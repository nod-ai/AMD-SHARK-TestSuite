import sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch_mlir.dynamo import _get_decomposition_table

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2Eamdshark_CHECK_DEF

# Create an instance of it for this test
E2Eamdshark_CHECK = dict(E2Eamdshark_CHECK_DEF)


class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            # print('create_mlp', n, m)
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            n = ln[i]

            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(
                    n,
                    m,
                    self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum",
                    sparse=False,
                )
            elif self.md_flag and n > self.md_threshold:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):
            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # create operators
            if ndevices <= 1:
                self.emb_l, w_list = self.create_emb(m_spa, ln_emb, weighted_pooling)
                if self.weighted_pooling == "learned":
                    self.v_W_l = nn.ParameterList()
                    for w in w_list:
                        self.v_W_l.append(Parameter(w))
                else:
                    self.v_W_l = w_list
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            # quantization
            self.quantize_emb = False
            self.emb_l_q = []
            self.quantize_bits = 32

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]
            # v_W_l is None for Kaggle DAC dataset
            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            if self.quantize_emb:
                s1 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                s2 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()

                if self.quantize_bits == 4:
                    QV = ops.quantized.embedding_bag_4bit_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                elif self.quantize_bits == 8:
                    QV = ops.quantized.embedding_bag_byte_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )

                ly.append(QV)
            else:
                E = emb_l[k]
                V = E(
                    sparse_index_group_batch,
                    sparse_offset_group_batch,
                    per_sample_weights=per_sample_weights,
                )
                ly.append(V)

        return ly

    #  using quantizing functions from caffe2/aten/src/ATen/native/quantized/cpu
    def quantize_embedding(self, bits):
        n = len(self.emb_l)
        self.emb_l_q = [None] * n
        for k in range(n):
            if bits == 4:
                self.emb_l_q[k] = ops.quantized.embedding_bag_4bit_prepack(
                    self.emb_l[k].weight
                )
            elif bits == 8:
                self.emb_l_q[k] = ops.quantized.embedding_bag_byte_prepack(
                    self.emb_l[k].weight
                )
            else:
                return
        self.emb_l = None
        self.quantize_emb = True
        self.quantize_bits = bits

    def interact_features(self, x, ly):
        # print(f'interact_features self.arch_interaction_op={self.arch_interaction_op}')
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            # print(f'bmm T.size={T.size()} torch.transpose(T, 1, 2)={torch.transpose(T, 1, 2).size()}')
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # print(f'bmm Z.size={Z.size()}')
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
            # print(f'bmm R.size={R.size()} x.size={x.size()}')
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        # print(f'    forward dense_x={dense_x.size()} lS_o={lS_o.size()} lS_i={lS_i.size()}')
        # print(f'    forward types dense_x={type(dense_x)} lS_o={type(lS_o)} lS_i={type(lS_i)}')
        # print(f'    forward size dense_x={dense_x.size()} lS_o={lS_o.size()} lS_i={lS_i.size()}')
        # print('dense_x=', dense_x)
        # print('lS_o=', lS_o)
        # print('lS_i=', lS_i)
        if self.ndevices <= 1:
            # single device run
            return self.sequential_forward(dense_x, lS_o, lS_i)
        else:
            print("Only single device run is support.")

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z


# End model definition

# Starts model instance creation
batch_size = 128

# Criteo Kaggle Display Advertisement Challenge Dataset
m_spa = 16
ln_emb = np.array(
    [
        1460,
        583,
        10131227,
        2202608,
        305,
        24,
        12517,
        633,
        3,
        93145,
        5683,
        8351593,
        3194,
        27,
        14992,
        5461306,
        10,
        5652,
        2173,
        4,
        7046547,
        18,
        15,
        286181,
        105,
        142572,
    ]
)
ln_bot = np.array([13, 512, 256, 64, 16])
ln_top = np.array([367, 512, 256, 1])

# Configure the model
dlrm_ref = DLRM_Net(
    m_spa,
    ln_emb,
    ln_bot,
    ln_top,
    arch_interaction_op="dot",
    arch_interaction_itself=False,
    sigmoid_bot=-1,
    sigmoid_top=ln_top.size - 2,
    sync_dense_params=True,
    loss_threshold=0.0,
    ndevices=-1,
    qr_flag=False,
    qr_operation=None,
    qr_collisions=None,
    qr_threshold=None,
    md_flag=False,
    md_threshold=None,
)
# Load the saved state_dict from the training
# model_path = "dlrm_kaggle.pt"
# print(f"INFO: Loading state_dict for the model from {model_path}")
# ld_model = torch.load(model_path, map_location=torch.device("cpu"))
# dlrm_ref.load_state_dict(ld_model["state_dict"])

# Sample test query from a single run on Kaggle DAC dataset
# 13 dense features organized as 1x13 tensor
dense_X = torch.tensor(
    [
        [
            0.0000,
            0.0000,
            1.6094,
            0.0000,
            9.3463,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ]
    ]
)

in_features = dense_X.size(1)

# 26 sparse features are looked up via two tensors
# lS_o (26x1) prodives the offset to
# lS_i (26x1) prodives the index to the EmbeddingBag
lS_o = torch.tensor(
    [
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
    ]
)

# Embedding values are summed if multiple indexes are specifed for a particular embedding table
lS_i = torch.tensor(
    [
        [3],
        [1],
        [1],
        [1],
        [0],
        [1],
        [678],
        [1],
        [0],
        [238],
        [294],
        [1],
        [269],
        [0],
        [17],
        [1],
        [5],
        [1],
        [0],
        [1],
        [1],
        [1],
        [1],
        [1],
        [2],
        [1],
    ]
)

# Repeat the same dense features
batch_dense_X = dense_X.repeat(batch_size, 1)
batch_dense_X = torch.rand((batch_size, in_features))

# Indices to embedding bags
# The same sparse features are used for each batch as we only have expected
# output for one query for now
# Each column is a new batch. The column number is specifed by batch_lS_o
# below
batch_lS_i = lS_i.repeat(1, batch_size)

# Offset controls the starting offset of the index vector below for each batch
# Make it increment by 1 for each batch
batch_lS_o = torch.zeros(ln_emb.size, batch_size, dtype=torch.int64)
for i in range(ln_emb.size):
    for j in range(1, batch_size):
        batch_lS_o[i][j] = j

# Run the model with the sample query and profiler
dlrm_ref.eval()
E2Eamdshark_CHECK["input"] = [batch_dense_X, batch_lS_o, batch_lS_i]
# Flag to prevent casting of input to a different dtype
keep_input_dtype = True

# model = make_fx(  # type: ignore[no-untyped-call]
#    dlrm_ref,
#    decomposition_table=_get_decomposition_table(),  # type: ignore[no-untyped-call]
# )(*test_input_list)

model = torch.jit.trace(dlrm_ref, (batch_dense_X, batch_lS_o, batch_lS_i))
model.eval()

print(f"INFO: Running inference using fx graph to generate reference data...")
E2Eamdshark_CHECK["output"] = model(batch_dense_X, batch_lS_o, batch_lS_i)
sorted, indices = torch.sort(E2Eamdshark_CHECK["output"], dim=0, descending=True)
top_n = batch_size if batch_size < 5 else 5
print(f"INFO: Clickthrough probability of top {top_n} ads:")
header_format = "{0:4s}|{1:20s}"
cell_format = "{0:4d}|{1:5.2f}"
print(header_format.format(4 * "-", 20 * "-"))
print(header_format.format("Ad", "Probability(%)"))
print(header_format.format(4 * "-", 20 * "-"))
for i in range(top_n):
    print(cell_format.format(indices[i][0].item(), sorted[i][0] * 100))

print("Input:", E2Eamdshark_CHECK["input"])
print("Output:", E2Eamdshark_CHECK["output"])
E2Eamdshark_CHECK["torchmlirimport"] = "compile"
