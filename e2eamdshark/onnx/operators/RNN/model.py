# run.pl creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing RNN
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy
import torch
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model
import sys

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2EAMDSHARK_CHECK_DEF

# Create an instance of it for this test
E2EAMDSHARK_CHECK = dict(E2EAMDSHARK_CHECK_DEF)


# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [10, 1, 5])
initial_h = make_tensor_value_info("initial_h", TensorProto.FLOAT, [1, 1, 20])

# Create tensor value info for W, R, B, sequence_lens
W = make_tensor_value_info(
    "W", TensorProto.FLOAT, [1, 20, 5]
)  # [num_directions, hidden_size, input_size]
R = make_tensor_value_info(
    "R", TensorProto.FLOAT, [1, 20, 20]
)  # [num_directions, hidden_size, hidden_size]
B = make_tensor_value_info(
    "B", TensorProto.FLOAT, [1, 40]
)  # [num_directions, 2*hidden_size]
sequence_lens = make_tensor_value_info(
    "sequence_lens", TensorProto.INT32, [1]
)  # [batch_size]

Y = make_tensor_value_info("Y", TensorProto.FLOAT, [10, 1, 1, 20])
Y_h = make_tensor_value_info("Y_h", TensorProto.FLOAT, [1, 1, 20])


rnnnode = make_node(
    op_type="RNN",
    inputs=[
        "X",  # input tensor
        "W",  # weight tensor for input gate
        "R",  # recurrence weight tensor for input gate
        "B",  # optional bias tensor for input gate
        "sequence_lens",  # optional tensor specifying lengths of the sequences
        "initial_h",  # optional initial value of the hidden
    ],
    outputs=[
        "Y",  # output tensor
        "Y_h",  # the last output value of the hidden
    ],
    hidden_size=20,
)

graph = make_graph(
    [rnnnode], "rnn_graph", [X, W, R, B, sequence_lens, initial_h], [Y, Y_h]
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 20

# Save the model
# save_model(onnx_model, "model.onnx")
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = onnxruntime.InferenceSession("model.onnx", None)


inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()

test_input = {
    "X": numpy.random.randn(10, 1, 5).astype(numpy.float32),  # inputs
    "W": numpy.random.randn(1, 20, 5).astype(
        numpy.float32
    ),  # weight tensor for input gate
    "R": numpy.random.randn(1, 20, 20).astype(
        numpy.float32
    ),  # recurrence weight tensor for input gate
    "B": numpy.random.randn(1, 40).astype(
        numpy.float32
    ),  # optional bias tensor for input gate
    "sequence_lens": numpy.array(
        [10], dtype=numpy.int32
    ),  # optional tensor specifying lengths of the sequences
    "initial_h": numpy.zeros((1, 1, 20)).astype(
        numpy.float32
    ),  # optional initial value of the hidden
}

# test_input and test_output are list of numpy arrays
# each index into list is one input or one output in the
# order it appears in the model

test_output = session.run(None, test_input)

print("Input:", test_input)
print("Output:", test_output)
# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2EAMDSHARK_CHECK["input"] = list(torch.from_numpy(arr) for arr in test_input.values())
E2EAMDSHARK_CHECK["output"] = list(torch.from_numpy(arr) for arr in test_output)
