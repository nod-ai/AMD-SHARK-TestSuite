# run.pl creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing GRU
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

# Define dimensions
seq_length = 10
batch_size = 1
input_size = 5
hidden_size = 20
num_directions = 1

# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [seq_length, batch_size, input_size])
initial_h = make_tensor_value_info("initial_h", TensorProto.FLOAT, [num_directions, batch_size, hidden_size])

# Create tensor value info for W, R, B, sequence_lens
W = make_tensor_value_info("W", TensorProto.FLOAT, [num_directions, 3*hidden_size, input_size])
R = make_tensor_value_info("R", TensorProto.FLOAT, [num_directions, 3*hidden_size, hidden_size])
B = make_tensor_value_info("B", TensorProto.FLOAT, [num_directions, 6*hidden_size])
sequence_lens = make_tensor_value_info("sequence_lens", TensorProto.INT32, [batch_size])

Y = make_tensor_value_info("Y", TensorProto.FLOAT, [seq_length, num_directions, batch_size, hidden_size])
Y_h = make_tensor_value_info("Y_h", TensorProto.FLOAT, [num_directions, batch_size, hidden_size])

grunode = make_node(
    op_type="GRU",
    inputs=[
        "X",
        "W",
        "R",
        "B",
        "sequence_lens",
        "initial_h",
    ],
    outputs=[
        "Y",
        "Y_h",
    ],
    hidden_size=hidden_size,
)

graph = make_graph(
    [grunode],
    "gru_graph",
    [X, W, R, B, sequence_lens, initial_h],
    [Y, Y_h]
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 20

# Save the model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = onnxruntime.InferenceSession("model.onnx", None)

inputs = session.get_inputs()
outputs = session.get_outputs()

test_input = {
    "X": numpy.random.randn(seq_length, batch_size, input_size).astype(numpy.float32),
    "W": numpy.random.randn(num_directions, 3*hidden_size, input_size).astype(numpy.float32),
    "R": numpy.random.randn(num_directions, 3*hidden_size, hidden_size).astype(numpy.float32),
    "B": numpy.random.randn(num_directions, 6*hidden_size).astype(numpy.float32),
    "sequence_lens": numpy.array([seq_length], dtype=numpy.int32),
    "initial_h": numpy.zeros((num_directions, batch_size, hidden_size)).astype(numpy.float32),
}

test_output = session.run(None, test_input)

print("Input:", test_input)
print("Output:", test_output)

E2EAMDSHARK_CHECK["input"] = list(torch.from_numpy(arr) for arr in test_input.values())
E2EAMDSHARK_CHECK["output"] = list(torch.from_numpy(arr) for arr in test_output)