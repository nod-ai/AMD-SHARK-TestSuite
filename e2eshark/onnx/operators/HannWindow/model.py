# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing HannWindow
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy, torch, sys
import onnxruntime
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
)

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.INT64, [1])

# Create an output
size = numpy.random.randint(low=1, high=50)
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [size])

# Create a node (NodeProto)
periodic = numpy.random.randint(2) if size > 1 else 1
hann_node = make_node(
    op_type="HannWindow",
    inputs=["X"],
    outputs=["Y"],
    periodic=periodic,
)

# Create the graph (GraphProto)
graph = make_graph(
    nodes=[hann_node],
    name="hann_graph",
    inputs=[X],
    outputs=[Y],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 17

# Save the model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = onnxruntime.InferenceSession("model.onnx", None)
model_input_X = numpy.array([size])
inputs = session.get_inputs()
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_X},
)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2ESHARK_CHECK["input"] = [
    torch.from_numpy(model_input_X),
]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
print("Periodic:", periodic)
