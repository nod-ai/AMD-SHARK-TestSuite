# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing LRN
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy, torch, sys
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

tensor_shape = [13, 19, 100, 200]
# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, tensor_shape)

# Create an output
Y = make_tensor_value_info("Y", TensorProto.FLOAT, tensor_shape)

# Create a node (NodeProto)
lrn_node = make_node(
    op_type="LRN", 
    inputs=["X"], 
    outputs=["Y"], 
    name="lrn_node",
    size=5,
    alpha = 0.002,
    beta = 0.65,
    bias = 3.0
)

# Create the graph (GraphProto)
lrn_graph = make_graph(
    [lrn_node],
    "lrn_graph",
    [X],
    [Y],
)

# Create the model (ModelProto)
onnx_model = make_model(lrn_graph)
onnx_model.opset_import[0].version = 17

# Save the model
# save_model(onnx_model, "model.onnx")
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = onnxruntime.InferenceSession("model.onnx", None)
model_input_X = numpy.random.randn(*tensor_shape).astype(numpy.float32)
# gets X in inputs[0] and Y in inputs[1]
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_X},
)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2ESHARK_CHECK["input"] = [
    torch.from_numpy(model_input_X)
]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
