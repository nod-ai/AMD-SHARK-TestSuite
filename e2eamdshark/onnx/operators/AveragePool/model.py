# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing AveragePool
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy, torch, sys
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_attribute
from onnx.checker import check_model

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2EAMDSHARK_CHECK_DEF

# Create an instance of it for this test
E2EAMDSHARK_CHECK = dict(E2EAMDSHARK_CHECK_DEF)

# Create an input (ValueInfoProto)
# X = make_tensor_value_info("X", TensorProto.FLOAT, [32, 384, 25, 25])
X = make_tensor_value_info("X", TensorProto.FLOAT, [1,1,4,4])

# Create an output
# Z = make_tensor_value_info("Z", TensorProto.FLOAT, [32, 384, 25, 25])
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [1,1,2,2])

# Create a node (NodeProto)
AveragePoolNode = make_node(
    "AveragePool", ["X"], ["Z"], "AveragePoolNode"  # node name  # inputs  # outputs
)
# Create attributes for AveragePoolNode
AveragePoolNode.attribute.append(make_attribute("ceil_mode", 1))
AveragePoolNode.attribute.append(make_attribute("count_include_pad", 0))
# AveragePoolNode.attribute.append(make_attribute("pads", [1,1,1,1]))
AveragePoolNode.attribute.append(make_attribute("kernel_shape", [2, 2]))
AveragePoolNode.attribute.append(make_attribute("strides", [1,1]))
AveragePoolNode.attribute.append(make_attribute("dilations", [2,2]))

# Create the graph (GraphProto)
graph = make_graph(
    [AveragePoolNode],
    "AveragePoolGraph",
    [X],
    [Z],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 19

# Save the model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())


session = onnxruntime.InferenceSession("model.onnx", None)
# model_input_X = numpy.random.randn(32, 384, 25, 25).astype(numpy.float32)
model_input_X = numpy.random.randn(1,1,4,4).astype(numpy.float32)
# gets X in inputs[0]
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_X},
)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2EAMDSHARK_CHECK["input"] = [
    torch.from_numpy(model_input_X),
]
E2EAMDSHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2EAMDSHARK_CHECK["input"])
print("Output:", E2EAMDSHARK_CHECK["output"])
