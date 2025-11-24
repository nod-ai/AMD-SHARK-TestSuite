# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing MaxPool
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy, torch, sys
import onnxruntime
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor,
    make_tensor_value_info,
)

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2EAMDSHARK_CHECK_DEF

# Create an instance of it for this test
E2EAMDSHARK_CHECK = dict(E2EAMDSHARK_CHECK_DEF)

# Create an input (ValueInfoProto)
input = make_tensor_value_info("input", TensorProto.FLOAT, [8, 3, 32, 32])
rois = make_tensor_value_info("rois", TensorProto.FLOAT, [2, 5])

# Create an output
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 2, 2])  # Adjust shape according to the specification

# Create a node (NodeProto) for MaxRoiPool
maxroipool_node = make_node(
    op_type="MaxRoiPool",
    inputs=["input", "rois"],
    outputs=["Y"],
    name="maxroipool_node",
    pooled_shape=[2, 2],
    spatial_scale=1.0,
)

# Create the graph (GraphProto)
graph = make_graph(
    nodes=[maxroipool_node],
    name="maxroipool_graph",
    inputs=[input, rois],
    outputs=[Y],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 21

# Save the model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = onnxruntime.InferenceSession("model.onnx", None)
model_input_X = numpy.random.randn(8, 3, 32, 32).astype(numpy.float32)
model_input_rois = numpy.array([
    [2, 1, 16, 9, 24],
    [7, 5, 5, 13, 13]
], dtype=numpy.float32)

inputs = session.get_inputs()
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_X, inputs[1].name: model_input_rois},
)

print("Input shape:", model_input_X.shape)
print("Output shape:", numpy.array(model_output[0]).shape)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2EAMDSHARK_CHECK["input"] = [torch.from_numpy(model_input_X), torch.from_numpy(model_input_rois)]
E2EAMDSHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2EAMDSHARK_CHECK["input"])
print("Output:", E2EAMDSHARK_CHECK["output"])