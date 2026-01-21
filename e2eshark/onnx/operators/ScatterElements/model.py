# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing Scatter
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy, torch, sys
import onnxruntime
from onnx import TensorProto, AttributeProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor,
    make_tensor_value_info,
    make_attribute,
)

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

# Create an input (ValueInfoProto)
data = make_tensor_value_info("data", TensorProto.FLOAT, [4])
updates = make_tensor_value_info("updates", TensorProto.FLOAT, [6])


# Create an output
output = make_tensor_value_info("output", TensorProto.FLOAT, [4])

indices_tensor = make_tensor(
    name="indices",
    data_type=TensorProto.INT64,
    dims=(6,),
    vals=[0, 1, 0, 1, 2, 1],
)

# Create a node (NodeProto) for indices
indices_node = make_node(
    "Constant",
    inputs=[],
    outputs=["indices"],
    value=indices_tensor,
)

# Create a node (NodeProto) for Scatter
scatter_node = make_node(
    op_type="ScatterElements",
    inputs=["data", "indices", "updates"],
    outputs=["output"],
    name="scatter_node",
    axis=0,
    # attributes=[
    #     make_attribute("axis", 0),
    # ]
)

print("BEGIN: scatter_node.attribute.append(make_attribute(reduction, 'add')")
# scatter_node.attribute.append(make_attribute("axis", 0))
scatter_node.attribute.append(make_attribute("reduction", 'add', attr_type = AttributeProto.STRING))
print("END: scatter_node.attribute.append(make_attribute(reduction, 'add')")


# Create the graph (GraphProto)
graph = make_graph(
    nodes=[indices_node, scatter_node],
    name="scatter_graph",
    inputs=[data, updates],
    outputs=[output],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 19

# Save the model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = onnxruntime.InferenceSession("model.onnx", None)
model_input_data= numpy.array([
    1.0, 2.0, 3.0, 4.0,
], dtype=numpy.float32)
model_input_updates = numpy.array([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0
], dtype=numpy.float32)

inputs = session.get_inputs()
outputs = session.get_outputs()

model_output = session.run(
    [
        outputs[0].name
    ],
    {
        inputs[0].name: model_input_data, 
        # inputs[1].name: model_input_indices, 
        inputs[1].name: model_input_updates
    },
)

print("Input shape:", model_input_data.shape)
print("Output shape:", numpy.array(model_output[0]).shape)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2ESHARK_CHECK["input"] = [
    torch.from_numpy(model_input_data),
    torch.from_numpy(model_input_updates),
]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
