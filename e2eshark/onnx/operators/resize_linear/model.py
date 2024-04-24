# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy, torch, sys
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_tensor,
    make_attribute
)
from onnx.checker import check_model

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)


# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 3, 4])  # Example shape

# Specify the shape of the output
shape_tensor = make_tensor(
    name="shape",
    data_type=TensorProto.INT64,  # Note: Use INT64 for the shape tensor
    dims=(4,),  # Result tensor will also be of rank 3.
    vals=[1, 2, 6, 8],
)

shape_node = make_node(
    "Constant",
    inputs=[],
    outputs=["shape"],
    value=shape_tensor,
)

# Create an output (ValueInfoProto)
# The output shape will depend on the shape tensor.
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2, 6, 8])
# Create an 'Reshape' node (NodeProto)
resize_node = make_node(
    "Resize",
    ["X", "", "", "shape"],
    ["Z"],
    "resize_node",  # node name  # inputs  # outputs
)

resize_node.attribute.append(make_attribute("mode", "linear"))


# Create the graph (GraphProto)
graph = make_graph(
    [shape_node, resize_node],  # Nodes in the graph
    "resize_graph",
    [X],
    [Z],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 19  # Ensure compatibility with your ONNX runtime

# Save the model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Initialize the ONNX runtime session and run inference
session = onnxruntime.InferenceSession("model.onnx", None)
model_input_X = numpy.random.randn(1, 2, 3, 4).astype(
    numpy.float32
)  # Match the input shape
inputs = session.get_inputs()
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_X},
)


print("Input shape:", model_input_X.shape)
print("Output shape:", numpy.array(model_output[0]).shape)

E2ESHARK_CHECK["input"] = [torch.from_numpy(model_input_X)]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
