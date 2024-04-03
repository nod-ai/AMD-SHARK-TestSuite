# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy, torch, sys
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)


# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])  # Example input shape

# Create an output (ValueInfoProto)
# The output shape is the same as the input shape for Sigmoid operation
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4])  # Same shape as input

# Create a 'Sigmoid' node (NodeProto)
sigmoid_node = make_node(
    "Sigmoid", ["X"], ["Z"], "sigmoid_node"  # op_type  # inputs  # outputs  # node name
)

# Create the graph (GraphProto)
graph = make_graph(
    [sigmoid_node],  # Nodes in the graph
    "sigmoid_graph",  # Name of the graph
    [X],  # Inputs to the graph
    [Z],  # Outputs of the graph
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 14  # Set the opset version to ensure compatibility

# Save the model
model_path = "model.onnx"
with open(model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Initialize the ONNX runtime session and run inference
session = onnxruntime.InferenceSession(model_path, None)
model_input_X = numpy.random.randn(3, 4).astype(numpy.float32)  # Match the input shape

inputs = session.get_inputs()
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_X},
)

print("Input shape:", model_input_X.shape)
print("Output shape:", numpy.array(model_output[0]).shape)

# things that need to be kept constant for every test:
# 1. Define Input and Output Shapes Clearly: Each test must start by defining the input tensor shapes using make_tensor_value_info, ensuring the dimensions match the expected input for the operation being tested.
# 2. There must be a E2ESHARK_CHECK['input']s and E2ESHARK_CHECK['output']s list so that the test harness code can access the input and output data for validation.
# 3. The model must be saved as model.onnx in the current working directory

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2ESHARK_CHECK["input"] = [torch.from_numpy(model_input_X)]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
