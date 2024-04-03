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
X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 5])  # Example input shape

# Create an output (ValueInfoProto)
# LSM returns a tensor of the same type as input
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4, 5])

# Create a 'LSM' node (NodeProto)
LSM_node = make_node(
    "LogSoftmax", ["X"], ["Y"], axis=0  # op_type  # inputs  # outputs  # node name
)


# Create the graph (GraphProto)
graph = make_graph(
    [LSM_node],  # Nodes in the graph
    "LSM_graph",  # Name of the graph
    [X],  # Inputs to the graph
    [Y],  # Outputs of the graph
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 1  # Set the opset version to ensure compatibility

# Save the model
model_path = "model.onnx"
with open(model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Initialize the ONNX runtime session and run inference
session = onnxruntime.InferenceSession(model_path, None)
model_input_X = numpy.random.randn(3, 4, 5).astype(numpy.float32)  # Match the input shape

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
