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
# Dynamic Quantize Linear returns a uint8 tensor, a f32 scale, and a uint8 zero point. 
Z = make_tensor_value_info("Z", TensorProto.UINT8, [3, 4])
Z_1 = make_tensor_value_info("Z_1", TensorProto.INT8, [3, 4])
S = make_tensor_value_info("scale", TensorProto.FLOAT, [])
P = make_tensor_value_info("zp", TensorProto.UINT8, [])
P_1 = make_tensor_value_info("zp_1", TensorProto.INT8, [])


# Create a 'DQL' node (NodeProto)
DQL_node = make_node(
    "DynamicQuantizeLinear", ["X"], ["Z", "scale", "zp"], "DQL_node"  # op_type  # inputs  # outputs  # node name
)
CastTensor_node = make_node(
    "Cast", ["Z"], ["Z_1"], to=3
)
CastZP_node = make_node(
    "Cast", ["zp"], ["zp_1"], to=3
)

# Create the graph (GraphProto)
graph = make_graph(
    [DQL_node, CastTensor_node, CastZP_node],  # Nodes in the graph
    "DQL_graph",  # Name of the graph
    [X],  # Inputs to the graph
    [Z_1, S, P_1],  # Outputs of the graph
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 11  # Set the opset version to ensure compatibility

#check_model(onnx_model)

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
    [outputs[0].name, outputs[1].name, outputs[2].name],
    {inputs[0].name: model_input_X},
)

print("Input shape:", model_input_X.shape)
print("Output shape:", numpy.array(model_output[0]).shape)

E2ESHARK_CHECK["input"] = [torch.from_numpy(model_input_X)]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
