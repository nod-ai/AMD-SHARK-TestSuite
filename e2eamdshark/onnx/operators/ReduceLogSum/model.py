# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing ReduceLogSum 
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

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2EAMDSHARK_CHECK_DEF

# Create an instance of it for this test
E2EAMDSHARK_CHECK = dict(E2EAMDSHARK_CHECK_DEF)

# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])

# Create an output (ValueInfoProto)
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1])

# Create a node (NodeProto)
reducelogsum_node = make_node(
    op_type="ReduceLogSum",
    inputs=["X"],
    outputs=["Y"]
)

# Create the graph (GraphProto)
graph = make_graph(
    nodes=[reducelogsum_node],
    name="reducelogsum_graph",
    inputs=[X],
    outputs=[Y],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 17

# Save the model
model_path = "model.onnx"
with open(model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Initialize the ONNX runtime session and run inference
session = onnxruntime.InferenceSession(model_path, None)
model_input_X = numpy.random.uniform(low = 0.1, high = 1, size = (2,3,4)).astype(numpy.float32)

inputs = session.get_inputs()
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_X},
)

print("Input shape:", model_input_X.shape)
print("Output shape:", numpy.array(model_output[0]).shape)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2EAMDSHARK_CHECK["input"] = [torch.from_numpy(model_input_X)]
E2EAMDSHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2EAMDSHARK_CHECK["input"])
print("Output:", E2EAMDSHARK_CHECK["output"])