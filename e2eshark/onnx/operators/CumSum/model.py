# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing CumSum
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy, torch, sys
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_tensor,
)
from onnx.checker import check_model

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])

axis_tensor = make_tensor(
    name="axis",
    data_type=TensorProto.INT32,
    dims=(1,),
    vals=[1],
)

axis_node = make_node(
    "Constant",
    inputs=[],
    outputs=["axis"],
    value=axis_tensor,
)

# Create an output
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [4, 5])

# Create a node (NodeProto)
cumsum_node = make_node(
    "CumSum", ["X", "axis"], ["Z"], "cumsum_node"  # node name  # inputs  # outputs
)

# Create the graph (GraphProto)
graph = make_graph(
    [axis_node, cumsum_node],  # 'axis_node' is now before 'cumsum_node'
    "cumsum_graph",
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
model_input_X = numpy.random.randn(4, 5).astype(numpy.float32)
inputs = session.get_inputs()
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_X},
)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2ESHARK_CHECK["input"] = [torch.from_numpy(model_input_X)]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
