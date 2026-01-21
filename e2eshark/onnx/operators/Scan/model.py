# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing GatherND
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

 # Given an input sequence [x1, ..., xN], sum up its elements using a scan
# returning the final state (x1+x2+...+xN) as well the scan_output
# [x1, x1+x2, ..., x1+x2+...+xN]
#
# create graph to represent scan body
sum_in = make_tensor_value_info(
    "sum_in", TensorProto.FLOAT, [2]
)
next = make_tensor_value_info(  # noqa: A001
    "next", TensorProto.FLOAT, [2]
)
sum_out = make_tensor_value_info(
    "sum_out", TensorProto.FLOAT, [2]
)
scan_out = make_tensor_value_info(
    "scan_out", TensorProto.FLOAT, [2]
)
add_node = make_node(
    "Add", inputs=["sum_in", "next"], outputs=["sum_out"]
)
id_node = make_node(
    "Identity", inputs=["sum_out"], outputs=["scan_out"]
)
scan_body = make_graph(
    [add_node, id_node], "scan_body", [sum_in, next], [sum_out, scan_out]
)
# create scan op node
INIT = make_tensor_value_info("INIT", TensorProto.FLOAT, [2])
X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [2])
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 2])
#  no_sequence_lens = ""  # optional input, not supplied
scan_node = make_node(
    "Scan",
    inputs=["INIT", "X"],
    outputs=["Y", "Z"],
    num_scan_inputs=1,
    body=scan_body,
)

# Create the graph (GraphProto)
graph = make_graph(
    [scan_node],
    "scan_graph",
    [INIT, X],
    [Y, Z],
)
# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 16

# Save the model
# save_model(onnx_model, "model.onnx")
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
session = onnxruntime.InferenceSession("model.onnx", None)
# create inputs for batch-size 1, sequence-length 3, inner dimension 2
model_initial = numpy.array([0, 0]).astype(numpy.float32).reshape((2))
model_x = numpy.array([1, 2, 3, 4, 5, 6]).astype(numpy.float32).reshape((3, 2))
# final state computed = [1 + 3 + 5, 2 + 4 + 6]
#  model_y = numpy.array([9, 12]).astype(np.float32).reshape((1, 2))
# scan-output computed
#  model_z = numpy.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((1, 3, 2))

# gets D in inputs[0] and I in inputs[1]
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name, outputs[1].name],
    {inputs[0].name: model_initial, 
     inputs[1].name: model_x},
)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2ESHARK_CHECK["input"] = [
    torch.from_numpy(model_initial),
    torch.from_numpy(model_x),
]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
