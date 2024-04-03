# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
# Issue: https://github.com/llvm/torch-mlir/issues/2764
# Description: ConstantOfShape whose input (tensor shape) is determined by a Constant node
# This construct is present in OPT and causes the ONNX importer to fail because
# of lack of support for ConstantOfShape with non-initializer input
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
E2ESHARK_CHECK = E2ESHARK_CHECK_DEF
const = make_node(
    "Constant",
    [],
    ["c_shape"],
    "const",
    value=numpy_helper.from_array(numpy.array([4], dtype=numpy.int64)),
)
cofshape = make_node(
    "ConstantOfShape",
    ["c_shape"],
    ["c_out"],
    "cofshape",
    value=numpy_helper.from_array(numpy.array([1], dtype=numpy.int64)),
)

# the part which does not change
outval = make_tensor_value_info("c_out", TensorProto.INT64, [None])
graph = make_graph([const, cofshape], "constgraph", [], [outval])
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 19
check_model(onnx_model)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = onnxruntime.InferenceSession("model.onnx", None)
model_input_X = numpy.array([])
# There is no input for this one
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()


model_output = session.run(
    [outputs[0].name],
    {},
)
# E2ESHARK_CHECK['input'] and E2ESHARK_CHECK['output'] are list of pytorch arrays
# each index into list is one input or one output in the
# order it appears in the model. for this test case it is empty
E2ESHARK_CHECK["input"] = []
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
