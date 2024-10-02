# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto, numpy_helper
import os
import onnx

from pathlib import Path
from e2e_testing.storage import TestTensors
from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test
from site import getsitepackages

### IMPORTANT: ###
# These node tests are added primarily to help debugging torch-mlir conversions on a *case-by-case* basis.
# Please consider looking into SHARK-TestSuite/iree_tests/onnx for a much more robust and efficient alternative to running e2e node tests through iree.


def get_tensor_from_pb(inputpb):
    proto = TensorProto()
    with open(inputpb, "rb") as f:
        proto.ParseFromString(f.read())
    t = numpy_helper.to_array(proto)
    return t


base_dir = getsitepackages()[0]
# you can also get the node tests from the git submodule instead by uncommenting:
# base_dir = str(Path(__file__).parents[3]) + "/third_party/onnx"
onnx_node_tests_dir = base_dir + "/onnx/backend/test/data/node/"

names = os.listdir(onnx_node_tests_dir) if os.path.exists(onnx_node_tests_dir) else []


class NodeTest(OnnxModelInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = onnx_node_tests_dir + self.name + "/model.onnx"

    def construct_inputs(self):
        inputs = len(self.ort_input_nodes)
        num_inputs = len(inputs)
        input_list = []
        for i in range(num_inputs):
            inputpb = f"{onnx_node_tests_dir}{self.name}/test_data_set_0/input_{i}.pb"
            input_list.append(get_tensor_from_pb(inputpb))
        return TestTensors(input_list)

    def forward(self, input):
        model = onnx.load(self.model)
        outputs = model.graph.output
        num_outputs = len(outputs)
        output_list = []
        for i in range(num_outputs):
            outputpb = f"{onnx_node_tests_dir}{self.name}/test_data_set_0/output_{i}.pb"
            output_list.append(get_tensor_from_pb(outputpb))
        return TestTensors(output_list)


for n in names:
    register_test(NodeTest, n)
