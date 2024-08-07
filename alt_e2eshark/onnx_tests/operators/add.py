# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_node, make_tensor_value_info

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test

class AddModel(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("X", TensorProto.FLOAT, [4, 5]),
            make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5]),
        ]
        self.output_vi = [make_tensor_value_info("Z", TensorProto.FLOAT, [4, 5])]

    def construct_nodes(self):
        self.node_list.append(make_node("Add", ["X", "Y"], ["Z"], "addnode"))

register_test(AddModel, "add_test")

