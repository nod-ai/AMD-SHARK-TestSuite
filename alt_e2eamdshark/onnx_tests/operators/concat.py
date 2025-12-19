# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_node, make_tensor_value_info

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test


class ConcatModel(BuildAModel):
    def construct_i_o_value_info(self):
        # Create an input (ValueInfoProto)
        X = make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 5])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 5])
        # Create an output
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4, 5])
        self.input_vi = [X, Y]
        self.output_vi = [Z]

    def construct_nodes(self):
        # Create a node (NodeProto)
        concat_node = make_node(
            op_type="Concat",
            inputs=["X", "Y"],
            outputs=["Z"],
            axis=0,
        )
        self.node_list = [concat_node]


register_test(ConcatModel, "concat_test")
