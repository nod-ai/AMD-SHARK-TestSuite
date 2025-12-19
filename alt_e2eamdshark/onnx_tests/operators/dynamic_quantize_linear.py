# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test


class DynamicQuantizeLinearModel(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])]
        self.output_vi = [
            make_tensor_value_info("Z", TensorProto.UINT8, [3, 4]),
            make_tensor_value_info("scale", TensorProto.FLOAT, []),
            make_tensor_value_info("zp", TensorProto.UINT8, []),
        ]

    def construct_nodes(self):
        # Create a 'DQL' node (NodeProto)
        self.node_list = [
            make_node(
                "DynamicQuantizeLinear",
                ["X"],
                ["Z", "scale", "zp"],
                "DQL_node",  # op_type  # inputs  # outputs  # node name
            )
        ]


register_test(DynamicQuantizeLinearModel, "dynamic_quanitze_linear_test")
