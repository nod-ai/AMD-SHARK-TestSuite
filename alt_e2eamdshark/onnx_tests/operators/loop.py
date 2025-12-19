# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_node, make_tensor_value_info, make_tensor, make_graph

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_with_name


@register_with_name("for_loop_basic")
class ForLoopBasic(BuildAModel):
    """
    x = x_init
    for i in range(10):
      x += i
    return x
    """

    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("x_init", TensorProto.INT64, [1]),
        ]
        self.output_vi = [
            make_tensor_value_info("x_result", TensorProto.INT64, [1]),
        ]

    def construct_nodes(self):
        body = make_graph(
            name="body",
            inputs=[
                make_tensor_value_info("iteration_num", TensorProto.INT64, [1]),
                make_tensor_value_info("keep_going", TensorProto.BOOL, [1]),
                make_tensor_value_info("x_in", TensorProto.INT64, [1]),
            ],
            nodes=[
                make_node("Add", ["x_in", "iteration_num"], ["x_out"]),
            ],
            outputs=[
                make_tensor_value_info("keep_going", TensorProto.BOOL, [1]),
                make_tensor_value_info("x_out", TensorProto.INT64, [1]),
            ],
        )
        self.initializers = [
            make_tensor("keep_going", TensorProto.BOOL, dims=[1], vals=[True]),
            make_tensor("max_trip_count", TensorProto.INT64, dims=[1], vals=[10]),
        ]
        self.node_list = [
            make_node(
                "Loop",
                ["max_trip_count", "keep_going", "x_init"],
                ["x_result"],
                body=body,
            ),
        ]
