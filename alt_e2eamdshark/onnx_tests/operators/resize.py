# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_tensor_value_info, make_tensor

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test


class ResizeModel(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("X", TensorProto.FLOAT, [1, 21, 65, 65])
        ]
        self.output_vi = [
            make_tensor_value_info("Y", TensorProto.FLOAT, [1, 21, 513, 513])
        ]

    def construct_nodes(self):
        C0T = make_tensor(
            "C0T",
            TensorProto.FLOAT,
            [4],
            [1.000000e00, 1.000000e00, 7.89230776, 7.89230776],
        )
        C1T = make_tensor(
            "C1T",
            TensorProto.FLOAT,
            [8],
            [
                0.000000e00,
                0.000000e00,
                0.000000e00,
                0.000000e00,
                1.000000e00,
                1.000000e00,
                1.000000e00,
                1.000000e00,
            ],
        )
        app_node = self.get_app_node()
        app_node("Constant", [], ["C0"], value=C0T)
        app_node("Constant", [], ["C1"], value=C1T)

        # this node has a lot of redundant info. Testing to see how it is treated.
        app_node(
            "Resize",
            ["X", "C1", "C0"],
            ["Y"],
            mode="linear",
            exclude_outside=0,
            coordinate_transformation_mode="half_pixel",
            cubic_coeff_a=-0.75,
            extrapolation_value=0.0,
            nearest_mode="round_prefer_floor",
        )


register_test(ResizeModel, "resize_test")
