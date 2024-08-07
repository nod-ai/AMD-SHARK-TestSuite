# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy
from onnx import TensorProto
import onnx
from onnx.helper import make_tensor_value_info

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test


class ShapeIntoCOSCombinationModel(BuildAModel):
    def construct_nodes(self):
        app_node = self.get_app_node()
        VT = onnx.numpy_helper.from_array(numpy.array([5], dtype=numpy.int64))
        app_node("Shape", ["B"], ["S"], name="shape_node")
        app_node("ConstantOfShape", ["S"], ["C"], value=VT)
        app_node("Expand", ["X", "C"], ["Y"], name="expand_node")

    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5]),
            make_tensor_value_info("B", TensorProto.FLOAT, [3]),
        ]
        self.output_vi = [
            make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])
        ]


register_test(ShapeIntoCOSCombinationModel, "shape_to_cos_test")
