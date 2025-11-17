# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_tensor_value_info

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test


class PadModel(BuildAModel):
    def construct_model(self):
        app_node = self.get_app_node()
        app_node("Constant", [], ["XP"], value_ints=[2,1,3,4])
        app_node("Pad", ["X","XP","XV"], ["XO"])
        app_node("Pad", ["Y","XP","YV"], ["YO"])
        app_node("Constant", [], ["ZP"], value_ints=[0,0,2,1,0,0,4,3])
        app_node("Pad", ["Z","ZP"], ["ZO"], mode="edge")
    
    def construct_i_o_value_info(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])
        Y = make_tensor_value_info("Y", TensorProto.INT32, [4, 5])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [1, 1, 4, 5])
        XV = make_tensor_value_info("XV", TensorProto.FLOAT, [])
        YV = make_tensor_value_info("YV", TensorProto.INT32, [])

        XO = make_tensor_value_info("XO", TensorProto.FLOAT, [-1, -1])
        YO = make_tensor_value_info("YO", TensorProto.INT32, [-1, -1])
        ZO = make_tensor_value_info("ZO", TensorProto.FLOAT, [-1, -1, -1, -1])
        self.input_vi = [X, XV, Y, YV, Z]
        self.output_vi = [XO, YO, ZO]

register_test(PadModel, "pad_test")
