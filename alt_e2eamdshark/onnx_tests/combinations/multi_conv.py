# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import (
    make_tensor_value_info,
    make_tensor,
)
from typing import Optional

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test


class MultipleConvBase(BuildAModel):
    def __init__(self, use_bias, use_clips, *args, **kwargs):
        self.use_bias = use_bias
        self.use_clips = use_clips
        super().__init__(*args, **kwargs)

    def construct_i_o_value_info(self):
        # float input tensor:
        AX0 = make_tensor_value_info("AX0", TensorProto.FLOAT, [1, 3, 513, 513])
        # quantized weight tensor inputs
        BK0 = make_tensor_value_info("BK0", TensorProto.INT8, [32, 3, 3, 3])
        BK1 = make_tensor_value_info("BK1", TensorProto.INT8, [32, 1, 3, 3])
        BK2 = make_tensor_value_info("BK2", TensorProto.INT8, [16, 32, 1, 1])
        # output tensor
        X3 = make_tensor_value_info("X3", TensorProto.FLOAT, [1, 16, 257, 257])
        self.input_vi = [AX0, BK0, BK1, BK2]
        self.output_vi = [X3]

    def construct_nodes(self):

        # scale for K1
        KS1T = make_tensor("KS1T", TensorProto.FLOAT, [], [5.000000e-01])
        # ZP is a trivial zeropoint.
        ZPT = make_tensor("ZPT", TensorProto.INT8, [], [0])
        # quantized bias values and corresponding scales
        if self.use_bias:
            B0T = make_tensor(
                "B0T",
                TensorProto.INT8,
                [32],
                [
                    8,
                    -58,
                    -45,
                    82,
                    10,
                    -40,
                    52,
                    71,
                    45,
                    81,
                    2,
                    14,
                    96,
                    49,
                    -42,
                    -34,
                    116,
                    -32,
                    11,
                    85,
                    95,
                    -51,
                    -82,
                    38,
                    7,
                    60,
                    25,
                    9,
                    80,
                    59,
                    92,
                    94,
                ],
            )
            B1T = make_tensor(
                "B1T",
                TensorProto.INT8,
                [32],
                [
                    28,
                    -14,
                    -5,
                    54,
                    0,
                    8,
                    44,
                    36,
                    5,
                    65,
                    39,
                    -1,
                    51,
                    36,
                    -14,
                    4,
                    39,
                    -17,
                    32,
                    30,
                    43,
                    -8,
                    2,
                    9,
                    22,
                    20,
                    43,
                    84,
                    28,
                    58,
                    72,
                    51,
                ],
            )
            B2T = make_tensor(
                "B2T",
                TensorProto.INT8,
                [16],
                [
                    48,
                    123,
                    -30,
                    -68,
                    14,
                    102,
                    -20,
                    -18,
                    -58,
                    65,
                    -19,
                    -12,
                    -82,
                    -58,
                    -38,
                    -59,
                ],
            )
            BS2T = make_tensor("BS2T", TensorProto.FLOAT, [], [2.500000e-01])
        BS0T = make_tensor("BS0T", TensorProto.FLOAT, [], [3.125000e-02])
        BS1T = make_tensor(
            "BS1T", TensorProto.FLOAT, [], [6.250000e-02]
        )  # also used to quantize/dequantize input between convs
        if self.use_clips:
            LT = make_tensor("LT", TensorProto.FLOAT, [], [0.0])
            UT = make_tensor("UT", TensorProto.FLOAT, [], [6.0])

        app_node = self.get_app_node()
        # Quantization Scheme Constants
        app_node("Constant", [], ["ZP"], value=ZPT)
        app_node("Constant", [], ["BS0"], value=BS0T)
        app_node("Constant", [], ["KS1"], value=KS1T)
        app_node("Constant", [], ["BS1"], value=BS1T)
        if self.use_bias:
            app_node("Constant", [], ["BS2"], value=BS2T)
            # bias constants and conversions
            app_node("Constant", [], ["AB0"], value=B0T)
            app_node("Constant", [], ["AB1"], value=B1T)
            app_node("Constant", [], ["AB2"], value=B2T)
            app_node("DequantizeLinear", ["AB0", "BS0", "ZP"], ["B0"])
            app_node("DequantizeLinear", ["AB1", "BS1", "ZP"], ["B1"])
            app_node("DequantizeLinear", ["AB2", "BS2", "ZP"], ["B2"])
        if self.use_clips:
            app_node("Constant", [], ["L"], value=LT)
            app_node("Constant", [], ["U"], value=UT)
        app_node("QuantizeLinear", ["AX0", "BS1", "ZP"], ["BX0"])
        app_node("DequantizeLinear", ["BX0", "BS1", "ZP"], ["X0"])
        app_node("DequantizeLinear", ["BK0", "BS0", "ZP"], ["K0"])
        app_node(
            "Conv",
            ["X0", "K0", "B0"] if self.use_bias else ["X0", "K0"],
            ["AX1"],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        )
        name1 = "AX1"
        if self.use_clips:
            name1 = "CX1"
            app_node("Clip", ["AX1", "L", "U"], [name1])
        app_node("QuantizeLinear", [name1, "BS1", "ZP"], ["BX1"])
        app_node("DequantizeLinear", ["BX1", "BS1", "ZP"], ["X1"])
        app_node("DequantizeLinear", ["BK1", "KS1", "ZP"], ["K1"])
        app_node(
            "Conv",
            ["X1", "K1", "B1"] if self.use_bias else ["X1", "K1"],
            ["AX2"],
            group=32,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        )
        name2 = "AX2"
        if self.use_clips:
            name2 = "CX2"
            app_node("Clip", ["AX2", "L", "U"], [name2])
        app_node("QuantizeLinear", [name2, "BS1", "ZP"], ["BX2"])
        app_node("DequantizeLinear", ["BX2", "BS1", "ZP"], ["X2"])
        app_node("DequantizeLinear", ["BK2", "BS1", "ZP"], ["K2"])
        app_node(
            "Conv",
            ["X2", "K2", "B2"] if self.use_bias else ["X2", "K2"],
            ["X3"],
            group=1,
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )


basic = lambda *args, **kwargs: MultipleConvBase(False, False, *args, **kwargs)
bias = lambda *args, **kwargs: MultipleConvBase(True, False, *args, **kwargs)
bias_clips = lambda *args, **kwargs: MultipleConvBase(True, True, *args, **kwargs)
clips = lambda *args, **kwargs: MultipleConvBase(False, True, *args, **kwargs)


register_test(basic, "multi_conv_basic")
register_test(bias, "multi_conv_bias")
register_test(bias_clips, "multi_conv_bias_clips")
register_test(clips, "multi_conv_clips")
