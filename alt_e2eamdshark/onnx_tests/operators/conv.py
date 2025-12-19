# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_tensor_value_info, make_tensor

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_with_name, register_test


class ConvRepro(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("X", TensorProto.FLOAT, [1, 256, 112, 112]),
            make_tensor_value_info("W", TensorProto.FLOAT, [256, 1, 3, 3]),
            make_tensor_value_info("B", TensorProto.FLOAT, [256]),
        ]
        self.output_vi = [
            make_tensor_value_info("Y", TensorProto.FLOAT, [1, 256, 56, 56])
        ]

    def construct_nodes(self):
        app_node = self.get_app_node()
        app_node(
            "Conv",
            ["X", "W", "B"],
            ["Y"],
            group=256,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        )


register_test(ConvRepro, "conv_depthwise_stride_2")


class QConvModelBase(BuildAModel):
    def __init__(self, specs, *args, **kwargs):
        (
            self.N,
            self.Cin,
            self.Hin,
            self.Win,
            self.Cout,
            self.groups,
            self.Hker,
            self.Wker,
            self.pads,
            self.dilations,
            self.strides,
        ) = specs
        self.Hout = int(
            (
                self.Hin
                + self.pads[0]
                + self.pads[2]
                - self.dilations[0] * (self.Hker - 1)
                - 1
            )
            / self.strides[0]
            + 1
        )
        self.Wout = int(
            (
                self.Win
                + self.pads[1]
                + self.pads[3]
                - self.dilations[1] * (self.Wker - 1)
                - 1
            )
            / self.strides[1]
            + 1
        )
        super().__init__(*args, **kwargs)

    def construct_i_o_value_info(self):
        # float input tensor:
        AX0 = make_tensor_value_info(
            "AX0", TensorProto.FLOAT, [self.N, self.Cin, self.Hin, self.Win]
        )
        # quantized weight tensor inputs
        BK = make_tensor_value_info(
            "BK",
            TensorProto.INT8,
            [self.Cout, int(self.Cin / self.groups), self.Hker, self.Wker],
        )
        self.input_vi = [AX0, BK]
        # output tensor
        X1 = make_tensor_value_info(
            "X1", TensorProto.FLOAT, [self.N, self.Cout, self.Hout, self.Wout]
        )
        self.output_vi = [X1]

    def construct_nodes(self):
        KZPT = make_tensor("ZPT", TensorProto.INT8, [], [0])
        KST = make_tensor("KST", TensorProto.FLOAT, [], [3.125000e-02])

        app_node = self.get_app_node()

        # Quantization Scheme Constants
        app_node("Constant", [], ["KZP"], value=KZPT)
        app_node("Constant", [], ["KS"], value=KST)
        app_node("DynamicQuantizeLinear", ["AX0"], ["BX0", "X0S", "X0ZP"])
        app_node("DequantizeLinear", ["BX0", "X0S", "X0ZP"], ["X0"])
        app_node("DequantizeLinear", ["BK", "KS", "KZP"], ["K"])
        app_node(
            "Conv",
            ["X0", "K"],
            ["X1"],
            group=self.groups,
            kernel_shape=[self.Hker, self.Wker],
            pads=self.pads,
            strides=self.strides,
        )


N = 2
Cin = 3
Hin = 12
Win = 12
Cout = 3
groups = Cin
Hker = 5
Wker = 3
pads = [2, 0, 2, 0]
dilations = [1, 1]
strides = [1, 1]

get_specs = lambda: (
    N,
    Cin,
    Hin,
    Win,
    Cout,
    groups,
    Hker,
    Wker,
    pads,
    dilations,
    strides,
)

depthwise_specs = get_specs()

groups = 1
pads = [0, 0, 0, 0]

basic_specs = get_specs()

pads = [1, 0, 2, 1]

asymmetric_pad_specs = get_specs()

Cin = 6
Cout = 9
groups = 3
pads = [0, 0, 0, 0]

grouped_specs = get_specs()


@register_with_name("qconv_depthwise")
class QConvDepthwise(QConvModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(depthwise_specs, *args, **kwargs)


@register_with_name("qconv_basic")
class QConvBasic(QConvModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(basic_specs, *args, **kwargs)


@register_with_name("qconv_asymmetric_pads")
class QConvAsymmetricPads(QConvModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(asymmetric_pad_specs, *args, **kwargs)


@register_with_name("qconv_grouped")
class QConvGrouped(QConvModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(grouped_specs, *args, **kwargs)


class QConv1DModel(BuildAModel):
    def update_sess_options(self):
        import onnxruntime

        self.sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

    def construct_nodes(self):
        app_node = self.get_app_node()
        app_node("QuantizeLinear", ["X", "Scale", "ZP"], ["QX"])
        app_node("DequantizeLinear", ["QX", "Scale", "ZP"], ["DQX"])
        app_node("DequantizeLinear", ["QW", "Scale", "ZP"], ["DQW"])
        app_node("Conv", ["DQX", "DQW"], ["Y"], group=1, kernel_shape=[5], pads=[2, 2])
        app_node("QuantizeLinear", ["Y", "Scale", "ZP"], ["QY"])
        app_node("DequantizeLinear", ["QY", "Scale", "ZP"], ["DQY"])

    def construct_i_o_value_info(self):
        self.input_vi = [make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 256])]
        self.output_vi = [make_tensor_value_info("DQY", TensorProto.FLOAT, [1, 1, 256])]

    def construct_initializers(self):
        self.initializers = [
            make_tensor("Scale", TensorProto.FLOAT, [], [0.025]),
            make_tensor("ZP", TensorProto.INT8, [], [2]),
            make_tensor("QW", TensorProto.INT8, [1, 1, 5], [-20, 15, 100, 27, -1]),
        ]


register_test(QConv1DModel, "qconv1d_basic")
