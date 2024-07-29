# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_tensor

from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_with_name

class QConvModelBase(OnnxModelInfo):
    def __init__(self, specs, *args, **kwargs):
        (self.N, self.Cin, self.Hin, self.Win, self.Cout, self.groups, self.Hker, self.Wker, self.pads, self.dilations, self.strides) = specs
        self.Hout = int((self.Hin + self.pads[0] + self.pads[2] - self.dilations[0]*(self.Hker - 1) - 1)/self.strides[0] + 1)
        self.Wout = int((self.Win + self.pads[1] + self.pads[3] - self.dilations[1]*(self.Wker - 1) - 1)/self.strides[1] + 1)
        super().__init__(*args, **kwargs)

    def construct_model(self):
        # float input tensor:
        AX0 = make_tensor_value_info("AX0", TensorProto.FLOAT, [self.N, self.Cin, self.Hin, self.Win])
        # quantized weight tensor inputs
        BK = make_tensor_value_info("BK", TensorProto.INT8, [self.Cout, int(self.Cin/self.groups), self.Hker, self.Wker])
        # output tensor
        X1 = make_tensor_value_info("X1", TensorProto.FLOAT, [self.N, self.Cout, self.Hout, self.Wout])
        
        KZPT = make_tensor("ZPT", TensorProto.INT8, [], [0])
        KST = make_tensor("KST", TensorProto.FLOAT, [], [3.125000e-02])
        
        node_list = []
        app_node = lambda op_ty, inputs, outputs, **kwargs: node_list.append(
            make_node(op_ty, inputs, outputs, **kwargs)
        )
        # Quantization Scheme Constants
        app_node("Constant", [], ["KZP"], value=KZPT)
        app_node("Constant", [], ["KS"], value=KST)
        app_node("DynamicQuantizeLinear", ["AX0"], ["BX0", "X0S", "X0ZP"])
        app_node("DequantizeLinear", ["BX0", "X0S", "X0ZP"], ["X0"])
        app_node("DequantizeLinear", ["BK", "KS", "KZP"], ["K"])
        app_node("Conv", ["X0", "K"], ["X1"],
            group=self.groups,
            kernel_shape=[self.Hker, self.Wker],
            pads=self.pads,
            strides=self.strides,
        )

        graph = make_graph(
            node_list,
            "main",
            [AX0, BK],
            [X1],
        )

        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 19

        onnx.save(onnx_model, self.model)

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

get_specs = lambda : (N, Cin, Hin, Win, Cout, groups, Hker, Wker, pads, dilations, strides)

depthwise_specs = get_specs()

groups = 1
pads = [0, 0, 0, 0]

basic_specs = get_specs()

pads = [1,0,2,1]

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
