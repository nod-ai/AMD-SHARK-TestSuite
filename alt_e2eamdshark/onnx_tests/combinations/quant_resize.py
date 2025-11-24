# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
import torch
from onnx.helper import (
    make_node,
    make_tensor_value_info,
    make_tensor,
)

from ..helper_classes import BuildAModel
from e2e_testing.storage import TestTensors
from e2e_testing.registry import register_test

class ResizeTransposeQModel(BuildAModel):
    def construct_nodes(self):
        app_node = self.get_app_node()

        ST = make_tensor("ST",TensorProto.FLOAT, [], [0.25])
        ZPT = make_tensor("ZPT",TensorProto.INT8, [], [0])
        C0T = make_tensor("C0T",TensorProto.FLOAT, [4], [1.0, 1.0, 7.89230776, 7.89230776])
        C1T = make_tensor("C1T",TensorProto.FLOAT, [8], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        app_node("Constant",[],["S"],value=ST)
        app_node("Constant",[],["ZP"],value=ZPT)
        app_node("Constant",[],["C0"],value=C0T)
        app_node("Constant",[],["C1"],value=C1T)

        app_node("QuantizeLinear",["X0","S","ZP"], ["QX0"])
        app_node("DequantizeLinear", ["QX0","S","ZP"], ["DQX0"])
        app_node("Resize",["DQX0","C1","C0"],["X1"], mode="linear")
        app_node("QuantizeLinear",["X1","S","ZP"], ["QX1"])
        app_node("DequantizeLinear", ["QX1","S","ZP"], ["DQX1"])
        app_node("Transpose", ["DQX1"], ["X2"], perm = [0, 2, 3, 1]) 
        app_node("QuantizeLinear",["X2","S","ZP"], ["QX2"])
        app_node("DequantizeLinear", ["QX2","S","ZP"], ["DQX2"])
    
    def construct_i_o_value_info(self):
        self.input_vi.append(make_tensor_value_info("X0", TensorProto.FLOAT, [1, 21, 65, 65]))
        self.output_vi.append(make_tensor_value_info("DQX2", TensorProto.FLOAT, [1, 513, 513, 21]))

register_test(ResizeTransposeQModel, "resize_tq")

class SmallerResizeTQModel(ResizeTransposeQModel):
    def construct_inputs(self):
        # input = torch.Tensor([[[[0.42, 0.93], [0.27, 0.06]]]]).to(dtype=torch.float32)
        # this is the quantized result:
        input = torch.Tensor([[[[0.5, 1.0], [0.25, 0.00]]]]).to(dtype=torch.float32) 
        return TestTensors((input,))

    def construct_nodes(self):
        app_node = self.get_app_node()
        ST = make_tensor("ST",TensorProto.FLOAT, [], [0.25])
        ZPT = make_tensor("ZPT",TensorProto.INT8, [], [0])
        C0T = make_tensor("C0T",TensorProto.FLOAT, [4], [1.0, 1.0, 1.5, 1.5])

        app_node("Constant",[],["S"],value=ST)
        app_node("Constant",[],["ZP"],value=ZPT)
        app_node("Constant",[],["C0"],value=C0T)

        app_node("QuantizeLinear",["X0","S","ZP"], ["QX0"])
        app_node("DequantizeLinear", ["QX0","S","ZP"], ["DQX0"])
        app_node("Resize",["DQX0","","C0"],["X1"], mode="linear")
        app_node("QuantizeLinear",["X1","S","ZP"], ["QX1"])
        app_node("DequantizeLinear", ["QX1","S","ZP"], ["DQX1"])
        app_node("Transpose", ["DQX1"], ["X2"], perm = [0, 2, 3, 1]) 
    
    def construct_i_o_value_info(self):
        self.input_vi = [make_tensor_value_info("X0", TensorProto.FLOAT, [1, 1, 2, 2])]
        # output_vi = [make_tensor_value_info("DQX1", TensorProto.FLOAT, [1, 1, 4, 4])]
        self.output_vi = [make_tensor_value_info("X2", TensorProto.FLOAT, [1, 3, 3, 1])]

register_test(SmallerResizeTQModel, "resize_tq_0")

class SmallResizeIdentityQ(SmallerResizeTQModel):
    def construct_nodes(self):
        super().construct_nodes()
        self.node_list.pop()
        self.node_list.append(make_node("Identity", ["DQX1"],["X2"]))

    def construct_i_o_value_info(self):
        self.input_vi = [make_tensor_value_info("X0", TensorProto.FLOAT, [1, 1, 2, 2])]
        self.output_vi = [make_tensor_value_info("X2", TensorProto.FLOAT, [1, 1, 3, 3])]

register_test(SmallResizeIdentityQ, "resize_tq_1")

class SmallResizeIdentityQ2(SmallerResizeTQModel):
    def construct_nodes(self):
        super().construct_nodes()
        self.node_list.pop()

    def construct_i_o_value_info(self):
        self.input_vi = [make_tensor_value_info("X0", TensorProto.FLOAT, [1, 1, 2, 2])]
        self.output_vi = [make_tensor_value_info("DQX1", TensorProto.FLOAT, [1, 1, 3, 3])]

register_test(SmallResizeIdentityQ2, "resize_tq_2")

class SmallResizeIdentityQ3(SmallResizeIdentityQ):
    def construct_i_o_value_info(self):
        self.input_vi = [make_tensor_value_info("X0", TensorProto.FLOAT, [1, 1, 2, 2])]
        self.output_vi = [make_tensor_value_info("DQX1", TensorProto.FLOAT, [1, 1, 3, 3])]

register_test(SmallResizeIdentityQ3, "resize_tq_3")
