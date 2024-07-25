# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy, torch, sys
import onnxruntime
from onnx import TensorProto
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_tensor
import os

from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test


class PadModel(OnnxModelInfo):
    def construct_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])
        Y = make_tensor_value_info("Y", TensorProto.INT32, [4, 5])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [1, 1, 4, 5])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [3, 4, 5])
        XP = make_tensor_value_info("XP", TensorProto.INT64, [4])
        XV = make_tensor_value_info("XV", TensorProto.FLOAT, [])
        YV = make_tensor_value_info("YV", TensorProto.INT32, [])
        ZP = make_tensor_value_info("ZP", TensorProto.INT64, [8])

        XO = make_tensor_value_info("XO", TensorProto.FLOAT, [-1, -1])
        YO = make_tensor_value_info("YO", TensorProto.INT32, [-1, -1])
        ZO = make_tensor_value_info("ZO", TensorProto.FLOAT, [-1, -1, -1, -1])
        # constantNode = make_node(op_type="Constant", inputs=[], outputs=["XP"], value_ints=[2,1,3,4])
        # padnodeX = make_node("Pad", ["X","XP","XV"], ["XO"])
        # padnodeY = make_node(op_type="Pad", inputs=["Y","XP","YV"], outputs=["YO"])
        constantNodeZ = make_node(op_type="Constant", inputs=[], outputs=["ZP"], value_ints=[0,0,2,1,0,0,4,3])
        padnodeZ = make_node(op_type="Pad", inputs=["Z","ZP"], outputs=["ZO"], mode="edge")
        # padgraph = make_graph([constantNode, padnodeX], "main", [X, XV], [XO])
        padgraph = make_graph([constantNodeZ, padnodeZ], "main", [Z], [ZO])
        onnx_model = make_model(padgraph)
        onnx_model.opset_import[0].version = 19

        onnx.save(onnx_model, self.model)

register_test(PadModel, "pad_test")
