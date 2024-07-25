# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info

from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test


class MultipleConvModel(OnnxModelInfo):
    def construct_model(self):
        AX0 = make_tensor_value_info("AX0", TensorProto.FLOAT, [1,3,513,513])
        AK0 = make_tensor_value_info("AK0", TensorProto.FLOAT, [32, 3, 3, 3])
        AK1 = make_tensor_value_info("AK1", TensorProto.FLOAT, [32, 1, 3, 3])
        AK2 = make_tensor_value_info("AK2", TensorProto.FLOAT, [16,32,1,1])
        X3 = make_tensor_value_info("X3", TensorProto.FLOAT, [1,16,257,257])

        node_list = []
        node_list.append(make_node(
            op_type="DynamicQuantizeLinear",
            inputs=["AX0"],
            outputs=["BX0","SX0","ZX0"],
        ))
        node_list.append(make_node(
            op_type="DynamicQuantizeLinear",
            inputs=["AK0"],
            outputs=["BK0","SK0","ZK0"],
        ))
        node_list.append(make_node(
            op_type="DequantizeLinear",
            inputs=["BX0","SX0","ZX0"],
            outputs=["X0"],
        ))
        node_list.append(make_node(
            op_type="DequantizeLinear",
            inputs=["BK0","SK0","ZK0"],
            outputs=["K0"],
        ))
        node_list.append(make_node(
            op_type="Conv",
            inputs=["X0","K0"],
            outputs=["AX1"],
            group=1,
            kernel_shape=[3,3],
            pads=[1,1,1,1],
            strides=[2,2],
        ))
        node_list.append(make_node(
            op_type="DynamicQuantizeLinear",
            inputs=["AX1"],
            outputs=["BX1","SX1","ZX1"],
        ))
        node_list.append(make_node(
            op_type="DynamicQuantizeLinear",
            inputs=["AK1"],
            outputs=["BK1","SK1","ZK1"],
        ))
        node_list.append(make_node(
            op_type="DequantizeLinear",
            inputs=["BX1","SX1","ZX1"],
            outputs=["X1"],
        ))
        node_list.append(make_node(
            op_type="DequantizeLinear",
            inputs=["BK1","SK1","ZK1"],
            outputs=["K1"],
        ))
        node_list.append(make_node(
            op_type="Conv",
            inputs=["X1","K1"],
            outputs=["AX2"],
            group=32,
            kernel_shape=[3,3],
            pads=[1,1,1,1],
            strides=[1,1],
        ))
        node_list.append(make_node(
            op_type="DynamicQuantizeLinear",
            inputs=["AX2"],
            outputs=["BX2","SX2","ZX2"],
        ))
        node_list.append(make_node(
            op_type="DynamicQuantizeLinear",
            inputs=["AK2"],
            outputs=["BK2","SK2","ZK2"],
        ))
        node_list.append(make_node(
            op_type="DequantizeLinear",
            inputs=["BX2","SX2","ZX2"],
            outputs=["X2"],
        ))
        node_list.append(make_node(
            op_type="DequantizeLinear",
            inputs=["BK2","SK2","ZK2"],
            outputs=["K2"],
        ))
        node_list.append(make_node(
            op_type="Conv",
            inputs=["X2","K2"],
            outputs=["X3"],
            group=1,
            kernel_shape=[1,1],
            pads=[0,0,0,0],
            strides=[1,1],
        ))

        graph=make_graph(
            node_list,
            "main",
            [AX0, AK0, AK1, AK2],
            [X3],
        )

        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 19

        onnx.save(onnx_model, self.model)

register_test(MultipleConvModel, "multi_conv")
        
