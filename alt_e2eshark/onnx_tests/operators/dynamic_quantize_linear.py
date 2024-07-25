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



class DynamicQuantizeLinearModel(OnnxModelInfo):
    def construct_model(self):
        input_X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
        output_Z = make_tensor_value_info("Z", TensorProto.UINT8, [3, 4])
        output_S = make_tensor_value_info("scale", TensorProto.FLOAT, [])
        output_P = make_tensor_value_info("zp", TensorProto.UINT8, [])

        # Create a 'DQL' node (NodeProto)
        DQL_node = make_node(
            "DynamicQuantizeLinear",
            ["X"],
            ["Z", "scale", "zp"],
            "DQL_node",  # op_type  # inputs  # outputs  # node name
        )

        # Create the graph (GraphProto)
        graph = make_graph(
            [DQL_node],  # Nodes in the graph
            "main",  # Name of the graph
            [input_X],  # Inputs to the graph
            [output_Z, output_S, output_P],  # Outputs of the graph
        )

        # Create the model (ModelProto)
        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = (
            11  # Set the opset version to ensure compatibility
        )

        # Save the model
        onnx.save(onnx_model, self.model)


register_test(DynamicQuantizeLinearModel, "dynamic_quanitze_linear_test")