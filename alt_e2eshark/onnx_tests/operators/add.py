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

# add a test by setting test_dict["my_test_name"] = OnnxModelInfo(info about test)
class AddModel(OnnxModelInfo):
    def construct_model(self):
        # Create an input (ValueInfoProto)
        X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5])

        # Create an output
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [4, 5])

        # Create a node (NodeProto)
        addnode = make_node(
            "Add", ["X", "Y"], ["Z"], "addnode"  # node name  # inputs  # outputs
        )

        # Create the graph (GraphProto)
        graph = make_graph(
            [addnode],
            "main",
            [X, Y],
            [Z],
        )

        # Create the model (ModelProto)
        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 19

        onnx.save(onnx_model, self.model)


register_test(AddModel, "add_test")

