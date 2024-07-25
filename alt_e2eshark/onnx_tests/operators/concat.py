
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

class ConcatModel(OnnxModelInfo):
    def construct_model(self):
        # Create an input (ValueInfoProto)
        X = make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 5])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 5])

        # Create an output
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4, 5])

        # Create a node (NodeProto)
        concat_node = make_node(
            op_type="Concat",
            inputs=["X", "Y"],
            outputs=["Z"],
            axis=0,
        )

        # Create the graph (GraphProto)
        graph = make_graph(
            nodes=[concat_node],
            name="main",
            inputs=[X, Y],
            outputs=[Z],
        )

        # Create the model (ModelProto)
        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 19

        onnx.save(onnx_model, self.model)


register_test(ConcatModel, "concat_test")