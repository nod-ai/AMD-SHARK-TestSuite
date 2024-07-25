# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy
from onnx import TensorProto
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info

from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test

class ShapeIntoCOSCombinationModel(OnnxModelInfo):
    def construct_model(self):
        input_X = make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5])
        input_B = make_tensor_value_info("B", TensorProto.FLOAT, [3])
        output_Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])

        shape_node = make_node("Shape", ["B"], ["S"], "shape_node")

        cos_node = make_node(
            "ConstantOfShape",
            ["S"],
            ["C"],
            value=onnx.numpy_helper.from_array(numpy.array([5], dtype=numpy.int64)),
        )
        expand_node = make_node("Expand", ["X", "C"], ["Y"], "expand_node")

        graph = make_graph(
            [shape_node, cos_node, expand_node],
            "main",
            [input_X, input_B],
            [output_Y],
        )

        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 11

        onnx.save(onnx_model, self.model)


register_test(ShapeIntoCOSCombinationModel, "shape_to_cos_test")