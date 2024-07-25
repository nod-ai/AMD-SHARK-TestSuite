# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_tensor

from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test


class ResizeModel(OnnxModelInfo):
    def construct_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [1,21,65,65])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [1, 21, 513, 513])
        C0T = make_tensor("C0T",TensorProto.FLOAT, [4], [1.000000e+00, 1.000000e+00, 7.89230776, 7.89230776])
        C1T = make_tensor("C1T",TensorProto.FLOAT, [8], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00])

        const_node0 = make_node(
            op_type="Constant",
            inputs=[],
            outputs=["C0"],
            value=C0T,
        )

        const_node1 = make_node(
            op_type="Constant",
            inputs=[],
            outputs=["C1"],
            value=C1T,
        )

        resize_node = make_node(
            op_type="Resize",
            inputs=["X","C1","C0"],
            outputs=["Y"],
            mode="linear",
            exclude_outside=0,
            coordinate_transformation_mode="half_pixel",
            cubic_coeff_a=-0.75,
            extrapolation_value=0.0,
            nearest_mode="round_prefer_floor",
        )

        graph = make_graph([const_node0, const_node1, resize_node],"main",[X],[Y])
        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 19
        onnx.save(onnx_model, self.model)

register_test(ResizeModel, "resize_test")
