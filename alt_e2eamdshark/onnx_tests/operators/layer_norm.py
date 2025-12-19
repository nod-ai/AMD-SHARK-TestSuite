# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_node, make_tensor_value_info

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test


class LayerNormalizationModel(BuildAModel):
    def construct_i_o_value_info(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT16, [2, 3, 5])
        Scale = make_tensor_value_info("Scale", TensorProto.FLOAT16, [5])
        B = make_tensor_value_info("B", TensorProto.FLOAT16, [5])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 3, 5])
        Mean = make_tensor_value_info("Mean", TensorProto.FLOAT, [2, 3, 1])
        InvStdDev = make_tensor_value_info("InvStdDev", TensorProto.FLOAT, [2, 3, 1])
        self.input_vi = [X, Scale, B]
        self.output_vi = [Y, Mean, InvStdDev]

    def construct_nodes(self):
        layer_norm_node = make_node(
            op_type="LayerNormalization",
            inputs=["X", "Scale", "B"],
            outputs=["Y", "Mean", "InvStdDev"],
            axis=-1,
            epsilon=1e-05,
            stash_type=1,
        )
        self.node_list = [layer_norm_node]


register_test(LayerNormalizationModel, "layer_norm_test")
