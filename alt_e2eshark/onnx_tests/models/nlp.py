# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from ..helper_classes import AzureDownloadableModel
from e2e_testing.registry import register_test
from e2e_testing.storage import load_test_txt_file
from e2e_testing.onnx_utils import get_node_shape_from_dim_param_dict
from e2e_testing.storage import TestTensors
import onnxruntime as ort
import numpy
from typing import Optional
import os

this_file = Path(__file__)
lists_dir = (this_file.parent).joinpath("external_lists")

model_names = []
for i in [1, 2, 3]:
    model_names += load_test_txt_file(lists_dir.joinpath(f"nlp-shard{i}.txt"))


def dim_param_constructor(dim_param_dict):
    class AzureWithDimParams(AzureDownloadableModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # TODO: check opset versions
            # print(f"Opset: {self.opset_version}")
            # self.update_opset_version_and_overwrite()

        def update_dim_param_dict(self):
            self.dim_param_dict = dim_param_dict

        def construct_inputs(self):
            """Overrides the parent class method to construct sample inputs with the correct dimensions."""
            default_inputs = super().construct_inputs()

            tensors = list(default_inputs.data)

            self.update_sess_options()
            session = ort.InferenceSession(self.model, self.sess_options)

            # nlp specific overrides
            for i, node in enumerate(session.get_inputs()):
                if node.name == "token_type_ids":
                    rng = numpy.random.default_rng(19)
                    int_dims = get_node_shape_from_dim_param_dict(
                        node, self.dim_param_dict
                    )
                    tensors[i] = rng.integers(0, 2, size=int_dims, dtype=numpy.int64)
            default_sample_inputs = TestTensors(tuple(tensors))
            return default_sample_inputs

    return AzureWithDimParams


# Default dimension parameters for NLP models

# TODO: Verify these dim params are valid for each model, or load them from a json
default_nlp_params = {
    "batch_size": 1,
    "seq_len": 128,
    "encoder_sequence_length": 128,
    "decoder_sequence_length": 128,
}
dim_aliases = [
    {"seq_len", "sequence_length"},
]
for alias_set in dim_aliases:
    found = set(alias_set).intersection(default_nlp_params.keys())
    if len(found) > 1:
        # check if the values are the same
        val = default_nlp_params[next(iter(found))]
        if not all(default_nlp_params[alias] == val for alias in found):
            raise ValueError(
                f"Multiple aliases for the same dimension have different values: {found}"
            )

    aliases = alias_set - found
    for alias in aliases:
        default_nlp_params[alias] = default_nlp_params[next(iter(found))]

# Register models with default parameters
for model_name in model_names:
    register_test(dim_param_constructor(default_nlp_params), model_name)

# Custom dimension parameters for specific models
custom_dim_params = {
    # Add custom dimension parameters for specific models here
    # Example:
    # "model_name": {"batch_size": 1, "seq_len": 256, "custom_dim": 64},
}

# Register models with custom parameters
for model_name, dim_params in custom_dim_params.items():
    if model_name in model_names:
        register_test(dim_param_constructor(dim_params), model_name)

# You can add more customizations or specific handling for certain models here
