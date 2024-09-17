# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from ..helper_classes import AzureDownloadableModel
from e2e_testing.registry import register_test
from e2e_testing.storage import load_test_txt_file

this_file = Path(__file__)
lists_dir = (this_file.parent).joinpath("external_lists")

model_names = load_test_txt_file(lists_dir.joinpath("nlp-pytorch.txt"))

def dim_param_constructor(dim_param_dict):
    class AzureWithDimParams(AzureDownloadableModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # TODO: check opset versions
            # print(f"Opset: {self.opset_version}")
            # self.update_opset_version_and_overwrite()

        def update_dim_param_dict(self):
            self.dim_param_dict = dim_param_dict

    return AzureWithDimParams

# Default dimension parameters for NLP models

default_nlp_params = {"batch_size": 1, "seq_len": 128}
dim_aliases = [
    {'seq_len', 'sequence_length'},
]
for alias_set in dim_aliases:
    found = set(alias_set).intersection(default_nlp_params.keys())
    if len(found) > 1:
        # check if the values are the same
        val = default_nlp_params[next(iter(found))]
        if not all(default_nlp_params[alias] == val for alias in found):
            raise ValueError(f"Multiple aliases for the same dimension have different values: {found}")
    
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

