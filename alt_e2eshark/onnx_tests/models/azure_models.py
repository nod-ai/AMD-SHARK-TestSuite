# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..helper_classes import AzureDownloadableModel
from e2e_testing.registry import register_test

# If the azure model is simple enough to use the default input generation in e2e_testing/onnx_utils.py, then add the name to the list here.
# Note: several of these models will likely fail numerics without further post-processing. Consider writing a custom child class if further post-processing is desired. 
model_names = [
    "RAFT_vaiq_int8",
    "pytorch-3dunet_vaiq_int8",
    "FCN_vaiq_int8",
    "u-net_brain_mri_vaiq_int8",
]

for t in model_names:
    register_test(AzureDownloadableModel, t)
