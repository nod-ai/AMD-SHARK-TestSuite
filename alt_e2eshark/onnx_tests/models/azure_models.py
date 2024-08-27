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

model_names = load_test_txt_file(lists_dir.joinpath("shark-test-suite.txt"))
for i in range(1,4):
    model_names += load_test_txt_file(lists_dir.joinpath(f"vai-hf-cnn-fp32-shard{i}.txt"))
    model_names += load_test_txt_file(lists_dir.joinpath(f"vai-int8-p0p1-shard{i}.txt"))
model_names += load_test_txt_file(lists_dir.joinpath("vai-vision-int8.txt"))

custom_registry = [
    "opt-125M-awq",
    "opt-125m-gptq",
    "DeepLabV3_resnet50_vaiq_int8",
]

# for simple models without dim params or additional customization, we should be able to register them directly with AzureDownloadableModel
# TODO: many of the models in the text files loaded from above will likely need to be registered with an alternative test info class.
for t in set(model_names).difference(custom_registry):
    register_test(AzureDownloadableModel, t)
