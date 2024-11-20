# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from e2e_testing.registry import register_test
from e2e_testing.storage import load_test_txt_file
from ..helper_classes import OnnxModelZooDownloadableModel
from .azure_models import custom_registry


this_file = Path(__file__)
lists_dir = (this_file.parent).joinpath("external_lists")
onnx_zoo_non_validated = load_test_txt_file(lists_dir.joinpath("onnx_model_zoo_non_validated_paths.txt"))
onnx_zoo_validated = load_test_txt_file(lists_dir.joinpath("onnx_model_zoo_validated_paths.txt"))

# Putting this inside the class contructor will
# call this repeatedly, which is wasteful.
model_path_map = {}
def build_model_to_path_map():
    for name in onnx_zoo_non_validated:
        test_name = name.split("/")[-2]
        model_path_map[test_name] = name

    for name in onnx_zoo_validated:
        test_name = '.'.join((name.split("/")[-1]).split('.')[:-2])
        model_path_map[test_name] = name


build_model_to_path_map()

url_map = lambda name : f'https://github.com/onnx/models/raw/refs/heads/main/{model_path_map[name]}'

meta_constructor = lambda is_validated, name : (lambda *args, **kwargs : OnnxModelZooDownloadableModel(is_validated, url_map(name),*args, **kwargs))

for t in set(onnx_zoo_non_validated).difference(custom_registry):
    t_split = t.split("/")[-2]
    register_test(meta_constructor(False, t_split), t_split)

for t in set(onnx_zoo_validated).difference(custom_registry):
    t_split = ".".join((t.split("/")[-1]).split(".")[:-2])
    register_test(meta_constructor(True, t_split), t_split)
