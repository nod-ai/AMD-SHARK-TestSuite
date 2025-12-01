# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from e2e_testing.registry import register_test
from e2e_testing.storage import load_test_txt_file
from ..helper_classes import HfOnnxModelZooNonLegacyModel

this_file = Path(__file__)
lists_dir = (this_file.parent).joinpath("external_lists")

hf_onnx_zoo_models_non_legacy = []
for hf_onnx_model_file in lists_dir.glob("hf_onnx_model_zoo_non_legacy_*.txt"):
    hf_onnx_zoo_models_non_legacy.extend(load_test_txt_file(hf_onnx_model_file))

meta_constructor = lambda model_name:  (lambda *args, **kwargs : HfOnnxModelZooNonLegacyModel(model_name, *args, **kwargs))

for model_name in hf_onnx_zoo_models_non_legacy:
    register_test(meta_constructor(model_name), model_name)
