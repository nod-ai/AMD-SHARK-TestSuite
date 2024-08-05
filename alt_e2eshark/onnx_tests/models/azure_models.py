# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
from pathlib import Path
from e2e_testing import azutils
from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test

CACHE_DIR = os.getenv('CACHE_DIR')

class AzureDownloadableModel(OnnxModelInfo):
    def __init__(self, name: str, onnx_model_path: str):
        opset_version = 21
        if not CACHE_DIR:
            raise RuntimeError("Please specify a cache directory path in the CACHE_DIR environment variable for storing large model files.")
        self.cache_dir = os.path.join(CACHE_DIR, name)
        super().__init__(name, onnx_model_path, opset_version)

    def construct_model(self):
        # try to find a .onnx file in the test-run dir
        # if that fails, check for zip file in cache
        # if that fails, try to download and setup from azure, then search again for a .onnx file

        # TODO: make the zip file structure more uniform so we don't need to search for extracted files
        model_dir = str(Path(self.model).parent)

        def find_models(model_dir):
            # search for a .onnx file in the ./test-run/testname/ dir
            found_models = []
            for root, dirs, files in os.walk(model_dir):
                for name in files:
                    if name[-5:] == ".onnx":  
                        found_models.append(os.path.abspath(os.path.join(root, name)))
            return found_models

        found_models = find_models(model_dir) 

        if len(found_models) == 0:
            azutils.pre_test_onnx_model_azure_download(
                self.name, self.cache_dir, self.model
            )
            found_models = find_models(model_dir)
        if len(found_models) == 1:
            self.model = found_models[0]
            return
        if len(found_models) > 1:
            print(f'Found multiple model.onnx files: {found_models}')
            print(f'Picking the first model found to use: {found_models[0]}')
            self.model = found_models[0]
            return
        raise OSError(f"No onnx model could be found, downloaded, or extracted to {model_dir}")

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
