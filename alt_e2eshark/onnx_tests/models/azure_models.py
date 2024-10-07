# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path
from ..helper_classes import AzureDownloadableModel
from e2e_testing.registry import register_test
from e2e_testing.storage import load_test_txt_file
import onnxruntime

this_file = Path(__file__)
lists_dir = (this_file.parent).joinpath("external_lists")

model_names = load_test_txt_file(lists_dir.joinpath("shark-test-suite.txt"))
for i in range(1, 4):
    model_names += load_test_txt_file(
        lists_dir.joinpath(f"vai-hf-cnn-fp32-shard{i}.txt")
    )
    model_names += load_test_txt_file(lists_dir.joinpath(f"vai-int8-p0p1-shard{i}.txt"))
model_names += load_test_txt_file(lists_dir.joinpath("vai-vision-int8.txt"))

custom_registry = [
    "opt-125M-awq",
    "opt-125m-gptq",
    "DeepLabV3_resnet50_vaiq_int8",
]

# if the model has significant shape issues, consider applying basic optimizations before import by adding to this list:
basic_opt = [
    "gcvit_base",
    "gcvit_small",
    "gcvit_tiny",
    "gcvit_xtiny",
    "gcvit_xxtiny",
    "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    "swinv2_base_window16_256.ms_in1k",
    "swinv2_base_window8_256.ms_in1k",
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
    "swinv2_small_window16_256.ms_in1k",
    "swinv2_small_window8_256.ms_in1k",
    "swinv2_tiny_window16_256.ms_in1k",
    "swinv2_tiny_window8_256.ms_in1k",
    "xcit_large_24_p16_224",
    "xcit_large_24_p16_224_dist",
    "xcit_large_24_p16_384_dist",
    "xcit_large_24_p8_224",
    "xcit_large_24_p8_224_dist",
    "xcit_large_24_p8_384_dist",
    "xcit_medium_24_p16_224",
    "xcit_medium_24_p16_224_dist",
    "xcit_medium_24_p16_384_dist",
    "xcit_medium_24_p8_224",
    "xcit_medium_24_p8_224_dist",
    "xcit_medium_24_p8_384_dist",
    "xcit_nano_12_p16_224",
    "xcit_nano_12_p16_224_dist",
    "xcit_nano_12_p16_384_dist",
    "xcit_nano_12_p8_224",
    "xcit_nano_12_p8_224_dist",
    "xcit_nano_12_p8_384_dist",
    "xcit_small_12_p16_224",
    "xcit_small_12_p16_224_dist",
    "xcit_small_12_p16_384_dist",
    "xcit_small_12_p8_224",
    "xcit_small_12_p8_224_dist",
    "xcit_small_12_p8_384_dist",
    "xcit_small_24_p16_224",
    "xcit_small_24_p16_224_dist",
    "xcit_small_24_p16_384_dist",
    "xcit_small_24_p8_224",
    "xcit_small_24_p8_224_dist",
    "xcit_small_24_p8_384_dist",
    "xcit_tiny_12_p16_224",
    "xcit_tiny_12_p16_224_dist",
    "xcit_tiny_12_p16_384_dist",
    "xcit_tiny_12_p8_224",
    "xcit_tiny_12_p8_224_dist",
    "xcit_tiny_12_p8_384_dist",
    "xcit_tiny_24_p16_224",
    "xcit_tiny_24_p16_224_dist",
    "xcit_tiny_24_p16_384_dist",
    "xcit_tiny_24_p8_224",
    "xcit_tiny_24_p8_224_dist",
    "xcit_tiny_24_p8_384_dist",
]

custom_registry += basic_opt
# for simple models without dim params or additional customization, we should be able to register them directly with AzureDownloadableModel
# TODO: many of the models in the text files loaded from above will likely need to be registered with an alternative test info class.
for t in set(model_names).difference(custom_registry):
    register_test(AzureDownloadableModel, t)


class AzureWithOpt(AzureDownloadableModel):
    def apply_ort_basic_optimizations(self):
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        optimized_model = str(Path(self.model).parent.joinpath("model.optimized.onnx"))
        opt.optimized_model_filepath = optimized_model
        session = onnxruntime.InferenceSession(self.model, opt)
        self.model = optimized_model
        del session

    def construct_model(self):
        super().construct_model()
        self.apply_ort_basic_optimizations()


for t in basic_opt:
    register_test(AzureWithOpt, t)
