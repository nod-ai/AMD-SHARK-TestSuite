# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path
from ..helper_classes import AzureDownloadableModel
from e2e_testing.registry import register_test
from e2e_testing.storage import load_test_txt_file
import onnx
import onnxruntime

this_file = Path(__file__)
lists_dir = (this_file.parent).joinpath("external_lists")

model_names = load_test_txt_file(lists_dir.joinpath("amdshark-test-suite.txt"))
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

no_opset_update = [
    "dm_nfnet_f2.dm_in1k",
    # getting test runner crash for the following.
    # TODO: unblock these when we externalize weights on import
    "dm_nfnet_f3.dm_in1k",
    "dm_nfnet_f4.dm_in1k",
    "vit_base_r50_s16_384.orig_in21k_ft_in1k",
    "vit_small_r26_s32_224.augreg_in21k_ft_in1k",
    "vit_small_r26_s32_384.augreg_in21k_ft_in1k",
    "vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k",
    "vit_tiny_r_s16_p8_384.augreg_in21k_ft_in1k",
    "vit_large_r50_s32_224.augreg_in21k_ft_in1k",
    "vit_large_r50_s32_384.augreg_in21k_ft_in1k",
]

# if the model has significant shape issues, consider applying basic optimizations before import by adding to this list:
basic_opt = [
    "bat_resnext26ts.ch_in1k",
    "botnet26t_256",
    "coat_mini",
    "coat_tiny",
    "crossvit_15_240",
    "crossvit_15_dagger_240",
    "crossvit_15_dagger_408",
    "crossvit_18_240",
    "crossvit_18_dagger_240",
    "crossvit_18_dagger_408",
    "crossvit_9_240",
    "crossvit_9_dagger_240",
    "crossvit_base_240",
    "crossvit_small_240",
    "crossvit_tiny_240",
    "davit_base.msft_in1k",
    "davit_small.msft_in1k",
    "davit_tiny.msft_in1k",
    "eca_botnext26ts_256",
    "ecaresnet269d",
    "gcvit_base",
    "gcvit_small",
    "gcvit_tiny",
    "gcvit_xtiny",
    "gcvit_xxtiny",
    "jx_nest_base",
    "jx_nest_small",
    "jx_nest_tiny",
    "levit_128.fb_dist_in1k",
    "levit_128s.fb_dist_in1k",
    "levit_192.fb_dist_in1k",
    "levit_256.fb_dist_in1k",
    "levit_384.fb_dist_in1k",
    "maxvit_base_tf_224.in1k",
    "maxvit_base_tf_384.in1k",
    "maxvit_base_tf_384.in21k_ft_in1k",
    "maxvit_base_tf_512.in1k",
    "maxvit_base_tf_512.in21k_ft_in1k",
    "maxvit_large_tf_224.in1k",
    "maxvit_large_tf_384.in1k",
    "maxvit_large_tf_384.in21k_ft_in1k",
    "maxvit_large_tf_512.in1k",
    "maxvit_large_tf_512.in21k_ft_in1k",
    "maxvit_small_tf_224.in1k",
    "maxvit_small_tf_384.in1k",
    "maxvit_small_tf_512.in1k",
    "maxvit_tiny_tf_224.in1k",
    "maxvit_tiny_tf_384.in1k",
    "maxvit_tiny_tf_512.in1k",
    "maxvit_xlarge_tf_384.in21k_ft_in1k",
    "maxvit_xlarge_tf_512.in21k_ft_in1k",
    "mvitv2_base",
    "mvitv2_large",
    "mvitv2_small",
    "mvitv2_tiny",
    "sebotnet33ts_256",
    "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    "swinv2_base_window16_256.ms_in1k",
    "swinv2_base_window8_256.ms_in1k",
    "swinv2_cr_small_224.sw_in1k",
    "swinv2_cr_small_ns_224.sw_in1k",
    "swinv2_cr_tiny_ns_224.sw_in1k",
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
    "swinv2_small_window16_256.ms_in1k",
    "swinv2_small_window8_256.ms_in1k",
    "swinv2_tiny_window16_256.ms_in1k",
    "swinv2_tiny_window8_256.ms_in1k",
    "twins_svt_base",
    "twins_svt_large",
    "twins_svt_small",
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

remove_metadata_props = [
    "resnetv2_50x1_bit.goog_in21k_ft_in1k_vaiq",
]

custom_registry += basic_opt
custom_registry += no_opset_update
custom_registry += remove_metadata_props

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


class AzureWithOptAndNoOpsetVersion(AzureWithOpt):
    def __init__(self, name: str, onnx_model_path: str):
        super().__init__(name, onnx_model_path)
        self.opset_version = None


for t in no_opset_update:
    register_test(AzureWithOptAndNoOpsetVersion, t)


class AzureRemoveMetadataProps(AzureDownloadableModel):
    def construct_model(self):
        super().construct_model()
        self.remove_duplicate_metadata_props()

    def remove_duplicate_metadata_props(self):
        model = onnx.load(self.model)
        metadata = model.metadata_props
        # unhashable type, so manually check for duplicates
        to_remove = []
        for i, m0 in enumerate(metadata):
            for j in range(i + 1, len(metadata)):
                if m0 == metadata[j]:
                    to_remove.append(i)
        num_removed = 0
        for i in to_remove:
            _ = metadata.pop(i - num_removed)
        mod_model_path = str(Path(self.model).parent / "model.modified.onnx")
        onnx.save(model, mod_model_path)
        self.model = mod_model_path


register_test(AzureRemoveMetadataProps, "resnetv2_50x1_bit.goog_in21k_ft_in1k_vaiq")

from ..helper_classes import TruncatedModel, get_truncated_constructor

const = get_truncated_constructor(TruncatedModel, AzureDownloadableModel, "mvitv2_tiny")
