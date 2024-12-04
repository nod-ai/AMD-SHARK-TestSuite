# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import onnx
import onnxruntime
from typing import List, Tuple

from e2e_testing.registry import register_test
from e2e_testing.storage import load_test_txt_file
from ..helper_classes import OnnxModelZooDownloadableModel


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

meta_constructor = lambda is_validated, name : (lambda *args, **kwargs : OnnxModelZooDownloadableModel(is_validated, url_map(name), *args, **kwargs))
meta_constructor_opt = lambda is_validated, name : (lambda *args, **kwargs : OnnxModelZooWithOpt(is_validated, url_map(name), *args, **kwargs))
meta_constructor_opt_no_opset = lambda is_validated, name : (lambda *args, **kwargs : OnnxModelZooWithOptAndNoOpsetVersion(is_validated, url_map(name), *args, **kwargs))
meta_constructor_remove_metadata = lambda is_validated, name : (lambda *args, **kwargs : OnnxModelZooRemoveMetadataProps(is_validated, url_map(name), *args, **kwargs))

# The custom registry list for ONNX Zoo Models.
# Elements of the registry are 2-tuples, where the first element
# is the model name, and the second is a boolean value indicating
# whether the model is validated (True) or not (False).
custom_registry: List[Tuple[str, bool]] = []

no_opset_update: List[Tuple[str, bool]] = []

# if the model has significant shape issues, consider applying
# basic optimizations before import by adding to this list.

basic_opt: List[Tuple[str, bool]] = [
    ("swin_large_patch4_window12_384_in22k_Opset17_timm", False),
    ("swin_base_patch4_window12_384_in22k_Opset17_timm", False),
    ("swin_large_patch4_window7_224_Opset16_timm", False),
    ("swin_base_patch4_window7_224_in22k_Opset16_timm", False),
    ("swin_s3_base_224_Opset16_timm", False),
    ("swin_small_patch4_window7_224_Opset16_timm", False),
    ("swin_base_patch4_window7_224_in22k_Opset17_timm", False),
    ("swin_large_patch4_window12_384_in22k_Opset16_timm", False),
    ("swin_large_patch4_window7_224_in22k_Opset17_timm", False),
    ("canine_Opset18_transformers", False),
    ("swin_large_patch4_window7_224_Opset17_timm", False),
    ("swin_base_patch4_window12_384_Opset16_timm", False),
    ("swin_small_patch4_window7_224_Opset17_timm", False),
    ("swin_large_patch4_window12_384_Opset16_timm", False),
    ("swin_large_patch4_window12_384_Opset17_timm", False),
    ("swin_s3_small_224_Opset16_timm", False),
    ("swin_s3_small_224_Opset17_timm", False),
    ("swin_s3_tiny_224_Opset17_timm", False),
    ("swin_base_patch4_window7_224_Opset17_timm", False),
    ("swin_tiny_patch4_window7_224_Opset17_timm", False),
    ("swin_base_patch4_window7_224_Opset16_timm", False),
    ("swin_tiny_patch4_window7_224_Opset16_timm", False),
    ("canine_Opset17_transformers", False),
    ("swin_large_patch4_window7_224_in22k_Opset16_timm", False),
    ("swin_base_patch4_window12_384_Opset17_timm", False),
    ("swin_base_patch4_window12_384_in22k_Opset16_timm", False),
    ("canine_Opset16_transformers", False),
    ("swin_s3_tiny_224_Opset16_timm", False),
    ("swin_s3_base_224_Opset17_timm", False),
    ("xlnetlmhead_Opset18_transformers", False),
    ("xlnet_Opset16_transformers", False),
    ("xlnetlmhead_Opset17_transformers", False),
    ("xlnet_Opset18_transformers", False),
    ("xlnetlmhead_Opset16_transformers", False),
    ("xlnet_Opset17_transformers", False),
    ("swin_t_Opset17_torch_hub", False),
    ("swin_t_Opset18_torch_hub", False),
    ("swin_s_Opset16_torch_hub", False),
    ("swin_t_Opset16_torch_hub", False),
    ("swin_s_Opset18_torch_hub", False),
    ("swin_b_Opset16_torch_hub", False),
    ("swin_b_Opset18_torch_hub", False),
    ("swin_s_Opset17_torch_hub", False),
    ("swin_b_Opset17_torch_hub", False),
    ("rain-princess-8", False),
    ("mosaic-8", False),
    ("udnie-8", False),
    ("pointilism-8", False),
    ("candy-8", False)
]


remove_metadata_props: List[Tuple[str, bool]] = []

custom_registry += basic_opt
custom_registry += no_opset_update
custom_registry += remove_metadata_props


custom_registry_model_names = list(map(lambda x: model_path_map[x[0]], custom_registry))
for t in set(onnx_zoo_non_validated).difference(custom_registry_model_names):
    t_split = t.split("/")[-2]
    register_test(meta_constructor(False, t_split), t_split)

for t in set(onnx_zoo_validated).difference(custom_registry_model_names):
    t_split = ".".join((t.split("/")[-1]).split(".")[:-2])
    register_test(meta_constructor(True, t_split), t_split)


class OnnxModelZooWithOpt(OnnxModelZooDownloadableModel):
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
    register_test(meta_constructor_opt(t[1], t[0]), t[0])


class OnnxModelZooWithOptAndNoOpsetVersion(OnnxModelZooWithOpt):
    def __init__(self, name: str, onnx_model_path: str, is_validated: bool, model_url: str):
        super().__init__(
            name=name,
            onnx_model_path=onnx_model_path,
            is_validated=is_validated,
            model_url=model_url
        )
        self.opset_version = None


for t in no_opset_update:
    register_test(meta_constructor_opt_no_opset(t[1], t[0]), t[0])


class OnnxModelZooRemoveMetadataProps(OnnxModelZooDownloadableModel):
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


for t in remove_metadata_props:
    register_test(meta_constructor_remove_metadata(t[1], t[0]), t[0])
