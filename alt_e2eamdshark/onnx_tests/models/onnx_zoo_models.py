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
onnx_zoo_non_validated = load_test_txt_file(
    lists_dir.joinpath("onnx_model_zoo_non_validated_paths.txt")
)
onnx_zoo_validated = load_test_txt_file(
    lists_dir.joinpath("onnx_model_zoo_validated_paths.txt")
)
onnx_zoo_unsupported = load_test_txt_file(
    lists_dir.joinpath("onnx_model_zoo_unsupported.txt")
)

onnx_zoo_non_validated = list(
    set(onnx_zoo_non_validated).difference(set(onnx_zoo_unsupported))
)
onnx_zoo_validated = list(set(onnx_zoo_validated).difference(set(onnx_zoo_unsupported)))

# Putting this inside the class contructor will
# call this repeatedly, which is wasteful.
model_path_map = {}


def build_model_to_path_map():
    for name in onnx_zoo_non_validated:
        test_name = name.split("/")[-2]
        model_path_map[test_name] = name

    for name in onnx_zoo_validated:
        test_name = ".".join((name.split("/")[-1]).split(".")[:-2])
        model_path_map[test_name] = name


build_model_to_path_map()

url_map = (
    lambda name: f"https://github.com/onnx/models/raw/refs/heads/main/{model_path_map[name]}"
)

meta_constructor = lambda is_validated, name: (
    lambda *args, **kwargs: OnnxModelZooDownloadableModel(
        is_validated, url_map(name), *args, **kwargs
    )
)
meta_constructor_opt = lambda is_validated, name: (
    lambda *args, **kwargs: OnnxModelZooWithOpt(
        is_validated, url_map(name), *args, **kwargs
    )
)
meta_constructor_opt_no_opset = lambda is_validated, name: (
    lambda *args, **kwargs: OnnxModelZooWithOptAndNoOpsetVersion(
        is_validated, url_map(name), *args, **kwargs
    )
)
meta_constructor_remove_metadata = lambda is_validated, name: (
    lambda *args, **kwargs: OnnxModelZooRemoveMetadataProps(
        is_validated, url_map(name), *args, **kwargs
    )
)

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
    ("swin_large_patch4_window7_224_in22k_Opset16_timm", False),
    ("swin_base_patch4_window12_384_Opset17_timm", False),
    ("swin_base_patch4_window12_384_in22k_Opset16_timm", False),
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
    ("vit_l_16_Opset16_torch_hub", False),
    ("vit_b_16_Opset18_torch_hub", False),
    ("vit_b_32_Opset18_torch_hub", False),
    ("vit_l_32_Opset17_torch_hub", False),
    ("vit_b_32_Opset17_torch_hub", False),
    ("vit_b_16_Opset17_torch_hub", False),
    ("vit_l_16_Opset17_torch_hub", False),
    ("vit_b_16_Opset16_torch_hub", False),
    ("vit_l_16_Opset18_torch_hub", False),
    ("vit_l_32_Opset18_torch_hub", False),
    ("vit_b_32_Opset16_torch_hub", False),
    ("vit_l_32_Opset16_torch_hub", False),
    ("blenderbot_Opset16_transformers", False),
    ("convnext_small_Opset18_torch_hub", False),
    ("dm_nfnet_f0_Opset16_timm", False),
    ("dm_nfnet_f0_Opset17_timm", False),
    ("dm_nfnet_f1_Opset16_timm", False),
    ("dm_nfnet_f1_Opset17_timm", False),
    ("dm_nfnet_f2_Opset16_timm", False),
    ("dm_nfnet_f2_Opset17_timm", False),
    ("dm_nfnet_f3_Opset16_timm", False),
    ("dm_nfnet_f3_Opset17_timm", False),
    ("dm_nfnet_f4_Opset16_timm", False),
    ("dm_nfnet_f4_Opset17_timm", False),
    ("jx_nest_base_Opset16_timm", False),
    ("jx_nest_base_Opset17_timm", False),
    ("jx_nest_base_Opset18_timm", False),
    ("jx_nest_small_Opset16_timm", False),
    ("jx_nest_small_Opset17_timm", False),
    ("jx_nest_small_Opset18_timm", False),
    ("jx_nest_tiny_Opset16_timm", False),
    ("jx_nest_tiny_Opset17_timm", False),
    ("jx_nest_tiny_Opset18_timm", False),
    ("m2m100_Opset17_transformers", False),
    ("resnetv2_101x1_bitm_Opset16_timm", False),
    ("resnetv2_101x1_bitm_Opset17_timm", False),
    ("resnetv2_101x1_bitm_in21k_Opset16_timm", False),
    ("resnetv2_101x1_bitm_in21k_Opset17_timm", False),
    ("resnetv2_152x2_bit_teacher_384_Opset16_timm", False),
    ("resnetv2_152x2_bit_teacher_384_Opset17_timm", False),
    ("resnetv2_152x2_bit_teacher_Opset16_timm", False),
    ("resnetv2_152x2_bit_teacher_Opset17_timm", False),
    ("resnetv2_152x2_bitm_Opset16_timm", False),
    ("resnetv2_152x2_bitm_Opset17_timm", False),
    ("resnetv2_50x1_bit_distilled_Opset16_timm", False),
    ("resnetv2_50x1_bit_distilled_Opset17_timm", False),
    ("resnetv2_50x1_bitm_Opset16_timm", False),
    ("resnetv2_50x1_bitm_Opset17_timm", False),
    ("resnetv2_50x1_bitm_in21k_Opset16_timm", False),
    ("resnetv2_50x1_bitm_in21k_Opset17_timm", False),
    ("resnetv2_50x3_bitm_Opset16_timm", False),
    ("resnetv2_50x3_bitm_Opset17_timm", False),
    ("tf_efficientnet_b0_Opset16_timm", False),
    ("tf_efficientnet_b0_Opset17_timm", False),
    ("tf_efficientnet_b0_ap_Opset16_timm", False),
    ("tf_efficientnet_b0_ap_Opset17_timm", False),
    ("tf_efficientnet_b0_ns_Opset16_timm", False),
    ("tf_efficientnet_b0_ns_Opset17_timm", False),
    ("tf_efficientnet_b1_Opset16_timm", False),
    ("tf_efficientnet_b1_Opset17_timm", False),
    ("tf_efficientnet_b1_ap_Opset16_timm", False),
    ("tf_efficientnet_b1_ap_Opset17_timm", False),
    ("tf_efficientnet_b1_ns_Opset16_timm", False),
    ("tf_efficientnet_b1_ns_Opset17_timm", False),
    ("tf_efficientnet_b2_Opset16_timm", False),
    ("tf_efficientnet_b2_Opset17_timm", False),
    ("tf_efficientnet_b2_ap_Opset16_timm", False),
    ("tf_efficientnet_b2_ap_Opset17_timm", False),
    ("tf_efficientnet_b2_ns_Opset16_timm", False),
    ("tf_efficientnet_b2_ns_Opset17_timm", False),
    ("tf_efficientnet_b3_Opset16_timm", False),
    ("tf_efficientnet_b3_Opset17_timm", False),
    ("tf_efficientnet_b3_ap_Opset16_timm", False),
    ("tf_efficientnet_b3_ap_Opset17_timm", False),
    ("tf_efficientnet_b3_ns_Opset16_timm", False),
    ("tf_efficientnet_b3_ns_Opset17_timm", False),
    ("tf_efficientnet_b4_Opset16_timm", False),
    ("tf_efficientnet_b4_Opset17_timm", False),
    ("tf_efficientnet_b4_ap_Opset16_timm", False),
    ("tf_efficientnet_b4_ap_Opset17_timm", False),
    ("tf_efficientnet_b4_ns_Opset16_timm", False),
    ("tf_efficientnet_b4_ns_Opset17_timm", False),
    ("tf_efficientnet_b5_Opset16_timm", False),
    ("tf_efficientnet_b5_Opset17_timm", False),
    ("tf_efficientnet_b5_ap_Opset16_timm", False),
    ("tf_efficientnet_b5_ap_Opset17_timm", False),
    ("tf_efficientnet_b5_ns_Opset16_timm", False),
    ("tf_efficientnet_b5_ns_Opset17_timm", False),
    ("tf_efficientnet_b6_Opset16_timm", False),
    ("tf_efficientnet_b6_Opset17_timm", False),
    ("tf_efficientnet_b6_ap_Opset16_timm", False),
    ("tf_efficientnet_b6_ap_Opset17_timm", False),
    ("tf_efficientnet_b6_ns_Opset16_timm", False),
    ("tf_efficientnet_b6_ns_Opset17_timm", False),
    ("tf_efficientnet_b7_Opset16_timm", False),
    ("tf_efficientnet_b7_Opset17_timm", False),
    ("tf_efficientnet_b7_ap_Opset16_timm", False),
    ("tf_efficientnet_b7_ap_Opset17_timm", False),
    ("tf_efficientnet_b7_ns_Opset16_timm", False),
    ("tf_efficientnet_b7_ns_Opset17_timm", False),
    ("tf_efficientnet_b8_Opset16_timm", False),
    ("tf_efficientnet_b8_Opset17_timm", False),
    ("tf_efficientnet_b8_ap_Opset16_timm", False),
    ("tf_efficientnet_b8_ap_Opset17_timm", False),
    ("tf_efficientnet_el_Opset16_timm", False),
    ("tf_efficientnet_el_Opset17_timm", False),
    ("tf_efficientnet_el_Opset18_timm", False),
    ("tf_efficientnet_em_Opset16_timm", False),
    ("tf_efficientnet_em_Opset17_timm", False),
    ("tf_efficientnet_em_Opset18_timm", False),
    ("tf_efficientnet_es_Opset16_timm", False),
    ("tf_efficientnet_es_Opset17_timm", False),
    ("tf_efficientnet_es_Opset18_timm", False),
    ("tf_efficientnet_l2_ns_475_Opset16_timm", False),
    ("tf_efficientnet_l2_ns_475_Opset17_timm", False),
    ("tf_efficientnet_l2_ns_Opset16_timm", False),
    ("tf_efficientnet_l2_ns_Opset17_timm", False),
    ("tf_efficientnet_lite0_Opset16_timm", False),
    ("tf_efficientnet_lite0_Opset17_timm", False),
    ("tf_efficientnet_lite0_Opset18_timm", False),
    ("tf_efficientnet_lite1_Opset16_timm", False),
    ("tf_efficientnet_lite1_Opset17_timm", False),
    ("tf_efficientnet_lite1_Opset18_timm", False),
    ("tf_efficientnet_lite2_Opset16_timm", False),
    ("tf_efficientnet_lite2_Opset17_timm", False),
    ("tf_efficientnet_lite2_Opset18_timm", False),
    ("tf_efficientnet_lite3_Opset16_timm", False),
    ("tf_efficientnet_lite3_Opset17_timm", False),
    ("tf_efficientnet_lite3_Opset18_timm", False),
    ("tf_efficientnet_lite4_Opset16_timm", False),
    ("tf_efficientnet_lite4_Opset17_timm", False),
    ("tf_efficientnet_lite4_Opset18_timm", False),
    ("tf_efficientnetv2_b0_Opset16_timm", False),
    ("tf_efficientnetv2_b0_Opset17_timm", False),
    ("tf_efficientnetv2_b1_Opset16_timm", False),
    ("tf_efficientnetv2_b1_Opset17_timm", False),
    ("tf_efficientnetv2_b2_Opset16_timm", False),
    ("tf_efficientnetv2_b2_Opset17_timm", False),
    ("tf_efficientnetv2_b3_Opset16_timm", False),
    ("tf_efficientnetv2_b3_Opset17_timm", False),
    ("tf_efficientnetv2_l_Opset16_timm", False),
    ("tf_efficientnetv2_l_Opset17_timm", False),
    ("tf_efficientnetv2_l_in21ft1k_Opset16_timm", False),
    ("tf_efficientnetv2_l_in21ft1k_Opset17_timm", False),
    ("tf_efficientnetv2_l_in21k_Opset16_timm", False),
    ("tf_efficientnetv2_l_in21k_Opset17_timm", False),
    ("tf_efficientnetv2_m_Opset16_timm", False),
    ("tf_efficientnetv2_m_Opset17_timm", False),
    ("tf_efficientnetv2_m_in21ft1k_Opset16_timm", False),
    ("tf_efficientnetv2_m_in21ft1k_Opset17_timm", False),
    ("tf_efficientnetv2_m_in21k_Opset16_timm", False),
    ("tf_efficientnetv2_m_in21k_Opset17_timm", False),
    ("tf_efficientnetv2_s_Opset16_timm", False),
    ("tf_efficientnetv2_s_Opset17_timm", False),
    ("tf_efficientnetv2_s_in21ft1k_Opset16_timm", False),
    ("tf_efficientnetv2_s_in21ft1k_Opset17_timm", False),
    ("tf_efficientnetv2_s_in21k_Opset16_timm", False),
    ("tf_efficientnetv2_s_in21k_Opset17_timm", False),
    ("tf_efficientnetv2_xl_in21ft1k_Opset16_timm", False),
    ("tf_efficientnetv2_xl_in21ft1k_Opset17_timm", False),
    ("tf_efficientnetv2_xl_in21k_Opset16_timm", False),
    ("tf_efficientnetv2_xl_in21k_Opset17_timm", False),
    ("tf_mixnet_l_Opset16_timm", False),
    ("tf_mixnet_l_Opset17_timm", False),
    ("tf_mixnet_m_Opset16_timm", False),
    ("tf_mixnet_m_Opset17_timm", False),
    ("tf_mixnet_s_Opset16_timm", False),
    ("tf_mixnet_s_Opset17_timm", False),
    ("tf_mobilenetv3_large_075_Opset16_timm", False),
    ("tf_mobilenetv3_large_075_Opset17_timm", False),
    ("tf_mobilenetv3_large_100_Opset16_timm", False),
    ("tf_mobilenetv3_large_100_Opset17_timm", False),
    ("tf_mobilenetv3_large_minimal_100_Opset16_timm", False),
    ("tf_mobilenetv3_large_minimal_100_Opset17_timm", False),
    ("tf_mobilenetv3_large_minimal_100_Opset18_timm", False),
    ("tf_mobilenetv3_small_075_Opset16_timm", False),
    ("tf_mobilenetv3_small_075_Opset17_timm", False),
    ("tf_mobilenetv3_small_100_Opset16_timm", False),
    ("tf_mobilenetv3_small_100_Opset17_timm", False),
    ("tf_mobilenetv3_small_minimal_100_Opset16_timm", False),
    ("tf_mobilenetv3_small_minimal_100_Opset17_timm", False),
    ("tf_mobilenetv3_small_minimal_100_Opset18_timm", False),
    ("twins_svt_base_Opset16_timm", False),
    ("twins_svt_base_Opset17_timm", False),
    ("twins_svt_large_Opset16_timm", False),
    ("twins_svt_large_Opset17_timm", False),
    ("twins_svt_small_Opset16_timm", False),
    ("twins_svt_small_Opset17_timm", False),
    ("xcit_nano_12_p8_224_Opset16_timm", False),
    ("funnel_Opset16_transformers", False),
    ("funnel_Opset17_transformers", False),
    ("funnel_Opset18_transformers", False),
    ("funnelbase_Opset16_transformers", False),
    ("funnelbase_Opset18_transformers", False),
    # Validated models
    ("candy-8", True),
    ("rain-princess-8", True),
    ("mosaic-8", True),
    ("udnie-8", True),
    ("pointilism-8", True),
    ("squeezenet1.0-6", True),
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
    def __init__(
        self, name: str, onnx_model_path: str, is_validated: bool, model_url: str
    ):
        super().__init__(
            name=name,
            onnx_model_path=onnx_model_path,
            is_validated=is_validated,
            model_url=model_url,
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
