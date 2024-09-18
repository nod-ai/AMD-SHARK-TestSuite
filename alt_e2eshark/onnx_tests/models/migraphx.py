# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..helper_classes import AzureDownloadableModel
from e2e_testing.registry import register_test

# TODOs:
# 1. just update the opset versions and re-upload to azure.
# 2. get tf models into onnx and upload
# 3. setup dim params for other misc models
# 4. reupload cadence model 1

def dim_param_constructor(dim_param_dict):
    class AzureWithDimParams(AzureDownloadableModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if (
                self.name == "migraphx_sd__unet__model"
                or self.name == "migraphx_sdxl__unet__model"
            ):
                # trying to update opset version seems to cause a crash or other issues.
                self.opset_version = None
                # even with the following, ort fails to allocate memory with default session options:
                self.sess_options.add_session_config_entry(
                    "use_device_allocator_for_initializers", "1"
                )
            self.update_opset_version_and_overwrite()

        def update_dim_param_dict(self):
            self.dim_param_dict = dim_param_dict

    return AzureWithDimParams


ORT_model_names = [
    "migraphx_ORT__bert_base_cased_1",  # batch_size, seq_len
    "migraphx_ORT__bert_base_uncased_1",  # batch_size, seq_len
    "migraphx_ORT__bert_large_uncased_1", # batch_size, seq_len
    "migraphx_ORT__distilgpt2_1",  # batch_size, seq_len
    "migraphx_ORT__onnx_models__bert_base_cased_1_fp16_gpu",  # batch_size, seq_len
    "migraphx_ORT__onnx_models__bert_large_uncased_1_fp16_gpu",  # batch_size, seq_len
    "migraphx_ORT__onnx_models__distilgpt2_1_fp16_gpu",  # batch_size, seq_len
]

llm_dict_0 = {"batch_size": 1, "seq_len": 128}
for name in ORT_model_names:
    register_test(dim_param_constructor(llm_dict_0), name)

static_dim_model_names = [
    "migraphx_bert__bert-large-uncased",  # need to specify input range for indices input [-2,1]
    "migraphx_cadene__dpn92i1",  # need to give names to nodes??? did this locally, need to reupload
    "migraphx_cadene__inceptionv4i16",
    "migraphx_cadene__resnext101_64x4di1",
    "migraphx_cadene__resnext101_64x4di16",
    "migraphx_onnx-misc__taau_low_res_downsample_d2s_for_infer_time_fp16_opset11",  # fp16 resize issue
    "migraphx_pytorch-examples__wlang_gru",
    "migraphx_pytorch-examples__wlang_lstm",  # also needs node names
    "migraphx_torchvision__densenet121i32",
    "migraphx_torchvision__inceptioni1",
    "migraphx_torchvision__inceptioni32",
    "migraphx_torchvision__resnet50i1",
    "migraphx_torchvision__resnet50i64",
    "migraphx_huggingface-transformers__bert_mrpc8",  # need to specify input range for indices input [-2,1]
]

for name in static_dim_model_names:
    register_test(dim_param_constructor(None), name)

misc_models = {
    "migraphx_agentmodel__AgentModel": {"batch": 1},
    "migraphx_bert__bertsquad-12": {
        "unk__492": 1,
        "unk__493": 1,
        "unk__494": 1,
        "unk__495": 1,
    },
    "migraphx_mlperf__bert_large_mlperf": {
        "batch_size": 1
    },  # need to specify input range for indices input [-2,1]
    "migraphx_mlperf__resnet50_v1": {"unk__616": 1},
    "migraphx_onnx-model-zoo__gpt2-10": {
        "input1_dynamic_axes_1": 1,
        "input1_dynamic_axes_2": 1,
        "input1_dynamic_axes_3": 1,
    },
    "migraphx_sd__unet__model": {
        "batch": 1,
        "channels": 4,
        "height": 512,
        "width": 512,
        "sequence": 64,
    },
    "migraphx_models__whisper-tiny-decoder" : {"batch_size" : 1, "decoder_sequence_length" : 64, "encoder_sequence_length / 2" : 32},
    "migraphx_models__whisper-tiny-encoder" : {"batch_size" : 1, "feature_size" : 80, "encoder_sequence_length" : 64},
    # this one crashes for some reason...
    "migraphx_sdxl__unet__model" : {"batch_size" : 1, "num_channels" : 4, "height" : 512, "width" : 512, "steps" : 2, "sequence_length" : 64}
}

for key, dim_param in misc_models.items():
    register_test(dim_param_constructor(dim_param), key)


### -------------------------------- ###
#        Truncated Model Tests         #
### -------------------------------- ###

# some smaller repros for failed to legalize cmd.stream.dispatch:

need_repro_dict = {
    "migraphx_ORT__bert_base_cased_1" : ["cased" , 4, "MatMul"],
    "migraphx_ORT__bert_base_uncased_1" : ["uncased", 1, "Transpose"],
    "migraphx_ORT__distilgpt2_1" : ["gpt", 3, "Add"],
    "migraphx_ORT__onnx_models__distilgpt2_1_fp16_gpu" : ["gptf16", 3, "Add"],
    "migraphx_onnx-model-zoo__gpt2-10" : ["gpt2_10", 0, "NonZero"],
}

from ..helper_classes import TruncatedModel, get_trucated_constructor

trunc_const = lambda key : get_trucated_constructor(TruncatedModel, dim_param_constructor(llm_dict_0), key)

for (key, value) in need_repro_dict.items():
    register_test(trunc_const(key)(value[1], value[2]), f"mi_trunc_{value[0]}_{value[1]}_{value[2]}")
