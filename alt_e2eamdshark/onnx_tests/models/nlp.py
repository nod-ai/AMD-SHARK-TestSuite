# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from ..helper_classes import AzureDownloadableModel
from e2e_testing.registry import register_test
from e2e_testing.storage import load_test_txt_file
from e2e_testing.onnx_utils import get_node_shape_from_dim_param_dict
from e2e_testing.storage import TestTensors
import onnxruntime as ort
import numpy
from typing import Optional
import onnx

this_file = Path(__file__)
lists_dir = (this_file.parent).joinpath("external_lists")

model_names = []
for i in [1, 2, 3]:
    model_names += load_test_txt_file(lists_dir.joinpath(f"nlp-shard{i}.txt"))

large_file_size_models = [
    "model--YuisekinAI-mistral-0.7B--yuiseki",
    "model--financial-summarization-pegasus--human-centered-summarization",
    "model--finetuned_gpt2-large_sst2_negation0.0001_pretrainedTrue_epochs1--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.001_pretrainedTrue_epochs1--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.001_pretrainedTrue_epochs2--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.001_pretrainedTrue_epochs3--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.01--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.01_pretrainedFalse_epochs10--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.01_pretrainedTrue_epochs1--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.05--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.0_pretrainedFalse--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.1_pretrainedFalse_epochs10--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.1_pretrainedTrue_epochs1--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.2--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.5--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.5_pretrainedFalse--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.8--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.8_pretrainedFalse--yuhuizhang",
    "model--flan-t5-large-samsum--oguuzhansahin",
    "model--long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP--pszemraj",
    "model--long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP13--pszemraj",
    "model--long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP14--pszemraj",
    "model--long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP15--pszemraj",
    "model--long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP17--pszemraj",
    "model--m2m100_418M-finetuned-kde4-en-to-pt_BR--danhsf",
    "model--m2m100_418M-fr--Jour",
    "model--m2m100_418M-fr--NDugar",
    "model--m2m100_418M-ja--vivek-307306",
    "model--mT5-base-HunSum-1--SZTAKI-HLT",
    "model--mT5_multilingual_XLSum--csebuetnlp",
    "model--manifestoberta-xlm-roberta-56policy-topics-sentence-2023-1-1--manifesto-project",
    "model--my_xlm-roberta-large-finetuned-conll03--BahAdoR0101",
    "model--pegasus-cnn_dailymail--google",
    "model--pegasus-large-book-summary--pszemraj",
    "model--pegasus-large-booksum--cnicu",
    "model--pegasus-large-summary-explain--pszemraj",
    "model--pegasus-xsum--google",
    "model--pegasus_summarizer--tuner007",
    "model--roberta-ner-multilingual--julian-schelb",
    "model--t5-large-finetuned-xsum-cnn--sysresearch101",
    "model--tglobal-large-booksum-WIP4-r1--pszemraj",
    "model--xlm-roberta-large-squad2--deepset",
    "model--xlmr-large-qa-fa--m3hrdadfi",
]


def dim_param_constructor(dim_param_dict):
    class AzureWithDimParams(AzureDownloadableModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.name in large_file_size_models:
                self.construct_model()
                self.update_model_without_ext_data()
                self.opset_version = None
            # TODO: check opset versions
            # print(f"Opset: {self.opset_version}")
            # self.update_opset_version_and_overwrite()

        def update_dim_param_dict(self):
            self.dim_param_dict = dim_param_dict

        def construct_inputs(self):
            """Overrides the parent class method to construct sample inputs with the correct dimensions."""
            default_inputs = super().construct_inputs()

            tensors = list(default_inputs.data)

            self.update_sess_options()
            session = ort.InferenceSession(self.model, self.sess_options)

            # nlp specific overrides
            rng = numpy.random.default_rng(19)
            for i, node in enumerate(session.get_inputs()):
                if node.name == "token_type_ids" or node.name == "attention_mask":
                    int_dims = get_node_shape_from_dim_param_dict(
                        node, self.dim_param_dict
                    )
                    tensors[i] = rng.integers(0, 2, size=int_dims, dtype=numpy.int64)
            if self.name == "model--s2t-medium-librispeech-asr--facebook":
                tensors[1] = rng.integers(
                    0, 2, size=tensors[1].shape, dtype=numpy.int64
                )
            default_sample_inputs = TestTensors(tuple(tensors))
            return default_sample_inputs

    return AzureWithDimParams


# Default dimension parameters for NLP models

# TODO: Verify these dim params are valid for each model, or load them from a json
default_nlp_params = {
    "batch_size": 1,
    "seq_len": 128,
    "encoder_sequence_length": 128,
    "decoder_sequence_length": 128,
}
dim_aliases = [
    {"seq_len", "sequence_length"},
]
for alias_set in dim_aliases:
    found = set(alias_set).intersection(default_nlp_params.keys())
    if len(found) > 1:
        # check if the values are the same
        val = default_nlp_params[next(iter(found))]
        if not all(default_nlp_params[alias] == val for alias in found):
            raise ValueError(
                f"Multiple aliases for the same dimension have different values: {found}"
            )

    aliases = alias_set - found
    for alias in aliases:
        default_nlp_params[alias] = default_nlp_params[next(iter(found))]

# Custom dimension parameters for specific models
custom_dim_params = {
    # Add custom dimension parameters for specific models here
    # Example:
    # "model_name": {"batch_size": 1, "seq_len": 256, "custom_dim": 64},
    "model--s2t-medium-librispeech-asr--facebook": {
        "batch_size": 1,
        "feature_size": 80,
        "encoder_sequence_length": 80,
        "decoder_sequence_length": 80,
    },
}

default_param_models = set(model_names).difference(set(custom_dim_params.keys()))

# Register models with default parameters
for model_name in default_param_models:
    register_test(dim_param_constructor(default_nlp_params), model_name)


# Register models with custom parameters
for model_name, dim_params in custom_dim_params.items():
    register_test(dim_param_constructor(dim_params), model_name)

# You can add more customizations or specific handling for certain models here
