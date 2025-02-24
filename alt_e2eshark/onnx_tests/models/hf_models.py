# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import requests
import torch

from pathlib import Path

from ..helper_classes import HfDownloadableModel
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors, load_test_txt_file

from transformers import (
    AutoTokenizer,
)

from torchvision import transforms
from PIL import Image

this_file = Path(__file__)
lists_dir = (this_file.parent).joinpath("external_lists")

model_repo_map = {}

task_list = [
    "text-generation",
    "feature-extraction",
    "fill-mask",
    "question-answering",
    "text-classification",
    "token-classification",
    "multiple-choice",
    "image-classification",
    "object-detection",
    "image-segmentation",
    "semantic-segmentation",
    "audio-classification",
]

# These are NLP model names that have a mismatch between tokenizer
# outputs and the model inputs, but do not fall under a particular
# model task. If a huge number of models that can be grouped
# in a common category fall under this list, a new meta_constructor
# should be created for them.
update_tokenizer_input_names = [
    "hf_paraphrase-multilingual-MiniLM-L12-v2",
    "hf_all-MiniLM-L6-v2",
    "hf_jina-embeddings-v2-small-en",
    "hf_all-MiniLM-L12-v2",
    "hf_msmarco-MiniLM-L6-cos-v5",
    "hf_paraphrase-MiniLM-L6-v2",
    "hf_multi-qa-MiniLM-L6-cos-v",
    "hf_bge-small-en-v1.5",
    "hf_llm-embedder",
    "hf_bert-base-nli-mean-tokens",
    "hf_LaBSE-en-ru",
    "hf_bge-large-en-v1.5",
    "hf_bert-base-turkish-cased-mean-nli-stsb-tr",
    "hf_mxbai-embed-large-v1",
    "hf_bge-base-en-v1.5",
    "hf_phobert-large-finetuned",
    "hf_bertweet-base-sentiment-analysis",
    "hf_bertweet-base-emotion-analysis",
    "hf_phobert-base-finetuned",
    "hf_bge-large-zh-v1.5",
    "hf_UAE-Large-V1",
    "hf_GIST-small-Embedding-v0",
    "hf_rubert-tiny2",
    "hf_bge-small-en",
    "hf_bge-large-en",
    "hf_GIST-Embedding-v0",
    "hf_GIST-large-Embedding-v0",
    "hf_paraphrase-MiniLM-L3-v2",
    "hf_LaBSE",
    "hf_opensearch-neural-sparse-encoding-doc-v2-distill",
    "hf_snowflake-arctic-embed-m",
    "hf_phobert-base-v2",
    "hf_phobert-base",
    "hf_bertweet-base",
    "hf_distilbert-base-uncased-finetuned-sst-2-english",
    "hf_checkpoints_1_16",
    "hf_mdeberta-v3-base-squad2",
    "hf_Medical-NER",
    "hf_deberta-v3-base-squad2",
    "hf_DeBERTa-v3-base-mnli-fever-anli",
    "hf_Debertalarg_model_multichoice_Version2",
    "hf_deberta-v3-large-squad2",
    "hf_output",
    "hf_piiranha-v1-detect-personal-information",
    "hf_deberta-v3-base-absa-v1.1",
    "hf_deberta-large-mnli",
    "hf_deberta-v3-base-zeroshot-v1.1-all-33",
    "hf_nli-deberta-v3-base",
    "hf_deberta-base",
    "hf_deberta_finetuned_pii",
    "hf_deberta-v3-large_boolq",
    "hf_deberta-v3-large",
    "hf_checkpoints_3_14",
    "hf_deberta-v3-base_finetuned_ai4privacy_v2",
    "hf_mxbai-rerank-xsmall-v1",
    "hf_mDeBERTa-v3-base-mnli-xnli",
    "hf_mDeBERTa-v3-xnli-ft-bs-multiple-choice",
    "hf_mxbai-rerank-base-v1",
    "hf_deberta-v3-base-injection",
    "hf_content",
    "hf_deberta-v3-base",
    "hf_deberta-v3-small",
    "hf_mdeberta-v3-base",
    "hf_deberta-v2-base-japanese",
    "hf_deberta-v2-base-japanese-char-wwm",
    "hf_deberta-v3-xsmall",
    "hf_distilbert_distilbert-base-uncased-15-epoch",
    "hf_distilbert-base-cased-distilled-squad",
    "hf_multi-qa-MiniLM-L6-cos-v1",
    "hf_distilbert-base-multilingual-cased-sentiments-student",
    "hf_distilbert-base-uncased-distilled-squad",
    "hf_distilbert_multiple_choice",
    "hf_distilbert-base-uncased",
    "hf_tiny-distilbert-base-cased-distilled-squad",
    "hf_keyphrase-extraction-distilbert-inspec",
    "hf_distilbert-extractive-qa-project",
    "hf_distilcamembert-base-ner",
    "hf_msmarco-distilbert-dot-v5",
    "hf_distilbert-SBD-en-judgements-laws",
    "hf_camembert-ner",
    "hf_distilbert-base-nli-stsb-mean-tokens",
    "hf_msmarco-distilbert-base-v4",
    "hf_distilbert-base-multilingual-cased-ner-hrl",
    "hf_distilbert-base-cased-finetuned-conll03-english",
    "hf_msmarco-distilbert-base-tas-b",
    "hf_distilbert-NER",
    "hf_distilbert_science_multiple_choice",
    "hf_multi-qa-distilbert-cos-v1",
    "hf_msmarco-distilbert-cos-v5",
    "hf_distilbert-base-nli-mean-tokens",
    "hf_distilbert-base-multilingual-cased",
    "hf_distilbert-base-cased",
]

# Add a basic_opt list to apply O1 to the models.
basic_opt = []


def get_tokenizer_from_model_path(model_repo_path: str, cache_dir: str | Path):
    trust_remote_code = False

    name = model_repo_path.split("/")[-1]
    if 'kobert' in name.lower():
        trust_remote_code = True

    return AutoTokenizer.from_pretrained(model_repo_path, cache_dir=cache_dir, trust_remote_code=True)


def build_repo_to_model_map():
    # The elements of the list are 2-tuples,
    # containing the repository path of the model,
    # and its task-type.
    hf_models_list = [
        (
            load_test_txt_file(
                lists_dir.joinpath(f"hf-model-paths/hf-{task}-model-list.txt")
            ),
            task,
        )
        for task in task_list
    ]

    for info in hf_models_list:
        for model in info[0]:
            model_name = model.split("/")[-1]
            model_repo_map[model_name] = (
                model,
                info[1],
            )


build_repo_to_model_map()

meta_constructor = lambda m_name: (
    lambda *args, **kwargs: HfDownloadableModel(
        model_repo_map[m_name][0], model_repo_map[m_name][1], *args, **kwargs
    )
)

# Meta constructor for all models that could use tokenizers to generate input tensors.
meta_constructor_tokenizer = lambda m_name: (
    lambda *args, **kwargs: HfModelWithTokenizers(
        model_repo_map[m_name][0], model_repo_map[m_name][1], *args, **kwargs
    )
)

# Meta constructor for all models that require image input.
meta_constructor_cv = lambda m_name: (
    lambda *args, **kwargs: HfModelWithImageSetup(
        model_repo_map[m_name][0], model_repo_map[m_name][1], *args, **kwargs
    )
)

# Meta constructor for all multiple choice models.
meta_constructor_random_input = lambda m_name: (
    lambda *args, **kwargs: HfModelWithRandomInput(
        model_repo_map[m_name][0], model_repo_map[m_name][1], *args, **kwargs
    )
)

# Meta constructor for all multiple choice models.
meta_constructor_multiple_choice = lambda m_name: (
    lambda *args, **kwargs: HfModelMultipleChoice(
        model_repo_map[m_name][0], model_repo_map[m_name][1], *args, **kwargs
    )
)


class HfModelWithTokenizers(HfDownloadableModel):
    def export_model(self, optim_level: str | None = None):
        # We won't need optim_level.
        del optim_level
        super().export_model("O1" if self.name in basic_opt else None)

    def construct_inputs(self):
        prompt = ["Deeds will not be less valiant because they are unpraised."]

        tokenizer = get_tokenizer_from_model_path(self.model_repo_path, self.cache_dir)
        if self.name in update_tokenizer_input_names:
            tokenizer.model_input_names = ["input_ids", "attention_mask"]

        tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = (*list(tokens.values()),)

        self.input_name_to_shape_map = {k: v.shape for (k, v) in tokens.items()}

        test_tensors = TestTensors(inputs)
        return test_tensors


class HfModelWithRandomInput(HfDownloadableModel):
    def export_model(self, optim_level: str | None = None):
        # We won't need optim_level.
        del optim_level
        super().export_model("O1" if self.name in basic_opt else None)

    def construct_inputs(self):
        inputs = torch.randn(1, 4, 16000)

        self.input_name_to_shape_map = {'input_ids': torch.Size([16000, 4]), 'attention_mask': torch.Size([16000, 4])}

        test_tensors = TestTensors(inputs)
        return test_tensors


class HfModelMultipleChoice(HfDownloadableModel):
    def export_model(self, optim_level: str | None = None):
        # We won't need optim_level.
        del optim_level
        super().export_model("O1" if self.name in basic_opt else None)

    def construct_inputs(self):
        tokenizer = get_tokenizer_from_model_path(self.model_repo_path, self.cache_dir)

        prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
        candidate1 = "The law does not apply to croissants and brioche."
        candidate2 = "The law applies to baguettes."

        # For Deberta/Roberta Models, the ONNX export will have a mismatch in the number of inputs.
        # See https://stackoverflow.com/questions/75948679/deberta-onnx-export-does-not-work-for-token-type-ids.
        if (
            "deberta" in self.name
            or "roberta" in self.name
            or self.name in update_tokenizer_input_names
        ):
            tokenizer.model_input_names = ["input_ids", "attention_mask"]

        tokens = tokenizer(
            [[prompt, candidate1], [prompt, candidate2]],
            return_tensors="pt",
            padding=True,
        )

        inputs = (*[input.unsqueeze_(0) for input in tokens.values()],)

        self.input_name_to_shape_map = {k: v.shape for (k, v) in tokens.items()}

        test_tensors = TestTensors(inputs)
        return test_tensors


class HfModelWithImageSetup(HfDownloadableModel):
    def export_model(self, optim_level: str | None = None):
        # We won't need optim_level.
        del optim_level
        super().export_model("O1" if self.name in basic_opt else None)

    def construct_inputs(self):
        def setup_test_image(height=224, width=224):
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            img = Image.open(requests.get(url, stream=True).raw)

            resize = transforms.Resize([height, width])
            img = resize(img)

            # Define a transform to convert
            # the image to torch tensor
            img_ycbcr = img.convert("YCbCr")

            # Convert the image to Torch tensor
            to_tensor = transforms.ToTensor()
            img_ycbcr = to_tensor(img_ycbcr)
            img_ycbcr.unsqueeze_(0)
            return img_ycbcr

        inputs = setup_test_image()

        # TODO: Figure out a way to remove the hardcoded
        # input name ('pixel_values') and load from model
        # config/input list.
        self.input_name_to_shape_map = {"pixel_values": inputs.shape}

        test_tensors = TestTensors((inputs,))
        return test_tensors


for t in model_repo_map.keys():
    match model_repo_map[t][1]:
        case (
            "question-answering"
            | "text-generation"
            | "feature-extraction"
            | "fill-mask"
            | "text-classification"
            | "token-classification"
        ):
            register_test(meta_constructor_tokenizer(t), t)
        case (
            "image-classification"
            | "object-detection"
            | "image-segmentation"
            | "semantic-segmentation"
        ):
            register_test(meta_constructor_cv(t), t)
        case "audio-classification":
            register_test(meta_constructor_random_input(t), t)
        case "multiple-choice":
            register_test(meta_constructor_multiple_choice(t), t)
        case _:
            register_test(meta_constructor(t), t)
