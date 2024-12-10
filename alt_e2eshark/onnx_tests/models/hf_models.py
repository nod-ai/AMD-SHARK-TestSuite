# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import requests

from pathlib import Path
from ..helper_classes import HfDownloadableModel
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors, load_test_txt_file

from transformers import AutoTokenizer
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
]


def build_repo_to_model_map():
    # The elements of the list are 2-tuples,
    # containing the repository path of the model,
    # and its task-type.
    hf_models_list = [
        (
            load_test_txt_file(lists_dir.joinpath(f"hf-model-paths/hf-{task}-model-list.txt")),
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


class HfModelWithTokenizers(HfDownloadableModel):
    def construct_inputs(self):
        model_dir = str(Path(self.model).parent)
        prompt = ["Deeds will not be less valiant because they are unpraised."]
        tokenizer = AutoTokenizer.from_pretrained(self.model_repo_path)
        inputs = tokenizer(prompt, return_tensors="pt").values()

        test_tensors = TestTensors(tuple(inputs))
        test_tensors.save_to(model_dir)
        return test_tensors


class HfModelWithImageSetup(HfDownloadableModel):
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

        model_dir = str(Path(self.model).parent)
        inputs = setup_test_image()
        test_tensors = TestTensors((inputs,))
        test_tensors.save_to(model_dir)
        return test_tensors


for t in model_repo_map.keys():
    match model_repo_map[t][1]:
        case (
            "question-answering"
            | "text-generation"
            | "multiple-choice"
            | "feature-extraction"
            | "fill-mask"
            | "text-classification"
        ):
            register_test(meta_constructor_tokenizer(t), t)
        case "image-classification" | "object-detection" | "image-segmentation":
            register_test(meta_constructor_cv(t), t)
        case _:
            register_test(meta_constructor(t), t)
