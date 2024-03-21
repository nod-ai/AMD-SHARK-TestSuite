# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree import runtime as ireert
import iree.compiler as ireec
from iree.compiler.ir import Context
import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import sys, argparse
import numpy as np

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)


class PromptEncoderModule(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        precision,
        hf_auth_token=None,
        do_classifier_free_guidance=True,
    ):
        super().__init__()
        self.torch_dtype = torch.float16 if precision == "fp16" else torch.float32
        self.text_encoder_model_1 = CLIPTextModel.from_pretrained(
            hf_model_name,
            subfolder="text_encoder",
            token=hf_auth_token,
        )
        self.text_encoder_model_2 = CLIPTextModelWithProjection.from_pretrained(
            hf_model_name,
            subfolder="text_encoder_2",
            token=hf_auth_token,
        )
        self.do_classifier_free_guidance = do_classifier_free_guidance

    #     self.tokenizer_1 = CLIPTokenizer.from_pretrained(
    #         hf_model_name,
    #         subfolder="tokenizer",
    #         token=hf_auth_token,
    #         model_max_length=max_length,
    #     )
    #     self.tokenizer_2 = CLIPTokenizer.from_pretrained(
    #         hf_model_name,
    #         subfolder="tokenizer_2",
    #         token=hf_auth_token,
    #         model_max_length=max_length,
    #     )
    # def tokenize(self, prompt, negative_prompt):
    #     text_input_ids_1 = self.tokenizer_1(
    #         prompt,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids
    #     uncond_input_ids_1 = self.tokenizer_2(
    #         negative_prompt,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids
    #     text_input_ids_2 = self.tokenizer_2(
    #         prompt,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids
    #     uncond_input_ids_2 = self.tokenizer_2(
    #         negative_prompt,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids
    #     return text_input_ids_1, uncond_input_ids_1, text_input_ids_2, uncond_input_ids_2

    def forward(
        self, text_input_ids_1, text_input_ids_2, uncond_input_ids_1, uncond_input_ids_2
    ):
        with torch.no_grad():
            prompt_embeds_1 = self.text_encoder_model_1(
                text_input_ids_1,
                output_hidden_states=True,
            )
            prompt_embeds_2 = self.text_encoder_model_2(
                text_input_ids_2,
                output_hidden_states=True,
            )
            neg_prompt_embeds_1 = self.text_encoder_model_1(
                uncond_input_ids_1,
                output_hidden_states=True,
            )
            neg_prompt_embeds_2 = self.text_encoder_model_2(
                uncond_input_ids_2,
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds_2[0]
            neg_pooled_prompt_embeds = neg_prompt_embeds_2[0]

            prompt_embeds_list = [
                prompt_embeds_1.hidden_states[-2],
                prompt_embeds_2.hidden_states[-2],
            ]
            neg_prompt_embeds_list = [
                neg_prompt_embeds_1.hidden_states[-2],
                neg_prompt_embeds_2.hidden_states[-2],
            ]

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            neg_prompt_embeds = torch.concat(neg_prompt_embeds_list, dim=-1)

            bs_embed, seq_len, _ = prompt_embeds.shape

            prompt_embeds = prompt_embeds.repeat(1, 1, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(
                bs_embed * 1, -1
            )
            add_text_embeds = pooled_prompt_embeds
            if self.do_classifier_free_guidance:
                neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(1, 1).view(
                    1, -1
                )
                neg_prompt_embeds = neg_prompt_embeds.repeat(1, 1, 1)
                neg_prompt_embeds = neg_prompt_embeds.view(bs_embed * 1, seq_len, -1)
                prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat(
                    [neg_pooled_prompt_embeds, add_text_embeds], dim=0
                )

            add_text_embeds = add_text_embeds.to(self.torch_dtype)
            prompt_embeds = prompt_embeds.to(self.torch_dtype)
            return prompt_embeds, add_text_embeds
        
model = PromptEncoderModule("stabilityai/stable-diffusion-xl-base-1.0", "fp32")

tokenizer_1 = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer",
)
tokenizer_2 = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer_2",
)
text_input_ids_list = []
uncond_input_ids_list = []

# Tokenize prompt and negative prompt.
tokenizers = [tokenizer_1, tokenizer_2]
for tokenizer in tokenizers:
    text_inputs = tokenizer(
        "A very fast car leaving a trail of fire as it screams along a mountain road, old school racing animation, retro 1980s anime style, 4k, motion blur, action shot, semi-realistic, nightwave, neon, tokyo",
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    uncond_input = tokenizer(
        "Watermark, blurry, oversaturated, low resolution, pollution",
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    uncond_input_ids = uncond_input.input_ids

    text_input_ids_list.extend([text_input_ids])
    uncond_input_ids_list.extend([uncond_input_ids])

E2ESHARK_CHECK["input"] = [text_input_ids_list[0], text_input_ids_list[1], uncond_input_ids_list[0], uncond_input_ids_list[1]]

E2ESHARK_CHECK["output"] = model.forward(text_input_ids_list[0], text_input_ids_list[1], uncond_input_ids_list[0], uncond_input_ids_list[1])

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
