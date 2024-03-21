# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# from @aviator19941's gist : https://gist.github.com/aviator19941/4e7967bd1787c83ee389a22637c6eea7

import torch
from diffusers import UNet2DConditionModel, PNDMScheduler
import sys

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

def get_schedulers(model_id):
    # TODO: Robust scheduler setup on pipeline creation -- if we don't
    # set batch_size here, the SHARK schedulers will
    # compile with batch size = 1 regardless of whether the model
    # outputs latents of a larger batch size, e.g. SDXL.
    # However, obviously, searching for whether the base model ID
    # contains "xl" is not very robust.

    batch_size = 2 if "xl" in model_id.lower() else 1

    schedulers = dict()
    schedulers["PNDM"] = PNDMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    return schedulers

class SDXLScheduledUnet(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        scheduler_id,
        height,
        width,
        batch_size,
        hf_auth_token=None,
        precision="fp32",
        num_inference_steps=1,
        return_index=False,
    ):
        super().__init__()
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.scheduler = get_schedulers(hf_model_name)[scheduler_id]
        if scheduler_id == "PNDM":
            num_inference_steps = num_inference_steps - 1
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.is_scale_input_called = True
        self.return_index = return_index
        if "Euler" in scheduler_id:
            self.scheduler._step_index = torch.tensor(0, dtype=torch.float16)

        if precision == "fp16":
            try:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                    variant="fp16",
                )
            except:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                )
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                hf_model_name,
                subfolder="unet",
                auth_token=hf_auth_token,
                low_cpu_mem_usage=False,
            )

    def initialize(self, sample):
        height = sample.shape[-2] * 8
        width = sample.shape[-1] * 8
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
        add_time_ids = add_time_ids.repeat(sample.shape[0], 1).type(self.dtype)
        timesteps = self.scheduler.timesteps
        step_indexes = torch.tensor(len(timesteps))
        sample = sample * self.scheduler.init_noise_sigma
        return sample.type(self.dtype), add_time_ids, step_indexes

    def forward(
        self, sample, prompt_embeds, text_embeds, time_ids, guidance_scale, step_index
    ):
        with torch.no_grad():
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
            t = self.scheduler.timesteps[step_index]
            sample = self.scheduler.scale_model_input(sample, t)
            latent_model_input = torch.cat([sample] * 2)
            noise_pred = self.unet.forward(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            sample = self.scheduler.step(noise_pred, t, sample, return_dict=False)[0]
            return sample.type(self.dtype)


model = SDXLScheduledUnet("stabilityai/stable-diffusion-xl-base-1.0", "PNDM", 1024, 1024, 1, num_inference_steps=30)

sample = torch.rand(
    1, 4, 1024 // 8, 1024 // 8, dtype=torch.float32
)
sample, add_time_ids, steps = model.initialize(sample)

# timestep = torch.zeros(1, dtype=torch.int64)
prompt_embeds = torch.rand(2 * 1, 64, 2048, dtype=torch.float32)
text_embeds = torch.rand(2 * 1, 1280, dtype=torch.float32)

E2ESHARK_CHECK["input"] = [sample.float(), prompt_embeds.float(), text_embeds.float(), add_time_ids.float(), torch.tensor([7.5], dtype=torch.float32), torch.tensor([0], dtype=torch.int64)]

E2ESHARK_CHECK["output"] = model.forward(sample.float(), prompt_embeds.float(), text_embeds.float(), add_time_ids.float(), torch.tensor([7.5], dtype=torch.float32), torch.tensor([0], dtype=torch.int64))

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])