# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import torch
from diffusers import AutoencoderKL

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)


class VaeModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        custom_vae="",
    ):
        super().__init__()
        self.vae = None
        if custom_vae in ["", None]:
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
        elif not isinstance(custom_vae, dict):
            try:
                # custom HF repo with no vae subfolder
                self.vae = AutoencoderKL.from_pretrained(
                    custom_vae,
                )
            except:
                # some larger repo with vae subfolder
                self.vae = AutoencoderKL.from_pretrained(
                    custom_vae,
                    subfolder="vae",
                )
        else:
            # custom vae as a HF state dict
            self.vae = AutoencoderKL.from_pretrained(
                hf_model_name,
                subfolder="vae",
            )
            self.vae.load_state_dict(custom_vae)

    def forward(self, inp):
        inp = 1 / 0.13025 * inp
        x = self.vae.decode(inp, return_dict=False)[0]
        return (x / 2 + 0.5).clamp(0, 1)

model = VaeModel("stabilityai/stable-diffusion-xl-base-1.0", "")

example_input = torch.ones(
    1,
    4,
    1024 // 8,
    1024 // 8,
    dtype=torch.float32,
)

E2ESHARK_CHECK["input"] = [example_input]

E2ESHARK_CHECK["output"] = model.forward(example_input)

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])