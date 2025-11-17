# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch, sys
import torch.nn as nn

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2Eamdshark_CHECK_DEF

# Create an instance of it for this test
E2Eamdshark_CHECK = dict(E2Eamdshark_CHECK_DEF)


class mlp_8192x2432x9728(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2432, 9728),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


model = mlp_8192x2432x9728()
E2Eamdshark_CHECK["input"] = torch.randn(8192, 2432)
E2Eamdshark_CHECK["output"] = model(E2Eamdshark_CHECK["input"]).detach()
print("Input:", E2Eamdshark_CHECK["input"])
print("Output:", E2Eamdshark_CHECK["output"])
