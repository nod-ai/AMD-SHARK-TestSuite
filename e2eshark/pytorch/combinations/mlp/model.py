# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch, sys
import torch.nn as nn

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)


class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 3 input, 4 output
            nn.Linear(3, 4),
            nn.ReLU(),
            # 3 input, 5 output
            nn.Linear(4, 5),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


model = mlp()
E2ESHARK_CHECK["input"] = torch.randn(8, 3)
E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"]).detach()
print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
