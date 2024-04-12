# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys, argparse
import torch
import torch.nn as nn
import torch_mlir


# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)


class op_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 4))

    def forward(self, x):
        return self.layers(x)


model = op_linear()
E2ESHARK_CHECK["input"] = torch.randn(8, 3)
E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])
print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
# Do not enforce any particular strategy for getting torch MLIR
# By default set it to None, set it to
# 'compile' : to force using torch_mllir.compile
# 'fximport' : to force using PyTorch 2.0 Fx Import
test_torchmlircompile = None
