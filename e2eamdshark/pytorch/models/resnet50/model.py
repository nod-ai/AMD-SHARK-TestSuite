# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import torch
from torchvision.models import resnet50, ResNet50_Weights

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2EAMDSHARK_CHECK_DEF

# Create an instance of it for this test
E2EAMDSHARK_CHECK = dict(E2EAMDSHARK_CHECK_DEF)

test_modelname = "resnet50"
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

E2EAMDSHARK_CHECK["input"] = E2EAMDSHARK_CHECK["input"] = torch.randn(1, 3, 224, 224)
E2EAMDSHARK_CHECK["output"] = model(E2EAMDSHARK_CHECK["input"])
print("Input:", E2EAMDSHARK_CHECK["input"])
print("Output:", E2EAMDSHARK_CHECK["output"])
