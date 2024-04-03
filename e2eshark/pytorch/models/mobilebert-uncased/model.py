# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from transformers import MobileBertForSequenceClassification, AutoTokenizer

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

# model origin: https://huggingface.co/google/mobilebert-uncased
test_modelname = "google/mobilebert-uncased"
tokenizer = AutoTokenizer.from_pretrained(test_modelname)
model = MobileBertForSequenceClassification.from_pretrained(
    test_modelname,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    torchscript=True,
)
model.config.pad_token_id = None
model.to("cpu")
model.eval()
E2ESHARK_CHECK["input"] = torch.randint(2, (1, 128))
E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])
# logit
E2ESHARK_CHECK["output_for_validation"] = [E2ESHARK_CHECK["output"][0]]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
# For geneartive AI models, input is int and should be kept that way for
# casted models as well
E2ESHARK_CHECK["inputtodtype"] = False

# Post process output to do:
# torch.nn.functional.softmax(output, -1)
# The output logits is the shape of (B, L).
# (batch size, num labels)
# This way we create a probability distribution for each possible label
# when classifying sentence.
E2ESHARK_CHECK["postprocess"] = [
    (torch.nn.functional.softmax, [-1], False, 0),
]
