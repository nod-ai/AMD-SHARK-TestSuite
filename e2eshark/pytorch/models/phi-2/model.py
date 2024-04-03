# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from transformers import AutoModelForCausalLM, AutoTokenizer

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

# model origin: https://huggingface.co/microsoft/phi-2
test_modelname = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(test_modelname)
model = AutoModelForCausalLM.from_pretrained(
    test_modelname,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    torchscript=True,
)
model.to("cpu")
model.eval()
prompt = "What is nature of our existence?"
encoding = tokenizer(prompt, return_tensors="pt")
E2ESHARK_CHECK["input"] = encoding["input_ids"].cpu()
E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])
E2ESHARK_CHECK["output_for_validation"] = [E2ESHARK_CHECK["output"][0]]

model_response = model.generate(
    E2ESHARK_CHECK["input"],
    do_sample=True,
    top_k=50,
    max_length=100,
    top_p=0.95,
    temperature=1.0,
)
print("Prompt:", prompt)
print("Response:", tokenizer.decode(model_response[0]))
print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
# For geneartive AI models, input is int and should be kept that way for
# casted models as well
E2ESHARK_CHECK["inputtodtype"] = False

# Post process output to do:
# torch.nn.functional.softmax(output, -1)
# The output logits is the shape of (B, S, V).
# (batch size, sequence length, unormalized scores for each possible token in vocabulary)
# This way we create a probability distribution for each possible token (vocabulary)
# for each position in the sequence by doing softmax over the last dimension.
E2ESHARK_CHECK["postprocess"] = [
    (torch.nn.functional.softmax, [-1], False, 0),
]
