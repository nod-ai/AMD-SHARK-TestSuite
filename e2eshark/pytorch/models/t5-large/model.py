# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from transformers import T5Model, AutoTokenizer

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

# model origin: https://huggingface.co/t5-large
test_modelname = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(test_modelname)
tokenization_kwargs = {
    "pad_to_multiple_of": 512,
    "padding": True,
    "return_tensors": "pt",
}
model = T5Model.from_pretrained(
    test_modelname,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    torchscript=True,
    return_dict=True,
)
model.to("cpu")
model.eval()
tokenizer("Studies have been shown that owning a dog is good for you", **tokenization_kwargs)
encoded_input_ids = tokenizer("Studies have been shown that owning a dog is good for you", 
    **tokenization_kwargs
).input_ids.cpu()
decoder_input_ids = tokenizer("Studies show that", 
    **tokenization_kwargs
).input_ids.cpu()
decoder_input_ids = model._shift_right(decoder_input_ids)
attention_mask = torch.ones(1, 512).to("cpu")
E2ESHARK_CHECK["input"] = [encoded_input_ids, attention_mask, decoder_input_ids]
E2ESHARK_CHECK["output"] = model(input_ids=encoded_input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
# last hidden state (contextual info, sequence of hidden-states at the output of the last layer of the decoder of the model.)
E2ESHARK_CHECK["output_for_validation"] = [E2ESHARK_CHECK["output"][0]]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
# For geneartive AI models, input is int and should be kept that way for
# casted models as well
E2ESHARK_CHECK["inputtodtype"] = False
