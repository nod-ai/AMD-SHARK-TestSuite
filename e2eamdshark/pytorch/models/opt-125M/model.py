# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from transformers import OPTForCausalLM, AutoTokenizer

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2Eamdshark_CHECK_DEF

# Create an instance of it for this test
E2Eamdshark_CHECK = dict(E2Eamdshark_CHECK_DEF)

# model origin: https://huggingface.co/facebook/opt-125M
test_modelname = "facebook/opt-125M"
tokenizer = AutoTokenizer.from_pretrained(test_modelname)
model = OPTForCausalLM.from_pretrained(
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
E2Eamdshark_CHECK["input"] = encoding["input_ids"].cpu()
E2Eamdshark_CHECK["output"] = model(E2Eamdshark_CHECK["input"])
model_response = model.generate(
    E2Eamdshark_CHECK["input"],
    do_sample=True,
    top_k=50,
    max_length=100,
    top_p=0.95,
    temperature=1.0,
)
print("Prompt:", prompt)
print("Response:", tokenizer.decode(model_response[0]).encode("utf-8"))
print("Input:", E2Eamdshark_CHECK["input"])
print("Output:", E2Eamdshark_CHECK["output"])
# For geneartive AI models, input is int and should be kept that way for
# casted models as well
E2Eamdshark_CHECK["inputtodtype"] = False
