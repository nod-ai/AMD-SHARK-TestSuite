# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2EAMDSHARK_CHECK_DEF

# Create an instance of it for this test
E2EAMDSHARK_CHECK = dict(E2EAMDSHARK_CHECK_DEF)

# model origin: https://huggingface.co/jlsilva/facebook-opt-125m-gptq4bit
test_modelname = "facebook/opt-125m"
quantizedmodelname = "jlsilva/facebook-opt-125m-gptq4bit"
kwargs = {
    "torch_dtype": torch.float32,
    "trust_remote_code": True,
}
quantization_config = GPTQConfig(bits=8, disable_exllama=True)
kwargs["quantization_config"] = quantization_config
kwargs["device_map"] = "cpu"
model = AutoModelForCausalLM.from_pretrained(quantizedmodelname, **kwargs)
# model.output_hidden_states = False
tokenizer = AutoTokenizer.from_pretrained(test_modelname)
prompt = "What is nature of our existence?"
encoding = tokenizer(prompt, return_tensors="pt")
E2EAMDSHARK_CHECK["input"] = encoding["input_ids"].cpu()
E2EAMDSHARK_CHECK["output"] = model(E2EAMDSHARK_CHECK["input"])
E2EAMDSHARK_CHECK["output"] = (E2EAMDSHARK_CHECK["output"].logits, E2EAMDSHARK_CHECK["output"].past_key_values)
E2EAMDSHARK_CHECK["output_for_validation"] = [E2EAMDSHARK_CHECK["output"][0]]

model_response = model.generate(
    E2EAMDSHARK_CHECK["input"],
    do_sample=True,
    top_k=50,
    max_length=100,
    top_p=0.95,
    temperature=1.0,
)
print("Prompt:", prompt)
print("Response:", tokenizer.decode(model_response[0]).encode("utf-8"))
print("Input:", E2EAMDSHARK_CHECK["input"])
print("Output:", E2EAMDSHARK_CHECK["output"])
# For geneartive AI models, input is int and should be kept that way for
# casted models as well
E2EAMDSHARK_CHECK["inputtodtype"] = False

# Post process output to do:
# torch.nn.functional.softmax(output, -1)
# The output logits is the shape of (B, S, V).
# (batch size, sequence length, unormalized scores for each possible token in vocabulary)
# This way we create a probability distribution for each possible token (vocabulary)
# for each position in the sequence by doing softmax over the last dimension.
E2EAMDSHARK_CHECK["postprocess"] = [
    (torch.nn.functional.softmax, [-1], False, 0),
]
