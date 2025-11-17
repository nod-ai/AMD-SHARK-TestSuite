import sys, argparse
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTQConfig,
)

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2Eamdshark_CHECK_DEF

# Create an instance of it for this test
E2Eamdshark_CHECK = dict(E2Eamdshark_CHECK_DEF)

# model origin: https://huggingface.co/TheBloke/Llama-2-7B-GPTQ
test_modelname = "TheBloke/Llama-2-7B-GPTQ"
kwargs = {
    "torch_dtype": torch.float32,
    "trust_remote_code": True,
}
quantization_config = GPTQConfig(bits=8, disable_exllama=True)
kwargs["quantization_config"] = quantization_config
kwargs["device_map"] = "cpu"
model = AutoModelForCausalLM.from_pretrained(
    test_modelname, low_cpu_mem_usage=True, attn_implementation="eager", **kwargs
)
# model.output_hidden_states = False
tokenizer = AutoTokenizer.from_pretrained(test_modelname)
prompt = "What is nature of our existence?"
encoding = tokenizer(prompt, return_tensors="pt")
E2Eamdshark_CHECK["input"] = encoding["input_ids"].cpu()
E2Eamdshark_CHECK["output"] = model(E2Eamdshark_CHECK["input"])
E2Eamdshark_CHECK["output"] = (E2Eamdshark_CHECK["output"].logits, E2Eamdshark_CHECK["output"].past_key_values)
E2Eamdshark_CHECK["output_for_validation"] = [E2Eamdshark_CHECK["output"][0]]
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

# Post process output to do:
# torch.nn.functional.softmax(output, -1)
# The output logits is the shape of (B, S, V).
# (batch size, sequence length, unormalized scores for each possible token in vocabulary)
# This way we create a probability distribution for each possible token (vocabulary)
# for each position in the sequence by doing softmax over the last dimension.
E2Eamdshark_CHECK["postprocess"] = [
    (torch.nn.functional.softmax, [-1], False, 0),
]
