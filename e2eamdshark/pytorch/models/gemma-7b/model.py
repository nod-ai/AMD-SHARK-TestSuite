import sys, argparse
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2EAMDSHARK_CHECK_DEF

# Create an instance of it for this test
E2EAMDSHARK_CHECK = dict(E2EAMDSHARK_CHECK_DEF)

# model origin: https://huggingface.co/google/gemma-7b
test_modelname = "google/gemma-7b"
token = os.getenv("HF_TOKEN", "")
if token == "":
    print(f"Huggingface token is required for this model ({test_modelname})." +
          "\nPlease set HF_TOKEN environment variable to a valid Huggingface token value and rerun the test." +
          "\nExample: HF_TOKEN=your_token python run.py")
tokenizer = AutoTokenizer.from_pretrained(test_modelname, token=token)
model = AutoModelForCausalLM.from_pretrained(
    test_modelname,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
    torchscript=True,
    token=token,
)
model.to("cpu")
model.eval()
model.output_hidden_states = False
prompt = "What is nature of our existence?"
encoding = tokenizer(prompt, return_tensors="pt")
E2EAMDSHARK_CHECK["input"] = encoding["input_ids"].cpu()
E2EAMDSHARK_CHECK["output"] = model(E2EAMDSHARK_CHECK["input"])
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
