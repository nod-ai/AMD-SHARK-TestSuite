# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file constitutes end  part of runmodel.py
# this is appended to the model.py in test dir

import argparse
import pickle

from turbine_models.model_builder import HFTransformerBuilder
import shark_turbine.aot as aot
from commonutils import getOutputTensorList, E2ESHARK_CHECK_DEF, postProcess

msg = "The script to run a model test"
parser = argparse.ArgumentParser(description=msg, epilog="")

parser.add_argument(
    "-d",
    "--todtype",
    choices=["default", "fp32", "fp16", "bf16"],
    default="default",
    help="If not default, casts model and input to given data type if framework supports model.to(dtype) and tensor.to(dtype)",
)
parser.add_argument(
    "-o",
    "--outfileprefix",
    default="model",
    help="Prefix of output files written by this model",
)
args = parser.parse_args()
outfileprefix = args.outfileprefix + "." + args.todtype


def getTorchDType(dtypestr):
    if dtypestr == "fp32":
        return torch.float32
    elif dtypestr == "fp16":
        return torch.float16
    elif dtypestr == "bf16":
        return torch.bfloat16
    else:
        print("Unknown dtype {dtypestr} returning torch.float32")
        return torch.float32


if args.todtype != "default":
    # convert the model to given dtype
    dtype = getTorchDType(args.todtype)
    model = model.to(dtype)
    # not all model need the input re-casted
    if E2ESHARK_CHECK["inputtodtype"]:
        E2ESHARK_CHECK["input"] = E2ESHARK_CHECK["input"].to(dtype)
    E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])

# create hugging face transformer model
turbine_model = HFTransformerBuilder(
    example_input=E2ESHARK_CHECK["input"],
    upload_ir=False,
    model=model,
    compile_to_vmfb=False,
)
if isinstance(E2ESHARK_CHECK["input"], list):
    module = aot.export(model, *E2ESHARK_CHECK["input"])
else:
    module = aot.export(model, E2ESHARK_CHECK["input"])
module_str = str(module.mlir_module)
torch_mlir_name = outfileprefix + ".pytorch.torch.mlir"
with open(torch_mlir_name, "w+") as f:
        f.write(module_str)

inputsavefilename = outfileprefix + ".input.pt"
outputsavefilename = outfileprefix + ".goldoutput.pt"

test_input_list = E2ESHARK_CHECK["input"]
test_output_list = E2ESHARK_CHECK["output"]

if not isinstance(E2ESHARK_CHECK["input"], list):
    test_input_list = [E2ESHARK_CHECK["input"]]

if isinstance(test_output_list, tuple):
    # handles only nested tuples for now
    print(f"Found tuple {len(test_output_list)} {test_output_list}")
    test_output_list = getOutputTensorList(E2ESHARK_CHECK["output"])

# model result expected to be List[Tensors]
if not isinstance(test_output_list, list):
    test_output_list = [E2ESHARK_CHECK["output"]]

E2ESHARK_CHECK["input"] = [t.detach() for t in test_input_list]
E2ESHARK_CHECK["output"] = [t.detach() for t in test_output_list]

E2ESHARK_CHECK["postprocessed_output"] = postProcess(E2ESHARK_CHECK)
# TBD, move to using E2ESHARK_CHECK pickle save
torch.save(E2ESHARK_CHECK["input"], inputsavefilename)
torch.save(E2ESHARK_CHECK["output"], outputsavefilename)

# Save the E2ESHARK_CHECK
with open("E2ESHARK_CHECK.pkl", "wb") as tchkf:
    pickle.dump(E2ESHARK_CHECK, tchkf)
