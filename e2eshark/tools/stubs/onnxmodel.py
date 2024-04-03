# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file constitutes end part of runmodel.py
# this is appended to the model.py in test dir

import numpy
import onnxruntime
import sys, argparse
import torch, pickle
from commonutils import getOutputTensorList, E2ESHARK_CHECK_DEF, postProcess

msg = "The script to run an ONNX model test"
parser = argparse.ArgumentParser(description=msg, epilog="")

parser.add_argument(
    "-d",
    "--todtype",
    choices=["default", "fp32", "fp16", "bf16"],
    default="default",
    help="If not default, casts model and input to given data type if framework supports model.to(dtype) and tensor.to(dtype)",
)
parser.add_argument(
    "-m",
    "--mode",
    choices=["direct", "onnx", "ort"],
    default="direct",
    help="Generate torch MLIR, ONNX or ONNX plus ONNX RT stub",
)
parser.add_argument(
    "-o",
    "--outfileprefix",
    default="model",
    help="Prefix of output files written by this model",
)
args = parser.parse_args()
if args.todtype != "default":
    print(
        "Onnx does not support model.to(dtype). Default dtype of the model will be used."
    )

runmode = args.mode
outfileprefix = args.outfileprefix
outfileprefix += "." + args.todtype
inputsavefilename = outfileprefix + ".input.pt"
outputsavefilename = outfileprefix + ".goldoutput.pt"


E2ESHARK_CHECK["postprocessed_output"] = postProcess(E2ESHARK_CHECK)
# TBD, remobe torch.save and use the .pkl instead
torch.save(E2ESHARK_CHECK["input"], inputsavefilename)
torch.save(E2ESHARK_CHECK["output"], outputsavefilename)

# Save the E2ESHARK_CHECK
with open("E2ESHARK_CHECK.pkl", "wb") as f:
    pickle.dump(E2ESHARK_CHECK, f)
