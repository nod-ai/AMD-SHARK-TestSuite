# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os, sys, argparse
import onnx


# Given an onxx model, find unique set of ops
def uniqueOnnxOps(model):
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    return ops


# Given an onnx model, return a dictionary of ops with frequency
def frequencyOfOPs(model):
    ops = {}
    for node in model.graph.node:
        if node.op_type in ops:
            ops[node.op_type] += 1
        else:
            ops[node.op_type] = 1
    return ops


if __name__ == "__main__":
    msg = "The script to print contents of an ONNX ProtoBuf file."
    parser = argparse.ArgumentParser(description=msg, epilog="")
    parser.add_argument(
        "inputfiles",
        nargs="*",
        help="Input ONNX file(s)",
    )
    parser.add_argument(
        "-u", "--uniqueops", action="store_true", help="Find unique ops in given file"
    )
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print in human readable format"
    )
    parser.add_argument(
        "-s", "--signature", action="store_true", help="Get model input output information"
    )
    parser.add_argument(
        "-f",
        "--frequency",
        action="store_true",
        help="Print number of occurrences i.e. frquency of ops",
    )

    args = parser.parse_args()
    filelist = args.inputfiles
    for onnxfile in filelist:
        if not os.path.exists(onnxfile):
            print("The given file ", onnxfile, " does not exist\n")
            sys.exit(1)

        model = onnx.load(onnxfile)
        # If it gets past it, model was opened successfully
        print("File:", onnxfile, "\n")

        if args.print:
            ofilename = os.path.basename(onnxfile) + ".json"
            ofile = open(ofilename, "w")
            print(model, file=ofile)
            ofile.close()
        if args.uniqueops:
            ops = uniqueOnnxOps(model)
            print("Number of unique ops:", len(ops), "\nOps: ", ops, "\n")
        if args.frequency:
            ops = frequencyOfOPs(model)
            sortedops = dict(
                sorted(ops.items(), key=lambda item: item[1], reverse=True)
            )
            count = 0
            for k, v in sortedops.items():
                print(k, ":", v)
                count += v
            print("Total instances of ops: ", count)
        if args.signature:
            # Get the input and output nodes of the model
            ofilename = os.path.basename(onnxfile) + "signature.json"
            with open(ofilename, "w") as ofile:
                print("INPUTS: \n", file=ofile)
                print(model.graph.input, file=ofile)
                print("------------------------\nOUTPUTS: \n", file=ofile)
                print(model.graph.output, file=ofile)
