# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os, time, glob, sys, zipfile, shutil
from multiprocessing import Pool
import argparse
import numpy as np
import torch, io
import struct, pickle, tabulate, statistics
from pathlib import Path
import shutil
import warnings
import datetime
import simplejson
import json
from multiprocessing import Manager
from tools.aztestsetup import pre_test_onnx_models_azure_download
from zipfile import ZipFile
from _run_helper import (
    getTestsList,
    getTestKind,
    changeToTestDir,
    concatenateFiles,
    loadE2eSharkCheckDictionary,
    uploadToBlobStorage,
    unzipONNXFile,
    loadTorchSave,
    getShapeString,
    loadRawBinaryAsTorchSensor,
    writeInferenceInputBinFile,
    getTestsListFromFile,
)

# Need to allow invocation of run.py from anywhere
sys.path.append(Path(__file__).parent)
from tools.stubs.commonutils import applyPostProcessPipeline


def launchCommand(args, scriptcommand, commandslog):
    if args.verbose:
        print("Launching:", scriptcommand, "[ Proc:", os.getpid(), "]")
    try:
        commandslog.write(scriptcommand)
        commandslog.write("\n")
        ret = os.system(scriptcommand)
        return ret
    except OSError as errormsg:
        print(
            "Invoking ",
            scriptcommand,
            " failed:",
            errormsg,
        )
        return 1


def logAndReturn(
    commandslog,
    timelog,
    resultdict,
    retval,
    uploadtestsList,
    cleanup,
    testName,
    uploadDict,
    dateAndTime,
):

    delete_list = ["mlir", "vmfb"]
    upload_list = ["mlir"]

    # Loop through everything in folder in current working directory
    if uploadtestsList or cleanup:
        listOfItems = os.listdir(os.getcwd())
        for item in listOfItems:
            file_type = item.split(".")[-1]
            if testName in uploadtestsList and file_type in upload_list:
                identifier = testName.replace("/", "_") + "/" + dateAndTime + "/" + item
                uploadToBlobStorage(
                    os.path.abspath(item), identifier, testName, uploadDict
                )
            if cleanup:
                if file_type in delete_list:  # If it isn't in the list for retaining
                    os.remove(item)  # Remove the item

    pickle.dump(resultdict, timelog)
    timelog.close()
    commandslog.close()
    return retval


def runOnnxToTorchMLIRGeneration(
    testName,
    modelname,
    mode,
    args,
    phases,
    scriptcommand,
    commandslog,
    timelog,
    onnxfilename,
    torchmlirOutputfilename,
    resultdict,
    uploadtestsList,
    cleanup,
    uploadDict,
    dateAndTime,
    torch_mlir_pythonpath,
):

    # If a torch mlit build is provided, use that else use iree-import-onnx
    if SHARED_TORCH_MLIR_BUILD:
        # start phases[1]
        curphase = phases[1]
        # Import ONNX into torch MLIR as torch.operator custom OP
        torchonnxfilename = modelname + "." + args.todtype + ".torch-onnx.mlir"
        logfilename = curphase + ".log"
        scriptcommand = (
            f"{torch_mlir_pythonpath} python -m torch_mlir.tools.import_onnx "
            + "--opset-version=21 "
            + onnxfilename
            + " -o "
            + torchonnxfilename
            + " 1> "
            + logfilename
            + " 2>&1"
        )
        start = time.time()
        if launchCommand(args, scriptcommand, commandslog):
            print("Test", testName, "failed [" + curphase + "]")
            end = time.time()
            resultdict[curphase] = ["failed", end - start]
            return logAndReturn(
                commandslog,
                timelog,
                resultdict,
                1,
                uploadtestsList,
                cleanup,
                testName,
                uploadDict,
                dateAndTime,
            )
        end = time.time()
        resultdict[curphase] = ["passed", end - start]

        # Lower torch ONNX to torch MLIR
        # start phases[2]
        curphase = phases[2]
        logfilename = curphase + ".log"
        commandstring = "/bin/torch-mlir-opt"
        if args.torchtolinalg:
            commandstring += " -pass-pipeline='builtin.module(func.func(convert-torch-onnx-to-torch),"
            commandstring += (
                "torch-lower-to-backend-contract,func.func(cse,canonicalize),"
            )
            commandstring += "torch-backend-to-linalg-on-tensors-backend-pipeline)' "
        else:
            commandstring += " -pass-pipeline='builtin.module(func.func(convert-torch-onnx-to-torch),"
            commandstring += (
                "torch-lower-to-backend-contract,func.func(cse,canonicalize))' "
            )
        # TORCH_MLIR_BUILD = path_config["TORCH_MLIR_BUILD"]
        # print(f"In RunTest - torch mlir build - {SHARED_TORCH_MLIR_BUILD}")
        scriptcommand = (
            SHARED_TORCH_MLIR_BUILD
            + commandstring
            + torchonnxfilename
            + " > "
            + torchmlirOutputfilename
            + " 2>"
            + logfilename
        )

        start = time.time()
        if launchCommand(args, scriptcommand, commandslog):
            print("Test", testName, "failed [" + curphase + "]")
            end = time.time()
            resultdict[curphase] = ["failed", end - start]
            return logAndReturn(
                commandslog,
                timelog,
                resultdict,
                1,
                uploadtestsList,
                cleanup,
                testName,
                uploadDict,
                dateAndTime,
            )
        end = time.time()
        resultdict[curphase] = ["passed", end - start]
    else:
        iree_import_onnx = "iree-import-onnx"
        curphase = phases[1]
        logfilename = curphase + ".log"
        # If local iree build provided use that instead of using the one installed
        # in user's env
        if SHARED_IREE_BUILD:
            iree_python_path = f"PYTHONPATH={SHARED_IREE_BUILD}/compiler/bindings/python"
            iree_import_onnx = f"{iree_python_path} python -m iree.compiler.tools.import_onnx"
        scriptcommand = (
            iree_import_onnx
            + " "
            + onnxfilename
            + " -o "
            + torchmlirOutputfilename
            + " 1> "
            + logfilename
            + " 2>&1"
        )
        start = time.time()
        if launchCommand(args, scriptcommand, commandslog):
            print("Test", testName, "failed [" + curphase + "]")
            end = time.time()
            resultdict[curphase] = ["failed", end - start]
            return logAndReturn(
                commandslog,
                timelog,
                resultdict,
                1,
                uploadtestsList,
                cleanup,
                testName,
                uploadDict,
                dateAndTime,
            )
        end = time.time()
        resultdict[curphase] = ["passed", end - start]

    return 0


def runTorchMLIRGeneration(
    testName,
    modelname,
    mode,
    args,
    phases,
    scriptcommand,
    commandslog,
    timelog,
    onnxfilename,
    torchmlirOutputfilename,
    resultdict,
    uploadtestsList,
    cleanup,
    uploadDict,
    dateAndTime,
):
    if args.verbose:
        print("Running torch MLIR generation for", testName)

    torch_mlir_pythonpath = ""
    if SHARED_TORCH_MLIR_BUILD:
        torch_mlir_pythonpath = f"PYTHONPATH={SHARED_TORCH_MLIR_BUILD}/tools/torch-mlir/python_packages/torch_mlir"
        scriptcommand = f"{torch_mlir_pythonpath} {scriptcommand}"

    # Phase = 0, Run the model.py first
    start = time.time()
    curphase = phases[0]
    if launchCommand(args, scriptcommand, commandslog):
        print("Test", testName, "failed [" + curphase + "]")
        end = time.time()
        resultdict[curphase] = ["failed", end - start]
        return logAndReturn(
            commandslog,
            timelog,
            resultdict,
            1,
            uploadtestsList,
            cleanup,
            testName,
            uploadDict,
            dateAndTime,
        )
    end = time.time()
    resultdict[curphase] = ["passed", end - start]
    if mode == "onnx" or mode == "ort":
        return runOnnxToTorchMLIRGeneration(
            testName,
            modelname,
            mode,
            args,
            phases,
            scriptcommand,
            commandslog,
            timelog,
            onnxfilename,
            torchmlirOutputfilename,
            resultdict,
            uploadtestsList,
            cleanup,
            uploadDict,
            dateAndTime,
            torch_mlir_pythonpath,
        )
    return 0


def runCodeGeneration(
    testName,
    args,
    phases,
    torchmlirOutputfilename,
    vmfbfilename,
    logfilename,
    commandslog,
    timelog,
    resultdict,
    uploadtestsList,
    cleanup,
    uploadDict,
    dateAndTime,
):
    if args.verbose:
        print("Running code generation for", testName)
    # Compile torch MLIR using IREE to binary to target backend
    curphase = phases[3]
    if (
        not os.path.exists(torchmlirOutputfilename)
        or not os.path.getsize(torchmlirOutputfilename) > 0
    ):
        print(f"The torch MLIR {torchmlirOutputfilename} does not exist or is empty.")
        print(f"Test {testName} failed [{curphase}]")
        return 1
    logfilename = curphase + ".log"
    commandname = ""
    if SHARED_IREE_BUILD:
        commandname = SHARED_IREE_BUILD + "/tools/"
    # Else pick from path
    commandname += (
        "iree-compile --iree-input-demote-i64-to-i32 --iree-hal-target-backends="
        + args.backend
        + " "
    )
    scriptcommand = (
        commandname
        + " "
        + torchmlirOutputfilename
        + " > "
        + vmfbfilename
        + " 2>"
        + logfilename
    )
    start = time.time()
    if launchCommand(args, scriptcommand, commandslog):
        print("Test", testName, "failed [" + curphase + "]")
        end = time.time()
        resultdict[curphase] = ["failed", end - start]
        return logAndReturn(
            commandslog,
            timelog,
            resultdict,
            1,
            uploadtestsList,
            cleanup,
            testName,
            uploadDict,
            dateAndTime,
        )
    end = time.time()
    resultdict[curphase] = ["passed", end - start]
    return 0


# ∣input−other∣ ≤ atol + rtol × ∣other∣
# Pytorch torch.testing defaults:
#    bf16: atol=1e-05, rtol=1.6e-02
#    fp16: atol=1e-05, rtol=1e-03
#    fp32: atol=1e-05, rtol=1.3e-06
def getTolerances(args, torchdtype):
    if args.tolerance:
        return tuple(args.tolerance)
    elif torchdtype == torch.bfloat16:
        return (1e-02, 1e-01)
    elif torchdtype == torch.float16:
        return (1e-04, 1e-03)
    return (1e-04, 1e-04)


def compareOutputs(args, goldoutput, infoutput, dtype):
    # if shapes do not match, we have a problem as comparison routines may crash
    # so gauard it
    inferencematched = False
    if infoutput.shape != goldoutput.shape:
        print(
            f"Shapes of two tensors do not match: gold: {goldoutput.shape} , inf: {infoutput.shape}"
        )
        inferencematched = False
    else:
        if args.zerotolerance:
            # If each element matches exactly only then torch.equal is true
            inferencematched = torch.equal(infoutput, goldoutput)
        else:
            atol, rtol = getTolerances(args, dtype)
            inferencematched = torch.allclose(infoutput, goldoutput, rtol, atol)
    return inferencematched


def runInference(
    testName,
    args,
    phases,
    vmfbfilename,
    modelinputptfilename,
    goldoutputptfilename,
    scriptcommand,
    commandslog,
    timelog,
    resultdict,
    uploadtestsList,
    cleanup,
    uploadDict,
    dateAndTime,
):
    if args.verbose:
        print("Running inference for", testName)
    curphase = phases[4]
    if not os.path.exists(vmfbfilename) or not os.path.getsize(vmfbfilename) > 0:
        print(
            "The compiled artefact",
            vmfbfilename,
            "does not exist or is empty. Make sure you have run previous phases.",
        )
        print("Test", testName, "failed [" + curphase + "]")
        return 1
    # read the gold output produced by model
    logfilename = curphase + ".log"
    getinfoutfilename = lambda i: "inference_output" + "." + str(i) + ".bin"
    # Each input or output loaded here is a python list of
    # torch tensor
    modelinputlist = loadTorchSave(modelinputptfilename)
    goldoutputlist = loadTorchSave(goldoutputptfilename)
    inputarg = ""
    if args.verbose:
        print(f"Loaded: {modelinputptfilename} and {goldoutputptfilename}")
        print(
            f"input list length: {len(modelinputlist)}, output list length: {len(goldoutputlist)}"
        )
    # If there is no input the do not pass --input
    if len(modelinputlist) > 0:

        for i, modelinput in enumerate(modelinputlist):
            if modelinput.numel() > 0:
                modelinputbinfilename = "inference_input." + str(i) + ".bin"
                if args.verbose:
                    print(f"Creating: {modelinputbinfilename}")
                writeInferenceInputBinFile(modelinput, modelinputbinfilename)
                inputshapestring = getShapeString(modelinput)
                inputarg += (
                    ' --input="'
                    + inputshapestring
                    + "=@"
                    + modelinputbinfilename
                    + '" '
                )
    if args.verbose:
        print(f"Created: inference_input.n.bin files")

    outputarg = ""
    commanddir = ""
    if SHARED_IREE_BUILD:
        commanddir = SHARED_IREE_BUILD + "/tools/"
    # else pick from path
    # expecting goldoutputlist to be a List[Tensors]
    # each tensor corresponding to a vmfb output
    for i in range(0, len(goldoutputlist)):
        infoutputfilename = getinfoutfilename(i)
        outputarg += " --output=@" + infoutputfilename + " "
    scriptcommand = (
        commanddir
        + "iree-run-module --module="
        + vmfbfilename
        + inputarg
        + outputarg
        + " > "
        + logfilename
        + " 2>&1"
    )

    start = time.time()

    if launchCommand(args, scriptcommand, commandslog):
        print("Test", testName, "failed [" + curphase + "]")
        end = time.time()
        resultdict[curphase] = ["failed", end - start]
        return logAndReturn(
            commandslog,
            timelog,
            resultdict,
            1,
            uploadtestsList,
            cleanup,
            testName,
            uploadDict,
            dateAndTime,
        )
    end = time.time()

    # Load the E2ESHARK_CHECK.pkl file saved by model run
    e2esharkDict = loadE2eSharkCheckDictionary()
    # get gold postprocessed output list
    goldpostoutputlist = e2esharkDict["postprocessed_output"]

    for i in range(0, len(goldoutputlist)):
        goldoutput = goldoutputlist[i]
        outputshape = goldoutput.size()

        torchdtype = goldoutput.dtype
        infoutputfilename = getinfoutfilename(i)
        if args.verbose:
            print(
                f"Out shape: {outputshape} Dtype: {torchdtype} Loading {infoutputfilename}"
            )
        infoutput = loadRawBinaryAsTorchSensor(
            infoutputfilename, outputshape, torchdtype
        )

        if args.verbose:
            inerencelog = open(logfilename, "a")
            torch.set_printoptions(profile="full")
            print(f"Gold reference[{i}]:\n{goldoutput}\n", file=inerencelog)
            print(f"Inference Output[{i}]:\n {infoutput}:\n", file=inerencelog)

        goldoutput_flat = goldoutput.flatten()
        infoutput_flat = infoutput.flatten()

        inferencematched = compareOutputs(
            args, goldoutput_flat, infoutput_flat, torchdtype
        )

        if not inferencematched or e2esharkDict.get("output_for_validation"):
            if i >= len(goldpostoutputlist):
                resultdict[curphase] = ["passed", end - start]
                return
            if args.postprocess and (e2esharkDict.get("postprocess")):
                functionPipeLine = e2esharkDict["postprocess"]
                goldpostoutput = goldpostoutputlist[i]
                infpostoutput = applyPostProcessPipeline(infoutput, functionPipeLine)
                # now compare the two
                if args.verbose:
                    print(f"gold post processed: {goldpostoutput}")
                    print(f"inference post processed: {infpostoutput}")
                torchdtype = infpostoutput.dtype
                goldoutput_flat = goldpostoutput.flatten()
                infoutput_flat = infpostoutput.flatten()
                inferencematched = compareOutputs(
                    args, goldoutput_flat, infoutput_flat, torchdtype
                )

        infoutput = infoutput_flat
        goldoutput = goldoutput_flat

        if not inferencematched:
            failedinflog = open("failedinference.log", "w")
            torch.set_printoptions(profile="full")
            print(f"Gold reference[output[{i}]]:\n{goldoutput}\n", file=failedinflog)
            print(f"Inference Output[output[{i}]]:\n{infoutput}:\n", file=failedinflog)
            atol, rtol = getTolerances(args, infoutput.dtype)
            diff = torch.abs(infoutput - goldoutput) <= (
                atol + rtol * torch.abs(goldoutput)
            )
            print(f"Element-wise difference[output[{i}]]:\n{diff}\n", file=failedinflog)
            percentdiff = torch.sum(diff).item() / diff.nelement() * 100
            print(
                f"Percentage element-wise match[{i}]:{percentdiff:.2f}%\n",
                file=failedinflog,
            )
            print("Test", testName, "failed [mismatch]")
            end = time.time()
            resultdict[curphase] = ["mismatch", end - start]
            return logAndReturn(
                commandslog,
                timelog,
                resultdict,
                1,
                uploadtestsList,
                cleanup,
                testName,
                uploadDict,
                dateAndTime,
            )

    resultdict[curphase] = ["passed", end - start]


def runTestUsingVAIML(args_tuple):
    (
        frameworkname,
        testName,
        args,
        toolsDirAbsPath,
        modelname,
        utilspy,
        testRunDir,
        testAbsPath,
        modelpy,
        runmodelpy,
        commandslog,
        timelog,
        resultdict,
        phases,
        vmfbfilename,
        modelinputptfilename,
        goldoutputptfilename,
    ) = args_tuple
    # TBD
    print("ERROR: Running test using VAI-ML is not implemented yet")
    sys.exit(1)
    return 0


def runTestUsingClassicalFlow(args_tuple):
    (
        frameworkname,
        testName,
        args,
        toolsDirAbsPath,
        modelname,
        utilspy,
        testRunDir,
        testAbsPath,
        modelpy,
        runmodelpy,
        commandslog,
        timelog,
        resultdict,
        phases,
        vmfbfilename,
        modelinputptfilename,
        goldoutputptfilename,
        uploadtestsList,
        uploadDict,
        dateAndTime,
    ) = args_tuple
    stubrunmodelpy = toolsDirAbsPath + "/stubs/pytorchmodel.py"
    mode = args.mode
    testargs = ""
    torchmlirOutputfilename = ""
    linalgmlirfilename = ""
    onnxfilename = ""
    if args.verbose:
        print(f"Running classical flow for test {testName}")
    # create a symlink to the utils file inside the test dir
    if not os.path.exists(utilspy):
        print(f"ERROR: {utilspy} file missing")
        sys.exit()
    symUtilspy = os.path.join(testRunDir, "commonutils.py")
    if not os.path.exists(symUtilspy):
        os.symlink(utilspy, symUtilspy)

    if args.torchtolinalg:
        torchmlirsuffix = ".linalg.mlir"
    else:
        torchmlirsuffix = ".torch.mlir"

    # If turbine, uses turbine's aot export into mlir module, but rest of flow is same
    if frameworkname == "pytorch":
        if mode == "turbine":
            stubrunmodelpy = toolsDirAbsPath + "/stubs/turbinemodel.py"
            torchmlirOutputfilename = (
                modelname + "." + args.todtype + ".pytorch" + torchmlirsuffix
            )
        else:
            stubrunmodelpy = toolsDirAbsPath + "/stubs/pytorchmodel.py"
            onnxfilename = modelname + "." + args.todtype + ".onnx"
            torchmlirOutputfilename = (
                modelname + "." + args.todtype + ".pytorch" + torchmlirsuffix
            )
            testargs += " --torchmlirimport " + args.torchmlirimport
    elif frameworkname == "onnx":
        # For onnx, dierct and onnx means same as direct generates/has onnx itself
        if mode == "direct" or mode == "turbine":
            mode = "onnx"
        stubrunmodelpy = toolsDirAbsPath + "/stubs/onnxmodel.py"
        torchmlirOutputfilename = (
            modelname + "." + args.todtype + ".onnx" + torchmlirsuffix
        )
        onnxfilename = "model.onnx"
        if args.run_as_static:
            testargs += " --run_as_static "
        if args.verbose:
            testargs += " --verbose "
        if getTestKind(testName) == "models":
            onnxfilename = testAbsPath + "/model.onnx"
            # Create soft link to the model.onnx
            unzipONNXFile(testName, testAbsPath, "model.onnx")
            if not os.path.exists("model.onnx"):
                os.symlink(onnxfilename, "model.onnx")
    else:
        print("Framework ", frameworkname, " not supported")
        return 1

    if mode == "turbine":
        testargs += " --todtype " + args.todtype + " --outfileprefix " + modelname
    else:
        testargs += (
            " --todtype "
            + args.todtype
            + " --mode "
            + mode
            + " --outfileprefix "
            + modelname
        )

    concatenateFiles(modelpy, stubrunmodelpy, runmodelpy)
    curphase = phases[0]
    logfilename = curphase + ".log"
    scriptcommand = (
        "python " + runmodelpy + " " + testargs + " 1> " + logfilename + " 2>&1"
    )

    if args.verbose:
        print(f"Running classical flow model-run for test {testName}")
    # phases 0 to 2
    if args.runfrom == "model-run":
        if runTorchMLIRGeneration(
            testName,
            modelname,
            mode,
            args,
            phases,
            scriptcommand,
            commandslog,
            timelog,
            onnxfilename,
            torchmlirOutputfilename,
            resultdict,
            uploadtestsList,
            args.cleanup,
            uploadDict,
            dateAndTime,
        ):
            return 1

    if args.runupto == "torch-mlir":
        print("Test", testName, "passed")
        return logAndReturn(
            commandslog,
            timelog,
            resultdict,
            0,
            uploadtestsList,
            args.cleanup,
            testName,
            uploadDict,
            dateAndTime,
        )

    if args.runfrom == "model-run" or args.runfrom == "torch-mlir":
        if runCodeGeneration(
            testName,
            args,
            phases,
            torchmlirOutputfilename,
            vmfbfilename,
            logfilename,
            commandslog,
            timelog,
            resultdict,
            uploadtestsList,
            args.cleanup,
            uploadDict,
            dateAndTime,
        ):
            return 1
    if args.runupto == "iree-compile":
        print("Test", testName, "passed")
        return logAndReturn(
            commandslog,
            timelog,
            resultdict,
            0,
            uploadtestsList,
            args.cleanup,
            testName,
            uploadDict,
            dateAndTime,
        )

    # run inference now
    if (
        args.runfrom == "model-run"
        or args.runfrom == "torch-mlir"
        or args.runfrom == "iree-compile"
    ):
        if runInference(
            testName,
            args,
            phases,
            vmfbfilename,
            modelinputptfilename,
            goldoutputptfilename,
            scriptcommand,
            commandslog,
            timelog,
            resultdict,
            uploadtestsList,
            args.cleanup,
            uploadDict,
            dateAndTime,
        ):
            return 1

    print("Test", testName, "passed")
    return logAndReturn(
        commandslog,
        timelog,
        resultdict,
        0,
        uploadtestsList,
        args.cleanup,
        testName,
        uploadDict,
        dateAndTime,
    )
    return 0


def runTest(aTuple):
    curdir = os.getcwd()
    # Do not construct absolute path here as this will run
    # in a new process and cur dir may change over time giving
    # unpredicatble results
    (frameworkname, testName, args, script_dir, run_dir, uploadDict, dateAndTime) = (
        aTuple
    )
    testRunDir = run_dir + "/" + testName
    modelname = os.path.basename(testName)
    modelinputptfilename = (
        testRunDir + "/" + modelname + "." + args.todtype + ".input.pt"
    )
    goldoutputptfilename = (
        testRunDir + "/" + modelname + "." + args.todtype + ".goldoutput.pt"
    )
    phases = ["model-run", "onnx-import", "torch-mlir", "iree-compile", "inference"]
    resultdict = {}
    for phase in phases:
        # Put status and time taken for each phase
        resultdict[phase] = ["notrun", 0.0]

    testAbsPath = script_dir + "/" + testName

    toolsDirAbsPath = script_dir + "/tools"

    utilspy = toolsDirAbsPath + "/stubs/commonutils.py"
    modelpy = testAbsPath + "/model.py"
    # This is the generated runmodel.py which will be run
    runmodelpy = "runmodel.py"
    # For args.framework == onnx, onnx is starting point, so mode is
    # forced to onnx if it is direct

    if args.verbose:
        print("Running:", testName, "[ Proc:", os.getpid(), "]")
    if changeToTestDir(testRunDir):
        return 1

    # set up upload utilities
    uploadtestsList = []
    if args.uploadtestsfile:
        uploadtestsfile = os.path.expanduser(args.uploadtestsfile)
        uploadtestsfile = os.path.abspath(uploadtestsfile)
        uploadtestsList = getTestsListFromFile(uploadtestsfile)

    # Open files to log commands run and time taken
    commandslog = open("commands.log", "w")
    timelog = open("time.pkl", "wb")
    vmfbfilename = modelname + "." + args.todtype + ".vmfb"
    retStatus = 0
    args_tuple = (
        frameworkname,
        testName,
        args,
        toolsDirAbsPath,
        modelname,
        utilspy,
        testRunDir,
        testAbsPath,
        modelpy,
        runmodelpy,
        commandslog,
        timelog,
        resultdict,
        phases,
        vmfbfilename,
        modelinputptfilename,
        goldoutputptfilename,
        uploadtestsList,
        uploadDict,
        dateAndTime,
    )
    if args.mode == "vaiml":
        runTestUsingVAIML(args_tuple)
    else:
        retStatus = runTestUsingClassicalFlow(args_tuple)

    os.chdir(curdir)
    if retStatus:
        return 1

    return 0


def initializer(tm_path, iree_path):
    global SHARED_TORCH_MLIR_BUILD, SHARED_IREE_BUILD
    SHARED_TORCH_MLIR_BUILD = tm_path
    SHARED_IREE_BUILD = iree_path


def runFrameworkTests(
    frameworkname, testsList, args, script_dir, run_dir, TORCH_MLIR_BUILD, IREE_BUILD
):
    # print(f"In runFrameworkTests - torch mlir build - {TORCH_MLIR_BUILD}")
    if len(testsList) == 0:
        return
    poolSize = args.jobs
    print(
        f"Framework:{frameworkname} mode={args.mode} backend={args.backend} runfrom={args.runfrom} runupto={args.runupto}"
    )
    print("Test list:", testsList)
    uniqueTestList = []
    [uniqueTestList.append(test) for test in testsList if test not in uniqueTestList]
    if not uniqueTestList:
        return
    if args.ci:
        if "pytorch/models/vicuna-13b-v1.3" in uniqueTestList:
            uniqueTestList.remove("pytorch/models/vicuna-13b-v1.3")
    uploadDict = Manager().dict({})
    dateAndTime = str(datetime.datetime.now(datetime.timezone.utc))
    tupleOfListArg = []
    # Create list of tuple(test, arg, run_dir) to allow launching tests in parallel
    [
        tupleOfListArg.append(
            (frameworkname, test, args, script_dir, run_dir, uploadDict, dateAndTime)
        )
        for test in uniqueTestList
    ]
    if args.verbose:
        print("Following tests will be run:", uniqueTestList)

    if args.ci:
        for i in range(0, len(tupleOfListArg)):
            initializer(TORCH_MLIR_BUILD, IREE_BUILD)
            runTest(tupleOfListArg[i])
    else:
        with Pool(poolSize, initializer, (TORCH_MLIR_BUILD, IREE_BUILD)) as p:
            result = p.map_async(runTest, tupleOfListArg)
            result.wait()
            if args.verbose:
                print("All tasks submitted to process pool completed")

    with open("upload_urls.json", "w") as convert_file:
        # convert_file.write(json.dumps(uploadDict._getvalue()))
        convert_file.write(
            simplejson.dumps(
                simplejson.loads(json.dumps(uploadDict._getvalue())),
                indent=4,
                sort_keys=True,
            )
        )


def getSummaryRows(listofstatusrows, listoftimerows, tableheader):
    summaryrows = []
    summarycountrow = [0] * len(tableheader)
    for row in listofstatusrows:
        summarycountrow[0] += 1
        for i in range(1, len(row)):
            if row[i] == "passed":
                summarycountrow[i] += 1
    summarycountrow = ["total-count"] + summarycountrow
    summaryrows += [summarycountrow]
    timevaluerows = [[float(str) for str in row[1:]] for row in listoftimerows]
    # Add average time
    times = [statistics.mean(tuple) for tuple in zip(*timevaluerows)]
    avgtimerows = ["average-time"] + [sum(times)] + [f"{i:.{3}f}" for i in times]
    summaryrows += [avgtimerows]
    times = [statistics.median(tuple) for tuple in zip(*timevaluerows)]
    # Add median time
    medtimerows = ["median-time"] + [sum(times)] + [f"{i:.{3}f}" for i in times]
    summaryrows += [medtimerows]
    return summaryrows


def generateReport(run_dir, testsList, args):
    reportdict = {}
    tableheader = []
    listoftimerows = []
    listofstatusrows = []
    passlist = []
    faillist = []
    for test in testsList:
        timelog = run_dir + "/" + test + "/" + "time.pkl"
        if os.path.exists(timelog) and os.path.getsize(timelog) > 0:
            with open(timelog, "rb") as logf:
                testdict = pickle.load(logf)
            reportdict[test] = testdict

    # Now have the dectionary of dictionary of log
    for test, testdict in reportdict.items():
        statustablerow = [test]
        timetablerow = [test]
        # First time build header
        if len(tableheader) == 0:
            tableheader += ["tests"]
            for k, v in testdict.items():
                tableheader += [k]

        # Now build the rows
        for k, v in testdict.items():
            statustablerow += [v[0]]
            timetablerow += [f"{v[1]:.{3}f}"]

        testfailed = [str for str in ["failed", "mismatch"] if str in statustablerow]
        if testfailed:
            faillist += [test]
        else:
            passlist += [test]
        listofstatusrows += [statustablerow]
        listoftimerows += [timetablerow]

    # Now add header and value rows and tabulate
    statustablerows = [tableheader] + listofstatusrows
    timetablerows = [tableheader] + listoftimerows

    # Build summary
    summaryrows = getSummaryRows(listofstatusrows, listoftimerows, tableheader)
    summarytableheader = ["items"] + tableheader
    summarytabelerows = [summarytableheader] + summaryrows
    statustable = tabulate.tabulate(
        statustablerows, headers="firstrow", tablefmt=args.reportformat
    )
    timetable = tabulate.tabulate(
        timetablerows, headers="firstrow", tablefmt=args.reportformat
    )
    summarytable = tabulate.tabulate(
        summarytabelerows, headers="firstrow", tablefmt=args.reportformat
    )
    suffix = "txt"
    if args.reportformat == "html":
        suffix = "html"
    elif args.reportformat == "pipe" or args.reportformat == "github":
        suffix = "md"

    # Now write out report files
    timetablefile = run_dir + "/timereport." + suffix
    timetablepkl = run_dir + "/timereport.pkl"
    statustablefile = run_dir + "/statusreport." + suffix
    statustablepkl = run_dir + "/statusreport.pkl"
    summarytablefile = run_dir + "/summaryreport." + suffix
    summarytablepkl = run_dir + "/summaryreport.pkl"
    passlistfile = run_dir + "/passed.txt"
    faillistfile = run_dir + "/failed.txt"
    runname = os.path.basename(run_dir)
    with open(statustablefile, "w") as statusf:
        print(
            f"Status report for run: {runname} using mode:{args.mode} todtype:{args.todtype} backend:{args.backend}\n",
            file=statusf,
        )
        print(statustable, file=statusf)
    with open(statustablepkl, "wb") as f:
        pickle.dump(statustablerows, f)
    print(f"Generated status report {statustablefile}")

    with open(timetablefile, "w") as timef:
        print(
            f"Time (in seconds) report for run: {runname} using mode:{args.mode} todtype:{args.todtype} backend:{args.backend}\n",
            file=timef,
        )
        print(timetable, file=timef)
    with open(timetablepkl, "wb") as f:
        pickle.dump(timetablerows, f)
    print(f"Generated time report {timetablefile}")

    with open(summarytablefile, "w") as summaryf:
        print(
            f"Summary (time in seconds) for run: {runname} using mode:{args.mode} todtype:{args.todtype} backend:{args.backend}\n",
            file=summaryf,
        )
        print(summarytable, file=summaryf)
    with open(summarytablepkl, "wb") as f:
        pickle.dump(summarytabelerows, f)
    print(f"Generated summary report {summarytablefile}")

    with open(passlistfile, "w") as f:
        for items in passlist:
            print(items, file=f)
    with open(faillistfile, "w") as f:
        for items in faillist:
            print(items, file=f)


def checkBuild(run_dir, args):
    IREE_BUILD = ""
    TORCH_MLIR_BUILD = ""
    if args.torchmlirbuild:
        TORCH_MLIR_BUILD = args.torchmlirbuild
        TORCH_MLIR_BUILD = os.path.expanduser(TORCH_MLIR_BUILD)
        TORCH_MLIR_BUILD = os.path.abspath(TORCH_MLIR_BUILD)
        if not os.path.exists(TORCH_MLIR_BUILD):
            print(
                "ERROR: Torch MLIR build directory",
                TORCH_MLIR_BUILD,
                "does not exist.",
            )
            sys.exit(1)

    if args.ireebuild:
        IREE_BUILD = args.ireebuild
        IREE_BUILD = os.path.expanduser(IREE_BUILD)
        IREE_BUILD = os.path.abspath(IREE_BUILD)
        if not os.path.exists(IREE_BUILD):
            print("ERROR: IREE build directory", IREE_BUILD, "does not exist.")
            sys.exit(1)
    if not os.path.exists(run_dir):
        try:
            os.mkdir(run_dir)
        except OSError as errormsg:
            print(
                "ERROR: Could not make run directory",
                run_dir,
                " Error message: ",
                errormsg,
            )
            sys.exit(1)
    return (TORCH_MLIR_BUILD, IREE_BUILD)


def main():
    global TORCH_MLIR_BUILD, IREE_BUILD
    msg = "The run.py script to run e2e shark tests"
    parser = argparse.ArgumentParser(prog="run.py", description=msg, epilog="")
    parser.add_argument(
        "-b",
        "--backend",
        choices=["llvm-cpu", "amd-aie", "rocm"],
        default="llvm-cpu",
        help="Target backend i.e. hardware to run on",
    )
    parser.add_argument(
        "-d",
        "--todtype",
        choices=["default", "fp32", "fp16", "bf16"],
        default="default",
        help="If not default, casts model and input to given data type if framework supports model.to(dtype) and tensor.to(dtype)",
    )
    parser.add_argument(
        "-f",
        "--frameworks",
        nargs="*",
        choices=["pytorch", "onnx", "tensorflow"],
        default=["pytorch"],
        help="Run tests for given framework(s)",
    )
    parser.add_argument(
        "-g",
        "--groups",
        nargs="*",
        choices=["operators", "combinations", "models"],
        default=["operators", "combinations"],
        help="Run given group of tests",
    )
    parser.add_argument(
        "-i",
        "--ireebuild",
        help="Path to the IREE build directory",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel processes to use per machine for running tests",
    )
    parser.add_argument(
        "-c",
        "--torchmlirbuild",
        help="Path to the torch-mlir build directory",
    )
    parser.add_argument(
        "--torchtolinalg",
        action="store_true",
        default=False,
        help="Have torch-mlir-opt to produce linalg instead of torch mlir and pass that to iree-compile",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["direct", "turbine", "onnx", "ort", "vaiml"],
        default="onnx",
        help="direct=Fx/TS->torch-mlir, turbine=aot-export->torch-mlir, onnx=exportonnx-to-torch-mlir, ort=exportonnx-to-ortep",
    )
    parser.add_argument(
        "--norun",
        action="store_true",
        default=False,
        help="Skip running of tests. Useful for generating test summary after the run",
    )
    parser.add_argument(
        "-p",
        "--postprocess",
        action="store_true",
        default=False,
        help="Compare post processed outputs if test has postprocessing installed for determining successful run",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate test report summary",
    )
    parser.add_argument(
        "--reportformat",
        choices=["pipe", "github", "html", "csv"],
        default="pipe",
        help="Format of the test report summary file. It takes subset of tablefmt value of python tabulate",
    )
    parser.add_argument(
        "--runfrom",
        choices=["model-run", "torch-mlir", "iree-compile"],
        default="model-run",
        help="Start from model-run, or torch MLIR, or IREE compiled artefact",
    )
    parser.add_argument(
        "--runupto",
        choices=["torch-mlir", "iree-compile", "inference"],
        default="inference",
        help="Run upto torch MLIR generation, IREE compilation, or inference",
    )
    parser.add_argument(
        "-r",
        "--rundirectory",
        default="test-run",
        help="The test run directory",
    )
    parser.add_argument(
        "-s",
        "--skiptestsfile",
        help="A file with lists of tests that should be skipped",
    )
    parser.add_argument(
        "--uploadtestsfile",
        help="A file with lists of tests that should be uploaded",
    )
    parser.add_argument(
        "-t",
        "--tests",
        nargs="*",
        help="Run given specific test(s) only",
    )
    parser.add_argument(
        "--testsfile",
        help="A file with lists of tests (starting with framework name) to run",
    )
    parser.add_argument(
        "--tolerance",
        help="Set abolsulte (atol) and relative (rtol) tolerances for comparing floating point numbers. Example: --tolerance 1e-03 1-04",
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "--torchmlirimport",
        choices=["compile", "fximport"],
        default="fximport",
        help="Use torch_mlir.torchscript.compile, or Fx importer",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print aditional messsages to show progress",
    )
    parser.add_argument(
        "-z",
        "--zerotolerance",
        action="store_true",
        default=False,
        help="Do not allow any tolerance in comparing results",
    )
    parser.add_argument(
        "--cachedir",
        help="Please select a dir with large free space to cache all torch, hf, turbine_tank model data",
        required=True,
    )
    parser.add_argument(
        "--cleanup",
        help="Space efficient testing (removing the large mlir, vmfb files during the model runs)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--run_as_static",
        action="store_true",
        default=False,
        help="makes the dim_params for model.onnx static with param/value dict given in model.py",
    )
    parser.add_argument(
        "--ci",
        help="Adjusted behavior, so that CI works and artifacts are in right place",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    cache_dir = args.cachedir


    cache_dir = os.path.expanduser(cache_dir)
    cache_dir = os.path.abspath(cache_dir)

    if not os.path.exists(cache_dir):
        print(f"ERROR: The Cache directory {cache_dir} does not exist.")
        sys.exit(1)
    # get the amount of GB available
    _, _, free = shutil.disk_usage(cache_dir)
    space_available = float(free) / pow(1024, 3)
    if space_available < 20:
        warnings.warn(
            "WARNING: Less than 20 GB of space available in selected cache directory. "
            + "Please choose directory with more space to avoid disk storage issues when running models."
        )

    os.environ["TORCH_HOME"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ["TURBINE_TANK_CACHE_DIR"] = cache_dir

    if args.skiptestsfile and args.testsfile:
        print(f"ERROR: Only one of --skiptestsfile or --testsfile can be used")
        sys.exit(1)

    # Perform check that only two values atol and rtol are provided for --tolerance
    if args.tolerance:
        tol = tuple(args.tolerance)
        if len(tol) != 2:
            print(
                f"ERROR: Incorrent number of arguments {len(tol)} for --tolerance {tol} provided. Provide absolute tolerance followed by relative tolerance floating point values with space in between. Example: --tolerance 1e-03 1e-04"
            )
            sys.exit(1)

    # Root dir where run.py is
    script_dir = os.path.dirname(os.path.realpath(__file__))
    run_dir = os.path.abspath(args.rundirectory)
    frameworks = args.frameworks
    TORCH_MLIR_BUILD, IREE_BUILD = checkBuild(run_dir, args)
    # assert

    print("Starting e2eshark tests. Using", args.jobs, "processes")
    print("Cache Directory: " + cache_dir)
    if args.tolerance:
        print(
            f"Tolerance for comparing floating point (atol, rtol) = {tuple(args.tolerance)}"
        )

    if args.verbose:
        print("Test run with arguments: ", vars(args))

    if TORCH_MLIR_BUILD:
        print("Torch MLIR build:", TORCH_MLIR_BUILD)
    else:
        print(
            "Note: No Torch MLIR build provided using --torchmlirbuild. iree-import-onnx will be used to convert onnx to torch onnx mlir"
        )

    if IREE_BUILD:
        print("IREE build:", IREE_BUILD)
    else:
        iree_in_path = shutil.which("iree-compile")
        if iree_in_path:
            print(
                f"IREE BUILD: IREE in PATH {os.path.dirname(iree_in_path)} will be used"
            )
        else:
            if args.runupto == "iree-compile" or args.runupto == "inference":
                print(
                    f"ERROR: Must have IREE in PATH or supply an IREE build using --ireebuild to runupto iree-compile or inference"
                )
                sys.exit(1)

    print("Test run directory:", run_dir)
    totalTestList = []
    skiptestslist = []
    # if args.tests used, that means run given specific tests, the --frameworks options will be
    # ignored in that case
    if args.skiptestsfile:
        skiptestslist = getTestsListFromFile(args.skiptestsfile)

    if args.tests or args.testsfile:
        testsList = []
        print("Since --tests or --testsfile was specified, --groups ignored")
        if args.tests:
            testsList = args.tests
            testsList = [item.strip().strip(os.sep) for item in testsList]
        if args.testsfile:
            testfile_path = os.path.expanduser(args.testsfile)
            testfile_path = os.path.abspath(testfile_path)
            testsList += getTestsListFromFile(testfile_path)

        # Strip leading/trailing slashes
        # Construct a dictionary of framework name and list of tests in them
        frameworktotests_dict = {"pytorch": [], "onnx": [], "tensorflow": []}
        for testName in testsList:
            if not os.path.exists(testName):
                print("ERROR: Test", testName, "does not exist")
                sys.exit(1)
            frameworkname = testName.split("/")[0]
            if frameworkname not in frameworktotests_dict:
                print(
                    "Test name must start with a valid framework name: pytorch, onnx, tensorflow. Invalid name:",
                    frameworkname,
                )
            frameworktotests_dict[frameworkname] += [testName]
        for framework in frameworktotests_dict:
            testsList = frameworktotests_dict[framework]
            testsList = [test for test in testsList if not test in skiptestslist]
            totalTestList += testsList
            if framework == "onnx":
                pre_test_onnx_models_azure_download(testsList, cache_dir, script_dir)
            if not args.norun:
                runFrameworkTests(
                    framework,
                    testsList,
                    args,
                    script_dir,
                    run_dir,
                    TORCH_MLIR_BUILD,
                    IREE_BUILD,
                )
    else:
        for framework in frameworks:
            testsList = getTestsList(framework, args.groups)
            testsList = [test for test in testsList if not test in skiptestslist]
            totalTestList += testsList
            if framework == "onnx":
                pre_test_onnx_models_azure_download(testsList, cache_dir, script_dir)
            if not args.norun:
                runFrameworkTests(
                    framework,
                    testsList,
                    args,
                    script_dir,
                    run_dir,
                    TORCH_MLIR_BUILD,
                    IREE_BUILD,
                )

    # report generation
    if args.report:
        generateReport(run_dir, totalTestList, args)

    if args.ci:
        today = datetime.date.today()
        path = script_dir + "/ci_reports"
        if not os.path.exists(path):
            os.mkdir(path)
        path += "/" + str(today)
        if not os.path.exists(path):
            os.mkdir(path)
        mode_path = path + f"/{args.mode}_reports"
        if not os.path.exists(mode_path):
            os.mkdir(mode_path)
        shutil.move(run_dir + "/statusreport.md", mode_path + "/statusreport.md")
        shutil.move(run_dir + "/summaryreport.md", mode_path + "/summaryreport.md")
        shutil.move(run_dir + "/timereport.md", mode_path + "/timereport.md")

    # When all processes are done, print
    print("\nCompleted run of e2e shark tests")
    if args.uploadtestsfile:
        print(
            "\nIf using the upload feature, you can find a map of the test name "
            + "to azure urls for your uploaded artifacts in upload_urls.json"
        )


if __name__ == "__main__":
    main()
