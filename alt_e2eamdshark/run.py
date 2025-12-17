# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import warnings
from pathlib import Path
import argparse
import re
from typing import List, Literal, Optional

# append alt_e2eamdshark dir to path to allow importing without explicit pythonpath management
TEST_DIR = str(Path(__file__).parent)
sys.path.append(TEST_DIR)

from e2e_testing.framework import *
from e2e_testing.logging_utils import log_exception, log_result, post_test_clean, log_error

# import frontend test configs:
from e2e_testing.test_configs.onnxconfig import (
    CLOnnxTestConfig,
    OnnxTestConfig,
    OnnxEpTestConfig,
    REDUCE_TO_LINALG_PIPELINE,
)

# import backends
from e2e_testing.backends import SimpleIREEBackend, OnnxrtIreeEpBackend, CLIREEBackend
from e2e_testing.storage import load_test_txt_file, load_json_dict
from utils.report import generate_report, save_dict

ALL_STAGES = [
    "setup",
    "import_model",
    "preprocessing",
    "compilation",
    "construct_inputs",
    "native_inference",
    "compiled_inference",
    "benchmark",
    "postprocessing",
]

DEFAULT_STAGES = [
    "setup",
    "import_model",
    "preprocessing",
    "compilation",
    "construct_inputs",
    "native_inference",
    "compiled_inference",
    "postprocessing",
]

def get_tests(groups: Literal["all", "combinations", "operators"], test_filter: Optional[str], testsfile: Optional[str]) -> List[str]:
    """imports tests based on groups and test_filter specification"""
    combinations = True if groups == "all" or groups == "combinations" else False
    models = True if groups == "all" or groups == "models" else False
    operators = True if groups == "all" or groups == "operators" else False

    from e2e_testing.registry import GLOBAL_TEST_LIST

    # importing the test generating files will register them to GLOBAL_TEST_LIST
    if combinations:
        import onnx_tests.combinations
    if models:
        import onnx_tests.models
    if operators:
        import onnx_tests.operators

    pre_test_list = GLOBAL_TEST_LIST

    if testsfile:
        test_names = load_test_txt_file(testsfile)
        pre_test_list = [t for t in GLOBAL_TEST_LIST if t.unique_name in test_names]

    if test_filter:
        test_list = [
            test for test in pre_test_list if re.match(test_filter, test.unique_name)
        ]
    else:
        test_list = pre_test_list

    test_list = sorted(test_list, key=lambda test : test.unique_name.lower())

    return test_list


def main(args):
    """Sets up config and test list based on CL args, then runs the tests"""

    # setup config
    if args.mode == "onnx-iree":
        pipeline = REDUCE_TO_LINALG_PIPELINE if args.torchtolinalg else []
        config = OnnxTestConfig(
            str(TEST_DIR), SimpleIREEBackend(device=args.device, hal_target_backend=args.backend, extra_args=args.iree_compile_args), pipeline
        )
    elif args.mode == "cl-onnx-iree":
        pipeline = REDUCE_TO_LINALG_PIPELINE if args.torchtolinalg else []
        config = CLOnnxTestConfig(
            str(TEST_DIR), CLIREEBackend(device=args.device, hal_target_backend=args.backend, target_chip=args.target_chip, extra_args=args.iree_compile_args), pipeline
        )
    elif args.mode == "ort-ep":
        # TODO: allow specifying provider explicitly from cl args.
        config = OnnxEpTestConfig(
            str(TEST_DIR), OnnxrtIreeEpBackend(device=args.device, hal_target_device=args.backend))
    else:
        raise NotImplementedError(f"unsupported mode: {args.mode}")

    # get test list
    test_list = get_tests(args.groups, args.test_filter, args.testsfile)
    test_list.sort()

    #setup test stages
    stages = ALL_STAGES if args.benchmark else DEFAULT_STAGES

    if args.stages:
        stages = args.stages
    if args.skip_stages:
        stages = [s for s in stages if s not in args.skip_stages]
    
    parent_log_dir = os.path.join(TEST_DIR, args.rundirectory)

    status_dict = run_tests(
        test_list,
        config,
        parent_log_dir,
        args.no_artifacts,
        args.verbose,
        stages,
        args.load_inputs,
        int(args.cleanup),
        args.get_metadata,
    )

    if args.report:
        generate_report(args, stages, status_dict)
        json_save_to = str(Path(args.report_file).parent.joinpath(Path(args.report_file).stem + ".json"))
        save_dict(status_dict, json_save_to)


def run_tests(
    test_list: List[Test], config: TestConfig, parent_log_dir: str, no_artifacts: bool, verbose: bool, stages: List[str], load_inputs: bool, cleanup: int, get_metadata=bool,
) -> Dict[str, Dict]:
    """runs tests in test_list based on config. Returns a dictionary containing the test statuses."""
    # TODO: multi-process
    # TODO: setup exception handling and better logging
    # TODO: log command-line reproducers for each step

    # set up a parent log directory to store results
    if not os.path.exists(parent_log_dir):
        os.makedirs(parent_log_dir)

    warnings.filterwarnings("ignore")

    if verbose:
        warnings.simplefilter("once", UserWarning)
        print(f"Stages to be run: {stages}")
        print(f'Test list: {[test.unique_name for test in test_list]}')

    status_dict = dict()

    for t in test_list:

        if verbose:
            print(f"running test {t.unique_name}...")

        # set log directory for the individual test
        log_dir = os.path.join(parent_log_dir, t.unique_name) + "/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # setup stage notifications
        ws = lambda curr_stage : " "*(max([len(s) for s in stages]) - len(curr_stage))
        notify_stage = lambda : print(f"\tRunning stage '{curr_stage}'..." + ws(curr_stage), end="\r")

        mean_time_ms = None
        try:
            # TODO: convert staging to an Enum and figure out how to specify staging from args
            # TODO: enable loading output/goldoutput bin files, vmfb, and mlir files if already present

            # set up test
            curr_stage = "setup"
            if curr_stage in stages:
                notify_stage()
                # build an instance of the test info class
                inst = t.model_constructor(t.unique_name, log_dir)
                options = inst.extra_options
                # this is highly onnx specific. 
                # TODO: Figure out how to factor this out of run.py
                if not os.path.exists(inst.model):
                    inst.construct_model()
                if get_metadata:
                    metadata = inst.get_metadata()
                    metadata_file = Path(log_dir) / "metadata.json"
                    save_dict(metadata, metadata_file)
            
            artifact_save_to = None if no_artifacts else log_dir
            # generate mlir from the instance using the config
            curr_stage = "import_model"
            if curr_stage in stages:
                notify_stage()
                model_artifact, func_name = config.import_model(
                    inst, save_to=artifact_save_to, extra_options=options.import_model_options
                )

            # apply config-specific preprocessing to the ModelArtifact
            curr_stage = "preprocessing"
            if curr_stage in stages:
                notify_stage()
                model_artifact = config.preprocess_model(
                    model_artifact, save_to=artifact_save_to
                )

            # compile mlir_module using config (calls backend compile)
            curr_stage = "compilation"
            if curr_stage in stages:
                notify_stage()
                compiled_artifact = config.compile(model_artifact, save_to=artifact_save_to, extra_options=options.compilation_options)

            # get inputs from inst
            curr_stage = "construct_inputs"
            if curr_stage in stages:
                notify_stage()
                if load_inputs:
                    inputs = inst.load_inputs(log_dir)
                else:
                    inputs = inst.construct_inputs()
                    inputs.save_to(log_dir)

            # run native inference
            curr_stage = "native_inference"
            if curr_stage in stages:
                notify_stage()
                golden_outputs_raw = inst.forward(inputs)
                # most reliable way to get output shapes for dynamic models
                # these shapes are needed to load the outputs from iree-run-module
                if isinstance(config, CLOnnxTestConfig):
                    config.tensor_info_dict[t.unique_name] = ([list(d.shape) for d in golden_outputs_raw.data], [d.dtype for d in golden_outputs_raw.to_torch().data])
                golden_outputs_raw.save_to(log_dir, base_stem="golden_output")

            # run inference with the compiled module
            curr_stage = "compiled_inference"
            if curr_stage in stages:
                notify_stage()
                outputs_raw = config.run(compiled_artifact, inputs, func_name=func_name, extra_options=options.compiled_inference_options)
                outputs_raw.save_to(log_dir, base_stem="output")

            # apply model-specific post-processing:
            curr_stage = "postprocessing"
            if curr_stage in stages:
                notify_stage()
                golden_outputs = inst.apply_postprocessing(golden_outputs_raw)
                outputs = inst.apply_postprocessing(outputs_raw)
                inst.save_processed_output(golden_outputs, log_dir, "golden_output")
                inst.save_processed_output(outputs, log_dir, "output")

        except Exception as e:
            status_dict[t.unique_name] = {"exit_status" : curr_stage, "time_ms" : mean_time_ms}
            log_error(status_dict, log_dir, curr_stage, t.unique_name)
            log_exception(e, log_dir, curr_stage, t.unique_name, verbose)
            post_test_clean(log_dir, cleanup, verbose)
            continue

        # benchmark inference time with compiled module
        curr_stage = "benchmark"
        if curr_stage in stages:
            notify_stage()
            # TODO: make repetitions configurable from a command line arg
            try:
                mean_time_ms = config.benchmark(compiled_artifact, inputs, repetitions=3, func_name=func_name, extra_options=options.compiled_inference_options)
            except Exception as e:
                # don't exit test because of a benchmarking failure
                mean_time_ms = "ERROR"
                log_exception(e, log_dir, curr_stage, t.unique_name, verbose)

        # store the results
        if "setup" and "native_inference" and "compiled_inference" in stages:
            try:
                result = TestResult(
                    name=t.unique_name,
                    input=inputs,
                    gold_output=golden_outputs,
                    output=outputs,
                )
                # log the results
                test_passed = log_result(result, log_dir, [1e-3, 1e-3])
                if test_passed:
                    status_dict[t.unique_name] = {"exit_status" : "PASS", "time_ms" : mean_time_ms}
                else:
                    status_dict[t.unique_name] = {"exit_status" : "Numerics", "time_ms" : mean_time_ms}
            except Exception as e:
                status_dict[inst.name] = {"exit_status" : "results-summary", "time_ms" : mean_time_ms}
                log_exception(e, log_dir, "results-summary", t.unique_name, verbose)
        
        if verbose:
            # "PASS" is only recorded if a results-summary is generated
            # if running a subset of ALL_STAGES, manually indicate "PASS".
            if t.unique_name not in status_dict.keys():
                status_dict[t.unique_name] = {"exit_status": "PASS", "time_ms" : mean_time_ms}
            if status_dict[t.unique_name]["exit_status"] == "PASS":
                print(f"\tPASSED" + " "*30)
            else:
                print(f"\tFAILED ({status_dict[t.unique_name]['exit_status']})" + " "*20)
        
        # test is complete, perform cleanup
        post_test_clean(log_dir,  cleanup, verbose)

    num_passes = [v["exit_status"] for v in status_dict.values()].count("PASS")
    print("\nTest Summary:")
    print(f"\tPASSES: {num_passes}\n\tTOTAL: {len(test_list)}")
    print(f"results stored in {parent_log_dir}")
    status_dict = dict(sorted(status_dict.items(), key=lambda item : item[0].lower()))
    return status_dict

def _get_argparse():
    msg = "The run.py script to run e2e amdshark tests"
    parser = argparse.ArgumentParser(prog="run.py", description=msg, epilog="")

    # test config related arguments:
    parser.add_argument(
        "-d",
        "--device",
        default="local-task",
        help="specifies the device for runtime config. E.g. local-task, local-sync, vulkan, hip, cuda",
    )
    parser.add_argument(
        "-b",
        "--backend",
        choices=["llvm-cpu", "amd-aie", "rocm", "hip", "cuda", "vmvx", "metal-spirv", "vulkan-spirv"],
        default="llvm-cpu",
        help="specifies the iree-hal-target-device / iree-hal-target-backends for compile phase",
    )
    parser.add_argument(
        "-c",
        "--target-chip",
        default="gfx942",
        help="specifies the target chip (gfx942 for hip)",
    )
    parser.add_argument(
        "-a",
        "--iree-compile-args",
        nargs="*",
        default = None,
        help="Manually specify a space-seperated list of extra args for iree-compile. Do not put `--` before the args.",
    )
    # parser.add_argument(
    #     "-f",
    #     "--framework",
    #     choices=["pytorch", "onnx", "tensorflow"],
    #     default="onnx",
    #     help="Run tests for given framework(s)",
    # )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["onnx-iree", "cl-onnx-iree", "ort-ep"],
        default="cl-onnx-iree",
        help="(cl-)onnx-iree : onnx->torch-mlir->IREE, "
             "cl-onnx-iree runs this through command line tools "
             "ort : onnx->run with custom ORT EP inference session",
    )
    parser.add_argument(
        "--torchtolinalg",
        action="store_true",
        default=False,
        help="for mode = onnx-iree: Have torch-mlir-opt convert to linalg before passing to iree-compile",
    )
    # parser.add_argument(
    #     "--torchmlirimport",
    #     choices=["compile", "fximport"],
    #     default="fximport",
    #     help="Use torch_mlir.torchscript.compile, or Fx importer",
    # )

    # arguments for customizing test-stages:
    parser.add_argument(
        "--stages",
        nargs="*",
        choices=ALL_STAGES,
        help="Manually specify which test stages to run.",
    )
    parser.add_argument(
        "--skip-stages",
        nargs="*",
        choices=ALL_STAGES,
        help="Manually specify which test stages to skip.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Convenience flag for enabling benchmark stage.",
    )
    parser.add_argument(
        "--load-inputs",
        action="store_true",
        default=False,
        help="If true, will try to load inputs from bin files.",
    )

    # test-list filtering arguments:
    parser.add_argument(
        "-g",
        "--groups",
        choices=["operators", "combinations", "models"],
        default="all",
        help="Run given group of tests",
    )
    parser.add_argument(
        "-t",
        "--test-filter",
        help="Run given specific test(s) only",
    )
    parser.add_argument(
        "--testsfile",
        help="A file with lists of test names to run",
    )

    # test tolerance
    parser.add_argument(
        "--tolerance",
        help="Set abolsulte (atol) and relative (rtol) tolerances for comparing floating point numbers. Example: --tolerance 1e-03 1-04",
        nargs="+",
        type=float,
    )

    # logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print aditional messsages to show progress",
    )
    parser.add_argument(
        "-r",
        "--rundirectory",
        default="test-run",
        help="The test run directory",
    )
    parser.add_argument(
        "--no-artifacts",
        help="If enabled, this flag will prevent saving mlir and vmfb files",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--cleanup",
        help="Specify cleanup level: 0 (nothing), 1 (size >= 500M), 2 (.mlir and .vmfb), 3 (all but .log), 4 (everything)",
        choices=['0','1','2','3','4'],
        default='0',
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate test report summary",
    )
    parser.add_argument(
        "--report-file",
        default="report.md",
        help="output filename for the report summary.",
    )
    parser.add_argument(
        "--get-metadata",
        action="store_true",
        default=False,
        help="save some model metadata to log_dir/metadata.json"
    )
    # parser.add_argument(
    #     "-d",
    #     "--todtype",
    #     choices=["default", "fp32", "fp16", "bf16"],
    #     default="default",
    #     help="If not default, casts model and input to given data type if framework supports model.to(dtype) and tensor.to(dtype)",
    # )
    # parser.add_argument(
    #     "-j",
    #     "--jobs",
    #     type=int,
    #     default=4,
    #     help="Number of parallel processes to use per machine for running tests",
    # )
    # parser.add_argument(
    #     "--norun",
    #     action="store_true",
    #     default=False,
    #     help="Skip running of tests. Useful for generating test summary after the run",
    # )
    # parser.add_argument(
    #     "--uploadtestsfile",
    #     help="A file with lists of tests that should be uploaded",
    # )
    return parser


if __name__ == "__main__":
    parser = _get_argparse()
    main(parser.parse_args())
