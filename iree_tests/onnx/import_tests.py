# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import onnx
from pathlib import Path
from onnx import numpy_helper
import shutil
import subprocess
import numpy as np
import sys

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent.parent

# The ONNX repo under third_party has test suite sources and generated files.
ONNX_REPO_ROOT = REPO_ROOT / "third_party/onnx"
ONNX_REPO_GENERATED_TESTS_ROOT = ONNX_REPO_ROOT / "onnx/backend/test/data"
NODE_TESTS_ROOT = ONNX_REPO_GENERATED_TESTS_ROOT / "node"

# Write imported files to our own 'generated' folder.
GENERATED_FILES_OUTPUT_ROOT = REPO_ROOT / "iree_tests/onnx/node/generated"

# Write lists of tests that passed/failed to import.
IMPORT_SUCCESSES_FILE = REPO_ROOT / "iree_tests/onnx/node/import_successes.txt"
IMPORT_FAILURES_FILE = REPO_ROOT / "iree_tests/onnx/node/import_failures.txt"


def find_onnx_tests(root_dir_path):
    test_dir_paths = [p for p in root_dir_path.iterdir() if p.is_dir()]
    print(f"Found {len(test_dir_paths)} tests")
    return sorted(test_dir_paths)


def convert_io_proto(proto_filename, type_proto):

    with open(proto_filename, "rb") as f:
        protobuf_content = f.read()
        if type_proto.HasField("tensor_type"):
            tensor = onnx.TensorProto()
            tensor.ParseFromString(protobuf_content)
            t = numpy_helper.to_array(tensor)
            return t
        else:
            print(f"Unsupported proto type: {type_proto}")
            return None


def import_onnx_files(test_dir_path, imported_dir_path):
    # This imports one 'test_[name]' subfolder from this:
    #
    #   test_[name]/
    #     model.onnx
    #     test_data_set_0/
    #       input_0.pb
    #       output_0.pb
    #
    # to this:
    #
    #   imported_dir_path/...
    #     test_[name]/
    #       model.mlir  (torch-mlir)
    #       input_0.npy
    #       output_0.npy
    #       test_data_flags.txt  (flagfile with --input=input_0.npy, --expected_output=)

    imported_dir_path.mkdir(parents=True, exist_ok=True)

    test_data_flagfile_path = imported_dir_path / "test_data_flags.txt"
    test_data_flagfile_lines = []

    # Import model.onnx to model.mlir.
    # TODO(scotttodd): copy the .onnx file into the generated folder? Useful for reproducing
    #                  could also add a symlink or other files with info
    #                  e.g. importer tool / version / flags used, reproducer command
    onnx_model_path = test_dir_path / "model.onnx"
    imported_model_path = imported_dir_path / "model.mlir"
    exec_args = [
        "iree-import-onnx",
        str(onnx_model_path),
        "-o",
        str(imported_model_path),
    ]
    ret = subprocess.run(exec_args, capture_output=True)
    if ret.returncode != 0:
        # print(
        #     f"  {imported_dir_path.name[5:]} import failed,\n    stdout: {ret.stdout},\n    stderr: {ret.stderr}",
        #     file=sys.stderr,
        # )
        print(f"  {imported_dir_path.name[5:]} import failed", file=sys.stderr)
        return False

    test_data_dirs = sorted(test_dir_path.glob("test_data_set*"))
    if len(test_data_dirs) != 1:
        print("WARNING: unhandled 'len(test_data_dirs) != 1'")
        return False

    # Convert input_*.pb and output_*.pb to .npy files.
    test_data_dir = test_data_dirs[0]
    test_inputs = list(test_data_dir.glob("input_*.pb"))
    test_outputs = list(test_data_dir.glob("output_*.pb"))
    model = onnx.load(onnx_model_path)
    for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        t = convert_io_proto(test_input, model.graph.input[i].type)
        if t is None:
            return False
        input_path = (imported_dir_path / test_input.stem).with_suffix(".npy")
        np.save(input_path, t)
        test_data_flagfile_lines.append(f"--input=@{input_path.name}\n")
    for i in range(len(test_outputs)):
        test_output = test_outputs[i]
        t = convert_io_proto(test_output, model.graph.output[i].type)
        if t is None:
            return False
        output_path = (imported_dir_path / test_output.stem).with_suffix(".npy")
        np.save(output_path, t)
        test_data_flagfile_lines.append(f"--expected_output=@{output_path.name}\n")

    with open(test_data_flagfile_path, "wt") as f:
        f.writelines(test_data_flagfile_lines)

    return True


if __name__ == "__main__":
    test_dir_paths = find_onnx_tests(NODE_TESTS_ROOT)

    # TODO(scotttodd): add flag to not clear output dir?
    print(f"Clearing old generated files from '{GENERATED_FILES_OUTPUT_ROOT}'")
    shutil.rmtree(GENERATED_FILES_OUTPUT_ROOT)
    GENERATED_FILES_OUTPUT_ROOT.mkdir(parents=True)

    print(f"Importing tests in '{NODE_TESTS_ROOT}'")

    print("******************************************************************")
    passed_imports = []
    failed_imports = []
    # TODO(scotttodd): parallelize this (or move into a test runner like pytest)
    for i in range(len(test_dir_paths)):
        test_dir_path = test_dir_paths[i]
        test_name = test_dir_path.name

        current_number = str(i).rjust(4, "0")
        progress_str = f"[{current_number}/{len(test_dir_paths)}]"
        print(f"{progress_str}: Importing {test_name}")

        imported_dir_path = Path(GENERATED_FILES_OUTPUT_ROOT) / test_name
        result = import_onnx_files(test_dir_path, imported_dir_path)
        if result:
            passed_imports.append(test_name)
        else:
            failed_imports.append(test_name)
            # Note: could comment this out to keep partially imported directories.
            shutil.rmtree(imported_dir_path)
    print("******************************************************************")

    passed_imports.sort()
    failed_imports.sort()

    with open(IMPORT_SUCCESSES_FILE, "wt") as f:
        f.write("\n".join(passed_imports))
    with open(IMPORT_FAILURES_FILE, "wt") as f:
        f.write("\n".join(failed_imports))

    print(f"Import pass count: {len(passed_imports)}")
    print(f"Import fail count: {len(failed_imports)}")
