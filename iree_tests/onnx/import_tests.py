# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import onnx
from multiprocessing import Pool
from pathlib import Path
from onnx import numpy_helper, version_converter
import shutil
import subprocess
import numpy as np
import sys
from import_tests_utils import get_shape_string, write_io_bin

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent.parent

# The ONNX repo under third_party has test suite sources and generated files.
ONNX_REPO_ROOT = REPO_ROOT / "third_party/onnx"
ONNX_REPO_GENERATED_TESTS_ROOT = ONNX_REPO_ROOT / "onnx/backend/test/data"
NODE_TESTS_ROOT = ONNX_REPO_GENERATED_TESTS_ROOT / "node"

# Convert test cases to at least this version using The ONNX Version Converter.
ONNX_CONVERTER_OUTPUT_MIN_VERSION = 17

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


def import_onnx_files_with_cleanup(test_dir_path):
    test_name = test_dir_path.name
    imported_dir_path = Path(GENERATED_FILES_OUTPUT_ROOT) / test_name
    result = import_onnx_files(test_dir_path, imported_dir_path)
    if not result:
        # Note: could comment this out to keep partially imported directories.
        shutil.rmtree(imported_dir_path)
    return (test_name, result)


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

    # Convert model.onnx up to ONNX_CONVERTER_OUTPUT_MIN_VERSION if needed.
    # TODO(scotttodd): stamp some info e.g. importer tool / version / flags used
    original_model_path = test_dir_path / "model.onnx"
    converted_model_path = imported_dir_path / "model.onnx"

    original_model = onnx.load_model(original_model_path)
    original_version = original_model.opset_import[0].version
    if original_version < ONNX_CONVERTER_OUTPUT_MIN_VERSION:
        try:
            converted_model = version_converter.convert_version(
                original_model, ONNX_CONVERTER_OUTPUT_MIN_VERSION
            )
            onnx.save(converted_model, converted_model_path)
        except:
            # Conversion failed. Do our best with the original file.
            # print(f"WARNING: ONNX conversion failed for {test_dir_path.name}")
            shutil.copy(original_model_path, converted_model_path)
    else:
        # No conversion needed.
        shutil.copy(original_model_path, converted_model_path)

    # Import converted model.onnx to model.mlir.
    imported_model_path = imported_dir_path / "model.mlir"
    exec_args = [
        "iree-import-onnx",
        str(converted_model_path),
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

    # Convert input_*.pb and output_*.pb -> .npy -> .bin (little endian)
    # systems with big endian byteordering may have problems with provided bins
    # Check using `sys.byteorder` and import tests locally if required.
    test_data_dir = test_data_dirs[0]
    test_inputs = sorted(list(test_data_dir.glob("input_*.pb")))
    test_outputs = sorted(list(test_data_dir.glob("output_*.pb")))
    model = onnx.load(converted_model_path)
    for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        t = convert_io_proto(test_input, model.graph.input[i].type)
        if t is None:
            return False
        input_path = (imported_dir_path / test_input.stem).with_suffix(".npy")
        np.save(input_path, t)  # Only for ref, actual comparison with .bin
        ss = get_shape_string(t)
        input_path_bin = (imported_dir_path / test_input.stem).with_suffix(".bin")
        write_io_bin(t, input_path_bin)
        test_data_flagfile_lines.append(
            f"--input={ss}=@{input_path_bin.name}\n"
        )
    for i in range(len(test_outputs)):
        test_output = test_outputs[i]
        t = convert_io_proto(test_output, model.graph.output[i].type)
        if t is None:
            return False
        output_path = (imported_dir_path / test_output.stem).with_suffix(".npy")
        np.save(output_path, t)  # Only for ref, actual comparison with .bin
        ss = get_shape_string(t)
        # required for signless output comparision
        if "xsi" in ss or "xui" in ss:
            ss = ss.replace("xsi", "xi")
            ss = ss.replace("xui", "xi")
        output_path_bin = (imported_dir_path / test_output.stem).with_suffix(".bin")
        write_io_bin(t, output_path_bin)
        test_data_flagfile_lines.append(
            f"--expected_output={ss}=@{output_path_bin.name}\n"
        )

    with open(test_data_flagfile_path, "wt") as f:
        f.writelines(test_data_flagfile_lines)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX test case importer.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        help="Number of parallel processes to use when importing test cases",
    )
    args = parser.parse_args()

    test_dir_paths = find_onnx_tests(NODE_TESTS_ROOT)

    # TODO(scotttodd): add flag to not clear output dir?
    print(f"Clearing old generated files from '{GENERATED_FILES_OUTPUT_ROOT}'")
    shutil.rmtree(GENERATED_FILES_OUTPUT_ROOT)
    GENERATED_FILES_OUTPUT_ROOT.mkdir(parents=True)

    print(f"Importing tests in '{NODE_TESTS_ROOT}'")
    print("******************************************************************")
    passed_imports = []
    failed_imports = []
    with Pool(args.jobs) as pool:
        results = pool.imap_unordered(
            import_onnx_files_with_cleanup, test_dir_paths
        )
        for result in results:
            if result[1]:
                passed_imports.append(result[0])
            else:
                failed_imports.append(result[0])
    print("******************************************************************")

    passed_imports.sort()
    failed_imports.sort()

    with open(IMPORT_SUCCESSES_FILE, "wt") as f:
        f.write("\n".join(passed_imports))
    with open(IMPORT_FAILURES_FILE, "wt") as f:
        f.write("\n".join(failed_imports))

    print(f"Import pass count: {len(passed_imports)}")
    print(f"Import fail count: {len(failed_imports)}")
