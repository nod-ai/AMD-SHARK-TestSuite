# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path
import shutil
import subprocess

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent.parent.parent
E2ESHARK_DIR = REPO_ROOT / "e2eshark"
TEST_RUN_DIR = E2ESHARK_DIR / "test-run"


def human_readable_size(size, decimal_places=2):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def extract_parameters(
    input_mlir_path, output_mlir_path, real_weights_output_path, splats_output_path
):
    # TODO: Have Turbine itself output with parameters. The passes in `iree-compile`
    #   run as part of "global optimization" and muck with the program before/after
    #   export so handling it correctly in the frontend would be preferred.

    # TODO: Try with a larger minimum export size, e.g.
    #   --iree-opt-minimum-parameter-export-size=16777216
    #   That was crashing in `iree-compile` around `serializeResourceRawData()` though

    print("  Extracting parameters...")
    exec_args = [
        "iree-compile",
        str(input_mlir_path),
        f"--iree-opt-parameter-archive-export-file={str(real_weights_output_path)}",
        f"--iree-opt-splat-parameter-archive-export-file={str(splats_output_path)}",
        "--compile-to=global-optimization",
        "-o",
        str(output_mlir_path),
    ]
    subprocess.check_call(exec_args)

    # Log file sizes.
    input_size = human_readable_size(input_mlir_path.stat().st_size)
    params_size = human_readable_size(output_mlir_path.stat().st_size)
    real_weights_size = human_readable_size(real_weights_output_path.stat().st_size)
    splats_size = human_readable_size(splats_output_path.stat().st_size)
    print(
        f"    {input_mlir_path.name} ({input_size}) -> {output_mlir_path.name} ({params_size})"
    )
    print(f"    {real_weights_output_path.name} size: {real_weights_size}")
    print(f"    {splats_output_path.name}       size: {splats_size}")


def convert_to_mlirbc(input_mlir_path, output_mlirbc_path):
    print("  Converting to mlirbc...")
    exec_args = [
        "iree-ir-tool",
        "cp",
        "--emit-bytecode",
        str(input_mlir_path),
        "-o",
        str(output_mlirbc_path),
    ]
    subprocess.check_call(exec_args)

    # Log file sizes.
    input_size = human_readable_size(input_mlir_path.stat().st_size)
    output_size = human_readable_size(output_mlirbc_path.stat().st_size)
    print(
        f"    {input_mlir_path.name} ({input_size}) -> {output_mlirbc_path.name} ({output_size})"
    )


def export_pytorch_model(model_name):
    input_dir = TEST_RUN_DIR / "pytorch/models" / model_name
    input_file = input_dir / f"{model_name}.default.pytorch.torch.mlir"
    output_dir = THIS_DIR / model_name

    print(f"Converting '{model_name}'")

    params_mlir_path = input_dir / f"{model_name}.mlir"
    params_mlirbc_path = input_dir / f"{model_name}.mlirbc"
    real_weights_irpa_path = input_dir / "real_weights.irpa"
    splats_irpa_path = input_dir / "splats.irpa"

    extract_parameters(
        input_mlir_path=input_file,
        output_mlir_path=params_mlir_path,
        real_weights_output_path=real_weights_irpa_path,
        splats_output_path=splats_irpa_path,
    )
    convert_to_mlirbc(
        input_mlir_path=params_mlir_path, output_mlirbc_path=params_mlirbc_path
    )

    print("Copying to iree_tests test case")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(params_mlirbc_path, output_dir / params_mlirbc_path.name)
    shutil.copy(splats_irpa_path, output_dir / splats_irpa_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2ESHARK test case importer.")
    parser.add_argument(
        "--model",
        help="Model name to export, e.g. 'opt-125M'",
    )
    args = parser.parse_args()

    export_pytorch_model(args.model)
