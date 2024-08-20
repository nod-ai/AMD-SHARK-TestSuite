# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import argparse

def _get_argparse():
    msg = "A script for setting up a .env file."
    parser = argparse.ArgumentParser(prog="write_env.py", description=msg, epilog="")

    parser.add_argument(
        "-i",
        "--iree-build",
        help="specify path to iree-build",
    )
    parser.add_argument(
        "-t",
        "--torch-mlir-build",
        help="specify path to torch-mlir/build",
    )
    parser.add_argument(
        "-c",
        "--cache",
        help="specify path to cache directory for downloading large models (e.g., '/home/username/.cache')",
    )
    parser.add_argument(
        "-a",
        "--azure-private-connection",
        help="specify azure-private-connection string for onnxprivatestorage",
    )
    return parser

def test_path(path: Path):
    if not path.exists():
        raise OSError(f'path: {path.absolute()} could not be resolved')

def main(args):
    s = ""
    pypaths = []

    if args.iree_build:
        iree_build_dir = Path(args.iree_build).resolve()
        test_path(iree_build_dir)
        compiler_bindings = iree_build_dir.joinpath("compiler/bindings/python")
        runtime_bindings = iree_build_dir.joinpath("runtime/bindings/python")
        test_path(compiler_bindings)
        test_path(runtime_bindings)
        pypaths.append(str(compiler_bindings))
        pypaths.append(str(runtime_bindings))

    if args.torch_mlir_build: 
        torch_mlir_build_dir = Path(args.torch_mlir_build).resolve()
        test_path(torch_mlir_build_dir)
        torch_mlir_bindings = torch_mlir_build_dir.joinpath("tools/torch-mlir/python_packages/torch_mlir/")
        test_path(torch_mlir_bindings)
        pypaths.append(str(torch_mlir_bindings))

    if args.cache:
        cache_dir = Path(args.cache).resolve()
        test_path(cache_dir)
        s += f"CACHE_DIR='{cache_dir}'\n"

    if args.azure_private_connection: 
        s += f'AZ_PRIVATE_CONNECTION="{args.azure_private_connection}"\n'

    if len(pypaths) > 0:
        pypathstr = ":".join(pypaths)
        s += f'PYTHONPATH="{pypathstr}"\n'

    if len(s) > 0:
        with open(".env", "w") as file:
            file.write(s)

    print("Check .env and run this script to export the variables to your environment (linux):")
    print("export $(cat .env | xargs)")
    
if __name__ == "__main__":
    parser = _get_argparse()
    main(parser.parse_args())