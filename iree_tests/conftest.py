# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from pathlib import Path
from typing import List
import pyjson5
import os
import pytest
import subprocess


@dataclass(frozen=True)
class IreeCompileAndRunTestSpec:
    """Specification for an IREE "compile and run" test."""

    # Directory where test input files are located.
    test_directory: Path

    # Name of input MLIR file in a format accepted by IREE (e.g. torch, tosa, or linalg dialect).
    # Including file suffix, e.g. 'model.mlir' or 'model.mlirbc'.
    input_mlir_name: str

    # Stem of input MLIR file, excluding file suffix, e.g. 'model'.
    input_mlir_stem: str

    # Name of flagfile in the same directory as the input MLIR, containing flags like:
    #   --input=@input_0.npy
    #   --expected_output=@output_0.npy
    data_flagfile_name: str

    # Name of the test configuration, e.g. "cpu".
    # This will be used in generated files and test case names.
    config_name: str

    # Flags to pass to `iree-compile`, e.g. ["--iree-hal-target-backends=llvm-cpu"].
    iree_compile_flags: List[str]

    # Flags to pass to `iree-run-module`, e.g. ["--device=local-task"].
    # These will be passed in addition to `--flagfile={data_flagfile_name}`.
    iree_run_module_flags: List[str]

    # True if compilation is expected to succeed. If false, the test will be marked XFAIL.
    expect_compile_success: bool

    # True if running is expected to succeed. If false, the test will be marked XFAIL.
    expect_run_success: bool

    # True to only compile the test and skip running.
    skip_run: bool


def pytest_collect_file(parent, file_path):
    if file_path.name.endswith(".mlir") or file_path.name.endswith(".mlirbc"):
        return MlirFile.from_parent(parent, path=file_path)


class MlirFile(pytest.File):
    """Collector for MLIR files accompanied by input/output."""

    def collect(self):
        # Expected directory structure:
        #   path/to/test_some_ml_operator/
        #     - *.mlir[bc]
        #     - test_data_flags.txt
        #   path/to/test_some_ml_model/
        #     ...

        test_directory = self.path.parent
        test_name = test_directory.name

        # Note: this could be a glob() if we want to support multiple
        # input/output test cases per test folder.
        test_data_flagfile_name = "test_data_flags.txt"
        if not (self.path.parent / test_data_flagfile_name).exists():
            # TODO(scotttodd): support just compiling but not testing by omitting
            #   data flags? May need another config file next to the .mlir.
            print(f"Missing test_data_flags.txt for test '{test_name}'")
            return []

        global _iree_test_configs
        for config in _iree_test_configs:
            if test_name in config["skip_compile_tests"]:
                continue

            expect_compile_success = (
                test_name not in config["expected_compile_failures"]
            )
            expect_run_success = test_name not in config["expected_run_failures"]
            skip_run = test_name in config["skip_run_tests"]
            config_name = config["config_name"]
            spec = IreeCompileAndRunTestSpec(
                test_directory=test_directory,
                input_mlir_name=self.path.name,
                input_mlir_stem=self.path.stem,
                data_flagfile_name=test_data_flagfile_name,
                config_name=config_name,
                iree_compile_flags=config["iree_compile_flags"],
                iree_run_module_flags=config["iree_run_module_flags"],
                expect_compile_success=expect_compile_success,
                expect_run_success=expect_run_success,
                skip_run=skip_run,
            )
            yield IreeCompileRunItem.from_parent(self, name=config_name, spec=spec)


class IreeCompileRunItem(pytest.Item):
    """Test invocation item for an IREE compile + run test case."""

    spec: IreeCompileAndRunTestSpec

    def __init__(self, spec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec

        # TODO(scotttodd): swap cwd for a temp path?
        self.test_cwd = self.spec.test_directory
        vmfb_name = f"{self.spec.input_mlir_stem}_{self.spec.config_name}.vmfb"

        self.compile_args = ["iree-compile", self.spec.input_mlir_name]
        self.compile_args.extend(self.spec.iree_compile_flags)
        self.compile_args.extend(["-o", str(vmfb_name)])

        self.run_args = ["iree-run-module", f"--module={vmfb_name}"]
        self.run_args.extend(self.spec.iree_run_module_flags)
        self.run_args.append(f"--flagfile={self.spec.data_flagfile_name}")

    def runtest(self):
        if not self.spec.expect_compile_success:
            self.add_marker(
                pytest.mark.xfail(
                    raises=IreeCompileException,
                    strict=True,
                    reason="Expected compilation to fail",
                )
            )
        self.test_compile()

        if not self.spec.expect_compile_success or self.spec.skip_run:
            return

        if not self.spec.expect_run_success:
            self.add_marker(
                pytest.mark.xfail(
                    raises=IreeRunException,
                    strict=True,
                    reason="Expected run to fail",
                )
            )
        self.test_run()

    def test_compile(self):
        proc = subprocess.run(self.compile_args, capture_output=True, cwd=self.test_cwd)
        if proc.returncode != 0:
            raise IreeCompileException(proc, self.test_cwd)

    def test_run(self):
        proc = subprocess.run(self.run_args, capture_output=True, cwd=self.test_cwd)
        if proc.returncode != 0:
            raise IreeRunException(proc, self.test_cwd, self.compile_args)

    def repr_failure(self, excinfo):
        """Called when self.runtest() raises an exception."""
        if isinstance(excinfo.value, (IreeCompileException, IreeRunException)):
            return "\n".join(excinfo.value.args)
        # TODO(scotttodd): XFAIL tests spew a ton of logs here when run with `pytest -rA`. Fix?
        return super().repr_failure(excinfo)

    def reportinfo(self):
        display_name = f"{self.path.parent.name}::{self.name}"
        return self.path, 0, f"IREE compile and run: {display_name}"

    # Defining this for pytest-retry to avoid an AttributeError.
    def _initrequest(self):
        pass


class IreeCompileException(Exception):
    """Compiler exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)  # Decode error or other: best we can do.

        super().__init__(
            f"Error invoking iree-compile\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n\n"
            f"Invoked with:\n"
            f"  cd {cwd} && {' '.join(process.args)}\n\n"
        )


class IreeRunException(Exception):
    """Runtime exception that preserves the command line and error output."""

    def __init__(
        self, process: subprocess.CompletedProcess, cwd: str, compile_args: List[str]
    ):
        # iree-run-module sends output to both stdout and stderr
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)  # Decode error or other: best we can do.
        try:
            outs = process.stdout.decode("utf-8")
        except:
            outs = str(process.stdout)  # Decode error or other: best we can do.

        super().__init__(
            f"Error invoking iree-run-module\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n"
            f"Stdout diagnostics:\n{outs}\n"
            f"Compiled with:\n"
            f"  cd {cwd} && {' '.join(compile_args)}\n\n"
            f"Run with:\n"
            f"  cd {cwd} && {' '.join(process.args)}\n\n"
        )


# TODO(scotttodd): move this setup code into a (scoped) function?
#   Is there some way to share state across pytest functions?

# Load a list of configuration files following this schema:
#   {
#     "config_name": str,
#     "iree_compile_flags": list of str,
#     "iree_run_module_flags": list of str,
#     "skip_compile_tests": list of str,
#     "skip_run_tests": list of str,
#     "expected_compile_failures": list of str,
#     "expected_run_failures": list of str
#   }
#
# For example, to test the on CPU with the `llvm-cpu`` backend on the `local-task` device:
#   {
#     "config_name": "cpu",
#     "iree_compile_flags": ["--iree-hal-target-backends=llvm-cpu"],
#     "iree_run_module_flags": ["--device=local-task"],
#     "skip_compile_tests": [],
#     "skip_run_tests": [],
#     "expected_compile_failures": ["test_abs"],
#     "expected_run_failures": ["test_add"],
#   }
#
# TODO(scotttodd): expand schema with more flexible include_tests/exclude_tests fields.
#   * One use case is wanting to run only a small, controlled subset of tests, without needing to
#     manually exclude any new tests that might be added in the future.
#
# First check for the `IREE_TEST_CONFIG_FILES` environment variable. If defined,
# this should point to a semicolon-delimited list of config file paths, e.g.
# `export IREE_TEST_CONFIG_FILES=/iree/config_cpu.json;/iree/config_gpu.json`.
_iree_test_configs = []
_iree_test_config_files = [
    config for config in os.getenv("IREE_TEST_CONFIG_FILES", "").split(";") if config
]

# If no config files were specified via the environment variable, default to in-tree config files.
if not _iree_test_config_files:
    THIS_DIR = Path(__file__).parent
    REPO_ROOT = THIS_DIR.parent
    _iree_test_config_files = [
        REPO_ROOT / "iree_tests/configs/config_cpu_llvm_sync.json",
        # REPO_ROOT / "iree_tests/configs/config_gpu_vulkan.json",
    ]

for config_file in _iree_test_config_files:
    with open(config_file) as f:
        _iree_test_configs.append(pyjson5.load(f))
