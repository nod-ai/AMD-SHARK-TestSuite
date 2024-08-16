# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from pathlib import Path
from typing import List
import argparse
import logging
import pyjson5
import os
import pytest
import subprocess
from ireers import *

IREE_TESTS_ROOT = Path(__file__).parent
TEST_DATA_FLAGFILE_NAME = "test_data_flags.txt"

# --------------------------------------------------------------------------- #
# pytest hooks
# https://docs.pytest.org/en/stable/reference/reference.html#initialization-hooks
# https://docs.pytest.org/en/stable/reference/reference.html#collection-hooks


def pytest_addoption(parser):
    # List of configuration files following this schema:
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
    # For example, to test on CPU with the `llvm-cpu` backend and `local-task` device:
    #   {
    #     "config_name": "cpu_llvm_task",
    #     "iree_compile_flags": ["--iree-hal-target-backends=llvm-cpu"],
    #     "iree_run_module_flags": ["--device=local-task"],
    #     "skip_compile_tests": [],
    #     "skip_run_tests": [],
    #     "expected_compile_failures": ["test_abs"],
    #     "expected_run_failures": ["test_add"],
    #   }
    #
    # The list of files can be specified in (by order of preference):
    #   1. The `--config-files` argument
    #       e.g. `pytest ... --config-files foo.json bar.json`
    #   2. The `IREE_TEST_CONFIG_FILES` environment variable
    #       e.g. `set IREE_TEST_CONFIG_FILES=foo.json;bar.json`
    #   3. A default config file used for testing the test suite itself
    default_config_files = [
        f for f in os.getenv("IREE_TEST_CONFIG_FILES", "").split(";") if f
    ]
    if not default_config_files:
        this_dir = Path(__file__).parent
        repo_root = this_dir.parent
        default_config_files = [
            repo_root / "iree_tests/configs/cpu_llvm_sync.json",
        ]
    parser.addoption(
        "--config-files",
        action="store",
        nargs="*",
        default=default_config_files,
        help="List of config JSON files used to build test cases",
    )

    parser.addoption(
        "--ignore-xfails",
        action="store_true",
        default=False,
        help="Ignores expected compile/run failures from configs, to print all error output",
    )

    parser.addoption(
        "--skip-all-runs",
        action="store_true",
        default=False,
        help="Skips all 'run' tests, overriding 'skip_run_tests' in configs",
    )

    parser.addoption(
        "--skip-tests-missing-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skips any tests that are missing required files",
    )


def pytest_sessionstart(session):
    session.config.iree_test_configs = []
    for config_file in session.config.getoption("config_files"):
        with open(config_file) as f:
            test_config = pyjson5.load(f)

            # Sanity check the config file structure before going any further.
            def check_field(field_name):
                if field_name not in test_config:
                    raise ValueError(
                        f"config file '{config_file}' is missing a '{field_name}' field"
                    )

            check_field("config_name")
            check_field("iree_compile_flags")
            check_field("iree_run_module_flags")

            session.config.iree_test_configs.append(test_config)


def pytest_collect_file(parent, file_path):
    if file_path.name == TEST_DATA_FLAGFILE_NAME:
        return MlirCompileRunTest.from_parent(parent, path=file_path)

    if file_path.suffix == ".json":
        with open(file_path) as f:
            test_cases_json = pyjson5.load(f)
            if test_cases_json.get("file_format", "") == "test_cases_v0":
                return MlirCompileRunTest.from_parent(parent, path=file_path)


# --------------------------------------------------------------------------- #


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

    # Name of the test configuration, e.g. "cpu_splats".
    # This will be used in generated files and test case names.
    test_name: str

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

    # True to skip the test entirely (but still define it).
    skip_test: bool


class MlirCompileRunTest(pytest.File):
    """Collector for MLIR -> compile -> run tests anchored on a file."""

    @dataclass(frozen=True)
    class TestCase:
        name: str
        mlir_file: str
        runtime_flagfile: str
        enabled: bool

    def check_for_lfs_files(self):
        """Checks if git LFS files are checked out."""
        have_lfs_files = True
        if self.path.stat().st_size < 1000:
            with open(self.path, "rt") as f:
                first_line = f.readline()
                if "git-lfs" in first_line:
                    have_lfs_files = False
        return have_lfs_files

    def check_for_remote_files(self, test_case_json):
        """Checks if all remote_files in a JSON test case exist on disk."""
        if "remote_files" not in test_case_json:
            return True

        have_all_files = True
        for remote_file_url in test_case_json["remote_files"]:
            remote_file = remote_file_url.rsplit("/", 1)[-1]
            if not (self.path.parent / remote_file).exists():
                test_case_name = test_case_json["name"]
                print(
                    f"Missing file '{remote_file}' for test {self.path.parent.name}::{test_case_name}"
                )
                have_all_files = False
        return have_all_files

    def discover_test_cases(self):
        """Discovers test cases in either test_data_flags.txt or *.json files."""
        test_cases = []

        skip_missing = self.config.getoption("skip_tests_missing_files")
        have_lfs_files = self.check_for_lfs_files()

        mlir_files = sorted(self.path.parent.glob("*.mlir*"))
        if len(mlir_files) == 0:
            if not skip_missing:
                raise FileNotFoundError(
                    f"Missing .mlir file for test {self.path.parent.name}"
                )
            return test_cases
        assert len(mlir_files) <= 1, "Test directories may only contain one .mlir file"
        mlir_file = mlir_files[0]

        if self.path.name == TEST_DATA_FLAGFILE_NAME:
            test_cases.append(
                MlirCompileRunTest.TestCase(
                    name="",
                    mlir_file=mlir_file,
                    runtime_flagfile=TEST_DATA_FLAGFILE_NAME,
                    enabled=have_lfs_files,
                )
            )
        elif self.path.suffix == ".json":
            with open(self.path) as f:
                test_cases_json = pyjson5.load(f)
                if test_cases_json.get("file_format", "") == "test_cases_v0":
                    for test_case_json in test_cases_json["test_cases"]:
                        test_case_name = test_case_json["name"]
                        have_remote_files = self.check_for_remote_files(test_case_json)
                        have_all_files = have_lfs_files and have_remote_files

                        if not skip_missing and not have_all_files:
                            raise FileNotFoundError(
                                f"Missing files for test {self.path.parent.name}::{test_case_name}"
                            )
                        test_cases.append(
                            MlirCompileRunTest.TestCase(
                                name=test_case_name,
                                mlir_file=mlir_file,
                                runtime_flagfile=test_case_json["runtime_flagfile"],
                                enabled=have_all_files,
                            )
                        )

        return test_cases

    def collect(self):
        # Expected directory structure:
        #   path/to/test_some_ml_operator/
        #     - *.mlir[bc]
        #     - test_data_flags.txt OR *.json using "test cases" schema
        #   path/to/test_some_ml_model/
        #     ...

        test_directory = self.path.parent
        relative_test_directory = test_directory.relative_to(IREE_TESTS_ROOT).as_posix()
        test_directory_name = test_directory.name

        test_cases = self.discover_test_cases()
        if len(test_cases) == 0:
            logging.getLogger().debug(f"No test cases for '{test_directory_name}'")
            return []

        for config in self.config.iree_test_configs:
            if relative_test_directory in config.get("skip_compile_tests", []):
                continue

            expect_compile_success = self.config.getoption(
                "ignore_xfails"
            ) or relative_test_directory not in config.get(
                "expected_compile_failures", []
            )
            expect_run_success = self.config.getoption(
                "ignore_xfails"
            ) or relative_test_directory not in config.get("expected_run_failures", [])
            skip_run = self.config.getoption(
                "skip_all_runs"
            ) or relative_test_directory in config.get("skip_run_tests", [])
            config_name = config["config_name"]

            for test_case in test_cases:
                # Generate test item names like 'model.mlir::cpu_llvm_sync::splats'.
                # These show up in pytest output.
                mlir_file = test_case.mlir_file
                name_parts = [
                    e for e in [mlir_file.name, config_name, test_case.name] if e
                ]
                item_name = "::".join(name_parts)
                # Generate simpler test names to use in filenames.
                test_name = config_name + "_" + test_case.name

                spec = IreeCompileAndRunTestSpec(
                    test_directory=test_directory,
                    input_mlir_name=mlir_file.name,
                    input_mlir_stem=mlir_file.stem,
                    data_flagfile_name=test_case.runtime_flagfile,
                    test_name=test_name,
                    iree_compile_flags=config["iree_compile_flags"],
                    iree_run_module_flags=config["iree_run_module_flags"],
                    expect_compile_success=expect_compile_success,
                    expect_run_success=expect_run_success,
                    skip_run=skip_run,
                    skip_test=not test_case.enabled,
                )
                yield IreeCompileRunItem.from_parent(self, name=item_name, spec=spec)


class IreeCompileRunItem(pytest.Item):
    """Test invocation item for an IREE compile + run test case."""

    spec: IreeCompileAndRunTestSpec

    def __init__(self, spec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec

        relative_test_directory = self.spec.test_directory.relative_to(
            IREE_TESTS_ROOT
        ).as_posix()
        self.user_properties.append(
            ("relative_test_directory_name", relative_test_directory)
        )
        self.user_properties.append(("input_mlir_name", self.spec.input_mlir_name))
        self.user_properties.append(("test_name", self.spec.test_name))

        if self.spec.skip_test:
            self.add_marker(
                pytest.mark.skip(f"{self.spec.test_name} missing required files")
            )
            return

        # TODO(scotttodd): swap cwd for a temp path?
        self.test_cwd = self.spec.test_directory

        self.run_args = []
        self.run_args.extend(self.spec.iree_run_module_flags)
        self.run_args.append(f"--flagfile={self.spec.data_flagfile_name}")

    def runtest(self):
        # TODO(scotttodd): log files needed by the test (remote files / git LFS)
        #     it should be easy to copy/paste commands from CI logs to get both
        #     the test files and the flags used with them

        # We want to test two phases: 'compile', and 'run'.
        # A test can be marked as expected to fail at either stage, with these
        # possible outcomes:

        # Expect 'compile' | Expect 'run' | Actual 'compile' | Actual 'run' | Result
        # ---------------- | ------------ | ---------------- | ------------ | ------
        #
        # PASS             | PASS         | PASS             | PASS         | PASS
        # PASS             | PASS         | FAIL             | N/A          | FAIL
        # PASS             | PASS         | PASS             | FAIL         | FAIL
        #
        # PASS             | FAIL         | PASS             | PASS         | XPASS
        # PASS             | FAIL         | FAIL             | N/A          | FAIL
        # PASS             | FAIL         | PASS             | FAIL         | XFAIL
        #
        # FAIL             | N/A          | PASS             | PASS         | XPASS
        # FAIL             | N/A          | FAIL             | N/A          | XFAIL
        # FAIL             | N/A          | PASS             | FAIL         | XPASS

        # * XFAIL and PASS are acceptable outcomes - they mean that the list of
        #   expected failures in the config file matched the test run.
        # * FAIL means that something expected to work did not. That's an error.
        # * XPASS means that a test is newly passing and can be removed from the
        #   expected failures list.

        if not self.spec.expect_compile_success:
            self.add_marker(
                pytest.mark.xfail(
                    raises=IreeCompileException,
                    strict=True,
                    reason="Expected compilation to fail (included in 'expected_compile_failures')",
                )
            )
        if not self.spec.expect_run_success:
            self.add_marker(
                pytest.mark.xfail(
                    raises=IreeRunException,
                    strict=True,
                    reason="Expected run to fail (included in 'expected_run_failures')",
                )
            )

        self.test_compile()

        if self.spec.skip_run:
            return

        try:
            self.test_run()
        except IreeRunException as e:
            if not self.spec.expect_compile_success:
                raise IreeXFailCompileRunException from e
            raise e

    def test_compile(self):
        mlir_name = self.spec.input_mlir_name
        vmfb_name = f"{self.spec.input_mlir_stem}_{self.spec.test_name}.vmfb"
        iree_compile(mlir_name, vmfb_name, self.spec.iree_compile_flags, self.test_cwd)

    def test_run(self):
        mlir_name = self.spec.input_mlir_name
        vmfb_name = f"{self.spec.input_mlir_stem}_{self.spec.test_name}.vmfb"
        compile_cmd = get_compile_cmd(
            mlir_name, vmfb_name, self.spec.iree_compile_flags
        )
        iree_run_module(vmfb_name, self.run_args, self.test_cwd, compile_cmd)

    def repr_failure(self, excinfo):
        """Called when self.runtest() raises an exception."""
        if isinstance(excinfo.value, (IreeCompileException, IreeRunException)):
            return "\n".join(excinfo.value.args)
        if isinstance(excinfo.value, IreeXFailCompileRunException):
            return (
                "Expected compile failure but run failed (move to 'expected_run_failures'):\n"
                + "\n".join(excinfo.value.__cause__.args)
            )
        return super().repr_failure(excinfo)

    def reportinfo(self):
        display_name = (
            f"{self.path.parent.name}::{self.spec.input_mlir_name}::{self.name}"
        )
        return self.path, 0, f"IREE compile and run: {display_name}"

    # Defining this for pytest-retry to avoid an AttributeError.
    def _initrequest(self):
        pass


class IreeXFailCompileRunException(Exception):
    pass
