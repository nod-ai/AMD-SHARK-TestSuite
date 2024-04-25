# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script to help with updating config.json files used with conftest.py.
#
# Usage:
#   1. Run tests with a config file
#   2. Collect log output from the test run
#   3. Parse through the logs to get lists of tests that need moving around
#      (TODO: automate that)
#   4. Sort those tests into sections in the `TESTS_` lists at the top of this file
#   5. Run the script: `python update_config_xfails.py --config-file=config.json`
#   6. Commit the modified script

import argparse
import json
import logging
import pyjson5

logger = logging.getLogger(__name__)

# TODO(scotttodd): pull these from pytest (direct), pytest logs (logger to file)
#                  or raw logs (regex)

# Test cases failing like this:
#   Expected compile failure but run failed (move to 'expected_run_failures')
TESTS_MOVE_COMPILE_FAILURE_TO_RUN_FAILURE = [
    # Add tests here, like:
    # "test_top_k",
]

# Test cases failing like this:
#   [XPASS(strict)] Expected compilation to fail (remove from 'expected_compile_failures')
TESTS_REMOVE_COMPILE_FAILURE = [
    #
]

# Test cases failing like this:
#   [XPASS(strict)] Expected run to fail (remove from 'expected_run_failures')
TESTS_REMOVE_RUN_FAILURE = [
    #
]

# Test cases with `Error invoking iree-compile` that _aren't_ marked XFAIL:
TESTS_ADD_COMPILE_FAILURE = [
    #
]

# Test cases with `Error invoking iree-run-module` that _aren't_ marked XFAIL:
TEST_ADD_RUN_FAILURE = [
    #
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file updater.")
    parser.add_argument(
        "--config-file",
        default="",
        required=True,
        help="Path to a config JSON file to update",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config_file = args.config_file

    logger.info(f"Reading config from '{config_file}'")
    with open(config_file, "r") as f:
        config = pyjson5.load(f)

        # Sanity check the config file structure before going any further.
        def check_field(field_name):
            if field_name not in config:
                raise ValueError(
                    f"config file '{config_file}' is missing a '{field_name}' field"
                )

        check_field("config_name")
        check_field("iree_compile_flags")
        check_field("iree_run_module_flags")

    logger.info(f"Updating config")
    for test in TESTS_MOVE_COMPILE_FAILURE_TO_RUN_FAILURE:
        if test in config["expected_compile_failures"]:
            config["expected_compile_failures"].remove(test)
        if test not in config["expected_run_failures"]:
            config["expected_run_failures"].append(test)
    for test in TESTS_REMOVE_COMPILE_FAILURE:
        if test in config["expected_compile_failures"]:
            config["expected_compile_failures"].remove(test)
    for test in TESTS_REMOVE_RUN_FAILURE:
        if test in config["expected_run_failures"]:
            config["expected_run_failures"].remove(test)
    for test in TESTS_ADD_COMPILE_FAILURE:
        if test not in config["expected_compile_failures"]:
            config["expected_compile_failures"].append(test)
    for test in TEST_ADD_RUN_FAILURE:
        if test not in config["expected_run_failures"]:
            config["expected_run_failures"].append(test)

    config["expected_compile_failures"].sort()
    config["expected_run_failures"].sort()

    logger.info(f"Writing updated config to '{config_file}'")
    with open(config_file, "w") as f:
        f.write(json.dumps(config, indent=2))
