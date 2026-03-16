# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
import traceback
import shutil
import subprocess
import time
from e2e_testing.framework import result_comparison

from typing import List, Dict
from e2e_testing.stage_handler import CompilationErrorHandler


def run_command_and_log(command: List[str], save_to: str, stage_name: str) -> None:
    """Runs command through subprocess.run and logs the command and error details (if present)"""
    # convert command list to a string
    script = subprocess.list2cmdline(command)
    # setup a commands subdirectory (if it doesn't exist)
    commands_dir = Path(save_to) / "commands"
    commands_dir.mkdir(exist_ok=True)
    # log the command
    commands_log = commands_dir / f"{stage_name}.commands.log"
    commands_log.write_text(script)

    # Log start time and command for verbose output
    start_time = time.time()
    timeout_seconds = 600  # Increased from 100 to 600 seconds (10 minutes)
    print(f"[{stage_name}] Starting command (timeout={timeout_seconds}s)...")

    # run the command
    try:
        ret = subprocess.run(script, shell=True, capture_output=True, timeout=timeout_seconds)
        elapsed_time = time.time() - start_time
        print(f"[{stage_name}] Completed in {elapsed_time:.2f}s")

        # Log timing information
        timing_log = commands_dir / f"{stage_name}.timing.log"
        timing_log.write_text(f"Execution time: {elapsed_time:.2f}s\nTimeout limit: {timeout_seconds}s\n")

    except subprocess.TimeoutExpired as e:
        elapsed_time = time.time() - start_time
        print(f"[{stage_name}] TIMEOUT after {elapsed_time:.2f}s (limit: {timeout_seconds}s)")
        # Log timeout details
        detail_dir = Path(save_to) / "detail"
        detail_dir.mkdir(exist_ok=True)
        detail_log = detail_dir / f"{stage_name}.detail.log"
        timeout_msg = f"subprocess.TimeoutExpired: Command exceeded {timeout_seconds}s timeout\n"
        timeout_msg += f"Actual execution time: {elapsed_time:.2f}s\n"
        timeout_msg += f"Command: {script}\n"
        if e.stdout:
            timeout_msg += f"\n--- STDOUT ---\n{e.stdout.decode()}\n"
        if e.stderr:
            timeout_msg += f"\n--- STDERR ---\n{e.stderr.decode()}\n"
        detail_log.write_text(timeout_msg)
        raise RuntimeError(f"Timeout after {elapsed_time:.2f}s (limit: {timeout_seconds}s): {script}")

    # if an error occured, log stderr and raise exception
    if ret.returncode != 0:
        elapsed_time = time.time() - start_time
        print(f"[{stage_name}] FAILED with exit code {ret.returncode} after {elapsed_time:.2f}s")
        detail_dir = Path(save_to) / "detail"
        detail_dir.mkdir(exist_ok=True)
        detail_log = detail_dir / f"{stage_name}.detail.log"
        error_content = f"Exit code: {ret.returncode}\nExecution time: {elapsed_time:.2f}s\n\n--- STDERR ---\n{ret.stderr.decode()}"
        if ret.stdout:
            error_content += f"\n\n--- STDOUT ---\n{ret.stdout.decode()}"
        detail_log.write_text(error_content)
        error_msg = f"failure executing command:\n{script}\n"
        error_msg += f"Exit code: {ret.returncode}\n"
        error_msg += f"Error detail in '{detail_log}'"
        raise RuntimeError(error_msg)


def log_result(result, log_dir, tol):
    # TODO: add more information for the result comparison (e.g., on verbose, add information on where the error is occuring, etc)
    summary = result_comparison(result, tol)
    num_match = 0
    num_total = 0
    for s in summary:
        num_match += s.sum().item()
        num_total += s.nelement()
    percent_correct = num_match / num_total if num_total != 0 else "N/A"
    with open(log_dir + "inference_comparison.log", "w+") as f:
        f.write(
            f"matching values with (rtol,atol) = {tol}: {num_match} of {num_total} = {percent_correct*100}%\n"
        )
        f.write(f"Test Result:\n{result}")
    return num_match == num_total


def log_error(status_dict: Dict[str, Dict], log_dir: str, stage: str, name: str):
    """
    Populate status_dict with error information for a given test.

    Args:
        status_dict: Dictionary to populate with error info
        log_dir: Base directory containing the test logs
        stage: The stage of the test (e.g., 'compilation', 'inference')
        name: The test name
    """
    if stage == "compilation":
        handler = CompilationErrorHandler(log_dir, stage)
        handler.populate_status_dict(status_dict, name)


def log_exception(e: Exception, path: str, stage: str, name: str, verbose: bool):
    """generates a log for an exception generated during a testing stage"""
    log_filename = path + stage + ".log"
    base_str = f"Failed test at stage {stage} with exception:\n{e}\n"
    with open(log_filename, "w") as f:
        f.write(base_str)
        if verbose:
            print(f"\tFAILED ({stage})" + " " * 20)
            traceback.print_exception(e, file=f)
        else:
            print(f"FAILED: {name}")


def scan_dir_del_if_large(dir, size_MB):
    remove_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            curr_file = os.path.join(root, name)
            size_bytes = os.path.getsize(curr_file)
            if size_bytes >= size_MB * (10**6):
                remove_files.append(curr_file)
    for file in remove_files:
        os.remove(file)
    return remove_files


def scan_dir_del_mlir_vmfb(dir):
    removed_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            curr_file = os.path.join(root, name)
            if name.endswith(".mlir") or name.endswith(".vmfb"):
                removed_files.append(curr_file)
    for file in removed_files:
        os.remove(file)
    return removed_files


def scan_dir_del_not_logs(dir):
    removed_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            curr_file = os.path.join(root, name)
            if not name.endswith(".log") and not name.endswith(".json"):
                removed_files.append(curr_file)
    for file in removed_files:
        os.remove(file)
    return removed_files


def post_test_clean(log_dir, cleanup, verbose):
    match cleanup:
        case 0:
            return
        case 1:
            files = scan_dir_del_if_large(log_dir, 500)
        case 2:
            files = scan_dir_del_mlir_vmfb(log_dir)
        case 3:
            files = scan_dir_del_not_logs(log_dir)
        case 4:
            shutil.rmtree(Path(log_dir))
