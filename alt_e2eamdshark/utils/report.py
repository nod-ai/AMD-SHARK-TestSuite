# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import json
import io
from typing import List, Dict

SUMMARY_STAGES = [
    "Setup",
    "IREE Compilation",
    "Gold Inference",
    "IREE Inference Invocation",
    "Inference Comparison",
]

STAGE_SIMPLIFICATION = {
    "setup": "Setup",
    "import_model": "IREE Compilation",
    "preprocessing": "IREE Compilation",
    "compilation": "IREE Compilation",
    "construct_inputs": "Gold Inference",
    "native_inference": "Gold Inference",
    "compiled_inference": "IREE Inference Invocation",
    "postprocessing": "Inference Comparison",
    "results-summary": "Inference Comparison",
    "Numerics": "Inference Comparison",
}


def save_dict(status_dict: Dict[str, Dict], status_dict_json: str):
    with io.open(status_dict_json, "w", encoding="utf8") as outfile:
        dict_str = json.dumps(
            status_dict,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=False,
        )
        outfile.write(dict_str)


def get_exit_status_counts(
    stages: List, status_dict: Dict[str, Dict], simplify: bool
) -> Dict[str, int]:
    use_stages = SUMMARY_STAGES if simplify else stages
    counts = {s: 0 for s in use_stages}
    for key, value in status_dict.items():
        if value["exit_status"] == "PASS":
            continue
        stage_name = (
            STAGE_SIMPLIFICATION[value["exit_status"]]
            if simplify
            else value["exit_status"]
        )
        counts[stage_name] += 1
    return counts


def get_stage_pass_counts(exit_counts: Dict[str, int], total: int) -> Dict[str, int]:
    counts = dict()
    running_total = total
    for key, value in exit_counts.items():
        if key == "PASS":
            continue
        running_total -= value
        counts[key] = running_total
    return counts


def safe_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def get_exit_status_string(counts: Dict[str, int], total) -> str:
    results_str = "## Fail Summary\n\n"
    results_str += f"**TOTAL TESTS = {total}**\n"
    results_str += f"|Stage|# Failed at Stage|% of Total|\n|--|--|--|\n"
    for key, value in counts.items():
        if key == "PASS":
            continue
        results_str += (
            f"| {key} | {value} | {round(safe_div(value, total)*100, 1)}% |\n"
        )
    return results_str


def get_stage_pass_string(counts: Dict[str, int], total: int) -> str:
    results_str = f"## Passing Summary\n\n"
    results_str += f"**TOTAL TESTS = {total}**\n"
    results_str += f"|Stage|# Passing|% of Total|% of Attempted|\n|--|--|--|--|\n"
    running_val = total
    for key, value in counts.items():
        if key == "Inference Comparison":
            key = "Inference Comparison (PASS)"
        results_str += f"| {key} | {value} | {round(safe_div(value, total)*100, 1)}% | {round(safe_div(value, running_val)*100, 1)}% |\n"
        running_val = value
    return results_str


def get_detail_string(status_dict: Dict[str, Dict]) -> str:
    report_string = "| Test | Exit Status | Mean Benchmark Time (ms) | Notes |\n"
    report_string += "|--|--|--|--|\n"
    for key, value in status_dict.items():
        report_string += f"| {key} | {value['exit_status']} | {value['time_ms']} | |\n"
    return report_string

def generate_error_report(status_dict: Dict[str, Dict], report_file: str):
    """Generate detailed error report for compilation failures, appending to existing report."""
    
    error_report_file = report_file.replace('.md', '_compilation_errors.md')
    # Filter for compilation failures only
    compilation_failures = {
        name: data for name, data in status_dict.items()
        if data.get('exit_status') == 'compilation' and 'compilation_status' in data
    }
    
    if not compilation_failures:
        return
    
    # Count by compilation status
    timeout_count = sum(1 for data in compilation_failures.values() if data.get('compilation_status') == 'timeout')
    error_count = sum(1 for data in compilation_failures.values() if data.get('compilation_status') == 'error')
    no_log_count = sum(1 for data in compilation_failures.values() if data.get('compilation_status') == 'no_log')
    
    with open(error_report_file, 'a', encoding='utf-8') as f:
        # Write compilation error summary
        f.write("\n# COMPILATION ERROR ANALYSIS\n\n")
        f.write(f"**Total compilation failures: {len(compilation_failures)}**\n\n")
        f.write("| Status | Count |\n")
        f.write("|---|---|\n")
        f.write(f"| Timeouts | {timeout_count} |\n")
        f.write(f"| Errors | {error_count} |\n")
        f.write(f"| No log | {no_log_count} |\n\n")
        
        # ERROR TESTS (grouped by error)
        if error_count > 0:
            f.write("## ERRORS (grouped by error)\n\n")
            # Group tests by error signature
            error_groups: Dict[str, List] = {}
            for name, data in compilation_failures.items():
                if data.get('compilation_status') == 'error':
                    error_key = data.get('error_signature', '') or "(no error details)"
                    if error_key not in error_groups:
                        error_groups[error_key] = []
                    error_groups[error_key].append((name, data))
            
            # Summary table sorted by count
            f.write("| Error | Count |\n")
            f.write("|---|---|\n")
            for error_signature, tests in sorted(error_groups.items(), key=lambda x: len(x[1]), reverse=True):
                error_escaped = error_signature.replace('|', '\\|').replace('\n', ' ')
                f.write(f"| {error_escaped} | {len(tests)} |\n")
            f.write("\n")
            
            f.write("## ERROR TESTS (grouped by error)\n\n")
            
            # Detailed tables grouped by error
            for error_signature, tests in sorted(error_groups.items(), key=lambda x: len(x[1]), reverse=True):
                error_escaped = error_signature.replace('|', '\\|').replace('\n', ' ')
                f.write(f"### Error: {error_escaped}\n\n")
                f.write(f"**Count: {len(tests)}**\n\n")
                f.write("| test_name | command | error |\n")
                f.write("|---|---|---|\n")
                for name, data in tests:
                    command = data.get('command', '').replace('|', '\\|').replace('\n', ' ')
                    error = data.get('error', '').replace('|', '\\|').replace('\n', ' ')
                    f.write(f"| {name} | {command} | {error} |\n")
                f.write("\n")
        
        # TIMEOUT TESTS
        if timeout_count > 0:
            f.write("## TIMEOUT TESTS\n\n")
            f.write("| test_name | command |\n")
            f.write("|---|---|\n")
            for name, data in compilation_failures.items():
                if data.get('compilation_status') == 'timeout':
                    command = data.get('command', '').replace('|', '\\|').replace('\n', ' ')
                    f.write(f"| {name} | {command} |\n")
            f.write("\n")
        
        # NO LOG FILE
        if no_log_count > 0:
            f.write("## NO LOG FILE\n\n")
            f.write("| test_name |\n")
            f.write("|---|\n")
            for name, data in compilation_failures.items():
                if data.get('compilation_status') == 'no_log':
                    f.write(f"| {name} |\n")
            f.write("\n")


def generate_report(
    args, og_stages: List, status_dict: Dict[str, Dict], *, simplify: bool = True
):
    """generates a markdown report for a test-run"""

    # add non-stage exit statuses:
    stages = og_stages
    stages.append("results-summary")
    stages.append("Numerics")

    # get some counts
    exit_counts = get_exit_status_counts(stages, status_dict, simplify=simplify)
    total = len(status_dict.keys())
    pass_counts = get_stage_pass_counts(exit_counts, total)

    # generate report summary strings
    results_str = get_stage_pass_string(pass_counts, total)
    exit_status_str = get_exit_status_string(exit_counts, total)

    # set up report detail
    args_string = (
        f"## Test Run Detail\nTest was run with the following arguments:\n{args}\n\n"
    )

    detail_string = get_detail_string(status_dict)

    # get a report file and write to it
    with open(args.report_file, "w") as file:
        file.write(results_str)
        file.write(exit_status_str)
        file.write(args_string)
        file.write(detail_string)

    generate_error_report(status_dict, args.report_file) 
