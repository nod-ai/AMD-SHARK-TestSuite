# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import json
import io

def save_dict(status_dict, status_dict_json):
    with io.open(status_dict_json, "w", encoding="utf8") as outfile:
        dict_str = json.dumps(status_dict, indent=4, sort_keys=True, separators=(',',': '), ensure_ascii=False)
        outfile.write(dict_str)

def generate_report(args, stages, status_dict):
    """generates a markdown report for a test-run"""

    # set up report summary
    stages.append("results-summary")
    stages.append("Numerics")
    stages.append("PASS")
    stages.reverse()
    counts = {s : 0 for s in stages}
    for (key, value) in status_dict.items():
        counts[value] += 1
    results_str = "## Summary\n\n|Stage|Count|\n|--|--|\n"
    results_str += f"| Total | {len(status_dict.keys())} |\n"
    for (key, value) in counts.items():
        results_str += f"| {key} | {value} |\n"

    # set up report detail
    report_string = f"\n## Test Run Detail \nTest was run with the following arguments:\n{args}\n\n"
    report_string += "| Test | Exit Status | Notes |\n"
    report_string += "|--|--|--|\n"
    for (key, value) in status_dict.items():
        report_string += f"| {key} | {value} | |\n"

    # get a report file and write to it 
    with open(args.report_file, "w") as file:
        file.write(results_str)
        file.write(report_string)