# Copyright 2025 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import sys
from pathlib import Path

MODEL = sys.argv[1]
BASELINE_JSON = Path(sys.argv[2])
CURRENT_JSON = Path(sys.argv[3])
MD_FILE = Path(sys.argv[4])

# --- File existence checks ---
if not BASELINE_JSON.is_file():
    print(f"[ERROR] Baseline JSON not found: {BASELINE_JSON}")
    sys.exit(1)

if not CURRENT_JSON.is_file():
    print(f"[ERROR] Current JSON not found: {CURRENT_JSON}")
    sys.exit(1)

if not MD_FILE.is_file():
    print(f"[ERROR] Markdown report not found: {MD_FILE}")
    sys.exit(1)

baseline = json.loads(BASELINE_JSON.read_text())
current = json.loads(CURRENT_JSON.read_text())

if MODEL not in baseline:
    print(f"[ERROR] {MODEL} not found in baseline json")
    sys.exit(1)

if MODEL not in current:
    print(f"[ERROR] {MODEL} not found in current run json")
    sys.exit(1)

expected_status = baseline[MODEL]["old_status"]
actual_status = current[MODEL]["exit_status"]

print(f"Model: {MODEL}")
print(f"Expected (old) status: {expected_status}")
print(f"Actual status:         {actual_status}")

# --- STATUS MATCH ---
if actual_status == expected_status:
    print(f"STATUS MATCH (Model: {MODEL} ) -> removing from regression table")

    lines = MD_FILE.read_text().splitlines()
    new_lines = []

    in_regression_section = False

    for line in lines:
        stripped = line.strip()

        # Enter regressions section
        if (
            stripped.startswith("##")
            and "Regression" in stripped
            and "Found" in stripped
        ):
            in_regression_section = True
            new_lines.append(line)
            continue

        # Exit section on next header
        if in_regression_section and stripped.startswith("##"):
            in_regression_section = False

        # Skip this model's row
        if in_regression_section and stripped.startswith(f"|{MODEL}|"):
            print(f"Removed row: {line}")
            continue

        new_lines.append(line)

    MD_FILE.write_text("\n".join(new_lines) + "\n")

# --- STATUS MISMATCH ---
else:
    print("STATUS MISMATCH (Old Status != New Status) -> keeping in regression table")
