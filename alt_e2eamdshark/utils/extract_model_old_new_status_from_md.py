# Copyright 2025 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import re
import sys
import json

out_path = Path("model_old_new_status.json")
md_path = Path(sys.argv[1])
text = md_path.read_text()

if re.search(r"##\s+No\s+Regression[s]?\s+Found", text, re.IGNORECASE):
    print("No regressions explicitly reported")
    sys.exit(0)

match = re.search(
    r"##\s+.+?\s+Regression[s]?\s+Found:?\s*\n+"
    r"(.*?)"
    r"\n##\s+(?:No|.+?)\s+Progression[s]?\s+Found:?",
    text,
    re.DOTALL,
)

if not match:
    print("No regressions found section")
    sys.exit(0)

section = match.group(1)

rows = re.findall(r"^\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|$", section, re.MULTILINE)

regressions = {}

for model, old_status, new_status in rows:
    model = model.strip()
    old_status = old_status.strip()
    new_status = new_status.strip()

    if model.lower() == "model name" or model == "---":
        continue

    regressions[model] = {
        "old_status": old_status,
        "new_status": new_status,
    }

with out_path.open("w") as f:
    json.dump(regressions, f, indent=2)

print(f"Wrote {len(regressions)} regressions to {out_path}")
