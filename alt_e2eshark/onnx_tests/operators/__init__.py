# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from importlib import import_module

this_dir = Path(__file__).parent

for file in this_dir.glob("*.py"):
    if file.stem.startswith("_") or file.stem in globals():
        continue
    import_module(f".{file.stem}", __package__)
