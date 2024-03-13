# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import pyjson5
import requests

THIS_DIR = Path(__file__).parent


# TODO(scotttodd): checksum / stamp to avoid re-downloading
# TODO(scotttodd): fast path for known hosts?
#     https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-download-python
def download_file(uri: str, dest: Path) -> Path:
    req = requests.get(uri, stream=True, timeout=60)
    if not req.ok:
        raise RuntimeError(f"Failed to fetch {uri}: {req.status_code} - {req.text}")
    with dest.open("wb") as dest_file:
        chunk_size = 512 * 1024 * 1024
        block = 0
        for data in req.iter_content(chunk_size=chunk_size):
            dest_file.write(data)
            block += chunk_size
            print(f"    {block}")
    return dest


# TODO(scotttodd): multithread, see onnx/import_tests.py

if __name__ == "__main__":
    for test_cases_path in Path(THIS_DIR).rglob("test_cases.json"):
        print(f"Processing {test_cases_path.relative_to(THIS_DIR)}")

        test_dir = test_cases_path.parent
        with open(test_cases_path) as f:
            test_cases_json = pyjson5.load(f)
            for test_case_json in test_cases_json["test_cases"]:
                for remote_file_group in test_case_json["remote_files"]:
                    base_url = remote_file_group["base_url"]
                    for remote_file in remote_file_group["files"]:
                        print(
                            f"  Downloading {remote_file} to {test_dir.relative_to(THIS_DIR)}"
                        )
                        uri = base_url + remote_file
                        dest = test_dir / remote_file
                        download_file(uri, dest)
