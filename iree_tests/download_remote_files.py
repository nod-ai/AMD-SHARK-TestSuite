# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from azure.storage.blob import ContainerClient
from pathlib import Path
import argparse
import pyjson5

THIS_DIR = Path(__file__).parent


# TODO(scotttodd): multithread? async?
# TODO(scotttodd): skip download if already exists? check some metadata
def download_azure_remote_files(
    test_dir: Path, container_client: ContainerClient, remote_file_group: dict
):
    base_blob_name = remote_file_group["azure_base_blob_name"]

    for remote_file in remote_file_group["files"]:
        print(f"  Downloading {remote_file} to {test_dir.relative_to(THIS_DIR)}")
        blob_name = base_blob_name + remote_file
        dest = test_dir / remote_file

        with open(dest, mode="wb") as local_blob:
            download_stream = container_client.download_blob(
                blob_name, max_concurrency=4
            )
            local_blob.write(download_stream.readall())


def download_for_test_case(test_dir: Path, test_case_json: dict):
    for remote_file_group in test_case_json["remote_file_groups"]:
        account_url = remote_file_group["azure_account_url"]
        container_name = remote_file_group["azure_container_name"]

        with ContainerClient(
            account_url,
            container_name,
            max_chunk_get_size=1024 * 1024 * 32,  # 32 MiB
            max_single_get_size=1024 * 1024 * 32,  # 32 MiB
        ) as container_client:
            download_azure_remote_files(test_dir, container_client, remote_file_group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote file downloader.")
    parser.add_argument(
        "--root-dir",
        default="",
        help="Root directory to search for files to download from (e.g. 'pytorch/models/resnet50')",
    )
    args = parser.parse_args()

    for test_cases_path in (THIS_DIR / args.root_dir).rglob("test_cases.json"):
        print(f"Processing {test_cases_path.relative_to(THIS_DIR)}")

        test_dir = test_cases_path.parent
        with open(test_cases_path) as f:
            test_cases_json = pyjson5.load(f)
            for test_case_json in test_cases_json["test_cases"]:
                download_for_test_case(test_dir, test_case_json)
