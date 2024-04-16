# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from azure.storage.blob import BlobClient, BlobProperties
from pathlib import Path
import argparse
import hashlib
import logging
import mmap
import pyjson5
import re
import os

THIS_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


def human_readable_size(size, decimal_places=2):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def get_remote_md5(remote_file: str, blob_properties: BlobProperties):
    content_settings = blob_properties.get("content_settings")
    if not content_settings:
        return None
    remote_md5 = content_settings.get("content_md5")
    if not remote_md5:
        logger.warning(
            f"  Remote file '{remote_file}' on Azure is missing the "
            "'content_md5' property, can't check if local matches remote"
        )
    return remote_md5


def get_local_md5(local_file_path: Path):
    if not local_file_path.exists() or local_file_path.stat().st_size == 0:
        return None

    with open(local_file_path) as file, mmap.mmap(
        file.fileno(), 0, access=mmap.ACCESS_READ
    ) as file:
        return hashlib.md5(file).digest()


def download_azure_remote_file(test_dir: Path, remote_file: str):
    remote_file_name = remote_file.rsplit("/", 1)[-1]
    relative_dir = test_dir.relative_to(THIS_DIR)

    # Extract path components from Azure URL to use with the Azure Storage Blobs
    # client library for Python (https://pypi.org/project/azure-storage-blob/).
    #
    # For example:
    #   https://sharkpublic.blob.core.windows.net/sharkpublic/path/to/blob.txt
    #                                            ^           ^
    #   account_url:    https://sharkpublic.blob.core.windows.net
    #   container_name: sharkpublic
    #   blob_name:      path/to/blob.txt
    #
    # Note: we could also use the generic handler (e.g. wget, 'requests'), but
    # the client library offers other APIs.

    result = re.search(r"(https.+\.net)/([^/]+)/(.+)", remote_file)
    account_url = result.groups()[0]
    container_name = result.groups()[1]
    blob_name = result.groups()[2]

    with BlobClient(
        account_url,
        container_name,
        blob_name,
        max_chunk_get_size=1024 * 1024 * 32,  # 32 MiB
        max_single_get_size=1024 * 1024 * 32,  # 32 MiB
    ) as blob_client:
        blob_properties = blob_client.get_blob_properties()
        blob_size_str = human_readable_size(blob_properties.size)
        remote_md5 = get_remote_md5(remote_file, blob_properties)

        cache_location = os.getenv("IREE_TEST_FILES", default="")
        if cache_location == "":
            os.environ["IREE_TEST_FILES"] = str(REPO_ROOT)
            cache_location = REPO_ROOT
        if cache_location == REPO_ROOT:
            local_dir_path = test_dir
            local_file_path = test_dir / remote_file_name
        else:
            cache_location = Path(os.path.expanduser(cache_location)).resolve()
            local_dir_path = cache_location / "iree_tests" / relative_dir
            local_file_path = cache_location / "iree_tests" / relative_dir / remote_file_name

        local_md5 = get_local_md5(local_file_path)

        if remote_md5 and remote_md5 == local_md5:
            logger.info(
                f"  Skipping '{remote_file_name}' download ({blob_size_str}) "
                "- local MD5 hash matches"
            )
            os.symlink(local_file_path, test_dir / remote_file_name)
            logger.info(
                f"  Created symlink for '{local_file_path}' to '{test_dir / remote_file_name}'"
            )
            return

        if not local_md5:
            logger.info(
                f"  Downloading '{remote_file_name}' ({blob_size_str}) "
                f"to '{relative_dir}'"
            )
        else:
            logger.info(
                f"  Downloading '{remote_file_name}' ({blob_size_str}) "
                f"to '{relative_dir}' (local MD5 does not match)"
            )

        if not os.path.isdir(local_dir_path):
            os.makedirs(local_dir_path)
        with open(local_file_path, mode="wb") as local_blob:
            download_stream = blob_client.download_blob(max_concurrency=4)
            local_blob.write(download_stream.readall())
        if str(cache_location) != str(REPO_ROOT):
            os.symlink(local_file_path, test_dir / remote_file_name)
            logger.info(
                f"  Created symlink for '{local_file_path}' to '{test_dir / remote_file_name}'"
            )


def download_generic_remote_file(test_dir: Path, remote_file: str):
    # TODO(scotttodd): use https://pypi.org/project/requests/
    raise NotImplementedError("generic remote file downloads not implemented yet")


def download_for_test_case(test_dir: Path, test_case_json: dict):
    # This is naive (greedy, serial) for now. We could batch downloads that
    # share a source:
    #   * Iterate over all files (across all included paths), building a list
    #     of files to download (checking hashes / local references before
    #     adding to the list)
    #   * (Optionally) Determine disk space needed/available and ask before
    #     continuing
    #   * Group files based on source (e.g. Azure container)
    #   * Start batched/parallel downloads

    if "remote_files" not in test_case_json:
        return

    for remote_file in test_case_json["remote_files"]:
        if "blob.core.windows.net" in remote_file:
            download_azure_remote_file(test_dir, remote_file)
        else:
            download_generic_remote_file(test_dir, remote_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote file downloader.")
    parser.add_argument(
        "--root-dir",
        default="",
        help="Root directory to search for files to download from (e.g. 'pytorch/models/resnet50')",
    )
    args = parser.parse_args()

    # Adjust logging levels.
    logging.basicConfig(level=logging.INFO)
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if log_name.startswith("azure"):
            logging.getLogger(log_name).setLevel(logging.WARNING)

    # TODO(scotttodd): build list of files _then_ download
    # TODO(scotttodd): report size needed for requested files and size available on disk

    for test_cases_path in (THIS_DIR / args.root_dir).rglob("*.json"):
        with open(test_cases_path) as f:
            test_cases_json = pyjson5.load(f)
            if test_cases_json.get("file_format", "") != "test_cases_v0":
                continue

            logger.info(f"Processing {test_cases_path.relative_to(THIS_DIR)}")

            test_dir = test_cases_path.parent
            for test_case_json in test_cases_json["test_cases"]:
                download_for_test_case(test_dir, test_case_json)
