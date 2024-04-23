# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from azure.storage.blob import BlobClient, BlobProperties
from pathlib import Path
from typing import Optional
import argparse
import hashlib
import logging
import mmap
import os
import pyjson5
import re

THIS_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


def human_readable_size(size, decimal_places=2):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def setup_cache_symlink_if_needed(
    cache_dir: Optional[Path], local_dir: Path, file_name: str
):
    """Creates a symlink from local_dir/file_name to cache_dir/file_name."""
    if not cache_dir:
        return

    local_file_path = local_dir / file_name
    cache_file_path = cache_dir / file_name
    if local_file_path.is_symlink():
        if os.path.samefile(str(local_file_path), str(cache_file_path)):
            # Symlink matches, no need to recreate.
            return
        os.remove(local_file_path)
    elif local_file_path.exists():
        logger.warning(
            f"  Local file '{local_file_path}' exists but cache_dir is set. Deleting and "
            "replacing with a symlink"
        )
        os.remove(local_file_path)
    os.symlink(cache_file_path, local_file_path)
    logger.info(f"  Created symlink for '{local_file_path}' to '{cache_file_path}'")


def get_azure_md5(remote_file: str, azure_blob_properties: BlobProperties):
    """Gets the content_md5 hash for a blob on Azure, if available."""
    content_settings = azure_blob_properties.get("content_settings")
    if not content_settings:
        return None
    azure_md5 = content_settings.get("content_md5")
    if not azure_md5:
        logger.warning(
            f"  Remote file '{remote_file}' on Azure is missing the "
            "'content_md5' property, can't check if local matches remote"
        )
    return azure_md5


def get_local_md5(local_file_path: Path):
    """Gets the content_md5 hash for a lolca file, if it exists."""
    if not local_file_path.exists() or local_file_path.stat().st_size == 0:
        return None

    with open(local_file_path) as file, mmap.mmap(
        file.fileno(), 0, access=mmap.ACCESS_READ
    ) as file:
        return hashlib.md5(file).digest()


def download_azure_remote_file(
    remote_file: str, test_dir: Path, cache_dir: Optional[Path]
):
    """
    Downloads a file from Azure into test_dir.

    If cache_dir is set, downloads there instead, creating a symlink from
    test_dir/file_name to cache_dir/file_name.
    """
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
        azure_md5 = get_azure_md5(remote_file, blob_properties)

        if cache_dir:
            local_file_path = cache_dir / remote_file_name
        else:
            local_file_path = test_dir / remote_file_name
        local_md5 = get_local_md5(local_file_path)

        if azure_md5 and azure_md5 == local_md5:
            logger.info(
                f"  Skipping '{remote_file_name}' download ({blob_size_str}) "
                "- local MD5 hash matches"
            )
            setup_cache_symlink_if_needed(cache_dir, test_dir, remote_file_name)
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

        with open(local_file_path, mode="wb") as local_blob:
            download_stream = blob_client.download_blob(max_concurrency=4)
            local_blob.write(download_stream.readall())
        setup_cache_symlink_if_needed(cache_dir, test_dir, remote_file_name)


def download_generic_remote_file(
    remote_file: str, test_dir: Path, cache_dir: Optional[Path]
):
    """
    Downloads a file from a generic URL into test_dir.

    If cache_dir is set, downloads there instead, creating a symlink from
    test_dir/file_name to cache_dir/file_name.
    """

    # TODO(scotttodd): use https://pypi.org/project/requests/
    raise NotImplementedError("generic remote file downloads not implemented yet")


def download_files_for_test_case(
    test_case_json: dict, test_dir: Path, cache_dir: Optional[Path]
):
    if "remote_files" not in test_case_json:
        return

    # This is naive (greedy, serial) for now. We could batch downloads that
    # share a source:
    #   * Iterate over all files (across all included paths), building a list
    #     of files to download (checking hashes / local references before
    #     adding to the list)
    #   * (Optionally) Determine disk space needed/available and ask before
    #     continuing
    #   * Group files based on source (e.g. Azure container)
    #   * Start batched/parallel downloads

    for remote_file in test_case_json["remote_files"]:
        if "blob.core.windows.net" in remote_file:
            download_azure_remote_file(remote_file, test_dir, cache_dir)
        else:
            download_generic_remote_file(remote_file, test_dir, cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote file downloader.")
    parser.add_argument(
        "--root-dir",
        default="",
        help="Root directory to search for files to download from (e.g. 'pytorch/models/resnet50')",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("IREE_TEST_FILES", default=""),
        help="Local cache directory to download into. If set, symlinks will be created pointing to "
        "this location",
    )
    args = parser.parse_args()

    # Adjust logging levels.
    logging.basicConfig(level=logging.INFO)
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if log_name.startswith("azure"):
            logging.getLogger(log_name).setLevel(logging.WARNING)

    # Resolve cache location.
    if args.cache_dir:
        args.cache_dir = Path(os.path.expanduser(args.cache_dir)).resolve()

    # TODO(scotttodd): build list of files _then_ download
    # TODO(scotttodd): report size needed for requested files and size available on disk

    for test_cases_path in (THIS_DIR / args.root_dir).rglob("*.json"):
        with open(test_cases_path) as f:
            test_cases_json = pyjson5.load(f)
            if test_cases_json.get("file_format", "") != "test_cases_v0":
                continue

            logger.info(f"Processing {test_cases_path.relative_to(THIS_DIR)}")

            test_dir = test_cases_path.parent
            relative_dir = test_dir.relative_to(THIS_DIR)

            # Expand directory structure in the cache matching the test tree.
            if args.cache_dir:
                cache_dir_for_test = args.cache_dir / "iree_tests" / relative_dir
                if not os.path.isdir(cache_dir_for_test):
                    os.makedirs(cache_dir_for_test)
            else:
                cache_dir_for_test = None

            for test_case_json in test_cases_json["test_cases"]:
                download_files_for_test_case(
                    test_case_json=test_case_json,
                    test_dir=test_dir,
                    cache_dir=cache_dir_for_test,
                )
