# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os, sys, argparse, shutil, zipfile
from azure.storage.blob import ContainerClient
from azure.core.exceptions import ResourceNotFoundError
from pathlib import Path
from zipfile import ZipFile

PRIVATE_CONN_STRING = os.environ.get("AZ_PRIVATE_CONNECTION", default="")
priv_container_name = "onnxprivatestorage"


def pre_test_onnx_model_azure_download(name, cache_dir, model_path):
    # This util helps setting up the e2eamdshark/onnx/models tests by ensuring
    # all the models-tests in the testsList have the required model.onnx file
    # testsList: expected to contain only onnx tests

    # if cache directory doesn't exist, then make it
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    # for the testList download all the onnx/models in cache_path
    download_and_setup_onnxmodel(cache_dir, name)

    model_dir = str(Path(model_path).parent)
    # if the the model exists for the test in the test dir, do nothing.
    # if it doesn't exist in the test directory but exists in cache dir, simply unzip cached model
    dest_file = os.path.join(cache_dir, "model.onnx.zip")
    print(f"Unzipping - {dest_file}...","\t")
    # model_file_path_cache may not exist for models which were not correctly downloaded,
    # skip unzipping such model files, only extract existing models
    if os.path.exists(dest_file):
        # Unzip the model in the model test dir
        with ZipFile(dest_file, "r") as zf:
            # onnx/model/testname already present in the zip file structure
            zf.extractall(model_dir)
            print(f'Unzipping succeded. Look for extracted contents in {model_dir}')
    else:
        print(f'Failed: path {dest_file} does not exist!')


def download_and_setup_onnxmodel(cache_dir, name):
    # Utility to download specified models (zip files) to cache dir
    # Download failure should not stop tests running entirely.
    # So downloads will be allowed to fail and corressponding
    # tests will fail with No model.onnx file found error

    # Azure Storage Creds for Public Onnx Models
    account_url = "https://onnxstorage.blob.core.windows.net"
    container_name = "onnxstorage"

    # Azure Storage Creds for Private Onnx Models - AZURE Login Required for access
    priv_account_url = "https://onnxprivatestorage.blob.core.windows.net"
    priv_container_name = "onnxprivatestorage"

    blob_dir = os.path.join("e2eamdshark/onnx/models/", name)
    blob_name = os.path.join(blob_dir, "model.onnx.zip")
    dest_file = os.path.join(cache_dir, "model.onnx.zip")
    if os.path.exists(dest_file):
        # model already in cache dir, skip download.
        # TODO: skip model downloading based on some comparison / update flag
        return
    # TODO: better organisation of models in tank and cache
    print(f"Begin download for {blob_name} to {dest_file}")

    try_private = False
    try:
        download_azure_blob(account_url, container_name, blob_name, dest_file)
    except Exception as e:
        try_private = True
        print(
            f"Unable to download model from public for {name}.\nError - {type(e).__name__}"
        )

    if try_private:
        print("Trying download from private storage")
        try:
            download_azure_blob(
                priv_account_url, priv_container_name, blob_name, dest_file
            )
        except Exception as e:
            print(f"Unable to download model for {name}.\nError - {type(e).__name__}")


def download_azure_blob(account_url, container_name, blob_name, dest_file):
    if container_name == priv_container_name:
        if PRIVATE_CONN_STRING == "":
            print(
                "Please set AZ_PRIVATE_CONNECTION environment variable with connection string for private azure storage account"
            )
        with ContainerClient.from_connection_string(
            conn_str=PRIVATE_CONN_STRING,
            container_name=container_name,
        ) as container_client:
            download_stream = container_client.download_blob(blob_name)
            with open(dest_file, mode="wb") as local_blob:
                local_blob.write(download_stream.readall())
    else:
        with ContainerClient(
            account_url,
            container_name,
            max_chunk_get_size=1024 * 1024 * 32,  # 32 MiB
            max_single_get_size=1024 * 1024 * 32,  # 32 MiB
        ) as container_client:
            download_stream = container_client.download_blob(
                blob_name, max_concurrency=4
            )
            with open(dest_file, mode="wb") as local_blob:
                local_blob.write(download_stream.readall())
