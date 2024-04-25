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

def getTestsListFromFile(testlistfile):
    testlist = []
    if not os.path.exists(testlistfile):
        print(f"The file {testlistfile} does not exist")
        sys.exit(1)
    with open(testlistfile, "r") as tf:
        testlist += tf.read().splitlines()
    testlist = [item.strip().strip(os.sep) for item in testlist]
    return testlist


def ziponnxmodel(onnxfile, targetmodelzip):
    with zipfile.ZipFile(targetmodelzip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(onnxfile)


def cleanup_e2eshark_test(testList, e2eshark_test_dir):
    for model in testList:
        onnxmodel = e2eshark_test_dir + "/" + model + "/model.onnx"
        onnxmodelzip = e2eshark_test_dir + "/" + model + "/model.onnx.zip"
        if os.path.exists(onnxmodel):
            print(f"Removing {onnxmodel}")
            os.remove(onnxmodel)
        if os.path.exists(onnxmodelzip):
            print(f"Removing {onnxmodelzip}")
            os.remove(onnxmodelzip)


def download_azure_blob(account_url, container_name, blob_name, dest_file):
    if container_name == priv_container_name:
        if PRIVATE_CONN_STRING == "":
            print("Please set AZ_PRIVATE_CONNECTION environment variable with connection string for private azure storage account")
        with ContainerClient.from_connection_string(
                conn_str=PRIVATE_CONN_STRING,
                container_name=container_name,
            ) as container_client:
                download_stream = container_client.download_blob(
                    blob_name
                )
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


def download_and_setup_onnxmodels(cache_dir, testList):
    # Utility to download specified models (zip files) to cache dir
    # testList : expected to contain list of test names of the format `onnx/model/testName`
    # Download failure should not stop tests running entirely.
    # So downloads will be allowed to fail and corressponding
    # tests will fail with No model.onnx file found error

    # Azure Storage Creds for Public Onnx Models
    account_url = "https://onnxstorage.blob.core.windows.net"
    container_name = "onnxstorage"

    # Azure Storage Creds for Private Onnx Models - AZURE Login Required for access
    priv_account_url = "https://onnxprivatestorage.blob.core.windows.net"
    priv_container_name = "onnxprivatestorage"

    for model in testList:
        blob_dir =  "e2eshark/" + model
        blob_name = blob_dir + "/model.onnx.zip"
        dest_file = cache_dir + "/" + blob_name
        if os.path.exists(dest_file):
            # model already in cache dir, skip download.
            # TODO: skip model downloading based on some comparison / update flag
            continue
        if not os.path.exists(cache_dir):
            print(f"ERROR : cache_dir path: {cache_dir}, does not exist!")
            sys.exit(1)
        if not os.path.isdir(cache_dir + "/" + blob_dir):
            print(f"DIR not found creating new {blob_dir}")
            os.makedirs(cache_dir + "/" + blob_dir, exist_ok=True)

        # TODO: better organisation of models in tank and cache
        print(
            f"Begin download for {blob_name} to {dest_file}"
        )

        try_private = False
        try:
            download_azure_blob(account_url, container_name, blob_name, dest_file)
        except Exception as e:
            try_private = True
            print(f"Unable to download model from public for {model}.\nError - {type(e).__name__}")

        if try_private:
            print("Trying download from private storage")
            try:
                download_azure_blob(priv_account_url, priv_container_name, blob_name, dest_file)
            except Exception as e:
                print(f"Unable to download model for {model}.\nError - {type(e).__name__}")


def pre_test_onnx_models_azure_download(testsList, cache_path, script_dir):
    # This util helps setting up the e2eshark/onnx/models tests by ensuring
    # all the models-tests in the testsList have the required model.onnx file
    # testsList: expected to contain only onnx tests

    # for the testList download all the onnx/models in cache_path
    model_tests_list = [test for test in testsList if 'models' in test]
    download_and_setup_onnxmodels(cache_path, model_tests_list)

    # if the the model exists for the test in the test dir, do nothing.
    # if it doesn't exist in the test directory but exists in cache dir, simply unzip cached model
    for test_name in model_tests_list:
        model_file_path_test = script_dir + '/' + test_name + '/model.onnx'
        model_file_path_cache = cache_path + '/e2eshark/' + test_name + '/model.onnx.zip'
        if not os.path.exists(model_file_path_test):
            print(f"Unzipping - {model_file_path_cache}")
            # model_file_path_cache may not exist for models which were not correctly downloaded,
            # skip unzipping such model files, only extract existing models
            if os.path.exists(model_file_path_cache):
                # Unzip the model in the model test dir
                with ZipFile(model_file_path_cache, 'r') as zf:
                    # onnx/model/testname already present in the zip file structure
                    zf.extract(test_name + '/model.onnx', path=script_dir)


def setup_e2eshark_test(modelpy, testList, sourcedir, model_root_dir):
    for model in testList:
        onnxmodel = sourcedir + "/" + model + ".onnx"
        if not os.path.exists(onnxmodel):
            print(f"The file {onnxmodel} does not exist")
            return

        testdir = model_root_dir + "/" + model
        if not os.path.exists(testdir):
            try:
                os.mkdir(testdir)
            except OSError as errormsg:
                print(
                    "ERROR: Could not make run directory",
                    testdir,
                    " Error message: ",
                    errormsg,
                )
                return
        targetmodel = testdir + "/model.onnx"
        targetmodelzip = targetmodel + ".zip"
        print(
            f"Setting up {testdir}: onnx model: {targetmodel}, onnx model zip: {targetmodelzip}"
        )
        if not os.path.exists(targetmodel):
            shutil.copy(onnxmodel, targetmodel)
            print(f"Copied {onnxmodel} to {targetmodel}")
        if not os.path.exists(targetmodelzip):
            ziponnxmodel(targetmodel, targetmodelzip)
        if os.path.exists(modelpy):
            targetmodelpy = testdir + "/model.py"
            shutil.copy(modelpy, targetmodelpy)


def upload_test_to_azure_storage(
    testList, model_root_dir, e2eshark_test_dir, azure_storage_url
):
    for model in testList:
        sourcemodelzip = model_root_dir + "/" + model + "/model.onnx.zip"
        targetmodelzip = e2eshark_test_dir + "/" + model + "/model.onnx.zip"
        command = "az storage blob upload --overwrite --account-name onnxstorage --container-name onnxstorage"
        command += " --name " + targetmodelzip
        command += " --file " + sourcemodelzip + " --auth-mode key"
        url = azure_storage_url + "/" + targetmodelzip
        print(f"{url}")
        os.system(command)


if __name__ == "__main__":
    msg = "The script to setup, upload and download e2eshark onnx model tests to/from Azure Storage"
    parser = argparse.ArgumentParser(description=msg, epilog="")
    parser.add_argument(
        "testlistfile",
        help="A file with names of tests listed on separate lines",
    )
    parser.add_argument(
        "-s",
        "--sourcedir",
        help="The source directory with <modelname>.onnx for setting up test(s)",
    )
    parser.add_argument(
        "-m",
        "--modelpy",
        help="The source template model.py to use for test setup",
    )
    parser.add_argument(
        "-c",
        "--cachedir",
        help="The cache directory into which model from Azure Storage should be downloaded",
    )
    parser.add_argument(
        "--setup",
        help="Copy source <modelname>.onnx from argument of --sourcedir and setup a e2eshark onnx model test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--cleanup",
        help="Remove the model.onnx and model.onnx.zip. Call --cleanup after uploading the model.onnx.zip to azure storage",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--upload",
        help="Upload test(s) to Azure Storage",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--download",
        help="Download test(s) from Azure Storage into cache directory",
        action="store_true",
        default=False,
    )
    account_url = "https://onnxstorage.blob.core.windows.net"
    container_name = "onnxstorage"
    azure_storage_url = account_url + "/" + container_name
    model_root_dir = "onnx/models"
    e2eshark_test_dir = "e2eshark/" + model_root_dir

    args = parser.parse_args()
    testList = getTestsListFromFile(args.testlistfile)

    if args.cleanup:
        cleanup_e2eshark_test(testList, model_root_dir)

    if args.setup:
        if not args.sourcedir:
            print(
                f"The --sourcedir arg where arg is directory containing <modelname>.onnx is required for settting up the test."
            )
            sys.exit(1)
        if not args.modelpy:
            print(
                f"The --modelpy arg where arg is model.py template is required for settting up the test."
            )
            sys.exit(1)

        sourcedir = args.sourcedir
        sourcedir.strip().strip(os.sep)
        if not os.path.exists(sourcedir):
            print(f"The directory {sourcedir} does not exist")
            sys.exit(1)
        if not os.path.exists(args.modelpy):
            print(f"The template model.py {args.modelpy} does not exist")
            sys.exit(1)
        setup_e2eshark_test(args.modelpy, testList, sourcedir, model_root_dir)

    if args.upload:
        upload_test_to_azure_storage(
            testList, model_root_dir, e2eshark_test_dir, azure_storage_url
        )

    if args.download:
        if not args.cachedir:
            print(
                f"The --cachedir for doalonding the models into is required for downloading models."
            )
            sys.exit(1)
        cachedir = args.cachedir
        cachedir.strip().strip(os.sep)
        if not os.path.exists(cachedir):
            print(f"The directory {cachedir} does not exist")
            sys.exit(1)

        download_and_setup_onnxmodels(cachedir, testList)
