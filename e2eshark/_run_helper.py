import os
import glob
import pickle
from azure.storage.blob import BlobServiceClient
import zipfile
import struct
import torch
import io


def concatenateFiles(inpfile1, inpfile2, outfile):
    f1 = open(inpfile1, "r")
    f2 = open(inpfile2, "r")
    ofile = open(outfile, "w")
    ofile.write(f1.read() + f2.read())
    f1.close()
    f2.close()
    ofile.close()


def getTestsList(frameworkname, test_types):
    testsList = []
    for test_type in test_types:
        globpattern = frameworkname + "/" + test_type + "/*"
        testsList += glob.glob(globpattern)
    return testsList


def getTestKind(testName):
    # extract second last name in test name and if that is "models" and
    # it may have zipped onnx files, unzip them if not already done so
    second_last_name_inpath = os.path.split(os.path.split(testName)[0])[1]
    return second_last_name_inpath


def changeToTestDir(run_dir):
    try:
        # If directory does not exist, make it
        os.makedirs(run_dir, exist_ok=True)
        os.chdir(run_dir)
        return 0
    except OSError as errormsg:
        print(
            "Could not make or change to test run directory",
            run_dir,
            " Error message: ",
            errormsg,
        )
        return 1


def loadE2eSharkCheckDictionary():
    e2esharkDict = None
    pklfilename = "E2ESHARK_CHECK.pkl"
    if os.path.exists(pklfilename):
        with open(pklfilename, "rb") as pkf:
            e2esharkDict = pickle.load(pkf)
    return e2esharkDict


def uploadToBlobStorage(file_path, file_name, testName, uploadDict):
    connection_string = os.getenv("AZURE_CONNECTION_STRING")
    container_name = "e2esharkuserartifacts"

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=file_name
    )
    blob = blob_client.from_connection_string(
        conn_str=connection_string,
        container_name=container_name,
        blob_name=blob_client.blob_name,
    )
    # we check to see if we already uploaded the blob (don't want to duplicate)
    if blob.exists():
        print(f"model artifacts have already been uploaded for this blob name")
        return
    # upload to azure storage container e2esharkuserartifacts
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data)
    dict_value = uploadDict.get(testName)
    if not dict_value:
        uploadDict[testName] = [blob_client.url]
    else:
        dict_value.append(blob_client.url)
        uploadDict[testName] = dict_value
    return


def unzipONNXFile(testName, abs_directory, unzipped_file_name):
    # Look for any unzipped file and if there is not already an unzipped file
    # then first time unzip it.
    abs_unzip_file_name = abs_directory + "/" + unzipped_file_name
    abs_zip_file_name = abs_unzip_file_name + ".zip"
    # this test dir does not have a zipped test file, so nothing to do
    if not os.path.exists(abs_zip_file_name):
        return 0
    # if not already unzipped, then
    if not os.path.exists(abs_unzip_file_name):
        if not os.access(abs_directory, os.W_OK):
            print(
                "The directory",
                abs_directory,
                "is not writeable. Could not unzip",
                abs_zip_file_name,
            )
            return 1
        with zipfile.ZipFile(abs_zip_file_name, "r") as zip_ref:
            zip_ref.extractall(abs_directory)

    return 0


def loadTorchSave(filename):
    with open(filename, "rb") as f:
        bindata = f.read()
    buf = io.BytesIO(bindata)
    loaded = torch.load(buf)
    return loaded


def getShapeString(torchtensor):
    inputshape = list(torchtensor.shape)
    inputshapestring = "x".join([str(item) for item in inputshape])
    dtype = torchtensor.dtype
    if dtype == torch.int64:
        inputshapestring += "xi64"
    elif dtype == torch.float32 or dtype == torch.float:
        inputshapestring += "xf32"
    elif dtype == torch.bfloat16 or dtype == torch.float16 or dtype == torch.int16:
        inputshapestring += "xbf16"
    elif dtype == torch.int8:
        inputshapestring += "xi8"
    elif dtype == torch.bool:
        inputshapestring += "xi1"
    else:
        print("In getShapeString, found an unsupported data type", dtype)
    return inputshapestring


def unpackBytearray(barray, num_elem, dtype):
    num_array = None
    if dtype == torch.int64:
        num_array = struct.unpack("q" * num_elem, barray)
    elif dtype == torch.float32 or dtype == torch.float:
        num_array = struct.unpack("f" * num_elem, barray)
    elif dtype == torch.bfloat16:
        num_array = struct.unpack("h" * num_elem, barray)
        temptensor = torch.tensor(num_array, dtype=torch.int16)
        rettensor = temptensor.view(dtype=torch.bfloat16)
        return rettensor
    elif dtype == torch.float16:
        num_array = struct.unpack("h" * num_elem, barray)
        temptensor = torch.tensor(num_array, dtype=torch.int16)
        rettensor = temptensor.view(dtype=torch.float16)
        return rettensor
    elif dtype == torch.int32:
        num_array = struct.unpack("l" * num_elem, barray)
    elif dtype == torch.int16:
        num_array = struct.unpack("h" * num_elem, barray)
    elif dtype == torch.int8:
        num_array = struct.unpack("b" * num_elem, barray)
    elif dtype == torch.uint8:
        num_array = struct.unpack("B" * num_elem, barray)
    elif dtype == torch.bool:
        num_array = struct.unpack("?" * num_elem, barray)
    else:
        print("In unpackBytearray, found an unsupported data type", dtype)
    rettensor = torch.tensor(num_array, dtype=dtype)
    return rettensor


def loadRawBinaryAsTorchSensor(binaryfile, shape, dtype):
    # Read the whole files as bytes
    with open(binaryfile, "rb") as f:
        binarydata = f.read()
    # Number of elements in tensor
    num_elem = torch.prod(torch.tensor(list(shape)))
    # Total bytes
    tensor_num_bytes = (num_elem * dtype.itemsize).item()
    barray = bytearray(binarydata[0:tensor_num_bytes])
    # for byte in barray:
    #     print(f"{byte:02X}", end="")
    rettensor = unpackBytearray(barray, num_elem, dtype)
    reshapedtensor = rettensor.reshape(list(shape))
    f.close()
    return reshapedtensor


def packTensor(modelinput):
    mylist = modelinput.flatten().tolist()
    dtype = modelinput.dtype
    if dtype == torch.int64:
        bytearr = struct.pack("%sq" % len(mylist), *mylist)
    elif dtype == torch.float32 or dtype == torch.float:
        bytearr = struct.pack("%sf" % len(mylist), *mylist)
    elif dtype == torch.bfloat16 or dtype == torch.float16:
        reinterprted = modelinput.view(dtype=torch.int16)
        mylist = reinterprted.flatten().tolist()
        bytearr = struct.pack("%sh" % len(mylist), *mylist)
    elif dtype == torch.int16:
        bytearr = struct.pack("%sh" % len(mylist), *mylist)
    elif dtype == torch.int8:
        bytearr = struct.pack("%sb" % len(mylist), *mylist)
    elif dtype == torch.bool:
        bytearr = struct.pack("%s?" % len(mylist), *mylist)
    else:
        print("In packTensor, found an unsupported data type", dtype)
    return bytearr


def writeInferenceInputBinFile(modelinput, modelinputbinfilename):
    with open(modelinputbinfilename, "wb") as f:
        bytearr = packTensor(modelinput)
        f.write(bytearr)


def convertNumToString(rows):
    strrows = []
    for row in rows:
        strrows += [[str(i) for i in row]]
    return strrows


def getTestsListFromFile(testlistfile):
    testlist = []
    if not os.path.exists(testlistfile):
        print(f"The file {testlistfile} does not exist")
    with open(testlistfile, "r") as tf:
        testlist += tf.read().splitlines()
    testlist = [item.strip().strip(os.sep) for item in testlist]
    return testlist
