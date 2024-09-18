# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import json
import numpy
import struct
import torch
from typing import Tuple, Optional, Dict, List, Any, Union
from pathlib import Path
import os

def get_shape_string(torch_tensor):
    input_shape = list(torch_tensor.shape)
    input_shape_string = "x".join([str(item) for item in input_shape])
    dtype = torch_tensor.dtype
    if dtype == torch.int64:
        input_shape_string += "xi64"
    elif dtype == torch.float32 or dtype == torch.float:
        input_shape_string += "xf32"
    elif dtype == torch.bfloat16 or dtype == torch.float16 or dtype == torch.int16:
        input_shape_string += "xbf16"
    elif dtype == torch.int8:
        input_shape_string += "xi8"
    elif dtype == torch.bool:
        input_shape_string += "xi1"
    else:
        print("In get_shape_string, found an unsupported data type", dtype)
    return input_shape_string


def unpack_bytearray(barray, num_elem, dtype):
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
        raise NotImplementedError(f"In unpack_bytearray, found an unsupported data type {dtype}")
    rettensor = torch.tensor(num_array, dtype=dtype)
    return rettensor


def load_raw_binary_as_torch_tensor(binaryfile, shape, dtype):
    '''given a shape and torch dtype, this will load a torch tensor from the specified binaryfile'''
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
    rettensor = unpack_bytearray(barray, num_elem, dtype)
    reshapedtensor = rettensor.reshape(list(shape))
    f.close()
    return reshapedtensor


def pack_tensor(modelinput):
    """stores a torch.Tensor into a binary file"""
    mylist = modelinput.flatten().tolist()
    dtype = modelinput.dtype
    if dtype == torch.int64:
        bytearr = struct.pack("%sq" % len(mylist), *mylist)
    elif dtype == torch.uint64:
        bytearr = struct.pack("%sQ" % len(mylist), *mylist)
    elif dtype == torch.int32:
        bytearr = struct.pack("%sl" % len(mylist), *mylist)
    elif dtype == torch.uint32:
        bytearr = struct.pack("%sL" % len(mylist), *mylist)
    elif dtype == torch.float64:
        bytearr = struct.pack("%sd" % len(mylist), *mylist)
    elif dtype == torch.float32 or dtype == torch.float:
        bytearr = struct.pack("%sf" % len(mylist), *mylist)
    elif dtype == torch.bfloat16 or dtype == torch.float16:
        reinterprted = modelinput.view(dtype=torch.int16)
        mylist = reinterprted.flatten().tolist()
        bytearr = struct.pack("%sh" % len(mylist), *mylist)
    elif dtype == torch.int16:
        bytearr = struct.pack("%sh" % len(mylist), *mylist)
    elif dtype == torch.uint16:
        bytearr = struct.pack("%sH" % len(mylist), *mylist)
    elif dtype == torch.int8:
        bytearr = struct.pack("%sb" % len(mylist), *mylist)
    elif dtype == torch.uint8:
        bytearr = struct.pack("%sB" % len(mylist), *mylist)
    elif dtype == torch.bool:
        bytearr = struct.pack("%s?" % len(mylist), *mylist)
    else:
        raise NotImplementedError(f"In pack_tensor, found an unsupported data type {dtype}")
    return bytearr


def write_inference_input_bin_file(modelinput, modelinputbinfilename):
    """Stores a modelinput to a specified binary file."""
    with open(modelinputbinfilename, "wb") as f:
        bytearr = pack_tensor(modelinput)
        f.write(bytearr)
        f.close()

def load_test_txt_file(filepath : Union[str, Path]) -> List[str]:
    with open(filepath, "r") as file:
        contents = file.read().split()
    return contents

def load_json_dict(filepath: Union[str, Path]) -> Dict[str, Any]:
    with open(filepath) as contents:
        loaded_dict = json.load(contents)
    return loaded_dict

class TestTensors:
    """storage class for tuples of tensor-like objects"""

    __slots__ = [
        "data",
        "type",
    ]

    def __init__(self, data: Tuple):
        self.data = data
        self.type = type(self.data[0])
        if not all([type(d) == self.type for d in data]):
            self.type == None

    def __repr__(self):
        return f"TestTensors({self.type}): {self.data.__repr__()}"

    def to_numpy(self) -> "TestTensors":
        """returns a copy of self as a numpy.ndarray type"""
        if self.type == torch.Tensor:
            new_data = tuple([d.numpy() for d in self.data])
            return TestTensors(new_data)
        elif self.type == numpy.ndarray:
            return TestTensors(self.data)
        else:
            raise TypeError(
                f"Unhandled TestTensors conversion from {self.type} to numpy.ndarray"
            )

    def to_torch(self) -> "TestTensors":
        """returns a copy of self as a torch.Tensor type"""
        if self.type == numpy.ndarray:
            new_data = tuple([torch.from_numpy(d) for d in self.data])
            return TestTensors(new_data)
        elif self.type == torch.Tensor:
            return TestTensors(self.data)
        else:
            raise TypeError(
                f"Unhandled TestTensors conversion from {self.type} to torch.Tensor"
            )

    def to_dtype(self, dtype, *, index: Optional[int] = None) -> "TestTensors":
        """returns a copy of self with a converted dtype (at a particular index, if specified)"""
        if self.type == numpy.ndarray:
            if index:
                try:
                    new_data = self.data
                    new_data[index] = new_data[index].astype(dtype)
                except Exception as e:
                    print("to_dtype failed due to excepton {e}.")
            else:
                new_data = tuple([d.astype(dtype) for d in self.data])
        if self.type == torch.Tensor:
            if index:
                try:
                    new_data = self.data
                    new_data[index] = new_data[index].to(dtype=dtype)
                except Exception as e:
                    print("to_dtype failed due to excepton {e}.")
            else:
                new_data = tuple([d.to(dtype=dtype) for d in self.data])
        return TestTensors(new_data)

    def save_to(self, path: str):
        """path should be of the form /path/to/log/folder/unformattedname"""
        if self.type == torch.Tensor:
            data = self.data
        else:
            data = self.to_torch().data
        for i in range(len(data)):
            write_inference_input_bin_file(data[i], path + f".{i}.bin")
    
    @staticmethod
    def load_from(shapes, torch_dtypes, dir_path: str, name: str = "input"):
        '''loads bin files. dir_path should end in a forward slash and should contain files of the type {name}.0.bin, {name}.1.bin, etc.'''
        tensor_list = []
        assert len(shapes) == len(torch_dtypes), "must provide same number of shapes and dtypes"
        for i in range(len(shapes)):
            shape = shapes[i]
            dtype = torch_dtypes[i]
            t = load_raw_binary_as_torch_tensor(os.path.join(dir_path, name + "." + str(i) + ".bin"), shape, dtype)
            tensor_list.append(t)
        return TestTensors(tuple(tensor_list))



