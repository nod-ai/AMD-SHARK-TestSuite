# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import json
import numpy
import struct
import torch
import glob
import onnx
from typing import Tuple, Optional, Dict, List, Any, Union
from pathlib import Path
import os

TYPE_STRING_FROM_TORCH_DTYPE = {
    torch.int64: "i64",
    torch.uint64: "ui64",
    torch.int32: "i32",
    torch.uint32: "ui32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "ui8",
    torch.bool: "i1",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}

PACK_CHAR_FROM_TORCH_DTYPE = {
    torch.int64: "q",
    torch.uint64: "Q",
    torch.int32: "i",
    torch.uint32: "I",
    torch.int16: "h",
    torch.uint16: "H",
    torch.int8: "b",
    torch.uint8: "B",
    torch.bool: "?",
    torch.float64: "d",
    torch.float32: "f",
    torch.float16: "h",
    torch.bfloat16: "h",
}


def get_shape_string(torch_tensor):
    input_shape = list(torch_tensor.shape)
    input_shape_string = "x".join([str(item) for item in input_shape])
    dtype = torch_tensor.dtype
    if dtype not in TYPE_STRING_FROM_TORCH_DTYPE.keys():
        raise NotImplementedError(
            "In get_shape_string, found an unsupported data type: " + str(dtype)
        )
    dtype_str = TYPE_STRING_FROM_TORCH_DTYPE[torch_tensor.dtype]
    return input_shape_string + "x" + dtype_str


def unpack_bytearray(barray, num_elem, dtype):
    if dtype not in PACK_CHAR_FROM_TORCH_DTYPE.keys():
        raise NotImplementedError(
            f"In unpack_bytearray, found an unsupported data type {dtype}"
        )
    num_array = struct.unpack(PACK_CHAR_FROM_TORCH_DTYPE[dtype] * num_elem, barray)
    # special handling for f16, bf16
    if dtype == torch.bfloat16 or dtype == torch.float16:
        temptensor = torch.tensor(num_array, dtype=torch.int16)
        return temptensor.view(dtype=dtype)
    return torch.tensor(num_array, dtype=dtype)


def load_raw_binary_as_torch_tensor(binaryfile, shape, dtype):
    """given a shape and torch dtype, this will load a torch tensor from the specified binaryfile"""
    # Read the whole files as bytes
    with open(binaryfile, "rb") as f:
        binarydata = f.read()
    # Number of elements in tensor
    num_elem = (
        torch.prod(torch.tensor(list(shape))) if len(shape) > 0 else torch.tensor([1])
    )
    # Total bytes
    tensor_num_bytes = (num_elem * dtype.itemsize).item()
    barray = bytearray(binarydata[0:tensor_num_bytes])
    # for byte in barray:
    #     print(f"{byte:02X}", end="")
    rettensor = unpack_bytearray(barray, num_elem, dtype)
    reshapedtensor = rettensor.reshape(list(shape))
    return reshapedtensor


def pack_tensor(modelinput):
    """stores a torch.Tensor into a binary file"""
    dtype = modelinput.dtype
    if dtype not in PACK_CHAR_FROM_TORCH_DTYPE.keys():
        raise NotImplementedError(
            f"In pack_tensor, found an unsupported data type {dtype}"
        )
    if dtype == torch.float16 or dtype == torch.bfloat16:
        modelinput = modelinput.view(dtype=torch.int16)
    mylist = modelinput.flatten().tolist()
    bytearr = struct.pack(
        f"%s{PACK_CHAR_FROM_TORCH_DTYPE[dtype]}" % len(mylist), *mylist
    )
    return bytearr


def write_inference_input_bin_file(modelinput, modelinputbinfilename):
    """Stores a modelinput to a specified binary file."""
    with open(modelinputbinfilename, "wb") as f:
        bytearr = pack_tensor(modelinput)
        f.write(bytearr)


def load_test_txt_file(filepath: Union[str, Path]) -> List[str]:
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
        self.type = None
        if len(data) > 0:
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
        """loads bin files. dir_path should end in a forward slash and should contain files of the type {name}.0.bin, {name}.1.bin, etc."""
        tensor_list = []
        pb_input_files = glob.glob("input_?.pb", root_dir=dir_path)

        # This condition should be only executed for inputs, so add a guard to ensure this.
        if len(pb_input_files) > 0 and name == "input":
            onnx_tensor_list = [onnx.load_tensor(os.path.join(dir_path, tensor)) for tensor in pb_input_files]
            for tensor in onnx_tensor_list:
                tensor_list.append(onnx.numpy_helper.to_array(tensor, base_dir=dir_path))
        else:
            assert len(shapes) == len(
                torch_dtypes
            ), "must provide same number of shapes and dtypes"
            for i in range(len(shapes)):
                shape = shapes[i]
                dtype = torch_dtypes[i]
                t = load_raw_binary_as_torch_tensor(
                    os.path.join(dir_path, name + "." + str(i) + ".bin"), shape, dtype
                )
                tensor_list.append(t)
        return TestTensors(tuple(tensor_list))
