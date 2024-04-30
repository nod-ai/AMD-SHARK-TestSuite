# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import struct
import numpy as np

# map numpy dtype -> (iree dtype, struct.pack format str)
dtype_map = {
    np.dtype("int64"): ("si64", "q"),
    np.dtype("uint64"): ("ui64", "Q"),
    np.dtype("int32"): ("si32", "i"),
    np.dtype("uint32"): ("ui32", "I"),
    np.dtype("int16"): ("si16", "h"),
    np.dtype("uint16"): ("ui16", "H"),
    np.dtype("int8"): ("si8", "b"),
    np.dtype("uint8"): ("ui8", "B"),
    np.dtype("float64"): ("f64", "d"),
    np.dtype("float32"): ("f32", "f"),
    np.dtype("float16"): ("f16", "e"),
    np.dtype("bool"): ("i1", "?"),
}


def get_shape_string(torchtensor):
    inputshape = list(torchtensor.shape)
    inputshapestring = "x".join([str(item) for item in inputshape])
    dtype = torchtensor.dtype
    if dtype in dtype_map:
        inputshapestring += f"x{dtype_map[dtype][0]}"
    else:
        print(
            f"WARNING: unsupported data type in get_shape_string() : '{dtype}'"
        )
    return inputshapestring


def pack_np_arr(arr):
    mylist = arr.flatten().tolist()
    dtype = arr.dtype
    # bf16 is converted to int16 when we `convert_io_proto` in `numpy_helper.to_array`
    bytearr = b""
    if dtype in dtype_map:
        bytearr = struct.pack(f"{len(mylist)}{dtype_map[dtype][1]}", *mylist)
    else:
        print(f"WARNING: unsupported data type in pack_np_arr() : '{dtype}'")
    return bytearr


def write_io_bin(ndarr, filename):
    with open(filename, "wb") as f:
        bytearr = pack_np_arr(ndarr)
        f.write(bytearr)
