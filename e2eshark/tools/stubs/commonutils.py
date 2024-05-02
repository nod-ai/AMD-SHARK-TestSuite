# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from PIL import Image
import torchvision.transforms as transforms
import requests

# These are pickle-saved and used by tools/stubs python and run.pl.
# If adding new fields, make sure the field has default value and have updated
# tools/stubs and run.pl to handle the new fields
E2ESHARK_CHECK_DEF = {
    # this is input applied to the model
    "input": None,
    # this is output gotten from the model
    "output": None,
    # output for validation
    "output_for_validation": None,
    # Controls how to import a graph from PyTorch into MLIR, options are: compile or fximport
    "torchmlirimport": "fximport",
    # By default, the input.to(dtype) is called, set it to False to not do so
    "inputtodtype": True,
    # Apply listed function with its arguments and return value selection repitively
    # to post-process an output. Each entry in list should be a tuple with four
    # entries: (function, [args other than input], isReturnTuple, indexOfTupleForTupleReturn)
    # Exmaple: "postprocess": [(torch.nn.functional.softmax, [0], False, 0), (torch.topk, [5], True, 1)]
    # which will be called as output = topk(torch.nn.functional.softmax(output, 0), 5)[1]
    "postprocess": None,
    # Store the post-processed output here
    "postprocessed_output": None,
}


# fix model op to be a list of tensors
# recursively flattens tuple of tuples to a list of single tuples in order
# example - (1, ((2,3),(4,5),(6,7))) -> [1,2,3,4,5,6,7]
def getOutputTensorList(test_out):
    def flatten_tuples(tup):
        if isinstance(tup, tuple):
            res = []
            for t in tup:
                res.extend(flatten_tuples(t))
            return res
        return [tup]

    test_op_list = flatten_tuples(test_out)
    return test_op_list


# functionPipeLine = (function, [args other than input], isReturnTuple, indexOfTupleForTupleReturn)
def applyPostProcessPipeline(item, functionPipeLine):
    if torch.any(torch.isnan(item)):
        print("NUMERICS ERROR: The tensor contains NaN values.")
    # Run post processing pipeline
    for func, argextra, isRetTuple, tupleIndex in functionPipeLine:
        if len(argextra) > 0:
            item = func(item, *argextra)
        else:
            item = func(item)
        if isRetTuple:
            item = item[tupleIndex]
    return item


# Apply post processing functions on output
def postProcess(e2esharkDict):
    if e2esharkDict.get("output_for_validation") is not None:
        test_output = e2esharkDict["output_for_validation"]
    else:
        test_output = e2esharkDict["output"]
    functionPipeLine = e2esharkDict["postprocess"]
    print(f"{functionPipeLine}")
    postprocess_output = []
    # Call chain of post processing -- run.pl will do same on backend inference output
    if e2esharkDict.get("postprocess"):
        for item in test_output:
            inp = item.clone()
            processed_item = applyPostProcessPipeline(inp, functionPipeLine)
            postprocess_output += [processed_item]
    else:
        postprocess_output = test_output
    return postprocess_output

# used for image inputs for onnx vision models
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def setup_test_image():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw)

    resize = transforms.Resize([224, 224])
    img = resize(img)
    
    # Define a transform to convert 
    # the image to torch tensor 
    img_ycbcr = img.convert('YCbCr')
    
    # Convert the image to Torch tensor 
    to_tensor = transforms.ToTensor()
    img_ycbcr = to_tensor(img_ycbcr)
    img_ycbcr.unsqueeze_(0)
    return img_ycbcr
