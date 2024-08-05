# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import torch
from e2e_testing.framework import OnnxModelInfo, SiblingModel
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors
from .azure_models import AzureDownloadableModel

class ImageClassificationModel(SiblingModel):
    def apply_postprocessing(self, output: TestTensors):
        processed_outputs = []
        for d in output.to_torch().data:
            processed_outputs.append(
                torch.sort(torch.topk(torch.nn.functional.softmax(d, 1), 2)[1])[0]
            )
        return TestTensors(processed_outputs)

# this will run resnet50_vaiq_int8 without post-processing
register_test(AzureDownloadableModel, "resnet50_vaiq_int8")

# this will run the same model, but with post-processing
constructor = lambda *args, **kwargs : ImageClassificationModel(AzureDownloadableModel, "resnet50_vaiq_int8", *args, **kwargs)
register_test(constructor, "resnet50_vaiq_int8_pp")