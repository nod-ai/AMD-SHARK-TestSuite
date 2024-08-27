# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import torch
import onnxruntime
from ..helper_classes import AzureDownloadableModel, SiblingModel, get_sibling_constructor
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors

class NoOptimizations(SiblingModel):
    def update_sess_options(self):
        self.sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

class ImageClassificationModel(SiblingModel):
    def apply_postprocessing(self, output: TestTensors):
        processed_outputs = []
        for d in output.to_torch().data:
            processed_outputs.append(
                torch.sort(torch.topk(torch.nn.functional.softmax(d, 1), 2)[1])[0]
            )
        return TestTensors(processed_outputs)

# this will run resnet50_vaiq_int8 without post-processing
# register_test(AzureDownloadableModel, "resnet50_vaiq_int8")
# register_test(AzureDownloadableModel, "ResNet50_vaiq")
# register_test(AzureDownloadableModel, "ResNet152")
# register_test(AzureDownloadableModel, "ResNet152_vaiq")
# register_test(AzureDownloadableModel, "ResNet152_vaiq_int8")

# this will run the same model, but with post-processing
constructor0 = get_sibling_constructor(ImageClassificationModel, AzureDownloadableModel, "resnet50_vaiq_int8")
constructor1 = get_sibling_constructor(ImageClassificationModel, AzureDownloadableModel, "ResNet50_vaiq")
register_test(constructor0, "resnet50_vaiq_int8_pp")
register_test(constructor1, "ResNet50_vaiq_pp")

# this will run the same model, but the gold inference will be generated without graph optimizations
constructor2 = get_sibling_constructor(NoOptimizations, AzureDownloadableModel, "resnet50_vaiq_int8")
constructor3 = get_sibling_constructor(NoOptimizations, AzureDownloadableModel, "ResNet50_vaiq")
register_test(constructor2, "resnet50_vaiq_int8_no_opt")
register_test(constructor3, "ResNet50_vaiq_no_opt")

# truncated_constructor = get_trucated_constructor(TruncatedModel, AzureDownloadableModel, "ResNet50_vaiq")
# for n in range(4, 10):
#     register_test(truncated_constructor(n,""), f"ResNet50_vaiq_trunc_{n}")