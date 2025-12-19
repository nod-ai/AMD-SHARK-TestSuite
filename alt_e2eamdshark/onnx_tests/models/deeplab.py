# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy
import onnxruntime
import torch
import urllib
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ..helper_classes import (
    AzureDownloadableModel,
    SiblingModel,
    get_sibling_constructor,
)
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors

label_map = numpy.array(
    [
        (0, 0, 0),  # background
        (128, 0, 0),  # aeroplane
        (0, 128, 0),  # bicycle
        (128, 128, 0),  # bird
        (0, 0, 128),  # boat
        (128, 0, 128),  # bottle
        (0, 128, 128),  # bus
        (128, 128, 128),  # car
        (64, 0, 0),  # cat
        (192, 0, 0),  # chair
        (64, 128, 0),  # cow
        (192, 128, 0),  # dining table
        (64, 0, 128),  # dog
        (192, 0, 128),  # horse
        (64, 128, 128),  # motorbike
        (192, 128, 128),  # person
        (0, 64, 0),  # potted plant
        (128, 64, 0),  # sheep
        (0, 192, 0),  # sofa
        (128, 192, 0),  # train
        (0, 64, 128),  # tv/monitor
    ]
)


class DeeplabModel(SiblingModel):
    def update_sess_options(self):
        self.sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

    def construct_inputs(self):
        filename = str(Path(self.model).parent.joinpath("input.png"))
        url = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png"
        try:
            urllib.URLopener().retrieve(url, filename)
        except:
            urllib.request.urlretrieve(url, filename)
        input_image = Image.open(filename)
        input_image = input_image.convert("RGB")
        self.img_size = input_image.size
        (shapes, dtypes) = self.get_signature(from_inputs=True)
        self.shape = shapes[0]
        if self.shape[1] == 3:
            self.spatial_shape = self.shape[2:]
            self.channels_last = False
        elif self.shape[3] == 3:
            self.spatial_shape = self.shape[1:3]
            self.channels_last = True
        else:
            return ValueError("Expected the input shape to have three channels (RGB)")
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize(self.spatial_shape),
            ]
        )
        input_tensor = preprocess(input_image)
        input_tensor = input_tensor.unsqueeze(0)
        if self.channels_last:
            input_tensor = input_tensor.transpose(1, 2).transpose(2, 3)
        return TestTensors((input_tensor,))

    def apply_postprocessing(self, output: TestTensors):
        processed_outputs = []
        for d in output.to_torch().data:
            if self.channels_last:
                c = torch.topk(torch.nn.functional.softmax(d, -1), 2)[-1][0, :, :, -1]
            else:
                c = torch.topk(torch.nn.functional.softmax(d, 1), 2, 1)[-1][0, -1, :, :]

            image = numpy.zeros(
                (self.spatial_shape[0], self.spatial_shape[1], 3)
            ).astype(numpy.uint8)
            for i in range(self.spatial_shape[0]):
                for j in range(self.spatial_shape[1]):
                    for k in range(3):
                        image[i, j, k] = label_map[c[i, j]][k]
            processed_outputs.append(image)
        return TestTensors(processed_outputs)

    def save_processed_output(self, output: TestTensors, save_to: str, name: str):
        c = 0
        for d in output.to_numpy().data:
            im = Image.fromarray(d)
            if self.img_size:
                im = im.resize(self.img_size)
            fp = save_to + name + "." + str(c) + ".jpeg"
            im.save(fp)
            c += 1


# base test (no post-processing or input mods)
register_test(AzureDownloadableModel, "deeplabv3")
register_test(AzureDownloadableModel, "DeepLabV3_resnet50_vaiq_int8")

# sibling test with all the bells & whistles
constructor0 = get_sibling_constructor(
    DeeplabModel, AzureDownloadableModel, "deeplabv3"
)
constructor1 = get_sibling_constructor(
    DeeplabModel, AzureDownloadableModel, "DeepLabV3_resnet50_vaiq_int8"
)
register_test(constructor0, "deeplabv3_real_with_pp")
register_test(constructor1, "DeepLabV3_resnet50_vaiq_int8_real_with_pp")
