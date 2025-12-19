# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from transformers import DeiTImageProcessor, DeiTForImageClassification
from PIL import Image
import requests

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2EAMDSHARK_CHECK_DEF

# Create an instance of it for this test
E2EAMDSHARK_CHECK = dict(E2EAMDSHARK_CHECK_DEF)

# model origin: https://huggingface.co/facebook/deit-small-distilled-patch16-224
test_modelname = "facebook/deit-small-distilled-patch16-224"
test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(test_image_url, stream=True).raw)

processor = DeiTImageProcessor.from_pretrained(test_modelname)
model = DeiTForImageClassification.from_pretrained(test_modelname)

model.to("cpu")
model.eval()

encoding = processor(images=image, return_tensors="pt")

E2EAMDSHARK_CHECK["input"] = encoding["pixel_values"].cpu()
E2EAMDSHARK_CHECK["output"] = model(E2EAMDSHARK_CHECK["input"]).logits
predicted_class_idx = E2EAMDSHARK_CHECK["output"].argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
print("Input:", E2EAMDSHARK_CHECK["input"])
print("Output:", E2EAMDSHARK_CHECK["output"])
# For geneartive AI models, input is int and should be kept that way for
# casted models as well
E2EAMDSHARK_CHECK["inputtodtype"] = False
