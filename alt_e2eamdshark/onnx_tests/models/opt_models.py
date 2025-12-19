# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy
import torch
from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors
from ..helper_classes import AzureDownloadableModel


class Opt125MAWQModelInfo(AzureDownloadableModel):
    def set_model_params(self, batch_size, sequence_length, past_sequence_length):
        self.dim_params = (
            "batch_size",
            "sequence_length",
            "past_sequence_length",
            "past_sequence_length + 1",
        )
        self.dim_values = (
            batch_size,
            sequence_length,
            past_sequence_length,
            past_sequence_length + 1,
        )

    def construct_inputs(self):
        self.set_model_params(1, 1, 0)
        pv_zip = zip(self.dim_params, self.dim_values)
        pv = dict(pv_zip)

        model_inputs = [
            numpy.random.randint(
                -1000,
                high=1000,
                size=(pv["batch_size"], pv["sequence_length"]),
                dtype=numpy.int64,
            )
        ]  # input_ids
        model_inputs.append(
            numpy.random.randint(
                -10,
                high=10,
                size=(pv["batch_size"], pv["past_sequence_length + 1"]),
                dtype=numpy.int64,
            )
        )  # attention_mask
        for i in range(2 * 12):
            model_inputs.append(
                numpy.random.rand(
                    pv["batch_size"], 12, pv["past_sequence_length"], 64
                ).astype(numpy.float32)
            )  # 12 key/value pairs
        return TestTensors(model_inputs)


register_test(Opt125MAWQModelInfo, "opt-125M-awq")

# hugging face example:
class Opt125MGPTQModelInfo(OnnxModelInfo):
    def construct_model(self):
        # model origin: https://huggingface.co/jlsilva/facebook-opt-125m-gptq4bit
        test_modelname = "facebook/opt-125m"
        quantizedmodelname = "jlsilva/facebook-opt-125m-gptq4bit"
        kwargs = {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
        }
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

        quantization_config = GPTQConfig(bits=8, disable_exllama=True)
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "cpu"
        self.pytorch_model = AutoModelForCausalLM.from_pretrained(
            quantizedmodelname, **kwargs
        )
        # model.output_hidden_states = False
        self.tokenizer = AutoTokenizer.from_pretrained(test_modelname)
        self.prompt = "What is nature of our existence?"
        self.encoding = self.tokenizer(self.prompt, return_tensors="pt")
        torch.onnx.export(
            self.pytorch_model,
            (self.encoding["input_ids"], self.encoding["attention_mask"]),
            self.model,
            opset_version=20,
        )

    def construct_inputs(self):
        if not self.__dict__.__contains__("tokenizer"):
            self.construct_model()
        return TestTensors(
            (self.encoding["input_ids"], self.encoding["attention_mask"])
        )

    def forward(self, input):
        response = self.pytorch_model.generate(
            input.data[0],
            do_sample=True,
            top_k=50,
            max_length=100,
            top_p=0.95,
            temperature=1.0,
        )
        return TestTensors((response,))

        # model_response = model.generate(
        #     E2EAMDSHARK_CHECK["input"],
        #     do_sample=True,
        #     top_k=50,
        #     max_length=100,
        #     top_p=0.95,
        #     temperature=1.0,
        # )
        # print("Response:", tokenizer.decode(model_response[0]))


register_test(Opt125MGPTQModelInfo, "opt-125m-gptq")


# this is to sample adding a static version of a test
# class Opt125MAWQStaticModel(Opt125MAWQModelInfo):
#     def construct_model(self):
#         self.set_model_params(1,1,0)
#         pv_zip = zip(self.dim_params, self.dim_values)
#         pv = dict(pv_zip)
#         model = onnx.load(self.model.rstrip("/model.onnx").rstrip(self.name) + "opt-125M-awq/onnx/models/opt-125M-awq/model.onnx")
#         for p in self.dim_params:
#             make_dim_param_fixed(model.graph, p, pv[p])
#         fix_output_shapes(model)
#         onnx.save(model, self.model)
# register_test(Opt125MAWQStaticModel, "static-opt-125M-awq")
