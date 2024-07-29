# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import onnxruntime
import torch
import abc
import os
from typing import Union, TypeVar, Tuple, NamedTuple, Dict, Optional, Callable
from e2e_testing.storage import TestTensors
from e2e_testing.onnx_utils import *

# This file two types of classes: framework-specific base classes for storing model info, and generic classes for testing infrastructure.

Module = TypeVar("Module")


class OnnxModelInfo:
    """Stores information about an onnx test: the filepath to model.onnx, how to construct/download it, and how to construct sample inputs for a test run."""

    def __init__(
        self,
        name: str,
        onnx_model_path: str,
        cache_dir: str,
        opset_version: Optional[int] = None,
    ):
        self.name = name
        self.model = onnx_model_path + "model.onnx"
        self.cache_dir = cache_dir
        self.opset_version = opset_version

    def forward(self, input: Optional[TestTensors] = None) -> TestTensors:
        """Applies self.model to self.input. Only override if necessary for specific models"""
        input = input.to_numpy().data
        if not os.path.exists(self.model):
            self.construct_model()
        session = onnxruntime.InferenceSession(self.model, None)
        session_inputs = session.get_inputs()
        session_outputs = session.get_outputs()

        model_output = session.run(
            [output.name for output in session_outputs],
            {session_inputs[i].name: input[i] for i in range(len(session_inputs))},
        )

        return TestTensors(model_output)

    def construct_model(self):
        """a method to be overwritten. To make a new test, define a subclass with an override for this method"""
        raise NotImplementedError(
            f"Model path {self.model} does not exist and no construct_model method is defined."
        )

    def construct_inputs(self):
        """can be overridden to generate specific inputs, but a default is provided for convenience"""
        if not os.path.exists(self.model):
            self.construct_model()
        return get_sample_inputs_for_onnx_model(self.model)

    def apply_postprocessing(self, output: TestTensors):
        """can be overridden to define post-processing methods for individual models"""
        return output
    
    def save_processed_output(self, output: TestTensors, save_to: str, name: str):
        """can be overridden to provide instructions on saving processed outputs (e.g., images, labels, text)"""

    # the following helper methods aren't meant to be overriden

    def get_signature(self, *, from_inputs=True):
        """Returns the input or output signature of self.model"""
        if not os.path.exists(self.model):
            self.construct_model()
        return get_signature_for_onnx_model(self.model, from_inputs=from_inputs)

    def load_inputs(self, dir_path):
        """computes the input signature of the onnx model and loads inputs from bin files"""
        shapes, dtypes = self.get_signature(from_inputs=True)
        try:
            return TestTensors.load_from(shapes, dtypes, dir_path, "input")
        except FileNotFoundError:
            print(
                "\tWarning: bin files missing. Generating new inputs. Please re-run this test without --load-inputs to save input bin files."
            )
            return self.construct_inputs()

    def load_outputs(self, dir_path):
        """computes the input signature of the onnx model and loads outputs from bin files"""
        shapes, dtypes = self.get_signature(from_inputs=False)
        return TestTensors.load_from(shapes, dtypes, dir_path, "output")

    def load_golden_outputs(self, dir_path):
        """computes the input signature of the onnx model and loads golden outputs from bin files"""
        shapes, dtypes = self.get_signature(from_inputs=False)
        return TestTensors.load_from(shapes, dtypes, dir_path, "golden_output")


TestModel = Union[OnnxModelInfo, torch.nn.Module]

CompiledArtifact = TypeVar("CompiledArtifact")


class TestConfig(abc.ABC):

    @abc.abstractmethod
    def mlir_import(self, program: TestModel, *, save_to: str) -> Module:
        """imports the test model to an MLIR Module"""

    @abc.abstractmethod
    def compile(self, mlir_module: Module, *, save_to: str) -> CompiledArtifact:
        """converts the test program to a compiled artifact"""
        pass

    @abc.abstractmethod
    def apply_torch_mlir_passes(self, mlir_module: Module, *, save_to: str) -> Module:
        """applies a (possibly empty) pass pipeline internal to self to provided mlir module"""

    @abc.abstractmethod
    def run(self, artifact: CompiledArtifact, input: TestTensors) -> TestTensors:
        """runs the input through the compiled artifact"""
        pass


class Test(NamedTuple):
    """Used to store the name and TestInfo constructor for a registered test"""

    unique_name: str
    model_constructor: Callable[[], TestModel]


class TestResult(NamedTuple):
    """Used to store associated input and output tensors from a test"""

    name: str
    input: TestTensors
    gold_output: TestTensors
    output: TestTensors


# TODO: find a better home for this random utility function
def result_comparison(test_result: TestResult, tol):
    """compares the output and gold_output stored in a TestResult instance with specified tolerance"""
    output = test_result.output.to_torch().data
    gold = test_result.gold_output.to_torch().data
    if len(output) != len(gold):
        raise ValueError(
            f"num outputs: {len(output)} doesn't match num golden: {len(gold)} for test {test_result.name}"
        )
    match = []
    for i in range(len(output)):
        match.append(torch.isclose(output[i].to(dtype=gold[i].dtype), gold[i], *tol))
    return match
