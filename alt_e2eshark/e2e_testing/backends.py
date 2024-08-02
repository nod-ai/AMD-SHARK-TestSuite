# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import abc
import onnxruntime as ort
from typing import TypeVar
from e2e_testing.storage import TestTensors
from e2e_testing.framework import CompiledOutput, ModelArtifact
from onnx import ModelProto

Invoker = TypeVar("Invoker")


class BackendBase(abc.ABC):

    @abc.abstractmethod
    def compile(self, module: ModelArtifact) -> CompiledOutput:
        """specifies how to compile an MLIR Module"""

    @abc.abstractmethod
    def load(self, artifact: CompiledOutput, func_name: str) -> Invoker:
        """loads the function with name func_name from compiled artifact. This method should return a function callable from python."""


from iree import compiler as ireec
from iree import runtime as ireert


class SimpleIREEBackend(BackendBase):
    '''This backend uses iree to compile and run MLIR modules for a specified hal_target_backend'''
    def __init__(self, *, device="local-task", hal_target_backend="llvm-cpu"):
        self.device = device
        self.hal_target_backend = hal_target_backend

    def compile(self, module, *, save_to: str = None):
        # compile to a vmfb for llvm-cpu
        b = ireec.tools.compile_str(
            str(module),
            target_backends=[self.hal_target_backend],
            extra_args=["--iree-input-demote-i64-to-i32"],
        )
        # log the vmfb
        if save_to:
            with open(save_to + "compiled_model.vmfb", "wb") as f:
                f.write(b)
        return b

    def load(self, artifact, *, func_name="main"):
        config = ireert.Config(self.device)
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.copy_buffer(ctx.instance, artifact)
        ctx.add_vm_module(vm_module)

        def func(x):
            x = x.data
            device_array = ctx.modules.module[func_name](*x)
            if isinstance(device_array, tuple):
                np_array = []
                for d in device_array:
                    np_array.append(d.to_host())
                return TestTensors(np_array)
            return TestTensors((device_array.to_host(),))

        return func


class OnnxrtIreeEpBackend(BackendBase):
    '''This backend uses onnxrt iree-ep to compile and run onnx models for a specified hal_target_backend'''
    def __init__(self, *, device="local-task", hal_target_backend="llvm-cpu", providers=["IreeExecutionProvider"], inter_op_num_threads=None, intra_op_num_threads=None):
        # may need the device and target_backend for the future (e.g., when IREE-EP has support for specifying)
        self.device = device
        self.hal_target_backend = hal_target_backend

        self.providers=providers
        # TODO: have more session options be optionally configurable by init args
        self.sess_opt = ort.SessionOptions()
        self.sess_opt.execution_mode  = ort.ExecutionMode.ORT_PARALLEL
        if inter_op_num_threads:
            self.sess_opt.inter_op_num_threads = inter_op_num_threads
        if intra_op_num_threads:
            self.sess_opt.intra_op_num_threads = intra_op_num_threads
        #  sess_opt.log_verbosity_level = 0

    def compile(self, model: ModelProto, *, save_to: str = None) -> ort.InferenceSession:
        session = ort.InferenceSession(
                   model.SerializeToString(),
                   self.sess_opt,
                   providers=self.providers,
               )
        # can't save an onnx runtime session
        return session

    def load(self, session: ort.InferenceSession, *, func_name=None) -> Invoker:
        def func(x: TestTensors):
            data = x.to_numpy().data
            session_inputs = session.get_inputs()
            session_outputs = session.get_outputs()
            model_output = session.run(
                [output.name for output in session_outputs],
                {session_inputs[i].name: data[i] for i in range(len(session_inputs))},
            )
            return TestTensors(model_output)

        return func
