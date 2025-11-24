# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import abc
import os
from pathlib import Path
from typing import TypeVar, List

from onnx import ModelProto
import onnxruntime as ort

from e2e_testing.framework import CompiledOutput, ModelArtifact, CompilerOptions, RuntimeOptions
from e2e_testing.logging_utils import run_command_and_log
from e2e_testing.storage import TestTensors, get_shape_string

Invoker = TypeVar("Invoker")


class BackendBase(abc.ABC):

    @abc.abstractmethod
    def compile(self, module: ModelArtifact, extra_options : CompilerOptions) -> CompiledOutput:
        """specifies how to compile an MLIR Module"""

    @abc.abstractmethod
    def load(self, artifact: CompiledOutput, func_name: str, extra_options : RuntimeOptions) -> Invoker:
        """loads the function with name func_name from compiled artifact. This method should return a function callable from python."""


def flag(arg : str) -> str:
    if arg.startswith("--"):
        return arg
    return f'--{arg}'

class SimpleIREEBackend(BackendBase):
    '''This backend uses iree to compile and run MLIR modules for a specified hal_target_backend'''
    def __init__(self, *, device="local-task", hal_target_backend="llvm-cpu", extra_args : List[str] = None):
        self.device = device
        self.hal_target_backend = hal_target_backend
        self.extra_args = [] if extra_args is None else [flag(a) for a in extra_args]
        if hal_target_backend == "rocm":
            self.extra_args += [
                f"--iree-hip-target={self.target_chip}",
            ]
        if hal_target_backend == "llvm-cpu":
            self.extra_args += [
                "--iree-llvmcpu-target-cpu=host",
            ]

    def compile(self, module, *, save_to: str = None, extra_options : CompilerOptions):
        from iree import compiler as ireec
        test_specific_args = list(extra_options.common_extra_args)
        if self.hal_target_backend in extra_options.backend_specific_flags.keys():
            test_specific_args += list(extra_options.backend_specific_flags[self.hal_target_backend])
        compile_args = self.extra_args + [flag(arg) for arg in test_specific_args]
        # compile to a vmfb for llvm-cpu
        b = ireec.tools.compile_str(
            str(module),
            target_backends=[self.hal_target_backend],
            extra_args=compile_args,
        )
        # log the vmfb
        if save_to:
            with open(os.path.join(save_to, "compiled_model.vmfb"), "wb") as f:
                f.write(b)
        return b

    def load(self, artifact, *, func_name="main", extra_options : RuntimeOptions):
        from iree import runtime as ireert
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

class CLIREEBackend(BackendBase):
    '''This backend calls iree through the command line to compile and run MLIR modules'''
    def __init__(self, *, device="local-task", hal_target_backend="llvm-cpu", target_chip = None, extra_args : List[str] = None):
        self.device = device
        self.hal_target_backend = hal_target_backend
        self.target_chip = target_chip
        self.extra_args = [] if extra_args is None else [flag(a) for a in extra_args]
        if hal_target_backend == "rocm":
            self.extra_args += [
                f"--iree-hip-target={self.target_chip}",
            ]
        if hal_target_backend == "llvm-cpu":
            self.extra_args += [
                "--iree-llvmcpu-target-cpu=host",
            ]
    
    def compile(self, module_path: str, *, save_to : str = None, extra_options : CompilerOptions) -> str:
        compile_command = ['iree-compile', module_path, f'--iree-hal-target-backends={self.hal_target_backend}']
        compile_command.extend(self.extra_args)
        # add test-specific flags
        test_specific_args = list(extra_options.common_extra_args)
        if self.hal_target_backend in extra_options.backend_specific_flags.keys():
            test_specific_args += list(extra_options.backend_specific_flags[self.hal_target_backend])
        compile_command.extend([flag(arg) for arg in test_specific_args])
        # set output path
        vmfb_path = os.path.join(save_to, "compiled_model.vmfb")
        compile_command.extend(['-o', vmfb_path])
        run_command_and_log(compile_command, save_to, "compilation")
        return vmfb_path
    
    def load(self, vmfb_path: str, *, func_name=None, extra_options : RuntimeOptions):
        """A bit hacky. func returns a script that would dump outputs to terminal output. Modified in config.run method"""
        test_specific_args = list(extra_options.common_extra_args)
        if self.hal_target_backend in extra_options.backend_specific_flags.keys():
            test_specific_args += list(extra_options.backend_specific_flags[self.hal_target_backend])
        run_dir = Path(vmfb_path).parent
        def func(x: TestTensors) -> List[str]:
            command = ["iree-run-module", f"--module='{vmfb_path}'", f"--device={self.device}"]
            command.extend([flag(arg) for arg in test_specific_args])
            if func_name:
                command.append(f"--function='{func_name}'")
            torch_inputs = x.to_torch().data
            for index, input in enumerate(torch_inputs):
                command.append(f"--input='{get_shape_string(input)}=@{run_dir}/input.{index}.bin'")
            return command
        return func
            

class OnnxrtIreeEpBackend(BackendBase):
    '''This backend uses onnxrt iree-ep to compile and run onnx models for a specified hal_target_backend'''
    def __init__(self, *, device="local-task", hal_target_device="llvm-cpu", extra_args : List[str] = None):
        self.device = device
        self.hal_target_device = hal_target_device
        self.extra_args = [] if extra_args is None else [flag(a) for a in extra_args]
        if hal_target_device == "hip":
            # some extra args for Mi250 - some of these may not work for other chips
            self.extra_args += [
                "--iree-hip-target=gfx90a",
            ]
        self.providers = ["IreeExecutionProvider"]
        # set provider options.
        provider_options_dict = dict()
        provider_options_dict["hal_target_device"] = self.hal_target_device
        provider_options_dict["device"] = self.device
        provider_options_dict["compile_time_flags"] = "+".join(self.extra_args)
        self.provider_options = [provider_options_dict]
        self.sess_opt = ort.SessionOptions()
        self.sess_opt.execution_mode  = ort.ExecutionMode.ORT_PARALLEL
        #  sess_opt.log_verbosity_level = 0
        #  self.sess_opt.log_severity_level = 0

    def compile(self, model: ModelProto, *, save_to: str = None, extra_options : CompilerOptions) -> ort.InferenceSession:
        if self.provider_options:
            provider_options_dict = self.provider_options[0]
            provider_options_dict["save_to"] = save_to

        session = ort.InferenceSession(
                   model.SerializeToString(),
                   self.sess_opt,
                   providers=self.providers,
                   provider_options=self.provider_options,
               )
        # can't save an onnx runtime session
        return session

    def load(self, session: ort.InferenceSession, *, func_name=None, extra_options : RuntimeOptions) -> Invoker:
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
