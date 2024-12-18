# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import abc
import onnxruntime as ort
from typing import TypeVar, List
from e2e_testing.storage import TestTensors, get_shape_string
from e2e_testing.framework import CompiledOutput, ModelArtifact, CompilerOptions, RuntimeOptions
from onnx import ModelProto
import os
from pathlib import Path

Invoker = TypeVar("Invoker")


class BackendBase(abc.ABC):

    @abc.abstractmethod
    def compile(self, module: ModelArtifact, extra_options : CompilerOptions) -> CompiledOutput:
        """specifies how to compile an MLIR Module"""

    @abc.abstractmethod
    def load(self, artifact: CompiledOutput, func_name: str, extra_options : RuntimeOptions) -> Invoker:
        """loads the function with name func_name from compiled artifact. This method should return a function callable from python."""


from iree import compiler as ireec
from iree import runtime as ireert


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
        test_specific_args = list(extra_options.common_extra_args)
        if self.hal_target_backend in extra_options.backend_specific_flags.keys():
            test_specific_args += list(extra_options.backend_specific_flags[self.hal_target_backend])
        compile_args = self.extra_args + [flag(arg) for arg in test_specific_args]
        vmfb_path = os.path.join(save_to, "compiled_model.vmfb")
        arg_string = f"--iree-hal-target-backends={self.hal_target_backend} "
        arg_string += ' '.join(compile_args)
        detail_log = os.path.join(save_to, "detail", "compilation.detail.log")
        commands_log = os.path.join(save_to, "commands", "compilation.commands.log")
        script = f"iree-compile {module_path} {arg_string} -o {vmfb_path} 1> {detail_log} 2>&1"
        with open(commands_log, "w") as file:
            file.write(script) 
        # remove old vmfb if it exists
        Path(vmfb_path).unlink(missing_ok=True)
        os.system(script)
        if not os.path.exists(vmfb_path):
            error_msg = f"failure executing command: \n{script}\n failed to produce a vmfb at {vmfb_path}.\n"
            error_msg += f"Error detail in '{detail_log}'"
            raise FileNotFoundError(error_msg)
        return vmfb_path
    
    def load(self, vmfb_path: str, *, func_name=None, extra_options : RuntimeOptions):
        """A bit hacky. func returns a script that would dump outputs to terminal output. Modified in config.run method"""
        test_specific_args = list(extra_options.common_extra_args)
        if self.hal_target_backend in extra_options.backend_specific_flags.keys():
            test_specific_args += list(extra_options.backend_specific_flags[self.hal_target_backend])
        run_dir = Path(vmfb_path).parent
        def func(x: TestTensors) -> str:
            script = f"iree-run-module --module='{vmfb_path}' --device={self.device} "
            for arg in test_specific_args:
                script += f'{flag(arg)} '
            if func_name:
                script += f"--function='{func_name}' "
            torch_inputs = x.to_torch().data
            for index, input in enumerate(torch_inputs):
                script += f"--input='{get_shape_string(input)}=@{run_dir}/input.{index}.bin' "
            return script
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
