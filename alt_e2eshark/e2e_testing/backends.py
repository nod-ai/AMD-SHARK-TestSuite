# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import abc
import onnxruntime as ort
from typing import TypeVar, List
from e2e_testing.storage import TestTensors, get_shape_string
from e2e_testing.framework import CompiledOutput, ModelArtifact
from onnx import ModelProto
import os
from pathlib import Path

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
    def __init__(self, *, device="local-task", hal_target_backend="llvm-cpu", extra_args : List[str] = None):
        self.device = device
        self.hal_target_backend = hal_target_backend
        if extra_args:
            self.extra_args = []
            for a in extra_args:
                if a[0:2] == "--":
                    self.extra_args.append(a)
                else:
                    self.extra_args.append("--" + a)
        elif hal_target_backend == "rocm":
            # some extra args for Mi300x - some of these may not work for other chips
            self.extra_args = [
                "--iree-rocm-target-chip=gfx942",
                # "--iree-global-opt-propagate-transposes=true",
                # "--iree-opt-outer-dim-concat=true",
                # "--iree-opt-const-eval=false",
                # "--iree-rocm-waves-per-eu=2",
                # "--iree-llvmgpu-enable-prefetch",
                # "--iree-flow-enable-aggressive-fusion",
                # "--iree-flow-enable-fuse-horizontal-contractions=true",
                # "--iree-opt-aggressively-propagate-transposes=true",
                # "--iree-codegen-llvmgpu-use-vector-distribution=true",
                # "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-preprocessing-pad-to-intrinsics{pad-target-type=conv}))",
                # maybe add iree-preprocessing-transpose-convolution-pipeline to preprocessing pipeline.
            ]
        elif hal_target_backend == "llvm-cpu":
            self.extra_args = [
                "--iree-input-demote-i64-to-i32",
                # "--iree-llvmcpu-fail-on-large-vector=0",
                # "--iree-llvmcpu-stack-allocation-limit=300000",
            ]

    def compile(self, module, *, save_to: str = None):
        # compile to a vmfb for llvm-cpu
        b = ireec.tools.compile_str(
            str(module),
            target_backends=[self.hal_target_backend],
            extra_args=self.extra_args,
        )
        # log the vmfb
        if save_to:
            with open(os.path.join(save_to, "compiled_model.vmfb"), "wb") as f:
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

class CLIREEBackend(BackendBase):
    '''This backend calls iree through the command line to compile and run MLIR modules'''
    def __init__(self, *, device="local-task", hal_target_backend="llvm-cpu", extra_args : List[str] = None):
        self.device = device
        self.hal_target_backend = hal_target_backend
        self.extra_args = []
        if extra_args:
            for a in extra_args:
                if a[0:2] == "--":
                    self.extra_args.append(a)
                else:
                    self.extra_args.append("--" + a)
    
    def compile(self, module_path: str, *, save_to : str = None) -> str:
        vmfb_path = os.path.join(save_to, "compiled_model.vmfb")
        arg_string = f"--iree-hal-target-backends={self.hal_target_backend} "
        for arg in self.extra_args:
            arg_string += arg
            arg_string += " "
        command_error_dump = os.path.join(save_to, "detail", "compilation.detail.log")
        commands_log = os.path.join(save_to, "commands", "compilation.commands.log")
        script = f"iree-compile {module_path} {arg_string}-o {vmfb_path} 1> {command_error_dump} 2>&1"
        with open(commands_log, "w") as file:
            file.write(script) 
        # remove old vmfb if it exists
        Path(vmfb_path).unlink(missing_ok=True)
        os.system(script)
        if not os.path.exists(vmfb_path):
            error_message = f"failure executing command: \n{script}\n failed to produce a vmfb at {vmfb_path}.\n"
            if os.path.exists(command_error_dump):
                error_message += "Error Details:\n\n"
                with open(command_error_dump, "r+") as file:
                    error_message += file.read()
            raise FileNotFoundError(error_message)
        return vmfb_path
    
    def load(self, vmfb_path: str, *, func_name=None):
        """A bit hacky. func returns a script that would dump outputs to terminal output. Modified in config.run method"""
        run_dir = Path(vmfb_path).parent
        def func(x: TestTensors) -> str:
            script = f"iree-run-module --module='{vmfb_path}' --device={self.device}"
            if func_name:
                script += f" --function='{func_name}'"
            torch_inputs = x.to_torch().data
            for index, input in enumerate(torch_inputs):
                script += f" --input='{get_shape_string(input)}=@{run_dir}/input.{index}.bin'"
            return script
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
