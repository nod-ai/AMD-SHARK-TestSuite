# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import abc
from typing import TypeVar
from e2e_testing.storage import TestTensors

CompiledArtifact = TypeVar("CompiledArtifact")
Invoker = TypeVar("Invoker")

# This file should contain customizations for how to compile mlir from various entrypoints


class BackendBase(abc.ABC):

    @abc.abstractmethod
    def compile(self, module) -> CompiledArtifact:
        """specifies how to compile an MLIR Module"""

    @abc.abstractmethod
    def load(self, artifact: CompiledArtifact, func_name: str) -> Invoker:
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
