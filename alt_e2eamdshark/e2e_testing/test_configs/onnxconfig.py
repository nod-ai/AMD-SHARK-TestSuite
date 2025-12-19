# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Tuple, Any

import onnx
from onnxruntime import InferenceSession

from e2e_testing.backends import BackendBase
from e2e_testing.framework import (
    TestConfig,
    OnnxModelInfo,
    Module,
    CompiledArtifact,
    ImporterOptions,
    CompilerOptions,
    RuntimeOptions,
)
from e2e_testing.logging_utils import run_command_and_log
from e2e_testing.storage import TestTensors

BACKEND_LEGAL_OPS = [
    "aten.flatten.using_ints",
    "aten.unflatten.int",
]

OPTION_STRING = "{backend-legal-ops=" + ",".join(BACKEND_LEGAL_OPS) + "}"

ONNX_TO_TORCH_BACKEND_PIPELINE = [
    f"torch-onnx-to-torch-backend-pipeline{OPTION_STRING}",
]

REDUCE_TO_LINALG_PIPELINE = [
    "torch-backend-to-linalg-on-tensors-backend-pipeline",
]


class OnnxEpTestConfig(TestConfig):
    """This is the basic testing configuration for onnx models"""

    def __init__(self, log_dir: str, backend: BackendBase):
        super().__init__()
        self.log_dir = log_dir
        self.backend = backend

    def import_model(
        self,
        model_info: OnnxModelInfo,
        *,
        save_to: str = None,
        extra_options: ImporterOptions,
    ) -> Tuple[onnx.ModelProto, None]:
        model = onnx.load(model_info.model)
        if model_info.opset_version:
            model = onnx.version_converter.convert_version(
                model, model_info.opset_version
            )
        # don't save the model, since it already exists in the log directory.
        return model, None

    def preprocess_model(
        self, model: onnx.ModelProto, *, save_to: str
    ) -> onnx.ModelProto:
        shaped_model = onnx.shape_inference.infer_shapes(model, data_prop=True)
        if save_to:
            onnx.save(shaped_model, save_to + "inferred_model.onnx")
        return shaped_model

    def compile(
        self,
        model: onnx.ModelProto,
        *,
        save_to: str = None,
        extra_options: CompilerOptions,
    ) -> InferenceSession:
        return self.backend.compile(model, save_to=save_to, extra_options=extra_options)

    def run(
        self,
        session: InferenceSession,
        inputs: TestTensors,
        *,
        func_name=None,
        extra_options: RuntimeOptions,
    ) -> TestTensors:
        func = self.backend.load(session, extra_options=extra_options)
        return func(inputs)

    def benchmark(self, artifact, input, repetitions, *, func_name=None, extra_options):
        raise NotImplementedError("benchmarking not yet supported for EP config.")


class OnnxTestConfig(TestConfig):
    """This is the basic testing configuration for onnx models. This should be initialized with a specific backend, and uses torch-mlir to import the onnx model to torch-onnx MLIR, and apply torch-mlir pre-proccessing passes if desired."""

    def __init__(
        self, log_dir: str, backend: BackendBase, torch_mlir_pipeline: Tuple[str, ...]
    ):
        super().__init__()
        self.log_dir = log_dir
        self.backend = backend
        if len(torch_mlir_pipeline) > 0:
            self.pass_pipeline = "builtin.module(" + ",".join(torch_mlir_pipeline) + ")"
        else:
            self.pass_pipeline = None

    def import_model(
        self,
        model_info: OnnxModelInfo,
        *,
        save_to: str = None,
        extra_options: ImporterOptions,
    ) -> Tuple[Module, str]:
        from torch_mlir.extras import onnx_importer
        from torch_mlir.dialects import torch as torch_d
        from torch_mlir.ir import Context

        model = onnx.load(model_info.model)
        if model_info.opset_version:
            model = onnx.version_converter.convert_version(
                model, model_info.opset_version
            )
        shaped_model = onnx.shape_inference.infer_shapes(model, data_prop=True)
        func_name = shaped_model.graph.name
        context = Context()
        torch_d.register_dialect(context)
        model_info = onnx_importer.ModelInfo(shaped_model)
        m = model_info.create_module(context=context)
        imp = onnx_importer.NodeImporter.define_function(
            model_info.main_graph, m.operation
        )
        imp.import_all()
        # log imported IR
        if save_to:
            with open(os.path.join(save_to, "model.torch_onnx.mlir"), "w") as f:
                f.write(str(m))
        return m, func_name

    def preprocess_model(self, mlir_module: Module, *, save_to: str = None) -> Module:
        from torch_mlir.passmanager import PassManager

        # if the pass pipeline is empty, return the original module
        if not self.pass_pipeline:
            return mlir_module
        # convert imported torch-onnx ir to torch
        onnx_to_torch_pipeline = (
            "builtin.module(" + ",".join(ONNX_TO_TORCH_BACKEND_PIPELINE) + ")"
        )
        with mlir_module.context as ctx:
            pm0 = PassManager.parse(onnx_to_torch_pipeline)
            pm0.run(mlir_module.operation)
            # log torch-mlir IR
            if save_to:
                with open(os.path.join(save_to, "model.torch.mlir"), "w") as f:
                    f.write(str(mlir_module))
            pm1 = PassManager.parse(self.pass_pipeline)
            pm1.run(mlir_module.operation)
            # log modified IR
            if save_to:
                with open(os.path.join(save_to, "model.modified.mlir"), "w") as f:
                    f.write(str(mlir_module))
        return mlir_module

    def compile(
        self,
        mlir_module: Module,
        *,
        save_to: str = None,
        extra_options: CompilerOptions,
    ) -> CompiledArtifact:
        return self.backend.compile(
            mlir_module, save_to=save_to, extra_options=extra_options
        )

    def run(
        self,
        artifact: CompiledArtifact,
        inputs: TestTensors,
        *,
        func_name="main",
        extra_options: RuntimeOptions,
    ) -> TestTensors:
        func = self.backend.load(
            artifact, func_name=func_name, extra_options=extra_options
        )
        return func(inputs)

    def benchmark(self, artifact, input, repetitions, *, func_name=None, extra_options):
        raise NotImplementedError("benchmarking not yet supported for OnnxTestConfig")


class CLOnnxTestConfig(TestConfig):
    """This is parallel to OnnxTestConfig, but uses command-line scripts for each stage."""

    def __init__(
        self, log_dir: str, backend: BackendBase, torch_mlir_pipeline: Tuple[str, ...]
    ):
        super().__init__()
        self.log_dir = log_dir
        self.backend = backend
        self.tensor_info_dict = dict()
        if len(torch_mlir_pipeline) > 0:
            self.pass_pipeline = "builtin.module(" + ",".join(torch_mlir_pipeline) + ")"
        else:
            self.pass_pipeline = None

    def import_model(
        self, program: OnnxModelInfo, *, save_to: str, extra_options: ImporterOptions
    ) -> Tuple[str, str]:
        if not save_to:
            raise ValueError("CLOnnxTestConfig requires saving artifacts")
        # set file paths
        mlir_file = os.path.join(save_to, "model.torch_onnx.mlir")
        # get a command line script
        import_command = [
            "python",
            "-m",
            "iree.compiler.tools.import_onnx",
            str(program.model),
        ]
        # add extra options to command
        bool_arg_defaults = {"large-model": False, "externalize-params": False}
        for underscore_key, value in extra_options._asdict().items():
            if value is None:
                continue
            key = underscore_key.replace("_", "-")
            if key not in bool_arg_defaults.keys():
                import_command.append(f"--{key}={value}")
            elif value != bool_arg_defaults[key]:
                import_command.extend([f"--{key}"])
        import_command.extend(["-o", mlir_file])
        run_command_and_log(import_command, save_to, "import_model")
        # get the func name
        # TODO put this as an OnnxModelInfo attr?
        model = onnx.load(program.model, load_external_data=False)
        func_name = model.graph.name
        return mlir_file, func_name

    def preprocess_model(self, mlir_module: str, *, save_to: str = None) -> Module:
        # if the pass pipeline is empty, return the original module
        if not self.pass_pipeline:
            return mlir_module
        # convert imported torch-onnx ir to torch
        onnx_to_torch_pipeline = (
            "builtin.module(" + ",".join(ONNX_TO_TORCH_BACKEND_PIPELINE) + ")"
        )
        # get paths
        torch_ir = os.path.join(save_to, "model.torch.mlir")
        linalg_ir = os.path.join(save_to, "model.modified.mlir")
        # generate scripts
        opt_tool = "torch-mlir-opt"
        use_tmopt = shutil.which("torch-mlir-opt")
        if use_tmopt is None:
            warnings.warn(
                "\nCould not find command line tool 'torch-mlir-opt'. Defaulting to 'iree-opt'."
            )
            opt_tool = "iree-opt"
        command0 = [
            opt_tool,
            f"-pass-pipeline='{onnx_to_torch_pipeline}'",
            mlir_module,
            "-o",
            torch_ir,
        ]
        command1 = [
            opt_tool,
            f"-pass-pipeline='{self.pass_pipeline}'",
            torch_ir,
            "-o",
            linalg_ir,
        ]
        run_command_and_log(command0, save_to, "preprocessing_torch_onnx_to_torch")
        run_command_and_log(command1, save_to, "preprocessing_torch_to_linalg")
        return linalg_ir

    def compile(
        self, mlir_module: str, *, save_to: str = None, extra_options: CompilerOptions
    ) -> str:
        return self.backend.compile(
            mlir_module, save_to=save_to, extra_options=extra_options
        )

    def run(
        self,
        artifact: str,
        inputs: TestTensors,
        *,
        func_name=None,
        extra_options: RuntimeOptions,
    ) -> TestTensors:
        run_dir = Path(artifact).parent
        test_name = run_dir.name
        func = self.backend.load(
            artifact, func_name=func_name, extra_options=extra_options
        )
        command = func(inputs)
        num_outputs = len(self.tensor_info_dict[test_name][0])
        command.extend(
            [
                f"--output=@'{os.path.join(run_dir, f'output.{i}.bin')}'"
                for i in range(num_outputs)
            ]
        )
        run_command_and_log(command, save_to=run_dir, stage_name="compiled_inference")
        return TestTensors.load_from(
            self.tensor_info_dict[test_name][0],
            self.tensor_info_dict[test_name][1],
            run_dir,
            "output",
        )

    def benchmark(
        self,
        artifact: str,
        inputs: TestTensors,
        repetitions: int = 5,
        *,
        func_name=None,
        extra_options=None,
    ) -> float:
        run_dir = Path(artifact).parent
        report_json = run_dir.joinpath("benchmark.json")
        func = self.backend.load(
            artifact, func_name=func_name, extra_options=extra_options
        )
        command = func(inputs)
        # replace "iree-run-module" with "iree-benchmark-module"
        command[0] = "iree-benchmark-module"
        command.extend(
            [
                f"--benchmark_repetitions={repetitions}",
                "--device_allocator=caching",
                f"--benchmark_out='{report_json}'",
                "--benchmark_out_format=json",
            ]
        )
        run_command_and_log(command, save_to=run_dir, stage_name="benchmark")
        # load benchmark time from report_json
        with open(report_json) as contents:
            loaded_dict = json.load(contents)
        mean_stats = loaded_dict["benchmarks"][-4]
        if mean_stats["name"] != f"BM_{func_name}/process_time/real_time_mean":
            raise ValueError("Name of benchmark item is unexpected")
        time = mean_stats["real_time"]
        return time
