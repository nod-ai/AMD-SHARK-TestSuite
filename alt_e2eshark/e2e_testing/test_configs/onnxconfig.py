# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import onnx
from torch_mlir.extras import onnx_importer
from torch_mlir.dialects import torch as torch_d
from torch_mlir.ir import Context
from e2e_testing.backends import BackendBase
from e2e_testing.framework import TestConfig, OnnxModelInfo, Module, CompiledArtifact
from e2e_testing.storage import TestTensors
from torch_mlir.passmanager import PassManager
from typing import Tuple, Any
from onnxruntime import InferenceSession
import os
from pathlib import Path

REDUCE_TO_LINALG_PIPELINE = [
    "torch-lower-to-backend-contract",
    "torch-backend-to-linalg-on-tensors-backend-pipeline",
]


class OnnxEpTestConfig(TestConfig):
    '''This is the basic testing configuration for onnx models'''
    def __init__(self, log_dir: str, backend: BackendBase):
        super().__init__()
        self.log_dir = log_dir
        self.backend = backend

    def import_model(self, model_info: OnnxModelInfo, *, save_to: str = None) -> Tuple[onnx.ModelProto, None]:
        model = onnx.load(model_info.model)
        if model_info.opset_version:
            model = onnx.version_converter.convert_version(
                model, model_info.opset_version
            )
        # don't save the model, since it already exists in the log directory.
        return model, None
    
    def preprocess_model(self, model: onnx.ModelProto, *, save_to: str) -> onnx.ModelProto:
        shaped_model = onnx.shape_inference.infer_shapes(model, data_prop=True)
        if save_to:
            onnx.save(shaped_model, save_to + "inferred_model.onnx")
        return shaped_model

    def compile(self, model: onnx.ModelProto, *, save_to: str = None) -> InferenceSession:
        return self.backend.compile(model, save_to=save_to)

    def run(self, session: InferenceSession, inputs: TestTensors, *, func_name=None) -> TestTensors:
        func = self.backend.load(session)
        return func(inputs)


class OnnxTestConfig(TestConfig):
    '''This is the basic testing configuration for onnx models. This should be initialized with a specific backend, and uses torch-mlir to import the onnx model to torch-onnx MLIR, and apply torch-mlir pre-proccessing passes if desired.'''
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

    def import_model(self, model_info: OnnxModelInfo, *, save_to: str = None) -> Tuple[Module, str]:
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
        # if the pass pipeline is empty, return the original module
        if not self.pass_pipeline:
            return mlir_module
        # convert imported torch-onnx ir to torch
        onnx_to_torch_pipeline = "builtin.module(func.func(convert-torch-onnx-to-torch))"
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

    def compile(self, mlir_module: Module, *, save_to: str = None) -> CompiledArtifact:
        return self.backend.compile(mlir_module, save_to=save_to)

    def run(self, artifact: CompiledArtifact, inputs: TestTensors, *, func_name="main") -> TestTensors:
        func = self.backend.load(artifact, func_name=func_name)
        return func(inputs)

class CLOnnxTestConfig(TestConfig):
    '''This is parallel to OnnxTestConfig, but uses command-line scripts for each stage.'''
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
    
    def import_model(self, program: OnnxModelInfo, *, save_to: str) -> Tuple[str, str]:
        if not save_to:
            raise ValueError("CLOnnxTestConfig requires saving artifacts")
        # setup a detail subdirectory
        os.makedirs(os.path.join(save_to, "detail"), exist_ok=True)
        # setup a commands subdirectory
        os.makedirs(os.path.join(save_to, "commands"), exist_ok=True)
        # set file paths
        mlir_file = os.path.join(save_to, "model.torch_onnx.mlir")
        detail_log = os.path.join(save_to, "detail", "import_model.detail.log")
        commands_log = os.path.join(save_to, "commands", "import_model.commands.log")
        # get a command line script
        script = "python -m torch_mlir.tools.import_onnx "
        script += str(program.model)
        script += " -o "
        script = script + mlir_file
        script += f" 1> {detail_log} 2>&1"
        # log the command
        with open(commands_log, "w") as file:
            file.write(script)
        # remove old mlir_file if present
        Path(mlir_file).unlink(missing_ok=True)
        # run the command
        os.system(script)
        # check if a new mlir file was generated
        if not os.path.exists(mlir_file):
            error_msg = f"failure executing command: \n{script}\n failed to produce mlir file {mlir_file}.\n"
            if os.path.exists(detail_log):
                error_msg += "Error detail:\n\n"
                with open(detail_log,"r+") as file:
                    error_msg += file.read()
            raise FileNotFoundError(error_msg)
        # store output signatures for loading the outputs of iree-run-module
        self.tensor_info_dict[program.name] = program.get_signature(from_inputs=False)
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
        onnx_to_torch_pipeline = "builtin.module(func.func(convert-torch-onnx-to-torch))"
        # get paths
        detail_log = os.path.join(save_to, "detail", "preprocessing.detail.log")
        commands_log = os.path.join(save_to, "commands", "preprocessing.commands.log")
        torch_ir = os.path.join(save_to, "model.torch.mlir")
        linalg_ir = os.path.join(save_to, "model.modified.mlir")
        # generate scripts
        script0 = f"torch-mlir-opt -pass-pipeline='{onnx_to_torch_pipeline}' {mlir_module} -o {torch_ir} 1> {detail_log} 2>&1"
        script1 = f"torch-mlir-opt -pass-pipeline='{self.pass_pipeline}' {torch_ir} -o {linalg_ir} 1> {detail_log} 2>&1"
        # remove old torch_ir
        Path(torch_ir).unlink(missing_ok=True)
        with open(commands_log, "w") as file:
            file.write(script0)
            file.write(script1)
        # run torch-onnx-to-torch
        os.system(script0)
        if not os.path.exists(torch_ir):
            error_msg = f"failure executing command: \n{script0}\n failed to produce mlir file {torch_ir}.\n"
            if os.path.exists(detail_log):
                error_msg += "Error detail:\n\n"
                with open(detail_log,"r+") as file:
                    error_msg += file.read()
            raise FileNotFoundError(error_msg)
        # remove old linalg ir
        Path(linalg_ir).unlink(missing_ok=True)
        # run torch-to-linalg pipeline
        os.system(script1)
        if not os.path.exists(linalg_ir):
            error_msg = f"failure executing command: \n{script1}\n failed to produce mlir file {linalg_ir}.\n"
            if os.path.exists(detail_log):
                error_msg += "Error detail:\n\n"
                with open(detail_log,"r+") as file:
                    error_msg += file.read()
            raise FileNotFoundError(error_msg)
        return linalg_ir
    
    def compile(self, mlir_module: str, *, save_to: str = None) -> str:
        return self.backend.compile(mlir_module, save_to=save_to)

    def run(self, artifact: str, inputs: TestTensors, *, func_name=None) -> TestTensors:
        run_dir = Path(artifact).parent
        test_name = run_dir.name
        detail_log = run_dir.joinpath("detail", "compiled_inference.detail.log")
        commands_log = run_dir.joinpath("commands", "compiled_inference.commands.log")
        func = self.backend.load(artifact, func_name=func_name)
        script = func(inputs)
        num_outputs = len(self.tensor_info_dict[test_name][0])
        output_files = []
        for i in range(num_outputs):
            output_files.append(os.path.join(run_dir, f"output.{i}.bin"))
            script += f" --output=@'{output_files[i]}'"
            # remove existing output files if they already exist
            # we use the existence of these files to check if the inference succeeded.
            Path(output_files[i]).unlink(missing_ok=True)
        # dump additional error messaging to the detail log.
        script += f" 1> {detail_log} 2>&1"
        with open(commands_log, "w") as file:
            file.write(script)
        os.system(script)
        for file in output_files:
            if not os.path.exists(file):
                error_msg = f"failure executing command: \n{script}\n failed to produce output file {file}.\n"
                if os.path.exists(detail_log):
                    error_msg += "Error detail:\n\n"
                    with open(detail_log,"r+") as file:
                        error_msg += file.read()
                raise FileNotFoundError(error_msg)
        return TestTensors.load_from(self.tensor_info_dict[test_name][0], self.tensor_info_dict[test_name][1], run_dir, "output")

