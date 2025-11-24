import os
import torch
from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModel, T5EncoderModel
from diffusers import FluxTransformer2DModel
from e2e_testing.framework import (
    OnnxModelInfo,
    ExtraOptions,
    ImporterOptions,
    RuntimeOptions,
    CompilerOptions,
)
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors
from pathlib import Path
from typing import Union


class FluxTransformerModelInfo(OnnxModelInfo):
    def __init__(self, name, onnx_model_path, opset_version=None):
        # set a cache dir for the hf model weights
        parent_cache_dir = os.getenv("CACHE_DIR")
        if not parent_cache_dir:
            raise RuntimeError(
                "Please specify a cache directory path in the CACHE_DIR environment variable for storing large model files."
            )
        self.cache_dir = os.path.join(parent_cache_dir, name)
        # hf model info
        self.hf_model_path = "black-forest-labs/FLUX.1-dev"
        self.model_dir = "transformer"
        # where to save the irpa file for iree-run-module
        # note, the importer will generate another param file at
        # "onnx_model_path / model.torch_onnx_params.irpa", since
        # some bias values are large enough to externalize, but weren't
        # part of the saftensors files.
        self.param_path = str(Path(onnx_model_path) / "model_params.irpa")
        # these are customizable:
        self.dynamic = False
        self.dynamo = False
        self.bs = 1
        self.max_len = 512
        self.img_height = 1024
        self.img_width = 1024
        self.compression_factor = 8
        self.torch_dtype = torch.float32
        self.update_customizable_vals()
        # these are determined from above
        self.latent_h = self.img_height // self.compression_factor
        self.latent_w = self.img_width // self.compression_factor
        self.latent_dim = (self.latent_h // 2) * (self.latent_w // 2)
        super().__init__(name, onnx_model_path, opset_version)

    def update_dim_param_dict(self):
        self.dim_param_dict = {
            "B": self.bs,
            "latent_dim": self.latent_dim,
            "L": self.max_len,
        }

    def update_customizable_vals(self):
        """override to modify dynamic, dyanmo, bs, max_len, img_height, img_width, compression_factor, or torch_dtype"""
        pass

    def update_extra_options(self):
        # we decided to set `export_params=False` and `do_constant_folding=False`
        # in the torch.onnx.export of this model. This will convert initializers
        # to inputs in the onnx model, which we need to convert back to util.global
        # ops in the importer. Therefore, we specify that the model has 7 actual inputs
        # for the `externalize_params` route in the iree-import-onnx tool. `large-model`
        # is additionally specified since we don't need to update the opset version
        # or check the model size.
        # The compile flags for rocm match what we are using for similar models.
        # Lastly, we add some runtime options specifically to point to the irpa files
        # generated for this test.
        if self.dynamic:
            rocm_compile_flags = tuple()
        else:
            rocm_compile_flags = (
                "iree-execution-model=async-external",
                "iree-global-opt-propagate-transposes=1",
                "iree-opt-const-eval=0",
                "iree-opt-outer-dim-concat=1",
                "iree-opt-aggressively-propagate-transposes=1",
                "iree-dispatch-creation-enable-aggressive-fusion",
                "iree-codegen-llvmgpu-use-vector-distribution=1",
                "iree-llvmgpu-enable-prefetch=1",
                "iree-codegen-gpu-native-math-precision=1",
                "iree-hip-legacy-sync=0",
                "iree-opt-data-tiling=0",
                "iree-vm-target-truncate-unsupported-floats",
                "iree-hal-force-indirect-command-buffers",
                "iree-preprocessing-pass-pipeline="
                "'builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), "
                "iree-preprocessing-transpose-convolution-pipeline, "
                "iree-preprocessing-pad-to-intrinsics, "
                "util.func(iree-preprocessing-generalize-linalg-matmul-experimental))'",
                "iree-dispatch-creation-enable-fuse-horizontal-contractions=1",
            )
        self.extra_options = ExtraOptions(
            import_model_options=ImporterOptions(
                externalize_inputs_threshold=7,
                num_elements_threshold=32,
                externalize_params=True,
                large_model=True,
            ),
            compilation_options=CompilerOptions(
                backend_specific_flags={
                    "rocm": rocm_compile_flags,
                    "hip": rocm_compile_flags,
                }
            ),
            compiled_inference_options=RuntimeOptions(
                common_extra_args=(
                    f"parameters=model={self.param_path}",
                    f'parameters=model={Path(self.model).parent / "model.torch_onnx_params.irpa"}',
                )
            ),
        )

    def construct_inputs(self):
        torch.manual_seed(0)
        shapes, dtypes = self.get_signature()
        return TestTensors(
            (
                torch.randn(*(shapes[0]), dtype=dtypes[0]),
                torch.randn(*(shapes[1]), dtype=dtypes[1]),
                torch.randn(*(shapes[2]), dtype=dtypes[2]),
                torch.ones(*(shapes[3]), dtype=dtypes[3]),
                torch.randn(*(shapes[4]), dtype=dtypes[4]),
                torch.randn(*(shapes[5]), dtype=dtypes[5]),
                torch.ones(*(shapes[6]), dtype=dtypes[6]),
            )
        )

    def construct_model(self):
        if os.path.isfile(self.model):
            return

        # we will use iree turbine to save params
        try:
            from iree.turbine.aot import save_module_parameters
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "The package iree-turbine (try pip install iree-turbine) "
                "is used to save hf model parameters to irpa format for flux."
            ) from e

        print("\nloading hf model...")
        model = FluxTransformer2DModel.from_pretrained(
            self.hf_model_path,
            subfolder=self.model_dir,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
        )
        print("hf model loaded.")

        print("saving params (this might take a while)...")
        save_module_parameters(self.param_path, model)
        print("params saved.")

        input_names = [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
        ]

        dynamic_axes = (
            {
                "hidden_states": {0: "B", 1: "latent_dim"},
                "encoder_hidden_states": {0: "B", 1: "L"},
                "pooled_projections": {0: "B"},
                "timestep": {0: "B"},
                "img_ids": {0: "latent_dim"},
                "txt_ids": {0: "L"},
                "guidance": {0: "B"},
            }
            if self.dynamic
            else {}
        )

        output_names = ["latent"]

        sample_inputs = self.construct_inputs().data

        print("exporting hf model to onnx (this might take a while)...")
        with torch.inference_mode():
            torch.onnx.export(
                model,
                sample_inputs,
                self.model,
                export_params=False,
                do_constant_folding=False,
                keep_initializers_as_inputs=True,
                opset_version=19,
                dynamo=self.dynamo,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        print("onnx model generated.")

        if not os.path.isfile(self.model):
            raise RuntimeError(
                f"Torch onnx export failed to produce an onnx model at {self.model}"
            )

    def forward(self, input):
        with torch.inference_mode():
            model = FluxTransformer2DModel.from_pretrained(
                self.hf_model_path,
                subfolder=self.model_dir,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
            )
            data = input.to_torch().data
            return TestTensors(
                model.forward(
                    data[0],
                    data[1],
                    data[2],
                    data[3],
                    data[4],
                    data[5],
                    data[6],
                    return_dict=False,
                )
            )

    def get_signature(self, *, from_inputs=True, leave_dynamic=False):
        config = FluxTransformer2DModel.load_config(
            self.hf_model_path, subfolder=self.model_dir
        )

        def d(dim: Union[str, int]) -> Union[str, int]:
            if (leave_dynamic and self.dynamic) or isinstance(dim, int):
                return dim
            return self.dim_param_dict[dim]

        B = d("B")
        L = d("L")
        latent_dim = d("latent_dim")

        if from_inputs:
            shapes = [
                [B, latent_dim, config["in_channels"]],
                [B, L, config["joint_attention_dim"]],
                [B, config["pooled_projection_dim"]],
                [B],
                [latent_dim, 3],
                [L, 3],
                [B],
            ]
        else:
            shapes = [[B, latent_dim, config["in_channels"]]]

        dtypes = len(shapes) * [self.torch_dtype]
        return shapes, dtypes


register_test(FluxTransformerModelInfo, "flux_1_dev_transformer")


class FluxTextEncoderModelInfo(FluxTransformerModelInfo):
    def update_customizable_vals(self):
        self.model_dir = "text_encoder"
        self.dynamo = False
        self.dynamic = True
        self.cls = CLIPTextModel
        self.bs = 1
        self.max_len = 77
        self.externalize_params = False

    def update_extra_options(self):
        if not self.externalize_params:
            self.extra_options = ExtraOptions()
            return
        importer_options = ImporterOptions(
            externalize_params=True,
            externalize_inputs_threshold=1,
        )
        runtime_options = RuntimeOptions(
            common_extra_args=(
                f"parameters=model={self.param_path} ",
                f'parameters=model={Path(self.model).parent / "model.torch_onnx_params.irpa"}',
            )
        )
        self.extra_options = ExtraOptions(
            import_model_options=importer_options,
            compiled_inference_options=runtime_options,
        )

    def update_dim_param_dict(self):
        self.dim_param_dict = {"B": 1}

    def construct_inputs(self):
        return TestTensors(
            (torch.zeros(self.dim_param_dict["B"], self.max_len, dtype=torch.int32),)
        )

    def get_signature(self, *, from_inputs=True, leave_dynamic=False):
        def d(dim: Union[str, int]) -> Union[str, int]:
            if (leave_dynamic and self.dynamic) or isinstance(dim, int):
                return dim
            return self.dim_param_dict[dim]

        B = d("B")
        if from_inputs:
            return [[B, self.max_len]], [[torch.int32]]

        return [[B, self.max_len, 768], [B, 768]], [
            [self.torch_dtype, self.torch_dtype]
        ]

    def construct_model(self):
        if os.path.isfile(self.model):
            return

        model = self.cls.from_pretrained(
            self.hf_model_path,
            subfolder=self.model_dir,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
        )

        if self.externalize_params:
            try:
                from iree.turbine.aot import save_module_parameters
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "The package iree-turbine (try pip install iree-turbine) "
                    "is used to save hf model parameters to irpa format for flux."
                ) from e
            save_module_parameters(self.param_path, model)
        input_names = ["input_ids"]
        dynamic_axes = {"input_ids": {0: "B"}} if self.dynamic else {}
        output_names = ["last_hidden_state"]
        sample_inputs = self.construct_inputs().data

        # CLIP export requires nightly pytorch due to bug in onnx parser
        with torch.inference_mode():
            # y = model.forward(sample_inputs[0])
            # for key, value in y.items():
            #     print(f'{key} : {value.shape}, {value.dtype}')
            torch.onnx.export(
                model,
                sample_inputs,
                self.model,
                export_params=(not self.externalize_params),
                do_constant_folding=(not self.externalize_params),
                opset_version=19,
                dynamo=self.dynamo,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

        if not os.path.isfile(self.model):
            raise RuntimeError(
                f"Torch onnx export failed to produce an onnx model at {self.model}"
            )

    def forward(self, inputs):
        model = self.cls.from_pretrained(
            self.hf_model_path,
            subfolder=self.model_dir,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
        )
        data = inputs.to_torch().data
        return TestTensors(
            model.forward(
                data[0],
                return_dict=False,
            )
        )


register_test(FluxTextEncoderModelInfo, "flux_1_dev_clip")


class FluxTextEncoder2ModelInfo(FluxTextEncoderModelInfo):
    def update_customizable_vals(self):
        self.model_dir = "text_encoder_2"
        self.dynamo = False
        self.dynamic = True
        self.cls = T5EncoderModel
        self.bs = 1
        self.max_len = 512
        self.externalize_params = True

    def get_signature(self, *, from_inputs=True, leave_dynamic=False):
        def d(dim: Union[str, int]) -> Union[str, int]:
            if (leave_dynamic and self.dynamic) or isinstance(dim, int):
                return dim
            return self.dim_param_dict[dim]

        B = d("B")
        if from_inputs:
            return [[B, self.max_len]], [[torch.int32]]

        return [[B, self.max_len, 4096]], [[self.torch_dtype]]


register_test(FluxTextEncoder2ModelInfo, "flux_1_dev_t5")
