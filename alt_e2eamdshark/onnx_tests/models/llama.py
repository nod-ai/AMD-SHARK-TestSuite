import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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


class LlamaModelInfo(OnnxModelInfo):
    def __init__(self, name, onnx_model_path, opset_version=None):
        # set a cache dir for the hf model weights
        parent_cache_dir = os.getenv("CACHE_DIR")
        if not parent_cache_dir:
            raise RuntimeError(
                "Please specify a cache directory path in the CACHE_DIR environment variable for storing large model files."
            )
        self.cache_dir = os.path.join(parent_cache_dir, name)
        # hf model info
        self.hf_model_path = "meta-llama/Llama-3.1-8B"
        self.param_path = str(Path(onnx_model_path) / "model_params.irpa")
        # these are customizable:
        self.dynamic = False
        self.dynamo = False
        self.batch_size  = 1
        self.max_seq_len = 512
        self.torch_dtype = torch.float32
        self.update_customizable_vals()
        super().__init__(name, onnx_model_path, opset_version)

    def update_dim_param_dict(self):
        self.dim_param_dict = {
            "B": self.batch_size,
            "L": self.max_seq_len,
        }

    def update_customizable_vals(self):
        """override to modify dynamic, dyanmo, batch_size, max_seq_len, or torch_dtype"""
        pass

    def update_extra_options(self):
        # we decided to set `export_params=False` and `do_constant_folding=False`
        # in the torch.onnx.export of this model. This will convert initializers
        # to inputs in the onnx model, which we need to convert back to util.global
        # ops in the importer. Therefore, we specify that the model has 3 actual inputs
        # for the `externalize_params` route in the iree-import-onnx tool. `large-model`
        # is additionally specified since we don't need to update the opset version
        # or check the model size.
        # TODO: figure out which rocm flags to use
        self.extra_options = ExtraOptions(
            import_model_options=ImporterOptions(
                externalize_inputs_threshold=3,
                num_elements_threshold=32,
                externalize_params=True,
                large_model=True,
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
        # TODO: Allow passing in position_ids, attn_mask, and past_key_value pairs.
        # Currently, a 0 tensor needs to be passed as the third input for the onnx model.
        # This is possibly due to a lack of position_ids provided for the export. 
        return TestTensors(
            (
                torch.ones(*(shapes[0]), dtype=dtypes[0]),
                torch.ones(*(shapes[1]), dtype=dtypes[1]),
                torch.tensor(0, dtype=dtypes[2]),
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
                "is used to save hf model parameters to irpa format."
            ) from e

        print("\nLoading Hugging Face model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_path,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
        )
        print("Model loaded.")

        print("Saving params (this might take a while)...")
        save_module_parameters(self.param_path, model)
        print("Params saved.")

        dynamic_axes = (
            {
                "input_ids": {0: "B", 1: "L"},
                "attention_mask": {0: "B", 1: "L"},
                "output": {0: "B", 1: "L"},
            }
            if self.dynamic
            else {}
        )
        inputs = self.construct_inputs().data

        print("Exporting model to ONNX (this might take a while)...")
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (inputs[0], inputs[1]),
                self.model,
                export_params=False,
                do_constant_folding=False,
                keep_initializers_as_inputs=True,
                opset_version=19,
                dynamo=self.dynamo,
                input_names=["input_ids", "attention_mask"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )
        print("ONNX model exported.")

        if not os.path.isfile(self.model):
            raise RuntimeError(
                f"Torch onnx export failed to produce an onnx model at {self.model}"
            )

    def forward(self, input):
        with torch.inference_mode():
            model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_path,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
            )
            data = input.to_torch().data
            output = model.forward(
                input_ids=data[0],
                attention_mask=data[1],
                return_dict=False,
            )
            # outputs = (logits, (k0, v0), ..., (k63, v63))
            # the following flattens the output tuples to a list
            output_list = []
            def recursively_unpack(item):
                if isinstance(item, tuple):
                    for t in item:
                        recursively_unpack(t)
                if isinstance(item, torch.Tensor):
                    output_list.append(item)
            recursively_unpack(output)
            return TestTensors(tuple(output_list))

    def get_signature(self, *, from_inputs=True, leave_dynamic=False):
        def d(dim: Union[str, int]) -> Union[str, int]:
            if (leave_dynamic and self.dynamic) or isinstance(dim, int):
                return dim
            return self.dim_param_dict[dim]

        B = d("B")
        L = d("L")

        if from_inputs:
            shapes = [
                [B, L],
                [B, L],
                [],
            ]
            dtypes = [
                torch.int64,
                torch.int64,
                torch.int64,
            ]
        else:
            # output:
            shapes = [[B, L, 128256]]
            dtypes = [torch.float32]
            for _ in range(0,64):
                shapes.append([B,8,L,128])
                dtypes.append(torch.float32)

        return shapes, dtypes

register_test(LlamaModelInfo, "llama3_1_8b")
