# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import requests
import tarfile
import shutil
import yaml
import os

from typing import List, Optional

import onnx
import onnxruntime

from onnx.helper import make_node, make_graph, make_model
from pathlib import Path
from e2e_testing import azutils
from e2e_testing.framework import (
    ExtraOptions,
    ImporterOptions,
    OnnxModelInfo,
    TestTensors,
)
from e2e_testing.onnx_utils import (
    modify_model_output,
    find_node,
    get_sample_inputs_for_onnx_model,
)

"""This file contains several helpful child classes of OnnxModelInfo."""


class HfDownloadableModel(OnnxModelInfo):
    """This class should be used to download models from Huggingface model hub."""

    def __init__(self, full_model_path, task_name, name, onnx_model_path):
        parent_cache_dir = os.getenv("CACHE_DIR")
        if not parent_cache_dir:
            raise RuntimeError(
                "Please specify a cache directory path in the CACHE_DIR environment variable "
                "for storing large model files."
            )
        opset_version = 21
        self.model_repo_path = full_model_path.replace("hf_", "")
        self.task = task_name
        # Appending the test name just adds one redundant level of nesting to the cache dir.
        # Use the value of the CACHE_DIR directly, the segregation of distinct models should
        # be handled by Huggingface itself.
        self.cache_dir = parent_cache_dir
        super().__init__(name, onnx_model_path, opset_version)

    def update_extra_options(self):
        # called in __init__
        # default to using opset version 21 for all ONNX Model Zoo models.
        self.extra_options = ExtraOptions(
            import_model_options=ImporterOptions(opset_version=21)
        )

    def export_model(self, optim_level: str | None = None):
        model_dir = str(Path(self.model).parent)

        # set HF cache explicitly to match provided CACHE_DIR
        os.environ["HF_HOME"] = self.cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = self.cache_dir

        # These tests require optimum. Importing here to avoid raising an exception on importing this file.
        try:
            from optimum.exporters.onnx import main_export
        except ImportError:
            print(
                "Failed to import ONNX Exporter module from optimum. "
                "Please install through `pip install optimum[exporters]`."
            )

        main_export(
            self.model_repo_path,
            output=model_dir,
            task=self.task,
            cache_dir=self.cache_dir,
            local_files_only=False,
            monolith=True,
            framework="pt",
            optimize=optim_level,
        )

    def __repr__(self):
        cls = self.__class__.__name__
        model_details = self.model_repo_path.split('/')
        model_path = '/hf_'.join(model_details)
        return f"{cls} (full_model_path={model_path}, task_name={self.task}, name={self.name}, onnx_model_path={os.path.dirname(self.model)})"

    def construct_model(self):
        model_dir = str(Path(self.model).parent)

        def find_models(model_dir):
            found_models = []
            for root, _, files in os.walk(model_dir):
                for name in files:
                    if name[-5:] == ".onnx":
                        found_models.append(os.path.abspath(os.path.join(root, name)))
            return found_models

        found_models = find_models(model_dir)

        if len(found_models) == 0:
            try:
                self.export_model()
            except:
                raise RuntimeError("Failed to Export class: ", self)

            found_models = find_models(model_dir)
        if len(found_models) == 1:
            self.model = found_models[0]
            return
        if len(found_models) > 1:
            print(f"Found multiple model.onnx files: {found_models}")
            print(f"Picking the first model found to use: {found_models[0]}")
            self.model = found_models[0]
            return
        raise OSError(
            f"No onnx model could be found, downloaded, or extracted to {model_dir}"
        )


class OnnxModelZooDownloadableModel(OnnxModelInfo):
    """This class should be used to download models from ONNX Model Zoo (onnx/models)."""

    def __init__(
        self, is_validated: bool, model_url: str, name: str, onnx_model_path: str
    ):
        opset_version = 21
        parent_cache_dir = os.getenv("CACHE_DIR")
        if not parent_cache_dir:
            raise RuntimeError(
                "Please specify a cache directory path in the CACHE_DIR environment variable for storing large model files."
            )
        self.is_validated = is_validated
        self.model_url = model_url
        self.cache_dir = os.path.join(parent_cache_dir, name)
        os.makedirs(self.cache_dir, exist_ok=True)
        # Validated models do not need a YAML file, we get the inputs directly.
        self.yaml_path = self.get_model_yaml() if not is_validated else None
        # super().__init__ will also call all update methods
        super().__init__(name, onnx_model_path, opset_version)

    def get_model_yaml(self) -> Optional[str]:
        """Tries to download turnkey_stats.yaml to self.cache_dir.
        Returns a path to the dowloaded file if sucessful."""
        # called in __init__
        yaml_path = os.path.join(self.cache_dir, "turnkey_stats.yaml")
        if not os.path.exists(yaml_path):
            turnkey_yaml_url = (
                "/".join(self.model_url.split("/")[:-1]) + "/turnkey_stats.yaml"
            )
            content = requests.get(turnkey_yaml_url).content
            # don't save an empty yaml file
            if content is None or len(content) == 0:
                return
            with open(yaml_path, "wb") as out_file:
                out_file.write(content)
        return yaml_path

    def update_extra_options(self):
        # called in __init__
        # default to using opset version 21 for all ONNX Model Zoo models.
        self.extra_options = ExtraOptions(
            import_model_options=ImporterOptions(opset_version=21)
        )

    def update_input_name_to_shape_map(self):
        # called in __init__
        # reads the input name to shape map from downloaded yaml file
        if not self.yaml_path:
            return
        turnkey_dict = dict()
        self.input_name_to_shape_map = dict()
        with open(self.yaml_path, "rb") as stream:
            turnkey_dict = yaml.safe_load(stream)
        if "onnx_input_dimensions" in turnkey_dict.keys():
            for dim_param in turnkey_dict["onnx_input_dimensions"]:
                self.input_name_to_shape_map[dim_param] = turnkey_dict[
                    "onnx_input_dimensions"
                ][dim_param]

    def unzip_model_archive(self, tar_path: str):
        """helper function for unzipping model and provided inputs to the test-dir"""
        model_dir = str(Path(self.model).parent)
        with tarfile.open(tar_path) as tar:
            for subdir_and_file in tar.getmembers():
                if "test_data_set_0/input" in subdir_and_file.name:
                    subdir_and_file.name = subdir_and_file.name.split("/")[-1]
                    tar.extract(subdir_and_file, path=model_dir)
                if ".onnx" in subdir_and_file.name:
                    subdir_and_file.name = "model.onnx"
                    tar.extract(subdir_and_file, path=model_dir)

    def construct_inputs(self) -> TestTensors:
        if not os.path.exists(self.model):
            self.construct_model()

        input_path = os.path.join(str(Path(self.model).parent), "input_0.pb")
        # if the download included input_*.pb files, load them:
        if os.path.exists(input_path):
            return self.load_inputs(str(Path(self.model).parent))

        # otherwise do default input generation
        return get_sample_inputs_for_onnx_model(
            self.model, self.dim_param_dict, self.input_name_to_shape_map
        )

    def construct_model(self):
        # Look in the test-run dir for the model file.
        # If it does not exist, pull it in from the model's Github URL, and try again.
        # final_model_path should look like this: <directory>/<test/model_name>/<test/model_name>.onnx
        model_dir = str(Path(self.model).parent)

        # search for a .onnx file in the ./test-run/testname/ dir
        found_models = []
        for root, _, files in os.walk(model_dir):
            for name in files:
                if name.endswith(".onnx"):
                    found_models.append(os.path.abspath(os.path.join(root, name)))

        # one model in test-dir: return
        if len(found_models) == 1:
            self.model = found_models[0]
            return

        # multiple models in test-dir: pick one
        if len(found_models) > 1:
            print(f"Found multiple model.onnx files: {found_models}")
            print(f"Picking the first model found to use: {found_models[0]}")
            self.model = found_models[0]
            return

        cache_file = os.path.join(
            self.cache_dir,
            self.model_url.split("/")[-1] if self.is_validated else "model.onnx",
        )

        # no model and cache doesn't exist: download files to cache
        if not os.path.exists(cache_file):
            print(f"Begin download for {self.name}")
            content = requests.get(self.model_url, stream=True).content
            if content is None or len(content) == 0:
                raise ValueError(
                    f"Contents downloaded from {self.model_url} are invalid. This is likely an invalid URL."
                )
            with open(cache_file, "wb") as model_out_file:
                model_out_file.write(content)

        if self.is_validated:
            self.unzip_model_archive(cache_file)
        else:
            os.symlink(cache_file, self.model)

        if not os.path.exists(self.model):
            raise OSError(
                f"No onnx model could be found, downloaded, or extracted to {model_dir}"
            )


class AzureDownloadableModel(OnnxModelInfo):
    """This class can be used for models in our azure storage (both private and public)."""

    def __init__(self, name: str, onnx_model_path: str):
        # TODO: Extract opset version from onnx.version.opset
        opset_version = 21
        parent_cache_dir = os.getenv("CACHE_DIR")
        if not parent_cache_dir:
            raise RuntimeError(
                "Please specify a cache directory path in the CACHE_DIR environment variable for storing large model files."
            )
        self.cache_dir = os.path.join(parent_cache_dir, name)
        super().__init__(name, onnx_model_path, opset_version)

    def update_extra_options(self):
        # Default to using opset version 21 for all Azure models.
        self.extra_options = ExtraOptions(
            import_model_options=ImporterOptions(opset_version=21)
        )

    def update_sess_options(self):
        self.sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

    def construct_model(self):
        # try to find a .onnx file in the test-run dir
        # if that fails, check for zip file in cache
        # if that fails, try to download and setup from azure, then search again for a .onnx file

        # TODO: make the zip file structure more uniform so we don't need to search for extracted files
        model_dir = str(Path(self.model).parent)

        def find_models(model_dir):
            # search for a .onnx file in the ./test-run/testname/ dir
            found_models = []
            for root, dirs, files in os.walk(model_dir):
                for name in files:
                    if name[-5:] == ".onnx":
                        found_models.append(os.path.abspath(os.path.join(root, name)))
            return found_models

        found_models = find_models(model_dir)

        if len(found_models) == 0:
            azutils.pre_test_onnx_model_azure_download(
                self.name, self.cache_dir, self.model
            )
            found_models = find_models(model_dir)
        if len(found_models) == 1:
            self.model = found_models[0]
            return
        if len(found_models) > 1:
            print(f"Found multiple model.onnx files: {found_models}")
            print(f"Picking the first model found to use: {found_models[0]}")
            self.model = found_models[0]
            return
        raise OSError(
            f"No onnx model could be found, downloaded, or extracted to {model_dir}"
        )


class SiblingModel(OnnxModelInfo):
    """convenience class for re-using an onnx model from another 'sibling' test"""

    def __init__(
        self,
        og_model_info_class: type,
        og_name: str,
        name,
        onnx_model_path,
        opset_version=None,
    ):
        # additionally store an instance of the sibling test
        run_dir = Path(onnx_model_path).parent
        og_model_path = os.path.join(run_dir, og_name)
        self.sibling_inst = og_model_info_class(og_name, og_model_path)
        self.opset_version = self.sibling_inst.opset_version
        super().__init__(name, onnx_model_path, opset_version)

    def construct_model(self):
        if not os.path.exists(self.sibling_inst.model):
            self.sibling_inst.construct_model()
        self.model = self.sibling_inst.model

    def update_dim_param_dict(self):
        self.sibling_inst.update_dim_param_dict()
        self.dim_param_dict = self.sibling_inst.dim_param_dict


def get_sibling_constructor(sibling_class, og_constructor, og_name):
    """Returns a constructor for the sibling class. Useful for convenient registration.

    Usage:

    class OGModelInfoClass:
        ...

    register_test(OGModelInfoClass, og_name)

    class NewSiblingModel(SiblingModel):
        ...

    sibling_constructor = get_sibling_constructor(NewSiblingModel, OGModelInfoClass, og_name)
    register_test(sibling_constructor, new_name)

    """
    return lambda *args, **kwargs: sibling_class(
        og_constructor, og_name, *args, **kwargs
    )


class TruncatedModel(SiblingModel):
    """This class will take the model.onnx from another test, and modify the output.

    Takes additional __init__ args: n (int) and op_type (str)
    If op_type = "", n will determine the position backwards from the original output node.
    If op_type isn't a null string, then n will be used to determine which node of that op_type will be returned as an output.

    Examples:
    op_type = "Conv" and n=2: This will set the output of the onnx model to the ouptput of the third Conv node in the graph.
    op_type = "" and n=2: This will set the output of the model to the second-to-last node before the original output.
    """

    def __init__(self, n: int, op_type: str, *args, **kwargs):
        self.n = n
        self.op_type = op_type
        super().__init__(*args, **kwargs)

    def construct_model(self):
        if not os.path.exists(self.sibling_inst.model):
            self.sibling_inst.construct_model()
        og_model = onnx.load(self.sibling_inst.model)
        inf_model = onnx.shape_inference.infer_shapes(og_model, data_prop=True)
        output_node = (
            -self.n
            if self.op_type == ""
            else find_node(inf_model, self.n, self.op_type)
        )
        new_model = modify_model_output(inf_model, output_node)
        onnx.save(new_model, self.model)
        from e2e_testing.onnx_utils import get_op_frequency

        print(get_op_frequency(self.model))


def get_truncated_constructor(truncated_class, og_constructor, og_name):
    """returns a function that takes in (n, op_type) and returns a constructor for the truncated class.

    Usage:

    class OGModelInfoClass:
        ...

    register_test(OGModelInfoClass, og_name)

    class NewTruncatedModel(TruncatedModel):
        ...

    truncated_constructor = get_truncated_constructor(NewTruncatedModel, OGModelInfoClass, og_name)
    register_test(truncated_constructor(2, "Conv"), og_name + "_2_Conv")
    register_test(truncated_constructor(5, ""), og_name + "_5")

    """
    return lambda n, op_type: (
        lambda *args, **kwargs: truncated_class(
            n, op_type, og_constructor, og_name, *args, **kwargs
        )
    )


class BuildAModel(OnnxModelInfo):
    """Convenience class for building an onnx model from scratch.
    If inheriting from this class:
    1. override construct_nodes(self) to add to the self.node_list. The get_app_node method may be helpful.
    2. override construct_i_o_value_info to add the input and output value infos to the lists self.input_vi and self.output_vi.
    3. optionally override other OnnxModelInfo methods if desired, e.g. construct_inputs.

    Example:

    class QuantizedRelu(BuildAModel):
        def construct_nodes(self):
            app_node = self.get_app_node()

            ST = make_tensor("ST", TensorProto.FLOAT, [], [0.025])
            ZPT = make_tensor("ZPT", TensorProto.INT8, [], [3])

            app_node("Constant", [], ["S"], value=ST)
            app_node("Constant", [], ["ZP"], value=ZPT)
            app_node("QuantizeLinear", ["X", "S", "ZP"], ["QX"])
            app_node("DequantizeLinear", ["QX", "S", "ZP"], ["DQX"])
            app_node("Relu", ["DQX"], ["Y"])
            app_node("QuantizeLinear", ["Y", "S", "ZP"], ["QY"])
            app_node("DequantizeLinear", ["QY", "S", "ZP"], ["DQY"])

        def construct_i_o_value_info(self):
            self.input_vi.append(make_tensor_value_info("X", TensorProto.FLOAT, [1,2,4]))
            self.output_vi.append(make_tensor_value_info("DQY", TensorProto.FLOAT, [1,2,4]))

    register_test(QuantizedRelu, "quantized_relu")

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # always construct the onnx model for these tests.
        # Useful for trial-error building of the test without having to delete the old onnx model.
        self.node_list = []
        self.input_vi = []
        self.output_vi = []
        self.initializers = None
        self.construct_model()

    def construct_nodes(self):
        """Needs to be overriden. Update self.node_list here with the nodes you want in your graph."""
        raise NotImplementedError("Please implement a construct_nodes method.")

    def construct_i_o_value_info(self):
        """Needs to be overridden. Update self.input_vi and self.output_vi with the lists of input and output value infos"""
        raise NotImplementedError("Please implement a construct_i_o_value_info method.")

    def construct_initializers(self):
        """Can be overridden. Use to define self.initializers as a list of TensorProtos."""
        pass

    def get_app_node(self):
        """Convenience function for defining a lambda that appends a new node to self.node_list"""
        return lambda op_type, inputs, outputs, **kwargs: self.node_list.append(
            make_node(op_type, inputs, outputs, **kwargs)
        )

    def construct_model(self):
        self.construct_nodes()
        self.construct_i_o_value_info()
        self.construct_initializers()
        graph = make_graph(
            self.node_list, "main", self.input_vi, self.output_vi, self.initializers
        )
        model = make_model(graph)
        onnx.save(model, self.model)
