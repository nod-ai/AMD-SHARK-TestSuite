import numpy
import onnx
import onnxruntime
import torch
from e2e_testing.storage import TestTensors
from typing import Optional
from pathlib import Path


def dtype_from_ort_node(node):
    '''infers a torch dtype from an ort node type of the form "tensor(dtype)"'''
    typestr = node.type
    if typestr[0:6] != "tensor":
        raise TypeError(f"node: {node} has invalid typestr {typestr}")
    dtypestr = typestr[7:-1]
    if dtypestr == "float":
        return torch.float
    if dtypestr == "float16":
        return torch.float16
    if dtypestr == "bfloat16":
        return torch.bfloat16
    if dtypestr == "int" or dtypestr == "int32":
        return torch.int32
    if dtypestr == "int64":
        return torch.int64
    if dtypestr == "int8":
        return torch.int8
    if dtypestr == "uint8":
        return torch.uint8
    if dtypestr == "bool":
        return torch.bool
    raise NotImplementedError(f"Unhandled dtype string found: {dtypestr}")

def get_node_shape_from_dim_param_dict(node: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg, dim_param_dict: Optional[dict[str, int]] = None, input_name_to_shape_map: Optional[dict[str, list[int]]] = None) -> list[int]:
    """Get the shape of a node, replacing any string dims with values from a dim_param_dict"""
    if input_name_to_shape_map and node.name in input_name_to_shape_map.keys():
        # Particularly used for models from ONNX Model Zoo,
        # the dim_param_dict here should contain the shapes
        # of the input nodes.
        return input_name_to_shape_map[node.name]

    int_dims = []
    for dim in node.shape:
        if isinstance(dim, str) and dim_param_dict:
            if not dim in dim_param_dict.keys():
                raise ValueError(f"input node {node.name} has a dim param='{dim}' not found in provided dim_param_dict: '{dim_param_dict}'")
            else:
                int_dims.append(dim_param_dict[dim])
                continue
        if not isinstance(dim, int):
            raise TypeError(
                f"input node '{node.name}' has dims={node.shape}. Node dim '{dim}' has invalid type: {type(dim)}\nexpected type: int.\nIf your model has dim_params, consider setting a self.dim_param_dict for this test. See: https://github.com/nod-ai/amdshark-TestSuite/blob/63f848a42a3e5e01d6c73de142ff182fb6f6e2d2/alt_e2eamdshark/onnx_tests/models/migraphx.py#L136"
            )
        if dim <= 0:
            raise ValueError(
                f"input node '{node.name}' has a non-positive dim: {dim}. Consider setting cutsom inputs for this test."
            )
        int_dims.append(dim)
    return int_dims


def generate_input_from_node(node: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg, dim_param_dict: Optional[dict[str, int]] = None, input_name_to_shape_map: Optional[dict[str, list[int]]] = None):
    """
    Generate a random input tensor for a given node

    Args:
        node: an onnx node
        dim_param_dict: a dictionary mapping onnx string dims to int values
    """


    int_dims = get_node_shape_from_dim_param_dict(node, dim_param_dict, input_name_to_shape_map)

    rng = numpy.random.default_rng(19)
    if node.type == "tensor(float)":
        return rng.random(int_dims).astype(numpy.float32)
    if node.type == "tensor(float16)":
        return rng.random(int_dims).astype(numpy.float16)
    if node.type == "tensor(bfloat16)":
        raise NotImplementedError("Numpy doesn't support bfloat16. Please consider modifying the boundary types.")
    if node.type == "tensor(int)" or node.type == "tensor(int32)":
        return rng.integers(0, 10000, size=int_dims, dtype=numpy.int32)
    if node.type == "tensor(int8)":
        return rng.integers(-127, 128, size=int_dims, dtype=numpy.int8)
    if node.type == "tensor(int64)":
        return rng.integers(0, 5, size=int_dims, dtype=numpy.int64)
    if node.type == "tensor(bool)":
        return rng.integers(0, 2, size=int_dims, dtype=bool)
    raise NotImplementedError(f"Found an unhandled dtype: {node.type}.")


def get_sample_inputs_for_onnx_model(model_path, dim_param_dict = None, input_name_to_shape_map = None) -> TestTensors:
    """A convenience function for generating sample inputs for an onnx model"""
    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = onnxruntime.InferenceSession(model_path, opt)
    inputs = s.get_inputs()
    sample_inputs = TestTensors(
        tuple([generate_input_from_node(node, dim_param_dict, input_name_to_shape_map) for node in inputs])
    )
    return sample_inputs


def get_signature_for_onnx_model(model_path, *, from_inputs: bool = True, dim_param_dict: Optional[dict[str, int]] = None, leave_dynamic: bool = False):
    """A convenience funtion for retrieving the input or output shapes and dtypes"""
    sess_opt = onnxruntime.SessionOptions()
    sess_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = onnxruntime.InferenceSession(model_path, sess_opt)
    if from_inputs:
        nodes = s.get_inputs()
    else:  # taking from outputs
        nodes = s.get_outputs()
    shapes = []
    dtypes = []
    for i in nodes:
        shape = i.shape
        for index, s in enumerate(shape):
            if dim_param_dict and not leave_dynamic and isinstance(s, str) and s in dim_param_dict.keys():
                shape[index] = dim_param_dict[s]
        shapes.append(shape)
        dtypes.append(dtype_from_ort_node(i))
    return shapes, dtypes


def get_op_frequency(model_or_path):
    if isinstance(model_or_path, str) or isinstance(model_or_path, Path):
        model = onnx.load(model_or_path)
    elif isinstance(model_or_path, onnx.ModelProto):
        model = model_or_path
    else:
        raise TypeError(f'Input argument must be a path, string, or onnx model.')
    op_freq = dict()
    for n in model.graph.node:
        if n.op_type in op_freq:
            op_freq[n.op_type] += 1
        else:
            op_freq[n.op_type] = 1
    return op_freq


def modify_model_output(model: onnx.ModelProto, final_node_key: int) -> onnx.ModelProto:
    """A helper function to change the output of an onnx model to a new output."""

    if final_node_key < 0:
        final_node_key += len(model.graph.node)

    final_node = model.graph.node[final_node_key]

    # clear old outputs
    n = len(model.graph.output)
    for _ in range(n):
        model.graph.output.pop()

    # add new outputs
    for vi in model.graph.value_info:
        if vi.name in final_node.output:
            model.graph.output.append(vi)

    # remove nodes after the final output
    for _ in range(final_node_key + 1, len(model.graph.node)):
        model.graph.node.pop()

    # remove unused nodes, inputs, value_info, and initializers before final output
    keep_node_names, keep_vi_names = find_minimal_graph(model.graph, final_node_key)

    def remove_unused(attr: str, keep_list):
        i = 0
        while i < len(model.graph.__getattribute__(attr)):
            obj = model.graph.__getattribute__(attr)[i]
            if obj.name not in keep_list:
                model.graph.__getattribute__(attr).pop(i)
            else:
                i += 1

    remove_unused("node", keep_node_names)
    remove_unused("input", keep_vi_names)
    remove_unused("value_info", keep_vi_names)
    remove_unused("initializer", keep_vi_names)
    return model


def find_minimal_graph(graph: onnx.GraphProto, top_key: int):
    keep_vi_names = set()
    keep_vi_names.update(set(graph.node[top_key].output))
    keep_names = set()
    i = top_key
    while i >= 0:
        node = graph.node[i]
        if node.name == '':
            node.name = f'node_{i}'
        if len(set(node.output).intersection(keep_vi_names)) != 0:
            keep_names.add(node.name)
            keep_vi_names.update(set(node.input))
        i -= 1

    return keep_names, keep_vi_names


def find_node(model: onnx.ModelProto, n: int, op_name: str) -> onnx.NodeProto:
    """returns the output names for the nth node in the onnx model with op_type given by op_name"""
    op_freq = get_op_frequency(model)
    N = op_freq[op_name]
    if n > N-1 or n < -N:
        raise ValueError(f"There are {N} nodes with op name {op_name} in model. Provided index {n} is OOB.\n{op_freq}")
    if n < 0:
        n += N
    match_counter = 0
    key = -1
    for nde in model.graph.node:
        key += 1
        if nde.op_type != op_name:
            continue
        if match_counter == n:
            node = nde
            break
        match_counter += 1
    return key


def update_no_ext(onnx_model_path: str, opset_version: int):
    """Models larger than 2GB do not allow updating opset version, so we update opset version without loading external data. Unfortunately, all external data references get wiped, and we need to manually trace through the graph and copy them back in."""
    model = onnx.load(onnx_model_path, load_external_data=False)
    new_model = onnx.version_converter.convert_version(
        model, opset_version
    )
    nv_init_map = {init.name: init for init in model.graph.initializer}
    for new_init in new_model.graph.initializer:
        if new_init.name not in nv_init_map.keys():
            continue
        old_init = nv_init_map[new_init.name]
        if old_init.data_location:
            new_init.Clear()
            new_init.CopyFrom(old_init)
    nv_node_map = {node.output[0]: node for node in model.graph.node}
    for node in new_model.graph.node:
        name = node.output[0]
        if name not in nv_node_map.keys():
            continue
        og_node = nv_node_map[name]
        attrs_to_replace = dict()
        for attr in og_node.attribute:
            if attr.t.external_data:
                attrs_to_replace[attr.name] = attr
        for new_attr in node.attribute:
            if new_attr.name in attrs_to_replace.keys():
                old_attr = attrs_to_replace[new_attr.name]
                new_attr.Clear()
                new_attr.CopyFrom(old_attr)

    onnx.save(new_model, onnx_model_path)

