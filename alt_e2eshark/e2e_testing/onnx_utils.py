import numpy
import onnx
import onnxruntime
import torch
from e2e_testing.storage import TestTensors


def dtype_from_ort_node(node):
    '''infers a torch dtype from an ort node type of the form "tensor(dtype)"'''
    typestr = node.type
    if typestr[0:6] != "tensor":
        raise TypeError(f"node: {node} has invalid typestr {typestr}")
    dtypestr = typestr[7:-1]
    if dtypestr == "float":
        return torch.float
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


def generate_input_from_node(node: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg):
    """A convenience function for generating sample inputs for an onnxruntime node"""
    for dim in node.shape:
        if not isinstance(dim, int):
            raise TypeError(
                f"input node '{node.name}' has a dim='{dim}', with invalid type: {type(dim)}\nexpected type: int.\nIf your model has dim_params, consider fixing them or setting custom inputs for this test."
            )
        if dim <= 0:
            raise ValueError(
                f"input node '{node.name}' has a non-positive dim: {dim}. Consider setting cutsom inputs for this test."
            )
    rng = numpy.random.default_rng(19)
    if node.type == "tensor(float)":
        return rng.random(node.shape).astype(numpy.float32)
    if node.type == "tensor(int)" or node.type == "tensor(int32)":
        return rng.integers(0, 10000, size=node.shape, dtype=numpy.int32)
    if node.type == "tensor(int8)":
        return rng.integers(-127, 128, size=node.shape, dtype=numpy.int8)
    if node.type == "tensor(int64)":
        return rng.integers(0, 5, size=node.shape, dtype=numpy.int64)
    if node.type == "tensor(bool)":
        return rng.integers(0, 2, size=node.shape, dtype=bool)
    raise NotImplementedError(f"Found an unhandled dtype: {node.type}.")


def get_sample_inputs_for_onnx_model(model_path):
    """A convenience function for generating sample inputs for an onnx model"""
    s = onnxruntime.InferenceSession(model_path, None)
    inputs = s.get_inputs()
    sample_inputs = TestTensors(
        tuple([generate_input_from_node(node) for node in inputs])
    )
    return sample_inputs


def get_signature_for_onnx_model(model_path, *, from_inputs: bool = True):
    """A convenience funtion for retrieving the input or output shapes and dtypes"""
    s = onnxruntime.InferenceSession(model_path, None)
    if from_inputs:
        nodes = s.get_inputs()
    else:  # taking from outputs
        nodes = s.get_outputs()
    shapes = []
    dtypes = []
    for i in nodes:
        shapes.append(i.shape)
        dtypes.append(dtype_from_ort_node(i))
    return shapes, dtypes


def get_op_frequency(model_path):
    model = onnx.load(model_path)
    op_freq = dict()
    for n in model.graph.node:
        if n.op_type in op_freq:
            op_freq[n.op_type] += 1
        else:
            op_freq[n.op_type] = 1
    return op_freq


def modify_model_output(model: onnx.ModelProto, final_node_key: int) -> onnx.ModelProto:
    """A helper function to change the output of an onnx model to a new output."""

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
        if len(set(node.output).intersection(keep_vi_names)) != 0:
            keep_names.add(node.name)
            keep_vi_names.update(set(node.input))
        i -= 1

    return keep_names, keep_vi_names


def find_node(model: onnx.ModelProto, n: int, op_name: str) -> onnx.NodeProto:
    """returns the output names for the nth node in the onnx model with op_type given by op_name"""
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
    if not node:
        raise ValueError(f"Could not find {n} nodes of type {op_name} in {model}")
    return key
