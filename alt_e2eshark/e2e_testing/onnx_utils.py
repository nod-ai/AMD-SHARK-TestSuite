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


def modify_model_output(
    model: onnx.ModelProto, new_output_name: str
) -> onnx.ModelProto:
    """A helper function to change the output of an onnx model to a new output. Helpful to use with node_output_name"""
    if len(model.graph.output) != 1:
        raise NotImplementedError(
            "This function currently only supports modifying models with one output."
        )
    for vi in model.graph.value_info:
        if vi.name == new_output_name:
            model.graph.output.pop()
            model.graph.output.append(vi)
    return model


def node_output_name(model: onnx.ModelProto, n: int, op_name: str) -> str:
    """returns the output name for the nth node in the onnx model with op_type given by op_name"""
    counter = 0
    for nde in model.graph.node:
        if nde.op_type != op_name:
            continue
        if counter == n:
            node = nde
            break
        counter += 1
    if not node:
        raise ValueError(f"Could not find {n} nodes of type {op_name} in {model}")
    return node.output[0]
