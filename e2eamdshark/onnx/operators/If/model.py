# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing add
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy, torch, sys
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2EAMDSHARK_CHECK_DEF

# Create an instance of it for this test
E2EAMDSHARK_CHECK = dict(E2EAMDSHARK_CHECK_DEF)


# condition has to be a float tensor
condition = make_tensor_value_info('condition', TensorProto.BOOL, [1])
input1 = make_tensor_value_info('input1', TensorProto.FLOAT, [2,3])
input2 = make_tensor_value_info('input2', TensorProto.FLOAT, [2,3])
output = make_tensor_value_info('output', TensorProto.FLOAT, [2,3])
output_then = make_tensor_value_info('output_then', TensorProto.FLOAT, [2,3])
output_else = make_tensor_value_info('output_else', TensorProto.FLOAT, [2,3])

then_branch = make_graph(
    nodes=[
        make_node('Add', ['input1', 'input2'], ['output_then'])
    ],
    name='then_branch',
    inputs=[],
    outputs=[output_then]
)

else_branch = make_graph(
    nodes=[
        make_node('Sub', ['input1', 'input2'], ['output_else'])
    ],
    name='else_branch',
    inputs=[],
    outputs=[output_else]
)

graph = make_graph(
    nodes=[
        make_node('If', ['condition'], ['output'], then_branch=then_branch, else_branch=else_branch)
    ],
    name='if_example',
    inputs=[condition, input1, input2],
    outputs=[output]
)

onnx_model = make_model(graph, producer_name='conditional_example')

onnx_model.opset_import[0].version = 19

# Save the model
# save_model(onnx_model, "model.onnx")
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())


session = onnxruntime.InferenceSession("model.onnx", None)
# gets X in inputs[0] and Y in inputs[1]
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()

def generate_input_from_node(node: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg):
    if node.type == "tensor(float)":
        return numpy.random.randn(*node.shape).astype(numpy.float32)
    if node.type == "tensor(int)":
        return numpy.random.randint(0, 10000, size=node.shape).astype(numpy.int32)
    if node.type == "tensor(bool)":
        return numpy.random.randint(0, 2, size=node.shape).astype(bool)
    
input_dict = {
    node.name: generate_input_from_node(node)
    for node in inputs
}

output_list = [
    node.name
    for node in outputs
]

model_output = session.run(
    output_list,
    input_dict,
)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2EAMDSHARK_CHECK["input"] = [
    torch.from_numpy(input_dict["condition"]),
    torch.from_numpy(input_dict["input1"]),
    torch.from_numpy(input_dict["input2"]),
]
E2EAMDSHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2EAMDSHARK_CHECK["input"])
print("Output:", E2EAMDSHARK_CHECK["output"])
