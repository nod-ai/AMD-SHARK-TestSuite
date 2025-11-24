# Tanke from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-84
import numpy, torch, sys
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_node, check_graph, check_model

import numpy as np
import onnx

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2EAMDSHARK_CHECK_DEF

# Create an instance of it for this test
E2EAMDSHARK_CHECK = dict(E2EAMDSHARK_CHECK_DEF)



# LOOP 
# Given a tensor x of values [x1, ..., xN], and initial tensor y
# sum up its elements using a scan
# returning the final state (y+x1+x2+...+xN) as well the scan_output
# [y+x1, y+x1+x2, ..., y+x1+x2+...+xN]

y_in = onnx.helper.make_tensor_value_info("y_in", onnx.TensorProto.FLOAT, [1])
y_out = onnx.helper.make_tensor_value_info("y_out", onnx.TensorProto.FLOAT, [1])
scan_out = onnx.helper.make_tensor_value_info(
    "scan_out", onnx.TensorProto.FLOAT, [1]
)
cond_in = onnx.helper.make_tensor_value_info(
    "cond_in", onnx.TensorProto.BOOL, []
)
cond_out = onnx.helper.make_tensor_value_info(
    "cond_out", onnx.TensorProto.BOOL, []
)
iter_count = onnx.helper.make_tensor_value_info(
    "iter_count", onnx.TensorProto.INT64, []
)

x_inp = np.array([1, 2, 3, 4, 5]).astype(np.float32)
y_inp = np.array([-2]).astype(np.float32)

cond = make_tensor_value_info('cond', TensorProto.BOOL, [])
y = make_tensor_value_info('y', TensorProto.FLOAT, [1])
res_y = make_tensor_value_info('res_y', TensorProto.FLOAT, [1])
res_scan = make_tensor_value_info('res_scan', TensorProto.FLOAT, [5,1])
trip_count = make_tensor_value_info('trip_count', TensorProto.INT64, [])

x_const_node = onnx.helper.make_node(
    "Constant",
    inputs=[],
    outputs=["x"],
    value=onnx.helper.make_tensor(
        name="const_tensor_x",
        data_type=onnx.TensorProto.FLOAT,
        dims=x_inp.shape,
        vals=x_inp.flatten().astype(float),
    ),
)

one_const_node = onnx.helper.make_node(
    "Constant",
    inputs=[],
    outputs=["one"],
    value=onnx.helper.make_tensor(
        name="const_tensor_one",
        data_type=onnx.TensorProto.INT64,
        dims=(),
        vals=[1],
    ),
)

axes_node = onnx.helper.make_node(
    "Constant",
    inputs=[],
    outputs=["axes"],
    value=onnx.helper.make_tensor(
        name="const_tensor_axes",
        data_type=onnx.TensorProto.INT64,
        dims=(),
        vals=[0],
    ),
)

i_add_node = onnx.helper.make_node(
    "Add", inputs=["iter_count", "one"], outputs=["end"]
)

start_unsqueeze_node = onnx.helper.make_node(
    "Unsqueeze", inputs=["iter_count", "axes"], outputs=["slice_start"]
)

end_unsqueeze_node = onnx.helper.make_node(
    "Unsqueeze", inputs=["end", "axes"], outputs=["slice_end"]
)

slice_node = onnx.helper.make_node(
    "Slice", inputs=["x", "slice_start", "slice_end"], outputs=["slice_out"]
)

y_add_node = onnx.helper.make_node(
    "Add", inputs=["y_in", "slice_out"], outputs=["y_out"]
)

identity_node = onnx.helper.make_node(
    "Identity", inputs=["cond_in"], outputs=["cond_out"]
)

scan_identity_node = onnx.helper.make_node(
    "Identity", inputs=["y_out"], outputs=["scan_out"]
)

loop_body_graph = onnx.helper.make_graph(
    [
        identity_node,
        x_const_node,
        one_const_node,
        i_add_node,
        axes_node,
        start_unsqueeze_node,
        end_unsqueeze_node,
        slice_node,
        y_add_node,
        # scan_identity_node,
    ],
    "loop_body",
    [iter_count, cond_in, y_in],
    [cond_out, y_out],
)

loop_node = onnx.helper.make_node(
    "Loop",
    inputs=["trip_count", "cond", "y"],
    outputs=["res_y"],
    body=loop_body_graph,
)

graph = onnx.helper.make_graph(
    nodes = [
        loop_node
    ],
    name="loop_example",
    inputs=[trip_count, cond, y],
    outputs=[res_y],
)

onnx_model = make_model(graph, producer_name='loop_example')
onnx_model.opset_import[0].version = 13


# Save the model
# save_model(onnx_model, "model.onnx")
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = onnxruntime.InferenceSession("model.onnx", None)
inputs = session.get_inputs()
outputs = session.get_outputs()

trip_count_inp = np.array(5).astype(np.int64)
cond_inp = np.array(1).astype(bool)

input_dict = {
    inputs[0].name : trip_count_inp, 
    inputs[1].name : cond_inp,
    inputs[2].name : y_inp,
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
    torch.from_numpy(input_dict["trip_count"]),
    torch.from_numpy(input_dict["cond"]),
    torch.from_numpy(input_dict["y"]),
]
E2EAMDSHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2EAMDSHARK_CHECK["input"])
print("Output:", E2EAMDSHARK_CHECK["output"])
