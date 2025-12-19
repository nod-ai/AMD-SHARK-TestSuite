from ..helper_classes import BuildAModel
from onnx import TensorProto
from onnx.helper import make_tensor, make_tensor_value_info
from e2e_testing.registry import register_test


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
        self.input_vi.append(make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 4]))
        self.output_vi.append(
            make_tensor_value_info("DQY", TensorProto.FLOAT, [1, 2, 4])
        )


register_test(QuantizedRelu, "quantized_relu")
