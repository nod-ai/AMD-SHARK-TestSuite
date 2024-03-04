module {
  func.func @test_scan_sum(%arg0: !torch.vtensor<[1,2],f32>, %arg1: !torch.vtensor<[1,3,2],f32>) -> (!torch.vtensor<[1,2],f32>, !torch.vtensor<[1,3,2],f32>) attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 8 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:2 = torch.operator "onnx.Scan"(%none, %arg0, %arg1) {torch.onnx.num_scan_inputs = 1 : si64} : (!torch.none, !torch.vtensor<[1,2],f32>, !torch.vtensor<[1,3,2],f32>) -> (!torch.vtensor<[1,2],f32>, !torch.vtensor<[1,3,2],f32>) {
    ^bb0(%arg2: !torch.vtensor<[2],f32>, %arg3: !torch.vtensor<[2],f32>):
      %1 = torch.operator "onnx.Add"(%arg2, %arg3) : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> 
      %2 = torch.operator "onnx.Identity"(%1) : (!torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> 
      torch.operator_terminator %1, %2 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>
    }
    return %0#0, %0#1 : !torch.vtensor<[1,2],f32>, !torch.vtensor<[1,3,2],f32>
  }
}

