## Summary

|Stage|Count|
|--|--|
| Total | 15 |
| PASS | 11 |
| Numerics | 0 |
| results-summary | 0 |
| postprocessing | 0 |
| compiled_inference | 0 |
| native_inference | 0 |
| construct_inputs | 0 |
| compilation | 4 |
| preprocessing | 0 |
| import_model | 0 |
| setup | 0 |

## Test Run Detail 
Test was run with the following arguments:
Namespace(device='local-task', backend='llvm-cpu', iree_compile_args=None, mode='onnx-iree', torchtolinalg=False, stages=None, skip_stages=None, load_inputs=False, groups='all', test_filter='test_conv', tolerance=None, verbose=True, rundirectory='test-run', no_artifacts=False, report=True, report_file='reports/test_conv.md')

| Test | Exit Status | Notes |
|--|--|--|
| test_conv_with_autopad_same | compilation | |
| test_conv_with_strides_and_asymmetric_padding | PASS | |
| test_conv_with_strides_no_padding | PASS | |
| test_conv_with_strides_padding | PASS | |
| test_convinteger_with_padding | PASS | |
| test_convinteger_without_padding | PASS | |
| test_convtranspose | PASS | |
| test_convtranspose_1d | PASS | |
| test_convtranspose_3d | PASS | |
| test_convtranspose_autopad_same | compilation | |
| test_convtranspose_dilations | PASS | |
| test_convtranspose_kernel_shape | compilation | |
| test_convtranspose_output_shape | compilation | |
| test_convtranspose_pad | PASS | |
| test_convtranspose_pads | PASS | |
