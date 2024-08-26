## Summary

|Stage|Count|
|--|--|
| Total | 18 |
| PASS | 14 |
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
Namespace(sources=['reports/test_add.json', 'reports/test_conv.json'], output='reports/test_conv_and_add.json', report=True, report_file='reports/test_conv_and_add.md')

| Test | Exit Status | Notes |
|--|--|--|
| test_add | PASS | |
| test_add_bcast | PASS | |
| test_add_uint8 | PASS | |
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
