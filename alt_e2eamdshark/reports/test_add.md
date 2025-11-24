## Summary

|Stage|Count|
|--|--|
| Total | 3 |
| PASS | 3 |
| Numerics | 0 |
| results-summary | 0 |
| postprocessing | 0 |
| compiled_inference | 0 |
| native_inference | 0 |
| construct_inputs | 0 |
| compilation | 0 |
| preprocessing | 0 |
| import_model | 0 |
| setup | 0 |

## Test Run Detail 
Test was run with the following arguments:
Namespace(device='local-task', backend='llvm-cpu', iree_compile_args=None, mode='onnx-iree', torchtolinalg=False, stages=None, skip_stages=None, load_inputs=False, groups='all', test_filter='test_add', tolerance=None, verbose=True, rundirectory='test-run', no_artifacts=False, report=True, report_file='reports/test_add.md')

| Test | Exit Status | Notes |
|--|--|--|
| test_add | PASS | |
| test_add_bcast | PASS | |
| test_add_uint8 | PASS | |
