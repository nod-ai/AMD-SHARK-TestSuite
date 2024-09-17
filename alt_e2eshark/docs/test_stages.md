## Test stages in run.py

The test runner performs tests in stages. If an error is thrown during a particular stage, then a `<curr_stage>.log` is generated in the `test-run/curr_test/` directory. 

### setup

This stage makes an instance of the test info class for a particular test. It currently calls `inst.construct_model`, but this may need to be modified if other frontends are to be added (e.g. pytorch, tf).

### import_model

Calls `config.import_model`. For the default config, this uses the `onnx_importer` tool from torch MLIR to convert an onnx model to MLIR in the torch dialect. 

Also stores the module name for future reference.

### preprocess_model

This also calls a config method of the same name. In the default config, this does nothing. If the `run.py` flag `--torchtolinalg` is specified, then this stage applies some torch-mlir passes (lower to backend contract, torch backend to linalg on tensors backend pipeline) to the imported model before calling iree-compile.

### compilation

This calls a config method `config.compile`. In the default config, this compiles the MLIR file with IREE. 

### construct_inputs

Calls the test info class method `inst.construct_inputs`. A failure during this stage is due to a poorly constructed test info class. We choose to seperate this stage from `setup` so that the compilation flow can be tested even if the test info class is not fully fleshed out. 

### native_inference

Calls the test info class method `inst.forward`. For onnx model info, this runs the original onnx model through onnxruntime CPU provider to generate gold_outputs.

### compiled_inference

Calls the `config.run` method, applying the compiled artifact to the constructed inputs in order to generate test outputs. 

### postprocessing

Applies test info specified postprocessing to the gold_outputs and outputs. 

## After a test run

After a test is run, the gold_outputs and outputs are compared for numerical accuracy. If the test fails numerics, it's exit status will be recorded as "Numerics". Otherwise it will be recorded as "PASS". 

