## Adding a new testing configuration

A configuration controls how to compile and run a model. The default configuration is to import and compile an onnx model with IREE, then run the compiled model with IREE runtime.

To add a new configuration, you will need to define a class `NewConfig` in `alt_e2eshark/e2e_testing/configs/` which inherits from `TestConfig`. In order for this new configuration to work properly, you will need to override the following methods:

- `import_model` : convert the model from it's framework to a config-specific format (e.g., custom IR).
- `preprocess_model` : (optional) apply some pre-processing instructions (e.g., IR passes)
- `compile` : compile the imported and preprocessed model to a compiled artifact
- `run` : apply the compiled artifact to some inputs

The last two methods are usually handled by a `backend` class, which defines how to manage the compilation and running of compiled artifacts.