 # e2eshark framework-to-iree-to-inference tests

 This test suite enables developers to add small (operator level) to large (full model)
 end-2-end tests that compare output of running a model in a Framework 
 (e.g. Pytorch, ONNX) to the output of running the IREE-compiled artefact of 
 the same model on a target backend (e.g. CPU, AIE). If the difference in outputs
 is within a tolerable limit, then the test is reported as have passed, else the 
 test is reported as have failed. In case of a failing test, the stage of the 
 failure is reported. 

 The test suite is organized starting with a framework name: pytorch, tensorflow, onnx. 
 For each framework category, multiple modes are tested. 

 - pytorch : starting model is a pytorch model (planned for later)
 - tensorflow : starting model is a tensorflow model (planned for later)
 - onnx : starting model is an onnx model generated using onnx python API or an existing onnx model
 
 The target backend can be any IREE supported backend: llvm-cpu, amd-aie etc.

## Contents
 The contents are as below.
 - e2e_testing/azutils.py : util functions for interfacing with azure
 - e2e_testing/backends.py : where test backends are defined. Add other backends here.
 - e2e_testing/framework.py : contains two types of classes: framework-specific base classes for storing model info, and generic classes for testing infrastructure.
 - e2e_testing/onnx_utils.py : onnx related util functions. These either infer information from an onnx model or modify an onnx model.
 - e2e_testing/registry.py : this contains the GLOBAL_TEST_REGISTRY, which gets updated when importing files with instances of `register_test(TestInfoClass, 'testname')`.
 - e2e_testing/storage.py : contains helper functions and classes for managing the storage of tensors.
 - e2e_testing/test_configs/onnxconfig.py : defines the onnx frontend test config. Other configs (e.g. pytorch, tensorflow) should be created in sibling files.
 - onnx_tests/ : contains files that define OnnxModelInfo child classes, which customize model/input generation for various kinds of tests. Individual tests are also registered here together with their corresponding OnnxModelInfo child class.
 - dev_requirements.txt : `pip install -r dev_requirements.txt` to install additional packages if you are using local builds of torch-mlir and iree.
 - requirements.txt : `pip install -r requirements.txt` to install packages for getting started immediately. This is mostly useful if you aren't trying to test local builds of IREE or torch-mlir.
 - run.py : Run `python run.py --help` to learn about the script. This is the script to run tests.
 
 The logs are created as .log files in the test-run sub directory. Examine the logs to find and fix 
 cause of any failure. You can specify -r 'your dir name' to the run.py to name your test run directory 
 as per your choice. The default name for the run directory is 'test-run'.

 Note that, you will be required to pass --cachedir argument to the run.py to point to a directory where 
 model weights etc. from external model serving repositories such as from Torch Vision, Hugging Face etc.
 will be downloaded. The downloaded data can be large, so set it to other than your home, 
 preferably with 100 GB or more free space.

## Setting up (Quick Start)

By default, a nightly build of torch_mlir and IREE is installed when you run:

```bash
python -m venv test_suite.venv /
source test_suite.venv/bin/activate /
pip install --upgrade pip /
pip install -r ./requirements.txt
```

Therefore, you are not required to have a local build of either torch mlir or iree.

## Setting up (using local build of torch-mlir or iree)

If you want to use a custom build of torch-mlir or iree, you need to build those projects with python bindings enabled. 

If you already installed `requirements.txt` to your venv, you can uninstall whatever package you want to replace, then activate the appropriate `.env` file for the project you want to use. For example,

```bash
# if starting with dev_requirements, this line is uneccessary:
pip uninstall iree-compiler iree-runtime
# set up python to find iree compiler and iree runtime
source /path/to/iree-build/.env && export PYTHONPATH
```

If you installed `dev_requirements.txt`, you won't need to uninstall iree-compiler, iree-runtime, or torch-mlir, since these aren't included there.

Unfortunately, the `.env` files in torch-mlir and iree completely replace the pythonpath instead of adding to it. So if you want to use a local build of both torch-mlir and iree, you could do something like:

```bash
export IREE_BUILD_DIR="<path to iree build dir>"
export TORCH_MLIR_BUILD_DIR="<path to torch-mlir build dir>"
source ${IREE_BUILD_DIR}/.env && export PYTHONPATH="${TORCH_MLIR_BUILD_DIR}/tools/torch-mlir/python_packages/torch_mlir/:${PYTHONPATH}"
```

If you are just a torch-mlir developer and don't want a custom IREE build, you can either make an `.env` file for torch-mlir with `torch-mlir/build_tools/write_env_file.sh`and use that to set your python path, or just use:

```bash
export PYTHONPATH="${TORCH_MLIR_BUILD_DIR}/tools/torch-mlir/python_packages/torch_mlir/"
```

## Adding a test

For onnx framework tests, you add a test in one of the model.py files contained in '/e2eshark/onnx_tests/'.

The OnnxModelInfo class simply requires that you define a function called "construct_model", which should define how to build the model.onnx file (be sure that the model.onnx file gets saved to your class' self.model, which should store the filepath to the model). 

We provide a conveninece function for generating inputs by default, but to override this for an individual test, you can redefine "construct_inputs" for your test class. 

Once a test class is generated, register the test with the test suite with:

```python
register_test(YourTestClassName,"name_of_test")
```

## Running a test

Here is an example of running the test we made in the previous section with some commonly used flags:

```bash
python run.py --torchtolinalg --cachedir="../cache_dir" -t name_of_test
```

This will generate a new folder './test-run/name_of_test/' which contains some artifacts generated during the test. These artifacts can be used to run command line scripts to debug various failures. 



