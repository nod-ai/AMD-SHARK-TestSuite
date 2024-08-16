# IREE Tests

This directory contains generated test suites for running through IREE's
compiler and runtime tools.

Each test suite has one folder per test program containing a few files:

```
[program name 1]/
  model.mlir
  input_0.npy
  output_0.npy
  test_data_flags.txt
```

Where:

* `model.mlir` is in a format that is ready for use with `iree-compile`
  (e.g. torch-mlir, stablehlo, tosa, linalg)
* `input_0.npy` and `output_0.npy` files correspond to any number of program
  inputs and outputs for one test case
* `test_data_flags.txt` is a flagfile for use with
  `iree-run-module --flagfile=test_data_flags.txt` of the format:

  ```
  --input=@input_0.npy
  --expected_output=@output_0.npy
  ```

Testing follows several stages:

```mermaid
graph LR
  Import -. "\n(offline)" .-> Compile
  Compile --> Run
```

Importing is run "offline" and the outputs are checked in to the repository for
ease of use in downstream projects and by developers who prefer to work directly
with `.mlir` files and native (C/C++) tools. Each test suite or test case may
also have its own import logic, with all test suites converging onto the
standard format described above.

Some large files are stored using [Git LFS](https://git-lfs.com/). When working
with these files please ensure that you have Git LFS installed:

```bash
$ git lfs install
```

Files that are too large for Git LFS (e.g. model weights) are stored on cloud
providers. Download these files with
[`download_remote_files.py`](./download_remote_files.py):

```bash
# All files
$ python download_remote_files.py

# Just files for one subdirectory
$ python download_remote_files.py --root-dir pytorch/models/resnet50
```

## Running tests

Tests are run using the [pytest](https://docs.pytest.org/en/stable/) framework.

A [`conftest.py`](conftest.py) file collects test cases from subdirectories,
wrapping each directory matching the format described above to one test case
per test configuration. Test configurations are defined in JSON config files
like [`configs/cpu_llvm_sync.json`](./configs/cpu_llvm_sync.json).

### Common venv setup with deps

```bash
$ python -m venv .venv
$ source .venv/bin/activate
$ python -m pip install -r iree_tests/requirements.txt
```

To use `iree-compile` and `iree-run-module` from Python packages:

```bash
$ python -m pip install --find-links https://iree.dev/pip-release-links.html \
  iree-compiler iree-runtime --upgrade
```

To use local versions of `iree-compile` and `iree-run-module`, put them on your
`$PATH` ahead of your `.venv/Scripts` directory:

```bash
$ export PATH=path/to/iree-build:$PATH
```

### Invoking pytest

Run tests:

```bash
$ pytest iree_tests
```

Run tests with parallelism (using
[pytest-xdist](https://pypi.org/project/pytest-xdist/)):

```bash
$ pytest iree_tests -n auto
```

Run tests using custom config files:

```bash
$ pytest iree_tests --config-files ./iree_tests/configs/gpu_vulkan.json

# OR set an environment variable
$ export IREE_TEST_CONFIG_FILES=/iree/cpu_llvm_sync.json;/iree/gpu_vulkan.json
$ pytest iree_tests
```

Run tests on CPU and print all errors:

```bash
$ pytest iree_tests -n auto --ignore-xfails \
    --config-files ./iree_tests/configs/cpu_llvm_sync.json
```

Run compilation tests only and print all errors:

```bash
$ pytest iree_tests -n auto --ignore-xfails --skip-all-runs \
    --config-files ./iree_tests/configs/cpu_llvm_sync.json
```

### Updating expected failure lists

Each config file uses with pytest includes a list of expected compile and run
failures like this:

```json
  "expected_compile_failures": [
    "test_acos",
  ],
  "expected_run_failures": [
    "test_add_uint8",
  ],
```

To update these lists using the results of a test run:

1. Run pytest with the `--report-log` option:

    ```bash
    $ pytest iree_tests \
      --report-log=/cpu_llvm_sync_logs.json \
      --config-files=cpu_llvm_sync.json \
      ...
    ```

2. Run the `update_config_xfails.py` script:

    ```bash
    $ python iree_tests/update_config_xfails.py \
      --log-file=/cpu_llvm_sync_logs.json \
      --config-file=cpu_llvm_sync.json
    ```

You can also update the config JSON files manually. The log output on its own
should give enough information for each test case (e.g.
"remove from 'expected_run_failures'" for newly passing tests), but there can be
1000+ test cases, so the automation can save time.

### Advanced pytest usage tips

Collect tests (but do not run them):

```bash
$ pytest iree_tests --collect-only

============================= test session starts =============================
platform win32 -- Python 3.11.2, pytest-8.0.2, pluggy-1.4.0
rootdir: D:\dev\projects\SHARK-TestSuite
plugins: xdist-3.5.0
collected 1047 items

<Dir SHARK-TestSuite>
  <Dir iree_tests>
    <Dir onnx>
      <Dir node>
        <Dir generated>
          <Dir test_abs>
            <MlirFile model.mlir>
              <IreeCompileRunItem cpu>
          <Dir test_acos>
            <MlirFile model.mlir>
              <IreeCompileRunItem cpu>
          ...

======================== 1047 tests collected in 4.34s ========================
```

Run tests from a specific subdirectory:

```bash
$ pytest iree_tests/simple

======================================= test session starts =======================================
platform win32 -- Python 3.11.2, pytest-8.0.2, pluggy-1.4.0
rootdir: D:\dev\projects\SHARK-TestSuite\iree_tests
configfile: pytest.ini
plugins: retry-1.6.2, timeout-2.2.0, xdist-3.5.0
collected 2 items

simple\abs\simple_abs.mlir .                                                                 [ 50%]
simple\abs_bc\simple_abs.mlirbc .                                                            [100%]

======================================== 2 passed in 2.48s ========================================
```

Run a filtered subset of tests (see
[Specifying which tests to run](https://docs.pytest.org/en/8.0.x/how-to/usage.html#specifying-which-tests-to-run)):

```bash
$ pytest iree_tests -k "test_sub_"

============================= test session starts =============================
platform win32 -- Python 3.11.2, pytest-8.0.2, pluggy-1.4.0
rootdir: D:\dev\projects\SHARK-TestSuite
plugins: xdist-3.5.0
collected 1047 items / 1044 deselected / 3 selected

iree_tests\onnx\node\generated\test_sub_bcast\model.mlir .               [ 33%]
iree_tests\onnx\node\generated\test_sub_example\model.mlir .             [ 66%]
iree_tests\onnx\node\generated\test_sub_uint8\model.mlir x               [100%]

================ 2 passed, 1044 deselected, 1 xfailed in 4.65s ================
```

Run tests with a summary of which tests passed and failed (see the docs on
[Producing a detailed summary report](https://docs.pytest.org/en/8.0.x/how-to/output.html#producing-a-detailed-summary-report)):

```bash
$ pytest iree_tests -n auto -rpfE

============================= test session starts =============================
platform win32 -- Python 3.11.2, pytest-8.0.2, pluggy-1.4.0
rootdir: D:\dev\projects\SHARK-TestSuite
plugins: xdist-3.5.0
64 workers [1047 items]
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx [  6%]
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx [ 13%]
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...xxxxxxxxxx.xx.x.xxxxx.x [ 20%]
xx.xxxxxxxxxxxxxxxxxxxxx.............x......x..xx.xxxxx.xxx.xxxxxxxxxxxx [ 27%]
xxxxxxxxxx.xxxxx.xxxxx..x.xxxxxxxx..xxxx.x..xxxx.x....x.x.xxxx.xxxx..xx. [ 34%]
........x.xx.xxxxx..x.x.xxxx.xxxx..xxxxxxx.xx.xxxx.xxx.x..xxxxxxxx.xx.x. [ 41%]
xxxx.x.xxx.xxxx.xxxx.x.xx.xxxxx.xxxxxxxx.xx..xxxxx.xx.xxxxxxx..x.xxxx.xx [ 48%]
xxxxxxx.xxxxxxxxxxxxxxxxxxx.xxxxxxx...x..xxxxxxxxxxxxx.x..xxxxxxxxxxxxxx [ 54%]
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.x.....xxxxxxxxxxxxx.xxxxxx.xxx..xxx.x. [ 61%]
xxxxx..x.xxx..x.....xx.x.x...x.xxxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxxxxxxxxx [ 68%]
x.xxxxxxxxxxxxx...x.xxxxxxxxx.xxxxxxx..xxxxxxxxx.x.xxxxxxxxxxxxxxxxxxxxx [ 75%]
xxxxxxxxxxx...xxxxx..xx.xxxxxxxxxxxx.........xx.xxxxxx.xxxxxxxxx.xxxxxxxx [ 82%]
xxxxxxxxxxxxxxxxxxx.xxxx.......xxxxx..xxx.x.....xxxxxxxxxxxxxxxxxx.xxxxx [ 89%]
xxxxxxxxxxxxxxxxxxxxx........xxxxx...x.xx..............xxxxxxx.xxx.xxxx. [ 96%]
...xxxx...xx..xxx.....................                                   [100%]
=========================== short test summary info ===========================
PASSED iree_tests/onnx/node/generated/test_and_bcast4v3d/model.mlir::cpu
PASSED iree_tests/onnx/node/generated/test_clip_example/model.mlir::cpu
...
====================== 238 passed, 809 xfailed in 35.79s ======================
```

Fail test collection if files (such as downloaded weights) are missing:

```bash
$ pytest -k resnet50 --no-skip-tests-missing-files
======================================= test session starts =======================================
platform win32 -- Python 3.11.2, pytest-8.0.2, pluggy-1.4.0
rootdir: D:\dev\projects\SHARK-TestSuite\iree_tests
configfile: pytest.ini
plugins: dependency-0.6.0, retry-1.6.2, timeout-2.2.0, xdist-3.5.0
collected 1248 items / 1 error / 1248 deselected / 0 selected

============================================= ERRORS ==============================================
____________________ ERROR collecting pytorch/models/resnet50/resnet50.mlirbc _____________________
conftest.py:260: in collect
    test_cases = self.discover_test_cases()
conftest.py:236: in discover_test_cases
    raise FileNotFoundError(
E   FileNotFoundError: Missing files for test resnet50::real_weights
----------------------------------------- Captured stdout -----------------------------------------
Missing file 'inference_input.0.bin' for test resnet50::real_weights
Missing file 'inference_output.0.bin' for test resnet50::real_weights
Missing file 'real_weights.irpa' for test resnet50::real_weights
===================================== short test summary info =====================================
ERROR pytorch/models/resnet50/resnet50.mlirbc - FileNotFoundError: Missing files for test resnet50::real_weights
!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
================================ 1248 deselected, 1 error in 2.95s ================================
```

## Available test suites

### Simple tests

These are hand-authored tests demonstratating simple features about how the
tools and test suite work.

### PyTorch models

#### Generating model test cases from e2eshark

> [!WARNING]
> UNDER CONSTRUCTION - this will change!

1. Setup venv for the [e2eshark/](/e2eshark/) directory by following that README:

    ```bash
    e2eshark$ python -m venv .venv
    e2eshark$ source .venv/bin/activate
    e2eshark$ python -m pip install -r requirements.txt
    e2eshark$ python -m pip install -e [PATH TO SHARK-Turbine REPO]/models
    ```

    Notes:

    * You may need to downgrade numpy:

        ```bash
        pip uninstall numpy
        pip install numpy<2.0
        ```

2. Run a test from e2eshark to generate artifact files:

    ```bash
    e2eshark$ python run.py \
      --cachedir ${CACHE_DIR} \
      --tests pytorch/models/resnet50 \
      --mode turbine

    e2eshark$ ls test-run/pytorch/models/resnet50/

    __pycache__/        inference_input.0.bin           resnet50.default.input.pt
    commands.log        inference_output.0.bin          resnet50.default.pytorch.torch.mlir
    commonutils.py@     iree-compile.log                resnet50.default.vmfb
    E2ESHARK_CHECK.pkl  model-run.log                   runmodel.py
    inference.log       resnet50.default.goldoutput.pt  time.pkl
    ```

    We want the program `.mlir` and input/output `.pt` files.

3. Run `import_from_e2eshark.py --model=[model_name]` to extract parameters
   (both splats and real weights), convert to `.mlirbc`, and copy test files
   into `iree_tests/`:

   ```bash
   iree_tests$ python ./pytorch/models/import_from_e2eshark.py --model=resnet50
   iree_tests$ ls ./pytorch/models/resnet50

   opt-125M.mlirbc  splats.irpa
   ```

4. Add a `splat_data_flags.txt` matching the input signature and using
    the splat parameters:

    ```txt
    --input="1x3x224x224xf32"
    --parameters=splats.irpa
    ```

5. Upload `inference_input`, `inference_output`, and `real_weights.irpa` files
   from the `test-run/` folder to Azure (e.g. using Azure Storage Explorer)

6. Add a `real_weights_data_flags.txt` and `test_cases.json` file for real
   weights, pointing at the uploaded remote files.

#### Generating model test cases from turbine/tank

As seen in [`iree_tests/pytorch/models`](./pytorch/models/), there are some models with the "-tank" suffix.
This refers to tests that were generated using the normal turbine flow.
For custom models, such as sd, sdxl, or stateless_llama, you can clone the turbine repo
and follow the setup instructions there (https://github.com/nod-ai/SHARK-Turbine).
Then, simply run the respective model with the appropriate command line args (for sd, sdxl edit this: https://github.com/nod-ai/SHARK-Turbine/blob/ean-sd-fp16/models/turbine_models/custom_models/sdxl_inference/sdxl_cmd_opts.py. otherwise, just direct command line args for llama. make sure to --compile_to vmfb).
Just as a side note, the unet_scheduler model requires diffusers dep changes, so make sure to use changes
in this branch: https://github.com/aviator19941/diffusers/tree/pndm_fx_v2.
Example run command (`python models/turbine_models/custom_models/sdxl_inference/sdxl_prompt_encoder.py`).
There is no easy way to get `.bin` or `.npy` files for your inputs and outputs.
You will have to edit the model runner files to convert the input and output tensors into `.bin` files,
so those are saved when running the flow. (example runner:
`models/turbine_models/custom_models/sdxl_inference/sdxl_prompt_encoder_runner.py`).
Then, run the runner with the appropriate command line args (vmfb path, device flags).
You should have all the artifacts needed to add to this TestSuite at that point.
Make sure to follow to follow appendix instructions to convert between different file types for weights and mlir.

### SHARK Tank models

These test cases are exported from https://github.com/nod-ai/sharktank.

## Steps to add test cases

* Follow instructions in https://github.com/nod-ai/sharktank/blob/main/docs/model_cookbook.md
* Convert the exported `.mlir` to `.mlirbc`:

    ```bash
    iree-ir-tool cp file.mlir --emit-bytecode -o file.mlirbc
    ```

* Create a test_cases.json file with parameters, inputs, and outputs
  * Parameters can come from Hugging Face by using URL from "download file"
  * TODO: inputs and outputs should be exportable from sharktank/shortfin
    (or a script here - need to run the tokenizer and optionally populate the
    KV cache for some models)

## Appendix

### Working with .mlirbc files

The [MLIR Bytecode Format](https://mlir.llvm.org/docs/BytecodeFormat/) (often
represented as `.mlirbc` files) can be used to store/transmit/load MLIR files
efficiently, but it is harder to inspect than text (`.mlir`) files.

To convert files IREE understands between `.mlir` and `.mlirbc`:

```bash
iree-ir-tool cp model.mlir --emit-bytecode -o model.mlirbc
iree-ir-tool cp model.mlirbc -o model.mlir
```

You can also run through `-opt` tools like `torch-mlir-opt` with no options,
if the tool includes all relevant MLIR dialects:

```bash
torch-mlir-opt model.mlirbc -o model.mlir
```

The
[MLIR VSCode extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir)
can also edit `.mlirbc` files as text.

### Working with large MLIR files

To simply strip weights:

```bash
iree-ir-tool strip-data model.mlir -o model_stripped.mlir
```

### Working with parameter files

To convert from .safetensors to .irpa (real weights):

```bash
iree-convert-parameters \
  --parameters=path/to/file.safetensors \
  --output=path/to/output.irpa
```

To strip constants and replace them with splats:

```bash
iree-convert-parameters \
  --parameters=path/to/parameters.[safetensors,irpa] \
  --strip \
  --output=path/to/output.irpa
```
