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

 - pytroch : starting model is a pytorch model
 - tensorflow : stating model is a tensorflow model (planned for later)
 - onnx : starting model is an onnx model generated using onnx python API or a an existing onnx model (zipped)
 
 Following, upto three, modes are supported based upon what is possible for a framework:

 - direct: Framework -> Framework graph (e.g. Torch Fx) -> Torch MLIR -> Compiled artefact -> Run target backend
 - onnx: Framework -> ONNX -> Import as Torch ONNX in Torch MLIR -> Torch MLIR -> Compiled artefact -> Run target backend
 - ort: Framework -> ONNX -> Load in IREE ONNX Runtime EP -> Compiled artefact -> Run target backend (planned for later)

 If Framework is 'onnx', then mode 'direct' will mean same as 'onnx'. For onnx/operators and onnx/combinations,  
 the onnx model should be created using ONNX Python APIs. For onnx/models, a prebuilt onnx model should be checked 
 in as a zip file.
 
 The target backend can be any IREE supported backend: llvm-cpu, amd-aie etc.

## Contents
 The contents are as below. Substitute 'Framework' by one of: pytorch, tensoflow, onnx
 - requirements.txt : 'pip install -r requirements.txt' to install needed additional 
                       packages not already in your venv or conda. If you have a venv or conda 
                       environment for building torch mlir or iree, you can install 
                       this on top of that. Also, peridically, run this step to keep packages current. 
                       Sometimes you may need to force installation: 'pip install --force -r requirements.txt'
 - run.py : Run 'python run.py --help' to learn about the script. This is the script to 
            run a specific test, all tests in a framework, all frameworks as per choice of a user
 - Framework/operators: This has operator level tests. Example: pytorch/operators/conv2d
 - Framework/combinations: This has small tests testing combination of operators such as 
                           pytorch/combinations/mlp testing a multi-layer perceptron which 
                           has torch.linear followed by torch.relu and optionally repeated a few times
 - Framework/models: This has full model test. Since this is full model test, you may need 
                     necesary permsisions to download a model such as for llama2 you will 
                     need hugging face token. You can either run 'huggingface-cli login' and 
                     enter the HF token before launching the run.py or set an environment variable 
                     named HF_TOKEN or provide it as "HF_TOKEN='token' python run.py 'options'"
 - gold/passed.txt : This has list of tests that are passing as of today. If your check-in increases
                     passing tests, please update this. After your changes in torch MLIR or IREE, 
                     run all the tests upto inference and make sure that your are passing at least 
                     at the level of gold/passed.txt
 - tools/stubs/onnxmodel.py : This is concatenated to the 'model.py' in the test directory to form a 
                              runmodel.py for tests of framework 'onnx'
 - tools/stubs/pytorchmodel.py : This is concatenated to 'model.py' in test directory to form a 
                                 runmodel.py for the tests of framework 'pytorch'
 - tools/stubs/commonutils.py: Utilities common to other tools/stubs files as well as model.py
                     This also defines a dictionary named as E2ESHARK_CHECK_DEF, an instance of
                     which is created for each test. This allows serializing input, output, any 
                     special controls, post processing recipe to be passed from model.py to run.py
 - tools/onnxutil.py : Allows examining an ONNX (protobuf) file
 - tools/reportutil.py: Given two or more run directories, diff or merge any of status, time or 
                      summary reports. For time and summary reports, it tell how many new 
                      passes (improvements) were seen
 - tools/aztestsetup.py: Setup, upload and download large models to/from Azure storage

 
 The logs are created as .log files in the test-run sub directory. Examine the logs to find and fix 
 cause of any failure. You can specify -r 'your dir name' to the run.py to name your test run directory 
 as per your choice. The default name for the run directory is 'test-run'.

 Note that, you will be required to pass --cachedir argument to the run.py to point to a directory where 
 model weights etc. from external model serving repositories such as from Torch Vision, Hugging Face etc.
 will be downloaded. The downloaded data can be large, so set it to other than your home, 
 preferably with 100 GB or more free space.

## Setting up

By default, a nightly build of torch_mlir and IREE is installed when you run 'pip install -r ./requirements.txt'
and that will be used to run tests. So, you are not required to have a local build of either torch mlir or iree.

But, if you want to test a local change in either torch mlir or iree, you can pass the build using switches.

If you have a local build of torch MLIR pass it using --torchmlirbuild (-c) option. Else, it will use
torch_mlir package from last nightly build and use iree tools directly such as iree-import-onnx to import
onnx into torch onnx mlir as opposed to running torch_mlir.tools.import_onnx.

If you made changed in IREE or you want to pull the latest fix in IREE, then create a local build and pass 
that using --ireebuild (-i). 

The arg value passed to --torchmlirbuild (-c) should point to a directory where 
bin/torch-mlir-opt can be found and the option passed to --ireebuild (-i) should point to 
a directory where tools/iree-compile and tools/iree-run-module can be found.

In all the exmaples in this README doc the -c option to pass a torch mlir build location 
or -i option to pass an IREE build location are optional.

**Prerequisites**: Before proceeding with build, make sure you have following installed on your machine

1. **_Git_**       : [GitInstallation](https://github.com/git-guides/install-git)
2. **_Python_**    : [PythonInstallation](https://www.python.org/downloads/)
3. **_cmake_**     : [cmakeInstallation](https://cmake.org/download/)
4. **_Ninja_**     : [NinjaInstallation](https://ninja-build.org/)
5. **_GCC(>=7.4)_**: [gccInstallation](https://gcc.gnu.org/install/)
6. **_CUDAToolkit(>=12)_** : [CUDAToolkitInstallation](https://developer.nvidia.com/cuda-downloads)


To get a local build of torch MLIR, follow:
https://github.com/llvm/torch-mlir/blob/main/docs/development.md

If you have a local build of torch MLIR build, you can build the torch_mlir python wheel and install it in your 
python env in addition, that will remove need to use PYTHONPATH when you are running model.py manually. Use any of
following options for allowing torch_mlir package to be available when running manually the model.py or runmodel.py.
If you run using run.py then this step is taken care of for you.

Option 1:

https://github.com/llvm/torch-mlir/blob/main/docs/development.md#build-python-packages 

Edit setup.py to insert following in cmake_args if you used clang to build torch-mlir 
```
 f"-DCMAKE_C_COMPILER=clang",
 f"-DCMAKE_CXX_COMPILER=clang++",
 ```
And then run following

```
CMAKE_GENERATOR=Ninja python setup.py bdist_wheel --dist-dir ./torch-mlir-wheel -v
pip uninstall torch-mlir
pip install torch-mlir-wheel/torch_mlir-0.0.1-cp310-cp310-linux_x86_64.whl
```
**Note**: torch_mlir wheel version may change so you should choose appropriate file name in the above command


An easier way than creating the above torch_mlir wheel is to simply set PYTHONPATH while running model.py or set and export
PYTHONPATH. 

Option 2:
```
PYHONPATH="Your torch mlir build dir"/tools/torch-mlir/python_packages/torch_mlir python model.py
```
Option 3: 
```
cd to "Your torch mlir root dir"
./build_tools/write_env_file.sh
source build/.env && export PYTHONPATH
```

To get a local build of IREE, follow (prefer the clang building option over gnu):
https://iree.dev/building-from-source/getting-started 

If using AMD AIE target, build following in addition:
https://github.com/nod-ai/iree-amd-aie

To get this test repo:
```
git clone https://github.com/nod-ai/SHARK-TestSuite.git
cd SHARK-TestSuite/e2eshark 
```
Set up python envonment, you can use venv or conda. 
Example to create a brand new conda env using python 3.11 is: 
```
conda create -n e2e python=3.11
conda activate e2e
pip install --upgrade pip

```
Make sure you do not skip the pip upgrade above as older pip may not able to handle pytorch deps
Then install needed packages as below:
```
pip install -r 'your local torch MLIR repo'/requirements.txt
pip install -r 'your local torch MLIR repo'/torchvision-requirements.txt
pip install -r ./requirements.txt
```
Optionally, as explained above, if you have built a local wheel of torch_mlir, then do (your whl file name may differ):
```
pip install 'your local torch MLIR repo'/torch-mlir-wheel/torch_mlir-0.0.1-cp310-cp310-linux_x86_64.whl
```

## Turbine Mode

If you are also interested in running through SHARK-Turbine follow these instructions as well:

```
git clone https://github.com/nod-ai/SHARK-Turbine
```

Now, go back to the TestSuite Repo, and make sure you are using same venv from all previous steps.

```
pip install -f https://openxla.github.io/iree/pip-release-links.html --upgrade -r 'your local SHARK Turbine repo'/core/iree-requirements.txt
pip install -e 'your local SHARK Turbine repo'/core[testing]
pip install -e 'your local SHARK Turbine repo'/models
```

Once setup, in any new shell you can activate the same env everytime you want to use it 
without needing to re-install requirements.txt's. 
Example:
```
conda activate e2e
```
## Examples

### Running tests

Example 1:

Note that the --cachedir command line argument is necessary for any run command. This is where torch, hugging face, and turbine tank data is cached. Please make sure to choose a directory with large free space.

Run the tests in operators, combinations folders of the default framework (i.e. pytorch),
Use framework to onnx to torch MLIR path (--mode onnx) and run upto inference (default) using llvm-cpu backend (default),
use four processor cores (default --jobs 4) on your machine, generate report file after finishing test run.

If the model you are running requires a huggingface token (llama, gemma), set the HF_TOKEN env variable as well.
Either set environment in shell (`export HF_TOKEN=your_token`) or add it on in command line
as shown below.

```
HF_TOKEN=your_token python ./run.py -c 'path_to_your_torch_mlir_build_dir' -i 'path_to_your_iree_build_dir'
--report --cachedir 'path_to_your_cache_dir'
```
You can see logs of test run inside test-run/'test sub-directory'. Start with commands.log file. 

The test-run/statusreport.md, test-run/timereport.md, and test-run/summaryreport.md will show nice tables 
like below to give you detailed status of pass/fail of each stage, time taken by each stage and total counts 
of passes for each phase. Furthermore, you can compare these reports using tools/reportutil.py to get either 
a merged view or diff of one or more runs.
```
Status report for run: fp32 using mode:onnx todtype:default backend:llvm-cpu

| tests                    | model-run | onnx-import | torch-mlir | iree-compile | inference |
| :----------------------- | :-------- | :---------- | :--------- | :----------- | :-------- |
| pytorch/operators/conv2d | passed    | passed      | passed     | passed       | passed    |
| pytorch/operators/linear | passed    | passed      | passed     | passed       | passed    |
| pytorch/combinations/mlp | passed    | passed      | passed     | passed       | passed    |


Time (in seconds) report for run: fp32 using mode:onnx todtype:default backend:llvm-cpu

| tests                    | model-run | onnx-import | torch-mlir | iree-compile | inference |
| :----------------------- | --------: | ----------: | ---------: | -----------: | --------: |
| pytorch/operators/conv2d |   3.15231 |    0.425977 |  0.0067625 |     0.570327 | 0.0300162 |
| pytorch/operators/linear |   3.11147 |    0.422572 | 0.00860119 |     0.825361 | 0.0320294 |
| pytorch/combinations/mlp |   3.16689 |    0.438611 | 0.00681615 |      1.04092 | 0.0127583 |

Summary (time in seconds) for run: fp32 using mode:onnx todtype:default backend:llvm-cpu

| items        |   tests | model-run | onnx-import | torch-mlir | iree-compile | inference |
| :----------- | ------: | --------: | ----------: | ---------: | -----------: | --------: |
| total-count  |       3 |         3 |           3 |          3 |            3 |         3 |
| average-time | 4.41714 |   3.14356 |    0.429053 | 0.00739328 |     0.812203 | 0.0249346 |
| median-time  | 4.44048 |   3.15231 |    0.425977 | 0.00681615 |     0.825361 | 0.0300162 |
```

The test-run/passed.txt has list of all tests that passed and test-run/failed.txt has list of all 
the tests that failed. After you make changes in your source code to fix torch MLIR or IREE, you can 
re-run just the failing tests by simply passing test-run/failed.txt as an input like below:
```
python ./run.py --cachedir 'path_to_your_cache_dir' -c 'path_to_your_torch_mlir_build_dir' -i 'path_to_your_iree_build_dir' --testsfile test-run/failed.txt
```
If you want to just generate report and skip run of tests then, you can pass --norun to skip running tests as below:
```
python ./run.py --cachedir 'path_to_your_cache_dir' -c 'path_to_your_torch_mlir_build_dir' -i 'path_to_your_iree_build_dir' --testsfile test-run/failed.txt --norun --report
```

Example 2:
You can start from and run upto a stage. There are four stages: model-run, torch-mlir, iree-compile and inference
Say if you tested upto torch-mlir and do not want to test further and come back later and test torch-mlir onwards
later. As long as you have not destroyed the test-run dir, you can run following two at different times 
(first should have generted the torch MLIR for second one to resume successfully):
```
python ./run.py --cachedir 'path_to_your_cache_dir' -c 'path_to_your_torch_mlir_build_dir' -i 'path_to_your_iree_build_dir' --frameworks pytorch onnx --runupto torch-mlir 

python ./run.py --cachedir 'path_to_your_cache_dir' -c 'path_to_your_torch_mlir_build_dir' -i 'path_to_your_iree_build_dir' --runfrom torch-mlir --runupto inference 

```
Example 3:
Run given test pytorch/models/opt-125M upto inference (default for --runupto) on target AMD AIE backend
```
python ./run.py --cachedir 'path_to_your_cache_dir' -c 'path_to_your_torch_mlir_build_dir' --frameworks onnx -i 'path_to_your_iree_build_dir' --tests pytorch/models/opt-125M --backend amd-aie
```

Example 4:
Print number of occureneces (frequency) of each operator kind in an ONNX file and print 
total count of operator instances

 ```
python ./tools/onnxutil.py onnx/models/resnet50_vaiq_int8/model.onnx -f
```

Example 5:
Merge reports from two different run directories named as 'fp32" and "bf16" and show a combined comparision view
```
python tools/reportutil.py --do merge fp32 bf16 
```
An example merged report:
```
| test-name                     | model-run-fp32 | model-run-bf16 | onnx-import-fp32 | onnx-import-bf16 | torch-mlir-fp32 | torch-mlir-bf16 | iree-compile-fp32 | iree-compile-bf16 | inference-fp32 | inference-bf16 |
| ----------------------------- | -------------- | -------------- | ---------------- | ---------------- | --------------- | --------------- | ----------------- | ----------------- | -------------- | -------------- |
| pytorch/models/llama2-7b-GPTQ | failed         | failed         | notrun           | notrun           | notrun          | notrun          | notrun            | notrun            | notrun         | notrun         |
| pytorch/models/opt-125M       | passed         | passed         | passed           | passed           | passed          | passed          | passed            | passed            | passed         | mismatch       |
| pytorch/models/llama2-7b-hf   | passed         | passed         | passed           | passed           | passed          | passed          | passed            | passed            | mismatch       | mismatch       |
```

Example 6:
Diff reports from two different run directories named as 'fp32" and "bf16" and show whether each test run matched or 
differed
```
python tools/reportutil.py --do diff fp32 bf16 -m status -v

The diff report for status of runs: fp32, bf16
| test-name                | model-run | onnx-import | torch-mlir | iree-compile | inference         |
| :----------------------- | :-------- | :---------- | :--------- | :----------- | :---------------- |
| pytorch/operators/linear | same      | same        | same       | same         | [passed,mismatch] |
| pytorch/combinations/mlp | same      | same        | same       | same         | same              |
| onnx/operators/gemm      | same      | same        | same       | same         | same              |

python tools/reportutil.py --do diff fp32 bf16 -m time

The diff report for time (in seconds) of runs: fp32, bf16
| test-name                | model-run | onnx-import |  torch-mlir | iree-compile |  inference |
| :----------------------- | --------: | ----------: | ----------: | -----------: | ---------: |
| pytorch/operators/linear |  0.227186 |  -0.0256374 | 0.000367403 |    0.0863338 |  0.0362182 |
| pytorch/combinations/mlp |  0.245063 |   -0.041338 | 0.000627995 |      0.15707 | -0.0194921 |
| onnx/operators/gemm      | -0.111096 |   0.0017724 |  0.00060463 |            0 |          0 |

python tools/reportutil.py -d diff fp32 bf16 -m summary
of
The diff report for summary (time in seconds) of runs: fp32, bf16
| items        |    tests | model-run | onnx-import | torch-mlir | iree-compile |  inference |
| :----------- | -------: | --------: | ----------: | ---------: | -----------: | ---------: |
| total-count  |        0 |         0 |           0 |          0 |            0 |         -1 |
| average-time | 0.241627 |  0.113173 |    0.026411 | 0.00141684 |    0.0643837 |  0.0362426 |
| median-time  | 0.209797 |  0.109462 |   0.0351717 | 0.00205493 |    0.0539219 | 0.00918627 |

python tools/reportutil.py -d diff fp32 bf16 fp32 -m summary -v -c 4,5

The diff report for summary (time in seconds) of runs: fp32, bf16, fp32

| items        | iree-compile     | inference        |
| :----------- | :--------------- | :--------------- |
| total-count  | same             | [3,2,3]          |
| average-time | [0.81,0.88,0.81] | [0.02,0.06,0.02] |
| median-time  | [0.83,0.88,0.83] | [0.03,0.04,0.03] |

if more than two runs are diffed, then when values differ the comma separated differing values are shown
```
The -1 under inference indicates, one test regressed in inference

### Running tests with upload

If you are interested in running tests, but want to upload the mlir files generated to Azure 
to share with others or for yourself, first you will have to set the AZURE_CONNECTION_STRING environment
variable. You can find this connection string here: 
https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/e2esharkuserartifacts/keys. 
If you don't have access to link above, you can ask Sai Enduri for the connection string.

Then, setup an upload_list.txt file with the names of the models you want to upload on. There is already one
setup at e2eshark/gold/upload_list.txt. You can just modify that one.

Optional: If you want to change what type of files are being uploaded, simply tweak `upload_list = ["mlir"]` in e2eshark/run.py to change or add more file types you want to upload (`upload_list = ["mlir", "log"]` for example).


With this connection string and upload list file, you can now run command like this:
```
python ./run.py -c ../../torch-mlir/build -i ../../iree-build --report --cachedir ~/.cache/huggingface --mode direct --tests pytorch/models/beit-base-patch16-224-pt22k-ft22k pytorch/models/bge-base-en-v1.5 pytorch/models/mit-b0 pytorch/models/bert-large-uncased pytorch/models/deit-small-distilled-patch16-224 --uploadtestsfile /home/sai/SHARK-TestSuite-fork/e2eshark/gold/upload_list.txt --cleanup
```

You can then find a json file (`upload_urls.json` in e2eshark directory) with the model names and links to the files uploaded for each model. You can just wget these links to download as it is public, so should be easy to share with others.

### Adding new tests

#### Adding test in framework pytorch
Let us say you wanted to add a new test to the framework "pytorch" to test maxpool  i.e. start with
pytorch model of maxpool, use run of the model as reference gold output and compare that with IREE compiled
output on your target backend. 

First google pytorch maxpool and read about the corresponding behavior of
maxpool operator in pytorch. For this example:
https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html provides description. 

Now take following steps:
1. Use an appropriate suffix to describe what you are intending to test say maxpool_2d_large
2. mkdir -p pytorch/operators/maxpool_2d_large
3. cd pytorch/operators/maxpool_2d_large
4. cp -pr pytorch/operators/conv2d/model.py .
5. modify model.py
6. You can run 'python ./model.py' to see input and output values

Once your model.py is ready, you can go to root of the e2eshark test directory and run test as below 
   ```
   python ./run.py --cachedir 'path_to_your_cache_dir' -c "your torch mlir build dir" --tests pytorch/operators/maxpool_2d_large --mode direct --runupto torch-mlir --torchtolinalg
   ```
   Note that I did not specify -i for IREE build as I am running only upto torch-mlir. Also, I have added 
   --torchtolinalg to make sure I test upto linalg lowerging as I am not running iree-compile

   Rerun above with --mode onnx if you want ONNX to get generated from pytorch and tested in addition.

   If you want to test upto inference, then provide your iree build in addition as -i option and run as

   ```
   python ./run.py --cachedir 'path_to_your_cache_dir' -c "your torch mlir build dir" -i 'path_to_your_iree_build_dir' --tests pytorch/operators/maxpool_2d_large --mode direct
   ```

   As before, run above with --mode onnx if you want ONNX to get generated from pytorch and tested in addition.

   cd into test-run/pytorch/operators/maxpool_2d_large and examine logs created. 
   Iterate over fixing your model.py and examing logs till you get it right. Given state of tools, tests may fail.
   If test fails because tool has bug, you are ready to add test. If test fails because of issue in model.py,
   you need to fix that before you can add test.
   
   Once you are satisfied, git add pytorch/operators/maxpool_2d_large/model.py, commit and publish it. 
   If that is a passing test and not in gold/passed.txt, add that passing test there as well

   #### Adding test in framework onnx

   Similarly to add a test in framework onnx, for say cumsum operator, First google 
   pytorch ONNX.Cumsum and study behavior of the operator. For example, study
   https://onnx.ai/onnx/operators/onnx__CumSum.html . 

   If you do not already know how to write an ONNX model using ONNX Python API, study
   https://onnx.ai/onnx/intro/python.html


   Then take following steps:
   1. Use an appropriate suffix to describe what you are intending to test say cumsum_small
   2. mkdir -p onnx/operators/cumsum_small
   3. cd onnx/operators/cumsum_small
   4. cp -pr onnx/operators/gemm/model.py .
   5. modify model.py to model behavior of ONNX.cumsum
   6. run 'python ./model.py' to see input and output values, verify them to be as desired 
   
   Then test it as: 

   ```
   python ./run.py --cachedir 'path_to_your_cache_dir' -c "your torch mlir build dir" -i 'path_to_your_iree_build_dir' --tests onnx/operators/cumsum_small --mode direct --runupto inference --torchtolinalg
   ```

   Then follow steps similar to the one described above for pytorch framework to test more and add 
   the test to the repo.

   ### Testing for different dtypes
   The model (test) writer should decide on a specific data type (dtype) to use. Name the test with a suffix 
   of dtype to indicate its dtype if not fp32. As an exmaple a 8-bit VAI quantized model (test) for resnet50 
   should preferably be named as restnet50_vaiq_int8 vs a fp32 model (test) as simply resnet50.

   If a framework has support for casting a model and tensor to a particular type then using switch --todtype you can
   run the same model with a different dtype. As an example, Pytorch has model.to(dtype) and tensor.to(dtype) for floating point types only, so for any of pytorch tests you can run test by passing --todtype 'an_allowed_value' . 
   
   Model, parameters and inputs are cast to new dtype. As long as framework run of model and IREE compiled run of 
   model match we are able to test torch-mlir and IREE, even if the values are not exactly the same as the original values, this way of testing can help us detect many low hanging bugs. 
   
   The exmaple run below casts the model and input tensor to bf16 for test pytorch/combinations/mlp (which is originally written as fp32 model):

   ```
   python ./run.py --cachedir 'path_to_your_cache_dir' -c "your torch mlir build dir" -i 'path_to_your_iree_build_dir' --tests pytorch/combinations/mlp --mode onnx --runupto inference --torchtolinalg --todtype bf16
   ```
