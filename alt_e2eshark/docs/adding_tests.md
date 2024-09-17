## Adding a new Onnx Test

### Where to Add a New Test

Tests are defined in one of the subdirectories in `alt_e2eshark/onnx_tests/`. If adding a new test, and none of the filenames are appropriate, feel free to make a new file, for example, `alt_e2eshark/onnx_tests/operators/new_example.py`. To include tests registered in `new_example.py`, please add 
```python
from .new_example import *
```
to `alt_e2eshark/onnx_tests/operators/model.py`.

### How to Add a New Onnx Test

All tests need to have an associated TestInfo class. This class determines how the onnx model is found, how to contruct inputs, etc. 

The base class for all onnx model tests is `OnnxModelInfo`, which is defined in `alt_e2eshark/e2e_testing/framework.py`.

The only method that absolutely must be overriden to properly define a new test info class is minimally `construct_model`, which should tell the test runner how to construct, download, or otherwise find the onnx model assosicated with a test. Other methods which may be useful to override:

- `construct_inputs`: We provide a somewhat robust random input generator, so it is not always necessary to override this. For a test which requires specific inputs to provide appropriate results, it is recommended to override this with custom input generation. E.g., for vision models, consider taking a tensor which represents an actual image file. 
- `update_sess_options` : This is the session options used during the onnxruntime session which generates our reference outputs. Disabling optimizations is recommended for some quantized models, since onnxruntime CPU implementations for certain operators (such as integer resize, for example) may deviate substantially when optimized from their original implementations. 
- `update_dim_param_dict` : if your model has dynamic dims (dim_params are string representations for these dynamic dims, e.g. "batch_size"), then you may override this method to define a `str -> int` dictionary in `self.dim_param_dict` to provide dims for constructing sample inputs. 
- `apply_postprocessing` : used to define a function for applying some post-processing logic to the test outputs before determining numerical accuracy. 
- `save_processed_output` : used to define how to save the postprocessed output for human inspection (e.g., save image outputs as a .jpeg or language model outputs as a .txt file).

Once a test info class is defined for your test, you can add it to the registry for our test runner by using the `register_test` function defined in `alt_e2eshark/e2e_testing/registry.py`. Example usage would be:

```python
from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test

# define a test class
class MyTestInfoClass(OnnxModelInfo):
    ...

# register the test with a unique name
register_test(MyTestInfoClass, "my_unique_test_name")

```

## Helpful Test Info Classes

Some convenient `OnnxModelInfo` derived classes are defined in `alt_e2eshark/onnx_tests/helper_classes.py`. Here is some information about a few of these helper classes:

### Adding Models from Azure Storage

For models which can be downloaded from one of our Azure storage blobs, a convenience base class, `AzureDownloadableModel` is defined in `alt_e2eshark/onnx_tests/helper_classes.py`. 

Unless your model has dynamic dim params, or inputs that should be constructed more carefully, this `AzureDownloadableModel` should be a sufficient test info class to register your test with. Since models on Azure have a unique name, `AzureDownloadableModel` is set up to automatically download and set up a test for the model with name matching the one being registered with the class. For example, 

```python
from onnx_tests.helper_classes import AzureDownoadableModel
from e2e_testing.registry import register_test

register_test(AzureDownloadableModel, "my_exact_model_name")
```

Would download the blob `e2eshark/onnx/models/my_exact_model_name/model.onnx.zip` from our public Azure storage (or private if you have a connection string for it), then unzip the contents into the test directory `test-run/my_exact_model_name/`. 

### Building a Model from Scratch

We have already registered node tests from the user's installed onnx package. If you want to test average pooling, for example, you could run 

```
python run.py -v -t test_average
```

To view these tests, look for `<your_site_packages_dir>/onnx/backend/test/data/node/`. 

If, for some reason, the coverage for these tests doesn't include a case you'd like to test (e.g., QDQ implementations, operator combinations, or a specific attribute combination not already covered), then we provide a helper TestInfo class called `BuildAModel`.

To use `BuildAModel`, you will need to inherit from it and override the following methods:

- `construct_nodes` (Mandatory Override) : use this to add nodes to the list of nodes `self.node_list`. During the pre-defined `construct_model` method, we will generate a graph with the nodes saved in `self.node_list`. There is a method for `BuildAModel` called `get_app_node()`, which returns a lambda function for automatically constructing a node and appending it to `self.node_list`. 
- `construct_i_o_value_info` (Mandatory Override) : use this method to add input value info to the list `self.input_vi`, and output value info to the list `self.output_vi`.
- `construct_initializers` : only override this if you'd like to add initializers to your onnx graph through `self.initializers`.
- Any of the optional methods from `OnnxModelInfo` you'd further like to customize. E.g. `construct_inputs`, `apply_postprocessing`. 

```python
from e2e_testing.registry import register_test
from onnx_tests.helper_classes import BuildAModel
# need these to define value info protos:
from onnx import TensorProto
from onnx.helper import make_tensor_value_info as make_vi

class AddSubtract(BuildAModel):
    def construct_nodes(self):
        app_node = self.get_app_node()
        # self.get_app_node() returns a lambda app_node. Usage:
        # app_node(op type, input list, output list, optional attribute kwargs)
        # will do 
        # from onnx.helper import make_node
        # self.node_list.append(make_node(args, kwargs))
        app_node("Add", ["X","Y"], ["Z"])
        app_node("Sub", ["Z", "X"], ["W"])
    
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_vi("X", TensorProto.FLOAT, [1,2,3]),
            make_vi("Y", TensorProto.FLOAT, [1]),
            ]
        self.output_vi = [make_vi("W", TensorProto.FLOAT, [1,2,3])]
        # note: "W" is neither a graph input nor graph output. Don't include it.

register_test(AddSubtract, "add_subtract")
```

For other examples, look in `alt_e2eshark/onnx_tests/combinations/` and `alt_e2eshark/onnx_tests/operators/`.

### Registering the same test twice

Tests need to have unique names. If you want to run a test twice with different options (e.g., postprocessing), it may be helpful to use `SiblingModel`. This will take another model info instance and re-use it's onnx model. You will not want to override `construct_model` since `SiblingModel` already overrides this. To register such a test, it is helpful to use the helper function `get_sibling_constructor` to generate a constructor for the sibling class. Here is an example:

```python
from e2e_testing.registry import register_test
from onnx_tests.helper_classes import AzureDownloadableModel, SiblingModel, get_sibling_constructor

class WithPostProcessing(SiblingModel):
    def apply_postprocessing(self, x):
        ...

# optionally register the original test
register_test(AzureDownloadableModel, "original_model_name")

# create a constructor for WithPostProcessing that passes the original test information
# i.e. original test = (AzureDownloadableModel, "original_model_name")
constructor = get_sibling_constructor(WithPostProcessing, AzureDownloadableModel, "original_model_name")

# register the sibling model which includes the new post-processing step
register_test(constructor, "new_test_name")
```

### Modifying Model Outputs

When debugging numerics, it is helpful to truncate an onnx model and return the results only up to a specific node. The helper class `TruncatedModel` is similar to `SiblingModel`, but the user additionally specifies a node index and op type in the constructor. These values are used to determine what node to use as a new output for the model.

```python
from e2e_testing.registry import register_test
from onnx_tests.helper_classes import AzureDownloadableModel, TruncatedModel, get_truncated_constructor

# optionally register the original test
register_test(AzureDownloadableModel, "original_model_name")

# create a constructor for TruncatedModel that passes the original test information
# i.e. original test = (AzureDownloadableModel, "original_model_name")
constructor = get_truncated_constructor(TruncatedModel, AzureDownloadableModel, "original_model_name")

# modify "original_model_name" to return on the first "Conv" node found. 
register_test(constructor(0, "Conv"), "new_test_name")

# if no op type specified, returns on the nth-to-last node in the graph
# this would return the tenth-from-last node in the graph
register_test(constructor(10, ""), "another_name")

# often useful to register multiple truncated tests.
# this would create ten tests that each return on one of the first ten MaxPool nodes
for i in range(0,10):
    register_test(constructor(i, "MaxPool"), f"truncated_{i}_MaxPool_test")

# although not often useful in practice, you could add additional customization to
# the truncated model class before making the constructor
class CustomTruncModel(TruncatedModel):
    def construct_inputs(self):
        # define specific input generation
        ...

custom_constructor = get_truncated_constructor(CustomTruncModel, AzureDownloadableModel, "original_model_name")

# register_test(custom_constructor( ....
```


