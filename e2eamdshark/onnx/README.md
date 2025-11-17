## Writing a model.py File for a New Onnx Model

### General Steps

Making a model.py for an unknown model.onnx file can be done in a few steps:

0. Copy boilerplate from another model.onnx file. 
1. Figure out what the inputs look like.
2. Make some randomized model inputs.
3. Start an inference session and run it.
4. Use E2Eamdshark_CHECK to record inputs and outputs and apply post-processing if desired.

Let's go into more depth into each step. 

### 0. Copy boilerplate from another model.onnx file

In addition to whatever additional python packages you choose to use, the boilerplate should minimally be:

```python
import numpy, torch, sys
import onnxruntime

# import from e2eamdshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2Eamdshark_CHECK_DEF

# Create an instance of it for this test
E2Eamdshark_CHECK = dict(E2Eamdshark_CHECK_DEF)
```

### 1. Figure out what the inputs look like

Frankly, I find it easiest to simply run the following python script to dump the inputs:

```python
import onnxruntime

session = onnxruntime.InferenceSession("model.onnx", None)
inputs = session.get_inputs()
for input in inputs:
    print(input)
```

For example you might see a printout like:

```
NodeArg(name='some_arg_name', type='tensor(int64)', shape=['batch_size', 'another_dim_param'])
NodeArg(name='another_arg_name', type='tensor(float)', shape=['batch_size', 12, 'another_dim_param * 2', 64])
```

This gives us everything we need to make some model inputs.

### 2. Make some randomized model inputs

Continuing the example above, we need to first specify the values for each dim_param. Here is an example of doing this:

```python
dim_params = (
    'batch_size',
    'another_dim_param',
    'another_dim_param * 2',
)
dim_values = (
    7,
    13,
    26,
)
pv = dict(zip(dim_params, dim_values))
```

It's useful to specify 'another_dim_param * 2' seperately if you want, for example, to make the model.onnx static before running the tests. 

Now that we chose some values for the dim_params, we can make some model inputs.

```python
some_arg_name = numpy.random.randint(
    -1000,
    high=1000,
    size=(pv["batch_size"], pv["another_dim_param"]),
    dtype=numpy.int64,
)
another_arg_name = numpy.random.rand(
    pv["batch_size"], 12, pv["another_dim_param * 2"], 64
).astype(numpy.float32)
model_inputs = [some_arg_name, another_arg_name]
```

### 3. Start an inference session and run it

This just involves including:

```python
# start a session
session = onnxruntime.InferenceSession("model.onnx", None)

# get inputs
inputs = session.get_inputs()
outputs = session.get_outputs()

model_outputs = session.run(
    [output.name for output in outputs],
    {inputs[i].name: model_inputs[i] for i in range(len(model_inputs))},
)
```

### 4. Use E2Eamdshark_CHECK to record inputs and outputs and apply post-processing if desired

At the very least, include:

```python
E2Eamdshark_CHECK["input"] = [torch.from_numpy(arr) for arr in model_inputs]
E2Eamdshark_CHECK["output"] = [torch.from_numpy(arr) for arr in model_outputs]

print("Input:", E2Eamdshark_CHECK["input"])
print("Output:", E2Eamdshark_CHECK["output"])
```

For some models, it may be fun to apply some post-processing. Here is an example contained in many model.py files for vision models:

```python
# Post process output to do:
# sort(topk(torch.nn.functional.softmax(output, 0), 2)[1])[0]
# Top most probability
E2Eamdshark_CHECK["postprocess"] = [
    (torch.nn.functional.softmax, [0], False, 0),
    (torch.topk, [2], True, 1),
    (torch.sort, [], True, 0),
]
```