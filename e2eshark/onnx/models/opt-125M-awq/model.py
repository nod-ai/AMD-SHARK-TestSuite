import numpy, torch, sys
import onnxruntime
import onnx

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)


# The generated or checked in onnx file must always be called model.onnx
# the tools/stubs/onnxmodel.py is appended to model.py
# to form runmodel.py in the rundirectory which is then taken
# through flow

# set some variable parameters (dynamic dims)
batch_size = 3 
sequence_length = 7 
past_sequence_length = 0 # setting this parameter to anything other than 0 results in an error

# get some model inputs
model_inputs = [numpy.random.rand(batch_size, sequence_length).astype(numpy.int64)] # input_ids
model_inputs.append(numpy.random.rand(batch_size, past_sequence_length + 1).astype(numpy.int64)) # attention_mask
for i in range(2*12):
    model_inputs.append(numpy.random.rand(batch_size, 12, past_sequence_length, 64).astype(numpy.float32)) # 12 key/value pairs

# start a session
session = onnxruntime.InferenceSession("model.onnx", None)

# get inputs
inputs = session.get_inputs()
# for input in inputs:
#     print(input)
outputs = session.get_outputs()

model_outputs = session.run(
    [output.name for output in outputs],
    {inputs[i].name: model_inputs[i] for i in range(26)},
)

E2ESHARK_CHECK["input"] = [torch.from_numpy(arr) for arr in model_inputs]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_outputs]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])

