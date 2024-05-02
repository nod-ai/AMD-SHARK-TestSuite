import numpy, torch, sys
import onnxruntime

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


# start an onnxrt session
session = onnxruntime.InferenceSession("model.onnx", None)

# Even if model is quantized, the inputs and outputs are
# not, so apply float32
model_input_X = numpy.random.rand(1, 3, 112, 224, 224).astype(numpy.float32)

# gets X in inputs[0] and Y in inputs[1]
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()


model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_X},
)[0]
E2ESHARK_CHECK["input"] = [torch.from_numpy(model_input_X)]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])

# Post process output to do:
E2ESHARK_CHECK["postprocess"] = [
    (torch.nn.functional.softmax, [0], False, 0),
    (torch.topk, [1], True, 1),
]
