import numpy, torch, sys
import onnxruntime
import onnx
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed, fix_output_shapes

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

import logging

# The generated or checked in onnx file must always be called model.onnx
# the tools/stubs/onnxmodel.py is appended to model.py
# to form runmodel.py in the rundirectory which is then taken
# through flow


def run(run_as_static: bool, logfile: str, verbose: bool):
    logger = logging.getLogger(__name__)
    if verbose:
        logging.basicConfig(filename=logfile, filemode="w", level=logging.INFO)
    dim_params = (
        "batch_size",
        "sequence_length",
        "past_sequence_length",
        "past_sequence_length + 1",
    )
    dim_values = (12, 1, 0, 1)

    pv_zip = zip(dim_params, dim_values)
    pv = dict(pv_zip)
    model = onnx.load("model.onnx")
    is_dynamic = bool(model.graph.input[0].type.tensor_type.shape.dim[0].dim_param)

    if run_as_static and is_dynamic:
        logger.info("Saving dynamic model...")
        onnx.save(model, "dynamic_model.onnx")
        logger.info("\t dynamic model saved as dynamic_model.onnx")

        logger.info("Setting model dim_params:")
        for p in dim_params:
            make_dim_param_fixed(model.graph, p, pv[p])
            logger.info(f"\t {p} = {pv[p]}")

        fix_output_shapes(model)
        logger.info("Overwriting file contents of model.onnx...")
        onnx.save(model, "model.onnx")
        logger.info("\t model.onnx is now a static model.")
    if not run_as_static and not is_dynamic:
        logger.info(
            "model.onnx is currently static. \nAttempting to retrieve dynamic model from dynamic_model.onnx..."
        )
        try:
            model = onnx.load("dynamic_model.onnx")
            logger.info(
                "\t dynamic model found! \nSaving dynamic model as model.onnx..."
            )
            onnx.save(model, "model.onnx")
            logger.info("\t dynamic model saved as model.onnx.")
        except Exception as e:
            logger.warning(
                f"\tError: {e.with_traceback} \nNo dynamic_model.onnx found. Please re-download the dynamic model. Test will proceed with static model."
            )

    # get some model inputs
    model_inputs = [
        numpy.random.randint(
            -1000,
            high=1000,
            size=(pv["batch_size"], pv["sequence_length"]),
            dtype=numpy.int64,
        )
    ]  # input_ids
    model_inputs.append(
        numpy.random.randint(
            -10,
            high=10,
            size=(pv["batch_size"], pv["past_sequence_length + 1"]),
            dtype=numpy.int64,
        )
    )  # attention_mask
    for i in range(2 * 12):
        model_inputs.append(
            numpy.random.rand(
                pv["batch_size"], 12, pv["past_sequence_length"], 64
            ).astype(numpy.float32)
        )  # 12 key/value pairs

    # start a session
    session = onnxruntime.InferenceSession("model.onnx", None)

    # get inputs
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    # run model
    model_outputs = session.run(
        [output.name for output in outputs],
        {inputs[i].name: model_inputs[i] for i in range(len(inputs))},
    )

    E2ESHARK_CHECK["input"] = [torch.from_numpy(arr) for arr in model_inputs]
    E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_outputs]

    print("Input:", E2ESHARK_CHECK["input"])
    print("Output:", E2ESHARK_CHECK["output"])
