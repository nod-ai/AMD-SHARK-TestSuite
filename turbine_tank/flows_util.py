from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
from turbine_models.custom_models.sd_inference import utils
from shark_turbine.aot import *
from iree.compiler.ir import Context
from turbine_models.turbine_tank import turbine_tank
import os
import compilation_util

def classic_flow(model, model_name, input, out, run_e2e, expected_err):
    vmfb_name = model_name.replace("/", "_") + ".vmfb"
    model.get_compiled_module(save_to=vmfb_name)

    # if model is not supposed to run e2e, exit at this point (mlir has been uploaded)
    if run_e2e is False:
        assert expected_err > 0
        return

    # run inference using iree runtime
    runner = vmfbRunner("local-task", vmfb_name)
    inputs = [ireert.asdevicearray(runner.config.device, input)]
    keys = list(runner.ctx.modules)
    key = keys[len(keys) - 1]
    results = runner.ctx.modules.__getattr__(key)["main"](*inputs)
    err = utils.largest_error(out.cpu().detach().numpy(), results)
    # cleanup
    os.remove(vmfb_name)
    # accuracy
    assert err < expected_err


def param_flow(model, model_name, model_type, input, out, run_e2e, expected_err):
    weight_name = model_name.replace("/", "_") + ".safetensors"
    mapper = {}
    utils.save_external_weights(mapper, model.model, "safetensors", weight_name)

    # seq2seq models differs from rest as it take two inputs (input_ids, decoder_input_ids)
    if model_type == "hf_seq2seq":

        class Seq2SeqModule(CompiledModule):
            params = export_parameters(
                model.model, external=True, external_scope="", name_mapper=mapper.get
            )

            def main(
                self,
                inp1=AbstractTensor(*(input[0].shape), dtype=input[0].dtype),
                inp2=AbstractTensor(*(input[1].shape), dtype=input[1].dtype),
            ):
                return jittable(model.model.forward)(inp1, inp2)

        inst = Seq2SeqModule(context=Context(), import_to="IMPORT")
        module_str = str(CompiledModule.get_mlir_module(inst))
    else:

        class GlobalModule(CompiledModule):
            params = export_parameters(
                model.model, external=True, external_scope="", name_mapper=mapper.get
            )

            def main(self, inp=AbstractTensor(*input.shape, dtype=input.dtype)):
                return jittable(model.model.forward)(inp)

        inst = GlobalModule(context=Context(), import_to="IMPORT")
        module_str = str(CompiledModule.get_mlir_module(inst))

    mlir_name = model_name.replace("/", "_") + ".mlir"
    with open(mlir_name, "w+") as f:
        f.write(module_str)

    model_name_upload = model_name.replace("/", "_")
    blob_name = turbine_tank.uploadToBlobStorage(
        str(os.path.abspath(mlir_name)),
        f"{model_name_upload}/{model_name_upload}-params.mlir",
    )

    os.remove(mlir_name)

    if run_e2e is False:
        assert expected_err > 0
        return

    vmfb_name = model_name.replace("/", "_")
    compilation_util.compile_to_vmfb(module_str, "cpu", "", "", vmfb_name)

    # run inference using iree runtime
    runner = vmfbRunner("local-task", vmfb_name + ".vmfb", weight_name)
    inputs = [ireert.asdevicearray(runner.config.device, input)]
    keys = list(runner.ctx.modules)
    key = keys[len(keys) - 1]
    results = runner.ctx.modules.__getattr__(key)["main"](*inputs)
    err = utils.largest_error(out.cpu().detach().numpy(), results)

    # clean up
    os.remove(vmfb_name + ".vmfb")
    os.remove(weight_name)

    if err < expected_err:
        new_blob_name = blob_name.split(".")
        new_blob_name = new_blob_name[0] + "-pass.mlir"
        turbine_tank.changeBlobName(blob_name, new_blob_name)

    # accuracy
    assert err < expected_err
