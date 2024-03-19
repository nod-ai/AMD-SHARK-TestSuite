import os

def launchCommand(scriptcommand):
    print("Launching:", scriptcommand, "[ Proc:", os.getpid(), "]")
    try:
        ret = os.system(scriptcommand)
        return ret
    except OSError as errormsg:
        print(
            "Invoking ",
            scriptcommand,
            " failed:",
            errormsg,
        )
        return 1
    
def main():
    scriptcommand = (
        "python /home/sai/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/sdxl_compiled_pipeline.py --precision=fp32 --input_mlir=test-run/pytorch/models/vae-decode/vae_decode.mlir,,test-run/pytorch/models/scheduled-unet/scheduled_unet.mlir --device=cpu --rt_device=local-task --iree_target_triple=x86_64-linux-gnu --num_inference_steps=30 1> sdxl.log 2>&1"
    )
    launchCommand(scriptcommand=scriptcommand)

if __name__ == "__main__":
    main()
