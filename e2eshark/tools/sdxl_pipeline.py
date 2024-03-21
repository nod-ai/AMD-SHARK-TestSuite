import os
import sys
import argparse

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--turbinedirectory",
        default="~/SHARK-Turbine",
        help="The test run directory",
    )
    parser.add_argument(
        "-r",
        "--rundirectory",
        default="test-run",
        help="The test run directory",
    )
    args = parser.parse_args()
    TURBINE_PATH = ""
    RUN_DIR_PATH = ""
    if args.turbinedirectory:
        TURBINE_PATH = args.turbinedirectory
        TURBINE_PATH = os.path.expanduser(TURBINE_PATH)
        TURBINE_PATH = os.path.abspath(TURBINE_PATH)
        if not os.path.exists(TURBINE_PATH):
            print(
                "ERROR: Turbine directory",
                TURBINE_PATH,
                "does not exist.",
            )
            sys.exit(1)

    if args.rundirectory:
        RUN_DIR_PATH = args.rundirectory
        RUN_DIR_PATH = os.path.expanduser(RUN_DIR_PATH)
        RUN_DIR_PATH = os.path.abspath(RUN_DIR_PATH)
        if not os.path.exists(RUN_DIR_PATH):
            print("ERROR: Run directory", RUN_DIR_PATH, "does not exist.")
            sys.exit(1)
    scriptcommand = (
        "python " + str(TURBINE_PATH) + "/models/turbine_models/custom_models/sdxl_inference/sdxl_compiled_pipeline.py --precision=fp32 --input_mlir=" + str(RUN_DIR_PATH) + "/pytorch/models/vae_decode/vae_decode.default.pytorch.torch.mlir," + str(RUN_DIR_PATH) +  "/pytorch/models/clip_prompt_encoder/clip_prompt_encoder.default.pytorch.torch.mlir," + str(RUN_DIR_PATH) + "/pytorch/models/scheduled_unet/scheduled_unet.default.pytorch.torch.mlir --device=cpu --rt_device=local-task --iree_target_triple=x86_64-linux-gnu --num_inference_steps=30 1> sdxl.log 2>&1"
    )
    launchCommand(scriptcommand=scriptcommand)

if __name__ == "__main__":
    main()
