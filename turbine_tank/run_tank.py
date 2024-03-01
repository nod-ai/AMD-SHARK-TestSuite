# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import unittest
import model_util
import os
import shutil

import turbine_models.tests.sd_test as sd_test
import os
from turbine_models.turbine_tank import turbine_tank

import pytest

parser = argparse.ArgumentParser()
parser.add_argument(
    "--download_ir",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="download IR from turbine tank to local cache",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.download_ir:
        turbine_tank.downloadModelArtifacts("CompVis/stable-diffusion-v1-4-clip")
        turbine_tank.downloadModelArtifacts("CompVis/stable-diffusion-v1-4-vae-decode")
        turbine_tank.downloadModelArtifacts("CompVis/stable-diffusion-v1-4-vae-encode")
        turbine_tank.downloadModelArtifacts("CompVis/stable-diffusion-v1-4-unet")
        turbine_tank.downloadModelArtifacts(
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2"
        )
        for model_name, _ in model_util.model_list:
            turbine_tank.downloadModelArtifacts(model_name)
    else:
        curr_dir = os.getcwd()
        temp_dir = os.path.join(curr_dir, "temp")
        os.mkdir(temp_dir)
        os.chdir(temp_dir)
        import turbine_models.tests.stateless_llama_test as stateless_llama_test

        # environment variable used to let the llama/sd tests know we are running from tank and want to upload
        os.environ["TURBINE_TANK_ACTION"] = "upload"

        # run existing turbine llama and sd tests integrated with turbine tank
        llama_suite = unittest.TestLoader().loadTestsFromModule(stateless_llama_test)
        unittest.TextTestRunner(verbosity=2).run(llama_suite)

        sd_suite = unittest.TestLoader().loadTestsFromModule(sd_test)
        unittest.TextTestRunner(verbosity=2).run(sd_suite)

        # cleanup
        os.chdir(curr_dir)
        shutil.rmtree(temp_dir)

        # runs tank_test.py (only pytest file in this directory, runs 30 models e2e)
        pytest.main(["-v", os.path.dirname(os.path.abspath(__file__))])
