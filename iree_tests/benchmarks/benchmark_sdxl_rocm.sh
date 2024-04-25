#!/bin/bash
#
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

THIS_DIR="$(cd $(dirname $0) && pwd)"
IREE_ROOT="$(cd ${THIS_DIR?}/.. && pwd)"
VAE_DECODE_DIR="${IREE_ROOT?}/pytorch/models/sdxl-vae-decode-tank"
SCHEDULED_UNET_DIR="${IREE_ROOT?}/pytorch/models/sdxl-scheduled-unet-3-tank"
PROMPT_ENCODER_DIR="${IREE_ROOT?}/pytorch/models/sdxl-prompt-encoder-tank"

echo "Echo compiling full sdxl pipeline"

iree-compile "${THIS_DIR?}/sdxl_pipeline_bench_f16.mlir" \
  --iree-hal-target-backends=rocm \
  --iree-rocm-target-chip=gfx90a \
  -o "${THIS_DIR?}/sdxl_full_pipeline_fp16_rocm.vmfb"

echo "Running sdxl benchmark"

iree-benchmark-module \
  --device=hip \
  --module="${PROMPT_ENCODER_DIR?}/model_gpu_rocm_real_weights.vmfb" \
  --parameters=model="${PROMPT_ENCODER_DIR?}/real_weights.irpa" \
  --module="${SCHEDULED_UNET_DIR?}/model_gpu_rocm_real_weights.vmfb" \
  --parameters=model="${SCHEDULED_UNET_DIR?}/real_weights.irpa" \
  --module="${VAE_DECODE_DIR?}/model_gpu_rocm_real_weights.vmfb" \
  --parameters=model="${VAE_DECODE_DIR?}/real_weights.irpa" \
  --module="${THIS_DIR?}/sdxl_full_pipeline_fp16_rocm.vmfb" \
  --function=tokens_to_image \
  --input=1x4x128x128xf16 \
  --input=1xf16 \
  --input=1x64xi64 \
  --input=1x64xi64 \
  --input=1x64xi64 \
  --input=1x64xi64 \
  --benchmark_repetitions=3

echo "Succesfully finished sdxl pipeline benchmark"
