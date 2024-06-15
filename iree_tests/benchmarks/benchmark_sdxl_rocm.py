# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from collections import namedtuple
import logging
from typing import Sequence
import subprocess

benchmark_dir = os.path.dirname(os.path.realpath(__file__))
iree_root = os.path.dirname(benchmark_dir)
prompt_encoder_dir = f"{iree_root}/pytorch/models/sd-clip/sdxl-prompt-encoder-tank"
scheduled_unet_dir = f"{iree_root}/pytorch/models/sd-unet/sdxl-scheduled-unet-3-tank"
vae_decode_dir = f"{iree_root}/pytorch/models/sd-vae/sdxl-vae-decode-tank"

def run_iree_command(args: Sequence[str] = ()):
    command = "Exec:", " ".join(args)
    logging.getLogger().info(command)
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stdout_v, stderr_v, = proc.stdout, proc.stderr
    return_code = proc.returncode
    if return_code == 0:
        return 0, proc.stdout
    logging.getLogger().info(f"Command failed with error: {proc.stderr}")
    return 1, proc.stdout

def run_sdxl_rocm_benchmark(rocm_chip, gpu_number):
    exec_args = [
        "iree-compile",
        f"{benchmark_dir}/sdxl_pipeline_bench_f16.mlir",
        "--iree-hal-target-backends=rocm",
        f"--iree-rocm-target-chip={rocm_chip}",
        "--iree-global-opt-propagate-transposes=true",
        "--iree-codegen-llvmgpu-use-vector-distribution",
        "--iree-codegen-gpu-native-math-precision=true",
        "--iree-rocm-waves-per-eu=2",
        "--iree-opt-outer-dim-concat=true",
        "--iree-llvmgpu-enable-prefetch",
        "-o",
        f"{benchmark_dir}/sdxl_full_pipeline_fp16_rocm.vmfb",
    ]
    # iree compile command for full sdxl pipeline
    ret_value, stdout = run_iree_command(exec_args)
    if ret_value == 1:
        return 1, stdout
    exec_args = [
        "iree-benchmark-module",
        f"--device=hip://{gpu_number}",
        "--device_allocator=caching",
        f"--module={prompt_encoder_dir}/model_gpu_rocm_real_weights.vmfb",
        f"--parameters=model={prompt_encoder_dir}/real_weights.irpa",
        f"--module={scheduled_unet_dir}/model_gpu_rocm_real_weights.vmfb",
        f"--parameters=model={scheduled_unet_dir}/real_weights.irpa",
        f"--module={vae_decode_dir}/model_gpu_rocm_real_weights.vmfb",
        f"--parameters=model={vae_decode_dir}/real_weights.irpa",
        f"--module={benchmark_dir}/sdxl_full_pipeline_fp16_rocm.vmfb",
        "--function=tokens_to_image",
        "--input=1x4x128x128xf16",
        "--input=1xf16",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--benchmark_repetitions=3",
    ]
    # iree benchmark command for full sdxl pipeline
    return run_iree_command(exec_args)

def run_sdxl_unet_rocm_benchmark(gpu_number):
    exec_args = [
        "iree-benchmark-module",
        f"--device=hip://{gpu_number}",
        "--device_allocator=caching",
        f"--module={scheduled_unet_dir}/model_gpu_rocm_real_weights.vmfb",
        f"--parameters=model={scheduled_unet_dir}/real_weights.irpa",
        "--function=run_forward",
        "--input=1x4x128x128xf16",
        "--input=2x64x2048xf16",
        "--input=2x1280xf16",
        "--input=2x6xf16",
        "--input=1xf16",
        "--input=1xi64",
        "--benchmark_repetitions=3",
    ]
    # iree benchmark command for full sdxl pipeline
    return run_iree_command(exec_args)

def run_sdxl_prompt_encoder_rocm_benchmark(gpu_number):
    exec_args = [
        "iree-benchmark-module",
        f"--device=hip://{gpu_number}",
        "--device_allocator=caching",
        f"--module={prompt_encoder_dir}/model_gpu_rocm_real_weights.vmfb",
        f"--parameters=model={prompt_encoder_dir}/real_weights.irpa",
        "--function=encode_prompts",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--input=1x64xi64",
        "--benchmark_repetitions=3",
    ]
    # iree benchmark command for full sdxl pipeline
    return run_iree_command(exec_args)

def run_sdxl_vae_decode_rocm_benchmark(gpu_number):
    exec_args = [
        "iree-benchmark-module",
        f"--device=hip://{gpu_number}",
        "--device_allocator=caching",
        f"--module={vae_decode_dir}/model_gpu_rocm_real_weights.vmfb",
        f"--parameters=model={vae_decode_dir}/real_weights.irpa",
        "--function=main",
        "--input=1x4x128x128xf16",
        "--benchmark_repetitions=3",
    ]
    # iree benchmark command for full sdxl pipeline
    return run_iree_command(exec_args)
    

BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)

def decode_output(bench_lines):
    benchmark_results = []
    for line in bench_lines:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark_name = split[0]
        time = " ".join(split[1:3])
        cpu_time = " ".join(split[3:5])
        iterations = split[5]
        user_counters = None
        if len(split) > 5:
            user_counters = split[6]
        benchmark_results.append(
            BenchmarkResult(
                benchmark_name=benchmark_name,
                time=time,
                cpu_time=cpu_time,
                iterations=iterations,
                user_counters=user_counters,
            )
        )
    return benchmark_results

def job_summary_process(ret_value, output):
    if ret_value == 1:
        logging.getLogger().info("Running SDXL ROCm benchmark failed. Exiting")
        return
    bench_lines = output.decode().split("\n")[3:]
    benchmark_results = decode_output(bench_lines)
    logging.getLogger().info(benchmark_results)
    benchmark_mean_time = float(benchmark_results[3].time.split()[0])
    return benchmark_mean_time

def test_sdxl_rocm_benchmark(goldentime_rocm_e2e, goldentime_rocm_unet, 
    goldentime_rocm_clip, goldentime_rocm_vae, gpu_number, rocm_chip):
    # if the benchmark returns 1, benchmark failed
    job_summary_lines = []

    # e2e benchmark
    ret_value, output = run_sdxl_rocm_benchmark(rocm_chip, gpu_number)
    benchmark_mean_time = job_summary_process(ret_value, output)
    assert benchmark_mean_time <= goldentime_rocm_e2e, "SDXL e2e benchmark time should not regress"
    mean_line = (f"E2E Benchmark Time: {str(benchmark_mean_time)} ms"
                 f" (golden time {goldentime_rocm_e2e} ms)")
    job_summary_lines.append(mean_line)
    logging.getLogger().info(mean_line)

    # unet benchmark
    ret_value, output = run_sdxl_unet_rocm_benchmark(gpu_number)
    benchmark_mean_time = job_summary_process(ret_value, output)
    assert benchmark_mean_time <= goldentime_rocm_unet, "SDXL unet benchmark time should not regress"
    mean_line = (f"Scheduled Unet Benchmark Time: {str(benchmark_mean_time)} ms"
                 f" (golden time {goldentime_rocm_unet} ms)")
    job_summary_lines.append(mean_line)
    logging.getLogger().info(mean_line)

    # prompt encoder benchmark
    ret_value, output = run_sdxl_prompt_encoder_rocm_benchmark(gpu_number)
    benchmark_mean_time = job_summary_process(ret_value, output)
    assert benchmark_mean_time <= goldentime_rocm_clip, "SDXL prompt encoder benchmark time should not regress"
    mean_line = (f"Prompt Encoder Benchmark Time: {str(benchmark_mean_time)} ms"
                 f" (golden time {goldentime_rocm_clip} ms)")
    job_summary_lines.append(mean_line)
    logging.getLogger().info(mean_line)

    # vae decode benchmark
    ret_value, output = run_sdxl_vae_decode_rocm_benchmark(gpu_number)
    benchmark_mean_time = job_summary_process(ret_value, output)
    assert benchmark_mean_time <= goldentime_rocm_vae, "SDXL vae decode benchmark time should not regress"
    mean_line = (f"VAE Decode Benchmark Time: {str(benchmark_mean_time)} ms"
                  f" (golden time {goldentime_rocm_vae} ms)")
    job_summary_lines.append(mean_line)
    logging.getLogger().info(mean_line)

    with open('job_summary.txt', 'w') as job_summary_file:
        for line in job_summary_lines:
            job_summary_file.write(line + "\n")
