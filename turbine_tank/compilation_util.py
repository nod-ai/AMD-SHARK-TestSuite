import iree.compiler as ireec

def compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name):
    flags = [
        "--iree-input-type=torch",
        "--mlir-print-debuginfo",
        "--mlir-print-op-on-diagnostic=false",
        "--iree-llvmcpu-target-cpu-features=host",
    ]
    if device == "cpu":
        flags.append("--iree-llvmcpu-enable-ukernels=all")
        device = "llvm-cpu"
    elif device == "vulkan":
        flags.extend(
            [
                "--iree-hal-target-backends=vulkan-spirv",
                "--iree-vulkan-target-triple=" + target_triple,
                "--iree-stream-resource-max-allocation-size=" + max_alloc,
            ]
        )
    elif device == "rocm":
        flags.extend(
            [
                "--iree-hal-target-backends=rocm",
                "--iree-rocm-target-chip=" + target_triple,
                "--iree-rocm-link-bc=true",
                "--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode",
            ]
        )
    elif device == "cuda":
        flags.extend(
            [
                "--iree-hal-target-backends=cuda",
                "--iree-hal-cuda-llvm-target-arch=" + target_triple,
            ]
        )
    else:
        raise ValueError(f"{device} has to be one of [cpu, vulkan, rocm, cuda]")

    flatbuffer_blob = ireec.compile_str(
        module_str,
        target_backends=[device],
        extra_args=flags,
    )
    with open(f"{safe_name}.vmfb", "wb+") as f:
        f.write(flatbuffer_blob)
    print("Saved to", safe_name + ".vmfb")
