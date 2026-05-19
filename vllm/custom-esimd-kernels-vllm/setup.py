import os
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension

root = Path(__file__).parent.resolve()

import torch
torch_include = str(Path(torch.__file__).parent / "include")

# Honor TORCH_XPU_ARCH_LIST for the AOT-compiled SPIR64 extensions below.
# Default to "bmg" so existing BMG-targeted builds are unchanged. Set
# TORCH_XPU_ARCH_LIST=ptl-u (or "bmg,ptl-u") to build for PTL Xe3 iGPU.
_AOT_DEVICE = os.environ.get("TORCH_XPU_ARCH_LIST", "bmg")

ext_modules = [
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels",
        sources=[
            "csrc/xpu/esimd_kernel.sycl",
            "csrc/xpu/torch_extension.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
]

### for lgrf esimd kernels (GDN conv fused — separate module, doubleGRF)
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels_lgrf",
        sources=[
            "csrc/xpu/esimd_kernel_lgrf.sycl",
            "csrc/xpu/torch_extension_lgrf.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen", "-Xs", f"-device {_AOT_DEVICE}",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### for lgrf esimd kernels

### MoE auxiliary kernels — no DPAS, standard compilation
# [skip-ptl-moe-aux] ext_modules.append(
# [skip-ptl-moe-aux]     SyclExtension(
# [skip-ptl-moe-aux]         name="custom_esimd_kernels_vllm.custom_esimd_kernels_moe",
# [skip-ptl-moe-aux]         sources=[
# [skip-ptl-moe-aux]             "csrc/xpu/esimd_kernel_moe.sycl",
# [skip-ptl-moe-aux]             "csrc/xpu/torch_extension_moe.cc",
# [skip-ptl-moe-aux]         ],
# [skip-ptl-moe-aux]         include_dirs=[
# [skip-ptl-moe-aux]             root / "include",
# [skip-ptl-moe-aux]             root / "csrc",
# [skip-ptl-moe-aux]         ],
# [skip-ptl-moe-aux]         extra_compile_args={
# [skip-ptl-moe-aux]             "cxx": ["-O3", "-std=c++17"],
# [skip-ptl-moe-aux]             "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
# [skip-ptl-moe-aux]                      f"-I{torch_include}"],
# [skip-ptl-moe-aux]         },
# [skip-ptl-moe-aux]         extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
# [skip-ptl-moe-aux]         py_limited_api=False,
# [skip-ptl-moe-aux]     )
# [skip-ptl-moe-aux] )
### MoE auxiliary kernels

### FP8 GEMM (M>1) — uses DPAS, compile with JIT only (no AOT to avoid device mismatch)
# [skip-ptl-fp8-gemm] ext_modules.append(
# [skip-ptl-fp8-gemm]     SyclExtension(
# [skip-ptl-fp8-gemm]         name="custom_esimd_kernels_vllm.custom_esimd_kernels_gemm",
# [skip-ptl-fp8-gemm]         sources=[
# [skip-ptl-fp8-gemm]             "csrc/xpu/esimd_kernel_gemm.sycl",
# [skip-ptl-fp8-gemm]             "csrc/xpu/torch_extension_gemm.cc",
# [skip-ptl-fp8-gemm]         ],
# [skip-ptl-fp8-gemm]         include_dirs=[
# [skip-ptl-fp8-gemm]             root / "include",
# [skip-ptl-fp8-gemm]             root / "csrc",
# [skip-ptl-fp8-gemm]         ],
# [skip-ptl-fp8-gemm]         extra_compile_args={
# [skip-ptl-fp8-gemm]             "cxx": ["-O3", "-std=c++17"],
# [skip-ptl-fp8-gemm]             "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
# [skip-ptl-fp8-gemm]                      f"-I{torch_include}"],
# [skip-ptl-fp8-gemm]         },
# [skip-ptl-fp8-gemm]         extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
# [skip-ptl-fp8-gemm]         py_limited_api=False,
# [skip-ptl-fp8-gemm]     )
# [skip-ptl-fp8-gemm] )
### FP8 GEMM kernels

### TopK V2 — vectorized softmax+topk for 512 experts (AOT for BMG)
# [skip-ptl-topk-v2] ext_modules.append(
# [skip-ptl-topk-v2]     SyclExtension(
# [skip-ptl-topk-v2]         name="custom_esimd_kernels_vllm.esimd_topk_v2",
# [skip-ptl-topk-v2]         sources=[
# [skip-ptl-topk-v2]             "csrc/xpu/esimd_kernel_topk_v2.sycl",
# [skip-ptl-topk-v2]             "csrc/xpu/torch_extension_topk_v2.cc",
# [skip-ptl-topk-v2]         ],
# [skip-ptl-topk-v2]         include_dirs=[
# [skip-ptl-topk-v2]             root / "include",
# [skip-ptl-topk-v2]             root / "csrc",
# [skip-ptl-topk-v2]         ],
# [skip-ptl-topk-v2]         extra_compile_args={
# [skip-ptl-topk-v2]             "cxx": ["-O3", "-std=c++17"],
# [skip-ptl-topk-v2]             "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
# [skip-ptl-topk-v2]                      "-fsycl-targets=spir64_gen", "-Xs", "-device bmg",
# [skip-ptl-topk-v2]                      f"-I{torch_include}"],
# [skip-ptl-topk-v2]         },
# [skip-ptl-topk-v2]         extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
# [skip-ptl-topk-v2]         py_limited_api=False,
# [skip-ptl-topk-v2]     )
# [skip-ptl-topk-v2] )
### TopK V2 kernels

### Eagle kernels (GDN + Page Attention) — from custom-esimd-kernels-vllm-eagle
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.eagle_ops",
        sources=[
            "csrc/eagle/eagle.sycl",
        ],
        include_dirs=[
            root / "csrc" / "eagle",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### Eagle kernels

### MoE Batch kernels (Router, TopK, Up/Down, Accumulate) — FP8
# [skip-ptl-moe-batch] ext_modules.append(
# [skip-ptl-moe-batch]     SyclExtension(
# [skip-ptl-moe-batch]         name="custom_esimd_kernels_vllm.moe_ops",
# [skip-ptl-moe-batch]         sources=[
# [skip-ptl-moe-batch]             "csrc/moe_batch/moe.sycl",
# [skip-ptl-moe-batch]         ],
# [skip-ptl-moe-batch]         include_dirs=[
# [skip-ptl-moe-batch]             root / "csrc" / "moe_batch",
# [skip-ptl-moe-batch]         ],
# [skip-ptl-moe-batch]         extra_compile_args={
# [skip-ptl-moe-batch]             "cxx": ["-O3", "-std=c++20"],
# [skip-ptl-moe-batch]             "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
# [skip-ptl-moe-batch]                      f"-I{torch_include}"],
# [skip-ptl-moe-batch]         },
# [skip-ptl-moe-batch]         extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
# [skip-ptl-moe-batch]         py_limited_api=False,
# [skip-ptl-moe-batch]     )
# [skip-ptl-moe-batch] )
### MoE Batch kernels (FP8)

### MoE INT4 Batch kernels (Router, TopK, Up/Down, Finalize) — INT4
# Re-enabled on PTL (XeLPG): the kernel imports the xmx namespace but
# doesn't actually issue dpas instructions, so it builds and runs on PTL.
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.moe_int4_ops",
        sources=[
            "csrc/moe_batch/moe_int4.sycl",
        ],
        include_dirs=[
            root / "csrc" / "moe_batch",
            root / "csrc" / "xpu" / "esimd_kernels",  # for moe_ops.h (TopK V2)
            root / "csrc",  # for relative includes
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### MoE INT4 Batch kernels

setup(
    name="custom-esimd-kernels-vllm",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
