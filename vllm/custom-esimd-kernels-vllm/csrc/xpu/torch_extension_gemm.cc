/* torch_extension_gemm.cc — Op registration for FP8 GEMM (M>1) ESIMD kernels.
 * Uses TORCH_LIBRARY_FRAGMENT to add ops to the existing library namespace.
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

#include "kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m) {
  // FP8 GEMM per-tensor scale: input [M, K] fp16, weight [N, K] fp8, output [M, N] fp16
  // Auto-dispatches based on M: GEMV for M<=3, DPAS for M>=2 (E4M3), WS fallback
  // Mutation annotation: the impl writes into `output` and returns it as
  // an alias. Declaring `(a!)` on both the input and return lets
  // torch.compile / AOT functionalization trace this correctly and
  // produce a valid cudagraph. Without the annotation, functionalization
  // sees it as a pure op and fails with "output spec mismatch".
  m.def("esimd_gemm_fp8_pert(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor(a!) output) -> ()");
  m.impl("esimd_gemm_fp8_pert", torch::kXPU, &esimd_gemm_fp8_pert);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels_gemm() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "custom_esimd_kernels_gemm", nullptr, 0, nullptr
    };
    return PyModule_Create(&module);
}
