#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

#include "kernel_ops.h"

TORCH_LIBRARY(custom_esimd_kernels_vllm, m) {
  // All GEMV variants write into their `output`/`o*` tensors in-place and
  // return them as aliases. Each output gets its own alias label so
  // auto_functionalized_v2 can track them independently.
  m.def("esimd_gemv_fp8_pern(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor(a!) output, int N, int K) -> ()");
  m.impl("esimd_gemv_fp8_pern", torch::kXPU, &esimd_gemv_fp8_pern);

  m.def("esimd_gemv_fp8_pern_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor(a!) o0, int N0, "
        "Tensor w1, Tensor s1, Tensor(b!) o1, int N1, "
        "int K) -> ()");
  m.impl("esimd_gemv_fp8_pern_fused2", torch::kXPU, &esimd_gemv_fp8_pern_fused2);

  m.def("esimd_gemv_fp8_pern_fused3(Tensor input, "
        "Tensor w0, Tensor s0, Tensor(a!) o0, int N0, "
        "Tensor w1, Tensor s1, Tensor(b!) o1, int N1, "
        "Tensor w2, Tensor s2, Tensor(c!) o2, int N2, "
        "int K) -> ()");
  m.impl("esimd_gemv_fp8_pern_fused3", torch::kXPU, &esimd_gemv_fp8_pern_fused3);

  // Per-tensor scale variants (N/K inferred from weight shape)
  m.def("esimd_gemv_fp8_pert(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor(a!) output) -> ()");
  m.impl("esimd_gemv_fp8_pert", torch::kXPU, &esimd_gemv_fp8_pert);

  m.def("esimd_gemv_fp8_pert_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor(a!) o0, "
        "Tensor w1, Tensor s1, Tensor(b!) o1) -> ()");
  m.impl("esimd_gemv_fp8_pert_fused2", torch::kXPU, &esimd_gemv_fp8_pert_fused2);

  m.def("esimd_gemv_fp8_pert_fused3(Tensor input, "
        "Tensor w0, Tensor s0, Tensor(a!) o0, "
        "Tensor w1, Tensor s1, Tensor(b!) o1, "
        "Tensor w2, Tensor s2, Tensor(c!) o2) -> ()");
  m.impl("esimd_gemv_fp8_pert_fused3", torch::kXPU, &esimd_gemv_fp8_pert_fused3);

  // INT4 GEMV with per-group scale (group_size=128)
  // Weight [N, K/2] uint8 packed, scale [N, K/128] fp16. N/K auto-detected.
  m.def("esimd_gemv_int4(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor(a!) output) -> ()");
  m.impl("esimd_gemv_int4", torch::kXPU, &esimd_gemv_int4);

  // Fused 2-matrix INT4 GEMV (GDN in_proj_qkvz + in_proj_ba)
  m.def("esimd_gemv_int4_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor(a!) o0, "
        "Tensor w1, Tensor s1, Tensor(b!) o1) -> ()");
  m.impl("esimd_gemv_int4_fused2", torch::kXPU, &esimd_gemv_int4_fused2);

  // Fused QKV Split + RMSNorm + RoPE
  // Writes into q_out, gate_out, k_out, v_out. Each gets its own alias
  // label (a!, b!, c!, d!) so functionalization can track them
  // independently. Return value aliases q_out (label a!).
  m.def("esimd_qkv_split_norm_rope(Tensor qkv_state, "
        "Tensor(a!) q_out, Tensor(b!) gate_out, "
        "Tensor(c!) k_out, Tensor(d!) v_out, "
        "Tensor norm_wq, Tensor norm_wk, Tensor positions, "
        "int q_heads, int kv_heads, bool attn_output_gate, "
        "int rotary_dim, Tensor cos_sin_cache) -> ()");
  m.impl("esimd_qkv_split_norm_rope", torch::kXPU, &esimd_qkv_split_norm_rope);

  // Fused ResidualAdd + RMSNorm + FP8 GEMV (post_attn_norm + router)
  // hidden_states and residual are updated in-place by the RMSNorm step;
  // output and normed_out are filled by the GEMV. Four independent aliases.
  // Return value aliases output (label c!).
  m.def("esimd_resadd_norm_gemv_fp8_pert(Tensor(a!) hidden_states, Tensor(b!) residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor(c!) output, Tensor(d!) normed_out, float eps) -> ()");
  m.impl("esimd_resadd_norm_gemv_fp8_pert", torch::kXPU, &esimd_resadd_norm_gemv_fp8_pert);

  // Fused ResidualAdd + RMSNorm + 2-matrix FP8 GEMV (input_norm + GDN in_proj)
  // Four mutable outputs: hidden_states, residual, o0, o1.
  m.def("esimd_resadd_norm_gemv2_fp8_pert(Tensor(a!) hidden_states, Tensor(b!) residual, "
        "Tensor norm_weight, "
        "Tensor w0, Tensor s0, Tensor(c!) o0, "
        "Tensor w1, Tensor s1, Tensor(d!) o1, "
        "float eps) -> ()");
  m.impl("esimd_resadd_norm_gemv2_fp8_pert", torch::kXPU, &esimd_resadd_norm_gemv2_fp8_pert);

  // Fused RMSNormGated + FP8 GEMV (out_proj for GDN layers)
  m.def("esimd_norm_gemv_fp8_pert(Tensor x, Tensor z, Tensor norm_weight, "
        "Tensor gemv_weight, Tensor gemv_scale, Tensor(a!) output, "
        "int HV, int V, float eps) -> ()");
  m.impl("esimd_norm_gemv_fp8_pert", torch::kXPU, &esimd_norm_gemv_fp8_pert);

  // Fused ResidualAdd + RMSNorm + INT4 GEMV (post_attn_norm + router)
  m.def("esimd_resadd_norm_gemv_int4_pert(Tensor(a!) hidden_states, Tensor(b!) residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor(c!) output, Tensor(d!) normed_out, float eps) -> ()");
  m.impl("esimd_resadd_norm_gemv_int4_pert", torch::kXPU, &esimd_resadd_norm_gemv_int4_pert);

  // Fused RMSNormGated + INT4 GEMV (out_proj for GDN layers)
  m.def("esimd_norm_gemv_int4_pert(Tensor x, Tensor z, Tensor norm_weight, "
        "Tensor gemv_weight, Tensor gemv_scale, Tensor(a!) output, "
        "int HV, int V, float eps) -> ()");
  m.impl("esimd_norm_gemv_int4_pert", torch::kXPU, &esimd_norm_gemv_int4_pert);

  // Single-row RMSNorm variant — overwrites hidden_states + residual.
  m.def("esimd_fused_add_rms_norm(Tensor(a!) hidden_states, Tensor(b!) residual, "
        "Tensor weight, float eps) -> ()");
  m.impl("esimd_fused_add_rms_norm", torch::kXPU, &esimd_fused_add_rms_norm);

  // Writes norm(x) * z into output.
  m.def("esimd_rms_norm_gated(Tensor x, Tensor z, Tensor weight, "
        "Tensor(a!) output, float eps) -> ()");
  m.impl("esimd_rms_norm_gated", torch::kXPU, &esimd_rms_norm_gated);

  // In-place variant: overwrites hidden_states with the normed output and
  // updates residual (x + residual). Return value aliases hidden_states.
  m.def("esimd_fused_add_rms_norm_batched(Tensor(a!) hidden_states, Tensor(b!) residual, "
        "Tensor weight, float eps) -> ()");
  m.impl("esimd_fused_add_rms_norm_batched", torch::kXPU, &esimd_fused_add_rms_norm_batched);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels() {
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "custom_esimd_kernels", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
