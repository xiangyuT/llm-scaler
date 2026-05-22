#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

#include "kernel_ops.h"

TORCH_LIBRARY(custom_esimd_kernels_vllm, m) {
  m.def("esimd_gemv_fp8_pern(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output, int N, int K) -> Tensor");
  m.impl("esimd_gemv_fp8_pern", torch::kXPU, &esimd_gemv_fp8_pern);

  m.def("esimd_gemv_fp8_pern_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, int N0, "
        "Tensor w1, Tensor s1, Tensor o1, int N1, "
        "int K) -> Tensor");
  m.impl("esimd_gemv_fp8_pern_fused2", torch::kXPU, &esimd_gemv_fp8_pern_fused2);

  m.def("esimd_gemv_fp8_pern_fused3(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, int N0, "
        "Tensor w1, Tensor s1, Tensor o1, int N1, "
        "Tensor w2, Tensor s2, Tensor o2, int N2, "
        "int K) -> Tensor");
  m.impl("esimd_gemv_fp8_pern_fused3", torch::kXPU, &esimd_gemv_fp8_pern_fused3);

  // Per-tensor scale variants (N/K inferred from weight shape)
  m.def("esimd_gemv_fp8_pert(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_fp8_pert", torch::kXPU, &esimd_gemv_fp8_pert);

  m.def("esimd_gemv_fp8_pert_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1) -> Tensor");
  m.impl("esimd_gemv_fp8_pert_fused2", torch::kXPU, &esimd_gemv_fp8_pert_fused2);

  m.def("esimd_gemv_fp8_pert_fused3(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1, "
        "Tensor w2, Tensor s2, Tensor o2) -> Tensor");
  m.impl("esimd_gemv_fp8_pert_fused3", torch::kXPU, &esimd_gemv_fp8_pert_fused3);

  // INT4 GEMV with per-group scale (group_size=128)
  // Weight [N, K/2] uint8 packed, scale [N, K/128] fp16. N/K auto-detected.
  m.def("esimd_gemv_int4(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor output) -> Tensor");
  m.impl("esimd_gemv_int4", torch::kXPU, &esimd_gemv_int4);

  // Fused 2-matrix INT4 GEMV (GDN in_proj_qkvz + in_proj_ba)
  m.def("esimd_gemv_int4_fused2(Tensor input, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1) -> Tensor");
  m.impl("esimd_gemv_int4_fused2", torch::kXPU, &esimd_gemv_int4_fused2);

  // Small-M INT4 GEMM (decode bsz>1). Same canonical layout, M in [1,4].
  m.def("esimd_gemm_int4_smallM(Tensor input, Tensor weight, "
        "Tensor weight_scale, Tensor output) -> Tensor");
  m.impl("esimd_gemm_int4_smallM", torch::kXPU, &esimd_gemm_int4_smallM);

  // Fused QKV Split + RMSNorm + RoPE
  m.def("esimd_qkv_split_norm_rope(Tensor qkv_state, "
        "Tensor q_out, Tensor gate_out, Tensor k_out, Tensor v_out, "
        "Tensor norm_wq, Tensor norm_wk, Tensor positions, "
        "int q_heads, int kv_heads, bool attn_output_gate, "
        "int rotary_dim, Tensor cos_sin_cache) -> Tensor");
  m.impl("esimd_qkv_split_norm_rope", torch::kXPU, &esimd_qkv_split_norm_rope);

  // Fused ResidualAdd + RMSNorm + FP8 GEMV (post_attn_norm + router)
  m.def("esimd_resadd_norm_gemv_fp8_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor output, Tensor normed_out, float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv_fp8_pert", torch::kXPU, &esimd_resadd_norm_gemv_fp8_pert);

  // Fused ResidualAdd + RMSNorm + 2-matrix FP8 GEMV (input_norm + GDN in_proj)
  m.def("esimd_resadd_norm_gemv2_fp8_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, "
        "Tensor w0, Tensor s0, Tensor o0, "
        "Tensor w1, Tensor s1, Tensor o1, "
        "float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv2_fp8_pert", torch::kXPU, &esimd_resadd_norm_gemv2_fp8_pert);

  // Fused RMSNormGated + FP8 GEMV (out_proj for GDN layers)
  m.def("esimd_norm_gemv_fp8_pert(Tensor x, Tensor z, Tensor norm_weight, "
        "Tensor gemv_weight, Tensor gemv_scale, Tensor output, "
        "int HV, int V, float eps) -> Tensor");
  m.impl("esimd_norm_gemv_fp8_pert", torch::kXPU, &esimd_norm_gemv_fp8_pert);

  // Fused ResidualAdd + RMSNorm + INT4 GEMV (post_attn_norm + router)
  m.def("esimd_resadd_norm_gemv_int4_pert(Tensor hidden_states, Tensor residual, "
        "Tensor norm_weight, Tensor gemv_weight, Tensor gemv_scale, "
        "Tensor output, Tensor normed_out, float eps) -> Tensor");
  m.impl("esimd_resadd_norm_gemv_int4_pert", torch::kXPU, &esimd_resadd_norm_gemv_int4_pert);

  // Fused RMSNormGated + INT4 GEMV (out_proj for GDN layers)
  m.def("esimd_norm_gemv_int4_pert(Tensor x, Tensor z, Tensor norm_weight, "
        "Tensor gemv_weight, Tensor gemv_scale, Tensor output, "
        "int HV, int V, float eps) -> Tensor");
  m.impl("esimd_norm_gemv_int4_pert", torch::kXPU, &esimd_norm_gemv_int4_pert);

  m.def("esimd_fused_add_rms_norm(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_fused_add_rms_norm", torch::kXPU, &esimd_fused_add_rms_norm);

  m.def("esimd_rms_norm_gated(Tensor x, Tensor z, Tensor weight, "
        "Tensor output, float eps) -> Tensor");
  m.impl("esimd_rms_norm_gated", torch::kXPU, &esimd_rms_norm_gated);

  m.def("esimd_fused_add_rms_norm_batched(Tensor hidden_states, Tensor residual, "
        "Tensor weight, float eps) -> Tensor");
  m.impl("esimd_fused_add_rms_norm_batched", torch::kXPU, &esimd_fused_add_rms_norm_batched);

  // ============ PTL non-DPAS replacements (XE2-only AOT bypass) ============

  // SDPA Decode (non-paged)
  m.def("esimd_sdpa_decode(Tensor q, Tensor k, Tensor v, "
        "bool is_causal, float? scale) -> Tensor");
  m.impl("esimd_sdpa_decode", torch::kXPU, &esimd_sdpa_decode);

  // SDPA Decode Varlen (paged KV cache) — replaces _vllm_fa2_C.varlen_fwd
  m.def("esimd_sdpa_decode_varlen(Tensor q, Tensor key_cache, "
        "Tensor value_cache, Tensor cu_seqlens_q, "
        "int max_seqlen_k, bool is_causal, float? scale, "
        "Tensor block_table, Tensor? seqused_k) -> Tensor");
  m.impl("esimd_sdpa_decode_varlen", torch::kXPU, &esimd_sdpa_decode_varlen);

  // SDPA Prefill DPAS (HD=256 only) — FA-2 style block-attention
  m.def("esimd_sdpa_prefill_dpas(Tensor q, Tensor key_cache, "
        "Tensor value_cache, Tensor cu_seqlens_q, Tensor seq_lens, "
        "bool is_causal, float? scale, Tensor block_table) -> Tensor");
  m.impl("esimd_sdpa_prefill_dpas", torch::kXPU, &esimd_sdpa_prefill_dpas);

  // GDN Attention — replaces _xpu_C.gdn_attention
  m.def("esimd_gdn_attention(Tensor core_attn_out, Tensor z, "
        "Tensor projected_states_qkvz, Tensor projected_states_ba, "
        "int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim, "
        "Tensor conv_state, Tensor ssm_state, "
        "Tensor conv_weights, Tensor? conv_bias, "
        "str activation, Tensor A_log, Tensor dt_bias, "
        "int num_prefills, int num_decodes, "
        "Tensor? has_initial_state, "
        "Tensor non_spec_query_start_loc, Tensor non_spec_state_indices_tensor, "
        "int num_actual_tokens, int tp_size, bool reorder_input) -> ()");
  m.impl("esimd_gdn_attention", torch::kXPU, &esimd_gdn_attention);

  // BF16 GEMV (single matrix)
  m.def("esimd_gemv_bf16(Tensor input, Tensor weight, Tensor output) -> Tensor");
  m.impl("esimd_gemv_bf16", torch::kXPU, &esimd_gemv_bf16);

  // BF16 GEMV fused (2 matrices sharing input)
  m.def("esimd_gemv_bf16_fused2(Tensor input, "
        "Tensor w0, Tensor o0, "
        "Tensor w1, Tensor o1) -> Tensor");
  m.impl("esimd_gemv_bf16_fused2", torch::kXPU, &esimd_gemv_bf16_fused2);

  // Paged KV cache scatter — graph-capture-safe replacement for
  // _C_cache_ops.reshape_and_cache_flash. In-place writes to key_cache /
  // value_cache; declares mutability via Tensor(a!) annotations.
  m.def("esimd_reshape_and_cache_flash(Tensor key, Tensor value, "
        "Tensor(a!) key_cache, Tensor(b!) value_cache, "
        "Tensor slot_mapping) -> ()");
  m.impl("esimd_reshape_and_cache_flash",
         torch::kXPU, &esimd_reshape_and_cache_flash);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels() {
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "custom_esimd_kernels", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
