#pragma once
#include "utils.h"
#include "int4_GEMV.h"  // INT4_GROUP_SIZE, int4_dequant<>

// =============================================================================
// Small-M INT4 GEMM (M=1..4): same canonical weight layout as esimd_gemv_int4,
// shared weight load across M input rows for weight-bandwidth amortization.
//
// Why this kernel exists:
//   esimd_gemv_int4 is hard-gated on M==1; vLLM's _esimd_int4_apply falls
//   through to dequant + torch.matmul for M>1, which dequants the entire
//   INT4 weight to fp16 every call and dominates step time at decode bsz>1.
//
//   PTL Xe3 LPG decode at bsz=1 is already at ~50% LPDDR utilization; the
//   weight read is the dominant byte cost. At small M (2..4), an in-kernel
//   "GEMV M times against the same weight tile" amortizes the weight load,
//   so aggregate throughput should scale nearly linearly with M up to LPDDR
//   saturation.
//
// Tensor shapes (same as esimd_gemv_int4 with the M dimension generalized):
//   input:  [M, K]              fp16
//   weight: [N, K/2]            uint8 (packed int4, GGML q4_0)
//   scale:  [N, K/GROUP_SIZE]   fp16
//   output: [M, N]              fp16
//
// WG layout (same as esimd_gemv_int4):
//   1 WG per output row n in {0..N-1}
//   K_SPLIT threads per WG split the K-dim reduction; partial sums reduced
//   via SLM at the end.
//
// Per-iteration FMA structure (per K-step of size VL=128):
//   1. Load weight tile (VL/2 packed bytes) ONCE.
//   2. Unpack to wf_even / wf_odd (float vectors of size VL/2).
//   3. Multiply by per-group scale (1 fp16 load per WG-iteration, broadcast).
//   4. For each m in [0..M):
//        Load M-th input slice [m, k:k+VL] (deinterleaved into iv_even/iv_odd).
//        FMA into M sets of accumulators.
//
// Output write:
//   Each WG (one per n) writes M outputs at output[m*N + n] for m in [0..M).
//   Inside the WG, lid=0 does the cross-lid reduction and the M writes.
//
// Compile-time M:
//   We instantiate M={1,2,3,4} as separate template specializations and
//   dispatch from the host based on the runtime M value. M=1 keeps the
//   existing esimd_gemv_int4 kernel as the canonical fast path; this file
//   covers M=2..4 (and an M=1 instantiation for code-path uniformity in
//   tests/bench).
// =============================================================================

template<int VL, int K_SPLIT, int M>
struct GEMM_int4_smallM_kernel {
    const fp16*    input;     // [M, K] fp16
    const uint8_t* weight;    // [N, K/2] uint8
    const fp16*    scale;     // [N, K/GROUP_SIZE] fp16
    fp16*          output;    // [M, N] fp16
    int N, K;
    int n_groups;             // K / GROUP_SIZE

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        // SLM for cross-thread reduction. M floats per thread.
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * M * sizeof(float)>();
        }

        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;
        int ks = lid * kp;

        // M sets of (even, odd) accumulators. Use C arrays of simd<> to keep
        // the inner loop unrolled by the SYCL compiler.
        simd<float, VL/2> acc_even[M];
        simd<float, VL/2> acc_odd[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            acc_even[m] = 0.0f;
            acc_odd[m]  = 0.0f;
        }

        const uint8_t* w_row = weight + (size_t)n * (K / 2);
        const fp16*    s_row = scale  + (size_t)n * n_groups;

        int group_idx = ks / INT4_GROUP_SIZE;

        for (int k = ks; k < ks + kp; k += VL) {
            // ---- Load weight tile ONCE per K-step (shared across M rows) ----
            simd<uint8_t, VL/2> raw = block_load<uint8_t, VL/2>(w_row + k / 2);

            // Unpack INT4 → two float[VL/2] (wf_even/wf_odd in [-8, 7]).
            simd<float, VL/2> wf_even, wf_odd;
            int4_dequant<VL>(raw, wf_even, wf_odd);

            // Apply scale (broadcast scalar fp16 → float).
            float gs = static_cast<float>(s_row[group_idx]);
            group_idx += VL / INT4_GROUP_SIZE;
            wf_even *= gs;
            wf_odd  *= gs;

            // ---- M independent FMAs against the dequanted weight tile ----
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                simd<fp16, VL> iv = block_load<fp16, VL>(input + (size_t)m * K + k);
                simd<fp16, VL/2> iv_even_h = iv.template select<VL/2, 2>(0).read();
                simd<fp16, VL/2> iv_odd_h  = iv.template select<VL/2, 2>(1).read();
                simd<float, VL/2> in_even = iv_even_h;
                simd<float, VL/2> in_odd  = iv_odd_h;
                acc_even[m] += in_even * wf_even;
                acc_odd[m]  += in_odd  * wf_odd;
            }
        }

        // ---- Reduce M partial sums per thread to scalars ----
        float my_sum[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            my_sum[m] = reduce<float>(acc_even[m], std::plus<>())
                      + reduce<float>(acc_odd[m],  std::plus<>());
        }

        if constexpr (K_SPLIT == 1) {
            // Single thread per WG: write M outputs directly.
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                output[(size_t)m * N + n] = fp16(my_sum[m]);
            }
        } else {
            // Cross-thread reduction via SLM. Layout: M*K_SPLIT floats,
            // [m=0,lid=0..K_SPLIT-1, m=1,lid=0..K_SPLIT-1, ...].
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                slm_block_store<float, 1>(
                    (m * K_SPLIT + lid) * sizeof(float),
                    simd<float, 1>(my_sum[m]));
            }
            barrier();
            if (lid == 0) {
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    simd<float, K_SPLIT> parts =
                        slm_block_load<float, K_SPLIT>(m * K_SPLIT * sizeof(float));
                    output[(size_t)m * N + n] =
                        fp16(reduce<float>(parts, std::plus<>()));
                }
            }
        }
    }
};


// =============================================================================
// Host dispatcher.  Same VL/K_SPLIT heuristic as esimd_gemv_int4
// (select_vl_ks_int4) — the per-output-row work shape doesn't change with M,
// only the per-thread accumulator count and SLM footprint do.
// =============================================================================

inline void GEMM_int4_smallM_host(
    uint8_t* input_data,
    uint8_t* weight_data,
    uint8_t* scale_data,
    uint8_t* output_data,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    sycl::queue& q) {

    auto* p_in  = reinterpret_cast<const fp16*>(input_data);
    auto* p_w   = reinterpret_cast<const uint8_t*>(weight_data);
    auto* p_sc  = reinterpret_cast<const fp16*>(scale_data);
    auto* p_out = reinterpret_cast<fp16*>(output_data);

    int n_groups = K / INT4_GROUP_SIZE;

    int vl, ks;
    select_vl_ks_int4(N, K, vl, ks);

    int global = N * ks;
    int local  = ks;

    // Two-axis dispatch: K_SPLIT in {1,2,4,8} × M in {1,2,3,4}.
    #define LAUNCH_GEMM_INT4(S, MM)                                      \
        q.submit([&](sycl::handler& h) {                                 \
            h.parallel_for(sycl::nd_range<1>(global, local),             \
                GEMM_int4_smallM_kernel<128, S, MM>{                     \
                    p_in, p_w, p_sc, p_out,                              \
                    (int)N, (int)K, n_groups});                          \
        });

    #define DISPATCH_KS(MM) do {                                         \
        if      (ks == 1) { LAUNCH_GEMM_INT4(1, MM); }                   \
        else if (ks == 2) { LAUNCH_GEMM_INT4(2, MM); }                   \
        else if (ks == 4) { LAUNCH_GEMM_INT4(4, MM); }                   \
        else if (ks == 8) { LAUNCH_GEMM_INT4(8, MM); }                   \
        else              { LAUNCH_GEMM_INT4(1, MM); }                   \
    } while (0)

    if      (M == 1) { DISPATCH_KS(1); }
    else if (M == 2) { DISPATCH_KS(2); }
    else if (M == 3) { DISPATCH_KS(3); }
    else if (M == 4) { DISPATCH_KS(4); }
    else {
        // M > 4 should not reach here; caller (esimd_gemm_int4 wrapper)
        // is expected to dispatch back to dequant+matmul or an MoE-style
        // tiled GEMM.
        // Fall through to M=1 loop as a graceful degradation.
        DISPATCH_KS(1);
    }

    #undef DISPATCH_KS
    #undef LAUNCH_GEMM_INT4
}
