/* q4_0_GEMM.h — GGUF q4_0 INT4 GEMM (prefill / batched M>=2), NON-DPAS.
 *
 * PTL Xe3 LPG (XeLPG) has NO XMX/DPAS matrix engine (confirmed: commit
 * cca9d9b + ocloc "Intrinsic is not supported by the <XeLPG> platform"). So
 * this is a plain SIMD ESIMD GEMM, NOT a port of the BMG DPAS int4_GEMM.h.
 *
 * Structure = q4_0_GEMV extended over M rows: each work-group owns one output
 * column n, dequantizes weight row n once per 32-block, and reuses those
 * dequantized weights across all M input rows. This amortizes the LPDDR
 * weight read across M (the reason a batched path beats M separate GEMV
 * calls), without needing a matrix engine.
 *
 *   * group_size = 32, INTERLEAVED nibble (same weight as q4_0_GEMV.h)
 *   * dequant: w = (nibble - 8) * scale  (symmetric, scale may be negative)
 *
 * input  [M, K]    fp16
 * weight [N, K/2]  uint8  (interleaved: byte j low=K_even, high=K_odd)
 * scale  [N, K/32] fp16
 * output [M, N]    fp16
 *
 * Grid: N work-groups, 1 thread each (M is looped inside; M is small at decode
 * batch / chunked-prefill tile). K reduction is sequential per WG. Correctness-
 * first; the decode GEMV (M=1) remains the optimized bandwidth path.
 *
 * Included into esimd_kernel.sycl (sycl/esimd headers + fp16 already in scope).
 */
#pragma once

static constexpr int Q4_0_GEMM_GROUP = 32;
static constexpr int Q4_0_GEMM_HALF = 16;

// Grid: N work-groups (one output column each), 1 thread per WG. Loops over M.
struct Q4_0_gemm_kernel {
    const fp16*    input;   // [M, K]
    const uint8_t* weight;  // [N, K/2]
    const fp16*    scale;   // [N, K/32]
    fp16*          output;  // [M, N]
    int M, N, K;
    int n_groups;           // K / 32

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;

        const uint8_t* w_row = weight + (size_t)n * (K / 2);
        const fp16*    s_row = scale  + (size_t)n * n_groups;

        int blocks = K / Q4_0_GEMM_GROUP;

        // Per-M running dot accumulators (M small; cap unroll at compile via loop).
        // We iterate blocks in the outer loop so each weight block is dequantized
        // once and reused across all M rows.
        for (int m0 = 0; m0 < M; m0++) {
            simd<float, Q4_0_GEMM_HALF> acc_even = 0.0f;
            simd<float, Q4_0_GEMM_HALF> acc_odd  = 0.0f;
            const fp16* in_row = input + (size_t)m0 * K;

            for (int b = 0; b < blocks; b++) {
                int k = b * Q4_0_GEMM_GROUP;
                simd<fp16, Q4_0_GEMM_GROUP> iv =
                    block_load<fp16, Q4_0_GEMM_GROUP>(in_row + k);
                simd<float, Q4_0_GEMM_HALF> in_even = iv.template select<Q4_0_GEMM_HALF, 2>(0);
                simd<float, Q4_0_GEMM_HALF> in_odd  = iv.template select<Q4_0_GEMM_HALF, 2>(1);

                simd<uint8_t, Q4_0_GEMM_HALF> raw =
                    block_load<uint8_t, Q4_0_GEMM_HALF>(w_row + k / 2);
                simd<uint16_t, Q4_0_GEMM_HALF> u16 = convert<uint16_t>(raw);
                simd<float, Q4_0_GEMM_HALF> w_even = convert<float>(u16 & 0x000F) - 8.0f;
                simd<float, Q4_0_GEMM_HALF> w_odd  = convert<float>((u16 >> 4) & 0x000F) - 8.0f;

                float s = static_cast<float>(s_row[b]);
                acc_even += in_even * (w_even * s);
                acc_odd  += in_odd  * (w_odd  * s);
            }

            float dot = reduce<float>(acc_even, std::plus<>())
                      + reduce<float>(acc_odd,  std::plus<>());
            output[(size_t)m0 * N + n] = fp16(dot);
        }
    }
};

inline void q4_0_gemm_host(
    const fp16* input, const uint8_t* weight, const fp16* scale, fp16* output,
    uint32_t M, uint32_t N, uint32_t K, sycl::queue& q) {
    int n_groups = (int)K / Q4_0_GEMM_GROUP;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)N, (size_t)1),
            Q4_0_gemm_kernel{input, weight, scale, output,
                             (int)M, (int)N, (int)K, n_groups});
    });
}
