/* q8_0_GEMV.h — GGUF q8_0 GEMV for Intel XPU (ESIMD), decode M=1.
 *
 * GGML block_q8_0 = { half d; int8 qs[32] }, group=32, SYMMETRIC (no min).
 * Real dequant (dequantize.cuh:71): w = d * qs, qs signed int8. We match this
 * exactly — NOT the skill's synthetic uint8+min q8_0 asset.
 *
 * Consumes the split-buffer repack (q8_0_repack_ref.py, bit-exact vs gguf-lib):
 *   input   [1, K]      fp16
 *   weight  [N, K]      int8   (signed quants, contiguous per row)
 *   scale   [N, K/32]   fp16   (per-block d)
 *   output  [1, N]      fp16
 *
 * dequant: w[n,k] = scale[n, k/32] * (float)qs[n,k]
 *
 * Structure mirrors q4_0_GEMV.h (K_SPLIT threads/WG + SLM reduce) rather than
 * the skill's ROWS=4 BMG layout — q8_0 in Qwen3.5 is the N=32 ssm_alpha/beta
 * tensors, so grid=N gives only 32 WGs; K_SPLIT keeps the 12 PTL cores busy.
 *
 * Included into esimd_kernel.sycl (utils.h provides fp16 + esimd namespace).
 */
#pragma once

static constexpr int Q8_0_GROUP = 32;  // elements per q8_0 block

// VL fixed at 32 (one q8_0 block per iter -> one scale load/iter).
// K_SPLIT distributes the K reduction across threads; kp = K/K_SPLIT must stay
// a multiple of 32 so no block is split across threads.
inline void select_ks_q8_0(uint32_t N, uint32_t K, int& ks) {
    ks = 1;
    if      (N <= 128 && K >= 2048) ks = 8;
    else if (N <= 512 && K >= 2048) ks = 4;
    int kp = K / ks;
    while ((kp % Q8_0_GROUP != 0) && ks > 1) {
        ks /= 2;
        kp = K / ks;
    }
}

template <int K_SPLIT>
struct Q8_0_gemv_kernel {
    const fp16*   input;   // [1, K]
    const int8_t* weight;  // [N, K]
    const fp16*   scale;   // [N, K/32]
    fp16*         output;  // [1, N]
    int N, K;
    int n_groups;          // K / 32

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }
        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;
        int kstart = lid * kp;

        simd<float, Q8_0_GROUP> acc = 0.0f;

        const int8_t* w_row = weight + (size_t)n * K;
        const fp16*   s_row = scale  + (size_t)n * n_groups;
        int group_idx = kstart / Q8_0_GROUP;

        for (int k = kstart; k < kstart + kp; k += Q8_0_GROUP) {
            // input: 32 fp16
            simd<fp16, Q8_0_GROUP> iv = block_load<fp16, Q8_0_GROUP>(input + k);
            // weight: 32 signed int8 -> float
            simd<int8_t, Q8_0_GROUP> raw = block_load<int8_t, Q8_0_GROUP>(w_row + k);
            simd<float, Q8_0_GROUP> wf = convert<float>(raw);
            // one fp16 scale per 32-block
            float s = static_cast<float>(s_row[group_idx]);
            group_idx += 1;
            acc += simd<float, Q8_0_GROUP>(iv) * (wf * s);
        }

        float my_sum = reduce<float>(acc, std::plus<>());

        if constexpr (K_SPLIT == 1) {
            output[n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                output[n] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};

inline void q8_0_gemv_host(
    const fp16* input, const int8_t* weight, const fp16* scale, fp16* output,
    uint32_t N, uint32_t K, sycl::queue& q) {
    int n_groups = K / Q8_0_GROUP;
    int ks;
    select_ks_q8_0(N, K, ks);
    int global = N * ks;
    int local = ks;

#define LAUNCH_Q8_0(S)                                                  \
    q.submit([&](sycl::handler& h) {                                    \
        h.parallel_for(sycl::nd_range<1>(global, local),                \
            Q8_0_gemv_kernel<S>{input, weight, scale, output,           \
                                (int)N, (int)K, n_groups});             \
    });

    if      (ks == 1) { LAUNCH_Q8_0(1) }
    else if (ks == 2) { LAUNCH_Q8_0(2) }
    else if (ks == 4) { LAUNCH_Q8_0(4) }
    else if (ks == 8) { LAUNCH_Q8_0(8) }
    else              { LAUNCH_Q8_0(1) }
#undef LAUNCH_Q8_0
}
