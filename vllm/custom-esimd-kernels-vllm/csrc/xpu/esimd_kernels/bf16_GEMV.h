#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;
using bf16  = sycl::ext::oneapi::bfloat16;
using namespace sycl;

// BF16 GEMV (W16A16, no quantisation).
// Input:  [1, K] bf16
// Weight: [N, K] bf16
// Output: [1, N] bf16
//
// No scale/zero-point — weights are exact bf16.
// FP32 accumulation.  Fused variant: one kernel dispatch for GEMV_COUNT
// matrices sharing the same input vector.
//
// ** PTL/XE3 constraints **
// - lsc_load_2d<bf16> produces WRONG results on PTL — use block_load only
// - Strided select across doubleGRF (stride-4, 256-elem) broken on PTL
// - SLM fence: use fence<local_barrier>() before barrier() on PTL
//
// Design: ROWS tiling with K_SPLIT, following xe2-esimd-gemv proven patterns.
// Work-group = ROWS × K_SPLIT threads.  Each thread handles one output row,
// computes K/K_SPLIT partial dot product.  K_SPLIT workers reduce via SLM.
// Multiple accumulators (8 rotating) to hide FMA latency.

// ── VL/ROWS/KS auto-selector for PTL ───────────────────────────────────
struct bf16_gemv_config {
    int vl;
    int rows;
    int ks;
};

inline bf16_gemv_config select_config_bf16(uint32_t N, uint32_t K) {
    // PTL: VL=128 proven safe, working with full model (48/48).
    int vl = 128;
    int rows = 4;
    int ks = 1;

    // For small N, reduce ROWS so we still have enough work-groups
    if (N < 64)  { rows = 1; }
    else if (N < 128) { rows = 2; }

    // For large K with small N, K_SPLIT helps
    if (N <= 128 && K >= 2048) { ks = 4; }
    else if (N <= 512 && K >= 2048) { ks = 2; }

    // Ensure K per worker is divisible by VL
    int kpt = K / ks;
    while (kpt % vl != 0 || vl > kpt) {
        if (vl > 32) { vl /= 2; }
        else if (ks > 1) { ks /= 2; kpt = K / ks; }
        else break;
    }
    return {vl, rows, ks};
}

// ── Fused BF16 GEMV kernel with ROWS tiling ────────────────────────────
template<int VL, int ROWS, int K_SPLIT, int GEMV_COUNT>
struct GEMV_bf16_fused_kernel {
    const bf16* input;               // [1, K]
    const bf16* weights[GEMV_COUNT]; // [N_i, K] each
    bf16*       outputs[GEMV_COUNT];
    int         N[GEMV_COUNT];
    int         N_cumsum[GEMV_COUNT];
    int         total_N;
    int         K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int LOCAL_SIZE = ROWS * K_SPLIT;
        constexpr int SLM_SIZE  = ROWS * K_SPLIT * sizeof(float);

        if constexpr (K_SPLIT > 1) {
            slm_init<SLM_SIZE>();
        }

        int local_id  = item.get_local_id(0);
        int group_id  = item.get_group(0);

        // Thread decomposition within work-group
        int k_tid   = local_id % K_SPLIT;    // which K-slice
        int row_tid = local_id / K_SPLIT;    // which row within tile

        // Global row index
        int n_global = group_id * ROWS + row_tid;

        // Route to correct matrix via cumulative-N
        int mat_idx = 0;
        int n_local = n_global;
        #pragma unroll
        for (int i = 0; i < GEMV_COUNT; i++) {
            if (n_global < N_cumsum[i]) {
                mat_idx = i;
                n_local = (i == 0) ? n_global : n_global - N_cumsum[i - 1];
                break;
            }
        }

        const bf16* w_ptr = weights[mat_idx];
        bf16*       o_ptr = outputs[mat_idx];
        int         n_max = N[mat_idx];

        // K-split range for this thread
        int kp     = K / K_SPLIT;
        int kstart = k_tid * kp;

        // 8 rotating accumulators to hide FMA latency
        simd<float, 8> partial_sums = 0.0f;
        int acc_idx = 0;

        if (row_tid < ROWS && n_local < n_max) {
            const bf16* w_row = w_ptr + (size_t)n_local * K;

            for (int k = kstart; k < kstart + kp; k += VL) {
                simd<float, VL> iv = convert<float>(
                    block_load<bf16, VL>(input + k));
                simd<float, VL> wv = convert<float>(
                    block_load<bf16, VL>(w_row + k));

                partial_sums[acc_idx] += reduce<float>(iv * wv, std::plus<>());
                acc_idx = (acc_idx + 1) & 0x7;
            }
        }

        float my_sum = reduce<float>(partial_sums, std::plus<>());

        if constexpr (K_SPLIT == 1) {
            // No SLM reduction needed — write directly
            if (row_tid < ROWS && n_local < n_max) {
                o_ptr[n_local] = bf16(my_sum);
            }
        } else {
            // SLM reduction across K_SPLIT workers
            uint32_t slm_off = (row_tid * K_SPLIT + k_tid) * sizeof(float);
            if (row_tid < ROWS) {
                slm_block_store<float, 1>(slm_off, simd<float, 1>(my_sum));
            }

            // PTL/XE3: fence before barrier for correct SLM visibility
            fence<fence_mask::local_barrier>();
            barrier();

            if (k_tid == 0 && row_tid < ROWS && n_local < n_max) {
                uint32_t slm_base = row_tid * K_SPLIT * sizeof(float);
                float final_sum;
                if constexpr (K_SPLIT == 2) {
                    simd<float, 2> r = slm_block_load<float, 2>(slm_base);
                    final_sum = r[0] + r[1];
                } else if constexpr (K_SPLIT == 4) {
                    simd<float, 4> r = slm_block_load<float, 4>(slm_base);
                    final_sum = reduce<float>(r, std::plus<>());
                } else if constexpr (K_SPLIT == 8) {
                    simd<float, 8> r = slm_block_load<float, 8>(slm_base);
                    final_sum = reduce<float>(r, std::plus<>());
                } else {
                    simd<float, K_SPLIT> r = slm_block_load<float, K_SPLIT>(slm_base);
                    final_sum = reduce<float>(r, std::plus<>());
                }
                o_ptr[n_local] = bf16(final_sum);
            }
        }
    }
};

// ── Host dispatch (auto-select) ────────────────────────────────────────
template<int GEMV_COUNT>
inline void GEMV_bf16_fused_host(
    uint8_t* input_data,
    uint8_t* weight_ptrs[GEMV_COUNT],
    uint8_t* output_ptrs[GEMV_COUNT],
    uint32_t Ns[GEMV_COUNT],
    uint32_t K,
    sycl::queue& q) {

    auto* p_in = reinterpret_cast<const bf16*>(input_data);

    uint32_t total_N = 0;
    for (int i = 0; i < GEMV_COUNT; i++) total_N += Ns[i];

    auto cfg = select_config_bf16(total_N, K);
    int vl   = cfg.vl;
    int rows = cfg.rows;
    int ks   = cfg.ks;

    int num_groups  = (total_N + rows - 1) / rows;
    int local_size  = rows * ks;
    int global_size = num_groups * local_size;

    // Macro: instantiate kernel template for given VL, ROWS, K_SPLIT
    #define LAUNCH_BF16(V, R, S) \
        q.submit([&](sycl::handler& h) { \
            GEMV_bf16_fused_kernel<V, R, S, GEMV_COUNT> kern; \
            kern.input = p_in; \
            kern.K = (int)K; \
            kern.total_N = (int)total_N; \
            uint32_t cum = 0; \
            for (int i = 0; i < GEMV_COUNT; i++) { \
                kern.weights[i] = reinterpret_cast<const bf16*>(weight_ptrs[i]); \
                kern.outputs[i] = reinterpret_cast<bf16*>(output_ptrs[i]); \
                kern.N[i] = (int)Ns[i]; \
                cum += Ns[i]; \
                kern.N_cumsum[i] = (int)cum; \
            } \
            h.parallel_for(sycl::nd_range<1>(global_size, local_size), kern); \
        });

    // Enumerate supported (VL, ROWS, K_SPLIT) combinations
    if      (vl == 128 && rows == 4 && ks == 1) { LAUNCH_BF16(128, 4, 1) }
    else if (vl == 128 && rows == 4 && ks == 2) { LAUNCH_BF16(128, 4, 2) }
    else if (vl == 128 && rows == 4 && ks == 4) { LAUNCH_BF16(128, 4, 4) }
    else if (vl == 128 && rows == 2 && ks == 1) { LAUNCH_BF16(128, 2, 1) }
    else if (vl == 128 && rows == 2 && ks == 2) { LAUNCH_BF16(128, 2, 2) }
    else if (vl == 128 && rows == 2 && ks == 4) { LAUNCH_BF16(128, 2, 4) }
    else if (vl == 128 && rows == 1 && ks == 1) { LAUNCH_BF16(128, 1, 1) }
    else if (vl == 128 && rows == 1 && ks == 4) { LAUNCH_BF16(128, 1, 4) }
    // Fallback
    else                                         { LAUNCH_BF16(128, 4, 1) }

    #undef LAUNCH_BF16
}
