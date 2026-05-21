/* fused_add_rms_norm_batched.h — Batched Fused residual add + RMSNorm.
 *
 * Multi-row version. Semantics:
 *   residual[i] += hidden[i]   (in-place)
 *   hidden[i] = rmsnorm(residual[i]) * weight   (Gemma w+1.0 baked in caller)
 *
 * v2 (Karen Xiang's design, ported from
 *  frameworks.ai.client-ai.esimd-kernels af68ede:csrc/rmsNorm.h):
 *   Each row is computed by a workgroup of (K/128) threads, each
 *   processing 128 elements; SLM cross-thread reduction for variance.
 *   K must be a multiple of 128. Up to K=8192 supported (cap = 64 threads).
 *
 *   Why v2: the prior single-thread-per-row design serialized the
 *   K-axis reduction (1 thread × K elements). Cold-call to a fresh buffer
 *   at exactly rows=128 also exhibited a deterministic max_abs=14.51 bug
 *   that the parallel-reduction algorithm here doesn't reproduce.
 */

#pragma once
#include "utils.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

inline void fused_add_rms_norm_batched_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* weight_ptr,
    int rows, int K, float eps, sycl::queue& q)
{
    constexpr int VL = 128;          // elements per thread
    int threads = K / VL;            // workgroup size
    sycl::range<1> global((size_t)rows * (size_t)threads);
    sycl::range<1> local((size_t)threads);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            __ESIMD_NS::slm_init(64 * sizeof(float));

            int row = (int)item.get_group(0);
            int tid = (int)item.get_local_linear_id();
            uint32_t off = (uint32_t)(VL * tid * sizeof(fp16));
            int active = K / VL;
            if (row >= rows) return;

            simd<fp16, VL> input_in;
            simd<fp16, VL> residual_in;
            simd<fp16, VL> weight_h;
            simd<float, VL> input;
            simd<float, 16> variance = 0.0f;

            if (tid < active) {
                input_in = block_load<fp16, VL>(hidden_ptr + row * K + VL * tid);
                residual_in.template bit_cast_view<uint8_t>()
                        .template select<VL * sizeof(fp16), 1>(0) =
                    xesimd::lsc_block_load<
                        uint8_t, VL * sizeof(fp16),
                        xesimd::lsc_data_size::default_size,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        (uint8_t*)residual_ptr + row * K * sizeof(fp16) + off);
                weight_h.template bit_cast_view<uint8_t>()
                        .template select<VL * sizeof(fp16), 1>(0) =
                    xesimd::lsc_block_load<
                        uint8_t, VL * sizeof(fp16),
                        xesimd::lsc_data_size::default_size,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        (uint8_t*)weight_ptr + off);
            } else {
                input_in = fp16(0);
                residual_in = fp16(0);
                weight_h = fp16(0);
            }

            // residual <- residual + input ; promote to fp32
            input = input_in;
            input = input + residual_in;

            // write back updated residual
            if (tid < active) {
                simd<fp16, VL> r_store = input;
                block_store<fp16, VL>(residual_ptr + row * K + VL * tid, r_store);
            }

            // sum_sq via 8x16-element accumulators
            #pragma unroll
            for (int ll = 0; ll < 8; ll++) {
                simd<float, 16> sq = input.select<16, 1>(ll * 16);
                sq = sq * sq;
                variance += sq;
            }
            variance.select<8, 1>(0) += variance.select<8, 1>(8);
            variance.select<4, 1>(0) += variance.select<4, 1>(4);
            variance.select<2, 1>(0) += variance.select<2, 1>(2);
            variance[0] += variance[1];

            __ESIMD_NS::slm_block_store<float, 1>(tid * sizeof(float), variance[0]);
            sycl::group_barrier(item.get_group());

            // cross-thread reduction (always 64 slots; zero-fill inactive)
            simd<float, 64> sum = __ESIMD_NS::slm_block_load<float, 64>(0);
            for (int i = active; i < 64; i++) sum[i] = 0.0f;
            sum.select<32, 1>(0) += sum.select<32, 1>(32);
            sum.select<16, 1>(0) += sum.select<16, 1>(16);
            sum.select<8, 1>(0)  += sum.select<8, 1>(8);
            sum.select<4, 1>(0)  += sum.select<4, 1>(4);
            sum.select<2, 1>(0)  += sum.select<2, 1>(2);
            sum[0] += sum[1];

            float inv_rms = 1.0f / sycl::sqrt(sum[0] / (float)K + eps);

            // normalize × weight (fp32 multiply to match torch fallback precision)
            simd<float, VL> weight_f = weight_h;
            simd<float, VL> normed = input * inv_rms * weight_f;

            if (tid < active) {
                simd<fp16, VL> out_h = normed;
                block_store<fp16, VL>(hidden_ptr + row * K + VL * tid, out_h);
            }
        });
    });
}
