#pragma once
#include "utils.h"

// =============================================================================
// Dense INT4 GEMM for prefill (M >= 8): canonical [N, K/2] uint8 weight layout
// + [N, K/group] fp16 scale, same as esimd_gemv_int4 / esimd_gemm_int4_smallM.
//
// Why this kernel exists:
//   _esimd_int4_apply (vllm/.../sym_int4.py) falls through to
//   `_dequant_int4_to_fp16 + torch.matmul` for M > 4. On a 4k-prompt prefill
//   that path produced ~1100 ms of torch elementwise traffic per request
//   (25% of prefill GPU), dominated by the bitwise/shift/cast/scale-mul
//   chain that materializes the full fp16 weight before each matmul.
//
//   This kernel is the dense analogue of moe_prefill_up_routed_int4_kernel:
//   same DPAS<8,8,fp16> tiles + inline INT4 dequant inside the K-loop, but
//   stripped of expert routing (input/output are contiguous, not scattered).
//
// Layout:
//   input:  [M, K]              fp16 (contiguous, row-major)
//   weight: [N, K/2]            uint8 (canonical INT4 storage, row-major,
//                                     low nibble = even k, high nibble = odd)
//   scale:  [N, K/GROUP_SIZE]   fp16 (per-group, GROUP_SIZE = BS = 128)
//   output: [M, N]              fp16 (contiguous)
//
// WG layout:
//   2D grid (M / MAX_M, N / N_TILE). Each WG handles MAX_M rows × N_TILE
//   output columns, iterating over the full K dimension.
//
// Per-WG compute:
//   For each K-block of BS = 128:
//     load 1 group of scales for the N_TILE columns
//     for k_sub in [0..BS/2 step 8]:
//       block-load input tile: [MAX_M, 16] fp16 = MS × 256 fp16
//       for ns in [0..NS]:
//         load 8 N-rows of weight: [8, 8] uint8
//         unpack to fp16, scale
//         DPAS for each MS in [0..MS]: acc[ms, ns] += b_tile[ms] @ a_tile
//
// Shape constraints:
//   MAX_M % 16 == 0    (DPAS row tile)
//   N_TILE % 8 == 0    (DPAS col tile)
//   BS % 32 == 0       (K step = 1 group of scales)
//   M % MAX_M == 0     (no remainder handling at the M tail)
//   N % N_TILE == 0    (no remainder handling at the N tail)
//   K % BS == 0        (no remainder handling at the K tail)
// =============================================================================

template<int BS = 128, int MAX_M = 32, int N_TILE = 16>
sycl::event esimd_gemm_int4_prefill_kernel(
    sycl::queue& q,
    const fp16* x,            // [M, K]
    const uint8_t* weight,    // [N, K/2]
    const fp16* scale,        // [N, K/GROUP_SIZE]
    fp16* output,             // [M, N]
    int M, int N, int K) {
    static_assert(MAX_M % 16 == 0, "MAX_M must be multiple of 16");
    static_assert(N_TILE % 8 == 0, "N_TILE must be multiple of 8");
    static_assert(BS % 32 == 0,    "BS must be multiple of 32");
    constexpr int MS     = MAX_M / 16;
    constexpr int NS     = N_TILE / 8;
    constexpr int ACC_SZ = MS * NS * 128;

    // Number of WGs in each dim of the 2D grid.
    const int n_m_tiles = M / MAX_M;
    const int n_n_tiles = N / N_TILE;
    const int K_groups  = K / BS;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class EsimdGemmInt4Prefill>(
            sycl::range<2>(n_m_tiles, n_n_tiles),
            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
                const int m_tile_idx = (int)id[0];
                const int n_tile_idx = (int)id[1];
                const int m_base     = m_tile_idx * MAX_M;
                const int n_base     = n_tile_idx * N_TILE;

                // Weight rows we own are [n_base, n_base + N_TILE).
                const uint8_t* w_ptr = weight + (size_t)n_base * (K / 2);

                // Per-row scale base + per-N stride for gather.
                const fp16* s_base = scale + (size_t)n_base * K_groups;
                const simd<uint32_t, N_TILE> scl_byte_off =
                    simd<uint32_t, N_TILE>(0u, 1u) *
                    (uint32_t)(K_groups * sizeof(fp16));

                // Per-M-row byte offsets for the input gather. Lane i = M-row
                // (m_base + i)'s start in `x`. The lsc_gather pattern below
                // produces VNNI layout for B that DPAS expects — see
                // probe at csrc/xpu/esimd_kernels/_dpas_layout_probe.h
                // (lsc_gather with 16 offsets × 8 uint32 each → 16 M-rows ×
                // 16 fp16 K-cols, laid out as (K-pair, M-row, K-within-pair),
                // which is the DPAS-fp16 B operand VNNI format).
                simd<uint32_t, MAX_M> in_off =
                    (simd<uint32_t, MAX_M>(0u, 1u) + (uint32_t)m_base) *
                    (uint32_t)(K * sizeof(fp16));

                // Accumulators: MS DPAS row-tiles × NS DPAS col-tiles.
                // Use float to avoid fp16 overflow at large K/M combinations
                // (M=1024 K=4096 partial sums can exceed fp16 range).
                simd<float, ACC_SZ> acc(0.0f);

                for (int blk = 0; blk < K_groups; blk++) {
                    // Gather N_TILE per-group scales (one per N-row).
                    const simd<fp16, N_TILE> sc =
                        gather<fp16, N_TILE>(s_base + blk, scl_byte_off);

                    for (int k_base = 0; k_base < BS / 2; k_base += 8) {
                        const int x_byte = blk * (BS / 2) + k_base;
                        // k_off in fp16 elements = blk * BS + k_base * 2.
                        // For lsc_gather on uint32 view, the per-lane offset
                        // is in BYTES added to in_off. k_off in bytes:
                        const uint32_t k_off_bytes =
                            (uint32_t)(blk * BS + k_base * 2) *
                            (uint32_t)sizeof(fp16);

                        // ---- Gather input tile [MAX_M rows × 16 fp16 K-cols] ----
                        // VNNI-layout output: lane i's 8 uint32 = M-row i's
                        // 16 fp16 K-cols, packed as (K-pair, K-within-pair).
                        simd<fp16, MS * 256> b_tile;
                        #pragma unroll
                        for (int ms = 0; ms < MS; ms++) {
                            simd<fp16, 256> btmp;
                            btmp.template bit_cast_view<uint32_t>() =
                                xesimd::lsc_gather<uint32_t, 8,
                                    xesimd::lsc_data_size::u32,
                                    xesimd::cache_hint::cached,
                                    xesimd::cache_hint::cached,
                                    16, uint32_t>(
                                    reinterpret_cast<const uint32_t*>(x),
                                    in_off.template select<16, 1>(ms * 16).read()
                                        + k_off_bytes);
                            b_tile.template select<256, 1>(ms * 256) = btmp;
                        }

                        // ---- For each NS DPAS-col-tile ----
                        #pragma unroll
                        for (int ns = 0; ns < NS; ns++) {
                            // Load weight tile: 8 N-rows × 8 cols (uint8).
                            // Each N-row's K-stride = K/2 bytes.
                            auto w_raw = xesimd::lsc_load_2d<uint8_t, 8, 8, 1,
                                false, false,
                                xesimd::cache_hint::cached,
                                xesimd::cache_hint::cached>(
                                w_ptr,
                                (unsigned)(K / 2 - 1),       // surface width-1
                                (unsigned)(N_TILE - 1),       // surface height-1
                                (unsigned)(K / 2 - 1),       // pitch-1
                                x_byte,                       // x (byte offset in row)
                                ns * 8);                      // y (row)

                            // Unpack int4 → int8 → fp16 (×scale).
                            // Low nibble at even k-position, high at odd.
                            simd<uint8_t, 128> wu;
                            wu.template select<64, 2>(0) = w_raw & (uint8_t)0x0F;
                            wu.template select<64, 2>(1) = w_raw >> 4;
                            simd<fp16, 128> a_tile =
                                wu.template bit_cast_view<int8_t>() - (int8_t)8;
                            #pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                a_tile.template select<16, 1>(r * 16) *=
                                    sc[ns * 8 + r];
                            }

                            // DPAS<8,8,float-acc>: acc[ms, ns] += b_tile[ms] @ a_tile
                            #pragma unroll
                            for (int ms = 0; ms < MS; ms++) {
                                const int idx = ms * NS * 128 + ns * 128;
                                simd<float, 128> a = acc.template select<128, 1>(idx);
                                simd<fp16, 256> bv = b_tile.template select<256, 1>(ms * 256);
                                a = sycl::ext::intel::esimd::xmx::dpas<8, 8, float, float, fp16, fp16>(
                                        a, bv, a_tile);
                                acc.template select<128, 1>(idx) = a;
                            }
                        }
                    }
                }

                // ---- Output scatter ----
                // acc layout (mirroring moe_prefill_up_routed_int4_kernel):
                //   acc[ms * NS * 128 + ns * 128 + r * 16 + lane]
                // is the value at (M-row = m_base + ms*16 + lane,
                //                  N-col = n_base + ns*8 + r/2 ?) ...
                // moe_prefill_up's scatter iterates `r` from 0..N_TILE and
                // emits a simd<fp16,16> covering 16 M-rows of one N-channel
                // (channel = n_base + r). We do the same, with contiguous
                // M-row offsets.
                #pragma unroll
                for (int ms = 0; ms < MS; ms++) {
                    simd<uint32_t, 16> lane_off =
                        (simd<uint32_t, 16>(0u, 1u) +
                         (uint32_t)(m_base + ms * 16)) *
                        (uint32_t)(N * sizeof(fp16));
                    #pragma unroll
                    for (int r = 0; r < N_TILE; r++) {
                        // acc is float; convert to fp16 for the output write.
                        simd<float, 16> v_f = acc.template select<16, 1>(
                            ms * N_TILE * 16 + r * 16);
                        simd<fp16, 16> v = convert<fp16>(v_f);
                        uint32_t ch = (uint32_t)(n_base + r) * (uint32_t)sizeof(fp16);
                        scatter<fp16, 16>(output, lane_off + ch, v);
                    }
                }
            });
    });
}
