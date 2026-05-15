#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cmath>
#include <algorithm>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl;

// Scaled Dot-Product Attention (Decode) kernel
//
// q:   [seq_len, n_heads,    HD]
// k:   [kv_len,  n_kv_heads, HD]
// v:   [kv_len,  n_kv_heads, HD]
// out: [seq_len, n_heads,    HD]
//
// Dispatch: nd_range<3>(range<3>(seq_len, n_heads, GS), range<3>(1, 1, GS))
// GS threads cooperate along kv_len via online softmax + SLM tree reduction.
// Supports GQA/MQA: n_kv_heads <= n_heads, n_heads must be divisible by n_kv_heads.

template <typename IT, int HD, int GS>
ESIMD_INLINE void sdpa_decode_kernel(
    IT* q_ptr,
    IT* k_ptr,
    IT* v_ptr,
    IT* out_ptr,
    int seq_len,
    int kv_len,
    int n_heads,
    int n_kv_heads,
    float scale,
    bool is_causal,
    sycl::nd_item<3>& ndi)
{
    const int seq_idx  = (int)ndi.get_group(0);
    const int head_idx = (int)ndi.get_group(1);
    const int lid      = (int)ndi.get_local_id(2);

    // GQA head mapping
    const int kv_head = head_idx * n_kv_heads / n_heads;

    // Load q vector for this (seq_pos, head)
    const int q_offset = (seq_idx * n_heads + head_idx) * HD;
    simd<IT, HD> q_vec = block_load<IT, HD>(q_ptr + q_offset);

    // Per-thread kv range
    const int kv_per_thread = (kv_len + GS - 1) / GS;
    const int kv_start = lid * kv_per_thread;
    const int kv_end   = std::min(kv_start + kv_per_thread, kv_len);

    // Online softmax state
    float local_max = -65504.0f;
    float local_sum = 0.0f;
    simd<float, HD> acc(0.0f);

    // --- Main loop over assigned kv positions ---
    // Causal limit: query at seq_idx attends to kv positions <= causal_limit
    // When kv_len > seq_len, kv cache contains prior context tokens.
    const int causal_limit = kv_len - seq_len + seq_idx;
    for (int kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        if (is_causal && kv_pos > causal_limit) break;

        const int kv_off = (kv_pos * n_kv_heads + kv_head) * HD;

        // Load k vector
        simd<IT, HD> k_vec = block_load<IT, HD>(k_ptr + kv_off);
        // Load v vector
        simd<IT, HD> v_vec = block_load<IT, HD>(v_ptr + kv_off);

        float score = sycl::ext::intel::esimd::detail::sum<float, float, HD>(
            convert<float>(q_vec) * convert<float>(k_vec)) * scale;

        // Online softmax update with fused V accumulation
        float new_max = std::max(local_max, score);
        float exp_1 = std::exp(local_max - new_max);
        float exp_2 = std::exp(score - new_max);
        local_sum = local_sum * exp_1 + exp_2;
        acc = acc * exp_1 + convert<float>(v_vec) * exp_2;
        local_max = new_max;
    }

    // --- SLM reduction across GS threads ---
    constexpr int SLM_MAX_OFF = 0;
    constexpr int SLM_SUM_OFF = GS * (int)sizeof(float);
    constexpr int SLM_ACC_OFF = 2 * GS * (int)sizeof(float);
    constexpr int SLM_SIZE    = SLM_ACC_OFF + GS * HD * (int)sizeof(float);
    slm_init<SLM_SIZE>();

    // Phase 1: all threads store local_max, local_sum, and acc to SLM
    slm_block_store<float, 1>(SLM_MAX_OFF + lid * (int)sizeof(float),
                               simd<float, 1>(local_max));
    slm_block_store<float, 1>(SLM_SUM_OFF + lid * (int)sizeof(float),
                               simd<float, 1>(local_sum));
    slm_block_store<float, HD>(SLM_ACC_OFF + lid * HD * (int)sizeof(float), acc);
    barrier();

    // Phase 2: thread 0 corrects, reduces, normalizes, and writes output
    if (lid == 0) {
        simd<float, GS> all_max = slm_block_load<float, GS>(SLM_MAX_OFF);
        float global_max = hmax<float>(all_max);
        const simd<float, GS> scales = exp<float, GS>(all_max - global_max);

        simd<float, GS> all_sum = slm_block_load<float, GS>(SLM_SUM_OFF);
        float global_sum = sycl::ext::intel::esimd::detail::sum<float, float, GS>(
            all_sum * scales);
        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

        simd<float, HD> final_acc(0.0f);
        #pragma unroll
        for (int i = 0; i < GS; i++) {
            final_acc += slm_block_load<float, HD>(
                SLM_ACC_OFF + i * HD * (int)sizeof(float)) * scales[i];
        }
        final_acc *= inv_sum;

        const int out_offset = (seq_idx * n_heads + head_idx) * HD;
        block_store<IT, HD>(out_ptr + out_offset, convert<IT>(final_acc));
    }
}

// Host dispatcher: submits the 3D kernel to the SYCL queue
template <typename IT, int HD, int GS>
inline void sdpa_decode_host(
    IT* q, IT* k, IT* v, IT* out,
    int seq_len, int kv_len, int n_heads, int n_kv_heads,
    float scale, bool is_causal,
    sycl::queue& dpcpp_queue)
{
    sycl::range<3> global_range(seq_len, n_heads, GS);
    sycl::range<3> local_range(1, 1, GS);

    dpcpp_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
                sdpa_decode_kernel<IT, HD, GS>(
                    q, k, v, out,
                    seq_len, kv_len, n_heads, n_kv_heads,
                    scale, is_causal, ndi);
            });
    });
}

// =============================================================================
// Variable-length SDPA Decode with paged KV cache
// =============================================================================
//
// q:           [total_q_tokens, n_heads, HD]        — packed queries
// key_cache:   [num_blocks, block_size, n_kv_heads, HD] — paged K cache (may be non-contiguous)
// value_cache: [num_blocks, block_size, n_kv_heads, HD] — paged V cache (may be non-contiguous)
// out:         [total_q_tokens, n_heads, HD]
// cu_seqlens_q:  [batch_size + 1]   (int32)  — cumulative query lengths
// seqused_k:     [batch_size]       (int32)  — KV length per sequence
// block_table:   [batch_size, max_num_blocks_per_seq] (int32)
//
// cache_block_stride: stride (in elements) between consecutive blocks in the
//     cache tensor. For a contiguous [num_blocks, block_size, n_kv_heads, HD]
//     tensor this equals block_size * n_kv_heads * HD. For hybrid models
//     (attention + mamba) where K/V are interleaved per block, it equals
//     2 * block_size * n_kv_heads * HD.
//
// Dispatch: nd_range<3>(range<3>(total_q_tokens, n_heads, GS),
//                       range<3>(1, 1, GS))
// Each workgroup handles one (q_token, head).
// GS threads cooperate along the KV dimension with online softmax + SLM reduction.

template <typename IT, int HD, int GS>
ESIMD_INLINE void sdpa_decode_varlen_kernel(
    IT*  q_ptr,
    IT*  k_cache_ptr,
    IT*  v_cache_ptr,
    IT*  out_ptr,
    int* cu_seqlens_q_ptr,
    int* seqused_k_ptr,
    int* block_table_ptr,
    int  batch_size,
    int  max_seqlen_k,
    int  n_heads,
    int  n_kv_heads,
    int  block_size,
    int  max_num_blocks_per_seq,
    int  cache_block_stride,
    float scale,
    bool is_causal,
    sycl::nd_item<3>& ndi)
{
    const int q_idx    = (int)ndi.get_group(0);   // index in [0, total_q_tokens)
    const int head_idx = (int)ndi.get_group(1);
    const int lid      = (int)ndi.get_local_id(2);

    // GQA head mapping
    const int kv_head = head_idx * n_kv_heads / n_heads;

    // --- Find which batch sequence this q_token belongs to (linear scan) ---
    int batch_idx = 0;
    int q_start = cu_seqlens_q_ptr[0];
    for (int b = 1; b <= batch_size; b++) {
        int next = cu_seqlens_q_ptr[b];
        if (q_idx < next) break;
        q_start = next;
        batch_idx = b;
    }
    const int q_pos_in_seq = q_idx - q_start;                     // position within this sequence's queries
    const int seq_q_len    = cu_seqlens_q_ptr[batch_idx + 1] - q_start;

    // KV length for this sequence
    const int kv_len = (seqused_k_ptr != nullptr)
                     ? seqused_k_ptr[batch_idx]
                     : max_seqlen_k;

    // Pointer into this sequence's block table row
    const int* seq_block_table = block_table_ptr
                               + batch_idx * max_num_blocks_per_seq;

    // stride constants for cache layout [num_blocks, block_size, n_kv_heads, HD]
    // cache_block_stride is passed from the host (tensor.stride(0)) to handle
    // both contiguous and interleaved (hybrid attention+mamba) layouts.
    const int cache_head_stride  = HD;                            // stride over heads
    const int cache_pos_stride   = n_kv_heads * HD;               // stride over positions within a block

    // Load q vector
    const int q_offset = (q_idx * n_heads + head_idx) * HD;
    simd<IT, HD> q_vec = block_load<IT, HD>(q_ptr + q_offset);

    // Per-thread kv range
    const int kv_per_thread = (kv_len + GS - 1) / GS;
    const int kv_start = lid * kv_per_thread;
    const int kv_end   = std::min(kv_start + kv_per_thread, kv_len);

    // Online softmax state
    float local_max = -65504.0f;
    float local_sum = 0.0f;
    simd<float, HD> acc(0.0f);

    // Causal limit
    const int causal_limit = kv_len - seq_q_len + q_pos_in_seq;

    for (int kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        if (is_causal && kv_pos > causal_limit) break;

        // Translate logical kv_pos to physical address via block table
        const int blk_idx   = kv_pos / block_size;
        const int blk_off   = kv_pos % block_size;
        const int phys_blk  = seq_block_table[blk_idx];
        const int cache_off = phys_blk * cache_block_stride
                            + blk_off  * cache_pos_stride
                            + kv_head  * cache_head_stride;

        simd<IT, HD> k_vec = block_load<IT, HD>(k_cache_ptr + cache_off);
        simd<IT, HD> v_vec = block_load<IT, HD>(v_cache_ptr + cache_off);

        float score = sycl::ext::intel::esimd::detail::sum<float, float, HD>(
            convert<float>(q_vec) * convert<float>(k_vec)) * scale;

        float new_max = std::max(local_max, score);
        float exp_1 = std::exp(local_max - new_max);
        float exp_2 = std::exp(score - new_max);
        local_sum = local_sum * exp_1 + exp_2;
        acc = acc * exp_1 + convert<float>(v_vec) * exp_2;
        local_max = new_max;
    }

    // --- SLM reduction across GS threads ---
    constexpr int SLM_MAX_OFF = 0;
    constexpr int SLM_SUM_OFF = GS * (int)sizeof(float);
    constexpr int SLM_ACC_OFF = 2 * GS * (int)sizeof(float);
    constexpr int SLM_SIZE    = SLM_ACC_OFF + GS * HD * (int)sizeof(float);
    slm_init<SLM_SIZE>();

    slm_block_store<float, 1>(SLM_MAX_OFF + lid * (int)sizeof(float),
                               simd<float, 1>(local_max));
    slm_block_store<float, 1>(SLM_SUM_OFF + lid * (int)sizeof(float),
                               simd<float, 1>(local_sum));
    slm_block_store<float, HD>(SLM_ACC_OFF + lid * HD * (int)sizeof(float), acc);
    barrier();

    if (lid == 0) {
        simd<float, GS> all_max = slm_block_load<float, GS>(SLM_MAX_OFF);
        float global_max = hmax<float>(all_max);
        const simd<float, GS> scales = exp<float, GS>(all_max - global_max);

        simd<float, GS> all_sum = slm_block_load<float, GS>(SLM_SUM_OFF);
        float global_sum = sycl::ext::intel::esimd::detail::sum<float, float, GS>(
            all_sum * scales);
        float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

        simd<float, HD> final_acc(0.0f);
        #pragma unroll
        for (int i = 0; i < GS; i++) {
            final_acc += slm_block_load<float, HD>(
                SLM_ACC_OFF + i * HD * (int)sizeof(float)) * scales[i];
        }
        final_acc *= inv_sum;

        const int out_offset = (q_idx * n_heads + head_idx) * HD;
        block_store<IT, HD>(out_ptr + out_offset, convert<IT>(final_acc));
    }
}

// Host dispatcher for varlen decode
template <typename IT, int HD, int GS>
inline void sdpa_decode_varlen_host(
    IT* q, IT* k_cache, IT* v_cache, IT* out,
    int* cu_seqlens_q, int* seqused_k, int* block_table,
    int total_q_tokens, int batch_size, int max_seqlen_k,
    int n_heads, int n_kv_heads,
    int block_size, int max_num_blocks_per_seq,
    int cache_block_stride,
    float scale, bool is_causal,
    sycl::queue& dpcpp_queue)
{
    sycl::range<3> global_range(total_q_tokens, n_heads, GS);
    sycl::range<3> local_range(1, 1, GS);

    dpcpp_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
                sdpa_decode_varlen_kernel<IT, HD, GS>(
                    q, k_cache, v_cache, out,
                    cu_seqlens_q, seqused_k, block_table,
                    batch_size, max_seqlen_k,
                    n_heads, n_kv_heads,
                    block_size, max_num_blocks_per_seq,
                    cache_block_stride,
                    scale, is_causal, ndi);
            });
    });
}
