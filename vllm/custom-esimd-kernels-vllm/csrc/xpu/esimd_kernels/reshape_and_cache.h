/* reshape_and_cache.h — Paged KV cache scatter for flash-style cache layout.
 *
 * Replacement for the PyTorch fallback in _ipex_ops.reshape_and_cache_flash
 * which uses .nonzero() and is therefore incompatible with SYCL command-graph
 * capture. This kernel is a pure dtype-agnostic 2-byte memory copy.
 *
 * Ported from referance/.../vllm-kernel-custom-windows/csrc/xpu/
 *                  reshape_and_cache.sycl
 *
 * key / value           : [num_tokens, num_kv_heads, head_size]  fp16/bf16
 * key_cache/value_cache : [num_blocks, block_size, num_kv_heads, head_size]
 * slot_mapping          : [num_tokens]  int64 (row-major linear index into
 *                                              the [num_blocks, block_size]
 *                                              flat KV cache).
 *
 * One workgroup per token. Each thread copies 128 elements (256 bytes).
 * Negative slot values (PAD_SLOT_ID = -1) are silently skipped.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace {

inline void reshape_and_cache_flash_host(
    uint8_t* key_ptr,
    uint8_t* value_ptr,
    uint8_t* key_cache_ptr,
    uint8_t* value_cache_ptr,
    const int64_t* slot_mapping_ptr,
    int64_t num_tokens,
    int64_t row_stride,        // num_kv_heads * head_size (in elements)
    int64_t cache_block_stride, // key_cache.stride(0) — elements between blocks
    int64_t cache_pos_stride,   // key_cache.stride(1) — elements between rows in a block
    int64_t block_size,         // tokens per block
    int64_t max_slot,           // num_blocks * block_size — out-of-range guard
    sycl::queue& dpcpp_queue)
{
    if (num_tokens == 0) return;

    int n_threads = (int)(row_stride / 128);
    if (n_threads <= 0) n_threads = 1;
    int64_t remainder = row_stride % 128;

    sycl::range<1> global((size_t)num_tokens * (size_t)n_threads);
    sycl::range<1> local((size_t)n_threads);
    sycl::nd_range<1> range(global, local);

    dpcpp_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(range,
            [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                using namespace sycl::ext::intel::esimd;
                int tok = (int)ndi.get_group(0);
                int tid = (int)ndi.get_local_id(0);
                if (tok >= num_tokens) return;

                int64_t slot = slot_mapping_ptr[tok];
                // Skip negative (PAD_SLOT_ID) and out-of-range slots; the
                // upstream torch fallback used `.nonzero()` to filter both.
                // Without this guard, profile_run / partial-batch input
                // can inject garbage indices that scribble over adjacent
                // model weights and corrupt subsequent decode steps.
                if (slot < 0 || slot >= max_slot) return;

                // Hybrid attn+mamba models use as_strided to interleave
                // K/V per block; cannot assume `cache.stride(0) ==
                // block_size * row_stride`. Compute via the supplied
                // strides instead.
                int64_t block_id      = slot / block_size;
                int64_t slot_in_block = slot - block_id * block_size;
                int64_t src_off =
                    (int64_t)tok * row_stride + (int64_t)tid * 128;
                int64_t dst_off =
                    block_id * cache_block_stride
                    + slot_in_block * cache_pos_stride
                    + (int64_t)tid * 128;

                int64_t count = 128;
                if (tid == (int)ndi.get_local_range(0) - 1 &&
                    remainder != 0) {
                    count = remainder;
                }

                // Use uint16_t for dtype-agnostic 2-byte copy
                // (works for both fp16 and bf16).
                if (count == 128) {
                    simd<uint16_t, 128> k_data =
                        block_load<uint16_t, 128>(
                            (const uint16_t*)key_ptr + src_off);
                    block_store<uint16_t, 128>(
                        (uint16_t*)key_cache_ptr + dst_off, k_data);

                    simd<uint16_t, 128> v_data =
                        block_load<uint16_t, 128>(
                            (const uint16_t*)value_ptr + src_off);
                    block_store<uint16_t, 128>(
                        (uint16_t*)value_cache_ptr + dst_off, v_data);
                } else {
                    for (int64_t i = 0; i < count; i++) {
                        ((uint16_t*)key_cache_ptr)[dst_off + i] =
                            ((const uint16_t*)key_ptr)[src_off + i];
                        ((uint16_t*)value_cache_ptr)[dst_off + i] =
                            ((const uint16_t*)value_ptr)[src_off + i];
                    }
                }
            });
    });
}

}  // anonymous namespace
