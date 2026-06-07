"""UT for esimd_sdpa_prefill_dpas on PTL Xe3.

Tests the FA-2 style block-attention prefill kernel ported from the
windows tree (task #135). Reference computed on CPU in fp32.
"""
import math
import torch
import pytest

import custom_esimd_kernels_vllm.custom_esimd_kernels_prefill_dpas  # registers op
esimd_sdpa_prefill_dpas = __import__('torch').ops.custom_esimd_kernels_vllm.esimd_sdpa_prefill_dpas


def _ref_paged_sdpa_prefill(q, key_cache, value_cache, cu_seqlens_q,
                            seq_lens, block_table, scale, is_causal):
    """Same reference as test_esimd_sdpa_decode_varlen.py — CPU fp32.

    Causal limit semantics: per-query position qi (0-indexed within its
    sequence's q-batch), causal limit = kv_len - seq_q_len + qi
    (matches the existing decode-varlen kernel's behaviour).
    """
    q = q.to("cpu", torch.float32)
    key_cache = key_cache.to("cpu", torch.float32)
    value_cache = value_cache.to("cpu", torch.float32)
    cu_seqlens_q = cu_seqlens_q.to("cpu")
    block_table = block_table.to("cpu")
    seq_lens = seq_lens.to("cpu")

    total_q, n_heads, hd = q.shape
    block_size = key_cache.shape[1]
    n_kv_heads = key_cache.shape[2]
    out = torch.zeros_like(q)
    B = cu_seqlens_q.shape[0] - 1

    for b in range(B):
        q_start = int(cu_seqlens_q[b])
        q_end = int(cu_seqlens_q[b + 1])
        seq_q_len = q_end - q_start
        if seq_q_len == 0:
            continue
        kv_len = int(seq_lens[b])

        # Gather K, V from paged blocks.
        k_seq = torch.empty(kv_len, n_kv_heads, hd, dtype=torch.float32)
        v_seq = torch.empty(kv_len, n_kv_heads, hd, dtype=torch.float32)
        for kv_pos in range(kv_len):
            blk_idx = kv_pos // block_size
            blk_off = kv_pos % block_size
            phys = int(block_table[b, blk_idx])
            k_seq[kv_pos] = key_cache[phys, blk_off]
            v_seq[kv_pos] = value_cache[phys, blk_off]

        # GQA expand
        ratio = n_heads // n_kv_heads
        k_seq = k_seq.repeat_interleave(ratio, dim=1)
        v_seq = v_seq.repeat_interleave(ratio, dim=1)

        for qi in range(seq_q_len):
            q_global = q_start + qi
            qv = q[q_global]
            scores = torch.einsum("hd,khd->hk", qv, k_seq) * scale
            if is_causal:
                causal_limit = kv_len - seq_q_len + qi
                idx = torch.arange(kv_len)
                scores = scores.masked_fill(
                    idx.unsqueeze(0) > causal_limit, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[q_global] = torch.einsum("hk,khd->hd", probs, v_seq)

    return out.to(q.dtype)


def _run_case(batch_q_lens, kv_lens, *, n_heads=16, n_kv_heads=2, hd=256,
              block_size=64, dtype=torch.float16, is_causal=True, seed=0,
              tol=None, shuffle_blocks=False):
    """Build inputs, run prefill kernel + CPU ref, compare."""
    torch.manual_seed(seed)
    device = "xpu"
    B = len(batch_q_lens)
    assert len(kv_lens) == B
    for ql, kl in zip(batch_q_lens, kv_lens):
        assert ql <= kl, f"q_len {ql} > kv_len {kl} (kv must include q prefix)"

    total_q = sum(batch_q_lens)
    cu_seqlens_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(batch_q_lens), dim=0).tolist()),
        dtype=torch.int32, device=device,
    )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    max_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
    total_blocks = sum((k + block_size - 1) // block_size for k in kv_lens)
    num_blocks = max(total_blocks + 4, 8)

    block_table = torch.full(
        (B, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
    if shuffle_blocks:
        # Realistic vllm paged-attention case: phys blocks are NOT
        # contiguous — they come from a free-list and get scattered.
        # Prior contiguous allocation hid the bug where the kernel
        # assumes one phys block holds PF_KV_CHUNK=128 KV tokens, but
        # block_size=64 splits them across 2 phys blocks.
        torch.manual_seed(seed + 1000)
        nb_total = sum((k + block_size - 1) // block_size for k in kv_lens)
        # Pick scrambled phys ids out of [1, num_blocks)
        perm = torch.randperm(num_blocks - 1)[:nb_total] + 1
        cursor_idx = 0
        for b in range(B):
            nb = (kv_lens[b] + block_size - 1) // block_size
            for j in range(nb):
                block_table[b, j] = perm[cursor_idx].item()
                cursor_idx += 1
    else:
        cursor = 1
        for b in range(B):
            nb = (kv_lens[b] + block_size - 1) // block_size
            for j in range(nb):
                block_table[b, j] = cursor
                cursor += 1

    q = (torch.randn(total_q, n_heads, hd, device=device, dtype=torch.float32)
         * 0.05).to(dtype)
    key_cache = (torch.randn(num_blocks, block_size, n_kv_heads, hd,
                             device=device, dtype=torch.float32) * 0.05
                 ).to(dtype)
    value_cache = (torch.randn(num_blocks, block_size, n_kv_heads, hd,
                               device=device, dtype=torch.float32) * 0.05
                   ).to(dtype)
    scale = 1.0 / math.sqrt(hd)

    out = esimd_sdpa_prefill_dpas(
        q.contiguous(), key_cache, value_cache,
        cu_seqlens_q, seq_lens, is_causal, scale,
        block_table.contiguous(),
    )
    torch.xpu.synchronize()

    ref = _ref_paged_sdpa_prefill(
        q, key_cache, value_cache, cu_seqlens_q, seq_lens,
        block_table, scale, is_causal,
    )

    out_cpu = out.to("cpu", torch.float32)
    ref_cpu = ref.to(torch.float32)
    diff = (out_cpu - ref_cpu).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_abs = ref_cpu.abs()
    ref_max = ref_abs.max().item()
    ref_mean = ref_abs.mean().item()
    # Relative error vs ref magnitude (avoid /0 with eps).
    eps = 1e-6
    rel = diff / (ref_abs + eps)
    rel_max = rel.max().item()
    rel_mean = rel.mean().item()
    # cosine similarity (per output token row, then mean) — proxy for direction match.
    out_flat = out_cpu.reshape(-1, ref_cpu.shape[-1])
    ref_flat = ref_cpu.reshape(-1, ref_cpu.shape[-1])
    cos = torch.nn.functional.cosine_similarity(out_flat, ref_flat, dim=-1)
    cos_min = cos.min().item()
    cos_mean = cos.mean().item()
    if tol is None:
        tol = 5e-3 if dtype == torch.float16 else 1e-2
    print(f"case batch_q={batch_q_lens} kv={kv_lens} dtype={dtype}:")
    print(f"  abs:   max={max_abs:.4g}  mean={mean_abs:.4g}  tol={tol}")
    print(f"  ref:   max={ref_max:.4g}  mean={ref_mean:.4g}")
    print(f"  rel:   max={rel_max:.4g}  mean={rel_mean:.4g}  "
          f"(rel_to_max={max_abs / max(ref_max, eps):.3%})")
    print(f"  cos:   min={cos_min:.6f}  mean={cos_mean:.6f}")
    assert max_abs < tol, f"abs diff {max_abs} > tol {tol}"


def test_single_short():
    """B=1, q=128, kv=128 (no prefix)."""
    _run_case([128], [128])


def test_single_with_prefix():
    """B=1, q=128, kv=256 (128-token prefix)."""
    _run_case([128], [256])


def test_single_q256_kv1024():
    _run_case([256], [1024])


def test_single_q924_kv924():
    """The 35B/4B production prefill shape from the unitrace."""
    _run_case([924], [924])


def test_b2_varlen():
    """B=2, mixed q lens with both having prefixes."""
    _run_case([128, 256], [256, 512])


def test_bf16_short():
    _run_case([128], [128], dtype=torch.bfloat16, tol=1e-2)


def test_bf16_q924():
    _run_case([924], [924], dtype=torch.bfloat16, tol=1e-2)


# === chunked-prefill mixed-batch coverage (production path #4) ===

def test_mixed_prefill_decode():
    """Chunked prefill: B=2, batch_q=[924, 1] kv=[924, 100].

    vLLM v1 chunked prefill bundles a prefill chunk with concurrent
    decode steps (q_len=1) into one varlen batch. The prefill_dpas
    kernel must handle q_len=1 rows correctly (FA-2 outer-tile padding).
    This is the actual 35B production batch shape.
    """
    _run_case([924, 1], [924, 100])


def test_mixed_decode_prefill():
    """Reverse order: decode first, then prefill. cu_seqlens_q layout matters."""
    _run_case([1, 924], [100, 924])


def test_mixed_two_prefill_one_decode():
    """B=3: q=[256, 1, 256] kv=[256, 50, 512] — prefill+decode+prefill+prefix."""
    _run_case([256, 1, 256], [256, 50, 512])


def test_short_prefill_chunks():
    """q_len smaller than PF_WG_Q_ROWS=128: q=64,kv=200."""
    _run_case([64], [200])


def test_q_at_block_boundary():
    """q at exact block boundaries (block_size=64)."""
    _run_case([64], [64])
    _run_case([192], [192])  # 3*block_size
    _run_case([320], [320])  # 5*block_size


def test_partial_q_tile():
    """q_len not multiple of PF_WG_Q_ROWS=128: q=130 (=128+2 partial tile)."""
    _run_case([130], [256])
    _run_case([200], [400])


def test_chunked_prefill_with_long_prefix():
    """Chunked prefill at high prefix ratio: q_len << kv_len.

    Mimics 35B prod: a 1024-token chunk landing into 4096-token context,
    then 256-token chunk into 8000-token context.
    """
    _run_case([1024], [2048])
    _run_case([512], [4096])
    _run_case([256], [8000])
    # Last chunk of a long prefill — chunk size 256 with 16k of prefix.
    _run_case([256], [16384])


def test_chunk_smaller_than_block():
    """q_len < block_size = 64."""
    _run_case([16], [128])
    _run_case([32], [256])
    _run_case([60], [256])  # near block boundary


# === non-contiguous phys block allocation (vllm paged-attention reality) ===

def test_shuffled_blocks_short():
    """B=1 q=128 kv=128 with scrambled phys block assignment."""
    _run_case([128], [128], shuffle_blocks=True)


def test_shuffled_blocks_q924():
    """Production prefill shape with phys blocks scrambled.

    35B prefill at mnbt=1024 with block_size=64 → 16 phys blocks per
    sequence. With paged-attention's free-list, these blocks are
    scattered in physical memory; the kernel must look up the block
    table for every PF_KV_CHUNK (or finer) tile, not assume contiguity.
    """
    _run_case([924], [924], shuffle_blocks=True)


def test_shuffled_blocks_long_prefix():
    """Chunked prefill: 256-token chunk into 8000-token prefix, scrambled."""
    _run_case([256], [8000], shuffle_blocks=True)


def test_shuffled_blocks_blocksize128():
    """Same as test_shuffled_blocks_q924 but block_size=128 (== PF_KV_CHUNK).

    Confirms that the kernel's contiguous-block assumption holds when
    block_size matches PF_KV_CHUNK. If this PASSes while
    test_shuffled_blocks_q924 FAILs, the root cause is exactly
    block_size < PF_KV_CHUNK splitting the chunk across phys blocks.
    """
    _run_case([924], [924], shuffle_blocks=True, block_size=128)


def test_shuffled_blocks_blocksize256():
    """Same as above but block_size=256."""
    _run_case([924], [924], shuffle_blocks=True, block_size=256)





if __name__ == "__main__":
    for fn in [
        test_single_short,
        test_single_with_prefix,
        test_single_q256_kv1024,
        test_single_q924_kv924,
        test_b2_varlen,
        test_bf16_short,
        test_bf16_q924,
        test_mixed_prefill_decode,
        test_mixed_decode_prefill,
        test_mixed_two_prefill_one_decode,
        test_short_prefill_chunks,
        test_q_at_block_boundary,
        test_partial_q_tile,
        test_chunked_prefill_with_long_prefix,
        test_chunk_smaller_than_block,
        test_shuffled_blocks_short,
        test_shuffled_blocks_q924,
        test_shuffled_blocks_long_prefix,
        test_shuffled_blocks_blocksize128,
        test_shuffled_blocks_blocksize256,
    ]:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
