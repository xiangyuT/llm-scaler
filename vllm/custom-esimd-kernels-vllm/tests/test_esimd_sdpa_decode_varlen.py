"""UT for esimd_sdpa_decode_varlen on PTL Xe3.

Reference computed on CPU (per project rule for PTL).
"""
import math
import torch
import pytest

from custom_esimd_kernels_vllm import esimd_sdpa_decode_varlen


def _ref_paged_sdpa_varlen(q, key_cache, value_cache, cu_seqlens_q,
                           seqused_k, block_table, scale, is_causal):
    """Reference SDPA over paged KV cache. Computed on CPU in fp32.

    q:           [total_q, n_heads, HD]
    key_cache:   [num_blocks, block_size, n_kv_heads, HD]
    value_cache: same
    cu_seqlens_q:[B+1] int32
    seqused_k:   [B] int32 or None (then full)
    block_table: [B, max_num_blocks_per_seq] int32
    Returns:     [total_q, n_heads, HD] same dtype as q
    """
    q = q.to("cpu", torch.float32)
    key_cache = key_cache.to("cpu", torch.float32)
    value_cache = value_cache.to("cpu", torch.float32)
    cu_seqlens_q = cu_seqlens_q.to("cpu")
    block_table = block_table.to("cpu")
    if seqused_k is not None:
        seqused_k = seqused_k.to("cpu")

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
        if seqused_k is not None:
            kv_len = int(seqused_k[b])
        else:
            kv_len = block_table.shape[1] * block_size

        # Materialize K, V for this sequence by gathering blocks.
        k_seq = torch.empty(kv_len, n_kv_heads, hd, dtype=torch.float32)
        v_seq = torch.empty(kv_len, n_kv_heads, hd, dtype=torch.float32)
        for kv_pos in range(kv_len):
            blk_idx = kv_pos // block_size
            blk_off = kv_pos % block_size
            phys_blk = int(block_table[b, blk_idx])
            k_seq[kv_pos] = key_cache[phys_blk, blk_off]
            v_seq[kv_pos] = value_cache[phys_blk, blk_off]

        # GQA expand
        head_ratio = n_heads // n_kv_heads
        k_seq = k_seq.repeat_interleave(head_ratio, dim=1)  # [kv_len, n_heads, HD]
        v_seq = v_seq.repeat_interleave(head_ratio, dim=1)

        for qi in range(seq_q_len):
            q_global = q_start + qi
            qv = q[q_global]                           # [n_heads, HD]
            scores = torch.einsum("hd,khd->hk", qv, k_seq) * scale  # [n_heads, kv_len]
            if is_causal:
                # The kernel uses causal_limit = kv_len - seq_q_len + qi
                causal_limit = kv_len - seq_q_len + qi
                mask_idx = torch.arange(kv_len)
                scores = scores.masked_fill(
                    mask_idx.unsqueeze(0) > causal_limit, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[q_global] = torch.einsum("hk,khd->hd", probs, v_seq)

    return out.to(q.dtype)


def _run_case(batch_q_lens, kv_lens, n_heads=8, n_kv_heads=2, hd=256,
              block_size=64, dtype=torch.float16, is_causal=True, seed=0):
    """Build inputs, run kernel and reference, compare."""
    torch.manual_seed(seed)
    device = "xpu"
    B = len(batch_q_lens)
    assert len(kv_lens) == B

    total_q = sum(batch_q_lens)
    cu_seqlens_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(batch_q_lens), dim=0).tolist()),
        dtype=torch.int32, device=device,
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    # Allocate enough blocks to cover the longest sequence.
    max_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
    total_blocks_needed = sum(
        (k + block_size - 1) // block_size for k in kv_lens
    )
    num_blocks = max(total_blocks_needed + 4, 8)

    block_table = torch.full(
        (B, max_blocks_per_seq), -1, dtype=torch.int32, device=device
    )
    cursor = 1  # leave block 0 unused as a sentinel
    for b in range(B):
        nb = (kv_lens[b] + block_size - 1) // block_size
        for j in range(nb):
            block_table[b, j] = cursor
            cursor += 1

    q = (torch.randn(total_q, n_heads, hd, device=device, dtype=torch.float32)
            * 0.05).to(dtype)
    key_cache = (
        torch.randn(num_blocks, block_size, n_kv_heads, hd,
                    device=device, dtype=torch.float32) * 0.05
    ).to(dtype)
    value_cache = (
        torch.randn(num_blocks, block_size, n_kv_heads, hd,
                    device=device, dtype=torch.float32) * 0.05
    ).to(dtype)

    scale = 1.0 / math.sqrt(hd)

    # Run kernel.
    out = esimd_sdpa_decode_varlen(
        q.contiguous(), key_cache, value_cache,
        cu_seqlens_q, max(kv_lens), is_causal, scale,
        block_table.contiguous(), seqused_k,
    )
    torch.xpu.synchronize()  # surface any async kernel error here

    # Reference on CPU.
    ref = _ref_paged_sdpa_varlen(
        q, key_cache, value_cache, cu_seqlens_q, seqused_k,
        block_table, scale, is_causal,
    )

    out_cpu = out.to("cpu", torch.float32)
    ref_cpu = ref.to(torch.float32)
    max_abs = (out_cpu - ref_cpu).abs().max().item()
    print(f"case batch_q={batch_q_lens} kv={kv_lens} hd={hd} dtype={dtype}: "
          f"max_abs={max_abs:.4g}")
    # fp16 tolerance — kernel does fp32 reduction, fp16 storage.
    tol = 5e-3 if dtype == torch.float16 else 1e-2
    assert max_abs < tol, f"abs diff {max_abs} > tol {tol}"


def test_decode_b1_short():
    """B=1, seq_q=1, kv_len=8 (single block) — pure decode."""
    _run_case([1], [8])


def test_decode_b1_two_blocks():
    _run_case([1], [80])


def test_decode_b2_varlen():
    _run_case([1, 1], [16, 100])


def test_prefill_b1_short():
    """B=1, seq_q=8, kv_len=8 — full prefill, no prior context."""
    _run_case([8], [8])


def test_prefill_b1_long():
    _run_case([64], [64])


def test_mixed_b2():
    """Prefill + decode mixed batch."""
    _run_case([16, 1], [16, 80])


def test_decode_b1_short_bf16():
    _run_case([1], [8], dtype=torch.bfloat16)


# --- Large-chunk chunked-prefill scenarios (the suspected crash path) ---
# Mimics what vllm dispatches at mnbt=128/256/512: a single long query batch
# with KV already containing a prefix.

def test_chunk_q256_kv256():
    """First chunk of a long prompt: q=256, kv=256 (no prefix)."""
    _run_case([256], [256])


def test_chunk_q256_kv512():
    """Second chunk: q=256, kv=512 (256 prefix + this chunk)."""
    _run_case([256], [512])


def test_chunk_q256_kv1024():
    _run_case([256], [1024])


def test_chunk_q256_kv4096():
    """4k context, last chunk before decode."""
    _run_case([256], [4096])


def test_chunk_q512_kv4096():
    _run_case([512], [4096])


def test_chunk_q128_kv4096():
    _run_case([128], [4096])


def test_chunk_q128_kv128():
    _run_case([128], [128])


def test_consecutive_chunks_q256_4chunks():
    """Mimic vllm chunked-prefill of a 4×256 = 1024-token prompt.

    Chunk i runs with (q_len=256, kv_len=256*(i+1)). The KV cache from
    earlier chunks must be live during later chunks (this is what vllm
    does between calls — same key_cache / value_cache tensor, more
    valid kv_len). Block table grows over chunks too.
    """
    torch.manual_seed(123)
    device = "xpu"
    n_heads, n_kv_heads, hd = 8, 2, 256
    block_size = 64
    chunk_q = 256
    n_chunks = 4
    total_kv = chunk_q * n_chunks  # 1024

    # Allocate cache big enough for the full prompt + sentinel block 0.
    blocks_per_seq = (total_kv + block_size - 1) // block_size
    num_blocks = blocks_per_seq + 4

    key_cache = (torch.randn(num_blocks, block_size, n_kv_heads, hd,
                             device=device, dtype=torch.float32) * 0.05
                 ).to(torch.float16)
    value_cache = (torch.randn(num_blocks, block_size, n_kv_heads, hd,
                               device=device, dtype=torch.float32) * 0.05
                   ).to(torch.float16)

    # Block table fixed shape: [B=1, blocks_per_seq]. Block 0 unused (sentinel).
    block_table = torch.full((1, blocks_per_seq), 0,
                             dtype=torch.int32, device=device)
    for j in range(blocks_per_seq):
        block_table[0, j] = j + 1  # blocks 1..blocks_per_seq

    scale = 1.0 / math.sqrt(hd)

    for chunk_id in range(n_chunks):
        kv_len = chunk_q * (chunk_id + 1)
        q = (torch.randn(chunk_q, n_heads, hd, device=device,
                         dtype=torch.float32) * 0.05).to(torch.float16)
        cu_seqlens_q = torch.tensor([0, chunk_q], dtype=torch.int32, device=device)
        seqused_k = torch.tensor([kv_len], dtype=torch.int32, device=device)

        out = esimd_sdpa_decode_varlen(
            q.contiguous(), key_cache, value_cache,
            cu_seqlens_q, kv_len, True, scale,
            block_table.contiguous(), seqused_k,
        )
        torch.xpu.synchronize()  # surface async crash here
        n_nan = torch.isnan(out.to("cpu", torch.float32)).sum().item()
        n_inf = torch.isinf(out.to("cpu", torch.float32)).sum().item()
        print(f"  chunk{chunk_id}  q=256 kv={kv_len}  "
              f"out.shape={tuple(out.shape)}  nan={n_nan} inf={n_inf}")
        assert n_nan == 0 and n_inf == 0, (
            f"chunk{chunk_id} kv={kv_len}: NaN/Inf in output")


if __name__ == "__main__":
    # Run inline so we get a clear pass/fail per case.
    for fn in [
        test_decode_b1_short,
        test_decode_b1_two_blocks,
        test_decode_b2_varlen,
        test_prefill_b1_short,
        test_prefill_b1_long,
        test_mixed_b2,
        test_decode_b1_short_bf16,
        test_chunk_q128_kv128,
        test_chunk_q128_kv4096,
        test_chunk_q256_kv256,
        test_chunk_q256_kv512,
        test_chunk_q256_kv1024,
        test_chunk_q256_kv4096,
        test_chunk_q512_kv4096,
        test_consecutive_chunks_q256_4chunks,
    ]:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
