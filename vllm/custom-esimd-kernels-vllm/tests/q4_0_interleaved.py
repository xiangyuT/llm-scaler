"""GGUF q4_0 -> INTERLEAVED repack (for DPAS GEMM + group-32 GEMV).

The DPAS int4 GEMM (ported from llm-scaler 6f61085 int4_GEMM.h) and the
group-32 GEMV both expect the INTERLEAVED nibble layout:
    byte j  ->  low nibble = element 2j (K_even),  high nibble = 2j+1 (K_odd)
(adjacent K pairs = DPAS VNNI layout).

GGUF q4_0 on disk is SPLIT-HALF (byte j: low=elem j (0..15), high=elem j+16).
So the repack is a within-32-block nibble permutation. The *values* are
unchanged — we only relabel which byte/nibble slot stores each element — so
the dequant of the interleaved layout must equal the split-half dequant
(validated in test_interleaved_vs_splithalf.py).

    repack_q4_0_interleaved(raw, N, K) -> (qweight [N,K/2] u8, scale [N,K/32] f16)
"""
import torch
from q4_0_ref import QK4_0, Q4_0_BLOCK_BYTES, dequant_q4_0 as _dequant_splithalf


def repack_q4_0_interleaved(raw, N, K):
    """GGUF q4_0 raw bytes -> interleaved (qweight [N,K/2] u8, scale [N,K/32] f16)."""
    blocks = K // QK4_0
    buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(
        N, blocks, Q4_0_BLOCK_BYTES)
    scale = buf[:, :, 0:2].contiguous().view(torch.float16).view(N, blocks)
    qs = buf[:, :, 2:18].contiguous()                  # [N, blocks, 16]

    # GGUF nibbles per block: nib[i] for i in 0..31
    #   i < 16 : low nibble of byte i
    #   i >=16 : high nibble of byte (i-16)
    lo = qs & 0x0F                                     # [N, blocks, 16] -> elems 0..15
    hi = (qs >> 4) & 0x0F                              # -> elems 16..31
    nib = torch.cat([lo, hi], dim=2)                   # [N, blocks, 32], nib[...,i]=elem i

    # Interleaved target: out_byte[j] = nib[2j] | nib[2j+1]<<4   (j in 0..15)
    even = nib[:, :, 0::2]                              # elems 0,2,4,...,30  [N,blocks,16]
    odd = nib[:, :, 1::2]                               # elems 1,3,5,...,31
    out = (even | (odd << 4)).to(torch.uint8).view(N, K // 2)
    return out.contiguous(), scale.contiguous()


def dequant_q4_0_interleaved(qweight, scale):
    """Interleaved (qweight [N,K/2] u8, scale [N,K/32] f16) -> dense [N,K] fp32.

    byte j: low nibble -> elem 2j, high nibble -> elem 2j+1.
    """
    N, Khalf = qweight.shape
    K = Khalf * 2
    blocks = K // QK4_0
    qw = qweight.view(N, blocks, 16).to(torch.int32)
    even = (qw & 0x0F) - 8         # [N, blocks, 16] -> elems 0,2,4,...
    odd = ((qw >> 4) & 0x0F) - 8   # -> elems 1,3,5,...
    sc = scale.to(torch.float32).view(N, blocks, 1)
    out = torch.empty(N, blocks, QK4_0, dtype=torch.float32)
    out[:, :, 0::2] = even.to(torch.float32) * sc
    out[:, :, 1::2] = odd.to(torch.float32) * sc
    return out.view(N, K)


if __name__ == "__main__":
    import sys
    from q4_0_ref import (read_gguf_header, load_q4_0_tensor, repack_q4_0)
    path = sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen3.5-9B-Q4_0.gguf"
    tensors, _ = read_gguf_header(path)
    q40 = [n for n, (d, t, o) in tensors.items() if t == "Q4_0" and len(d) == 2]
    name = "blk.0.attn_gate.weight" if "blk.0.attn_gate.weight" in q40 else q40[0]
    raw, N, K = load_q4_0_tensor(path, name)

    # split-half dense (proven == GGML spec) vs interleaved dense
    qw_s, sc_s = repack_q4_0(raw, N, K)
    qw_i, sc_i = repack_q4_0_interleaved(raw, N, K)
    n = min(N, 8)
    dense_s = _dequant_splithalf(qw_s[:n].contiguous(), sc_s[:n].contiguous())
    dense_i = dequant_q4_0_interleaved(qw_i[:n].contiguous(), sc_i[:n].contiguous())
    diff = (dense_s - dense_i).abs().max().item()
    print(f"{name}: N={N} K={K}")
    print(f"  scale identical: {torch.equal(sc_s, sc_i)}")
    print(f"  MAX |split-half dense - interleaved dense| over [{n},{K}] = {diff:.3e}")
    assert diff == 0.0, "interleaved repack changes values!"
    print("PASS: interleaved repack preserves GGML values (bit-exact vs split-half)")
