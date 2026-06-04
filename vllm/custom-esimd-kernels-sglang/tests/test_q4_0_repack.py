"""Cross-validate repack_q4_0 + dequant_q4_0 against the raw GGML spec.

Two independent decode paths must agree element-by-element:
  (A) raw 18-byte block decoded inline per GGML reference (no repack)
  (B) my repack_q4_0 -> dequant_q4_0 pipeline

If they match, the repack is provably lossless and spec-correct.
"""
import struct
import torch
from q4_0_ref import (read_gguf_header, load_q4_0_tensor, repack_q4_0,
                      dequant_q4_0, QK4_0, Q4_0_BLOCK_BYTES)


def decode_raw_spec(raw, N, K, max_rows=4):
    """(A) Decode straight from 18-byte blocks per GGML q4_0 reference."""
    blocks_per_row = K // QK4_0
    out = torch.zeros(min(N, max_rows), K, dtype=torch.float32)
    for n in range(min(N, max_rows)):
        row_off = n * blocks_per_row * Q4_0_BLOCK_BYTES
        for b in range(blocks_per_row):
            o = row_off + b * Q4_0_BLOCK_BYTES
            d = struct.unpack("<e", raw[o:o + 2])[0]
            qs = raw[o + 2:o + 18]  # 16 bytes
            for j in range(16):
                byte = qs[j]
                lo = (byte & 0x0F) - 8
                hi = ((byte >> 4) & 0x0F) - 8
                out[n, b * 32 + j] = lo * d        # low nibble -> elem j
                out[n, b * 32 + j + 16] = hi * d   # high nibble -> elem j+16
    return out


def main(path="/models/Qwen3.5-9B-Q4_0.gguf"):
    tensors, _ = read_gguf_header(path)
    q40 = [n for n, (d, t, o) in tensors.items() if t == "Q4_0" and len(d) == 2]
    name = "blk.0.ffn_down.weight" if "blk.0.ffn_down.weight" in q40 else q40[0]
    raw, N, K = load_q4_0_tensor(path, name)
    print(f"tensor {name}: N={N} K={K}")

    # (B) repack pipeline
    qw, sc = repack_q4_0(raw, N, K)
    deq_b = dequant_q4_0(qw[:4].contiguous(), sc[:4].contiguous())  # [4, K]

    # (A) raw spec decode
    deq_a = decode_raw_spec(raw, N, K, max_rows=4)

    diff = (deq_a - deq_b).abs().max().item()
    print(f"(A) raw-spec  first row[:6] = {[round(x,5) for x in deq_a[0,:6].tolist()]}")
    print(f"(B) repack    first row[:6] = {[round(x,5) for x in deq_b[0,:6].tolist()]}")
    print(f"(A) raw-spec  row0[16:22]   = {[round(x,5) for x in deq_a[0,16:22].tolist()]}")
    print(f"(B) repack    row0[16:22]   = {[round(x,5) for x in deq_b[0,16:22].tolist()]}")
    print(f"\nMAX |A - B| over [4,{K}] = {diff:.3e}")
    assert diff == 0.0, "repack does NOT match GGML spec!"
    print("PASS: repack_q4_0 + dequant_q4_0 == GGML raw spec (bit-exact)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen3.5-9B-Q4_0.gguf")
