"""GGUF q4_0 reference: dequant + repack to ESIMD-friendly layout.

Ground-truth (pure torch, no custom kernel) for validating the SYCL q4_0 GEMV.

GGML q4_0 on-disk block (18 bytes, 32 elements):
    struct { fp16 d; uint8 qs[16]; }
  - qs[i] low  nibble -> element i        (i in 0..15)   [SPLIT-HALF]
  - qs[i] high nibble -> element i+16     (i in 16..31)
  - dequant: w = (nibble - 8) * d         (symmetric, d can be negative)

Repack (load-time, INT4 stays INT4 — NO dequant, preserves bandwidth win):
    qweight  [N, K/2] uint8   — qs bytes, blocks laid contiguously per row
    scale    [N, K/32] fp16   — the per-block d, extracted out of the block

This repacked layout is what the SYCL q4_0 GEMV kernel consumes:
  - group_size = 32 (NOT 128 like the symmetric ESIMD int4 kernel)
  - nibble mapping = split-half within each 32-block (NOT interleaved)
"""
import struct
import torch

QK4_0 = 32          # elements per q4_0 block
Q4_0_BLOCK_BYTES = 18  # 2 (fp16 d) + 16 (qs)


def read_gguf_header(path):
    """Minimal GGUF v3 reader -> (tensors, data_start). tensors: name->(dims,ttype,offset)."""
    GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0",
            12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}

    def rd(f, fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

    def rstr(f):
        (ln,) = rd(f, "<Q")
        return f.read(ln).decode("utf-8", "ignore")

    def skip(f, vt):
        if vt == 8:
            return rstr(f)
        if vt == 9:
            (et,) = rd(f, "<I")
            (ln,) = rd(f, "<Q")
            for _ in range(ln):
                skip(f, et)
            return None
        s = {0: "<b", 1: "<B", 2: "<h", 3: "<H", 4: "<i", 5: "<I",
             6: "<f", 7: "<?", 10: "<q", 11: "<Q", 12: "<d"}
        (v,) = rd(f, s[vt])
        return v

    with open(path, "rb") as f:
        f.read(4)
        rd(f, "<I")
        (nt,) = rd(f, "<Q")
        (nkv,) = rd(f, "<Q")
        alignment = 32
        for _ in range(nkv):
            k = rstr(f)
            (vt,) = rd(f, "<I")
            v = skip(f, vt)
            if k == "general.alignment" and isinstance(v, int):
                alignment = v
        tensors = {}
        for _ in range(nt):
            nm = rstr(f)
            (nd,) = rd(f, "<I")
            dims = rd(f, f"<{nd}Q")
            (tt,) = rd(f, "<I")
            (off,) = rd(f, "<Q")
            tensors[nm] = (dims, GGML.get(tt, tt), off)
        pos = f.tell()
        data_start = (pos + alignment - 1) // alignment * alignment
    return tensors, data_start


def load_q4_0_tensor(path, name):
    """Read a q4_0 tensor's raw bytes. Returns (raw_bytes, N, K).

    GGUF stores dims reversed vs torch row-major: dims=(K, N) -> [N rows, K cols].
    """
    tensors, data_start = read_gguf_header(path)
    dims, ttype, off = tensors[name]
    assert ttype == "Q4_0", f"{name} is {ttype}, not Q4_0"
    K = dims[0]
    N = dims[1] if len(dims) > 1 else 1
    assert K % QK4_0 == 0
    nblocks = N * (K // QK4_0)
    nbytes = nblocks * Q4_0_BLOCK_BYTES
    with open(path, "rb") as f:
        f.seek(data_start + off)
        raw = f.read(nbytes)
    return raw, N, K


def repack_q4_0(raw, N, K):
    """GGUF q4_0 raw bytes -> (qweight [N,K/2] u8, scale [N,K/32] f16).

    Splits each 18-byte block into its 16 qs bytes (contiguous) and its
    fp16 scale d. INT4 stays INT4 — no dequant.
    """
    blocks_per_row = K // QK4_0
    buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(
        N, blocks_per_row, Q4_0_BLOCK_BYTES)
    # scale d = first 2 bytes of each block, as fp16
    d_bytes = buf[:, :, 0:2].contiguous()
    scale = d_bytes.view(torch.float16).view(N, blocks_per_row)
    # qs = remaining 16 bytes
    qs = buf[:, :, 2:18].contiguous().view(N, blocks_per_row * 16)  # [N, K/2]
    return qs, scale


def dequant_q4_0(qweight, scale):
    """Repacked (qweight [N,K/2] u8, scale [N,K/32] f16) -> dense [N,K] fp32.

    split-half: within block b, byte j (j in 0..15):
        low  nibble -> elem b*32 + j
        high nibble -> elem b*32 + j + 16
    """
    N, Khalf = qweight.shape
    K = Khalf * 2
    blocks = K // QK4_0
    qw = qweight.view(N, blocks, 16).to(torch.int32)
    lo = (qw & 0x0F) - 8           # [N, blocks, 16] -> elems 0..15
    hi = ((qw >> 4) & 0x0F) - 8    # -> elems 16..31
    sc = scale.to(torch.float32).view(N, blocks, 1)
    lo = lo.to(torch.float32) * sc
    hi = hi.to(torch.float32) * sc
    out = torch.empty(N, blocks, QK4_0, dtype=torch.float32)
    out[:, :, 0:16] = lo
    out[:, :, 16:32] = hi
    return out.view(N, K)


def gemv_ref(x_fp16, qweight, scale):
    """Reference GEMV: out[n] = sum_k x[k] * dequant_w[n,k].  x:[K], out:[N]."""
    w = dequant_q4_0(qweight, scale)               # [N, K] fp32
    return (x_fp16.to(torch.float32) @ w.t())       # [N]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen3.5-9B-Q4_0.gguf"
    # pick a real q4_0 linear weight
    tensors, _ = read_gguf_header(path)
    q40 = [n for n, (d, t, o) in tensors.items() if t == "Q4_0" and len(d) == 2]
    name = "blk.0.ffn_down.weight" if "blk.0.ffn_down.weight" in q40 else q40[0]
    raw, N, K = load_q4_0_tensor(path, name)
    qw, sc = repack_q4_0(raw, N, K)
    print(f"{name}: N={N} K={K}")
    print(f"  qweight {tuple(qw.shape)} {qw.dtype}   scale {tuple(sc.shape)} {sc.dtype}")
    w = dequant_q4_0(qw, sc)
    print(f"  dequant [{N},{K}] range [{w.min():.5f}, {w.max():.5f}] mean|w|={w.abs().mean():.5f}")
    # sanity GEMV
    torch.manual_seed(0)
    x = torch.randn(K, dtype=torch.float16)
    out = gemv_ref(x, qw, sc)
    print(f"  gemv out[:4] = {out[:4].tolist()}")
    print(f"  gemv out range [{out.min():.4f}, {out.max():.4f}]")
