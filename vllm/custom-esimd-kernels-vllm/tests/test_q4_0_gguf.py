"""Validate the XPU esimd_gemv_q4_0 kernel against the q4_0 reference.

Runs the SYCL kernel on real Qwen3.5-9B-Q4_0.gguf weights and compares to
gemv_ref (proven bit-exact vs GGML spec in test_repack_vs_spec.py).
"""
import sys
import torch
from q4_0_ref import (read_gguf_header, load_q4_0_tensor, repack_q4_0,
                      gemv_ref, QK4_0)

import custom_esimd_kernels_vllm  # noqa: F401 — registers torch.ops
from custom_esimd_kernels_vllm import esimd_gemv_q4_0


def run_one(path, name, seed=0):
    raw, N, K = load_q4_0_tensor(path, name)
    qw, sc = repack_q4_0(raw, N, K)            # cpu: [N,K/2] u8, [N,K/32] f16

    torch.manual_seed(seed)
    x = torch.randn(K, dtype=torch.float16)

    # reference (cpu fp32 accumulation)
    ref = gemv_ref(x, qw, sc)                  # [N] fp32

    # xpu kernel
    dev = "xpu"
    x_x = x.view(1, K).to(dev)
    qw_x = qw.to(dev)
    sc_x = sc.to(dev)
    out_x = torch.empty(1, N, dtype=torch.float16, device=dev)
    esimd_gemv_q4_0(x_x, qw_x, sc_x, out_x)
    torch.xpu.synchronize()
    got = out_x.view(N).float().cpu()

    # fp16 kernel accumulation vs fp32 ref: expect small relative error
    abs = (got - ref).abs()
    rel = abs / (ref.abs() + 1e-3)
    max_abs = abs.max().item()
    max_rel = rel.max().item()
    mean_abs = abs.mean().item()
    ok = max_rel < 0.02 or max_abs < 0.05
    print(f"  {name}: N={N} K={K}")
    print(f"    ref[:4] = {[round(v,4) for v in ref[:4].tolist()]}")
    print(f"    got[:4] = {[round(v,4) for v in got[:4].tolist()]}")
    print(f"    max_abs={max_abs:.4e}  mean_abs={mean_abs:.4e}  max_rel={max_rel:.4e}"
          f"   {'PASS' if ok else 'FAIL'}")
    return ok


def main(path="/models/Qwen3.5-9B-Q4_0.gguf"):
    print(f"torch {torch.__version__}  xpu={torch.xpu.is_available()}")
    tensors, _ = read_gguf_header(path)
    q40 = [n for n, (d, t, o) in tensors.items() if t == "Q4_0" and len(d) == 2]
    # test a few shapes: embedding (huge N), a square attn proj, an ffn
    cands = ["blk.0.attn_qkv.weight", "blk.3.attn_q.weight",
             "blk.0.attn_gate.weight", "token_embd.weight"]
    picks = [c for c in cands if c in q40][:3]
    if not picks:
        picks = q40[:2]
    all_ok = True
    for nm in picks:
        all_ok &= run_one(path, nm)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import os
    rc = main(sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen3.5-9B-Q4_0.gguf")
    os._exit(rc)
