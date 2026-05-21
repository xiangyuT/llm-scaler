"""Microbench: moe_forward_full_int4 wall time at varying n_tokens.

Production shapes (Qwen3.5-35B-A3B TP=2 per-rank):
  H=2048, D=256, D_S=256, E=256, top_k=8

Set VLLM_MOE_PREFILL_MIN_TOKENS=1 to force prefill path,
    VLLM_MOE_PREFILL_MIN_TOKENS=99999 to force decode-style path.
"""
import os
import time
import torch
import numpy as np

DEVICE = "xpu"
GROUP_SIZE = 128
PACK_FACTOR = 8


def quantize_int4(weight_fp16, group_size=GROUP_SIZE):
    N, K = weight_fp16.shape
    n_groups = K // group_size
    w = weight_fp16.float().numpy().reshape(N, n_groups, group_size)
    max_abs = np.abs(w).max(axis=2)
    sc = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float16)
    sce = sc[:, :, None].astype(np.float32)
    q = np.round(w / sce).clip(-8, 7).astype(np.int32) + 8
    qf = q.reshape(N, K)
    packed = np.zeros((N, K // PACK_FACTOR), dtype=np.uint32)
    qg = qf.reshape(N, K // PACK_FACTOR, PACK_FACTOR).astype(np.uint32)
    for b in range(PACK_FACTOR):
        packed |= (qg[:, :, b] & 0xF) << (b * 4)
    return torch.from_numpy(packed.view(np.int32)), torch.from_numpy(sc)


def main():
    from custom_esimd_kernels_vllm import moe_int4_ops

    label = os.environ.get("LABEL", "?")
    H, D, D_S, E, TK = 2048, 256, 256, 256, 8
    NUM_SHARED = 1
    print(f"=== bench [{label}] (H={H} D={D} E={E} TK={TK}) "
          f"VLLM_MOE_PREFILL_MIN_TOKENS={os.environ.get('VLLM_MOE_PREFILL_MIN_TOKENS','unset')} ===")

    torch.manual_seed(42)
    w13 = (torch.randn(E, 2 * D, H) * 0.02).half()
    w2 = (torch.randn(E, H, D) * 0.02).half()
    sgu = (torch.randn(2 * D_S, H) * 0.02).half()
    sdw = (torch.randn(H, D_S) * 0.02).half()
    sgw = (torch.randn(1, H) * 0.02).half()

    w13_qw_list, w13_sc_list, w2_qw_list, w2_sc_list = [], [], [], []
    for e in range(E):
        qw, sc = quantize_int4(w13[e]); w13_qw_list.append(qw); w13_sc_list.append(sc)
        qw, sc = quantize_int4(w2[e]);  w2_qw_list.append(qw);  w2_sc_list.append(sc)
    w13_qw = torch.stack(w13_qw_list).to(DEVICE)
    w13_sc = torch.stack(w13_sc_list).to(DEVICE)
    w2_qw = torch.stack(w2_qw_list).to(DEVICE)
    w2_sc = torch.stack(w2_sc_list).to(DEVICE)
    sgu_d = sgu.to(DEVICE); sdw_d = sdw.to(DEVICE); sgw_d = sgw.to(DEVICE)
    _dummy = torch.empty(0, device=DEVICE, dtype=torch.float16)

    print(f"\n{'n_tokens':>8} | {'ms/call':>9}")
    print("-" * 22)
    for n in [16, 32, 64, 128, 256]:
        x = (torch.randn(n, H) * 0.1).half().to(DEVICE)
        logits = (torch.randn(n, E) * 0.1).half().to(DEVICE)

        for _ in range(3):
            _ = moe_int4_ops.moe_forward_full_int4(
                x, logits, w13_qw, w13_sc, sgu_d, _dummy,
                w2_qw, w2_sc, sdw_d, _dummy, sgw_d,
                TK, NUM_SHARED, E, True)
        torch.xpu.synchronize()
        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = moe_int4_ops.moe_forward_full_int4(
                x, logits, w13_qw, w13_sc, sgu_d, _dummy,
                w2_qw, w2_sc, sdw_d, _dummy, sgw_d,
                TK, NUM_SHARED, E, True)
        torch.xpu.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000
        print(f"{n:>8} | {ms:>8.2f}")


if __name__ == "__main__":
    main()
