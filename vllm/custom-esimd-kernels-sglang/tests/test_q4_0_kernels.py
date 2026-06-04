"""Minimal q4_0 XPU kernel test — synthetic, tiny, step-by-step timed.

Isolates: (1) does the kernel return at all? (2) is it numerically correct?
Uses tiny shapes so CPU ref is instant and a hang must be the kernel.
"""
import os, sys, time
import torch

print("step0: import", flush=True)
t = time.time()
import custom_esimd_kernels_sglang  # noqa
from custom_esimd_kernels_sglang import esimd_gemv_q4_0, esimd_gemm_q4_0
print(f"  import OK ({time.time()-t:.2f}s)", flush=True)


def make_q4_0(N, K, seed=0):
    """Synthetic INTERLEAVED q4_0: qweight [N,K/2] u8, scale [N,K/32] f16.
    Returns (qweight, scale, dense_w [N,K] fp32) so we have exact ground truth.
    Interleaved: byte j low nibble = elem 2j (even K), high = elem 2j+1 (odd K).
    """
    torch.manual_seed(seed)
    blocks = K // 32
    q = torch.randint(0, 16, (N, blocks, 32), dtype=torch.int32)
    scale = (torch.rand(N, blocks) * 0.02 - 0.01).to(torch.float16)
    sc = scale.float().view(N, blocks, 1)
    w = (q - 8).float() * sc                         # [N, blocks, 32]
    dense = w.view(N, K)
    # interleaved pack: byte j = nib[2j] | nib[2j+1]<<4
    even = q[:, :, 0::2]                              # elems 0,2,...,30
    odd = q[:, :, 1::2]                               # elems 1,3,...,31
    qbytes = (even | (odd << 4)).to(torch.uint8).view(N, K // 2)
    return qbytes.contiguous(), scale.contiguous(), dense


def main():
    for (N, K) in [(2, 32), (4, 128), (64, 256), (256, 4096)]:
        print(f"\n=== N={N} K={K} ===", flush=True)
        qw, sc, dense = make_q4_0(N, K)
        torch.manual_seed(1)
        x = torch.randn(K, dtype=torch.float16)
        ref = (x.float() @ dense.t())                 # [N] fp32, instant

        print("  -> to xpu", flush=True)
        t = time.time()
        x_x = x.view(1, K).to("xpu")
        qw_x = qw.to("xpu"); sc_x = sc.to("xpu")
        out_x = torch.empty(1, N, dtype=torch.float16, device="xpu")
        torch.xpu.synchronize()
        print(f"     h2d done ({time.time()-t:.2f}s); calling kernel", flush=True)

        t = time.time()
        esimd_gemv_q4_0(x_x, qw_x, sc_x, out_x)
        torch.xpu.synchronize()
        print(f"     kernel returned ({time.time()-t:.2f}s)", flush=True)

        got = out_x.view(N).float().cpu()
        abs_ = (got - ref).abs()
        max_abs = abs_.max().item()
        max_rel = (abs_ / (ref.abs() + 1e-3)).max().item()
        ok = max_rel < 0.02 or max_abs < 0.05
        print(f"     ref[:4]={[round(v,4) for v in ref[:4].tolist()]}", flush=True)
        print(f"     got[:4]={[round(v,4) for v in got[:4].tolist()]}", flush=True)
        print(f"     max_abs={max_abs:.3e} max_rel={max_rel:.3e}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    # --- prefill GEMM (M>=2) ---
    print("\n=== GEMM (M>=2) ===", flush=True)
    for (M, N, K) in [(2, 16, 128), (8, 64, 256), (32, 256, 4096), (5, 4096, 4096)]:
        qw, sc, dense = make_q4_0(N, K, seed=7)
        torch.manual_seed(2)
        x = torch.randn(M, K, dtype=torch.float16)
        ref = (x.float() @ dense.t())                 # [M, N] fp32
        x_x = x.to("xpu"); qw_x = qw.to("xpu"); sc_x = sc.to("xpu")
        out_x = torch.empty(M, N, dtype=torch.float16, device="xpu")
        t = time.time()
        esimd_gemm_q4_0(x_x, qw_x, sc_x, out_x)
        torch.xpu.synchronize()
        got = out_x.float().cpu()
        abs_ = (got - ref).abs()
        max_abs = abs_.max().item()
        max_rel = (abs_ / (ref.abs() + 1e-3)).max().item()
        ok = max_rel < 0.02 or max_abs < 0.06
        print(f"  M={M} N={N} K={K}: kernel {time.time()-t:.2f}s  "
              f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
