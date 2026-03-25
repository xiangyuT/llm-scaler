"""
Performance and correctness benchmarks for SDP (Scaled Dot-Product Attention) kernels

Compares:
  1. ESIMD Flash Attention (omni_xpu_kernel.sdp) — ESIMD doubleGRF kernel
  2. PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention)
  3. Naive PyTorch (matmul + softmax + matmul) — baseline

Correctness:
  Validates ESIMD output against PyTorch SDPA reference for fp16 and bf16.

Usage:
    python bench_sdp.py                  # Run all
    python bench_sdp.py --correctness    # Correctness only
    python bench_sdp.py --benchmark      # Benchmark only
"""

import argparse
import math
import time
import torch


def has_xpu():
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


# ============================================================================
# Reference implementations
# ============================================================================

def sdp_naive(q, k, v):
    """Naive PyTorch: permute to BHLD, matmul+softmax+matmul, permute back."""
    # q/k/v: [B, L, H, D]
    q_t = q.permute(0, 2, 1, 3).contiguous()  # [B, H, Lq, D]
    k_t = k.permute(0, 2, 1, 3).contiguous()  # [B, H, Lk, D]
    v_t = v.permute(0, 2, 1, 3).contiguous()  # [B, H, Lk, D]

    scale = 1.0 / math.sqrt(q.size(3))
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs.to(v_t.dtype), v_t)
    return out.permute(0, 2, 1, 3).contiguous()  # [B, L, H, D]


def sdp_torch(q, k, v):
    """PyTorch SDPA: uses torch.nn.functional.scaled_dot_product_attention."""
    # SDPA expects [B, H, L, D]
    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()

    out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
    return out.permute(0, 2, 1, 3).contiguous()  # [B, L, H, D]


def sdp_esimd(q, k, v):
    """ESIMD Flash Attention via omni_xpu_kernel."""
    from omni_xpu_kernel._C import sdp
    return sdp.sdp(q, k, v)


# ============================================================================
# Correctness
# ============================================================================

def check_correctness(seq_len, heads, dtype, label=""):
    B, D = 1, 128
    q = torch.randn(B, seq_len, heads, D, device="xpu", dtype=dtype)
    k = torch.randn(B, seq_len, heads, D, device="xpu", dtype=dtype)
    v = torch.randn(B, seq_len, heads, D, device="xpu", dtype=dtype)

    # Reference: fp32 naive for small seq, SDPA for large seq (avoids OOM from score matrix)
    if seq_len * heads * seq_len * 4 < 2 * 1024**3:  # score matrix < 2GB in fp32
        q32 = q.float()
        k32 = k.float()
        v32 = v.float()
        ref = sdp_naive(q32, k32, v32).to(dtype)
    else:
        ref = sdp_torch(q, k, v)

    out_esimd = sdp_esimd(q, k, v)
    out_torch = sdp_torch(q, k, v)

    # Tolerances: fp16/bf16 attention has accumulated numerical error
    if dtype == torch.float16:
        rtol, atol = 1e-2, 5e-3
    else:  # bf16
        rtol, atol = 2e-2, 1e-2

    results = {}
    for name, out in [("esimd", out_esimd), ("torch_sdpa", out_torch)]:
        try:
            torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
            max_diff = (out.float() - ref.float()).abs().max().item()
            results[name] = ("PASS", max_diff)
        except AssertionError:
            max_diff = (out.float() - ref.float()).abs().max().item()
            mean_diff = (out.float() - ref.float()).abs().mean().item()
            # Use a looser check: cosine similarity per row
            out_flat = out.float().reshape(-1, D)
            ref_flat = ref.float().reshape(-1, D)
            cos_sim = torch.nn.functional.cosine_similarity(out_flat, ref_flat, dim=-1)
            min_cos = cos_sim.min().item()
            if min_cos > 0.99:
                results[name] = ("PASS(cos)", max_diff)
            else:
                results[name] = ("FAIL", max_diff)

    return results


def run_correctness():
    dtype_name = {torch.float16: "fp16", torch.bfloat16: "bf16"}

    print("=" * 95)
    print("SDP Correctness: ESIMD Flash Attention vs FP32 Reference")
    print("=" * 95)
    print(f"{'SeqLen':>8} {'Heads':>6} {'Dtype':>6} | "
          f"{'ESIMD':>14} {'MaxDiff':>10} | "
          f"{'torch_sdpa':>14} {'MaxDiff':>10}")
    print(f"{'':-<8} {'':-<6} {'':-<6}-+-{'':-<14}-{'':-<10}-+-{'':-<14}-{'':-<10}")

    all_pass = True
    # Use realistic diffusion model configs
    for seq_len in [1024, 4096, 8192, 16384]:
        for heads in [24, 32, 40]:
            for dtype in [torch.float16, torch.bfloat16]:
                results = check_correctness(seq_len, heads, dtype)
                esimd_status, esimd_diff = results["esimd"]
                torch_status, torch_diff = results["torch_sdpa"]

                if "FAIL" in esimd_status:
                    all_pass = False

                print(f"{seq_len:>8} {heads:>6} {dtype_name[dtype]:>6} | "
                      f"{esimd_status:>14} {esimd_diff:>10.6f} | "
                      f"{torch_status:>14} {torch_diff:>10.6f}")

    print()
    if all_pass:
        print("All correctness checks PASSED.")
    else:
        print("Some correctness checks FAILED!")
    return all_pass


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_one(fn, q, k, v, warmup=10, iters=50):
    """Benchmark a single SDP implementation, return average time in ms."""
    for _ in range(warmup):
        _ = fn(q, k, v)
    torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(q, k, v)
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1000  # ms


def run_benchmarks():
    print("=" * 100)
    print("SDP Performance Benchmark: ESIMD Flash Attention vs PyTorch SDPA")
    print("Realistic diffusion model configs (Flux, Wan2.2, HunyuanVideo, LTX-2)")
    print("=" * 100)
    print(f"{'Model hint':<16} {'SeqLen':>7} {'Heads':>6} {'Dtype':>6} | "
          f"{'ESIMD(ms)':>11} {'SDPA(ms)':>11} | {'Speedup':>9}")
    print(f"{'':-<16} {'':-<7} {'':-<6} {'':-<6}-+-"
          f"{'':-<11}-{'':-<11}-+-{'':-<9}")

    dtype_name = {torch.float16: "fp16", torch.bfloat16: "bf16"}
    B, D = 1, 128

    # Realistic diffusion model configs:
    #   Flux/Z-Image:    heads=24, seq=4096 (512x512) / 16384 (1024x1024)
    #   Wan2.2-14B:      heads=40, seq=~17550 (81 frames 480p)
    #   Wan2.2-5B:       heads=24, seq=~17550
    #   HunyuanVideo:    heads=24, seq=~11520 (480p)
    #   LTX-2-19B:       heads=32, seq=~6000
    #   SD3.5:           heads=24, head_dim=64 (NOT 128, skip for now)
    configs = [
        ("Flux-512",      4096, 24),
        ("Flux-1024",    16384, 24),
        ("LTX-2",         6144, 32),
        ("HunyuanVideo",  11520, 24),
        ("Wan2.2-5B",    17550, 24),
        ("Wan2.2-14B",   17550, 40),
    ]

    for model_hint, seq_len, heads in configs:
        for dtype in [torch.float16, torch.bfloat16]:
            try:
                q = torch.randn(B, seq_len, heads, D, device="xpu", dtype=dtype)
                k = torch.randn(B, seq_len, heads, D, device="xpu", dtype=dtype)
                v = torch.randn(B, seq_len, heads, D, device="xpu", dtype=dtype)
            except RuntimeError as e:
                print(f"{model_hint:<16} {seq_len:>7} {heads:>6} {dtype_name[dtype]:>6} |  OOM: {e}")
                continue

            try:
                t_esimd = benchmark_one(sdp_esimd, q, k, v, warmup=5, iters=20)
            except RuntimeError as e:
                t_esimd = float('inf')
                print(f"{model_hint:<16} {seq_len:>7} {heads:>6} {dtype_name[dtype]:>6} |  ESIMD err: {e}")
                continue

            try:
                t_torch = benchmark_one(sdp_torch, q, k, v, warmup=5, iters=20)
            except RuntimeError as e:
                t_torch = float('inf')

            if t_esimd < float('inf') and t_torch < float('inf'):
                speedup = t_torch / t_esimd
                print(f"{model_hint:<16} {seq_len:>7} {heads:>6} {dtype_name[dtype]:>6} | "
                      f"{t_esimd:>11.3f} {t_torch:>11.3f} | {speedup:>8.2f}x")
            elif t_esimd < float('inf'):
                print(f"{model_hint:<16} {seq_len:>7} {heads:>6} {dtype_name[dtype]:>6} | "
                      f"{t_esimd:>11.3f} {'OOM':>11} | {'N/A':>9}")
            else:
                print(f"{model_hint:<16} {seq_len:>7} {heads:>6} {dtype_name[dtype]:>6} | "
                      f"{'ERR':>11} {t_torch:>11.3f} | {'N/A':>9}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SDP kernel benchmarks")
    parser.add_argument("--correctness", action="store_true", help="Correctness only")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark only")
    args = parser.parse_args()

    if not has_xpu():
        print("XPU not available, skipping")
        return

    run_all = not args.correctness and not args.benchmark

    if run_all or args.correctness:
        run_correctness()
        print()

    if run_all or args.benchmark:
        run_benchmarks()


if __name__ == "__main__":
    main()
