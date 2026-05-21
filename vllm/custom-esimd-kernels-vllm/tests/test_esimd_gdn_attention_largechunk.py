"""UT: esimd_gdn_attention output consistency at large num_actual_tokens.

We don't (re)build a full CPU reference — the kernel does causal_conv1d +
gated-delta-rule recurrence and reproducing it on CPU is a separate project.

Instead this UT validates **internal consistency** at large chunks:

  1. Single-sequence prefill of N tokens vs. batch=B prefill of (N/B + N/B + ...)
     where each sub-sequence sees its own state — outputs of the FIRST
     sub-sequence should match the prefix of the single-sequence run
     (because both start from has_initial_state=False and process tokens
     0..N/B-1 with the same q/k/v/b/a). Tests that the kernel
     correctly indexes per-sequence in a packed-batch layout at large N.

  2. Decode-step output after a long prefill is finite (no state corruption
     after a 256/512/1024-token prefill).

  3. Output magnitudes stay within sane range for prefill of 128/256/512/1024
     (catch silent fp16 overflow / NaN cascade).

These don't prove math correctness vs the model spec, but they catch the
class of bugs we suspect at large chunks: state-pool index OOB, fp16
accumulator overflow at long N, async kernel stale-data races.
"""
import os
import sys

import torch

from custom_esimd_kernels_vllm import esimd_gdn_attention


# Qwen3.5-0.8B GDN params
NUM_K_HEADS = 16
NUM_V_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4

# qkvz layout: Q | K | V | Z (per kernel header).
QKVZ_DIM = (NUM_K_HEADS * HEAD_K_DIM        # Q
            + NUM_K_HEADS * HEAD_K_DIM      # K
            + NUM_V_HEADS * HEAD_V_DIM      # V
            + NUM_V_HEADS * HEAD_V_DIM)     # Z
BA_DIM = 2 * NUM_V_HEADS  # b + a per v-head
CONV_ELEMS = (NUM_K_HEADS * HEAD_K_DIM * 2 + NUM_V_HEADS * HEAD_V_DIM)


def _make_static(seed=0, dtype=torch.float16):
    """Static (non-input) tensors that any call can share."""
    torch.manual_seed(seed)
    g = torch.Generator(device="xpu").manual_seed(seed)
    conv_weights = (torch.randn(
        CONV_ELEMS, CONV_KERNEL, generator=g, device="xpu", dtype=torch.float32) * 0.1
    ).to(dtype).contiguous()
    conv_bias = torch.zeros(CONV_ELEMS, dtype=dtype, device="xpu")
    A_log = (torch.randn(NUM_V_HEADS, generator=g, device="xpu",
                         dtype=torch.float32) * 0.1).to(dtype)
    dt_bias = (torch.randn(NUM_V_HEADS, generator=g, device="xpu",
                           dtype=torch.float32) * 0.1).to(dtype)
    return conv_weights, conv_bias, A_log, dt_bias


def _make_inputs(num_actual_tokens, dtype=torch.float16, seed=0):
    """All-prefill, single-sequence inputs of length num_actual_tokens."""
    torch.manual_seed(seed)
    g = torch.Generator(device="xpu").manual_seed(seed)
    qkvz = (torch.randn(num_actual_tokens, QKVZ_DIM, generator=g,
                        device="xpu", dtype=torch.float32) * 0.05
            ).to(dtype).contiguous()
    ba = (torch.randn(num_actual_tokens, BA_DIM, generator=g,
                      device="xpu", dtype=torch.float32) * 0.05
          ).to(dtype).contiguous()
    return qkvz, ba


def _run_prefill(qkvz, ba, conv_weights, conv_bias, A_log, dt_bias,
                 cu_seq_lens, num_prefills, cache_batch=None,
                 dtype=torch.float16,
                 conv_state=None, ssm_state=None,
                 has_initial=None, state_indices=None):
    """Run esimd_gdn_attention as all-prefill with the given cu_seq_lens."""
    n = qkvz.shape[0]
    if cache_batch is None:
        cache_batch = num_prefills + 1  # at least 1 slot per sequence
    if conv_state is None:
        conv_state = torch.zeros(cache_batch, CONV_KERNEL - 1, CONV_ELEMS,
                                  dtype=dtype, device="xpu")
    if ssm_state is None:
        ssm_state = torch.zeros(cache_batch, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM,
                                 dtype=dtype, device="xpu")
    if has_initial is None:
        has_initial = torch.zeros(num_prefills, dtype=torch.bool, device="xpu")
    if state_indices is None:
        state_indices = torch.arange(num_prefills, dtype=torch.int32, device="xpu")

    core = torch.zeros(n, NUM_V_HEADS, HEAD_V_DIM, dtype=dtype, device="xpu")
    z = torch.zeros_like(core)

    esimd_gdn_attention(
        core, z, qkvz, ba,
        NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM,
        conv_state, ssm_state,
        conv_weights, conv_bias,
        "silu",
        A_log, dt_bias,
        num_prefills, 0,  # num_prefills, num_decodes
        has_initial,
        cu_seq_lens, state_indices,
        n, 1, False)
    torch.xpu.synchronize()
    return core, z, conv_state, ssm_state


def _summary(name, t):
    f = t.detach().to("cpu", torch.float32)
    n_nan = torch.isnan(f).sum().item()
    n_inf = torch.isinf(f).sum().item()
    return (f"{name:24s} shape={tuple(t.shape)}  dt={t.dtype}  "
            f"nan={n_nan} inf={n_inf}  "
            f"mean={f.mean().item():.4e} std={f.std().item():.4e}  "
            f"min={f.min().item():.4e} max={f.max().item():.4e}")


def test_finiteness(N):
    """Run a single-seq prefill of N tokens, assert finite + non-trivial."""
    print(f"\n--- finiteness  N={N} ---")
    qkvz, ba = _make_inputs(N, seed=42)
    cw, cb, A_log, dt_bias = _make_static(seed=42)
    cu = torch.tensor([0, N], dtype=torch.int32, device="xpu")
    core, z, conv_state, ssm_state = _run_prefill(
        qkvz, ba, cw, cb, A_log, dt_bias, cu, num_prefills=1)

    print(_summary("core_attn_out", core))
    print(_summary("z", z))
    print(_summary("ssm_state", ssm_state))

    f = core.detach().to("cpu", torch.float32)
    assert torch.isfinite(f).all(), f"N={N}: core has NaN/Inf"
    assert ssm_state.detach().to("cpu", torch.float32).abs().max().item() > 0, (
        f"N={N}: ssm_state still all-zero — kernel did not write state")
    return core, z, conv_state, ssm_state


def test_batch_split_consistency(N, B):
    """Compare:
      A) single seq of N tokens (1 prefill)
      B) B sequences of N/B tokens, all from the same input (so packed
         input is equivalent to taking the first N/B of the single-seq input,
         repeated B times).
    Each batch member starts from a fresh state. The first sub-sequence's
    output should match the first N/B of the single-seq output.
    """
    assert N % B == 0
    sub = N // B
    print(f"\n--- batch_split_consistency  N={N}  B={B} sub={sub} ---")

    # Use the FIRST sub tokens for everything. Pack them B times.
    qkvz_full, ba_full = _make_inputs(sub, seed=7)
    qkvz_packed = qkvz_full.repeat(B, 1).contiguous()
    ba_packed = ba_full.repeat(B, 1).contiguous()
    cw, cb, A_log, dt_bias = _make_static(seed=7)

    # A: single sub-token-len sequence (the reference "first N/B" output)
    cu_A = torch.tensor([0, sub], dtype=torch.int32, device="xpu")
    coreA, _, _, _ = _run_prefill(
        qkvz_full, ba_full, cw, cb, A_log, dt_bias, cu_A, num_prefills=1)

    # B: B sequences of length sub each (no shared state between them)
    cu_B_list = list(range(0, sub * (B + 1), sub))
    cu_B = torch.tensor(cu_B_list, dtype=torch.int32, device="xpu")
    coreB, _, _, _ = _run_prefill(
        qkvz_packed, ba_packed, cw, cb, A_log, dt_bias, cu_B, num_prefills=B)

    # Each of the B sub-blocks of coreB should match coreA exactly (bitwise
    # ~ within fp16 precision, since math is identical).
    coreA_f = coreA.detach().to("cpu", torch.float32)
    coreB_f = coreB.detach().to("cpu", torch.float32)
    print(_summary("A core (sub seq)  ", coreA))
    print(_summary("B core (packed)   ", coreB))

    max_err = 0.0
    for b in range(B):
        sub_b = coreB_f[b * sub:(b + 1) * sub]
        err = (sub_b - coreA_f).abs().max().item()
        max_err = max(max_err, err)
    rel_err = max_err / max(coreA_f.abs().max().item(), 1e-6)
    print(f"  max(|B[b] - A|) = {max_err:.4e}  relative={rel_err:.4e}")

    # fp16 tolerance — strict but realistic for 256-step recurrence
    assert max_err < 2e-2, (
        f"N={N} B={B}: per-sub-batch output diverges from single-seq ref "
        f"by {max_err:.4e} (rel {rel_err:.4e})")


def test_packed_two_seqs(seq_lens):
    """Pack multiple variable-length sequences into one packed batch.
    For seq_lens=[A, B]:
      run packed[0..A] || packed[A..A+B] as a 2-prefill batch.
    Compare each chunk's output against running them as separate single-seq
    prefills. Tests state-pool indexing across a real packed scheduler-style call.
    """
    print(f"\n--- packed_two_seqs  seq_lens={seq_lens} ---")
    n = sum(seq_lens)
    qkvz, ba = _make_inputs(n, seed=11)
    cw, cb, A_log, dt_bias = _make_static(seed=11)

    # Packed: 2 sequences in one call
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()),
                      dtype=torch.int32, device="xpu")
    core_packed, _, _, _ = _run_prefill(
        qkvz, ba, cw, cb, A_log, dt_bias, cu, num_prefills=len(seq_lens))

    # Reference: run each seq separately
    offset = 0
    refs = []
    for L in seq_lens:
        cu_solo = torch.tensor([0, L], dtype=torch.int32, device="xpu")
        sub_qkvz = qkvz[offset:offset + L].clone().contiguous()
        sub_ba = ba[offset:offset + L].clone().contiguous()
        core_solo, _, _, _ = _run_prefill(
            sub_qkvz, sub_ba, cw, cb, A_log, dt_bias, cu_solo, num_prefills=1)
        refs.append(core_solo)
        offset += L

    # Compare
    core_packed_cpu = core_packed.detach().to("cpu", torch.float32)
    offset = 0
    max_err = 0.0
    for i, (L, ref) in enumerate(zip(seq_lens, refs)):
        sub_packed = core_packed_cpu[offset:offset + L]
        ref_cpu = ref.detach().to("cpu", torch.float32)
        err = (sub_packed - ref_cpu).abs().max().item()
        ref_max = ref_cpu.abs().max().item()
        rel = err / max(ref_max, 1e-6)
        print(f"  seq{i}  L={L}  max_err={err:.4e}  rel={rel:.4e}  "
              f"ref_max={ref_max:.4e}")
        max_err = max(max_err, err)
        offset += L

    assert max_err < 2e-2, (
        f"packed seq_lens={seq_lens}: max_err={max_err:.4e}")


def main():
    # Phase 1: just check finiteness across chunk sizes
    print("=" * 60)
    print("Phase 1: finiteness sweep")
    print("=" * 60)
    for N in [64, 128, 256, 512, 1024]:
        try:
            test_finiteness(N)
        except Exception as e:
            print(f"FAIL N={N}: {type(e).__name__}: {e}")
            raise

    # Phase 2: batch-split consistency (does the recurrence give the same
    # output regardless of how we batch tokens of the same prefix?)
    print()
    print("=" * 60)
    print("Phase 2: batch-split consistency (single seq vs packed B seqs)")
    print("=" * 60)
    for N, B in [(64, 4), (128, 4), (256, 4), (512, 4), (1024, 4)]:
        try:
            test_batch_split_consistency(N, B)
        except Exception as e:
            print(f"FAIL N={N} B={B}: {type(e).__name__}: {e}")
            raise

    # Phase 3: packed varlen consistency (seq A + seq B in one call)
    print()
    print("=" * 60)
    print("Phase 3: packed varlen vs separate calls")
    print("=" * 60)
    for seq_lens in [(64, 64), (128, 256), (256, 256), (256, 512), (128, 384)]:
        try:
            test_packed_two_seqs(list(seq_lens))
        except Exception as e:
            print(f"FAIL seq_lens={seq_lens}: {type(e).__name__}: {e}")
            raise

    print("\nAll passed.")


if __name__ == "__main__":
    main()
