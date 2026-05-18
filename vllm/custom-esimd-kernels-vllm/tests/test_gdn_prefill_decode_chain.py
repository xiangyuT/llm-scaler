"""Chain UT: prefill (esimd_gdn_attention) → decode step 1 → decode step 2.

Mirrors what vllm does on Qwen3.5-0.8B per request:
  1. Prefill: build ssm_state in slot[s].
  2. Decode N: read ssm_state[s], advance recurrence, write back.

The vllm-side smoke test on PTL produces ' Paris !!!!!!' — first decode step
correct, second onward NaN. This UT isolates whether the bug is in
esimd_gdn_conv_fused_seq itself, in the cross-kernel ssm_state hand-off,
or somewhere upstream in vllm.
"""
import torch

from custom_esimd_kernels_vllm import (
    esimd_gdn_attention,
    esimd_gdn_conv_fused_seq,
)


# Qwen3.5-0.8B GDN params:
NUM_K_HEADS = 16
NUM_V_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4


def _build_state_pool(cache_batch=2, dtype=torch.float16, device="xpu"):
    """Allocate the GDN state pool that both kernels share."""
    qkvz_dim = (
        NUM_K_HEADS * HEAD_K_DIM
        + NUM_K_HEADS * HEAD_K_DIM
        + NUM_V_HEADS * HEAD_V_DIM
        + NUM_V_HEADS * HEAD_V_DIM
    )
    conv_elems = NUM_K_HEADS * HEAD_K_DIM * 2 + NUM_V_HEADS * HEAD_V_DIM
    conv_state = torch.zeros(
        cache_batch, CONV_KERNEL - 1, conv_elems,
        dtype=dtype, device=device,
    ).contiguous()
    ssm_state = torch.zeros(
        cache_batch, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM,
        dtype=dtype, device=device,
    ).contiguous()
    return conv_state, ssm_state, qkvz_dim, conv_elems


def _build_weights(conv_elems, dtype=torch.float16, device="xpu"):
    conv_weights = (
        torch.randn(conv_elems, CONV_KERNEL, dtype=torch.float32, device=device) * 0.05
    ).to(dtype).contiguous()
    conv_bias = torch.zeros(conv_elems, dtype=dtype, device=device).contiguous()
    A_log = torch.linspace(0.0, 2.0, NUM_V_HEADS, dtype=torch.float32, device=device).to(dtype)
    dt_bias = torch.zeros(NUM_V_HEADS, dtype=dtype, device=device)
    return conv_weights, conv_bias, A_log, dt_bias


def _run_prefill(qkvz_prefill, ba_prefill, conv_state, ssm_state,
                 conv_weights, conv_bias, A_log, dt_bias, slot_idx,
                 prefill_tokens):
    device = qkvz_prefill.device
    core_attn_out = torch.zeros(
        prefill_tokens, NUM_V_HEADS, HEAD_V_DIM,
        device=device, dtype=qkvz_prefill.dtype,
    ).contiguous()
    z_out = torch.zeros_like(core_attn_out).contiguous()

    cu = torch.tensor([0, prefill_tokens], dtype=torch.int32, device=device)
    has_init = torch.tensor([False], dtype=torch.bool, device=device)
    state_idx = torch.tensor([slot_idx], dtype=torch.int32, device=device)

    esimd_gdn_attention(
        core_attn_out, z_out,
        qkvz_prefill, ba_prefill,
        NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM,
        conv_state, ssm_state,
        conv_weights, conv_bias,
        "silu", A_log, dt_bias,
        1,  # num_prefills
        0,  # num_decodes
        has_init,
        cu, state_idx,
        prefill_tokens,
        1,  # tp_size
        False,  # reorder_input
    )
    torch.xpu.synchronize()
    return core_attn_out, z_out


def _run_decode_step(qkvz_dec, ba_dec, conv_state, ssm_state,
                     conv_weights, conv_bias, A_log, dt_bias, slot_idx,
                     scale=None):
    device = qkvz_dec.device
    core_attn_out = torch.zeros(
        1, NUM_V_HEADS, HEAD_V_DIM,
        device=device, dtype=qkvz_dec.dtype,
    ).contiguous()
    z_out = torch.zeros_like(core_attn_out).contiguous()
    state_idx = torch.tensor([slot_idx], dtype=torch.int32, device=device)
    if scale is None:
        scale = HEAD_K_DIM ** -0.5

    esimd_gdn_conv_fused_seq(
        qkvz_dec, conv_state, conv_weights, conv_bias,
        state_idx,                       # conv_state_indices
        A_log, dt_bias,
        ba_dec,
        ssm_state, state_idx,            # ssm_state_indices
        core_attn_out, z_out,
        1,  # N (decode batch)
        NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM,
        scale,
    )
    torch.xpu.synchronize()
    return core_attn_out, z_out


def _summary(name, t):
    t_cpu = t.detach().to("cpu", torch.float32)
    nan_count = int(torch.isnan(t_cpu).sum())
    inf_count = int(torch.isinf(t_cpu).sum())
    if nan_count == 0 and inf_count == 0:
        return f"{name}: max_abs={t_cpu.abs().max().item():.4g} mean_abs={t_cpu.abs().mean().item():.4g}"
    return f"{name}: NaN={nan_count} Inf={inf_count}"


def test_chain_5_prefill_3_decode():
    torch.manual_seed(0)
    device = "xpu"
    dtype = torch.float16
    slot = 0

    conv_state, ssm_state, qkvz_dim, conv_elems = _build_state_pool(
        cache_batch=2, dtype=dtype, device=device,
    )
    conv_weights, conv_bias, A_log, dt_bias = _build_weights(
        conv_elems, dtype=dtype, device=device,
    )

    # Prefill input
    P = 5
    qkvz_p = (torch.randn(P, qkvz_dim, dtype=torch.float32, device=device) * 0.05).to(dtype)
    ba_p = (torch.randn(P, 2 * NUM_V_HEADS, dtype=torch.float32, device=device) * 0.05).to(dtype)

    # NOTE: the prefill kernel expects GQA-interleaved qkvz, decode expects
    # sequential. For a self-test using the same input we don't care about
    # the absolute correctness, only that ssm_state propagates without
    # corruption between calls.
    _run_prefill(
        qkvz_p, ba_p, conv_state, ssm_state, conv_weights, conv_bias,
        A_log, dt_bias, slot, P,
    )
    print(_summary("after_prefill ssm_state", ssm_state))
    print(_summary("after_prefill conv_state", conv_state))
    assert torch.isfinite(ssm_state.float()).all(), "prefill produced non-finite ssm_state"

    # Decode N steps in a row, checking state stays finite each step.
    for step in range(3):
        qkvz_d = (torch.randn(1, qkvz_dim, dtype=torch.float32, device=device) * 0.05).to(dtype)
        ba_d = (torch.randn(1, 2 * NUM_V_HEADS, dtype=torch.float32, device=device) * 0.05).to(dtype)
        out, z = _run_decode_step(
            qkvz_d, ba_d, conv_state, ssm_state,
            conv_weights, conv_bias, A_log, dt_bias, slot,
        )
        print(_summary(f"decode[{step}] core_attn_out", out))
        print(_summary(f"decode[{step}] ssm_state", ssm_state))
        print(_summary(f"decode[{step}] conv_state", conv_state))
        assert torch.isfinite(out.float()).all(), \
            f"decode step {step} produced non-finite output"
        assert torch.isfinite(ssm_state.float()).all(), \
            f"decode step {step} corrupted ssm_state"


def test_chain_decode_only_zero_init():
    """Decode without prior prefill — `has_initial_state=False` semantics
    in vllm are emulated here by ssm_state == 0 and conv_state == 0.
    """
    torch.manual_seed(0)
    device = "xpu"
    dtype = torch.float16
    slot = 0

    conv_state, ssm_state, qkvz_dim, conv_elems = _build_state_pool(
        cache_batch=2, dtype=dtype, device=device,
    )
    conv_weights, conv_bias, A_log, dt_bias = _build_weights(
        conv_elems, dtype=dtype, device=device,
    )
    for step in range(3):
        qkvz_d = (torch.randn(1, qkvz_dim, dtype=torch.float32, device=device) * 0.05).to(dtype)
        ba_d = (torch.randn(1, 2 * NUM_V_HEADS, dtype=torch.float32, device=device) * 0.05).to(dtype)
        out, _ = _run_decode_step(
            qkvz_d, ba_d, conv_state, ssm_state,
            conv_weights, conv_bias, A_log, dt_bias, slot,
        )
        print(_summary(f"zero-init decode[{step}] out", out))
        print(_summary(f"zero-init decode[{step}] ssm_state", ssm_state))
        assert torch.isfinite(out.float()).all()
        assert torch.isfinite(ssm_state.float()).all()


if __name__ == "__main__":
    for fn in [test_chain_5_prefill_3_decode, test_chain_decode_only_zero_init]:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
