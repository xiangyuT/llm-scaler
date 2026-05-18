"""Smoke UT for esimd_gdn_attention on PTL Xe3.

We don't reimplement the full GDN math on CPU — the kernel is too involved
(causal_conv1d split + gated delta rule recurrence) and doing so risks
shipping a subtly-wrong reference. Instead this UT validates:

  1. The kernel launches without DEVICE_LOST under shapes Qwen3.5-0.8B uses.
  2. Outputs (core_attn_out, z, conv_state, ssm_state) are finite.
  3. ssm_state is not all-zero after a non-trivial prefill.
  4. Repeated runs (mimicking decode after prefill) stay finite.

Numerical correctness is then validated end-to-end by comparing vllm
serve output against the SGL reference for the same prompt.
"""
import torch
import pytest

from custom_esimd_kernels_vllm import esimd_gdn_attention


# Qwen3.5-0.8B GDN params (from config.json):
NUM_K_HEADS = 16
NUM_V_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4


def _build_inputs(num_prefill_tokens, num_decodes,
                  cache_batch=4, dtype=torch.float16):
    """Construct kernel inputs for a mixed prefill+decode batch."""
    device = "xpu"

    # Layout: prefills first, then decodes (1 token each).
    # In vllm dispatch, num_actual_tokens = num_prefill_tokens + num_decodes.
    num_prefills = 1 if num_prefill_tokens > 0 else 0
    num_actual_tokens = num_prefill_tokens + num_decodes

    # qkvz_dim: Q (n_k * hk) + K (n_k * hk) + V (n_v * hv) + Z (n_v * hv).
    # All num_k_heads == num_v_heads here, so it's:
    qkvz_dim = (NUM_K_HEADS * HEAD_K_DIM
                + NUM_K_HEADS * HEAD_K_DIM
                + NUM_V_HEADS * HEAD_V_DIM
                + NUM_V_HEADS * HEAD_V_DIM)
    ba_dim = 2 * NUM_V_HEADS  # b + a per v-head

    # conv1d operates over (Q | K | V) which is conv_elems = (n_k*hk)*2 + n_v*hv
    conv_elems = (NUM_K_HEADS * HEAD_K_DIM * 2 + NUM_V_HEADS * HEAD_V_DIM)

    projected_states_qkvz = (
        torch.randn(num_actual_tokens, qkvz_dim, device=device, dtype=torch.float32) * 0.05
    ).to(dtype).contiguous()
    projected_states_ba = (
        torch.randn(num_actual_tokens, ba_dim, device=device, dtype=torch.float32) * 0.05
    ).to(dtype).contiguous()

    core_attn_out = torch.zeros(
        num_actual_tokens, NUM_V_HEADS, HEAD_V_DIM,
        device=device, dtype=dtype,
    ).contiguous()
    z = torch.zeros_like(core_attn_out).contiguous()

    # conv_state: [cache_batch, kernel-1, conv_elems]
    conv_state = torch.zeros(
        cache_batch, CONV_KERNEL - 1, conv_elems,
        device=device, dtype=dtype,
    ).contiguous()
    # ssm_state: [cache_batch, num_v_heads, head_v_dim, head_k_dim] (H, V, K)
    # Kernel reinterpret_casts as scalar_t — must match model dtype.
    ssm_state = torch.zeros(
        cache_batch, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM,
        device=device, dtype=dtype,
    ).contiguous()

    conv_weights = torch.randn(
        conv_elems, CONV_KERNEL, device=device, dtype=dtype,
    ).contiguous() * 0.1
    conv_bias = torch.zeros(conv_elems, device=device, dtype=dtype).contiguous()

    # A_log / dt_bias dtype must match the model dtype — the kernel
    # reinterpret_casts their data_ptr() as scalar_t (fp16/bf16).
    A_log = (torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32) * 0.1).to(dtype)
    dt_bias = (torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32) * 0.1).to(dtype)

    # Batch breakdown
    B = num_prefills + num_decodes
    cu_lens = [0]
    if num_prefills:
        cu_lens.append(num_prefill_tokens)
    for _ in range(num_decodes):
        cu_lens.append(cu_lens[-1] + 1)
    non_spec_query_start_loc = torch.tensor(
        cu_lens, dtype=torch.int32, device=device)

    # Slot indices into the cache for each sequence in the batch.
    non_spec_state_indices_tensor = torch.arange(
        B, dtype=torch.int32, device=device)

    has_initial_state = torch.zeros(B, dtype=torch.bool, device=device)

    return dict(
        core_attn_out=core_attn_out,
        z=z,
        projected_states_qkvz=projected_states_qkvz,
        projected_states_ba=projected_states_ba,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens,
    )


def _run_kernel(inp):
    esimd_gdn_attention(
        inp["core_attn_out"],
        inp["z"],
        inp["projected_states_qkvz"],
        inp["projected_states_ba"],
        NUM_K_HEADS,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
        inp["conv_state"],
        inp["ssm_state"],
        inp["conv_weights"],
        inp["conv_bias"],
        "silu",
        inp["A_log"],
        inp["dt_bias"],
        inp["num_prefills"],
        inp["num_decodes"],
        inp["has_initial_state"],
        inp["non_spec_query_start_loc"],
        inp["non_spec_state_indices_tensor"],
        inp["num_actual_tokens"],
        1,      # tp_size
        False,  # reorder_input
    )
    torch.xpu.synchronize()  # surface async DEVICE_LOST/OOB here


def _assert_finite(name, t):
    t_cpu = t.detach().to("cpu", torch.float32)
    n_nan = torch.isnan(t_cpu).sum().item()
    n_inf = torch.isinf(t_cpu).sum().item()
    assert n_nan == 0, f"{name}: {n_nan} NaN values"
    assert n_inf == 0, f"{name}: {n_inf} Inf values"


def test_prefill_only():
    inp = _build_inputs(num_prefill_tokens=8, num_decodes=0)
    _run_kernel(inp)
    _assert_finite("core_attn_out", inp["core_attn_out"])
    _assert_finite("z", inp["z"])
    _assert_finite("conv_state", inp["conv_state"])
    _assert_finite("ssm_state", inp["ssm_state"])
    # ssm_state should have been written.
    assert inp["ssm_state"].abs().max().item() > 0.0, (
        "ssm_state still all-zero after prefill — kernel did not update state")


def test_decode_only_with_initial_state():
    inp = _build_inputs(num_prefill_tokens=0, num_decodes=2)
    # Seed initial state to non-zero so decode has something to read.
    inp["ssm_state"].normal_(mean=0.0, std=0.05)
    inp["conv_state"].normal_(mean=0.0, std=0.05)
    inp["has_initial_state"].fill_(True)
    _run_kernel(inp)
    _assert_finite("core_attn_out", inp["core_attn_out"])
    _assert_finite("z", inp["z"])
    _assert_finite("conv_state", inp["conv_state"])
    _assert_finite("ssm_state", inp["ssm_state"])


def test_mixed_prefill_decode():
    inp = _build_inputs(num_prefill_tokens=16, num_decodes=2)
    _run_kernel(inp)
    _assert_finite("core_attn_out", inp["core_attn_out"])
    _assert_finite("z", inp["z"])
    _assert_finite("conv_state", inp["conv_state"])
    _assert_finite("ssm_state", inp["ssm_state"])


def test_prefill_then_decode_chain():
    """Run a prefill, then a decode that consumes the resulting state."""
    inp1 = _build_inputs(num_prefill_tokens=12, num_decodes=0)
    _run_kernel(inp1)
    _assert_finite("ssm_state after prefill", inp1["ssm_state"])

    # Build a decode that re-uses the same conv_state/ssm_state slot[0].
    inp2 = _build_inputs(num_prefill_tokens=0, num_decodes=1)
    inp2["conv_state"] = inp1["conv_state"]
    inp2["ssm_state"] = inp1["ssm_state"]
    inp2["has_initial_state"].fill_(True)
    inp2["non_spec_state_indices_tensor"] = torch.tensor(
        [0], dtype=torch.int32, device="xpu")
    _run_kernel(inp2)
    _assert_finite("core_attn_out (decode)", inp2["core_attn_out"])
    _assert_finite("ssm_state after decode", inp2["ssm_state"])


def test_profile_run_shapes():
    """Mirror what vllm profile_run dispatches: max_num_batched_tokens=2048
    all-prefill from a single sequence."""
    inp = _build_inputs(num_prefill_tokens=2048, num_decodes=0, cache_batch=2)
    _run_kernel(inp)
    _assert_finite("core_attn_out", inp["core_attn_out"])
    _assert_finite("z", inp["z"])
    _assert_finite("conv_state", inp["conv_state"])
    _assert_finite("ssm_state", inp["ssm_state"])


if __name__ == "__main__":
    for fn in [
        test_prefill_only,
        test_decode_only_with_initial_state,
        test_mixed_prefill_decode,
        test_prefill_then_decode_chain,
        test_profile_run_shapes,
    ]:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
