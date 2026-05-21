"""Correctness UT for the prefill-path moe_forward_full_int4 (GGML layout).

Uses the *same* moe_forward_full_int4 entry but with n_tokens >= 16 so the
new XMX-DPAS prefill kernel fires (gated on VLLM_MOE_PREFILL_MIN_TOKENS).
Computes a CPU reference (pure PyTorch) and compares cosine similarity.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "xpu"
GROUP_SIZE = 128
PACK_FACTOR = 8


def quantize_int4(weight_fp16: torch.Tensor, group_size: int = GROUP_SIZE):
    """Quantize FP16 [N, K] -> int32 packed nibbles + fp16 scales."""
    N, K = weight_fp16.shape
    assert K % group_size == 0
    n_groups = K // group_size

    w = weight_fp16.float().numpy().reshape(N, n_groups, group_size)
    max_abs = np.abs(w).max(axis=2)
    scale_np = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float16)

    scale_e = scale_np[:, :, None].astype(np.float32)
    q = np.round(w / scale_e).clip(-8, 7).astype(np.int32) + 8
    q_flat = q.reshape(N, K)

    packed = np.zeros((N, K // PACK_FACTOR), dtype=np.uint32)
    q_grouped = q_flat.reshape(N, K // PACK_FACTOR, PACK_FACTOR).astype(np.uint32)
    for b in range(PACK_FACTOR):
        packed |= (q_grouped[:, :, b] & 0xF) << (b * 4)
    return torch.from_numpy(packed.view(np.int32)), torch.from_numpy(scale_np)


def dequantize_int4(qweight, scales, N, K, group_size=GROUP_SIZE):
    qw = qweight.numpy().view(np.uint32)
    sc = scales.numpy().astype(np.float32)
    n_groups = K // group_size

    out = np.zeros((N, K), dtype=np.float32)
    for b in range(PACK_FACTOR):
        out[:, b::PACK_FACTOR] = ((qw >> (b * 4)) & 0xF).astype(np.float32) - 8.0
    out = out.reshape(N, n_groups, group_size) * sc[:, :, None]
    return torch.from_numpy(out.reshape(N, K).astype(np.float16))


def ref_moe_forward(x, logits, w13_dq, shared_gate_up, w2_dq, shared_down,
                    shared_gate_w, top_k):
    n_tokens, hidden_size = x.shape
    intermediate_size = w2_dq.shape[2]
    shared_inter = shared_gate_up.shape[0] // 2
    x_f = x.float()

    probs = F.softmax(logits.float(), dim=-1)
    topk_weight, topk_idx = torch.topk(probs, top_k, dim=-1)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    topk_weight = topk_weight.half()

    final = torch.zeros(n_tokens, hidden_size, dtype=torch.float32)
    for t in range(n_tokens):
        routed_sum = torch.zeros(hidden_size, dtype=torch.float32)
        for ki in range(top_k):
            eid = topk_idx[t, ki].item()
            rw = topk_weight[t, ki].float()
            gw = w13_dq[eid, :intermediate_size, :].float()
            uw = w13_dq[eid, intermediate_size:, :].float()
            g_out = x_f[t] @ gw.t()
            u_out = x_f[t] @ uw.t()
            inter = (g_out / (1 + torch.exp(-g_out))) * u_out
            dw = w2_dq[eid].float()
            routed_sum += rw * (inter @ dw.t())

        sgu = shared_gate_up.float()
        sg = x_f[t] @ sgu[:shared_inter, :].t()
        su = x_f[t] @ sgu[shared_inter:, :].t()
        sinter = (sg / (1 + torch.exp(-sg))) * su
        sdown = sinter @ shared_down.float().t()
        gate_sigmoid = torch.sigmoid(x_f[t] @ shared_gate_w.float().t()).item()
        final[t] = routed_sum + sdown * gate_sigmoid
    return final.half()


def cos_sim(a, b):
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    return (a_f @ b_f / (a_f.norm() * b_f.norm() + 1e-9)).item()


def main():
    from custom_esimd_kernels_vllm import moe_int4_ops

    # Force prefill path
    os.environ["VLLM_MOE_PREFILL_MIN_TOKENS"] = "1"

    cfg = os.environ.get("CFG", "small")
    if cfg == "35b":
        # Real Qwen3.5-35B-A3B per-rank shape (TP=2 → intermediate_size=256)
        H, D, D_S, E, TK = 2048, 256, 256, 256, 8
    else:
        H, D, D_S, E, TK = 256, 128, 128, 16, 4
    NUM_SHARED = 1

    print(f"=== test_moe_prefill_int4 (H={H} D={D} E={E} TK={TK}) ===")
    for n_tokens in [16, 32, 48]:
        torch.manual_seed(42)
        w13 = (torch.randn(E, 2 * D, H) * 0.02).half()
        w2 = (torch.randn(E, H, D) * 0.02).half()
        sgu = (torch.randn(2 * D_S, H) * 0.02).half()
        sdw = (torch.randn(H, D_S) * 0.02).half()
        sgw = (torch.randn(1, H) * 0.02).half()
        x = (torch.randn(n_tokens, H) * 0.1).half()
        logits = (torch.randn(n_tokens, E) * 0.1).half()

        w13_qw_list, w13_sc_list, w2_qw_list, w2_sc_list = [], [], [], []
        for e in range(E):
            qw, sc = quantize_int4(w13[e])
            w13_qw_list.append(qw); w13_sc_list.append(sc)
            qw, sc = quantize_int4(w2[e])
            w2_qw_list.append(qw); w2_sc_list.append(sc)
        w13_qw = torch.stack(w13_qw_list)
        w13_sc = torch.stack(w13_sc_list)
        w2_qw = torch.stack(w2_qw_list)
        w2_sc = torch.stack(w2_sc_list)

        w13_dq = torch.stack([dequantize_int4(w13_qw[e], w13_sc[e], 2 * D, H)
                              for e in range(E)])
        w2_dq = torch.stack([dequantize_int4(w2_qw[e], w2_sc[e], H, D)
                             for e in range(E)])

        # Reference on CPU
        ref = ref_moe_forward(x, logits, w13_dq, sgu, w2_dq, sdw, sgw, TK)

        _dummy = torch.empty(0, device=DEVICE, dtype=torch.float16)
        out = moe_int4_ops.moe_forward_full_int4(
            x.to(DEVICE), logits.to(DEVICE),
            w13_qw.to(DEVICE), w13_sc.to(DEVICE),
            sgu.to(DEVICE), _dummy,
            w2_qw.to(DEVICE), w2_sc.to(DEVICE),
            sdw.to(DEVICE), _dummy,
            sgw.to(DEVICE),
            TK, NUM_SHARED, E, True).cpu()

        cos = cos_sim(ref, out)
        mae = (ref.float() - out.float()).abs().max().item()
        ok = cos > 0.95 and mae < 1.0
        status = "PASS" if ok else "FAIL"
        print(f"  n_tokens={n_tokens:>3d}  cos={cos:.6f}  mae={mae:.4f}  [{status}]")
        if not ok:
            print(f"    ref[:5]={ref[0,:5].tolist()}")
            print(f"    out[:5]={out[0,:5].tolist()}")
            sys.exit(1)
    print("All prefill INT4 GGML tests passed.")


if __name__ == "__main__":
    main()
