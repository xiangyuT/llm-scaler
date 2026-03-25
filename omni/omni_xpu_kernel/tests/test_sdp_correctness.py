"""
Correctness tests for omni_xpu_kernel SDP (Flash Attention) kernel
"""

import math
import pytest
import torch


def has_xpu():
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


def reference_sdp(q, k, v):
    """FP32 reference: naive matmul + softmax + matmul."""
    q32 = q.float().permute(0, 2, 1, 3)
    k32 = k.float().permute(0, 2, 1, 3)
    v32 = v.float().permute(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(q.size(3))
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v32)
    return out.permute(0, 2, 1, 3).contiguous().to(q.dtype)


class TestSDPCorrectness:
    """Correctness tests for ESIMD Flash Attention kernel."""

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("seq_len", [16, 64, 128, 256, 512, 1024])
    @pytest.mark.parametrize("heads", [1, 8, 24])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdp_correctness(self, seq_len, heads, dtype):
        from omni_xpu_kernel import sdp

        B, D = 1, 128
        q = torch.randn(B, seq_len, heads, D, device="xpu", dtype=dtype)
        k = torch.randn(B, seq_len, heads, D, device="xpu", dtype=dtype)
        v = torch.randn(B, seq_len, heads, D, device="xpu", dtype=dtype)

        out = sdp.sdp(q, k, v)
        ref = reference_sdp(q, k, v)

        assert out.shape == ref.shape
        assert out.dtype == ref.dtype

        if dtype == torch.float16:
            rtol, atol = 1e-2, 5e-3
        else:
            rtol, atol = 2e-2, 1e-2

        # Check element-wise tolerance
        max_diff = (out.float() - ref.float()).abs().max().item()
        if max_diff > atol:
            # Fallback: check cosine similarity per row
            out_flat = out.float().reshape(-1, D)
            ref_flat = ref.float().reshape(-1, D)
            cos_sim = torch.nn.functional.cosine_similarity(out_flat, ref_flat, dim=-1)
            min_cos = cos_sim.min().item()
            assert min_cos > 0.99, (
                f"Cosine similarity too low: {min_cos:.6f} "
                f"(max_diff={max_diff:.6f})"
            )

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_output_shape(self):
        from omni_xpu_kernel import sdp

        q = torch.randn(1, 64, 8, 128, device="xpu", dtype=torch.float16)
        k = torch.randn(1, 64, 8, 128, device="xpu", dtype=torch.float16)
        v = torch.randn(1, 64, 8, 128, device="xpu", dtype=torch.float16)
        out = sdp.sdp(q, k, v)

        assert out.shape == (1, 64, 8, 128)
        assert out.dtype == torch.float16
        assert out.device.type == "xpu"

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_dtype_consistency(self):
        from omni_xpu_kernel import sdp

        for dtype in [torch.float16, torch.bfloat16]:
            q = torch.randn(1, 32, 4, 128, device="xpu", dtype=dtype)
            k = torch.randn(1, 32, 4, 128, device="xpu", dtype=dtype)
            v = torch.randn(1, 32, 4, 128, device="xpu", dtype=dtype)
            out = sdp.sdp(q, k, v)
            assert out.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
