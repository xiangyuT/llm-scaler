"""
Scaled Dot-Product Attention (ESIMD Flash Attention)

High-performance Flash Attention using Intel ESIMD doubleGRF kernels.

Example:
    import torch
    from omni_xpu_kernel import sdp

    q = torch.randn(1, 512, 24, 128, device="xpu", dtype=torch.float16)
    k = torch.randn(1, 512, 24, 128, device="xpu", dtype=torch.float16)
    v = torch.randn(1, 512, 24, 128, device="xpu", dtype=torch.float16)
    out = sdp.sdp(q, k, v)
"""

import torch


def _get_native():
    """Get the native sdp module."""
    from .. import _load_extension
    return _load_extension().sdp


def sdp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    ESIMD Flash Attention for Intel XPU.

    Uses a doubleGRF ESIMD kernel with tiled Flash Attention algorithm,
    fusing Q*K^T scaling, softmax, and S*V multiply into a single kernel.

    Args:
        q: Query tensor [B, Lq, H, 128] fp16 or bf16
        k: Key tensor [B, Lk, H, 128] fp16 or bf16
        v: Value tensor [B, Lk, H, 128] fp16 or bf16

    Returns:
        Output tensor [B, Lq, H, 128] same dtype as q

    Note:
        - B must be 1, head_dim must be 128
        - All tensors must be contiguous and on XPU
        - Supports fp16 and bf16
    """
    return _get_native().sdp(q, k, v)


__all__ = ["sdp"]
