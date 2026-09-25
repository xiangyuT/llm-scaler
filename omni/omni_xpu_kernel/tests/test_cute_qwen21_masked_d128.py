"""Focused B70 sidecar gate; Luna runs it only after an exact AOT build."""

import json
import hashlib
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


MATRIX = json.loads((Path(__file__).parents[1] / "omni_xpu_kernel/cute/experimental/qwen21_masked/matrix.json").read_text())
REAL_ROOT = os.environ.get("QWEN21_MASKED_REAL_ROOT")
_DENSE_BEFORE_NEW_DSO = None
_EXPERIMENT_REQUESTED = any(key in os.environ for key in (
    "QWEN21_MASKED_D128_DSO", "QWEN21_MASKED_D128_DSO_SHA256"))
pytestmark = pytest.mark.skipif(
    not _EXPERIMENT_REQUESTED,
    reason="experimental masked D128 sidecar requires explicit DSO selection",
)


@pytest.fixture(scope="module")
def masked_ops():
    global _DENSE_BEFORE_NEW_DSO
    path = os.environ.get("QWEN21_MASKED_D128_DSO")
    expected_sha = os.environ.get("QWEN21_MASKED_D128_DSO_SHA256")
    if not path or not Path(path).is_file():
        raise RuntimeError("QWEN21_MASKED_D128_DSO must name Luna's exact new sidecar")
    if not expected_sha or len(expected_sha) != 64 or any(
            digit not in "0123456789abcdef" for digit in expected_sha):
        raise RuntimeError("QWEN21_MASKED_D128_DSO_SHA256 must be the exact lowercase DSO digest")
    if not REAL_ROOT or not Path(REAL_ROOT).is_dir():
        raise RuntimeError("QWEN21_MASKED_REAL_ROOT must contain the required captures")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise RuntimeError("experimental masked D128 DSO identity mismatch")
    from omni_xpu_kernel import cute
    q, k, v = qkv(31, 65)
    dense_before = cute.sdp_bhld_d128(q, k, v).clone()
    assert dense_before.shape == q.shape
    _DENSE_BEFORE_NEW_DSO = (q, k, v, dense_before)
    # The installed dense DSO is now loaded before the new namespace in this
    # same process; isolated one-DSO success cannot hide C++ symbol interpose.
    torch.ops.load_library(path)
    torch.set_float32_matmul_precision("highest")
    assert torch.get_float32_matmul_precision() == "highest"
    return torch.ops.qwen21_masked_d128


def qkv(q_len, kv_len, seed=20260925):
    torch.manual_seed(seed)
    def value(length):
        return torch.randn((1, length, 32, 128), device="xpu", dtype=torch.bfloat16).permute(0, 2, 1, 3)
    return value(q_len), value(kv_len), value(kv_len)


def qkv_mixed_offset(q_len, kv_len, q_layout):
    """Legal active strides with aligned nonzero offsets and mixed K/V layout."""
    torch.manual_seed(20260926)

    def packed_offset(length):
        offset = 8  # BF16 elements: 16-byte-aligned active data pointer.
        storage = torch.empty(offset + 32 * length * 128,
                              device="xpu", dtype=torch.bfloat16)
        view = storage.as_strided((1, 32, length, 128),
                                  (32 * length * 128, length * 128, 128, 1),
                                  offset)
        view.copy_(torch.randn(view.shape, device="xpu", dtype=view.dtype))
        assert view.storage_offset() == offset
        return view

    def blhd_offset(length):
        backing = torch.randn((1, length + 2, 32, 128),
                              device="xpu", dtype=torch.bfloat16)
        view = backing[:, 1:1 + length].permute(0, 2, 1, 3)
        assert view.storage_offset() == 4096
        return view

    q = packed_offset(q_len) if q_layout == "packed" else blhd_offset(q_len)
    return q, packed_offset(kv_len), blhd_offset(kv_len)


def prefix_mask(q_len, kv_len):
    prefix = max(0, kv_len - q_len)
    rows = torch.arange(q_len, device="xpu")[:, None]
    cols = torch.arange(kv_len, device="xpu")[None, :]
    return (cols <= prefix + rows).contiguous()


def oracle(q, k, v, mask):
    out = torch.empty(q.shape, device="xpu", dtype=torch.float32)
    for head in range(0, 32, 4):
        kk = k[:, head:head + 4].float().transpose(-1, -2)
        vv = v[:, head:head + 4].float()
        for row in range(0, q.shape[2], 128):
            visible = mask[row:row + 128]
            scores = q[:, head:head + 4, row:row + 128].float() @ kk
            scores *= 128 ** -0.5
            scores.masked_fill_(~visible, float("-inf"))
            probabilities = scores.softmax(-1)
            probabilities = torch.where(
                visible.any(-1)[None, None, :, None], probabilities,
                torch.zeros_like(probabilities),
            )
            out[:, head:head + 4, row:row + 128] = probabilities @ vv
    return out


def assert_screen(actual, reference):
    delta = actual.float() - reference
    relative_rms = float(delta.square().mean().sqrt() /
                         reference.square().mean().sqrt().clamp_min(1e-12))
    maximum = float(delta.abs().max())
    limit = 0.02 * max(1.0, float(reference.abs().max()))
    assert actual.shape == reference.shape and bool(torch.isfinite(actual).all())
    assert relative_rms <= 0.01 and maximum <= limit


@pytest.mark.parametrize("name", ["sdp", "sdp_split4"])
@pytest.mark.parametrize("geometry", MATRIX["geometries"], ids=lambda row: row["label"])
def test_seven_actual_masked_geometry_classes(masked_ops, name, geometry):
    q, k, v = qkv(geometry["q"], geometry["kv"])
    mask = prefix_mask(geometry["q"], geometry["kv"])
    expected = oracle(q, k, v, mask)
    assert_screen(F.scaled_dot_product_attention(q, k, v, attn_mask=mask), expected)
    actual = getattr(masked_ops, name)(q, k, v, mask)
    assert_screen(actual, expected)
    assert actual.data_ptr() not in {q.data_ptr(), k.data_ptr(), v.data_ptr()}
    assert actual.stride() == q.stride()


@pytest.mark.parametrize("name", ["sdp", "sdp_split4"])
@pytest.mark.parametrize("q_layout", ["packed", "blhd"])
def test_mixed_qkv_layout_offsets_and_input_immutability(masked_ops, name, q_layout):
    q, k, v = qkv_mixed_offset(9, 65, q_layout)
    mask = prefix_mask(9, 65)
    mask[:, ::7] = False
    inputs = (q, k, v, mask)
    before = tuple(t.clone() for t in inputs)
    expected = oracle(q, k, v, mask)
    assert_screen(F.scaled_dot_product_attention(q, k, v, attn_mask=mask), expected)
    actual = getattr(masked_ops, name)(q, k, v, mask)
    assert_screen(actual, expected)
    expected_stride = ((32 * 9 * 128, 9 * 128, 128, 1) if q_layout == "packed"
                       else (9 * 4096, 128, 4096, 1))
    assert actual.stride() == expected_stride
    assert actual.storage_offset() == 0
    assert actual.data_ptr() not in {q.data_ptr(), k.data_ptr(), v.data_ptr()}
    for tensor, original in zip(inputs, before):
        assert torch.equal(tensor, original)


@pytest.mark.parametrize("name", ["sdp", "sdp_split4"])
@pytest.mark.parametrize("q_len,kv_len", [(1, 1), (7, 31), (9, 65), (17, 1025), (31, 4097)])
def test_actual_mask_bits_holes_and_tails(masked_ops, name, q_len, kv_len):
    q, k, v = qkv(q_len, kv_len)
    mask = prefix_mask(q_len, kv_len)
    mask[:, ::7] = False       # same shape, not a triangular/prefix mask
    if q_len > 1:
        mask[0] = False        # no visible K: output must be zero
        mask[-1] = True        # all visible K, regardless of Q/KV lengths
    expected = oracle(q, k, v, mask)
    assert_screen(F.scaled_dot_product_attention(q, k, v, attn_mask=mask), expected)
    actual = getattr(masked_ops, name)(q, k, v, mask)
    assert_screen(actual, expected)
    if q_len > 1:
        assert torch.count_nonzero(actual[:, :, 0]) == 0


@pytest.mark.parametrize("name", ["sdp", "sdp_split4"])
def test_exact_zero_only_for_balanced_visible_pairs(masked_ops, name):
    q, k, v = qkv(8, 64)
    q.zero_()                  # uniform logits over every visible key
    paired = torch.randn_like(v[:, :, ::2])
    v[:, :, ::2] = paired
    v[:, :, 1::2] = -paired
    mask = torch.ones((8, 64), device="xpu", dtype=torch.bool)
    actual = getattr(masked_ops, name)(q, k, v, mask)
    assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("name", ["sdp", "sdp_split4"])
def test_peaked_and_irregular_cancellation_use_fp32_oracle(masked_ops, name):
    q, k, v = qkv(17, 1025)
    mask = prefix_mask(17, 1025)
    mask[:, 1::9] = False  # Unpaired visible keys: exact zero is not asserted.
    v[:, :, 1::2] = -v[:, :, 1::2]
    for cq, ck in ((q * 4, k * 4), (torch.zeros_like(q), k)):
        expected = oracle(cq, ck, v, mask)
        assert_screen(getattr(masked_ops, name)(cq, ck, v, mask), expected)


@pytest.mark.parametrize("name", ["sdp", "sdp_split4"])
def test_zero_v_and_globally_invisible_kv_do_not_affect_output(masked_ops, name):
    q, k, v = qkv(9, 65)
    mask = prefix_mask(9, 65)
    mask[:, ::4] = False  # Every row excludes these columns, regardless of prefix.
    op = getattr(masked_ops, name)

    zero_v = torch.zeros_like(v)
    assert torch.count_nonzero(op(q, k, zero_v, mask)) == 0
    assert torch.count_nonzero(op(q, k, zero_v, torch.zeros_like(mask))) == 0

    expected = oracle(q, k, v, mask)
    before = op(q, k, v, mask)
    assert_screen(before, expected)
    altered_k, altered_v = k.clone(), v.clone()
    altered_k[:, :, ::4] += 32
    altered_v[:, :, ::4] -= 16
    assert_screen(oracle(q, altered_k, altered_v, mask), expected)
    after = op(q, altered_k, altered_v, mask)
    assert_screen(after, expected)
    assert torch.equal(after, before)


@pytest.mark.parametrize("name", ["sdp", "sdp_split4"])
def test_reject_unsupported_mask_contract(masked_ops, name):
    q, k, v = qkv(9, 65)
    masks = [torch.ones((9, 65), device="xpu", dtype=torch.float32),
             torch.ones((1, 1, 9, 65), device="xpu", dtype=torch.bool),
             torch.ones((9, 130), device="xpu", dtype=torch.bool)[:, ::2],
             torch.ones((9, 64), device="xpu", dtype=torch.bool)]
    for mask in masks:
        with pytest.raises(RuntimeError):
            getattr(masked_ops, name)(q, k, v, mask)
    with pytest.raises(RuntimeError, match="BF16 Q/K/V"):
        getattr(masked_ops, name)(q, k.to(torch.float16), v,
                                  torch.ones((9, 65), device="xpu", dtype=torch.bool))
    with pytest.raises(RuntimeError, match="one XPU device"):
        getattr(masked_ops, name)(q, k, v,
                                  torch.ones((9, 65), device="cpu", dtype=torch.bool))
    overlapping_q = q.as_strided(q.shape, (q.stride(0), q.stride(1), 128, 1))
    with pytest.raises(RuntimeError):
        getattr(masked_ops, name)(overlapping_q, k, v,
                                  torch.ones((9, 65), device="xpu", dtype=torch.bool))
    grad_q = q.detach().requires_grad_(True)
    with pytest.raises(RuntimeError):
        getattr(masked_ops, name)(grad_q, k, v,
                                  torch.ones((9, 65), device="xpu", dtype=torch.bool))


def replay(value, meta):
    shape, stride, offset = meta["shape"], meta["stride"], meta["storage_offset"]
    span = offset + 1 + sum((dim - 1) * step for dim, step in zip(shape, stride))
    storage = torch.empty(span, dtype=value.dtype, device="xpu")
    out = storage.as_strided(shape, stride, offset)
    out.copy_(value.to("xpu"))
    return out


@pytest.mark.parametrize("name", ["sdp", "sdp_split4"])
@pytest.mark.parametrize("geometry", [row for row in MATRIX["geometries"] if row["real_capture"]],
                         ids=lambda row: row["label"])
def test_real_masked_qkv(masked_ops, name, geometry):
    label = geometry["real_capture"].removeprefix("workspace-artifacts:")
    path = Path(REAL_ROOT) / label
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == geometry["real_capture_sha256"]
    sample = torch.load(path, map_location="cpu", weights_only=True)
    meta = sample["metadata"]
    q, k, v = (replay(sample[key], meta[key]) for key in ("q", "k", "v"))
    mask = replay(sample["mask"], meta["mask"])
    assert q.shape[1] == geometry["q"] and k.shape[1] == geometry["kv"]
    assert mask.dtype == torch.bool and mask.is_contiguous()
    qh, kh, vh = (t.view(1, t.shape[1], 32, 128).transpose(1, 2) for t in (q, k, v))
    expected = oracle(qh, kh, vh, mask)
    assert_screen(getattr(masked_ops, name)(qh, kh, vh, mask), expected)


def test_installed_dense_and_new_masked_coexist_in_one_process(masked_ops):
    from omni_xpu_kernel import cute
    q, k, v, before = _DENSE_BEFORE_NEW_DSO
    mask = torch.ones((31, 65), device="xpu", dtype=torch.bool)
    assert_screen(masked_ops.sdp(q, k, v, mask), oracle(q, k, v, mask))
    after = cute.sdp_bhld_d128(q, k, v)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
