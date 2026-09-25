"""Focused segmented prefix/current BF16 correctness; Luna runs on B70."""

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


_SELECTED = any(name in os.environ for name in
                ("QWEN21_SEGMENTED_D128_DSO", "QWEN21_SEGMENTED_D128_SHA256"))
pytestmark = pytest.mark.skipif(not _SELECTED, reason="segmented D128 sidecar opt-in")
OLD_MASKED_SHA = "70792b845574710d0807b4a1f9f1a968bd459a1f4d50d8e8293a32105014b470"
OLD_DIRECT_SHA = "7bc575e11c52b5779db80de99fb0ca3b90e036f9bf161d5321bf1f5aedc1a127"
REAL_HITS = {
    "r0": ("capture-other-attempt-001/r0-cache-hit-attempt-002/capture/attempt-lchhcvt9/hit-layer15-target.pt",
           "7159375210c19c92973458b8ca3ce3d7ce0479f5ee3be5f279be1af3d43cc4c4", 335),
    "r1": ("capture-other-attempt-001/cells/r1-official-target-only/capture/attempt-3yghpukr/hit-layer15-target.pt",
           "3306f2860a33db4b85d9a2c799230c7afda97ae5abcffac50eacd0e66fc3eeb3", 4078),
    "r2": ("capture-r2-001/r2/capture/attempt-2814budp/hit-layer15-target.pt",
           "2e9a7a3fc7743821bda8fe63fe70730c36e390d5c8004127f67e090c52c35c29", 8156),
    "r3": ("capture-other-attempt-001/cells/r3-official-pair-plus-sofa-mixed/capture/attempt-likjwqzj/hit-layer15-target.pt",
           "f9eab1c345345afecc5e5f98417581b25c8598b4f76c09f0417dc6f0e6fc2e55", 10659),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_helper():
    path = Path(__file__).with_name("test_cute_qwen21_masked_d128.py")
    spec = importlib.util.spec_from_file_location("segmented_frozen_fp32_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen FP32 helper unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load_helper()


@pytest.fixture(scope="module")
def segmented():
    path = os.environ.get("QWEN21_SEGMENTED_D128_DSO")
    expected = os.environ.get("QWEN21_SEGMENTED_D128_SHA256")
    if (not path or not Path(path).is_file() or not expected or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
            or digest(path) != expected):
        raise RuntimeError("exact segmented sidecar path/SHA required")
    real_root = os.environ.get("QWEN21_SEGMENTED_REAL_ROOT")
    if not real_root or not Path(real_root).is_dir():
        raise RuntimeError("QWEN21_SEGMENTED_REAL_ROOT must contain four target-hit captures")
    from omni_xpu_kernel import cute
    q, k, v = contract.qkv(17, 65)
    dense_before = cute.sdp_bhld_d128(q, k, v).clone()
    for env, sha in (("QWEN21_MASKED_D128_DSO", OLD_MASKED_SHA),
                     ("QWEN21_MASKED_DIRECT_DSO", OLD_DIRECT_SHA)):
        old = os.environ.get(env)
        if not old or not Path(old).is_file() or digest(old) != sha:
            raise RuntimeError(f"previous sidecar identity missing: {env}")
        torch.ops.load_library(old)
    mask = torch.ones((17, 65), dtype=torch.bool, device="xpu")
    old_masked = torch.ops.qwen21_masked_d128.sdp(q, k, v, mask).clone()
    old_direct = torch.ops.qwen21_masked_direct_d128.sdp_direct(q, k, v, mask).clone()
    torch.ops.load_library(path)
    op = torch.ops.qwen21_segmented_d128.sdp_prefix
    if "sdp_prefix(Tensor q, Tensor k_current, Tensor v_current, Tensor k_prefix, Tensor v_prefix) -> Tensor" not in str(op.default._schema):
        raise RuntimeError("segmented sidecar schema mismatch")
    torch.testing.assert_close(cute.sdp_bhld_d128(q, k, v), dense_before, rtol=0, atol=0)
    torch.testing.assert_close(torch.ops.qwen21_masked_d128.sdp(q, k, v, mask),
                               old_masked, rtol=0, atol=0)
    torch.testing.assert_close(torch.ops.qwen21_masked_direct_d128.sdp_direct(q, k, v, mask),
                               old_direct, rtol=0, atol=0)
    torch.set_float32_matmul_precision("highest")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("FP32 precision was not fixed")
    return op


def values(q_len, prefix_len, current_len, seed=20260925):
    torch.manual_seed(seed)
    def one(length):
        return torch.randn((1, length, 32, 128), device="xpu",
                           dtype=torch.bfloat16).permute(0, 2, 1, 3)
    return one(q_len), one(current_len), one(current_len), one(prefix_len), one(prefix_len)


def screen(op, q, kc, vc, kp, vp):
    joined_k = torch.cat((kp, kc), dim=2)
    joined_v = torch.cat((vp, vc), dim=2)
    all_visible = torch.ones((q.shape[2], joined_k.shape[2]),
                             device="xpu", dtype=torch.bool)
    reference = contract.oracle(q, joined_k, joined_v, all_visible)
    contract.assert_screen(F.scaled_dot_product_attention(q, joined_k, joined_v), reference)
    actual = op(q, kc, vc, kp, vp)
    contract.assert_screen(actual, reference)
    q_len = q.shape[2]
    head_stride, sequence_stride, width_stride = actual.stride()[1:]
    packed = head_stride == q_len * 128 and (q_len == 1 or sequence_stride == 128)
    blhd_backed = head_stride == 128 and (q_len == 1 or sequence_stride == 4096)
    if (actual.shape != q.shape or actual.dtype != torch.bfloat16
            or width_stride != 1 or not (packed or blhd_backed)
            or actual.data_ptr() in {
                q.data_ptr(), kc.data_ptr(), vc.data_ptr(),
                kp.data_ptr(), vp.data_ptr()}):
        raise AssertionError("segmented output layout/alias contract changed")


@pytest.mark.parametrize("q_len,prefix_len,current_len", [
    (1, 1, 1), (7, 31, 17), (17, 65, 33), (129, 127, 128),
])
def test_prefix_and_current_tails_share_one_softmax(segmented, q_len, prefix_len, current_len):
    screen(segmented, *values(q_len, prefix_len, current_len))


def test_packed_bhld_five_input_layout(segmented):
    torch.manual_seed(20260925)
    def packed(length):
        return torch.randn((1, 32, length, 128), device="xpu", dtype=torch.bfloat16)
    screen(segmented, packed(9), packed(17), packed(17), packed(31), packed(31))


@pytest.mark.parametrize("prefix_wins", [True, False])
def test_cross_boundary_peaked_logits_and_values(segmented, prefix_wins):
    q, kc, vc, kp, vp = values(9, 31, 17)
    q.zero_()
    kc.zero_()
    kp.zero_()
    q[:, :, :, 0] = 8
    kp[:, :, :, 0] = 1 if prefix_wins else -1
    kc[:, :, :, 0] = -1 if prefix_wins else 1
    vp.fill_(100)
    vc.fill_(-100)
    screen(segmented, q, kc, vc, kp, vp)


def test_zero_v_and_nonzero_offsets_preserve_inputs(segmented):
    backing = torch.randn((1, 26, 32, 128), device="xpu", dtype=torch.bfloat16)
    q = backing[:, 7:24].permute(0, 2, 1, 3)
    _, kc, vc, kp, vp = values(17, 31, 33)
    prefix_pair = torch.stack((kp.transpose(1, 2), vp.transpose(1, 2)), dim=1)
    kp, vp = (part.transpose(1, 2) for part in prefix_pair.unbind(1))
    inputs = (q, kc, vc, kp, vp)
    copies = tuple(t.clone() for t in inputs)
    screen(segmented, *inputs)
    for tensor, original in zip(inputs, copies):
        assert torch.equal(tensor, original)
    assert torch.count_nonzero(segmented(q, kc, torch.zeros_like(vc),
                                         kp, torch.zeros_like(vp))) == 0


def test_native_rejects_unsupported_contract(segmented):
    q, kc, vc, kp, vp = values(9, 31, 17)
    with pytest.raises(RuntimeError):
        segmented(q, kc.to(torch.float16), vc, kp, vp)
    with pytest.raises(RuntimeError):
        segmented(q, kc, vc, kp.cpu(), vp)
    with pytest.raises(RuntimeError):
        segmented(q, kc[:, :, ::2], vc[:, :, ::2], kp, vp)
    with pytest.raises(RuntimeError):
        segmented(q.detach().requires_grad_(True), kc, vc, kp, vp)
    storage = torch.empty(8 + q.numel(), device="xpu", dtype=q.dtype)
    offset_q = storage.as_strided(q.shape, (q.numel(), 128, 4096, 1), 8)
    assert offset_q.data_ptr() % 64 == 16
    with pytest.raises(RuntimeError, match="64-byte-aligned"):
        segmented(offset_q, kc, vc, kp, vp)


@pytest.mark.parametrize("cell", list(REAL_HITS))
def test_real_target_hit_qkv_split_without_input_copy(segmented, cell):
    relative, expected_sha, prefix_len = REAL_HITS[cell]
    path = Path(os.environ["QWEN21_SEGMENTED_REAL_ROOT"]) / relative
    if not path.is_file() or digest(path) != expected_sha:
        raise RuntimeError(f"target-hit capture identity mismatch: {cell}")
    sample = torch.load(path, map_location="cpu", weights_only=True)
    meta = sample["metadata"]
    if sample["mask"] is not None or meta["heads"] != 32:
        raise RuntimeError("target-hit capture is not unmasked B1/H32")
    q, k, v = (contract.replay(sample[name], meta[name]) for name in ("q", "k", "v"))
    if k.shape[1] - q.shape[1] != prefix_len or v.shape != k.shape:
        raise RuntimeError("capture prefix/current relation changed")
    qh = q.view(1, q.shape[1], 32, 128).transpose(1, 2)
    def split(value):
        prefix = value[:, :prefix_len].view(1, prefix_len, 32, 128).transpose(1, 2)
        current = value[:, prefix_len:].view(1, q.shape[1], 32, 128).transpose(1, 2)
        if prefix.untyped_storage().data_ptr() != value.untyped_storage().data_ptr():
            raise RuntimeError("reference replay unexpectedly copied prefix KV")
        return prefix, current
    kp, kc = split(k)
    vp, vc = split(v)
    screen(segmented, qh, kc, vc, kp, vp)
