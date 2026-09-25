"""Focused opt-in direct epilogue gate; reuses the passed masked FP32 contract."""

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch


_REQUESTED = any(key in os.environ for key in (
    "QWEN21_MASKED_DIRECT_DSO", "QWEN21_MASKED_DIRECT_DSO_SHA256"))
pytestmark = pytest.mark.skipif(
    not _REQUESTED, reason="experimental direct masked D128 requires explicit DSO",
)
OLD_DSO_SHA = "70792b845574710d0807b4a1f9f1a968bd459a1f4d50d8e8293a32105014b470"
INSTALLED_DENSE_SHA = "c195494b5dd3deb1e7b161e71683f85a29ad53426a2fa4a16d8b97e3aee04c4b"


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _checked_path(path, expected):
    if not path or not Path(path).is_file() or _digest(path) != expected:
        raise RuntimeError(f"exact DSO identity mismatch: {path}")
    return str(Path(path).resolve())


_HELPER = Path(__file__).with_name("test_cute_qwen21_masked_d128.py")
_SPEC = importlib.util.spec_from_file_location("qwen21_masked_frozen_math_contract", _HELPER)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("passed masked numerical helper source is unavailable")
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


@pytest.fixture(scope="module")
def ops():
    direct_path = os.environ.get("QWEN21_MASKED_DIRECT_DSO")
    direct_sha = os.environ.get("QWEN21_MASKED_DIRECT_DSO_SHA256")
    old_path = os.environ.get("QWEN21_MASKED_D128_DSO")
    real_root = os.environ.get("QWEN21_MASKED_REAL_ROOT")
    if not direct_sha or len(direct_sha) != 64 or any(c not in "0123456789abcdef" for c in direct_sha):
        raise RuntimeError("explicit direct DSO needs its full lowercase SHA-256")
    if not real_root or not Path(real_root).is_dir():
        raise RuntimeError("QWEN21_MASKED_REAL_ROOT must contain the five retained captures")
    direct_path = _checked_path(direct_path, direct_sha)
    old_path = _checked_path(old_path, OLD_DSO_SHA)
    from omni_xpu_kernel import cute
    _checked_path(cute._find_extension(), INSTALLED_DENSE_SHA)
    torch.set_float32_matmul_precision("highest")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("FP32 oracle precision differs from the frozen screen")

    q, k, v = contract.qkv(31, 65)
    mask = torch.ones((31, 65), device="xpu", dtype=torch.bool)
    dense_before = cute.sdp_bhld_d128(q, k, v).clone()
    torch.ops.load_library(old_path)
    old_before = torch.ops.qwen21_masked_d128.sdp(q, k, v, mask).clone()
    torch.ops.load_library(direct_path)
    namespace = torch.ops.qwen21_masked_direct_d128
    reference = contract.oracle(q, k, v, mask)
    contract.assert_screen(namespace.sdp_direct(q, k, v, mask), reference)
    contract.assert_screen(namespace.sdp(q, k, v, mask), reference)
    contract.assert_screen(namespace.sdp_split4(q, k, v, mask), reference)
    torch.testing.assert_close(cute.sdp_bhld_d128(q, k, v), dense_before, rtol=0, atol=0)
    torch.testing.assert_close(torch.ops.qwen21_masked_d128.sdp(q, k, v, mask),
                               old_before, rtol=0, atol=0)
    return namespace


@pytest.mark.parametrize("name", ["sdp", "sdp_split4", "sdp_direct"])
@pytest.mark.parametrize("geometry", contract.MATRIX["geometries"],
                         ids=lambda row: row["label"])
def test_new_dso_seven_current_geometry_classes(ops, name, geometry):
    contract.test_seven_actual_masked_geometry_classes(ops, name, geometry)


@pytest.mark.parametrize("name", ["sdp", "sdp_split4", "sdp_direct"])
@pytest.mark.parametrize("q_len,kv_len", [(1, 1), (7, 31), (9, 65), (17, 1025), (31, 4097)])
def test_new_dso_mask_bits_holes_and_tails(ops, name, q_len, kv_len):
    contract.test_actual_mask_bits_holes_and_tails(ops, name, q_len, kv_len)


@pytest.mark.parametrize("name", ["sdp", "sdp_split4", "sdp_direct"])
@pytest.mark.parametrize("q_layout", ["packed", "blhd"])
def test_new_dso_mixed_qkv_layout_offsets_and_immutability(ops, name, q_layout):
    contract.test_mixed_qkv_layout_offsets_and_input_immutability(
        ops, name, q_layout)


@pytest.mark.parametrize("name", ["sdp", "sdp_split4", "sdp_direct"])
def test_new_dso_balanced_visible_pairs(ops, name):
    contract.test_exact_zero_only_for_balanced_visible_pairs(ops, name)


@pytest.mark.parametrize("name", ["sdp", "sdp_split4", "sdp_direct"])
def test_new_dso_peaked_irregular_and_zero_v(ops, name):
    contract.test_peaked_and_irregular_cancellation_use_fp32_oracle(ops, name)
    contract.test_zero_v_and_globally_invisible_kv_do_not_affect_output(
        ops, name)


@pytest.mark.parametrize("name", ["sdp", "sdp_split4", "sdp_direct"])
def test_new_dso_negative_native_contract(ops, name):
    contract.test_reject_unsupported_mask_contract(ops, name)


@pytest.mark.parametrize("name", ["sdp", "sdp_split4", "sdp_direct"])
@pytest.mark.parametrize("geometry", [row for row in contract.MATRIX["geometries"]
                                     if row["real_capture"]],
                         ids=lambda row: row["label"])
def test_new_dso_five_real_masked_captures(ops, name, geometry):
    if contract.REAL_ROOT != os.environ.get("QWEN21_MASKED_REAL_ROOT"):
        raise RuntimeError("real capture root changed after the frozen helper import")
    contract.test_real_masked_qkv(ops, name, geometry)


def test_direct_nonempty_nan_logits_match_fp32_and_torch(ops):
    q, k, v = contract.qkv(8, 8)
    q.zero_()
    k.zero_()
    q[:, :, :, 0] = float("nan")
    mask = torch.ones((8, 8), device="xpu", dtype=torch.bool)
    reference = contract.oracle(q, k, v, mask)
    torch_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    direct = ops.sdp_direct(q, k, v, mask)
    assert bool(torch.isnan(reference).all())
    assert bool(torch.isnan(torch_out).all())
    assert bool(torch.isnan(direct).all())
