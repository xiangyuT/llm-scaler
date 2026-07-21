"""Portable control-flow tests for the fused Lumina INT8 FFN route."""

from __future__ import annotations

import importlib.util
import sys
import types
import weakref
from pathlib import Path

import torch

_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
_PATCHES = _PLUGIN / "patches"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_patch(monkeypatch, candidate):
    package = types.ModuleType("omnixpu_ffn_test")
    package.__path__ = [str(_PLUGIN)]
    patches = types.ModuleType("omnixpu_ffn_test.patches")
    patches.__path__ = [str(_PATCHES)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    _load_module("omnixpu_ffn_test.patches.debug", _PATCHES / "debug.py")
    patch = _load_module(
        "omnixpu_ffn_test.patches.patch_int8_ffn",
        _PATCHES / "patch_int8_ffn.py",
    )

    class FeedForward:
        def forward(self, x):
            self.original_calls += 1
            return x + 1

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_ops = types.ModuleType("comfy.ops")
    comfy_ldm = types.ModuleType("comfy.ldm")
    comfy_ldm.__path__ = []
    comfy_lumina = types.ModuleType("comfy.ldm.lumina")
    comfy_lumina.__path__ = []
    comfy_model = types.ModuleType("comfy.ldm.lumina.model")
    comfy_model.FeedForward = FeedForward
    comfy.ops = comfy_ops
    comfy.ldm = comfy_ldm
    comfy_ldm.lumina = comfy_lumina
    comfy_lumina.model = comfy_model

    omni = types.ModuleType("omni_xpu_kernel")
    omni.int8 = candidate
    for name, module in (
        ("comfy", comfy),
        ("comfy.ops", comfy_ops),
        ("comfy.ldm", comfy_ldm),
        ("comfy.ldm.lumina", comfy_lumina),
        ("comfy.ldm.lumina.model", comfy_model),
        ("omni_xpu_kernel", omni),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return patch, FeedForward, comfy_ops


class _Candidate:
    def __init__(self):
        self.calls = []
        self.up_refs = ()
        self.up_inputs_released_at_down = False

    def int8_linear_shared_input(self, x, *args, **kwargs):
        self.calls.append(("shared", kwargs))
        up1, up3 = x + 2, x + 3
        self.up_refs = (weakref.ref(up1), weakref.ref(up3))
        return up1, up3

    def fused_silu_mul_quantize_rowwise(self, x1, x2):
        self.calls.append(("fused", {}))
        return x1.to(torch.int8), torch.ones((*x1.shape[:-1], 1))

    def fused_silu_mul(self, x1, x2):
        self.calls.append(("fused_float", {}))
        return x1 + x2

    def rotate_convrot(self, x, group_size):
        self.calls.append(("convrot", {"group_size": group_size}))
        return x

    def quantize_int8_rowwise(self, x):
        self.calls.append(("quantize", {}))
        return x.to(torch.int8), torch.ones((*x.shape[:-1], 1))

    def int8_linear_prequantized(self, q, scale, *args, **kwargs):
        self.calls.append(("down", kwargs))
        self.up_inputs_released_at_down = all(ref() is None for ref in self.up_refs)
        return q.to(torch.float32) + 4

    def int8_linear(self, x, *args, **kwargs):
        self.calls.append(("linear", kwargs))
        return x.to(torch.float32) + 5


def test_ffn_environment_switch_is_nested_under_int8(monkeypatch):
    config_path = _PLUGIN / "config.py"
    monkeypatch.setenv("OMNIXPU_ENABLE", "1")
    monkeypatch.setenv("OMNIXPU_INT8", "1")
    monkeypatch.setenv("OMNIXPU_INT8_FFN", "0")
    config = _load_module("omnixpu_ffn_config_disabled", config_path)
    assert config.config.int8
    assert not config.config.int8_ffn

    monkeypatch.setenv("OMNIXPU_INT8", "0")
    monkeypatch.setenv("OMNIXPU_INT8_FFN", "1")
    config = _load_module("omnixpu_ffn_config_parent_disabled", config_path)
    assert not config.config.int8
    assert not config.config.int8_ffn


def test_cpu_input_uses_original_comfy_route(monkeypatch):
    candidate = _Candidate()
    patch, FeedForward, comfy_ops = _load_patch(monkeypatch, candidate)
    comfy_ops.run_every_op = lambda: None

    assert patch.apply() == (True, "")
    module = FeedForward()
    module.original_calls = 0
    x = torch.zeros((2, 4), dtype=torch.bfloat16)
    torch.testing.assert_close(module.forward(x), x + 1)
    assert module.original_calls == 1
    assert candidate.calls == []
    assert patch.get_stats() == {
        "routed": 0,
        "fallback": 1,
        "reasons": {"device": 1},
    }


def test_eligible_route_chains_all_three_kernel_apis(monkeypatch):
    candidate = _Candidate()
    patch, FeedForward, comfy_ops = _load_patch(monkeypatch, candidate)
    interruptions = []
    comfy_ops.run_every_op = lambda: interruptions.append(True)
    assert patch.apply() == (True, "")

    x = torch.zeros((2, 4), dtype=torch.bfloat16)
    weight = patch._Weight(
        qdata=torch.zeros((4, 4), dtype=torch.int8),
        scale=torch.ones((4, 1)),
        convrot=False,
        convrot_groupsize=256,
    )
    monkeypatch.setattr(
        patch, "_route_inputs", lambda module, input: ((weight, weight, weight), "")
    )

    module = FeedForward()
    module.original_calls = 0
    output = module.forward(x)
    assert interruptions == [True]
    assert module.original_calls == 0
    assert [name for name, _ in candidate.calls] == ["shared", "fused", "down"]
    assert candidate.calls[0][1]["out_dtype"] == torch.bfloat16
    assert candidate.calls[2][1]["out_dtype"] == torch.bfloat16
    assert candidate.up_inputs_released_at_down
    torch.testing.assert_close(output, torch.full_like(output, 6.0))


def test_convrot_route_uses_staged_fused_producer(monkeypatch):
    candidate = _Candidate()
    patch, FeedForward, comfy_ops = _load_patch(monkeypatch, candidate)
    comfy_ops.run_every_op = lambda: None
    assert patch.apply() == (True, "")

    x = torch.zeros((2, 4), dtype=torch.bfloat16)
    up = patch._Weight(
        qdata=torch.zeros((4, 4), dtype=torch.int8),
        scale=torch.ones((4, 1)),
        convrot=False,
        convrot_groupsize=256,
    )
    down = patch._Weight(
        qdata=torch.zeros((4, 4), dtype=torch.int8),
        scale=torch.ones((4, 1)),
        convrot=True,
        convrot_groupsize=4,
    )
    monkeypatch.setattr(
        patch, "_route_inputs", lambda module, input: ((up, up, down), "")
    )

    module = FeedForward()
    module.original_calls = 0
    output = module.forward(x)
    assert module.original_calls == 0
    assert [name for name, _ in candidate.calls] == [
        "shared",
        "fused_float",
        "convrot",
        "quantize",
        "down",
    ]
    assert candidate.calls[2][1]["group_size"] == 4
    assert candidate.up_inputs_released_at_down
    torch.testing.assert_close(output, torch.full_like(output, 9.0))


def test_weight_extraction_rejects_dynamic_weight_transform(monkeypatch):
    candidate = _Candidate()
    patch, _, _ = _load_patch(monkeypatch, candidate)
    x = torch.zeros((2, 4), dtype=torch.bfloat16)
    params = types.SimpleNamespace(
        scale=torch.ones((4, 1)),
        orig_dtype=torch.bfloat16,
        transposed=False,
        convrot=False,
        convrot_groupsize=256,
    )
    weight = types.SimpleNamespace(
        _layout_cls="TensorWiseINT8Layout",
        _qdata=torch.zeros((4, 4), dtype=torch.int8),
        _params=params,
    )
    module = types.SimpleNamespace(
        weight=weight,
        bias=None,
        quant_format="int8_tensorwise",
        layout_type="TensorWiseINT8Layout",
        _full_precision_mm=False,
        comfy_force_cast_weights=False,
        weight_function=[lambda value: value],
        bias_function=[],
    )
    extracted, reason = patch._module_weight(module, x)
    assert extracted is None
    assert reason == "weight_function"


def test_omnigen_routes_share_ffn_and_qkv_quantization(monkeypatch):
    candidate = _Candidate()
    patch, _, comfy_ops = _load_patch(monkeypatch, candidate)
    interruptions = []
    comfy_ops.run_every_op = lambda: interruptions.append(True)

    class LuminaFeedForward:
        def forward(self, x):
            self.original_calls += 1
            return x + 10

    class Attention:
        def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
            self.original_calls += 1
            return hidden_states + 20

    omnigen_package = types.ModuleType("comfy.ldm.omnigen")
    omnigen_package.__path__ = []
    omnigen_model = types.ModuleType("comfy.ldm.omnigen.omnigen2")
    omnigen_model.LuminaFeedForward = LuminaFeedForward
    omnigen_model.Attention = Attention
    omnigen_model.apply_rotary_emb = lambda tensor, _: tensor

    def optimized_attention(query, key, value, heads, mask, **kwargs):
        return query.transpose(1, 2).reshape(query.shape[0], query.shape[2], -1)

    omnigen_model.optimized_attention_masked = optimized_attention
    monkeypatch.setitem(sys.modules, "comfy.ldm.omnigen", omnigen_package)
    monkeypatch.setitem(
        sys.modules, "comfy.ldm.omnigen.omnigen2", omnigen_model
    )
    patch._omni_int8 = candidate

    weight = patch._Weight(
        qdata=torch.zeros((4, 4), dtype=torch.int8),
        scale=torch.ones((4, 1)),
        convrot=True,
        convrot_groupsize=4,
    )
    monkeypatch.setattr(
        patch,
        "_route_named_inputs",
        lambda module, x, names: ((weight, weight, weight), ""),
    )
    monkeypatch.setattr(
        patch,
        "_route_qkv_inputs",
        lambda module, hidden, encoder: ((weight, weight, weight), ""),
    )

    assert patch._install_omnigen_routes(comfy_ops) == 2

    ffn = LuminaFeedForward()
    ffn.original_calls = 0
    x = torch.zeros((1, 2, 4), dtype=torch.bfloat16)
    ffn.forward(x)
    assert ffn.original_calls == 0
    assert [name for name, _ in candidate.calls] == ["shared", "linear"]

    candidate.calls.clear()
    attention = Attention()
    attention.original_calls = 0
    attention.heads = 1
    attention.kv_heads = 1
    attention.dim_head = 4
    attention.norm_q = lambda tensor: tensor
    attention.norm_k = lambda tensor: tensor
    attention.to_out = [lambda tensor: tensor]
    output = attention.forward(x, x)
    assert attention.original_calls == 0
    assert output.shape == x.shape
    assert [name for name, _ in candidate.calls] == [
        "convrot",
        "quantize",
        "down",
        "down",
        "down",
    ]
    assert interruptions == [True, True]


def test_boogu_joint_qkv_shares_each_stream_quantization(monkeypatch):
    candidate = _Candidate()
    patch, _, comfy_ops = _load_patch(monkeypatch, candidate)
    interruptions = []
    comfy_ops.run_every_op = lambda: interruptions.append(True)

    omnigen_package = types.ModuleType("comfy.ldm.omnigen")
    omnigen_package.__path__ = []
    omnigen_model = types.ModuleType("comfy.ldm.omnigen.omnigen2")
    boogu_package = types.ModuleType("comfy.ldm.boogu")
    boogu_package.__path__ = []
    boogu_model = types.ModuleType("comfy.ldm.boogu.model")

    class BooguDoubleStreamProcessor:
        def forward(self, attn, img, instruct, *args, **kwargs):
            self.original_calls += 1
            return torch.cat([instruct, img], dim=1) + 20

    boogu_model.BooguDoubleStreamProcessor = BooguDoubleStreamProcessor
    boogu_model.apply_rotary_emb = lambda tensor, _: tensor

    def optimized_attention(query, key, value, heads, mask, **kwargs):
        return query.transpose(1, 2).reshape(query.shape[0], query.shape[2], -1)

    boogu_model.optimized_attention_masked = optimized_attention
    for name, module in (
        ("comfy.ldm.omnigen", omnigen_package),
        ("comfy.ldm.omnigen.omnigen2", omnigen_model),
        ("comfy.ldm.boogu", boogu_package),
        ("comfy.ldm.boogu.model", boogu_model),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    patch._omni_int8 = candidate
    weight = patch._Weight(
        qdata=torch.zeros((4, 4), dtype=torch.int8),
        scale=torch.ones((4, 1)),
        convrot=True,
        convrot_groupsize=4,
    )
    monkeypatch.setattr(
        patch,
        "_route_qkv_inputs",
        lambda module, hidden, encoder, *names: ((weight, weight, weight), ""),
    )
    assert patch._install_omnigen_routes(comfy_ops) == 1

    processor = BooguDoubleStreamProcessor()
    processor.original_calls = 0
    processor.instruct_out = lambda tensor: tensor
    processor.img_out = lambda tensor: tensor
    attn = types.SimpleNamespace(
        heads=1,
        kv_heads=1,
        dim_head=4,
        norm_q=lambda tensor: tensor,
        norm_k=lambda tensor: tensor,
        to_out=[lambda tensor: tensor],
    )
    image = torch.zeros((1, 2, 4), dtype=torch.bfloat16)
    instruction = torch.zeros((1, 1, 4), dtype=torch.bfloat16)
    output = processor.forward(attn, image, instruction, None)

    assert processor.original_calls == 0
    assert output.shape == (1, 3, 4)
    assert [name for name, _ in candidate.calls] == [
        "convrot",
        "quantize",
        "down",
        "down",
        "down",
        "convrot",
        "quantize",
        "down",
        "down",
        "down",
    ]
    assert interruptions == [True]
