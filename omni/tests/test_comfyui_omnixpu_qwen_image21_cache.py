"""Weight-free Qwen 2.1 adapter regression with installed ComfyUI and Torch."""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
import types

import pytest
import torch

PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"


def load_adapter():
    spec = importlib.util.spec_from_file_location(
        "omnixpu_qwen_cache_test", PLUGIN / "adapters/qwen_image21_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def comfy_runtime():
    if not torch.xpu.is_available():
        pytest.skip("requires installed XPU Torch")
    comfy_root = Path(os.environ.get("COMFYUI_ROOT", "/llm/ComfyUI"))
    if not comfy_root.is_dir():
        pytest.skip("requires installed ComfyUI")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(comfy_root))
        patch.setenv("OMNIXPU_PROVIDER_BOOTSTRAP", "auto")
        name = "_comfyui_omnixpu_runtime_bootstrap"
        bootstrap = sys.modules.get(name)
        if bootstrap is None:
            spec = importlib.util.spec_from_file_location(
                name, PLUGIN / "runtime_bootstrap.py"
            )
            bootstrap = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, name, bootstrap)
            spec.loader.exec_module(bootstrap)
        state = bootstrap.bootstrap(dynamic_vram_override=False)
        assert state["providers"]["comfy_kitchen.xpu"]["status"] == "active", state
        # With an installed runtime, missing or incompatible model imports fail
        # collection of the fixture instead of silently skipping its checks.
        qwen = importlib.import_module("comfy.ldm.qwen_image21.model")
        mm = importlib.import_module("comfy.model_management")
        mp = importlib.import_module("comfy.model_patcher")
        yield qwen, mm, mp


@pytest.fixture
def runtime(monkeypatch, comfy_runtime):
    qwen, mm, mp = comfy_runtime
    adapter = load_adapter()
    cache = qwen.PoseBranchCache
    # Permit use after the real plugin startup, while restoring that state
    # after each isolated apply/compatibility test.
    originals = []
    for owner, name in ((cache, "select"), (cache, "put"), (mp.ModelPatcher, "__init__")):
        method = getattr(owner, name)
        original = getattr(method, adapter._MARKER, method)
        monkeypatch.setattr(owner, name, original)
        originals.append(original)
    monkeypatch.setattr(mm.args, "disable_pinned_memory", False)
    monkeypatch.setattr(mm, "NUM_STREAMS", 2)
    return types.SimpleNamespace(adapter=adapter, cache=cache, qwen=qwen, mm=mm,
                                 mp=mp, original_put=originals[1], device=mm.get_torch_device())


def activate(runtime):
    ok, reason = runtime.adapter.apply()
    assert ok, reason


def test_apply_is_idempotent(runtime):
    activate(runtime)
    first = runtime.cache.select, runtime.cache.put, runtime.mp.ModelPatcher.__init__
    activate(runtime)
    assert first == (runtime.cache.select, runtime.cache.put, runtime.mp.ModelPatcher.__init__)


def test_unsupported_contract_is_atomic(runtime, monkeypatch):
    def unknown_put(self, layer, value):
        raise AssertionError("must not execute")
    monkeypatch.setattr(runtime.cache, "put", unknown_put)
    before = runtime.cache.select, runtime.mp.ModelPatcher.__init__
    ok, reason = runtime.adapter.apply()
    assert not ok and "signature" in reason
    assert runtime.cache.put is unknown_put
    assert before == (runtime.cache.select, runtime.mp.ModelPatcher.__init__)


def test_missing_cache_api_is_not_partially_applied(runtime, monkeypatch):
    before = runtime.cache.select, runtime.cache.put, runtime.mp.ModelPatcher.__init__
    monkeypatch.delattr(runtime.qwen, "PoseBranchCache")
    ok, reason = runtime.adapter.apply()
    assert not ok and "cache API is unavailable" in reason
    assert before == (runtime.cache.select, runtime.cache.put, runtime.mp.ModelPatcher.__init__)


def test_older_prefix_contract_is_not_partially_applied(runtime, monkeypatch):
    def _forward(self, x, timesteps, context, ref_latents=None, image_slots=None,
                 transformer_options={}, **kwargs):
        prefix_len = 1
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        if self.prefix_cache_enabled and prefix_len > 0 and not blocks_replace:
            return x
    monkeypatch.setattr(runtime.qwen.QwenImage21Transformer2DModel, "_forward", _forward)
    before = runtime.cache.select, runtime.cache.put, runtime.mp.ModelPatcher.__init__
    ok, reason = runtime.adapter.apply()
    assert not ok and "wrapper/cache contract" in reason
    assert before == (runtime.cache.select, runtime.cache.put, runtime.mp.ModelPatcher.__init__)


def test_non_xpu_runtime_does_not_install(runtime, monkeypatch):
    monkeypatch.setattr(runtime.mm, "get_torch_device", lambda: torch.device("cpu"))
    before = runtime.cache.select, runtime.cache.put, runtime.mp.ModelPatcher.__init__
    assert not runtime.adapter.apply()[0]
    assert before == (runtime.cache.select, runtime.cache.put, runtime.mp.ModelPatcher.__init__)


def test_changed_cache_body_is_not_partially_applied(runtime, monkeypatch):
    def put(self, i, t):
        return None
    monkeypatch.setattr(runtime.cache, "put", put)
    before = runtime.cache.select, runtime.cache.put, runtime.mp.ModelPatcher.__init__
    ok, reason = runtime.adapter.apply()
    assert not ok and "contract" in reason
    assert before == (runtime.cache.select, runtime.cache.put, runtime.mp.ModelPatcher.__init__)


@pytest.mark.parametrize("store", ["cpu", "xpu"])
def test_repeated_and_interleaved_slot_selection(runtime, store):
    activate(runtime)
    cache = runtime.cache(store_device=store)
    keys = [torch.full((1, 8), value, device=runtime.device) for value in (0., 1.)]
    try:
        for key in keys:
            cache.select(key)
        for index in (1, 1, 0, 0, 1, 0):
            assert cache.select(keys[index], create=False)
            assert cache.slot is cache.slots[-1]
            assert torch.equal(cache.slot["key"].to(runtime.device), keys[index])
            assert len(cache.slots) == 2
        assert not cache.select(keys[0] + 2, create=False)
        assert cache.slot is None and len(cache.slots) == 2
    finally:
        cache.free()


@pytest.mark.parametrize("storage_dtype", ["default", "int8", "int4"])
def test_pinned_quantized_cache_matches_original(runtime, storage_dtype):
    activate(runtime)
    cache = runtime.cache(store_device="cpu", dtype=storage_dtype)
    reference = runtime.cache(store_device="cpu", dtype=storage_dtype)
    key = torch.zeros(1, 8, device=runtime.device)
    cache.select(key)
    reference.select(key)
    values = [torch.randn(1, 2, 65, 4, 128, device=runtime.device,
                          dtype=torch.bfloat16) for _ in range(4)]
    try:
        for index, value in enumerate(values):
            runtime.original_put(reference, index, value)
            cache.put(index, value)
            assert cache.slot["blocks"][index].is_pinned()
            assert torch.equal(cache.slot["blocks"][index], reference.slot["blocks"][index])
        # Torch-owned host allocations must not enter the CUDA unregister list.
        assert not cache.slot["pinned"]
        actual, expected = [], []
        for index in range(4):
            cache.prefetch(index, runtime.device, torch.bfloat16)
            assert cache._pending[index][1] is not None
            actual.append(cache.take(index, runtime.device, torch.bfloat16, 1).clone())
            cache.prefetch(index + 1, runtime.device, torch.bfloat16)
            expected.append(reference.take(index, runtime.device, torch.bfloat16, 1).clone())
        for value, ref in zip(actual, expected):
            assert torch.equal(value, ref)
    finally:
        cache.free()
        reference.free()
    assert not cache.slots and not cache._pending and not cache._staging


@pytest.mark.parametrize("source_device,store,disabled", [
    ("cpu", "cpu", False), ("xpu", "xpu", False), ("xpu", "cpu", True),
])
def test_original_storage_routes(runtime, monkeypatch, source_device, store, disabled):
    activate(runtime)
    monkeypatch.setattr(runtime.mm.args, "disable_pinned_memory", disabled)
    cache = runtime.cache(store_device=store)
    cache.select(torch.zeros(1, 1, device=source_device))
    value = torch.ones(1, 2, 3, 1, 128, device=source_device)
    try:
        cache.put(0, value)
        assert not cache.slot["blocks"][0].is_pinned()
        assert cache.slot["blocks"][0].device.type == store
        assert torch.equal(cache.take(0, torch.device(source_device), value.dtype, 1), value)
    finally:
        cache.free()


def test_pinned_oom_falls_back_but_other_errors_propagate(runtime, monkeypatch):
    value = torch.ones(2, 8, device=runtime.device)
    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("fixture pinned capacity")
    monkeypatch.setattr(torch, "empty", oom)
    result = runtime.adapter._copy_pinned_cpu(value, torch.device("cpu"))
    assert not result.is_pinned() and torch.equal(result, value.cpu())
    def failed(*args, **kwargs):
        raise RuntimeError("fixture unexpected runtime")
    monkeypatch.setattr(torch, "empty", failed)
    with pytest.raises(RuntimeError, match="unexpected runtime"):
        runtime.adapter._copy_pinned_cpu(value, torch.device("cpu"))


def test_wan_input_cache_batch_reuse_and_eviction(runtime, monkeypatch):
    activate(runtime)
    cache = runtime.cache(store_device="cpu")
    a = torch.zeros(1, 8, device=runtime.device)
    value = torch.randn(1, 17, 128, device=runtime.device, dtype=torch.bfloat16)
    try:
        cache.select(a)
        cache.put(0, value)
        assert cache.slot["blocks"][0].is_pinned()
        # Wan Animate2 stores its first batch row and expands on consumption.
        actual = cache.take(0, runtime.device, value.dtype, 2)
        assert torch.equal(actual, value.repeat(2, 1, 1))
        monkeypatch.setattr(runtime.mm, "get_free_memory", lambda device: 0)
        cache.select(a + 1)
        assert len(cache.slots) == 1 and not cache.filled(1)
        assert cache.memory_bytes() == 0
    finally:
        cache.free()


def make_patcher(runtime):
    import comfy.ops
    import comfy.supported_models
    config = comfy.supported_models.QwenImage21(dict(
        image_model="qwen_image21", in_channels=64, out_channels=64, num_layers=2,
        num_attention_heads=1, attention_head_dim=128, context_in_dim=128, mlp_ratio=3,
        fused_mlp=True,
    ))
    config.set_inference_dtype(torch.float32, None, runtime.device)
    config.custom_operations = comfy.ops.manual_cast
    model = config.get_model({}, device=runtime.device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "text_norm.weight" in name or name.endswith("bias"):
                param.zero_()
            elif "norm" in name and param.ndim == 1:
                param.fill_(1)
            else:
                param.normal_(0, 0.02)
    return runtime.mp.ModelPatcher(model, runtime.device, torch.device("cpu"))


def model_options(patcher):
    import comfy.sampler_helpers
    options = copy.deepcopy(patcher.model_options)
    comfy.sampler_helpers.prepare_model_patcher(patcher, {}, options)
    return options["transformer_options"]


def test_patcher_clone_and_sampler_wrapper_registration(runtime):
    activate(runtime)
    patcher = make_patcher(runtime)
    clone = patcher.clone()
    key = runtime.adapter._WRAPPER_KEY
    for item in (patcher, clone):
        assert len(item.get_wrappers("diffusion_model", key)) == 1
        assert len(model_options(item)["wrappers"]["diffusion_model"][key]) == 1


@pytest.mark.parametrize("kind", ["post_input", "attn1_patch", "single_block", "replace"])
def test_prefix_changes_bypass_cache_through_model_patcher(runtime, kind):
    activate(runtime)
    patcher = make_patcher(runtime)
    model = patcher.model.diffusion_model
    options = model_options(patcher)
    x = torch.randn(1, 64, 3, 3, device=runtime.device)
    context = torch.randn(1, 5, 128, device=runtime.device)
    signal = torch.randn_like(context)
    t = torch.ones(1, device=runtime.device)
    control = {"amount": 0.0}
    if kind == "attn1_patch":
        def patch(q, k, v, pe, attn_mask, extra_options):
            prefix = extra_options["img_slice"][0]
            k, v = k.clone(), v.clone()
            k[:, :, :prefix] += control["amount"] * signal[:, :prefix].unsqueeze(1)
            v[:, :, :prefix] += control["amount"] * signal[:, :prefix].unsqueeze(1)
            return {"q": q, "k": k, "v": v, "pe": pe}
    else:
        def patch(args):
            image = args["img"].clone()
            image[:, :5] += control["amount"] * signal
            return {**args, "img": image}
    with torch.inference_mode():
        patcher.pre_run()
        try:
            model(x, t, context, transformer_options=options)
            assert model.prefix_cache is not None
            if kind == "replace":
                def replace(args, extra):
                    result = extra["original_block"](args)
                    result["img"] = result["img"].clone()
                    result["img"][:, :5] += control["amount"] * signal
                    return result
                options["patches_replace"] = {"dit": {("single_block", 0): replace}}
            else:
                options["patches"] = {kind: [patch]}
            model(x, t, context, transformer_options=options)
            assert model.prefix_cache is None and model.prefix_cache_enabled
            control["amount"] = 2
            actual = model(x, t, context, transformer_options=options)
            model.reset_prefix_cache(False)
            expected = model(x, t, context, transformer_options=options)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            assert not model.prefix_cache_enabled
            model.reset_prefix_cache(True)
            options.pop("patches", None)
            options.pop("patches_replace", None)
            model(x, t, context, transformer_options=options)
            assert model.prefix_cache is not None and model.prefix_cache.filled(2)
        finally:
            patcher.cleanup()
    assert model.prefix_cache is None and not model.prefix_cache_enabled


def test_prefix_guard_restores_state_after_exception(runtime):
    activate(runtime)
    patcher = make_patcher(runtime)
    model = patcher.model.diffusion_model
    options = model_options(patcher)
    def fail(args):
        raise RuntimeError("fixture interrupted patch")
    options["patches"] = {"post_input": [fail]}
    with torch.inference_mode():
        patcher.pre_run()
        try:
            with pytest.raises(RuntimeError, match="interrupted patch"):
                model(torch.ones(1, 64, 2, 2, device=runtime.device),
                      torch.ones(1, device=runtime.device),
                      torch.ones(1, 5, 128, device=runtime.device), transformer_options=options)
            assert model.prefix_cache is None and model.prefix_cache_enabled
        finally:
            patcher.cleanup()


def test_non_xpu_prefix_wrapper_is_transparent():
    adapter = load_adapter()
    calls = []
    class Executor:
        class_obj = object()  # No cache API needed on the original CPU path.
        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "original"
    x = torch.ones(1)
    options = {"patches": {"post_input": [object()]}}
    assert adapter._guard_prefix_cache(Executor(), x, x, x,
                                      transformer_options=options) == "original"
    assert calls[0][0][-1] is options
