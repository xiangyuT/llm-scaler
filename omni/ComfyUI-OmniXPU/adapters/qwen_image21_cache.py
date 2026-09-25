"""Adapt Qwen Image 2.1 prefix caching and its shared Wan cache on XPU."""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
import textwrap
from pathlib import Path

import torch

log = logging.getLogger("ComfyUI-OmniXPU")
_MARKER = "__omnixpu_qwen_image21_cache_original__"
_WRAPPER_KEY = "omnixpu_qwen_image21_prefix_cache"
_PREFIX_PATCHES = ("post_input", "attn1_patch", "single_block")
_SEGMENTED_MARKER = "__omnixpu_qwen21_segmented_prefix_original__"
_SEGMENTED_DSO_ENV = "OMNIXPU_EXPERIMENTAL_QWEN21_SEGMENTED_DSO"
_SEGMENTED_SHA_ENV = "OMNIXPU_EXPERIMENTAL_QWEN21_SEGMENTED_SHA256"
_SEGMENTED_LOADED = None
_SEGMENTED_MAX_ELEMENTS = (1 << 31) - 1


def _segmented_prefix_inputs(q, k, v, prefix_k, prefix_v, heads, options):
    """Return five no-copy BHLD views only for the explicit safe cache-hit path."""
    if (not isinstance(options, dict)
            or not os.environ.get(_SEGMENTED_DSO_ENV)
            or not os.environ.get(_SEGMENTED_SHA_ENV)
            or os.environ.get("OMNI_ATTN_BACKEND", "auto").lower() not in ("auto", "cute")
            or torch.compiler.is_compiling() or heads != 32
            or options.get("optimized_attention_override")
            or options.get("patches") or options.get("patches_replace")):
        return None
    try:
        import omni_xpu_kernel
        if omni_xpu_kernel.__xpu_target__ != "bmg":
            return None
        tensors = (q, k, v, prefix_k, prefix_v)
        if (q.device.type != "xpu" or q.dtype != torch.bfloat16
                or any(t.device != q.device or t.dtype != q.dtype or t.requires_grad
                       for t in tensors)
                or any(t.ndim != 4 or t.shape[0] != 1 or t.shape[2:] != (32, 128)
                       or t.stride()[1:] != (4096, 128, 1)
                       or t.data_ptr() % 64 for t in tensors)):
            return None
        q_len, current_len, prefix_len = q.shape[1], k.shape[1], prefix_k.shape[1]
        if (min(q_len, current_len, prefix_len) <= 0
                or v.shape != k.shape or prefix_v.shape != prefix_k.shape
                or any(length * 4096 > _SEGMENTED_MAX_ELEMENTS
                       for length in (q_len, current_len, prefix_len))
                or prefix_len + current_len > _SEGMENTED_MAX_ELEMENTS
                or ((prefix_len + 63) // 64 + (current_len + 63) // 64) * 64
                   > _SEGMENTED_MAX_ELEMENTS):
            return None
        index = q.device.index if q.device.index is not None else torch.xpu.current_device()
        if getattr(torch.xpu.get_device_properties(index), "device_id", None) != 0xE223:
            return None
        return tuple(t.permute(0, 2, 1, 3) for t in tensors)
    except (AttributeError, ImportError, TypeError, ValueError):
        return None


def _run_segmented_prefix(prepared):
    global _SEGMENTED_LOADED
    configured = (os.environ[_SEGMENTED_DSO_ENV], os.environ[_SEGMENTED_SHA_ENV])
    if _SEGMENTED_LOADED != configured:
        if _SEGMENTED_LOADED is not None:
            raise RuntimeError("segmented prefix sidecar identity cannot change in process")
        path = Path(configured[0]).resolve()
        if not path.is_file() or len(configured[1]) != 64:
            raise RuntimeError("segmented prefix sidecar path/SHA is incomplete")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != configured[1]:
            raise RuntimeError("segmented prefix sidecar SHA mismatch")
        torch.ops.load_library(str(path))
        _SEGMENTED_LOADED = configured
    out = torch.ops.qwen21_segmented_d128.sdp_prefix(*prepared)
    q = prepared[0]
    if tuple(out.shape) != tuple(q.shape) or out.dtype != q.dtype:
        raise RuntimeError("segmented prefix sidecar output contract mismatch")
    return out.transpose(1, 2).reshape(1, q.shape[2], 4096)


def _segmented_contract_error(qwen):
    original = qwen.prefix_cached_attention
    if hasattr(original, _SEGMENTED_MARKER):
        return None
    path_set = _SEGMENTED_DSO_ENV in os.environ
    sha_set = _SEGMENTED_SHA_ENV in os.environ
    if not path_set and not sha_set:
        return None
    if not path_set or not sha_set:
        return "segmented prefix opt-in requires both DSO path and SHA"
    if not os.environ[_SEGMENTED_DSO_ENV] or not os.environ[_SEGMENTED_SHA_ENV]:
        return "segmented prefix opt-in requires nonempty DSO path and SHA"
    if tuple(inspect.signature(original).parameters) != (
            "prefix_k", "prefix_v", "transformer_options"):
        return "unsupported prefix_cached_attention signature"
    source = inspect.getsource(original)
    if (source.count("torch.cat([prefix_k, k], dim=1).flatten(2)") != 1
            or source.count("torch.cat([prefix_v, v], dim=1).flatten(2)") != 1):
        return "unsupported prefix_cached_attention source contract"
    return None


def _install_segmented_prefix(qwen):
    original = qwen.prefix_cached_attention
    error = _segmented_contract_error(qwen)
    if error is not None:
        return False, error
    if hasattr(original, _SEGMENTED_MARKER) or not os.environ.get(_SEGMENTED_DSO_ENV):
        return True, None

    @functools.wraps(original)
    def factory(prefix_k, prefix_v, transformer_options={}):
        fallback = original(prefix_k, prefix_v, transformer_options)

        def attention(q, k, v, heads):
            prepared = _segmented_prefix_inputs(
                q, k, v, prefix_k, prefix_v, heads, transformer_options)
            if prepared is None:
                return fallback(q, k, v, heads)
            return _run_segmented_prefix(prepared)

        return attention

    setattr(factory, _SEGMENTED_MARKER, original)
    qwen.prefix_cached_attention = factory
    return True, None


def _rewrite(function, replacements, helpers=None):
    """Adapt exact upstream statements while retaining quantization/lifecycle code."""
    if function.__closure__:
        raise ValueError("cache method unexpectedly captures closure state")
    source = textwrap.dedent(inspect.getsource(function))
    if not source.startswith(f"def {function.__name__}("):
        raise ValueError("unsupported decorated cache method")
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f"unsupported {function.__name__} cache contract")
        source = source.replace(before, after)
    namespace = dict(function.__globals__)
    namespace.update(helpers or {})
    exec(compile(source, inspect.getfile(function), "exec"), namespace)
    return functools.update_wrapper(namespace[function.__name__], function)


def _copy_pinned_cpu(tensor, device):
    try:
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device=device, pin_memory=True)
    except torch.OutOfMemoryError:
        # Match upstream's pageable fallback when host registration cannot fit.
        log.warning("[OmniXPU] KV cache pinned allocation exhausted; using pageable RAM")
        return tensor.to(device, copy=True)
    # Cache fill completes before CPU key/slot reuse. Prefetch back to XPU uses
    # the upstream stream waits and staging buffers; Torch owns host lifetime.
    host.copy_(tensor, non_blocking=False)
    return host


def _guard_prefix_cache(executor, x, timesteps, context, ref_latents=None,
                        image_slots=None, transformer_options=None, **kwargs):
    options = {} if transformer_options is None else transformer_options
    patches = options.get("patches", {})
    unsafe = any(patches.get(name) for name in _PREFIX_PATCHES)
    unsafe = unsafe or bool(options.get("patches_replace", {}).get("dit", {}))
    model = executor.class_obj
    if x.device.type != "xpu" or not unsafe:
        return executor(x, timesteps, context, ref_latents, image_slots, options, **kwargs)

    enabled = model.prefix_cache_enabled
    model.reset_prefix_cache(False)
    try:
        return executor(x, timesteps, context, ref_latents, image_slots, options, **kwargs)
    finally:
        # Never restore a stale slot, including after an interrupted forward.
        model.reset_prefix_cache(enabled)


def apply():
    try:
        import comfy.ldm.qwen_image21.model as qwen
        import comfy.model_base as model_base
        import comfy.model_management as mm
        import comfy.model_patcher as model_patcher
        from comfy.patcher_extension import WrappersMP
    except ModuleNotFoundError as exc:
        if exc.name in {"comfy", "comfy.ldm.qwen_image21", "comfy.ldm.qwen_image21.model"}:
            return False, "Qwen Image 2.1 is not available in this ComfyUI"
        raise

    if mm.get_torch_device().type != "xpu":
        return False, "Qwen Image 2.1 cache adapter requires XPU"

    try:
        cache = qwen.PoseBranchCache
        patcher = model_patcher.ModelPatcher
        originals = (cache.select, cache.put, patcher.__init__)
    except AttributeError as exc:
        return False, f"Qwen Image 2.1 cache API is unavailable: {exc}"
    if all(hasattr(method, _MARKER) for method in originals):
        return _install_segmented_prefix(qwen)
    if any(hasattr(method, _MARKER) for method in originals):
        return False, "Qwen Image 2.1 cache adapter is partially applied"

    original_select, original_put, original_init = originals
    try:
        if tuple(inspect.signature(original_select).parameters) != ("self", "k", "create"):
            raise ValueError("unsupported PoseBranchCache.select signature")
        if inspect.signature(original_select).parameters["create"].default is not True:
            raise ValueError("unsupported PoseBranchCache.select default")
        if tuple(inspect.signature(original_put).parameters) != ("self", "i", "t"):
            raise ValueError("unsupported PoseBranchCache.put signature")
        forward = qwen.QwenImage21Transformer2DModel._forward
        if tuple(inspect.signature(forward).parameters) != (
            "self", "x", "timesteps", "context", "ref_latents", "image_slots",
            "transformer_options", "kwargs",
        ):
            raise ValueError("unsupported Qwen Image 2.1 forward signature")
        entry = inspect.getsource(qwen.QwenImage21Transformer2DModel.forward)
        body = inspect.getsource(forward)
        if ("WrappersMP.DIFFUSION_MODEL" not in entry
                or "self._forward" not in entry
                or "self.prefix_cache_enabled and prefix_len > 0 and not hooked" not in body
                or any(f'patches.get("{name}")' not in body for name in _PREFIX_PATCHES)):
            raise ValueError("unsupported Qwen Image 2.1 wrapper/cache contract")
        qwen_type = model_base.QwenImage21
        if not callable(getattr(patcher, "add_wrapper_with_key", None)):
            raise ValueError("ModelPatcher diffusion wrappers are unavailable")
        reset = inspect.getsource(qwen.QwenImage21Transformer2DModel.reset_prefix_cache)
        if "self.prefix_cache.free()" not in reset or "self.prefix_cache_enabled = enabled" not in reset:
            raise ValueError("unsupported Qwen Image 2.1 cache reset contract")
        select_xpu = _rewrite(original_select, (
            ("for s in self.slots:", "for index, s in enumerate(self.slots):"),
            ("self.slots.remove(s)", "self.slots.pop(index)"),
        ))
        put_xpu = _rewrite(original_put, (
            ("t = t.to(self.store_device, copy=True)",
             "t = _omnixpu_copy_pinned_cpu(t, self.store_device)"),
            ("if comfy.model_management.pin_memory(t):",
             "if not t.is_pinned() and comfy.model_management.pin_memory(t):"),
        ), {"_omnixpu_copy_pinned_cpu": _copy_pinned_cpu})
        segmented_error = _segmented_contract_error(qwen)
        if segmented_error is not None:
            raise ValueError(segmented_error)
    except (AttributeError, OSError, TypeError, ValueError, SyntaxError) as exc:
        return False, str(exc)

    @functools.wraps(original_select)
    def select(self, k, create=True):
        if k.device.type != "xpu":
            return original_select(self, k, create=create)
        return select_xpu(self, k, create=create)

    @functools.wraps(original_put)
    def put(self, i, t):
        if (t.device.type != "xpu" or self.store_device.type != "cpu"
                or mm.args.disable_pinned_memory or mm.in_training or t.requires_grad):
            return original_put(self, i, t)
        return put_xpu(self, i, t)

    @functools.wraps(original_init)
    def init(self, model, load_device, offload_device, *args, **kwargs):
        original_init(self, model, load_device, offload_device, *args, **kwargs)
        if isinstance(model, qwen_type) and load_device.type == "xpu":
            self.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY,
                                      _guard_prefix_cache)

    for replacement, original in zip((select, put, init), originals):
        setattr(replacement, _MARKER, original)
    cache.select, cache.put, patcher.__init__ = select, put, init
    return _install_segmented_prefix(qwen)
