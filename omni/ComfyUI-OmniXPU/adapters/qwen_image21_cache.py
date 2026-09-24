"""Adapt Qwen Image 2.1 prefix caching and its shared Wan cache on XPU."""

from __future__ import annotations

import functools
import inspect
import logging
import textwrap

import torch

log = logging.getLogger("ComfyUI-OmniXPU")
_MARKER = "__omnixpu_qwen_image21_cache_original__"
_WRAPPER_KEY = "omnixpu_qwen_image21_prefix_cache"
_PREFIX_PATCHES = ("post_input", "attn1_patch", "single_block")


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
        return True, "already patched"
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
    return True, None
