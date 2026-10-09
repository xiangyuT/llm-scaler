"""Per-model public ComfyUI wrappers for resident execution graphs."""
from __future__ import annotations

from collections import Counter
import inspect
import os
import threading
import weakref

import torch

from ..graph_runtime import GraphRuntime, XPUBackend
from ..graph_constants import ResidentEmbedND

KEY = "omnixpu.execution_graph"
_STATES = weakref.WeakSet()
_TOTAL_COUNTS = Counter()
_TOTAL_REASONS = Counter()
_STATS_LOCK = threading.Lock()
_MODEL_TYPES = {
    ("comfy.ldm.modules.diffusionmodules.openaimodel", "UNetModel"),
    ("comfy.ldm.lumina.model", "NextDiT"),
}


def feature_enabled():
    return (os.environ.get("OMNIXPU_ENABLE", "1") != "0"
            and os.environ.get("OMNIXPU_XPU_GRAPH", "1") != "0")


def apply():
    from comfy.patcher_extension import WrappersMP
    if not all(hasattr(WrappersMP, name) for name in ("DIFFUSION_MODEL", "SAMPLER_SAMPLE")):
        return False, "public model/sampler wrappers unavailable"
    if not all(hasattr(torch.xpu, name) for name in ("XPUGraph", "graph", "Stream")):
        return False, "PyTorch XPU graph APIs unavailable"
    return True, "per-model opt-in; sampler-scoped resident graphs"


class ModelGraphState:
    def __init__(self, patcher=None, *, enabled=True, runtime_factory=None):
        self.patcher = weakref.ref(patcher) if patcher is not None else None
        self.enabled = enabled
        self.runtime_factory = runtime_factory or (lambda device: GraphRuntime(torch, XPUBackend(torch, device)))
        self.local = threading.local()
        self.stats = Counter()
        self.reasons = Counter()
        self.poisoned = False
        _STATES.add(self)

    def on_model_patcher_clone(self):
        return ModelGraphState(enabled=self.enabled, runtime_factory=self.runtime_factory)

    def sampler_wrapper(self, executor, *args, **kwargs):
        if self.poisoned:
            raise RuntimeError("previous XPU graph cleanup failed; restart the process")
        if getattr(self.local, "active", False):
            return executor(*args, **kwargs)
        self.local.active = True
        self.local.runtime = None
        before_stats, before_reasons = self.stats.copy(), self.reasons.copy()
        self.stats["samples"] += 1
        try:
            return executor(*args, **kwargs)
        finally:
            runtime = self.local.runtime
            try:
                if runtime is not None:
                    runtime.close()
                    self.stats["graph_teardowns"] += 1
            finally:
                if runtime is not None:
                    self.poisoned = self.poisoned or runtime.poisoned
                    self.stats.update(runtime.stats)
                    self.reasons.update(runtime.reasons)
                self.local.runtime = None
                self.local.active = False
                with _STATS_LOCK:
                    _TOTAL_COUNTS.update(self.stats - before_stats)
                    _TOTAL_REASONS.update(self.reasons - before_reasons)

    def eligibility(self, executor, options):
        patcher = self.patcher() if self.patcher else None
        module = executor.class_obj
        if not self.enabled or not feature_enabled():
            return "disabled"
        if not getattr(self.local, "active", False):
            return "outside managed sampler"
        if patcher is None or patcher.is_dynamic():
            return "non-dynamic model clone required"
        if getattr(patcher.model, "model_lowvram", False):
            return "model is partially loaded"
        if getattr(module, "training", True) or torch.is_grad_enabled():
            return "inference-only graph path"
        if (type(module).__module__, type(module).__name__) not in _MODEL_TYPES:
            return "model forward has not been qualified"
        if len(executor.wrappers) != 1:
            return "additional diffusion wrappers"
        if patcher.model_options.get("torch_compile_kwargs"):
            return "compile plus graph is not qualified"
        if patcher.hook_patches or patcher.weight_wrapper_patches:
            return "dynamic weight hooks"
        if options.get("patches") or options.get("patches_replace"):
            return "forward patches are not qualified"
        if options.get("prefetch_dynamic_vbars"):
            return "dynamic prefetch is not qualified"
        if any(key.startswith("diffusion_model.") and key != "diffusion_model.rope_embedder"
               for key in patcher.object_patches):
            return "diffusion object patches are not qualified"
        if type(module).__name__ == "NextDiT":
            if not isinstance(module.rope_embedder, ResidentEmbedND):
                return "native position constant adapter required"
            if any(type(value) not in (int, float) for value in options.get("rope_options", {}).values()):
                return "dynamic position options are not qualified"
        if os.environ.get("OMNI_ATTN_BACKEND", "auto") == "esimd":
            return "stateful ESIMD attention is not qualified"
        if os.environ.get("AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC", "0") != "0":
            return "allocation compiler combination is not qualified"
        from comfy_aimdo import control
        mode = getattr(control, "get_xpu_allocator_mode", lambda: None)()
        if mode not in (None, "native_hook"):
            return "native Torch caching allocator required"
        parameters = list(module.parameters())
        if not parameters or parameters[0].device.type != "xpu":
            return "XPU-resident parameters required"
        device = parameters[0].device
        if any(type(p) not in (torch.Tensor, torch.nn.Parameter)
               or p.device != device or p.dtype not in (torch.float16, torch.bfloat16, torch.float32)
               for p in parameters):
            return "fully resident FP16/BF16/FP32 parameters required"
        return None

    def diffusion_wrapper(self, executor, *args, **kwargs):
        bound = inspect.signature(executor.original).bind(*args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments.get("transformer_options", {})
        reason = self.eligibility(executor, options)
        if reason:
            self.reasons[reason] += 1
            self.stats["skipped"] += 1
            return executor(*args, **kwargs)
        module = executor.class_obj
        parameters = list(module.parameters())
        device = parameters[0].device
        if self.local.runtime is None:
            self.local.runtime = self.runtime_factory(device)
        patcher = self.patcher()
        identity = (id(module), str(patcher.patches_uuid),
                    str(getattr(patcher.model, "current_weight_patches_uuid", None)),
                    tuple(p.data_ptr() for p in parameters))
        if type(module).__name__ == "NextDiT":
            rope = module.rope_embedder.original
            identity += ((id(rope), rope.dim, rope.theta, tuple(rope.axes_dim),
                          module.patch_size, module.pad_tokens_multiple, module.masked_pad_multiple),)
        # These registries are consumed by public wrappers outside the captured
        # core. Keep tensor/control arguments and model math unchanged.
        options = dict(options)
        options.pop("wrappers", None)
        options.pop("callbacks", None)
        bound.arguments["transformer_options"] = options
        return self.local.runtime(executor, bound.args, bound.kwargs, identity=identity)


def _on_clone(parent, clone):
    state = clone.get_attachment(KEY)
    if state is not None:
        bind(clone, state)


def bind(model, state):
    from comfy.patcher_extension import CallbacksMP, WrappersMP
    state.patcher = weakref.ref(model)
    model.set_attachments(KEY, state)
    for wrapper_type, function in ((WrappersMP.DIFFUSION_MODEL, state.diffusion_wrapper),
                                   (WrappersMP.SAMPLER_SAMPLE, state.sampler_wrapper)):
        model.remove_wrappers_with_key(wrapper_type, KEY)
        model.add_wrapper_with_key(wrapper_type, KEY, function)
    model.remove_callbacks_with_key(CallbacksMP.ON_CLONE, KEY)
    model.add_callback_with_key(CallbacksMP.ON_CLONE, KEY, _on_clone)


def get_stats():
    with _STATS_LOCK:
        stats, reasons = _TOTAL_COUNTS.copy(), _TOTAL_REASONS.copy()
    for state in list(_STATES):
        runtime = getattr(state.local, "runtime", None)
        if runtime is not None:
            stats.update(runtime.stats)
            reasons.update(runtime.reasons)
    return {"counts": dict(stats), "reasons": dict(reasons)}
