"""Explicit MODEL node; original workflows keep their existing execution."""
from ..adapters.xpu_graph import ModelGraphState, bind
from ..graph_constants import add_native_position_patch


class OmniXPUGraph:
    CATEGORY = "OmniXPU/experimental"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = "Experimental resident XPU graph. Enabled and disabled modes both use a non-dynamic model clone."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "enabled": ("BOOLEAN", {"default": True})}}

    def patch(self, model, enabled):
        clone = model.clone(disable_dynamic=True)
        if enabled:
            add_native_position_patch(clone)
        bind(clone, ModelGraphState(clone, enabled=enabled))
        return (clone,)
