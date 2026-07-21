"""Route supported Lumina/Z-Image INT8 FFNs through fused Omni XPU kernels."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .debug import log_debug_event

log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_int8_ffn_original__"
_omni_int8 = None
_routed_calls = 0
_fallback_calls = 0
_fallback_reasons: dict[str, int] = {}


@dataclass(frozen=True)
class _Weight:
    qdata: torch.Tensor
    scale: torch.Tensor
    convrot: bool
    convrot_groupsize: int


def _module_weight(module: Any, x: torch.Tensor) -> tuple[_Weight | None, str]:
    """Extract an already-resident TensorWise INT8 weight without moving it."""
    weight = getattr(module, "weight", None)
    if weight is None:
        return None, "missing_weight"
    if getattr(module, "quant_format", None) != "int8_tensorwise":
        return None, "quant_format"
    if getattr(module, "layout_type", None) != "TensorWiseINT8Layout":
        return None, "layout"
    if getattr(module, "_full_precision_mm", False):
        return None, "full_precision_mm"
    if getattr(module, "comfy_force_cast_weights", False):
        return None, "force_cast_weights"
    if len(getattr(module, "weight_function", ())) != 0:
        return None, "weight_function"
    if len(getattr(module, "bias_function", ())) != 0:
        return None, "bias_function"
    if getattr(module, "bias", None) is not None:
        return None, "bias"
    if getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
        return None, "weight_layout"

    qdata = getattr(weight, "_qdata", None)
    params = getattr(weight, "_params", None)
    scale = getattr(params, "scale", None)
    if not isinstance(qdata, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return None, "weight_storage"
    if qdata.dtype != torch.int8 or qdata.ndim != 2:
        return None, "weight_storage"
    if qdata.device != x.device or scale.device != x.device:
        return None, "offloaded_weight"
    if getattr(params, "orig_dtype", None) != x.dtype:
        return None, "logical_dtype"
    if getattr(params, "transposed", False):
        return None, "transposed_weight"

    return (
        _Weight(
            qdata=qdata,
            scale=scale,
            convrot=bool(getattr(params, "convrot", False)),
            convrot_groupsize=int(getattr(params, "convrot_groupsize", 256)),
        ),
        "",
    )


def _route_inputs(
    module: Any, x: Any
) -> tuple[tuple[_Weight, _Weight, _Weight] | None, str]:
    return _route_named_inputs(module, x, ("w1", "w3", "w2"))


def _route_named_inputs(
    module: Any,
    x: Any,
    names: tuple[str, str, str],
) -> tuple[tuple[_Weight, _Weight, _Weight] | None, str]:
    if not isinstance(x, torch.Tensor):
        return None, "input_type"
    if x.device.type != "xpu":
        return None, "device"
    if x.dtype not in (torch.float16, torch.bfloat16):
        return None, "input_dtype"
    if x.ndim not in (2, 3) or x.shape[-1] == 0:
        return None, "input_shape"
    if x.requires_grad:
        return None, "requires_grad"

    weights = []
    for name in names:
        linear = getattr(module, name, None)
        if linear is None:
            return None, f"missing_{name}"
        extracted, reason = _module_weight(linear, x)
        if extracted is None:
            return None, f"{name}_{reason}"
        weights.append(extracted)

    w1, w3, w2 = weights
    input_features = x.shape[-1]
    if w1.qdata.shape[1] != input_features or w3.qdata.shape[1] != input_features:
        return None, "up_input_shape"
    if w1.qdata.shape[0] != w3.qdata.shape[0]:
        return None, "up_output_shape"
    if w2.qdata.shape != (input_features, w1.qdata.shape[0]):
        return None, "down_shape"
    if (w1.convrot, w1.convrot_groupsize) != (
        w3.convrot,
        w3.convrot_groupsize,
    ):
        return None, "up_convrot_mismatch"
    for name, weight in (("up", w1), ("down", w2)):
        if weight.convrot:
            size = weight.convrot_groupsize
            remaining = size
            while remaining > 1 and remaining % 4 == 0:
                remaining //= 4
            if size < 4 or remaining != 1:
                return None, f"{name}_convrot_groupsize"
            if weight.qdata.shape[1] % size != 0:
                return None, f"{name}_convrot_shape"

    return (w1, w3, w2), ""


def _route_qkv_inputs(
    module: Any,
    hidden_states: Any,
    encoder_hidden_states: Any,
    names: tuple[str, str, str] = ("to_q", "to_k", "to_v"),
) -> tuple[tuple[_Weight, _Weight, _Weight] | None, str]:
    if hidden_states is not encoder_hidden_states:
        return None, "cross_attention"
    if not isinstance(hidden_states, torch.Tensor):
        return None, "input_type"
    if hidden_states.device.type != "xpu":
        return None, "device"
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        return None, "input_dtype"
    if hidden_states.ndim not in (2, 3) or hidden_states.shape[-1] == 0:
        return None, "input_shape"
    if hidden_states.requires_grad:
        return None, "requires_grad"

    weights = []
    for name in names:
        linear = getattr(module, name, None)
        if linear is None:
            return None, f"missing_{name}"
        extracted, reason = _module_weight(linear, hidden_states)
        if extracted is None:
            return None, f"{name}_{reason}"
        weights.append(extracted)

    q, k, v = weights
    input_features = hidden_states.shape[-1]
    for name, weight in zip(names, weights):
        if weight.qdata.shape[1] != input_features:
            return None, f"{name}_input_shape"
    if (q.convrot, q.convrot_groupsize) != (
        k.convrot,
        k.convrot_groupsize,
    ) or (q.convrot, q.convrot_groupsize) != (
        v.convrot,
        v.convrot_groupsize,
    ):
        return None, "convrot_mismatch"
    if q.convrot:
        size = q.convrot_groupsize
        remaining = size
        while remaining > 1 and remaining % 4 == 0:
            remaining //= 4
        if size < 4 or remaining != 1:
            return None, "convrot_groupsize"
        if input_features % size != 0:
            return None, "convrot_shape"
    return (q, k, v), ""


def _record_fallback(reason: str) -> None:
    global _fallback_calls
    _fallback_calls += 1
    _fallback_reasons[reason] = _fallback_reasons.get(reason, 0) + 1


def get_stats() -> dict[str, Any]:
    return {
        "routed": _routed_calls,
        "fallback": _fallback_calls,
        "reasons": dict(_fallback_reasons),
    }


def _run_shared_qkv(
    x: torch.Tensor,
    weights: tuple[_Weight, _Weight, _Weight],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = weights[0]
    if first.convrot:
        x = _omni_int8.rotate_convrot(x, first.convrot_groupsize)
    x_q, x_scale = _omni_int8.quantize_int8_rowwise(x)
    return tuple(
        _omni_int8.int8_linear_prequantized(
            x_q,
            x_scale,
            weight.qdata,
            weight.scale,
            out_dtype=x.dtype,
        )
        for weight in weights
    )


def _install_omnigen_routes(comfy_ops: Any) -> int:
    """Install exact shared-quantization routes used by OmniGen2 and Boogu."""
    if not hasattr(_omni_int8, "int8_linear"):
        return 0
    try:
        import comfy.ldm.omnigen.omnigen2 as omnigen_model
    except ImportError:
        return 0

    installed = 0
    feed_forward = getattr(omnigen_model, "LuminaFeedForward", None)
    if feed_forward is not None and hasattr(feed_forward, "forward") and not hasattr(
        feed_forward.forward, _PATCH_MARKER
    ):
        original_forward = feed_forward.forward

        def _omnigen_ffn_forward(self, x):
            global _routed_calls

            weights, reason = _route_named_inputs(
                self, x, ("linear_1", "linear_3", "linear_2")
            )
            if weights is None:
                _record_fallback(f"omnigen_ffn_{reason}")
                return original_forward(self, x)

            w1, w3, w2 = weights
            comfy_ops.run_every_op()
            up1, up3 = _omni_int8.int8_linear_shared_input(
                x,
                w1.qdata,
                w1.scale,
                w3.qdata,
                w3.scale,
                out_dtype=x.dtype,
                convrot=w1.convrot,
                convrot_groupsize=w1.convrot_groupsize,
            )
            # Keep OmniGen2/Boogu's in-place SwiGLU boundary byte-for-byte.
            gated = F.silu(up1, inplace=True).mul_(up3)
            del up1, up3
            output = _omni_int8.int8_linear(
                gated,
                w2.qdata,
                w2.scale,
                out_dtype=x.dtype,
                convrot=w2.convrot,
                convrot_groupsize=w2.convrot_groupsize,
            )
            _routed_calls += 1
            log_debug_event(
                "kernel",
                "int8_omnigen_swiglu_mlp",
                {"input": x, "output": output},
                details={"route": "shared_up+exact_swiglu+down"},
            )
            return output

        setattr(_omnigen_ffn_forward, _PATCH_MARKER, original_forward)
        feed_forward.forward = _omnigen_ffn_forward
        installed += 1

    attention = getattr(omnigen_model, "Attention", None)
    if attention is not None and hasattr(attention, "forward") and not hasattr(
        attention.forward, _PATCH_MARKER
    ):
        original_attention_forward = attention.forward

        def _omnigen_attention_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask=None,
            image_rotary_emb=None,
            transformer_options={},
        ):
            global _routed_calls

            weights, reason = _route_qkv_inputs(
                self, hidden_states, encoder_hidden_states
            )
            if weights is None:
                _record_fallback(f"omnigen_qkv_{reason}")
                return original_attention_forward(
                    self,
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    image_rotary_emb,
                    transformer_options,
                )

            comfy_ops.run_every_op()
            query, key, value = _run_shared_qkv(hidden_states, weights)
            batch_size = hidden_states.shape[0]
            query = query.view(batch_size, -1, self.heads, self.dim_head)
            key = key.view(batch_size, -1, self.kv_heads, self.dim_head)
            value = value.view(batch_size, -1, self.kv_heads, self.dim_head)
            query = self.norm_q(query)
            key = self.norm_k(key)
            if image_rotary_emb is not None:
                query = omnigen_model.apply_rotary_emb(query, image_rotary_emb)
                key = omnigen_model.apply_rotary_emb(key, image_rotary_emb)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.kv_heads < self.heads:
                repeats = self.heads // self.kv_heads
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)
            output = omnigen_model.optimized_attention_masked(
                query,
                key,
                value,
                self.heads,
                attention_mask,
                skip_reshape=True,
                transformer_options=transformer_options,
            )
            output = self.to_out[0](output)
            _routed_calls += 1
            log_debug_event(
                "kernel",
                "int8_omnigen_qkv",
                {"input": hidden_states, "output": output},
                details={"route": "shared_convrot+quant+three_gemm"},
            )
            return output

        setattr(
            _omnigen_attention_forward,
            _PATCH_MARKER,
            original_attention_forward,
        )
        attention.forward = _omnigen_attention_forward
        installed += 1

    try:
        import comfy.ldm.boogu.model as boogu_model
    except ImportError:
        boogu_model = None
    joint_attention = (
        getattr(boogu_model, "BooguDoubleStreamProcessor", None)
        if boogu_model is not None
        else None
    )
    if (
        joint_attention is not None
        and hasattr(joint_attention, "forward")
        and not hasattr(joint_attention.forward, _PATCH_MARKER)
    ):
        original_joint_forward = joint_attention.forward

        def _boogu_joint_forward(
            self,
            attn,
            img_hidden_states,
            instruct_hidden_states,
            rotary_emb,
            attention_mask=None,
            transformer_options={},
        ):
            global _routed_calls

            img_weights, img_reason = _route_qkv_inputs(
                self,
                img_hidden_states,
                img_hidden_states,
                ("img_to_q", "img_to_k", "img_to_v"),
            )
            instruct_weights, instruct_reason = _route_qkv_inputs(
                self,
                instruct_hidden_states,
                instruct_hidden_states,
                ("instruct_to_q", "instruct_to_k", "instruct_to_v"),
            )
            if img_weights is None or instruct_weights is None:
                reason = img_reason if img_weights is None else instruct_reason
                _record_fallback(f"boogu_joint_qkv_{reason}")
                return original_joint_forward(
                    self,
                    attn,
                    img_hidden_states,
                    instruct_hidden_states,
                    rotary_emb,
                    attention_mask,
                    transformer_options,
                )

            comfy_ops.run_every_op()
            img_q, img_k, img_v = _run_shared_qkv(
                img_hidden_states, img_weights
            )
            instruct_q, instruct_k, instruct_v = _run_shared_qkv(
                instruct_hidden_states, instruct_weights
            )
            batch_size = img_hidden_states.shape[0]
            instruct_length = instruct_hidden_states.shape[1]
            query = torch.cat([instruct_q, img_q], dim=1)
            key = torch.cat([instruct_k, img_k], dim=1)
            value = torch.cat([instruct_v, img_v], dim=1)
            query = query.view(batch_size, -1, attn.heads, attn.dim_head)
            key = key.view(batch_size, -1, attn.kv_heads, attn.dim_head)
            value = value.view(batch_size, -1, attn.kv_heads, attn.dim_head)
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if rotary_emb is not None:
                query = boogu_model.apply_rotary_emb(query, rotary_emb)
                key = boogu_model.apply_rotary_emb(key, rotary_emb)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if attn.kv_heads < attn.heads:
                repeats = attn.heads // attn.kv_heads
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)
            output = boogu_model.optimized_attention_masked(
                query,
                key,
                value,
                attn.heads,
                attention_mask,
                skip_reshape=True,
                transformer_options=transformer_options,
            )
            instruct_output = self.instruct_out(output[:, :instruct_length])
            img_output = self.img_out(output[:, instruct_length:])
            output = torch.cat([instruct_output, img_output], dim=1)
            output = attn.to_out[0](output)
            _routed_calls += 1
            log_debug_event(
                "kernel",
                "int8_boogu_joint_qkv",
                {"image": img_hidden_states, "instruction": instruct_hidden_states},
                details={"route": "two_shared_convrot+quant+six_gemm"},
            )
            return output

        setattr(_boogu_joint_forward, _PATCH_MARKER, original_joint_forward)
        joint_attention.forward = _boogu_joint_forward
        installed += 1
    return installed


def apply():
    global _omni_int8

    try:
        from omni_xpu_kernel import int8 as _candidate
    except ImportError:
        return False, "omni_xpu_kernel.int8 not available"

    required = (
        "int8_linear_shared_input",
        "fused_silu_mul",
        "fused_silu_mul_quantize_rowwise",
        "rotate_convrot",
        "quantize_int8_rowwise",
        "int8_linear_prequantized",
    )
    missing = [name for name in required if not hasattr(_candidate, name)]
    if missing:
        return False, f"omni_xpu_kernel.int8 missing {', '.join(missing)}"

    try:
        import comfy.ops as comfy_ops
        import comfy.ldm.lumina.model as lumina_model
    except ImportError as exc:
        return False, f"Lumina FeedForward unavailable ({exc})"

    feed_forward = getattr(lumina_model, "FeedForward", None)
    if feed_forward is None or not hasattr(feed_forward, "forward"):
        return False, "Lumina FeedForward.forward not found"
    if hasattr(feed_forward.forward, _PATCH_MARKER):
        return True, ""

    _omni_int8 = _candidate
    original_forward = feed_forward.forward

    def _forward(self, x):
        global _routed_calls

        weights, reason = _route_inputs(self, x)
        if weights is None:
            _record_fallback(reason)
            log_debug_event(
                "dispatch",
                "lumina.FeedForward",
                {"input": x},
                details={"route": "comfy", "reason": reason},
                verbose_only=True,
            )
            return original_forward(self, x)

        w1, w3, w2 = weights
        comfy_ops.run_every_op()
        up1, up3 = _omni_int8.int8_linear_shared_input(
            x,
            w1.qdata,
            w1.scale,
            w3.qdata,
            w3.scale,
            out_dtype=x.dtype,
            convrot=w1.convrot,
            convrot_groupsize=w1.convrot_groupsize,
        )
        if w2.convrot:
            gated = _omni_int8.fused_silu_mul(up1, up3)
            del up1, up3
            rotated = _omni_int8.rotate_convrot(
                gated, w2.convrot_groupsize
            )
            del gated
            gated_q, gated_scale = _omni_int8.quantize_int8_rowwise(rotated)
            del rotated
            route = "shared_up+fused_swiglu+convrot+quant+prequant_down"
        else:
            gated_q, gated_scale = _omni_int8.fused_silu_mul_quantize_rowwise(
                up1, up3
            )
            del up1, up3
            route = "shared_up+fused_swiglu_quant+prequant_down"
        output = _omni_int8.int8_linear_prequantized(
            gated_q,
            gated_scale,
            w2.qdata,
            w2.scale,
            out_dtype=x.dtype,
        )
        _routed_calls += 1
        log_debug_event(
            "kernel",
            "int8_swiglu_mlp",
            {
                "input": x,
                "up_weight": w1.qdata,
                "gate_weight": w3.qdata,
                "down_weight": w2.qdata,
                "output": output,
            },
            details={
                "backend": "omni_xpu",
                "route": route,
                "up_convrot": w1.convrot,
                "down_convrot": w2.convrot,
            },
        )
        return output

    setattr(_forward, _PATCH_MARKER, original_forward)
    feed_forward.forward = _forward
    omnigen_routes = _install_omnigen_routes(comfy_ops)
    log.info(
        "[OmniXPU] INT8: routed eligible Lumina FFN and %d OmniGen2 paths",
        omnigen_routes,
    )
    return True, ""


__all__ = ["apply", "get_stats"]
