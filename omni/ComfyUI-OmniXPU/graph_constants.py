"""Sampler-owned native NextDiT position constants, using public object patches.

Native EmbedND uses CPU FP64 on devices without FP64. Keep that exact math
outside capture. Qualified NextDiT constructs IDs from shapes and constant
options, so their outputs can be retained for the corresponding graph key.
"""
import torch

from .graph_runtime import get_constant_call, input_signature


class ResidentEmbedND(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.eval()

    def forward(self, ids):
        call = get_constant_call()
        if call is None:
            return self.original(ids)
        index = call.index
        call.index += 1
        contract = (input_signature(ids, torch), self.original.dim,
                    self.original.theta, tuple(self.original.axes_dim))
        if call.phase == "warmup":
            output = self.original(ids)
            snapshot = ids.detach().cpu().clone()
            if index == len(call.values):
                call.values.append([contract, snapshot, output, True])
            else:
                saved = call.values[index]
                saved[3] = saved[3] and contract == saved[0] and torch.equal(snapshot, saved[1])
            return output
        if index >= len(call.values):
            raise RuntimeError("native position constant was not warmed")
        contract_before, _, output, valid = call.values[index]
        if not valid or contract != contract_before:
            raise RuntimeError("native position constants changed during warmup")
        return output


def add_native_position_patch(model):
    """Only the exact native module contract is qualified; never mutate source."""
    module = getattr(model.model, "diffusion_model", None)
    if (type(module).__module__, type(module).__name__) != ("comfy.ldm.lumina.model", "NextDiT"):
        return
    path = "diffusion_model.rope_embedder"
    if path in model.object_patches:
        return
    original = module.rope_embedder
    if (type(original).__module__, type(original).__name__) == ("comfy.ldm.flux.layers", "EmbedND"):
        model.add_object_patch(path, ResidentEmbedND(original))
