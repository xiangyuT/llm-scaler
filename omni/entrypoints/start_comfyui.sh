#!/usr/bin/env bash
set -euo pipefail

export no_proxy="${no_proxy:-localhost,127.0.0.1}"

default_aimdo_library=/opt/venv/lib/python3.12/site-packages/comfy_aimdo/aimdo_xpu.so
aimdo_library="${OMNI_COMFY_AIMDO_LIBRARY:-$default_aimdo_library}"
if [[ ! -f "$aimdo_library" ]]; then
    echo "AIMDO XPU library not found: $aimdo_library" >&2
    exit 2
fi
export AIMDO_XPU_ALLOCATOR_MODE=native_hook
export LD_PRELOAD="$aimdo_library${LD_PRELOAD:+:$LD_PRELOAD}"

# Keep runtime VRAM headroom configurable for model switching.
reserve_vram_gb="${OMNI_COMFYUI_RESERVE_VRAM_GB:-4}"
if [[ ! "$reserve_vram_gb" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "OMNI_COMFYUI_RESERVE_VRAM_GB must be a nonnegative number" >&2
    exit 2
fi

exec python /llm/ComfyUI/main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --reserve-vram "$reserve_vram_gb" \
    --enable-dynamic-vram \
    --enable-manager \
    "$@"
