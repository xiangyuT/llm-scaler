"""Alias package: custom_esimd_kernels_sglang -> custom_esimd_kernels_vllm.

sglang's XPU paths import `custom_esimd_kernels_sglang` (rename landed in
sglang commit 081615ea), but the kernel package on disk is still named
`custom_esimd_kernels_vllm`. This shim re-exports the real package under the
sglang name so both import styles work without renaming/rebuilding the kernel
package. Drop-in until the kernel package itself is renamed.
"""
import custom_esimd_kernels_vllm as _impl
from custom_esimd_kernels_vllm import *  # noqa: F401,F403 — re-export all ops

# Preserve __all__ so `from custom_esimd_kernels_sglang import X` resolves.
__all__ = getattr(_impl, "__all__", [])

# Make `import custom_esimd_kernels_sglang.ops` resolve to the real ops module.
import sys as _sys
from custom_esimd_kernels_vllm import ops as _ops  # noqa: E402
_sys.modules[__name__ + ".ops"] = _ops
ops = _ops
