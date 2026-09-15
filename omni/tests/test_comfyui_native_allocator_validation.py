"""Host fixtures for the installed-image native allocator gate; no XPU work."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest


@pytest.fixture
def validator(monkeypatch):
    path = Path(__file__).parents[1] / "tools/validate_comfyui_image.py"
    spec = importlib.util.spec_from_file_location("native_allocator_validator_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "comfy_aimdo.control", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    return module


@pytest.fixture
def control():
    stats = {
        "tracked_alloc_calls": 3, "tracked_alloc_bytes": 4096,
        "runtime_oom_calls": 0, "unknown_device_calls": 0,
        "unknown_free_calls": 0, "dropped_metadata_calls": 0,
        "duplicate_pointer_calls": 0,
    }
    return SimpleNamespace(
        get_xpu_allocator_mode=lambda: "native_hook", _xpu_allocator_ready=True,
        _torch_allocator=None, lib=SimpleNamespace(xpu_ur_hook_is_interposed=lambda: True),
        get_xpu_ur_hook_stats=lambda: dict(stats), stats=stats,
    )


def test_native_gate_accepts_observed_hooks_without_torch_allocator_takeover(validator, control):
    assert validator.require_aimdo_native_allocator(control) == control.stats


@pytest.mark.parametrize("field,value,diagnostic", [
    ("get_xpu_allocator_mode", lambda: "global", "allocator mode"),
    ("_xpu_allocator_ready", False, "not ready"),
    ("lib", None, "not loaded"),
    ("_torch_allocator", object(), "retain Torch"),
    ("_torch_xpu_empty_cache_original", object(), "retain Torch"),
    ("_torch_xpu_memory_stats_original", object(), "retain Torch"),
    ("_torch_xpu_reset_peak_stats_original", object(), "retain Torch"),
    ("lib", SimpleNamespace(xpu_ur_hook_is_interposed=lambda: False), "not interposed"),
    ("lib", SimpleNamespace(), "not interposed"),
    ("get_xpu_ur_hook_stats", None, "statistics API is missing"),
    ("get_xpu_ur_hook_stats", lambda: {}, "incomplete or invalid"),
])
def test_native_gate_rejects_missing_or_incompatible_runtime(validator, control, field, value, diagnostic):
    setattr(control, field, value)
    with pytest.raises(RuntimeError, match=diagnostic):
        validator.require_aimdo_native_allocator(control)


@pytest.mark.parametrize("counter", [
    "runtime_oom_calls", "dropped_metadata_calls", "duplicate_pointer_calls",
])
def test_native_hook_errors_cannot_pass(validator, control, counter):
    control.stats[counter] = 1
    with pytest.raises(RuntimeError, match=counter):
        validator.require_aimdo_native_allocator(control)


def test_successful_untracked_forwarding_is_observed_without_rejection(validator, control):
    control.stats.update(unknown_device_calls=2, unknown_free_calls=7)
    observed = validator.require_aimdo_native_allocator(control)
    assert observed["unknown_device_calls"] == 2
    assert observed["unknown_free_calls"] == 7


@pytest.mark.parametrize("counter", ["unknown_device_calls", "unknown_free_calls"])
def test_forwarding_counters_still_require_valid_native_integer_fields(validator, control, counter):
    del control.stats[counter]
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        validator.require_aimdo_native_allocator(control)
    control.stats[counter] = -1
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        validator.require_aimdo_native_allocator(control)


@pytest.mark.parametrize("value", [-1, 1.5, "1", True, None])
def test_invalid_hook_counter_values_cannot_pass(validator, control, value):
    control.stats["tracked_alloc_calls"] = value
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        validator.require_aimdo_native_allocator(control)


def test_hook_counter_query_errors_propagate(validator, control):
    control.get_xpu_ur_hook_stats = mock.Mock(side_effect=RuntimeError("native query failed"))
    with pytest.raises(RuntimeError, match="native query failed"):
        validator.require_aimdo_native_allocator(control)


def test_audio_gate_requires_both_tracked_calls_and_bytes_to_advance(validator):
    before = {"tracked_alloc_calls": 2, "tracked_alloc_bytes": 2048}
    validator.require_aimdo_tracked_allocation(before, {"tracked_alloc_calls": 3, "tracked_alloc_bytes": 4096})
    for after in (before, {"tracked_alloc_calls": 3, "tracked_alloc_bytes": 2048},
                  {"tracked_alloc_calls": 2, "tracked_alloc_bytes": 4096}):
        with pytest.raises(RuntimeError, match="did not track"):
            validator.require_aimdo_tracked_allocation(before, after)


def test_cli_prepares_verified_preload_before_provider_initialization(validator, monkeypatch, tmp_path):
    library = tmp_path / "verified-provider.so"
    library.write_bytes(b"fixture identity; never loaded")
    arguments = ["/llm/tools/validate_comfyui_image.py", "--allow-dirty-source"]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-B", *arguments])
    monkeypatch.setenv("LD_PRELOAD", "/existing/observer.so")
    class Restart(BaseException):
        pass
    with mock.patch.object(validator.subprocess, "run", return_value=subprocess.CompletedProcess(
            [], 0, str(library) + "\n", "")) as resolver, \
            mock.patch.object(validator.os, "execvpe", side_effect=Restart) as restart, \
            mock.patch.object(validator, "activate_runtime_providers") as activate:
        with pytest.raises(Restart):
            validator.main()
    resolver.assert_called_once_with(
        [sys.executable, str(validator.OMNIXPU_RUNTIME_BOOTSTRAP), "--native-preload-path"],
        check=True, capture_output=True, text=True,
    )
    executable, argv, environment = restart.call_args.args
    assert executable == sys.executable
    assert argv == [sys.executable, "-B", *arguments]
    assert environment["LD_PRELOAD"] == str(library) + ":/existing/observer.so"
    activate.assert_not_called()


def test_preloaded_cli_does_not_restart_but_environment_alone_cannot_pass(validator, control, monkeypatch, tmp_path):
    library = tmp_path / "verified-provider.so"
    library.write_bytes(b"fixture identity; never loaded")
    monkeypatch.setenv("LD_PRELOAD", str(library))
    with mock.patch.object(validator.subprocess, "run", return_value=subprocess.CompletedProcess(
            [], 0, str(library) + "\n", "")), mock.patch.object(validator.os, "execvpe") as restart:
        validator.prepare_native_allocator_preload()
    restart.assert_not_called()
    control.lib.xpu_ur_hook_is_interposed = lambda: False
    with pytest.raises(RuntimeError, match="not interposed"):
        validator.require_aimdo_native_allocator(control)


def test_resolver_failure_is_fatal_without_a_global_fallback(validator, monkeypatch):
    monkeypatch.setenv("AIMDO_XPU_ALLOCATOR_MODE", "global")
    with mock.patch.object(validator.subprocess, "run", side_effect=subprocess.CalledProcessError(
            1, ["runtime_bootstrap.py", "--native-preload-path"])), \
            mock.patch.object(validator.os, "execvpe") as restart:
        with pytest.raises(subprocess.CalledProcessError):
            validator.prepare_native_allocator_preload()
    restart.assert_not_called()


@pytest.mark.parametrize("response", ["empty", "multiline", "missing", "relative"])
def test_invalid_resolver_output_never_restarts_or_initializes_runtime(validator, tmp_path, response):
    library = tmp_path / "provider.so"
    library.write_bytes(b"fixture identity; never loaded")
    stdout = {
        "empty": "\n", "multiline": str(library) + "\n" + str(library) + "\n",
        "missing": str(tmp_path / "missing.so") + "\n", "relative": "provider.so\n",
    }[response]
    with mock.patch.object(validator.subprocess, "run", return_value=subprocess.CompletedProcess(
            [], 0, stdout, "")), mock.patch.object(validator.os, "execvpe") as restart:
        with pytest.raises(RuntimeError, match="verified library path"):
            validator.prepare_native_allocator_preload()
    restart.assert_not_called()


@pytest.mark.parametrize("loaded", ["torch", "comfy_aimdo.control"])
def test_preload_is_not_retrofitted_after_runtime_initialization(validator, monkeypatch, loaded):
    monkeypatch.setitem(sys.modules, loaded, SimpleNamespace())
    with mock.patch.object(validator.subprocess, "run") as resolver:
        with pytest.raises(RuntimeError, match="before AIMDO or Torch"):
            validator.prepare_native_allocator_preload()
    resolver.assert_not_called()


def test_kitchen_capabilities_return_the_observed_complete_set(validator):
    observed = {"dequantize_gguf", "dequantize_int8_simple", "dequantize_int8_simple_dtype",
        "int8_linear", "mm_int8", "quantize_int8_rowwise", "quantize_int8_tensorwise",
        "svdquant_w4a16_linear", "sol_attn", "additional_installed_capability"}
    assert validator.require_kitchen_xpu_capabilities({"available": True, "capabilities": list(observed)}) == observed
