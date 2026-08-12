"""Tests for the focused ComfyUI image entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ENTRYPOINT = Path(__file__).parents[1] / "entrypoints" / "start_comfyui.sh"


def _run_entrypoint(
    tmp_path: Path,
    *,
    reserve: str | None = None,
    extra_arguments: tuple[str, ...] = ("--disable-all-custom-nodes",),
):
    capture = tmp_path / "args.txt"
    environment_capture = tmp_path / "environment.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$OMNI_TEST_CAPTURE\"\n"
        "printf 'AIMDO_XPU_ALLOCATOR_MODE=%s\\nLD_PRELOAD=%s\\n' "
        "\"${AIMDO_XPU_ALLOCATOR_MODE:-}\" \"${LD_PRELOAD:-}\" "
        "> \"$OMNI_TEST_ENVIRONMENT_CAPTURE\"\n"
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}:{environment['PATH']}"
    environment["OMNI_TEST_CAPTURE"] = str(capture)
    environment["OMNI_TEST_ENVIRONMENT_CAPTURE"] = str(environment_capture)
    environment["OMNI_COMFY_AIMDO_LIBRARY"] = "/lib/x86_64-linux-gnu/libc.so.6"
    environment.pop("AIMDO_XPU_ALLOCATOR_MODE", None)
    environment.pop("LD_PRELOAD", None)
    if reserve is None:
        environment.pop("OMNI_COMFYUI_RESERVE_VRAM_GB", None)
    else:
        environment["OMNI_COMFYUI_RESERVE_VRAM_GB"] = reserve
    completed = subprocess.run(
        ["bash", str(ENTRYPOINT), *extra_arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    arguments = capture.read_text().splitlines() if capture.exists() else []
    captured_environment = (
        environment_capture.read_text().splitlines()
        if environment_capture.exists()
        else []
    )
    return completed, arguments, captured_environment


def test_entrypoint_reserves_four_gib_by_default(tmp_path):
    completed, arguments, environment = _run_entrypoint(tmp_path)

    assert completed.returncode == 0
    assert arguments == [
        "/llm/ComfyUI/main.py",
        "--listen",
        "0.0.0.0",
        "--port",
        "8188",
        "--reserve-vram",
        "4",
        "--enable-dynamic-vram",
        "--enable-manager",
        "--disable-all-custom-nodes",
    ]
    assert environment == [
        "AIMDO_XPU_ALLOCATOR_MODE=native_hook",
        "LD_PRELOAD=/lib/x86_64-linux-gnu/libc.so.6",
    ]


def test_entrypoint_loads_custom_nodes_by_default(tmp_path):
    completed, arguments, _ = _run_entrypoint(tmp_path, extra_arguments=())

    assert completed.returncode == 0
    assert "--enable-manager" in arguments
    assert "--enable-dynamic-vram" in arguments
    assert "--disable-all-custom-nodes" not in arguments


def test_entrypoint_allows_explicit_reserve_override(tmp_path):
    completed, arguments, _ = _run_entrypoint(tmp_path, reserve="6.5")

    assert completed.returncode == 0
    reserve_index = arguments.index("--reserve-vram")
    assert arguments[reserve_index + 1] == "6.5"


def test_entrypoint_rejects_invalid_reserve(tmp_path):
    completed, arguments, environment = _run_entrypoint(tmp_path, reserve="four")

    assert completed.returncode == 2
    assert arguments == []
    assert environment == []
    assert "must be a nonnegative number" in completed.stderr


def test_entrypoint_rejects_missing_aimdo_library(tmp_path):
    missing = tmp_path / "missing-aimdo-xpu.so"
    process_environment = os.environ.copy()
    process_environment["PATH"] = f"{tmp_path}:{process_environment['PATH']}"
    process_environment["OMNI_TEST_CAPTURE"] = str(tmp_path / "missing-args.txt")
    process_environment["OMNI_TEST_ENVIRONMENT_CAPTURE"] = str(
        tmp_path / "missing-environment.txt"
    )
    process_environment["OMNI_COMFY_AIMDO_LIBRARY"] = str(missing)

    completed = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        check=False,
        capture_output=True,
        text=True,
        env=process_environment,
    )

    assert completed.returncode == 2
    assert not (tmp_path / "missing-args.txt").exists()
    assert not (tmp_path / "missing-environment.txt").exists()
    assert "AIMDO XPU library not found" in completed.stderr
