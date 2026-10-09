"""Match ComfyUI pre-device startup before collecting real-XPU graph tests.

Run under the image runtime entrypoint with the provider's verified LD_PRELOAD.
Device mapping and idle admission belong to the invoking development runner.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--junitxml", required=True)
    args = parser.parse_args()
    if not os.environ.get("LD_PRELOAD"):
        raise RuntimeError("verified native-hook preload is required")
    comfy = Path(os.environ.get("OMNIXPU_TEST_COMFYUI_ROOT", "/llm/ComfyUI"))
    sys.path.insert(0, str(comfy))
    sys.argv = ["main.py", "--enable-dynamic-vram", "--reserve-vram", "4"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import comfy_aimdo.control
    assert "torch" not in sys.modules, "allocator takeover must precede Torch import"
    plugin = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
    runpy.run_path(str(plugin / "prestartup_script.py"))
    import torch
    assert torch.xpu.device_count() == 1
    props = torch.xpu.get_device_properties(0)
    assert props.device_id == 0xE223
    assert str(props.uuid) == "868023e2-0000-0000-cc00-000000000000"
    assert comfy_aimdo.control.init_devices([(0, 4 * 1024**3)])
    os.environ["OMNIXPU_GRAPH_DEVICE_TESTS"] = "1"
    import pytest
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider",
                                 str(Path(__file__).with_name("test_comfyui_omnixpu_xpu_graph_device.py")),
                                 "--junitxml=" + args.junitxml]))


if __name__ == "__main__":
    main()
