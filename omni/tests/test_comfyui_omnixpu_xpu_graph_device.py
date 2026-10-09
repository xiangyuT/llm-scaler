"""Real XPU capture and public native-model wrapper integration (opt-in device)."""
import importlib
import os
from pathlib import Path
import sys
import types

import pytest
import torch

PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
COMFY = Path(os.environ.get("OMNIXPU_TEST_COMFYUI_ROOT", "/llm/ComfyUI"))


@pytest.fixture(scope="module")
def live():
    if os.environ.get("OMNIXPU_GRAPH_DEVICE_TESTS", "0") != "1":
        pytest.skip("real device cases require the explicit graph test launcher")
    if not torch.xpu.is_available():
        pytest.skip("real XPU runtime required")
    sys.path.insert(0, str(COMFY))
    bootstrap = sys.modules.get("_comfyui_omnixpu_runtime_bootstrap")
    if bootstrap is None:
        pytest.fail("use launch_xpu_graph_tests.py to activate native_hook before importing Torch")
    assert bootstrap.get_state()["providers"]["comfy_aimdo.xpu"]["status"] == "active"
    assert torch.xpu.device_count() == 1
    props = torch.xpu.get_device_properties(0)
    assert props.device_id == 0xE223
    assert str(props.uuid) == "868023e2-0000-0000-cc00-000000000000"
    torch.xpu.set_device(0)
    package = types.ModuleType("omnixpu_graph_device")
    package.__path__ = [str(PLUGIN)]
    sys.modules[package.__name__] = package
    runtime = importlib.import_module(package.__name__ + ".graph_runtime")
    adapter = importlib.import_module(package.__name__ + ".adapters.xpu_graph")
    node = importlib.import_module(package.__name__ + ".nodes.xpu_graph")
    assert adapter.apply()[0]
    return runtime, adapter, node


@pytest.mark.parametrize("operation", ["pointwise", "cute"])
def test_native_xpu_capture_replay_changes_inputs_and_preserves_outputs(live, operation):
    runtime, _, _ = live
    device = torch.device("xpu", 0)
    if operation == "pointwise":
        function = lambda x: torch.sin(x * 2.0) + x
        shape, dtype = (8, 64), torch.float32
    else:
        from omni_xpu_kernel import cute
        function = lambda x: cute.sdp_bhld_d128(x, x, x)
        shape, dtype = (1, 32, 256, 128), torch.bfloat16
    with torch.inference_mode():
        engine = runtime.GraphRuntime(torch, runtime.XPUBackend(torch, device))
        retained = []
        try:
            for index in range(6):
                x = torch.randn(shape, dtype=dtype, device=device) * 0.1
                expected = function(x)
                actual = engine(function, (x,), {})
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                retained.append((actual, actual.cpu().clone()))
            assert engine.stats["captures"] == 1
            assert engine.stats["replays"] == 3
            for value, snapshot in retained:
                torch.testing.assert_close(value.cpu(), snapshot, rtol=0, atol=0)
        finally:
            engine.close()
        assert not engine.entries


def test_native_unet_public_sampler_wrappers_and_clone(live):
    _, adapter, node = live
    import comfy.ops
    from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor, WrappersMP
    device = torch.device("xpu", 0)
    with torch.inference_mode():
        unet = UNetModel(
            image_size=8, in_channels=4, model_channels=32, out_channels=4,
            num_res_blocks=1, channel_mult=[1],
            num_heads=4, use_spatial_transformer=True, context_dim=32,
            transformer_depth=[0], transformer_depth_middle=0,
            transformer_depth_output=[0, 0], dtype=torch.float32,
            device=device, operations=comfy.ops.disable_weight_init,
        ).eval()
        for parameter in unet.parameters():
            parameter.uniform_(-0.02, 0.02)
        parent = torch.nn.Module()
        parent.diffusion_model = unet
        parent.model_lowvram = False
        parent.device = device
        patcher = ModelPatcher(parent, device, torch.device("cpu"),
                               size=sum(p.numel() * p.element_size() for p in unet.parameters()))
        model = node.OmniXPUGraph().patch(patcher, True)[0]
        state = model.get_attachment(adapter.KEY)
        cloned = model.clone()
        assert cloned.get_attachment(adapter.KEY) is not state
        options = {"wrappers": model.wrappers}
        def sampler():
            for index in range(6):
                x = torch.randn((2, 4, 8, 8), device=device)
                t = torch.ones(2, device=device) * (index + 1)
                expected = unet._forward(x, t, transformer_options={})
                actual = unet(x, t, transformer_options=options)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        executor = WrapperExecutor.new_executor(sampler, model.get_all_wrappers(WrappersMP.SAMPLER_SAMPLE))
        executor.execute()
        assert state.stats["captures"] == 1, dict(state.reasons)
        assert state.stats["replays"] == 3
        assert state.local.runtime is None and not state.local.active
        model.cleanup()


def test_native_nextdit_public_position_patch(live):
    _, adapter, node = live
    import comfy.ops
    from comfy.ldm.lumina.model import NextDiT
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor, WrappersMP
    device = torch.device('xpu', 0)
    with torch.inference_mode():
        module = NextDiT(in_channels=4, dim=256, n_layers=1, n_refiner_layers=1,
                         n_heads=8, multiple_of=32, cap_feat_dim=32,
                         axes_dims=[8, 12, 12], qk_norm=True,
                         z_image_modulation=True, pad_tokens_multiple=8,
                         dtype=torch.bfloat16, device=device,
                         operations=comfy.ops.disable_weight_init).eval()
        for parameter in module.parameters():
            parameter.uniform_(-0.02, 0.02)
        parent = torch.nn.Module()
        parent.diffusion_model, parent.model_lowvram = module, False
        parent.device = device
        patcher = ModelPatcher(parent, device, torch.device('cpu'),
                               size=sum(p.numel() * p.element_size() for p in module.parameters()))
        model = node.OmniXPUGraph().patch(patcher, True)[0]
        model.patch_model(load_weights=False)
        state = model.get_attachment(adapter.KEY)
        options = {'wrappers': model.wrappers}
        def sampler():
            for index in range(6):
                x = torch.randn((1, 4, 8, 8), device=device, dtype=torch.bfloat16)
                context = torch.randn((1, 16, 32), device=device, dtype=torch.bfloat16)
                t = torch.ones(1, device=device) * (index + 1) / 8
                expected = module._forward(x, t, context, 16, transformer_options={})
                actual = module(x, t, context, 16, transformer_options=options)
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        try:
            WrapperExecutor.new_executor(sampler, model.get_all_wrappers(WrappersMP.SAMPLER_SAMPLE)).execute()
            assert state.stats['captures'] == 1, dict(state.reasons)
            assert state.stats['replays'] == 3
            assert state.stats['position_constants'] == 2
            assert state.local.runtime is None
        finally:
            model.unpatch_model(unpatch_weights=False)


def test_existing_zimage_bf16_forward_replay_numerics(live):
    model_path = os.environ.get('OMNIXPU_GRAPH_TEST_MODEL')
    if not model_path:
        pytest.skip('existing BF16 weight path must be selected explicitly')
    # Activate the actual source-overlay plugin adapters, matching the workflow.
    spec = importlib.util.spec_from_file_location('ComfyUI-OmniXPU', PLUGIN / '__init__.py',
                                                 submodule_search_locations=[str(PLUGIN)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    adapter = importlib.import_module(spec.name + '.adapters.xpu_graph')
    node = package.NODE_CLASS_MAPPINGS['OmniXPUGraph']
    import comfy.sd
    import comfy.model_management
    from comfy.patcher_extension import WrapperExecutor, WrappersMP
    import json
    device = torch.device('xpu', 0)
    records = []
    with torch.inference_mode():
        model = node().patch(comfy.sd.load_diffusion_model(model_path, disable_dynamic=True), True)[0]
        comfy.model_management.load_models_gpu([model], force_full_load=True)
        module = model.model.diffusion_model
        assert module.in_channels == 16
        width = module.cap_embedder[-1].in_features
        state = model.get_attachment(adapter.KEY)
        options = {'wrappers': model.wrappers}
        def sampler():
            for index in range(6):
                generator = torch.Generator(device=device).manual_seed(39110 + index)
                x = torch.randn((1, 16, 128, 128), generator=generator, device=device, dtype=torch.bfloat16) * 0.1
                context = torch.randn((1, 32, width), generator=generator, device=device, dtype=torch.bfloat16) * 0.1
                t = torch.ones(1, device=device) * (0.9 - index / 8)
                expected = module._forward(x, t, context, 32, transformer_options={})
                repeated = module._forward(x, t, context, 32, transformer_options={})
                actual = module(x, t, context, 32, transformer_options=options)
                delta = (actual.float() - expected.float()).abs()
                calibration = (repeated.float() - expected.float()).abs()
                records.append({'seed': 39110 + index, 'max_abs': delta.max().item(),
                                'rms': delta.square().mean().sqrt().item(),
                                'eager_repeat_max_abs': calibration.max().item(),
                                'eager_repeat_rms': calibration.square().mean().sqrt().item()})
                torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
        try:
            WrapperExecutor.new_executor(sampler, model.get_all_wrappers(WrappersMP.SAMPLER_SAMPLE)).execute()
            assert state.stats['captures'] == 1, dict(state.reasons)
            assert state.stats['replays'] == 3
            assert state.stats['position_constants'] == 2
            assert state.local.runtime is None
        finally:
            output = os.environ.get('OMNIXPU_GRAPH_NUMERICS_OUTPUT')
            if output:
                Path(output).write_text(json.dumps({'records': records, 'counts': dict(state.stats)}, indent=2) + '\n')
            model.cleanup()
            comfy.model_management.unload_all_models()
