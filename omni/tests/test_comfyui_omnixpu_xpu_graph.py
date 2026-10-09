"""CPU protocol checks; real capture/device evidence is a separate phase."""
from contextlib import contextmanager
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"


@pytest.fixture
def modules(monkeypatch):
    name = "omnixpu_graph_test"
    package = types.ModuleType(name)
    package.__path__ = [str(PLUGIN)]
    monkeypatch.setitem(sys.modules, name, package)
    for key in list(sys.modules):
        if key.startswith(name + "."):
            monkeypatch.delitem(sys.modules, key)
    return (importlib.import_module(name + ".graph_runtime"),
            importlib.import_module(name + ".adapters.xpu_graph"))


class FakeGraph:
    def __init__(self, backend):
        self.backend = backend
        self.operation = None

    def replay(self):
        self.backend.events.append("replay")
        if self.backend.replay_error:
            raise self.backend.replay_error
        self.operation()

    def reset(self):
        self.backend.events.append("reset")


class FakeBackend:
    def __init__(self, capture_error=None, replay_error=None, cleanup_error=None):
        self.capture_error, self.replay_error = capture_error, replay_error
        self.cleanup_error = cleanup_error
        self.graph = None
        self.events = []

    @contextmanager
    def execution(self, inputs):
        yield

    @contextmanager
    def capture(self, graph):
        if self.capture_error:
            raise self.capture_error
        self.graph = graph
        try:
            yield
        finally:
            self.graph = None

    def new_graph(self):
        return FakeGraph(self)

    def quiesce(self):
        self.events.append("wait")

    def recover(self, graph):
        if self.cleanup_error:
            raise self.cleanup_error
        self.quiesce()
        graph.reset()

    def retain_outputs(self, output):
        return output


class Core:
    def __init__(self, backend):
        self.backend, self.calls = backend, 0

    def __call__(self, x, scale=2.0, transformer_options=None):
        self.calls += 1
        output = x * scale
        if self.backend.graph:
            self.backend.graph.operation = lambda: output.copy_(x * scale)
        return output


def runtime(modules, **options):
    backend = FakeBackend()
    engine = modules[0].GraphRuntime(torch, backend, **options)
    return engine, backend, Core(backend)


def test_real_warm_calls_then_replay_owns_output_snapshots(modules):
    engine, backend, core = runtime(modules)
    for index in range(3):
        assert torch.equal(engine(core, (torch.ones(4) * index,), {}), torch.ones(4) * index * 2)
    first = engine(core, (torch.ones(4) * 3,), {})
    later = engine(core, (torch.ones(4) * 7,), {})
    assert torch.equal(first, torch.ones(4) * 6)
    assert torch.equal(later, torch.ones(4) * 14)
    assert first.data_ptr() != later.data_ptr()
    assert core.calls == 4
    assert engine.stats["captures"] == 1 and engine.stats["replays"] == 2
    engine.close()
    assert backend.events[-2:] == ["wait", "reset"]


@pytest.mark.parametrize("change", ["shape", "dtype", "constant", "identity"])
def test_signature_and_weight_identity_recapture(modules, change):
    engine, _, core = runtime(modules, warmup_calls=1)
    engine(core, (torch.ones(4),), {}, identity=("weights-a",))
    engine(core, (torch.ones(4),), {}, identity=("weights-a",))
    x, kwargs, identity = torch.ones(4), {}, ("weights-a",)
    if change == "shape":
        x = torch.ones(5)
    elif change == "dtype":
        x = x.double()
    elif change == "constant":
        kwargs = {"scale": 3.0}
    else:
        identity = ("weights-b",)
    engine(core, (x,), kwargs, identity=identity)
    engine(core, (x,), kwargs, identity=identity)
    assert engine.stats["captures"] == 2
    engine.close()


def test_cache_eviction_waits_before_reset(modules):
    engine, backend, core = runtime(modules, warmup_calls=1, max_entries=1)
    for size in (4, 5):
        engine(core, (torch.ones(size),), {})
        engine(core, (torch.ones(size),), {})
    assert engine.stats["evictions"] == 1
    reset = backend.events.index("reset")
    assert backend.events[reset - 1] == "wait"
    engine.close()


def test_capture_failure_recovers_once_then_keeps_eager(modules):
    backend = FakeBackend(capture_error=RuntimeError("unsupported graph operation"))
    engine = modules[0].GraphRuntime(torch, backend, warmup_calls=1)
    core = Core(backend)
    for _ in range(3):
        assert torch.equal(engine(core, (torch.ones(4),), {}), torch.ones(4) * 2)
    assert engine.stats["capture_failures"] == 1
    assert backend.events == ["wait", "reset"]
    engine.close()


@pytest.mark.parametrize("error", [RuntimeError("device replay failed"), torch.OutOfMemoryError("out of memory")])
def test_replay_errors_are_not_eager_retries(modules, error):
    backend = FakeBackend(replay_error=error)
    engine = modules[0].GraphRuntime(torch, backend, warmup_calls=1)
    core = Core(backend)
    engine(core, (torch.ones(4),), {})
    with pytest.raises(type(error)):
        engine(core, (torch.ones(4),), {})
    assert core.calls == 2
    engine.close()


def test_oom_during_capture_is_not_retried(modules):
    backend = FakeBackend(capture_error=torch.OutOfMemoryError("out of memory"))
    engine = modules[0].GraphRuntime(torch, backend, warmup_calls=1)
    core = Core(backend)
    engine(core, (torch.ones(4),), {})
    with pytest.raises(torch.OutOfMemoryError):
        engine(core, (torch.ones(4),), {})
    assert core.calls == 1
    engine.close()


def test_failed_cleanup_stops_runtime(modules):
    backend = FakeBackend(capture_error=RuntimeError("unsupported"), cleanup_error=RuntimeError("still recording"))
    engine = modules[0].GraphRuntime(torch, backend, warmup_calls=1)
    core = Core(backend)
    engine(core, (torch.ones(4),), {})
    with pytest.raises(RuntimeError, match="restart"):
        engine(core, (torch.ones(4),), {})
    with pytest.raises(RuntimeError, match="closed"):
        engine(core, (torch.ones(4),), {})


def test_unknown_objects_do_not_enter_capture(modules):
    engine, backend, _ = runtime(modules, warmup_calls=1)
    marker = object()
    for _ in range(3):
        assert engine(lambda value: value, (marker,), {}) is marker
    assert backend.events == [] and engine.stats["captures"] == 0
    engine.close()


def test_sampler_finally_closes_graph_on_cancellation(modules):
    _, adapter = modules
    backend = FakeBackend()
    state = adapter.ModelGraphState(runtime_factory=lambda device: None)
    engine = modules[0].GraphRuntime(torch, backend, warmup_calls=1)
    core = Core(backend)
    def sampler():
        state.local.runtime = engine
        engine(core, (torch.ones(4),), {})
        engine(core, (torch.ones(4),), {})
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        state.sampler_wrapper(sampler)
    assert backend.events[-2:] == ["wait", "reset"]
    assert not state.local.active and state.local.runtime is None
    assert state.stats["captures"] == 1


def test_model_clone_does_not_share_runtime(modules):
    state = modules[1].ModelGraphState(enabled=False)
    clone = state.on_model_patcher_clone()
    assert clone is not state and clone.local is not state.local
    assert not clone.enabled and clone.patcher is None


def test_positional_transformer_options_bind_without_duplicate_kwargs(modules, monkeypatch):
    backend = FakeBackend()
    class Patcher:
        patches_uuid = "model-patches"
        model = types.SimpleNamespace(current_weight_patches_uuid="loaded-patches")
    patcher = Patcher()
    module = torch.nn.Linear(4, 4).eval()
    state = modules[1].ModelGraphState(patcher, runtime_factory=lambda device:
                                     modules[0].GraphRuntime(torch, backend, warmup_calls=1))
    monkeypatch.setattr(state, "eligibility", lambda executor, options: None)
    seen = []
    def function(x, timestep, context=None, y=None, control=None, transformer_options=None):
        seen.append(dict(transformer_options))
        output = x * transformer_options["scale"]
        if backend.graph:
            backend.graph.operation = lambda: output.copy_(x * transformer_options["scale"])
        return output
    class Executor:
        class_obj = module
        original = staticmethod(function)
        def __call__(self, *args, **kwargs):
            return function(*args, **kwargs)
    options = {"scale": 3.0, "wrappers": {"opaque": object()}, "callbacks": object()}
    def sampler():
        outputs = [state.diffusion_wrapper(Executor(), torch.ones(4) * i, torch.ones(1),
                                          None, None, None, options) for i in range(3)]
        assert torch.equal(outputs[1], torch.ones(4) * 3)
        assert torch.equal(outputs[2], torch.ones(4) * 6)
    state.sampler_wrapper(sampler)
    assert all(value == {"scale": 3.0} for value in seen)
    assert state.stats["captures"] == 1 and state.stats["replays"] == 2
    assert "wrappers" in options and "callbacks" in options


@pytest.mark.parametrize("master, graph, expected", [("1", "1", True), ("0", "1", False), ("1", "0", False)])
def test_kill_switches_leave_feature_disabled(modules, monkeypatch, master, graph, expected):
    monkeypatch.setenv("OMNIXPU_ENABLE", master)
    monkeypatch.setenv("OMNIXPU_XPU_GRAPH", graph)
    assert modules[1].feature_enabled() is expected


def test_forward_container_annotations_do_not_corrupt_static_input_tree(modules):
    engine, backend, core = runtime(modules, warmup_calls=1)
    def annotated(x, transformer_options):
        transformer_options["block"] = ("output", 1)
        transformer_options["nested"].append("generated")
        return core(x)
    for value in range(4):
        options = {"nested": []}
        output = engine(annotated, (torch.ones(4) * value,), {"transformer_options": options})
        assert torch.equal(output, torch.ones(4) * value * 2)
    assert engine.stats["captures"] == 1 and engine.stats["replays"] == 3
    engine.close()


@pytest.mark.parametrize("layout", ["transposed", "sliced", "expanded"])
def test_static_input_layout_is_preserved_or_left_eager(modules, layout):
    engine, _, core = runtime(modules, warmup_calls=1)
    if layout == "transposed":
        x = torch.arange(16.0).reshape(4, 4).t()
    elif layout == "sliced":
        x = torch.arange(16.0)[::2]
    else:
        x = torch.ones(1).expand(4)
    for _ in range(3):
        torch.testing.assert_close(engine(core, (x,), {}), x * 2)
    assert engine.stats["captures"] == (1 if layout == "transposed" else 0)
    engine.close()


def test_conditioning_uuid_is_an_immutable_signature_value(modules):
    from uuid import uuid4
    identifier = uuid4()
    assert modules[0].tree_map(identifier, lambda value: value, torch) is identifier
    assert modules[0].input_signature(identifier, torch) != modules[0].input_signature(uuid4(), torch)


def test_failed_device_cleanup_blocks_new_managers_until_process_restart(modules):
    runtime = modules[0]
    backend = object.__new__(runtime.XPUBackend)
    backend.device = torch.device("xpu", 0)
    backend.poison()
    try:
        with pytest.raises(RuntimeError, match="restart"):
            runtime.XPUBackend(torch, backend.device)
        with pytest.raises(RuntimeError, match="restart"):
            with backend.execution(()):
                pytest.fail("poisoned context must not execute")
    finally:
        runtime._POISONED_DEVICES.clear()


@pytest.mark.parametrize("change", [False, True])
def test_native_position_constants_preserve_original_math_and_reject_changed_ids(modules, change):
    constants_module = importlib.import_module("omnixpu_graph_test.graph_constants")
    class Embed(torch.nn.Module):
        dim, theta, axes_dim = 8, 10000, [2, 2, 4]
        calls = 0
        def forward(self, ids):
            self.calls += 1
            return ids.cos().unsqueeze(1)
    original = Embed()
    wrapper = constants_module.ResidentEmbedND(original)
    values = []
    ids = torch.arange(12.0).reshape(1, 4, 3)
    for index in range(3):
        current = ids + (index if change else 0)
        with modules[0].constant_call("warmup", values):
            torch.testing.assert_close(wrapper(current), current.cos().unsqueeze(1), rtol=0, atol=0)
    assert original.calls == 3
    if change:
        with pytest.raises(RuntimeError, match="constants changed"):
            with modules[0].constant_call("capture", values):
                wrapper(ids)
    else:
        with modules[0].constant_call("capture", values):
            torch.testing.assert_close(wrapper(ids), ids.cos().unsqueeze(1), rtol=0, atol=0)
    assert original.calls == 3
    assert modules[0].get_constant_call() is None
    # Calls outside the managed graph scope always use the original algorithm.
    wrapper(ids)
    assert original.calls == 4


def test_native_position_constants_are_released_with_sampler_runtime(modules):
    engine, _, core = runtime(modules)
    engine(core, (torch.ones(4),), {})
    assert engine.constants
    engine.close()
    assert not engine.constants


def test_position_options_with_tensor_content_are_not_graph_qualified(modules):
    constants_module = importlib.import_module("omnixpu_graph_test.graph_constants")
    Native = type("NextDiT", (torch.nn.Module,), {"__module__": "comfy.ldm.lumina.model"})
    module = Native().eval()
    module.rope_embedder = constants_module.ResidentEmbedND(torch.nn.Identity())
    class Patcher:
        model = types.SimpleNamespace(model_lowvram=False)
        model_options, object_patches, hook_patches, weight_wrapper_patches = {}, {}, {}, {}
        def is_dynamic(self):
            return False
    patcher = Patcher()
    state = modules[1].ModelGraphState(patcher)
    state.local.active = True
    executor = types.SimpleNamespace(class_obj=module, wrappers=[state.diffusion_wrapper])
    with torch.inference_mode():
        assert state.eligibility(executor, {"rope_options": {"scale_x": torch.tensor(1.0)}}) == "dynamic position options are not qualified"


def test_eviction_cleanup_failure_prevents_another_capture(modules):
    engine, backend, core = runtime(modules, warmup_calls=1, max_entries=1)
    for _ in range(2):
        engine(core, (torch.ones(4),), {})
    graph = next(iter(engine.entries.values()))[0]
    def failed_reset():
        raise RuntimeError("graph reset failed")
    graph.reset = failed_reset
    with pytest.raises(RuntimeError, match="restart"):
        engine(core, (torch.ones(5),), {})
    assert engine.poisoned and engine.closed
    with pytest.raises(RuntimeError, match="closed"):
        engine(core, (torch.ones(5),), {})


def test_warm_position_caches_share_the_signature_budget(modules):
    constants_module = importlib.import_module("omnixpu_graph_test.graph_constants")
    class Embed(torch.nn.Module):
        dim, theta, axes_dim = 2, 10000, [2]
        def forward(self, ids):
            return ids.cos()
    wrapper = constants_module.ResidentEmbedND(Embed())
    engine, _, core = runtime(modules, max_entries=2)
    def function(x):
        wrapper(torch.arange(x.numel(), dtype=x.dtype).reshape(1, -1, 1))
        return core(x)
    for size in range(4, 14):
        engine(function, (torch.ones(size),), {})
        assert len(engine.constants) <= 2 and len(engine.warmed) <= 2
        assert len(engine.entries) <= 2 and len(engine.signatures) <= 2
    assert engine.stats["signature_evictions"] == 8
    engine.close()
    assert not engine.signatures and not engine.constants


def test_rejected_signatures_are_bounded_too(modules):
    backend = FakeBackend(capture_error=RuntimeError("unsupported"))
    engine = modules[0].GraphRuntime(torch, backend, warmup_calls=1, max_entries=2)
    core = Core(backend)
    for size in range(4, 10):
        for _ in range(2):
            engine(core, (torch.ones(size),), {})
        assert len(engine.rejected) <= 2
    assert engine.stats["capture_failures"] == 6
    engine.close()


def test_failed_cleanup_is_not_a_successful_teardown_and_survives_clone(modules):
    import weakref
    backend = FakeBackend(capture_error=RuntimeError("unsupported"), cleanup_error=RuntimeError("recording"))
    state = modules[1].ModelGraphState()
    engine = modules[0].GraphRuntime(torch, backend, warmup_calls=1)
    core = Core(backend)
    def sampler():
        state.local.runtime = engine
        engine(core, (torch.ones(4),), {})
        engine(core, (torch.ones(4),), {})
    with pytest.raises(RuntimeError, match="restart"):
        state.sampler_wrapper(sampler)
    assert state.stats["graph_teardowns"] == 0 and state.poisoned
    assert state.on_model_patcher_clone().poisoned
    assert engine in modules[0]._FAILED_RUNTIMES
    held_graph = weakref.ref(engine.failed_resources[0])
    assert held_graph() is not None
    with pytest.raises(RuntimeError, match="restart"):
        state.sampler_wrapper(lambda: pytest.fail("poisoned sampler executed"))


def test_cancelled_sampler_can_restart_with_a_new_runtime(modules):
    state = modules[1].ModelGraphState()
    engines = []
    def sampler(cancel):
        engine, _, core = runtime(modules, warmup_calls=1)
        engines.append(engine)
        state.local.runtime = engine
        for _ in range(2):
            engine(core, (torch.ones(4),), {})
        if cancel:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        state.sampler_wrapper(sampler, True)
    state.sampler_wrapper(sampler, False)
    assert all(engine.closed and not engine.entries for engine in engines)
    assert engines[0] is not engines[1]
    assert state.stats["captures"] == 2 and state.stats["graph_teardowns"] == 2


def test_overlapping_sampler_counters_are_added_once(modules):
    import concurrent.futures
    import threading
    state = modules[1].ModelGraphState()
    barrier = threading.Barrier(2)
    def sampler():
        engine, _, core = runtime(modules, warmup_calls=1)
        state.local.runtime = engine
        for _ in range(2):
            engine(core, (torch.ones(4),), {})
        barrier.wait(timeout=5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as workers:
        jobs = [workers.submit(state.sampler_wrapper, sampler) for _ in range(2)]
        for job in jobs:
            job.result(timeout=10)
    counts = modules[1].get_stats()["counts"]
    assert counts["samples"] == counts["captures"] == counts["graph_teardowns"] == 2


@pytest.mark.parametrize("option", [None, {}, {"scale_x": 1.0}])
def test_native_null_or_scalar_position_options_do_not_fail(modules, monkeypatch, option):
    constants_module = importlib.import_module("omnixpu_graph_test.graph_constants")
    Native = type("NextDiT", (torch.nn.Module,), {"__module__": "comfy.ldm.lumina.model"})
    module = Native().eval()
    module.rope_embedder = constants_module.ResidentEmbedND(torch.nn.Identity())
    class Patcher:
        model = types.SimpleNamespace(model_lowvram=False)
        model_options, object_patches, hook_patches, weight_wrapper_patches = {}, {}, {}, {}
        def is_dynamic(self):
            return False
    patcher = Patcher()
    state = modules[1].ModelGraphState(patcher)
    state.local.active = True
    control = types.SimpleNamespace(get_xpu_allocator_mode=lambda: None)
    monkeypatch.setitem(sys.modules, "comfy_aimdo", types.SimpleNamespace(control=control))
    executor = types.SimpleNamespace(class_obj=module, wrappers=[state.diffusion_wrapper])
    with torch.inference_mode():
        assert state.eligibility(executor, {"rope_options": option}) == "XPU-resident parameters required"


@pytest.mark.parametrize("hook", ["pre", "post", "training_child"])
def test_hidden_module_hooks_or_training_children_keep_eager(modules, hook):
    Native = type("UNetModel", (torch.nn.Module,),
                  {"__module__": "comfy.ldm.modules.diffusionmodules.openaimodel"})
    module = Native()
    module.child = torch.nn.Dropout()
    module.eval()
    if hook == "training_child":
        module.child.train()
    elif hook == "pre":
        module.child.register_forward_pre_hook(lambda *args: None)
    else:
        module.register_forward_hook(lambda *args: None)
    class Patcher:
        model = types.SimpleNamespace(model_lowvram=False)
        def is_dynamic(self):
            return False
    patcher = Patcher()
    state = modules[1].ModelGraphState(patcher)
    state.local.active = True
    executor = types.SimpleNamespace(class_obj=module, wrappers=[state.diffusion_wrapper])
    with torch.inference_mode():
        assert state.eligibility(executor, {}) == "training children or module forward hooks are not qualified"


def test_cpu_metadata_is_rejected_before_xpu_stream_work(modules):
    backend = object.__new__(modules[0].XPUBackend)
    backend.device, backend.torch = torch.device("xpu", 0), torch
    with pytest.raises(modules[0].UnsupportedGraphInput, match="model XPU"):
        backend.validate_inputs((torch.ones(4),))


def test_new_callable_with_identical_inputs_cannot_reuse_old_graph(modules):
    engine, backend, _ = runtime(modules, warmup_calls=1)
    def make(scale):
        def function(x):
            output = x * scale
            if backend.graph:
                backend.graph.operation = lambda: output.copy_(x * scale)
            return output
        return function
    for scale in (2, 3):
        function = make(scale)
        for _ in range(2):
            assert torch.equal(engine(function, (torch.ones(4),), {}), torch.ones(4) * scale)
    assert engine.stats["captures"] == 2
    engine.close()


def qualification_state(modules):
    Native = type("UNetModel", (torch.nn.Module,),
                  {"__module__": "comfy.ldm.modules.diffusionmodules.openaimodel"})
    module = Native().eval()
    class Patcher:
        model = types.SimpleNamespace(model_lowvram=False)
        model_options, object_patches, hook_patches, weight_wrapper_patches = {}, {}, {}, {}
        def is_dynamic(self):
            return False
    patcher = Patcher()
    state = modules[1].ModelGraphState(patcher)
    state.local.active = True
    executor = types.SimpleNamespace(class_obj=module, wrappers=[state.diffusion_wrapper])
    return patcher, state, executor


@pytest.mark.parametrize("backend", ["ESIMD", "EsImD"])
def test_esimd_opt_out_matches_attention_case_normalization(modules, monkeypatch, backend):
    patcher, state, executor = qualification_state(modules)
    monkeypatch.setenv("OMNI_ATTN_BACKEND", backend)
    with torch.inference_mode():
        assert state.eligibility(executor, {}) == "stateful ESIMD attention is not qualified"


def test_global_module_hooks_are_not_silently_frozen(modules):
    patcher, state, executor = qualification_state(modules)
    handle = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
    try:
        with torch.inference_mode():
            assert state.eligibility(executor, {}) == "global module forward hooks are not qualified"
    finally:
        handle.remove()
