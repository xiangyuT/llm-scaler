"""Inference-only XPU execution graphs with sampler-owned lifetimes."""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import contextmanager
import logging
import os
import threading
import traceback
from uuid import UUID
from types import SimpleNamespace

from .adapters.errors import is_fatal_accelerator_error

_CAPTURE_LOCK = threading.Lock()
_POISONED_DEVICES = set()
_FAILED_RUNTIMES = []  # Keep unsafe graph/storage owners alive until restart.
_CONSTANT_CALL = threading.local()


@contextmanager
def constant_call(phase, values):
    previous = getattr(_CONSTANT_CALL, "current", None)
    current = SimpleNamespace(phase=phase, values=values, index=0)
    _CONSTANT_CALL.current = current
    try:
        yield
        if phase == "capture" and current.index != len(values):
            raise RuntimeError("constant call sequence changed")
    finally:
        _CONSTANT_CALL.current = previous


def get_constant_call():
    return getattr(_CONSTANT_CALL, "current", None)


class UnsupportedGraphInput(ValueError):
    pass


def tree_map(value, tensor_fn, torch):
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor:
            raise UnsupportedGraphInput("tensor subclasses are not supported")
        return tensor_fn(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise UnsupportedGraphInput("dictionary keys must be strings")
        return {key: tree_map(item, tensor_fn, torch) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(tree_map(item, tensor_fn, torch) for item in value)
    if value is None or type(value) in (bool, int, float, str, UUID):
        return value
    raise UnsupportedGraphInput(f"unsupported argument type: {type(value).__name__}")


def input_signature(value, torch):
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor:
            raise UnsupportedGraphInput("tensor subclasses are not supported")
        return ("tensor", tuple(value.shape), tuple(value.stride()), str(value.dtype),
                str(value.device), value.requires_grad)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise UnsupportedGraphInput("dictionary keys must be strings")
        return ("dict", tuple((key, input_signature(item, torch)) for key, item in value.items()))
    if type(value) in (list, tuple):
        return (type(value).__name__, tuple(input_signature(item, torch) for item in value))
    if value is None or type(value) in (bool, int, float, str, UUID):
        return (type(value).__name__, value)
    raise UnsupportedGraphInput(f"unsupported argument type: {type(value).__name__}")


def copy_inputs(target, source, torch):
    if isinstance(target, torch.Tensor):
        target.copy_(source)
    elif isinstance(target, dict):
        for key in target:
            copy_inputs(target[key], source[key], torch)
    elif type(target) in (list, tuple):
        for dst, src in zip(target, source):
            copy_inputs(dst, src, torch)


def clone_input(tensor):
    clone = tensor.clone()
    if clone.stride() != tensor.stride():
        raise UnsupportedGraphInput("input layout cannot be preserved by a static clone")
    return clone


class XPUBackend:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.check_health()
        with torch.xpu.device(device):
            self.stream = torch.xpu.Stream(device=device)

    def check_health(self):
        if str(self.device) in _POISONED_DEVICES:
            raise RuntimeError("previous XPU graph cleanup failed on this device; restart the process")

    def poison(self):
        _POISONED_DEVICES.add(str(self.device))

    def validate_inputs(self, inputs):
        def validate(tensor):
            if tensor.device != self.device or tensor.layout != self.torch.strided:
                raise UnsupportedGraphInput("graph inputs must be strided tensors on the model XPU")
            return tensor
        tree_map(inputs, validate, self.torch)

    @contextmanager
    def execution(self, inputs):
        self.check_health()
        xpu = self.torch.xpu
        caller = xpu.current_stream(self.device)
        self.stream.wait_stream(caller)
        def record(tensor):
            tensor.record_stream(self.stream)
            return tensor
        tree_map(inputs, record, self.torch)
        with xpu.device(self.device), xpu.stream(self.stream):
            try:
                yield
            finally:
                caller.wait_stream(self.stream)

    def new_graph(self):
        return self.torch.xpu.XPUGraph()

    @contextmanager
    def capture(self, graph):
        # The outer stream context restores the caller even if capture_end fails.
        with self.torch.xpu.device(self.device), self.torch.xpu.stream(self.stream):
            with self.torch.xpu.graph(graph, stream=self.stream):
                yield

    def quiesce(self):
        self.stream.synchronize()

    def recover(self, graph):
        with self.torch.xpu.device(self.device), self.torch.xpu.stream(self.stream):
            if self.torch.xpu.is_current_stream_capturing():
                raise RuntimeError("XPU capture remains active; restart the process")
        self.quiesce()
        graph.reset()

    def retain_outputs(self, value):
        caller = self.torch.xpu.current_stream(self.device)
        return tree_map(value, lambda tensor: tensor.record_stream(caller) or tensor, self.torch)


class GraphRuntime:
    """Warm using real calls, then replay; never run discarded model forwards."""
    def __init__(self, torch, backend, *, warmup_calls=3, max_entries=2):
        if warmup_calls < 1 or max_entries < 1:
            raise ValueError("warmup_calls and max_entries must be positive")
        self.torch, self.backend = torch, backend
        self.warmup_calls, self.max_entries = warmup_calls, max_entries
        self.entries = OrderedDict()
        self.signatures = OrderedDict()
        self.warmed = Counter()
        self.constants = {}
        self.rejected = {}
        self.stats = Counter()
        self.reasons = Counter()
        self.closed = False
        self.poisoned = False

    def _poison(self, resources=None):
        self.closed = self.poisoned = True
        self.failed_resources = resources
        _FAILED_RUNTIMES.append(self)
        if hasattr(self.backend, "poison"):
            self.backend.poison()

    def _admit_signature(self, key, owner):
        if key in self.signatures:
            self.signatures.move_to_end(key)
            return
        if len(self.signatures) >= self.max_entries:
            old_key = next(iter(self.signatures))
            old = self.entries.get(old_key)
            try:
                self.backend.quiesce()
                if old is not None:
                    old[0].reset()
            except BaseException as error:
                self._poison()
                raise RuntimeError("XPU graph eviction cleanup failed; restart the process") from error
            if old is not None:
                del self.entries[old_key]
                self.stats["evictions"] += 1
            self.signatures.pop(old_key)
            self.constants.pop(old_key, None)
            self.warmed.pop(old_key, None)
            self.rejected.pop(old_key, None)
            self.stats["signature_evictions"] += 1
        self.signatures[key] = owner

    def _eager(self, function, args, kwargs, reason=None):
        self.stats["eager"] += 1
        if reason:
            self.reasons[reason] += 1
        return function(*args, **kwargs)

    def __call__(self, function, args, kwargs, *, identity=(), callable_identity=None):
        if self.closed:
            raise RuntimeError("execution graph runtime is closed")
        inputs = (args, kwargs)
        try:
            if hasattr(self.backend, "validate_inputs"):
                self.backend.validate_inputs(inputs)
            owner = function if callable_identity is None else callable_identity
            function_key = (id(getattr(owner, "__func__", owner)), id(getattr(owner, "__self__", None)))
            key = (identity, function_key, input_signature(inputs, self.torch))
        except UnsupportedGraphInput as error:
            return self._eager(function, args, kwargs, str(error))
        # Retain the callable as well as its identity, preventing Python ID
        # reuse from selecting another function's captured graph.
        self._admit_signature(key, owner)
        if key in self.rejected:
            return self._eager(function, args, kwargs, self.rejected[key])
        if key not in self.entries and self.warmed[key] < self.warmup_calls:
            self.warmed[key] += 1
            with self.backend.execution(inputs), constant_call("warmup", self.constants.setdefault(key, [])):
                result = self._eager(function, args, kwargs)
            return self.backend.retain_outputs(result)

        entry = self.entries.get(key)
        graph = None
        static = None
        output = None
        capturing = entry is None
        capture_complete = not capturing
        try:
            with self.backend.execution(inputs):
                if capturing:
                    static = tree_map(inputs, clone_input, self.torch)
                    graph = self.backend.new_graph()
                    # Native forwards annotate transformer_options in place.
                    # Own an immutable container tree for future input copies;
                    # the captured forward gets separate containers referencing
                    # the same static tensors.
                    call_inputs = tree_map(static, lambda tensor: tensor, self.torch)
                    with _CAPTURE_LOCK, self.backend.capture(graph), constant_call("capture", self.constants.get(key, [])):
                        output = function(*call_inputs[0], **call_inputs[1])
                    # Validate output structure before retaining or replaying it.
                    input_signature(output, self.torch)
                    capture_complete = True
                    entry = (graph, static, output)
                    self.entries[key] = entry
                    self.stats["captures"] += 1
                    self.stats["position_constants"] += len(self.constants.get(key, []))
                else:
                    self.entries.move_to_end(key)
                    copy_inputs(entry[1], inputs, self.torch)
                entry[0].replay()
                # A later replay overwrites graph outputs; callers own snapshots.
                result = tree_map(entry[2], lambda tensor: tensor.clone(), self.torch)
            self.stats["replays"] += 1
            return self.backend.retain_outputs(result)
        except BaseException as error:
            if graph is None and isinstance(error, UnsupportedGraphInput):
                self.rejected[key] = str(error)
                self.constants.pop(key, None)
                return self._eager(function, args, kwargs, str(error))
            if entry is not None:
                graph = entry[0]
            if graph is not None:
                try:
                    self.backend.recover(graph)
                except BaseException as cleanup_error:
                    self._poison((graph, static, output))
                    raise RuntimeError("XPU graph cleanup failed; restart the process") from cleanup_error
            self.entries.pop(key, None)
            if capture_complete or is_fatal_accelerator_error(error) or not isinstance(error, RuntimeError):
                raise
            reason = f"capture unsupported: {error}"
            if os.environ.get("OMNIXPU_XPU_GRAPH_DEBUG", "0") == "1":
                sites = traceback.extract_tb(error.__traceback__)
                logging.info("[OmniXPU graph] capture failure sites: %s",
                             " -> ".join(f"{site.filename}:{site.lineno} {site.name}" for site in sites))
            self.rejected[key] = reason
            self.constants.pop(key, None)
            self.stats["capture_failures"] += 1
            return self._eager(function, args, kwargs, reason)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.backend.quiesce()
            for graph, _, _ in self.entries.values():
                graph.reset()
        except BaseException:
            self._poison()
            raise
        self.entries.clear()
        self.signatures.clear()
        self.warmed.clear()
        self.constants.clear()
        self.rejected.clear()
