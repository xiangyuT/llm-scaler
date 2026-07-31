from types import SimpleNamespace

import torch

from omni_xpu_kernel import cute


class FakeTensor:
    def __init__(self, shape=(1, 63699, 7, 128)):
        self.shape = shape
        self.dtype = torch.bfloat16
        self.device = SimpleNamespace(type="xpu")


def fake_ops(calls):
    def general(q, k, v):
        calls.append("general")
        return "general-output"

    def h3(q, k, v):
        calls.append("h3")
        return "h3-output"

    return SimpleNamespace(
        cute_fmha=SimpleNamespace(sdp=general),
        cute_h3_bf16=SimpleNamespace(sdp=h3),
    )


def test_exact_h3_contract_routes_to_isolated_sidecar(monkeypatch):
    calls = []
    q = FakeTensor()
    k = FakeTensor()
    v = FakeTensor()
    k.device = q.device
    v.device = q.device
    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(cute, "supports_h3_bf16", lambda: True)
    monkeypatch.setattr(cute.torch, "ops", fake_ops(calls))

    assert cute.sdp(q, k, v) == "h3-output"
    assert calls == ["h3"]


def test_nonexact_contract_keeps_general_sidecar(monkeypatch):
    calls = []
    q = FakeTensor(shape=(1, 63698, 7, 128))
    k = FakeTensor(shape=q.shape)
    v = FakeTensor(shape=q.shape)
    k.device = q.device
    v.device = q.device
    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(cute, "supports_h3_bf16", lambda: True)
    monkeypatch.setattr(cute.torch, "ops", fake_ops(calls))

    assert cute.sdp(q, k, v) == "general-output"
    assert calls == ["general"]


def test_h3_route_can_be_disabled_for_e2e_ab(monkeypatch):
    calls = []
    q = FakeTensor()
    k = FakeTensor()
    v = FakeTensor()
    k.device = q.device
    v.device = q.device
    monkeypatch.setenv("OMNI_CUTE_H3_BF16", "0")
    monkeypatch.setattr(cute, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(cute, "supports_h3_bf16", lambda: True)
    monkeypatch.setattr(cute.torch, "ops", fake_ops(calls))

    assert cute.sdp(q, k, v) == "general-output"
    assert calls == ["general"]
