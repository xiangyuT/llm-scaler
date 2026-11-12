"""Quoted text and actual token boundaries must remain distinguishable."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.mark.parametrize("pieces,expected", [
    (["reason", "</think>", "answer"], None),
    (["quoted </", "think> in reasoning", "</think>", " answer "],
     ("answer", "quoted </think> in reasoning")),
    (["reason", "</think>", "answer mentions </think>"], None),
    (["unfinished reasoning quotes </think>"], None),
])
def test_actual_token_boundary(pieces, expected):
    path=Path(__file__).parents[1]/"ComfyUI-OmniXPU/adapters/qwen35_text_boundary.py"
    spec=importlib.util.spec_from_file_location("qwen35_text_boundary_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    closing=next((i for i,s in enumerate(pieces) if s=="</think>"),len(pieces))
    class Tokenizer:
        unk_token_id=-1
        def convert_tokens_to_ids(self,text):return closing
        def decode(self,ids,**kwargs):return "</think>"
    class Clip:
        def decode(self,ids):return "".join(pieces[i] for i in ids)
    assert module._split_at_token(Clip(),Tokenizer(),list(range(len(pieces))))==expected


def test_patch_targets_registered_node_not_duplicate_import(monkeypatch):
    path=Path(__file__).parents[1]/"ComfyUI-OmniXPU/adapters/qwen35_text_boundary.py"
    spec=importlib.util.spec_from_file_location("qwen35_registered_boundary_test",path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    io=types.SimpleNamespace(NodeOutput=lambda *args:args)
    class Registered:
        @classmethod
        def execute(cls,clip):
            reasoning,sep,text=clip.decode(clip.generate()).partition("</think>")
            return io.NodeOutput(text.strip(),reasoning.strip())
    class Duplicate(Registered):
        execute=classmethod(Registered.execute.__func__)
    owner=types.ModuleType("registered_textgen");owner.io=io
    owner.TextGenerateV2=Registered;Registered.__module__=owner.__name__
    nodes=types.ModuleType("nodes");nodes.NODE_CLASS_MAPPINGS={"TextGenerateV2":Registered}
    qwen=types.ModuleType("comfy.text_encoders.qwen35")
    class Outer:
        clip="q"
    qwen.Qwen35ImageTokenizer=Outer
    duplicate=types.ModuleType("comfy_extras.nodes_textgen");duplicate.TextGenerateV2=Duplicate;duplicate.io=io
    for name,value in {"nodes":nodes,owner.__name__:owner,"comfy":types.ModuleType("comfy"),
                       "comfy.text_encoders":types.ModuleType("comfy.text_encoders"),qwen.__name__:qwen,
                       "comfy_extras":types.ModuleType("comfy_extras"),duplicate.__name__:duplicate}.items():
        monkeypatch.setitem(sys.modules,name,value)
    assert adapter.apply()==(True,None)
    assert getattr(Registered.execute.__func__,adapter._MARKER,False)
    assert not getattr(Duplicate.execute.__func__,adapter._MARKER,False)
    # Executor clones copy the registered descriptor, not the package import.
    clone=type("RegisteredClone",(Registered,),dict(Registered.__dict__))
    assert clone.execute.__func__ is Registered.execute.__func__
