"""Quoted text and actual token boundaries must remain distinguishable."""
import importlib.util
from pathlib import Path

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
