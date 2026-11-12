"""Keep a quoted closing tag in Qwen3.5 reasoning out of the visible answer."""

from functools import wraps


_MARKER = "_omnixpu_qwen35_token_boundary"


def _split_at_token(clip, tokenizer, ids):
    closing = tokenizer.convert_tokens_to_ids("</think>")
    if closing is None or closing == tokenizer.unk_token_id:
        raise RuntimeError("Qwen3.5 tokenizer has no closing thinking token")
    if tokenizer.decode([closing], skip_special_tokens=True) != "</think>":
        raise RuntimeError("Qwen3.5 closing token is not preserved by decode")
    try:
        index = ids.index(closing)
    except ValueError:
        return None
    reasoning = clip.decode(ids[:index])
    if "</think>" not in reasoning:
        return None  # The upstream first-string split already chose this token.
    return (clip.decode(ids[index + 1:]).strip(),
            reasoning.replace("<think>", "", 1).strip())


class _Clip:
    def __init__(self, original):
        self.original = original
        self.ids = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def generate(self, *args, **kwargs):
        self.ids = self.original.generate(*args, **kwargs)
        return self.ids


def apply():
    try:
        from comfy.text_encoders.qwen35 import Qwen35ImageTokenizer
        from comfy_extras import nodes_textgen
    except ImportError:
        return False, "Qwen3.5 text generation is unavailable"
    cls = nodes_textgen.TextGenerateV2
    method = cls.__dict__.get("execute")
    if not isinstance(method, classmethod):
        return False, "TextGenerateV2 no longer exposes classmethod execute"
    original = method.__func__
    if getattr(original, _MARKER, False):
        return True, None
    if "partition" not in original.__code__.co_names or "</think>" not in original.__code__.co_consts:
        return False, "TextGenerateV2 no longer uses the first-string thinking split"

    @wraps(original)
    def execute(cls, clip, *args, **kwargs):
        outer = getattr(clip, "tokenizer", None)
        if not isinstance(outer, Qwen35ImageTokenizer):
            return original(cls, clip, *args, **kwargs)
        tokenizer = getattr(outer, outer.clip).tokenizer
        proxy = _Clip(clip)
        result = original(cls, proxy, *args, **kwargs)
        if not isinstance(proxy.ids, list) or any(type(i) is not int for i in proxy.ids):
            raise RuntimeError("Qwen3.5 generation no longer returns a token ID list")
        split = _split_at_token(clip, tokenizer, proxy.ids)
        return result if split is None else nodes_textgen.io.NodeOutput(*split)

    setattr(execute, _MARKER, True)
    cls.execute = classmethod(execute)
    return True, None
