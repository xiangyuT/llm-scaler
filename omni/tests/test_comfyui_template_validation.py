"""Fixture tests for installed template provenance; no image or device work."""

import base64
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def installed(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "tools/validate_comfyui_image.py"
    spec = importlib.util.spec_from_file_location("template_validator_test", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    files = []
    hashes = {}
    for name in validator.REQUIRED_MINIMAX_H3_TEMPLATES:
        entry = importlib.metadata.PackagePath("comfyui_workflow_templates_json/templates/" + name)
        target = tmp_path / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"fixture": name, "nodes": []}))
        digest = hashlib.sha256(target.read_bytes()).digest()
        entry.hash = importlib.metadata.FileHash("sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("="))
        entry.size = target.stat().st_size
        files.append(entry)
        hashes[name] = digest.hex()
    pinned = "video_minimax_h3_t2v.json"
    monkeypatch.setattr(validator, "PINNED_MINIMAX_H3_TEMPLATE_HASHES", {pinned: hashes[pinned]})
    distribution = SimpleNamespace(files=files, locate_file=lambda entry: tmp_path / entry)
    return SimpleNamespace(validator=validator, distribution=distribution, hashes=hashes,
        selected=next(entry for entry in files if entry.name == pinned), root=tmp_path)


def test_complete_installed_templates_match_wheel_record_and_current_pin(installed):
    assert installed.validator.require_minimax_h3_templates(installed.distribution) == installed.hashes


@pytest.mark.parametrize("change", ["missing", "duplicate", "no_record"])
def test_required_template_needs_one_record_entry(installed, change):
    if change == "missing":
        installed.distribution.files.remove(installed.selected)
    elif change == "duplicate":
        installed.distribution.files.append(installed.selected)
    else:
        installed.distribution.files = None
    with pytest.raises(RuntimeError, match="one installed RECORD entry"):
        installed.validator.require_minimax_h3_templates(installed.distribution)


def test_missing_installed_template_is_rejected(installed):
    (installed.root / installed.selected).unlink()
    with pytest.raises(RuntimeError, match="missing MiniMax H3 file"):
        installed.validator.require_minimax_h3_templates(installed.distribution)


@pytest.mark.parametrize("change", ["missing_hash", "algorithm", "digest", "size"])
def test_template_must_match_record_hash_and_size(installed, change):
    if change == "missing_hash":
        installed.selected.hash = None
    elif change == "algorithm":
        installed.selected.hash = importlib.metadata.FileHash("md5=fixture")
    elif change == "digest":
        installed.selected.hash = importlib.metadata.FileHash("sha256=wrong")
    else:
        installed.selected.size += 1
    with pytest.raises(RuntimeError, match="differs from installed RECORD"):
        installed.validator.require_minimax_h3_templates(installed.distribution)


def test_changed_template_bytes_cannot_pass_installed_record(installed):
    (installed.root / installed.selected).write_text('{"nodes":["changed"]}')
    with pytest.raises(RuntimeError, match="differs from installed RECORD"):
        installed.validator.require_minimax_h3_templates(installed.distribution)


def test_updated_record_cannot_replace_the_reviewed_official_template(installed):
    path = installed.root / installed.selected
    path.write_text('{"nodes":["changed"]}')
    digest = hashlib.sha256(path.read_bytes()).digest()
    installed.selected.hash = importlib.metadata.FileHash("sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("="))
    installed.selected.size = path.stat().st_size
    with pytest.raises(RuntimeError, match="official template hash"):
        installed.validator.require_minimax_h3_templates(installed.distribution)


def test_malformed_json_cannot_pass_a_matching_record(installed):
    path = installed.root / installed.selected
    path.write_text('{invalid JSON}')
    digest = hashlib.sha256(path.read_bytes()).digest()
    installed.selected.hash = importlib.metadata.FileHash("sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("="))
    installed.selected.size = path.stat().st_size
    with pytest.raises(json.JSONDecodeError):
        installed.validator.require_minimax_h3_templates(installed.distribution)
