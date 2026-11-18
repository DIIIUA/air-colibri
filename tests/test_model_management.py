"""CLI model selection and listing must agree and avoid importing the GPU runtime."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aircolibri_cli.model_source import local_model_candidates, resolve_model_dir, runtime_model_dir, ModelNotFoundError


def test_help_and_catalog_work_without_torch(tmp_path):
    (tmp_path/"src").mkdir()
    (tmp_path/"src/aircolibri_integrated.py").touch()
    model = tmp_path/"models/01-small"
    model.mkdir(parents=True)
    (model/"config.json").write_text("{}")
    (model/"model.safetensors").write_bytes(b"listing fixture")
    script = '''
import sys, importlib.abc
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('CLI listing imported Torch')
sys.meta_path.insert(0,NoTorch())
from aircolibri_cli.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    root = Path(__file__).resolve().parents[1]
    for args in (["--help"], ["models","--aircolibri-root",str(tmp_path),"--json"]):
        result = subprocess.run([sys.executable,"-c",script,*args],cwd=root,capture_output=True,text=True)
        assert result.returncode == 0, result.stderr
        if args[0] == "models":
            rows = json.loads(result.stdout)
            assert rows[0]["index"] == 1 and rows[0]["name"] == "01-small"
            assert rows[0]["status"] == "available" and "metadata_supported" not in rows[0]


def test_catalog_numbers_and_names_resolve_same_folder(tmp_path, monkeypatch):
    from aircolibri_cli import model_source
    for name in ("02-second", "01-first"):
        folder = tmp_path/"models"/name
        folder.mkdir(parents=True)
        (folder/"weights.gguf").touch()
    monkeypatch.setattr(model_source,"check_supported",lambda folder: None)
    choices = local_model_candidates(tmp_path)
    assert [p.name for p in choices] == ["01-first","02-second"]
    assert resolve_model_dir("2", tmp_path) == resolve_model_dir("02-second",tmp_path) == choices[1].resolve()
    with pytest.raises(ModelNotFoundError,match="outside"):
        resolve_model_dir("3",tmp_path)


def test_nested_download_marker_is_resolved_but_cannot_escape(tmp_path):
    (tmp_path/"quant").mkdir()
    marker = tmp_path/"aircolibri-source.json"
    marker.write_text(json.dumps({"runtime_subdir":"quant"}))
    assert runtime_model_dir(tmp_path) == (tmp_path/"quant").resolve()
    marker.write_text(json.dumps({"runtime_subdir":".."}))
    with pytest.raises(ModelNotFoundError,match="outside"):
        runtime_model_dir(tmp_path)


@pytest.mark.parametrize("value,repo,file,revision", [
    ("Qwen/Qwen3.5-0.8B","Qwen/Qwen3.5-0.8B",None,None),
    ("https://huggingface.co/Qwen/Qwen3.5-0.8B","Qwen/Qwen3.5-0.8B",None,None),
    ("https://huggingface.co/owner/model/tree/main","owner/model",None,"main"),
    ("https://huggingface.co/owner/model/resolve/main/sub/model-Q4.gguf?download=true","owner/model","sub/model-Q4.gguf","main"),
    ("https://huggingface.co/owner/model/blob/rev/model.safetensors","owner/model",None,"rev"),
    ("owner/model::model-Q6.gguf","owner/model","model-Q6.gguf",None),
])
def test_hf_page_and_file_links_normalize(value, repo, file, revision):
    from aircolibri_cli.model_reference import parse_hf_reference
    parsed = parse_hf_reference(value)
    assert (parsed.repo_id, parsed.filename, parsed.revision) == (repo, file, revision)


def test_download_directory_ownership_and_external_catalog_registration(tmp_path, monkeypatch):
    from aircolibri_cli import model_hub
    root, target = tmp_path/"app", tmp_path/"external model"
    def downloaded(reference, allow_download, *, local_dir, **kwargs):
        (local_dir/"config.json").write_text("{}")
        (local_dir/"model.safetensors").write_bytes(b"fixture")
        return local_dir
    monkeypatch.setattr(model_hub,"hf_snapshot",downloaded)
    path, marker, receipt = model_hub.download_model("https://huggingface.co/owner/model",root,directory=target)
    assert path == target and receipt["status"] == "downloaded"
    assert local_model_candidates(root) == [target]
    assert json.loads(marker.read_text())["repo_id"] == "owner/model"
    # Retry the same download; another model/revision must not overwrite it.
    model_hub.download_model("owner/model",root,directory=target)
    with pytest.raises(ModelNotFoundError,match="different"):
        model_hub.download_model("other/model",root,directory=target)
    with pytest.raises(ModelNotFoundError,match="revision"):
        model_hub.download_model("owner/model",root,directory=target,revision="new")
    assert (target/"model.safetensors").read_bytes() == b"fixture"


def test_download_command_really_verifies_runtime_packing(tmp_path, monkeypatch):
    from aircolibri_cli import cli, model_hub
    from test_cpu_independence import tiny_checkpoint
    target = tmp_path/"downloaded"
    def downloaded(reference, allow_download, *, local_dir, **kwargs):
        tiny_checkpoint(local_dir)
        return local_dir
    monkeypatch.setattr(model_hub,"hf_snapshot",downloaded)
    monkeypatch.setattr(model_hub,"register_model",lambda root,folder: None)
    output = tmp_path/"result.json"
    assert cli.main(["download","https://huggingface.co/example/tiny","--directory",str(target),
                     "--device","cpu","--weight-format","q6","--json-out",str(output)]) == 0
    result = json.loads(output.read_text("utf-8"))
    prep = result["preparation"]
    assert result["status"] == "ready" and prep["finite_first_forward"]
    assert prep["packed_projections"]["bits"] == [6] and prep["packed_projections"]["tiled_calls"] > 0
    assert {p.name for p in target.glob("*.safetensors")} == {"model.safetensors"}
