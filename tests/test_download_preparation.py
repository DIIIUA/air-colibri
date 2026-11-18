"""Hub snapshots must become runnable sources without a manual conversion step."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aircolibri_cli.model_source import hf_snapshot, ModelNotFoundError


def hub(monkeypatch, cached, downloaded, files):
    import huggingface_hub
    calls = []
    def snapshot(repo_id, *, allow_patterns, local_files_only=False):
        calls.append(dict(repo=repo_id, patterns=allow_patterns, local=local_files_only))
        return str(cached if local_files_only else downloaded)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(list_repo_files=lambda **kw: files))
    return calls


def test_partial_hf_cache_downloads_then_automatically_packs_and_generates(tmp_path, monkeypatch):
    from test_cpu_independence import tiny_checkpoint
    from aircolibri_cli import cli
    partial, full = tmp_path/"partial", tmp_path/"full"
    partial.mkdir(); full.mkdir()
    (partial/"config.json").write_text("{}")
    tiny_checkpoint(full)
    calls = hub(monkeypatch, partial, full, [p.name for p in full.iterdir()])
    args = cli.build_parser().parse_args(["ask", "--model", "example/tiny", "--device", "cpu",
                                        "--weight-format", "q6", "Hi"])
    with cli._engine_from_args(args) as u:
        events = list(u.stream_chat([{"role":"user", "content":"Hi"}], max_new_tokens=3,
                                    enable_thinking=False, stop_on_eos=False))
        done = next(e for e in events if e["type"] == "done")
        assert done["metrics"]["generated_tokens"] == 3
        stats = u.status_snapshot()["packed_projections"]
        assert stats["modules"] > 0 and stats["bits"] == [6]
        assert stats["tiled_calls"] > 0
    assert [c["local"] for c in calls] == [True, False]
    assert set(p.name for p in full.glob("*.safetensors")) == {"model.safetensors"}


def test_complete_cache_needs_no_network_and_missing_shard_is_not_complete(tmp_path, monkeypatch):
    from test_cpu_independence import tiny_checkpoint
    tiny_checkpoint(tmp_path)
    calls = hub(monkeypatch, tmp_path, tmp_path, [])
    assert hf_snapshot("example/tiny", False) == tmp_path
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"a":"missing.safetensors"}}))
    with pytest.raises(ModelNotFoundError, match="incomplete"):
        hf_snapshot("example/tiny", False)
    assert all(c["local"] for c in calls)


def test_gguf_selection_fetches_exact_split_and_not_other_quantizations(tmp_path, monkeypatch):
    partial, full = tmp_path/"partial", tmp_path/"full"
    partial.mkdir(); (full/"sub").mkdir(parents=True)
    names = [f"sub/model-Q4-0000{i}-of-00002.gguf" for i in (1,2)]
    for name in names:
        (full/name).write_bytes(b"fixture")
    calls = hub(monkeypatch, partial, full, names+["other-Q6.gguf"])
    assert hf_snapshot("example/model::"+names[0], True) == full/"sub"
    selected = [p for p in calls[-1]["patterns"] if p.endswith(".gguf")]
    assert selected == names
    assert not any("safetensors" in p for p in calls[-1]["patterns"])


def test_ambiguous_gguf_repo_requires_selection_before_any_weight_download(tmp_path, monkeypatch):
    calls = hub(monkeypatch, tmp_path, tmp_path, ["model-Q4.gguf", "model-Q6.gguf"])
    with pytest.raises(ModelNotFoundError, match="Select one GGUF"):
        hf_snapshot("example/model", True)
    assert len(calls) == 1 and calls[0]["local"]


@pytest.mark.parametrize("filename", ["../model.gguf", "/model.gguf", "sub\\model.gguf", "*.gguf"])
def test_gguf_selector_cannot_escape_or_fetch_all_variants(filename):
    with pytest.raises(ModelNotFoundError):
        hf_snapshot("example/model::"+filename, True)


def test_doctor_explicit_repo_downloads_but_default_stays_offline(tmp_path, monkeypatch, capsys):
    from aircolibri_cli import cli
    calls = []
    def resolve(model, root, allow_download):
        calls.append(allow_download)
        raise ModelNotFoundError("test stops after resolving")
    monkeypatch.setattr(cli, "resolve_model_dir", resolve)
    for model in (["--model", "example/model"], []):
        args = cli.build_parser().parse_args(["doctor", "--device", "cpu", *model])
        assert cli.cmd_doctor(args) == cli.EXIT_MODEL
    assert calls == [True, False]


def test_unavailable_hub_is_a_concise_model_error(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import OfflineModeIsEnabled
    hub(monkeypatch, tmp_path, tmp_path, [])
    def unavailable(**kwargs):
        raise OfflineModeIsEnabled("offline")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(list_repo_files=unavailable))
    with pytest.raises(ModelNotFoundError, match="Cannot access Hugging Face model"):
        hf_snapshot("example/model", True)
