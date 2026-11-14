"""CPU checks for model discovery and direct Hugging Face shard reading (tiny synthetic checkpoints)."""
import json
import os
import time

import pytest
import torch
from safetensors.torch import load_file, save_file

from aircolibri_cli.model_source import (
    HFShardedSource,
    PreadSafetensorsReader,
    UnsupportedModelError,
    check_supported,
    detect_format,
    find_local_models,
    load_text_config,
    resolve_model_dir,
)

TEXT_CFG = {
    "model_type": "qwen3_5_text",
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "vocab_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "layer_types": ["linear_attention", "full_attention"],
    "tie_word_embeddings": False,
}


def _make_hf_checkpoint(path, nested=True, separate_mtp=True):
    path.mkdir(parents=True, exist_ok=True)
    cfg = {"model_type": "qwen3_5", "text_config": TEXT_CFG} if nested else TEXT_CFG
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    p = "model.language_model." if nested else "model."
    shard1 = {
        f"{p}embed_tokens.weight": torch.randn(32, 64, dtype=torch.bfloat16),
        f"{p}layers.0.mlp.gate_proj.weight": torch.randn(128, 64, dtype=torch.bfloat16),
        f"{p}layers.0.input_layernorm.weight": torch.randn(64, dtype=torch.bfloat16),
    }
    shard2 = {
        f"{p}layers.1.mlp.gate_proj.weight": torch.randn(128, 64, dtype=torch.bfloat16),
        f"{p}layers.1.self_attn.q_proj.weight": torch.randn(128, 64, dtype=torch.bfloat16),
        f"{p}norm.weight": torch.randn(64, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(32, 64, dtype=torch.bfloat16),
        "model.visual.patch.weight": torch.randn(4, 4, dtype=torch.bfloat16),
    }
    save_file(shard1, str(path / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(path / "model-00002-of-00002.safetensors"))
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
    mtp = {"mtp.fc.weight": torch.randn(64, 128, dtype=torch.bfloat16), "mtp.norm.weight": torch.randn(64, dtype=torch.bfloat16)}
    if separate_mtp:
        save_file(mtp, str(path / "model-mtp.safetensors"))
    else:
        save_file(mtp, str(path / "model-00003-of-00003.safetensors"))
        weight_map.update({k: "model-00003-of-00003.safetensors" for k in mtp})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    return shard1, shard2, mtp


@pytest.mark.parametrize("nested,separate_mtp", [(True, True), (False, False)])
def test_hf_shards_map_to_canonical_names(tmp_path, nested, separate_mtp):
    d = tmp_path / "Qwen3.5-tiny"
    shard1, shard2, mtp = _make_hf_checkpoint(d, nested=nested, separate_mtp=separate_mtp)
    assert detect_format(d) == "hf"
    check_supported(d)
    src = HFShardedSource(d)
    assert src.num_layers == 2
    assert set(src.layer_entries(0)) == {"model.layers.0.mlp.gate_proj.weight", "model.layers.0.input_layernorm.weight"}
    assert set(src.layer_entries(1)) == {"model.layers.1.mlp.gate_proj.weight", "model.layers.1.self_attn.q_proj.weight"}
    assert set(src.globals_entries()) == {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    assert src.has_mtp() and set(src.mtp_entries()) == set(mtp)
    assert src.tensor_shape(src.layer_entries(1)["model.layers.1.self_attn.q_proj.weight"]) == (128, 64)

    reader = PreadSafetensorsReader(max_handles=1)
    got = reader.read_entries(src.layer_entries(0))
    p = "model.language_model." if nested else "model."
    assert torch.equal(got["model.layers.0.mlp.gate_proj.weight"], shard1[f"{p}layers.0.mlp.gate_proj.weight"])
    lm = reader.read_entries({"x": src.globals_entries()["lm_head.weight"]})["x"]
    assert torch.equal(lm, shard2["lm_head.weight"])
    reader.close()

    cfg = load_text_config(d)
    assert cfg.model_type == "qwen3_5_text" and cfg.num_hidden_layers == 2


def test_layer_identity_changes_when_a_shard_changes(tmp_path):
    d = tmp_path / "m"
    _make_hf_checkpoint(d)
    src = HFShardedSource(d)
    before0, before1 = src.layer_identity(0)["sha256"], src.layer_identity(1)["sha256"]
    assert before0 == HFShardedSource(d).layer_identity(0)["sha256"]
    shard = d / "model-00002-of-00002.safetensors"
    data = load_file(str(shard))
    data["model.language_model.layers.1.mlp.gate_proj.weight"] = torch.zeros(128, 64, dtype=torch.bfloat16)
    time.sleep(0.01)
    save_file(data, str(shard))
    os.utime(shard, None)
    after = HFShardedSource(d)
    assert after.layer_identity(0)["sha256"] == before0
    assert after.layer_identity(1)["sha256"] != before1


def test_pread_reader_matches_safetensors(tmp_path):
    tensors = {"a": torch.arange(12, dtype=torch.float32).reshape(3, 4), "b": torch.randn(5, dtype=torch.bfloat16),
               "c": torch.randint(0, 255, (7,), dtype=torch.uint8)}
    f = tmp_path / "layer-x.safetensors"
    save_file(tensors, str(f))
    reader = PreadSafetensorsReader()
    got = reader(str(tmp_path), "layer-x", ["a", "b", "c"])
    for k, v in tensors.items():
        assert got[k].dtype == v.dtype and torch.equal(got[k], v)
    reader.close()


def test_autodetect_skips_unsupported_models(tmp_path):
    models = tmp_path / "models"
    olmoe = models / "OLMoE-1B-7B-0125-Instruct"
    olmoe.mkdir(parents=True)
    (olmoe / "config.json").write_text(json.dumps({"model_type": "olmoe", "num_hidden_layers": 16}), encoding="utf-8")
    save_file({"x": torch.zeros(1)}, str(olmoe / "model.safetensors"))
    with pytest.raises(UnsupportedModelError):
        check_supported(olmoe)
    assert find_local_models(tmp_path) == []
    _make_hf_checkpoint(models / "Qwen3.5-9B")
    assert [p.name for p in find_local_models(tmp_path)] == ["Qwen3.5-9B"]
    assert resolve_model_dir(None, tmp_path).name == "Qwen3.5-9B"
    with pytest.raises(UnsupportedModelError):
        resolve_model_dir(str(olmoe), tmp_path)
