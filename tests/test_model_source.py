"""CPU checks for model discovery and direct Hugging Face shard reading (tiny synthetic checkpoints)."""
import json
import os
import struct
import time

import pytest
import torch
from safetensors.torch import load_file, save_file

from aircolibri_cli.model_source import (
    FAST_KERNEL_FIELDS,
    GGUFModelSource,
    HFShardedSource,
    PreadSafetensorsReader,
    UnsupportedModelError,
    check_supported,
    detect_format,
    find_local_models,
    load_text_config,
    open_model_source,
    resolve_model_dir,
    runtime_profile,
)
from aircolibri_cli.unified_runtime import plan_for_checkpoint

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


def _make_plain_decoder(path, model_type="llama", layers=2):
    path.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": model_type,
        "num_hidden_layers": layers,
        "hidden_size": 32,
        "intermediate_size": 64,
        "vocab_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "tie_word_embeddings": False,
    }
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    tensors = {
        "model.embed_tokens.weight": torch.zeros(16, 32, dtype=torch.bfloat16),
        "model.norm.weight": torch.zeros(32, dtype=torch.bfloat16),
        "lm_head.weight": torch.zeros(16, 32, dtype=torch.bfloat16),
    }
    for i in range(layers):
        tensors[f"model.layers.{i}.input_layernorm.weight"] = torch.zeros(32, dtype=torch.bfloat16)
    save_file(tensors, str(path / "model.safetensors"))


def test_capability_follows_geometry_and_qwen_stays_default(tmp_path):
    models = tmp_path / "models"
    llama = models / "llama-tiny"
    _make_plain_decoder(llama)
    check_supported(llama)
    profile = runtime_profile(llama)
    assert profile["fast_qwen"] is False
    assert profile["backend"] == "hf-bf16"
    assert profile["weight_format"] == "bf16"
    planned, plan, has_mtp = plan_for_checkpoint(
        llama, mode="normal", total_bytes=8 * 2**30, free_bytes=7 * 2**30, enable_mtp=True
    )
    assert planned["mtp_enabled"] is False and has_mtp is False
    assert plan.full_residency is True

    qwen = models / "Qwen3.5-9B"
    _make_hf_checkpoint(qwen)
    qwen_profile = runtime_profile(qwen)
    assert qwen_profile["fast_qwen"] is False
    assert any("hidden_size" in item for item in qwen_profile["why_not_fast"])
    assert resolve_model_dir(None, tmp_path).name == "Qwen3.5-9B"
    assert resolve_model_dir(str(llama), tmp_path).name == "llama-tiny"

    matched = tmp_path / "matched-qwen"
    matched.mkdir()
    cfg = {
        **FAST_KERNEL_FIELDS,
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 32,
        "intermediate_size": 12288,
        "vocab_size": 248320,
        "layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8,
    }
    (matched / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    fast = runtime_profile(matched)
    assert fast["fast_qwen"] is True
    assert fast["backend"] == "qwen35-q4"
    assert fast["weight_format"] == "q4"


def test_gguf_folder_is_detected_and_does_not_become_the_default(tmp_path):
    models = tmp_path / "models"
    gguf = models / "Qwen3.6-35B-A3B-uncensored"
    gguf.mkdir(parents=True)
    (gguf / "weights.gguf").write_bytes(b"not-a-real-gguf")
    assert detect_format(gguf) == "gguf"
    with pytest.raises(UnsupportedModelError):
        check_supported(gguf)
    assert find_local_models(tmp_path) == []
    _make_hf_checkpoint(models / "Qwen3.5-9B")
    assert [p.name for p in find_local_models(tmp_path)] == ["Qwen3.5-9B"]
    assert resolve_model_dir(None, tmp_path).name == "Qwen3.5-9B"


def _pack_q8_0(values: torch.Tensor) -> bytes:
    """Packs a float tensor whose last dim is a multiple of 32 into GGUF Q8_0 blocks."""
    flat = values.detach().float().reshape(-1, 32)
    amax = flat.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 127.0).to(torch.float16)
    safe_d = torch.where(scale.float() > 0, scale.float(), torch.ones_like(scale, dtype=torch.float32))
    qs = torch.clamp(torch.round(flat / safe_d), -128, 127).to(torch.int8)
    scale_u8 = scale.numpy().view("uint8")
    qs_u8 = qs.numpy().view("uint8")
    out = bytearray(flat.shape[0] * 34)
    for i in range(flat.shape[0]):
        out[i * 34 : i * 34 + 2] = scale_u8[i].tobytes()
        out[i * 34 + 2 : (i + 1) * 34] = qs_u8[i].tobytes()
    return bytes(out)


def _write_tiny_qwen35moe_gguf(gguf_path):
    """Writes a minimal valid GGUF v3 ``qwen35moe`` checkpoint (2 layers: linear + full attn, 4 experts)."""
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    b2u = list(bytes_to_unicode().values())
    tokens = b2u + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    vocab = len(tokens)  # 259
    hidden = 32
    head_dim = 16
    n_heads = 2
    n_kv = 1
    k_heads = 2
    v_heads = 2
    k_dim = 16
    v_dim = 16
    conv_k = 4
    conv_channels = k_heads * k_dim * 2 + v_heads * v_dim  # 96
    n_exp = 4
    top_k = 2
    exp_inter = 32
    shared_inter = 32

    meta = [
        ("general.architecture", 8, "qwen35moe"),
        ("general.alignment", 4, 32),
        ("qwen35moe.block_count", 4, 2),
        ("qwen35moe.context_length", 4, 512),
        ("qwen35moe.embedding_length", 4, hidden),
        ("qwen35moe.attention.head_count", 4, n_heads),
        ("qwen35moe.attention.head_count_kv", 4, n_kv),
        ("qwen35moe.attention.key_length", 4, head_dim),
        ("qwen35moe.attention.value_length", 4, head_dim),
        ("qwen35moe.attention.layer_norm_rms_epsilon", 6, 1e-6),
        ("qwen35moe.full_attention_interval", 4, 2),
        ("qwen35moe.ssm.conv_kernel", 4, conv_k),
        ("qwen35moe.ssm.state_size", 4, k_dim),
        ("qwen35moe.ssm.group_count", 4, k_heads),
        ("qwen35moe.ssm.time_step_rank", 4, v_heads),
        ("qwen35moe.ssm.inner_size", 4, v_heads * v_dim),
        ("qwen35moe.rope.freq_base", 6, 10000.0),
        ("qwen35moe.rope.dimension_count", 4, 8),
        ("qwen35moe.rope.dimension_sections", 9, (5, [2, 1, 1, 0])),
        ("qwen35moe.expert_count", 4, n_exp),
        ("qwen35moe.expert_used_count", 4, top_k),
        ("qwen35moe.expert_feed_forward_length", 4, exp_inter),
        ("qwen35moe.expert_shared_feed_forward_length", 4, shared_inter),
        ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.pre", 8, "qwen2"),
        ("tokenizer.ggml.tokens", 9, (8, tokens)),
        ("tokenizer.ggml.merges", 9, (8, [])),
        ("tokenizer.ggml.eos_token_id", 4, vocab - 1),
        (
            "tokenizer.chat_template",
            8,
            "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ),
    ]

    torch.manual_seed(0)
    tensors = []

    def add_f32(name, torch_shape, val=None):
        t = (torch.randn(torch_shape, dtype=torch.float32) * 0.02) if val is None else val.to(torch.float32)
        tensors.append((name, tuple(reversed(torch_shape)), 0, t.contiguous().numpy().tobytes()))

    def add_q8(name, torch_shape):
        t = torch.randn(torch_shape, dtype=torch.float32) * 0.05
        tensors.append((name, tuple(reversed(torch_shape)), 8, _pack_q8_0(t)))

    add_q8("token_embd.weight", (vocab, hidden))
    add_f32("output_norm.weight", (hidden,), torch.full((hidden,), 1.25))
    add_q8("output.weight", (vocab, hidden))

    # Layer 0: linear_attention + MoE
    add_f32("blk.0.attn_norm.weight", (hidden,), torch.ones(hidden))
    add_f32("blk.0.post_attention_norm.weight", (hidden,), torch.ones(hidden))
    add_q8("blk.0.attn_qkv.weight", (conv_channels, hidden))
    add_q8("blk.0.attn_gate.weight", (v_heads * v_dim, hidden))
    add_f32("blk.0.ssm_ba.weight", (v_heads * 2, hidden))  # wait: separate ssm_beta and ssm_alpha in _qwen35!
    tensors.pop()
    add_f32("blk.0.ssm_beta.weight", (v_heads, hidden))
    add_f32("blk.0.ssm_alpha.weight", (v_heads, hidden))
    add_f32("blk.0.ssm_conv1d.weight", (conv_channels, conv_k))
    add_f32("blk.0.ssm_dt.bias", (v_heads,), torch.full((v_heads,), 0.1))
    add_f32("blk.0.ssm_a", (v_heads,), torch.full((v_heads,), -1.5))
    add_f32("blk.0.ssm_norm.weight", (v_dim,), torch.ones(v_dim))
    add_q8("blk.0.ssm_out.weight", (hidden, v_heads * v_dim))

    # Layer 1: full_attention + MoE
    add_f32("blk.1.attn_norm.weight", (hidden,), torch.ones(hidden))
    add_f32("blk.1.post_attention_norm.weight", (hidden,), torch.ones(hidden))
    add_q8("blk.1.attn_q.weight", (n_heads * head_dim * 2, hidden))
    add_q8("blk.1.attn_k.weight", (n_kv * head_dim, hidden))
    add_q8("blk.1.attn_v.weight", (n_kv * head_dim, hidden))
    add_q8("blk.1.attn_output.weight", (hidden, n_heads * head_dim))
    add_f32("blk.1.attn_q_norm.weight", (head_dim,), torch.ones(head_dim))
    add_f32("blk.1.attn_k_norm.weight", (head_dim,), torch.ones(head_dim))

    for l_idx in (0, 1):
        add_f32(f"blk.{l_idx}.ffn_gate_inp.weight", (n_exp, hidden))
        add_f32(f"blk.{l_idx}.ffn_gate_inp_shexp.weight", (hidden,))
        add_q8(f"blk.{l_idx}.ffn_gate_shexp.weight", (shared_inter, hidden))
        add_q8(f"blk.{l_idx}.ffn_up_shexp.weight", (shared_inter, hidden))
        add_q8(f"blk.{l_idx}.ffn_down_shexp.weight", (hidden, shared_inter))
        add_q8(f"blk.{l_idx}.ffn_gate_exps.weight", (n_exp, exp_inter, hidden))
        add_q8(f"blk.{l_idx}.ffn_up_exps.weight", (n_exp, exp_inter, hidden))
        add_q8(f"blk.{l_idx}.ffn_down_exps.weight", (n_exp, hidden, exp_inter))

    def _s(text: str) -> bytes:
        b = text.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _val(vtype: int, v) -> bytes:
        if vtype == 4:
            return struct.pack("<I", int(v))
        if vtype == 5:
            return struct.pack("<i", int(v))
        if vtype == 6:
            return struct.pack("<f", float(v))
        if vtype == 8:
            return _s(str(v))
        if vtype == 9:
            elem_t, items = v
            return struct.pack("<IQ", elem_t, len(items)) + b"".join(_val(elem_t, it) for it in items)
        raise AssertionError(vtype)

    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, vt, val in meta:
        buf += _s(k) + struct.pack("<I", vt) + _val(vt, val)

    offsets = []
    cur = 0
    for _, _, _, raw in tensors:
        cur = (cur + 31) // 32 * 32
        offsets.append(cur)
        cur += len(raw)

    for (name, gguf_dims, gtype, _), off in zip(tensors, offsets):
        buf += _s(name) + struct.pack("<I", len(gguf_dims))
        for d in gguf_dims:
            buf += struct.pack("<Q", int(d))
        buf += struct.pack("<IQ", int(gtype), int(off))

    data_start = (len(buf) + 31) // 32 * 32
    buf += b"\x00" * (data_start - len(buf))
    for (_, _, _, raw), off in zip(tensors, offsets):
        pad = (data_start + off) - len(buf)
        if pad > 0:
            buf += b"\x00" * pad
        buf += raw
    gguf_path.write_bytes(bytes(buf))
    return vocab


def test_tiny_gguf_qwen35moe_source_and_sliced_experts_forward(tmp_path):
    from src.aircolibri_integrated import AirColibriModel

    d = tmp_path / "Qwen3.6-tiny-gguf"
    d.mkdir(parents=True)
    vocab = _write_tiny_qwen35moe_gguf(d / "tiny.gguf")

    assert detect_format(d) == "gguf"
    check_supported(d)
    # Zero sidecar files written into the GGUF folder!
    assert [p.name for p in d.iterdir()] == ["tiny.gguf"]

    prof = runtime_profile(d)
    assert prof["fast_qwen"] is False
    assert prof["backend"] == "hf-bf16"
    assert prof["num_experts"] == 4 and prof["num_experts_per_tok"] == 2
    assert any("num_experts=4" in item for item in prof["why_not_fast"])

    planned, plan, has_mtp = plan_for_checkpoint(
        d, mode="normal", total_bytes=8 * 2**30, free_bytes=7 * 2**30, enable_mtp=True
    )
    assert planned["mtp_enabled"] is False and has_mtp is False
    assert plan.full_residency is True

    src = open_model_source(d)
    assert isinstance(src, GGUFModelSource)
    try:
        l0 = src.layer_entries(0)
        assert "model.layers.0.mlp.experts.gate_up_proj" not in l0
        assert "model.layers.0.mlp.experts.down_proj" not in l0
        assert src.tensor_shape(l0["model.layers.0.linear_attn.conv1d.weight"]) == (96, 1, 4)
        assert src.tensor_shape(l0["model.layers.0.mlp.shared_expert_gate.weight"]) == (1, 32)
        # output_norm stored as 1.25 in GGUF -> SubtractOne yields 0.25 for Qwen3_5MoeRMSNorm (1 + w).
        norm_w = src.open_globals(dtype=torch.bfloat16).pop("model.norm.weight")
        assert torch.allclose(norm_w.float(), torch.full((32,), 0.25), atol=1e-2)
    finally:
        src.close()

    with AirColibriModel(
        str(d),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=torch.device("cpu"),
        weight_format="bf16",
        model_source=open_model_source(d),
    ) as eng:
        assert "gate_up_proj" not in eng.model.model.layers[0].mlp.experts._parameters
        logits, past = eng._forward_step(torch.tensor([[1, 2, 3]], dtype=torch.long), prefetch_next=True)
        assert logits.shape == (1, vocab)
        assert torch.isfinite(logits).all()
        logits2, _ = eng._forward_step(
            torch.tensor([[4]], dtype=torch.long),
            past_key_values=past,
            prefetch_next=False,
            cache_position=torch.tensor([3], dtype=torch.long),
        )
        assert logits2.shape == (1, vocab)
        assert torch.isfinite(logits2).all()

    assert [p.name for p in d.iterdir()] == ["tiny.gguf"]


def test_hf_moe_per_expert_shards_route_on_demand(tmp_path):
    from src.aircolibri_integrated import AirColibriModel

    d = tmp_path / "olmoe-tiny"
    d.mkdir(parents=True)
    cfg = {
        "model_type": "olmoe",
        "num_hidden_layers": 2,
        "hidden_size": 32,
        "intermediate_size": 16,
        "vocab_size": 24,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "tie_word_embeddings": False,
    }
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    # Minimal char-level fast tokenizer so AirColibriModel can instantiate offline.
    tok_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{"id": 0, "content": "<eos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "vocab": {f"t{i}": i for i in range(24)}, "merges": []},
    }
    (d / "tokenizer.json").write_text(json.dumps(tok_json), encoding="utf-8")
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "<eos>",
                "chat_template": "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}<eos>{% endfor %}assistant:",
            }
        ),
        encoding="utf-8",
    )

    torch.manual_seed(1)
    tensors = {
        "model.embed_tokens.weight": torch.randn(24, 32, dtype=torch.bfloat16) * 0.05,
        "model.norm.weight": torch.ones(32, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(24, 32, dtype=torch.bfloat16) * 0.05,
    }
    for i in range(2):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = torch.ones(32, dtype=torch.bfloat16)
        tensors[p + "post_attention_layernorm.weight"] = torch.ones(32, dtype=torch.bfloat16)
        tensors[p + "self_attn.q_proj.weight"] = torch.randn(32, 32, dtype=torch.bfloat16) * 0.05
        tensors[p + "self_attn.k_proj.weight"] = torch.randn(32, 32, dtype=torch.bfloat16) * 0.05
        tensors[p + "self_attn.v_proj.weight"] = torch.randn(32, 32, dtype=torch.bfloat16) * 0.05
        tensors[p + "self_attn.o_proj.weight"] = torch.randn(32, 32, dtype=torch.bfloat16) * 0.05
        tensors[p + "self_attn.q_norm.weight"] = torch.ones(32, dtype=torch.bfloat16)
        tensors[p + "self_attn.k_norm.weight"] = torch.ones(32, dtype=torch.bfloat16)
        tensors[p + "mlp.gate.weight"] = torch.randn(4, 32, dtype=torch.bfloat16) * 0.05
        for e in range(4):
            tensors[f"{p}mlp.experts.{e}.gate_proj.weight"] = torch.randn(16, 32, dtype=torch.bfloat16) * 0.05
            tensors[f"{p}mlp.experts.{e}.up_proj.weight"] = torch.randn(16, 32, dtype=torch.bfloat16) * 0.05
            tensors[f"{p}mlp.experts.{e}.down_proj.weight"] = torch.randn(32, 16, dtype=torch.bfloat16) * 0.05
    save_file(tensors, str(d / "model.safetensors"))

    src = open_model_source(d)
    assert all("mlp.experts." not in k for k in src.layer_entries(0))
    src.close()

    with AirColibriModel(
        str(d),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=torch.device("cpu"),
        weight_format="bf16",
        model_source=open_model_source(d),
    ) as eng:
        logits, _ = eng._forward_step(torch.tensor([[1, 2]], dtype=torch.long), prefetch_next=False)
        assert logits.shape == (1, 24)
        assert torch.isfinite(logits).all()


def test_extended_gguf_dequantizers_and_nvrtc_exactness():
    from transformers.integrations.gguf.dequant import dequantize as hf_dequantize
    from aircolibri_cli.gguf_reader import (
        GGML_TYPE_MXFP4,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q8_K,
    )
    from aircolibri_cli.model_source import dequantize_gguf_tensor

    # Q5_0: 2B fp16 d (0.5) + 4B qh (alternating 0xAA) + 16B qs (0x21) -> 2 blocks = 44 bytes
    d_half = torch.tensor([0.5], dtype=torch.float16).numpy().tobytes()
    q5_0_blk = d_half + bytes([0xAA, 0x55, 0xAA, 0x55]) + bytes([0x21] * 16)
    out_q5_0 = dequantize_gguf_tensor(torch.frombuffer(bytearray(q5_0_blk * 2), dtype=torch.uint8), GGML_TYPE_Q5_0)
    assert out_q5_0.shape == (64,) and torch.isfinite(out_q5_0).all()
    # First nibble lo=1, qh bit0=0 -> (1 - 16)*0.5 = -7.5
    assert abs(float(out_q5_0[0].item()) - (-7.5)) < 1e-4

    # Q5_1: 2B fp16 d (0.5) + 2B fp16 m (1.0) + 4B qh (0) + 16B qs (0x32)
    m_half = torch.tensor([1.0], dtype=torch.float16).numpy().tobytes()
    q5_1_blk = d_half + m_half + bytes([0, 0, 0, 0]) + bytes([0x32] * 16)
    out_q5_1 = dequantize_gguf_tensor(torch.frombuffer(bytearray(q5_1_blk), dtype=torch.uint8), GGML_TYPE_Q5_1)
    assert out_q5_1.shape == (32,) and abs(float(out_q5_1[0].item()) - 2.0) < 1e-4

    # Q8_K: 4B fp32 d (0.25) + 256B int8 qs (all 4) + 32B int16 bsums -> 292B
    q8_k_blk = struct.pack("<f", 0.25) + bytes([4] * 256) + bytes(32)
    out_q8_k = dequantize_gguf_tensor(torch.frombuffer(bytearray(q8_k_blk), dtype=torch.uint8), GGML_TYPE_Q8_K)
    assert out_q8_k.shape == (256,) and torch.allclose(out_q8_k.float(), torch.ones(256), atol=1e-5)

    # MXFP4: 1B E8M0 e=128 (d = 2^(128-128) = 1.0, lut doubled -> E2M1 idx 2 is 2*0.5=1.0 -> 2.0/2? Wait:
    # e=127 gives ldexp(1.0, 127-128) = 0.5; lut[2]=2 -> 2 * 0.5 = 1.0)
    mxfp4_blk = bytes([127]) + bytes([0x22] * 16)
    out_mxfp4 = dequantize_gguf_tensor(torch.frombuffer(bytearray(mxfp4_blk), dtype=torch.uint8), GGML_TYPE_MXFP4)
    assert out_mxfp4.shape == (32,) and torch.allclose(out_mxfp4.float(), torch.ones(32), atol=1e-5)

    if torch.cuda.is_available():
        torch.manual_seed(42)
        dev = torch.device("cuda:0")
        for gtype, block_bytes in (
            (GGML_TYPE_Q8_0, 34),
            (GGML_TYPE_Q4_0, 18),
            (GGML_TYPE_Q4_K, 144),
            (GGML_TYPE_Q5_K, 176),
            (GGML_TYPE_Q6_K, 210),
        ):
            u8 = torch.randint(0, 256, (64 * block_bytes,), dtype=torch.uint8, device=dev)
            # Keep fp16 scale headers finite so random bit patterns never hit fp16 NaN/Inf
            v = u8.view(64, block_bytes)
            if gtype in (GGML_TYPE_Q8_0, GGML_TYPE_Q4_0):
                v[:, 1] &= 0x3F
            elif gtype in (GGML_TYPE_Q4_K, GGML_TYPE_Q5_K):
                v[:, 1] &= 0x3F
                v[:, 3] &= 0x3F
            elif gtype == GGML_TYPE_Q6_K:
                v[:, 209] &= 0x3F
            got = dequantize_gguf_tensor(u8, gtype, dtype=torch.bfloat16)
            ref = hf_dequantize(u8, gtype, dtype=torch.bfloat16)
            assert torch.equal(got, ref), f"NVRTC mismatch for ggml_type={gtype}"


def test_gguf_zero_mmap_and_mid_layer_stream_cancel(tmp_path, monkeypatch):
    import mmap
    import numpy as np
    from aircolibri_cli.model_source import load_runtime_config, load_runtime_tokenizer
    from aircolibri_cli.unified_runtime import GenerationCancelled
    from src.aircolibri_integrated import AirColibriModel

    d = tmp_path / "gguf-nommap"
    d.mkdir(parents=True)
    _write_tiny_qwen35moe_gguf(d / "tiny.gguf")

    def _forbid_mmap(*_a, **_kw):
        raise AssertionError("mmap / np.memmap must never be called on GGUF files")

    monkeypatch.setattr(mmap, "mmap", _forbid_mmap)
    monkeypatch.setattr(np, "memmap", _forbid_mmap)

    cfg = load_runtime_config(d)
    tok = load_runtime_tokenizer(d, config=cfg)
    assert tok.eos_token_id == 258

    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    with AirColibriModel(
        str(d),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=dev,
        weight_format="bf16",
        model_source=open_model_source(d),
        expert_cache_bytes=128 * 1024 if dev.type == "cuda" else 0,
    ) as eng:
        # Cancel right inside layer 1 during streamed forward, then verify clean ring reset + recovery
        seen = {"layers": 0}

        def _cancel_on_second_layer():
            seen["layers"] += 1
            if seen["layers"] >= 2:
                raise GenerationCancelled("decode")

        eng._step_cancel_check = _cancel_on_second_layer
        with pytest.raises(GenerationCancelled):
            eng._forward_step(torch.tensor([[1, 2, 3]], dtype=torch.long), prefetch_next=True)
        assert eng._failed is False
        eng.reset_after_cancel()

        lg_a, _ = eng._forward_step(torch.tensor([[1, 2, 3]], dtype=torch.long), prefetch_next=False)
        lg_b, _ = eng._forward_step(torch.tensor([[1, 2, 3]], dtype=torch.long), prefetch_next=False)
        assert torch.equal(lg_a, lg_b)
        if dev.type == "cuda":
            stats = eng.model_source.expert_cache_stats()
            assert stats is not None and stats["slots"] > 0 and stats["hits"] > 0

    assert [p.name for p in d.iterdir()] == ["tiny.gguf"]


def test_bf16_session_prefix_reuse_on_gguf_and_olmoe(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("AirColibriUnifiedEngine requires CUDA")
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine

    # Tiny GGUF qwen35moe hybrid: verify Turn 1 -> park -> Turn 2 unpark reuses prefix and matches
    # running the exact same full Turn-2 token sequence from scratch (cached_prefix_len=0).
    gguf_dir = tmp_path / "gguf-chat"
    gguf_dir.mkdir(parents=True)
    _write_tiny_qwen35moe_gguf(gguf_dir / "tiny.gguf")
    with AirColibriUnifiedEngine(cpu_first=False, model_dir=gguf_dir, vram_mode="normal", verbose=False) as ueng:
        m1 = [{"role": "user", "content": "hi"}]
        r1 = [
            e
            for e in ueng.stream_chat(
                m1, max_new_tokens=3, enable_thinking=False, use_session_cache=True, stop_on_eos=False
            )
            if e["type"] == "done"
        ][0]
        assert len(ueng.session.fed) > 0 and ueng.session.parked_backbone is not None

        m2 = m1 + [{"role": "assistant", "content": r1["answer_text"]}, {"role": "user", "content": "ok"}]
        full_ids, reused, past_kv, _ = ueng._prepare_prompt_and_cache(
            m2, enable_thinking=False, use_session_cache=True
        )
        assert reused == len(ueng.session.fed) and reused > 0 and past_kv is not None

        with torch.no_grad():
            lg_unparked, _ = ueng.engine._forward_step(
                torch.tensor([full_ids[reused:]], dtype=torch.long),
                past_key_values=past_kv,
                prefetch_next=False,
                cache_position=torch.arange(reused, len(full_ids)),
            )
            # Live incremental cache without CPU parking must match unparked cache bit-for-bit (0.0 diff)
            _, p = ueng.engine._forward_step(
                torch.tensor([full_ids[: reused - 2]], dtype=torch.long),
                past_key_values=None,
                prefetch_next=False,
                cache_position=torch.arange(0, reused - 2),
            )
            _, p = ueng.engine._forward_step(
                torch.tensor([[full_ids[reused - 2]]], dtype=torch.long),
                past_key_values=p,
                prefetch_next=False,
                cache_position=torch.tensor([reused - 2]),
            )
            _, p = ueng.engine._forward_step(
                torch.tensor([[full_ids[reused - 1]]], dtype=torch.long),
                past_key_values=p,
                prefetch_next=False,
                cache_position=torch.tensor([reused - 1]),
            )
            lg_live, _ = ueng.engine._forward_step(
                torch.tensor([full_ids[reused:]], dtype=torch.long),
                past_key_values=p,
                prefetch_next=False,
                cache_position=torch.arange(reused, len(full_ids)),
            )
            lg_zero, _ = ueng.engine._forward_step(
                torch.tensor([full_ids], dtype=torch.long),
                past_key_values=None,
                prefetch_next=False,
                cache_position=torch.arange(0, len(full_ids)),
            )
        assert torch.equal(lg_unparked, lg_live)
        cos = torch.nn.functional.cosine_similarity(lg_unparked.float(), lg_zero.float(), dim=-1).item()
        assert cos > 0.995, cos
    assert [p.name for p in gguf_dir.iterdir()] == ["tiny.gguf"]


def test_staging_pool_coalesce_tasks():
    from aircolibri_cli.model_source import _ParallelSliceStagingPool

    tasks = [
        ("a.gguf", 100, 0, 64),
        ("a.gguf", 164, 64, 64),
        ("a.gguf", 228, 128, 128),
        ("a.gguf", 400, 256, 64),  # file offset gap -> split
        ("a.gguf", 464, 320, 200),  # exceeds max_chunk_bytes=200 when merged -> split
        ("b.gguf", 664, 520, 32),  # different file -> split
    ]
    merged = _ParallelSliceStagingPool._coalesce_tasks(tasks, max_chunk_bytes=256)
    assert merged == [
        ("a.gguf", 100, 0, 256),
        ("a.gguf", 400, 256, 64),
        ("a.gguf", 464, 320, 200),
        ("b.gguf", 664, 520, 32),
    ]


def test_nvrtc_deltanet_kernels_and_zero_fallback_warnings(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("NVRTC GatedDeltaNet kernels require CUDA")

    import inspect
    import logging
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as q35m
    from src.aircolibri_integrated import AirColibriModel
    from src.colibri_deltanet_nvrtc import (
        KERNEL_CALL_STATS,
        nvrtc_causal_conv1d_fn,
        nvrtc_causal_conv1d_update,
        nvrtc_chunk_gated_delta_rule,
        nvrtc_recurrent_gated_delta_rule,
        nvrtc_rmsnorm_gated_silu,
        nvrtc_rmsnorm_zero_centered,
    )

    dev = torch.device("cuda:0")
    torch.manual_seed(42)

    # Compare against unwrapped PyTorch reference functions from transformers
    def original_reference(fn):
        # Works both before and after NVRTC installation; never compare a kernel
        # against its own wrapper or assume the position of a default argument.
        fallback = inspect.signature(fn).parameters.get("_fb")
        original = inspect.unwrap(fallback.default if fallback is not None else fn)
        assert original.__module__.startswith("transformers."), original
        return original

    ref_conv_fn = original_reference(q35m.causal_conv1d_fn)
    ref_conv_upd = original_reference(q35m.causal_conv1d_update)
    ref_rec = original_reference(q35m.torch_recurrent_gated_delta_rule)
    ref_chk = original_reference(q35m.torch_chunk_gated_delta_rule)

    # 1) Non-contiguous transposed CausalConv1d prefill & update (exact BF16 match)
    x_nc = torch.randn(2, 130, 64, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    w_cv = torch.randn(64, 1, 4, device=dev, dtype=torch.bfloat16)
    b_cv = torch.randn(64, device=dev, dtype=torch.bfloat16)
    got_cv = nvrtc_causal_conv1d_fn(x_nc, w_cv.squeeze(1), bias=b_cv, activation="silu")
    exp_cv = ref_conv_fn(x_nc, w_cv.squeeze(1), bias=b_cv, activation="silu")
    assert torch.equal(got_cv, exp_cv)

    st_a = torch.randn(2, 64, 4, device=dev, dtype=torch.bfloat16)
    st_b = st_a.clone()
    x_step = torch.randn(2, 1, 64, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    got_up = nvrtc_causal_conv1d_update(x_step, st_a, w_cv.squeeze(1), bias=b_cv, activation="silu")
    exp_up = ref_conv_upd(x_step, st_b, w_cv.squeeze(1), bias=b_cv, activation="silu")
    assert torch.equal(got_up, exp_up)
    assert torch.equal(st_a, st_b)

    # 2) Register-resident D=128 GatedDeltaNet (chunk & recurrent) + generic D=16
    for D in (128, 16):
        q = torch.randn(1, 96, 2, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, 96, 2, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, 96, 4, D, device=dev, dtype=torch.bfloat16)
        g = -torch.rand(1, 96, 4, device=dev, dtype=torch.float32) * 0.5
        beta = torch.sigmoid(torch.randn(1, 96, 4, device=dev, dtype=torch.bfloat16))
        out_nv, st_nv = nvrtc_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        out_rf, st_rf = ref_chk(
            q.repeat_interleave(2, dim=2),
            k.repeat_interleave(2, dim=2),
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert (out_nv.float() - out_rf.float()).abs().max().item() < 1e-3
        assert (st_nv - st_rf).abs().max().item() < 1e-5

        q1, k1, v1 = q[:, :1], k[:, :1], v[:, :1]
        g1, b1 = g[:, :1], beta[:, :1]
        o1_nv, s1_nv = nvrtc_recurrent_gated_delta_rule(
            q1, k1, v1, g=g1, beta=b1, initial_state=st_rf, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        o1_rf, s1_rf = ref_rec(
            q1.repeat_interleave(2, dim=2),
            k1.repeat_interleave(2, dim=2),
            v1,
            g=g1,
            beta=b1,
            initial_state=st_rf,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert (o1_nv.float() - o1_rf.float()).abs().max().item() < 1e-3
        assert (s1_nv - s1_rf).abs().max().item() < 1e-5

    # 3) Fused RMSNormGated + zero-centered RMSNorm (exact BF16 match)
    rn_g = q35m.Qwen3_5MoeRMSNormGated(128, eps=1e-6).to(device=dev, dtype=torch.bfloat16)
    rn_z = q35m.Qwen3_5MoeRMSNorm(128, eps=1e-6).to(device=dev, dtype=torch.bfloat16)
    hx = torch.randn(4, 128, device=dev, dtype=torch.bfloat16)
    gt = torch.randn(4, 128, device=dev, dtype=torch.bfloat16)
    assert torch.equal(nvrtc_rmsnorm_gated_silu(hx, gt, rn_g.weight, 1e-6), rn_g(hx, gt))
    assert torch.equal(nvrtc_rmsnorm_zero_centered(hx, rn_z.weight, 1e-6), rn_z(hx))

    # 4) End-to-end synthetic GGUF Qwen3.5-MoE prefill + decode emits zero fallback warnings
    gguf_dir = tmp_path / "gguf-nvrtc-warncheck"
    gguf_dir.mkdir(parents=True)
    _write_tiny_qwen35moe_gguf(gguf_dir / "tiny.gguf")

    captured: list[str] = []

    class _WarnCatcher(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    tf_logger = logging.getLogger("transformers")
    catcher = _WarnCatcher(level=logging.WARNING)
    tf_logger.addHandler(catcher)
    before = dict(KERNEL_CALL_STATS)
    try:
        with AirColibriModel(
            str(gguf_dir),
            pipeline_mode="triple_ring_prefetch",
            vram_resident_layers=2,
            num_slots=2,
            device=dev,
            weight_format="bf16",
            model_source=open_model_source(gguf_dir),
            expert_cache_bytes=128 * 1024,
        ) as eng:
            _, pkv = eng._forward_step(
                torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
                past_key_values=None,
                prefetch_next=False,
                cache_position=torch.arange(0, 6),
            )
            _, _ = eng._forward_step(
                torch.tensor([[7]], dtype=torch.long),
                past_key_values=pkv,
                prefetch_next=False,
                cache_position=torch.tensor([6]),
            )
    finally:
        tf_logger.removeHandler(catcher)

    fallback_msgs = [m for m in captured if "falling back to its reference PyTorch implementation" in m]
    assert fallback_msgs == [], f"Unexpected transformers fallback warnings: {fallback_msgs}"
    for key in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "rmsnorm_gated",
        "rmsnorm_zero_centered",
    ):
        assert KERNEL_CALL_STATS[key] > before[key], f"Kernel {key} was not invoked during model forward"


def _write_tiny_hf_tokenizer(folder, vocab_size: int = 64, *, with_chat_template: bool = True):
    tok_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "<eos>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": {f"t{i}": i for i in range(vocab_size)},
            "merges": [],
        },
    }
    (folder / "tokenizer.json").write_text(json.dumps(tok_json), encoding="utf-8")
    tcfg = {"eos_token": "<eos>"}
    if with_chat_template:
        tcfg["chat_template"] = "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}<eos>{% endfor %}assistant:"
    (folder / "tokenizer_config.json").write_text(json.dumps(tcfg), encoding="utf-8")


def _serialize_gguf_v3(gguf_path, meta_items, tensor_items):
    """Writes a valid GGUF v3 file from ``meta_items=[(key, vtype, val)]`` and ``tensor_items=[(name, torch_t)]``."""

    def _s(text: str) -> bytes:
        b = text.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _val(vtype: int, v) -> bytes:
        if vtype == 4:
            return struct.pack("<I", int(v))
        if vtype == 5:
            return struct.pack("<i", int(v))
        if vtype == 6:
            return struct.pack("<f", float(v))
        if vtype == 8:
            return _s(str(v))
        if vtype == 9:
            elem_t, items = v
            return struct.pack("<IQ", elem_t, len(items)) + b"".join(_val(elem_t, it) for it in items)
        raise AssertionError(vtype)

    encoded = []
    for name, t in tensor_items:
        arr = t.detach().to(torch.float32).contiguous()
        gguf_dims = tuple(reversed(arr.shape)) if arr.ndim > 0 else (1,)
        encoded.append((name, gguf_dims, 0, arr.numpy().tobytes()))

    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<IQQ", 3, len(encoded), len(meta_items))
    for k, vt, val in meta_items:
        buf += _s(k) + struct.pack("<I", vt) + _val(vt, val)

    offsets = []
    cur = 0
    for _, _, _, raw in encoded:
        cur = (cur + 31) // 32 * 32
        offsets.append(cur)
        cur += len(raw)

    for (name, gguf_dims, gtype, _), off in zip(encoded, offsets):
        buf += _s(name) + struct.pack("<I", len(gguf_dims))
        for d in gguf_dims:
            buf += struct.pack("<Q", int(d))
        buf += struct.pack("<IQ", int(gtype), int(off))

    data_start = (len(buf) + 31) // 32 * 32
    buf += b"\x00" * (data_start - len(buf))
    for (_, _, _, raw), off in zip(encoded, offsets):
        pad = (data_start + off) - len(buf)
        if pad > 0:
            buf += b"\x00" * pad
        buf += raw
    gguf_path.write_bytes(bytes(buf))


def _gguf_bpe_meta(vocab_size: int = 259, *, with_template: bool = True, with_merges: bool = True):
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    b2u = list(bytes_to_unicode().values())
    extra = [f"<tok_{i}>" for i in range(max(0, vocab_size - len(b2u)))]
    tokens = (b2u + extra)[:vocab_size]
    items = [
        ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.pre", 8, "gpt2"),
        ("tokenizer.ggml.tokens", 9, (8, tokens)),
        ("tokenizer.ggml.eos_token_id", 4, len(tokens) - 1),
    ]
    if with_merges:
        items.append(("tokenizer.ggml.merges", 9, (8, [])))
    if with_template:
        items.append(
            (
                "tokenizer.chat_template",
                8,
                "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant: ",
            )
        )
    return items, len(tokens)


def test_ternary_tq1_tq2_and_q8_1_dequantizers():
    from aircolibri_cli.model_source import (
        GGML_TYPE_Q8_1,
        GGML_TYPE_TQ1_0,
        GGML_TYPE_TQ2_0,
        dequantize_gguf_tensor,
    )

    # Q8_1: 32 int8 + fp16 d + fp16 s -> 36 bytes (note layout: d fp16 [2B], s fp16 [2B], qs int8[32])
    d_h = torch.tensor([0.5], dtype=torch.float16).numpy().tobytes()
    s_h = torch.tensor([8.0], dtype=torch.float16).numpy().tobytes()
    qs = bytes([(i - 16) & 0xFF for i in range(32)])
    q81 = dequantize_gguf_tensor(torch.frombuffer(bytearray(d_h + s_h + qs), dtype=torch.uint8), GGML_TYPE_Q8_1)
    expected_q81 = torch.tensor([(i - 16) * 0.5 for i in range(32)], dtype=torch.bfloat16)
    assert q81.shape == (32,) and torch.equal(q81, expected_q81)

    # TQ2_0: 64 bytes of 2-bit trits (0->-1, 1->0, 2->+1) + 2B fp16 d -> 66 bytes for 256 elems
    # Byte 0x90 = (0 << 0) | (0 << 2) | (1 << 4) | (2 << 6) -> lane0=-1, lane1=-1, lane2=0, lane3=+1
    d_tq = torch.tensor([2.0], dtype=torch.float16).numpy().tobytes()
    tq2_raw = bytes([0x90] * 64) + d_tq
    tq2 = dequantize_gguf_tensor(torch.frombuffer(bytearray(tq2_raw), dtype=torch.uint8), GGML_TYPE_TQ2_0)
    assert tq2.shape == (256,)
    assert torch.equal(tq2[:32].float(), torch.full((32,), -2.0))
    assert torch.equal(tq2[64:96].float(), torch.zeros(32))
    assert torch.equal(tq2[96:128].float(), torch.full((32,), 2.0))

    # TQ1_0: 48B qs + 4B qh + 2B fp16 d -> 54 bytes for 256 elems; all-zero bytes map to trit 0 -> -1 * d
    tq1_zero = bytes(48) + bytes(4) + d_tq
    tq1 = dequantize_gguf_tensor(torch.frombuffer(bytearray(tq1_zero), dtype=torch.uint8), GGML_TYPE_TQ1_0)
    assert tq1.shape == (256,)
    assert torch.equal(tq1.float(), torch.full((256,), -2.0))


def test_gpt2_hf_custom_safetensors_and_gguf_exact_logit_parity(tmp_path):
    from transformers import AutoModelForCausalLM, GPT2Config
    from src.aircolibri_integrated import AirColibriModel

    torch.manual_seed(7)
    vocab = 259
    cfg = GPT2Config(
        vocab_size=vocab,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=64,
        n_positions=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=vocab - 2,
        eos_token_id=vocab - 1,
    )
    ref = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for p in ref.parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn_like(p) * 0.04)
            else:
                p.copy_(torch.randn_like(p) * 0.02 + (1.0 if p.shape == (32,) else 0.0))
        out_pre = ref(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
        ref_pre = out_pre.logits[:, -1, :].float()
        out_dec = ref(torch.tensor([[5]]), past_key_values=out_pre.past_key_values, use_cache=True)
        ref_dec = out_dec.logits[:, -1, :].float()

    # 1) HF checkpoint with custom single-file safetensors name (not model.safetensors)
    hf_dir = tmp_path / "gpt2-hf-custom"
    hf_dir.mkdir(parents=True)
    cfg.save_pretrained(hf_dir)
    _write_tiny_hf_tokenizer(hf_dir, vocab_size=vocab)
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items() if not k.endswith(".attn.bias")}
    # Tied lm_head omitted from disk to verify tied-embedding resolution
    sd.pop("lm_head.weight", None)
    save_file(sd, str(hf_dir / "custom_gpt2_weights.safetensors"))

    with AirColibriModel(
        str(hf_dir),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=torch.device("cpu"),
        weight_format="bf16",
        model_source=open_model_source(hf_dir),
    ) as eng:
        lg_pre, past = eng._forward_step(
            torch.tensor([[1, 2, 3, 4]]),
            past_key_values=None,
            prefetch_next=True,
            cache_position=torch.arange(0, 4),
        )
        lg_dec, _ = eng._forward_step(
            torch.tensor([[5]]),
            past_key_values=past,
            prefetch_next=False,
            cache_position=torch.tensor([4]),
        )
    assert (lg_pre.float() - ref_pre).abs().max().item() < 1e-2
    assert (lg_dec.float() - ref_dec).abs().max().item() < 1e-2
    assert int(lg_pre.argmax()) == int(ref_pre.argmax())
    assert int(lg_dec.argmax()) == int(ref_dec.argmax())

    # 2) GGUF gpt2 checkpoint (llama.cpp stores Conv1D 2D weights transposed as [out, in])
    gguf_dir = tmp_path / "gpt2-gguf"
    gguf_dir.mkdir(parents=True)
    tok_meta, _ = _gguf_bpe_meta(vocab)
    meta = [
        ("general.architecture", 8, "gpt2"),
        ("gpt2.block_count", 4, 2),
        ("gpt2.context_length", 4, 64),
        ("gpt2.embedding_length", 4, 32),
        ("gpt2.feed_forward_length", 4, 64),
        ("gpt2.attention.head_count", 4, 4),
        ("gpt2.attention.layer_norm_epsilon", 6, float(cfg.layer_norm_epsilon)),
        *tok_meta,
    ]
    g_tensors = [
        ("token_embd.weight", sd["transformer.wte.weight"]),
        ("position_embd.weight", sd["transformer.wpe.weight"]),
        ("output_norm.weight", sd["transformer.ln_f.weight"]),
        ("output_norm.bias", sd["transformer.ln_f.bias"]),
    ]
    for i in range(2):
        p = f"transformer.h.{i}."
        g_tensors.extend(
            [
                (f"blk.{i}.attn_norm.weight", sd[p + "ln_1.weight"]),
                (f"blk.{i}.attn_norm.bias", sd[p + "ln_1.bias"]),
                (f"blk.{i}.attn_qkv.weight", sd[p + "attn.c_attn.weight"].t().contiguous()),
                (f"blk.{i}.attn_qkv.bias", sd[p + "attn.c_attn.bias"]),
                (f"blk.{i}.attn_output.weight", sd[p + "attn.c_proj.weight"].t().contiguous()),
                (f"blk.{i}.attn_output.bias", sd[p + "attn.c_proj.bias"]),
                (f"blk.{i}.ffn_norm.weight", sd[p + "ln_2.weight"]),
                (f"blk.{i}.ffn_norm.bias", sd[p + "ln_2.bias"]),
                (f"blk.{i}.ffn_up.weight", sd[p + "mlp.c_fc.weight"].t().contiguous()),
                (f"blk.{i}.ffn_up.bias", sd[p + "mlp.c_fc.bias"]),
                (f"blk.{i}.ffn_down.weight", sd[p + "mlp.c_proj.weight"].t().contiguous()),
                (f"blk.{i}.ffn_down.bias", sd[p + "mlp.c_proj.bias"]),
            ]
        )
    _serialize_gguf_v3(gguf_dir / "gpt2.gguf", meta, g_tensors)

    with AirColibriModel(
        str(gguf_dir),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=torch.device("cpu"),
        weight_format="bf16",
        model_source=open_model_source(gguf_dir),
    ) as eng_g:
        gg_pre, g_past = eng_g._forward_step(
            torch.tensor([[1, 2, 3, 4]]),
            past_key_values=None,
            prefetch_next=True,
            cache_position=torch.arange(0, 4),
        )
        gg_dec, _ = eng_g._forward_step(
            torch.tensor([[5]]),
            past_key_values=g_past,
            prefetch_next=False,
            cache_position=torch.tensor([4]),
        )
    assert (gg_pre.float() - ref_pre).abs().max().item() < 1e-2
    assert (gg_dec.float() - ref_dec).abs().max().item() < 1e-2
    assert int(gg_pre.argmax()) == int(ref_pre.argmax())
    assert int(gg_dec.argmax()) == int(ref_dec.argmax())


def test_mamba_hf_and_gguf_recurrent_cache_parity(tmp_path):
    from transformers import AutoModelForCausalLM, MambaConfig
    from aircolibri_cli.unified_runtime import ColibriKVPrefixSession
    from src.aircolibri_integrated import AirColibriModel

    torch.manual_seed(11)
    vocab = 259
    cfg = MambaConfig(
        vocab_size=vocab,
        hidden_size=32,
        state_size=16,
        num_hidden_layers=2,
        conv_kernel=4,
        expand=2,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )
    ref = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "A_log" in name:
                p.copy_(torch.linspace(0.5, 1.5, p.numel(), dtype=p.dtype).reshape(p.shape))
            elif p.ndim >= 2:
                p.copy_(torch.randn_like(p) * 0.04)
            else:
                p.copy_(torch.ones_like(p) * 0.5)
        r0 = ref(torch.tensor([[1, 2, 3]]), use_cache=True)
        ref_pre = r0.logits[:, -1, :].float()
        r1 = ref(torch.tensor([[4]]), cache_params=r0.cache_params, use_cache=True)
        ref_dec = r1.logits[:, -1, :].float()

    hf_dir = tmp_path / "mamba-hf"
    hf_dir.mkdir(parents=True)
    cfg.save_pretrained(hf_dir)
    _write_tiny_hf_tokenizer(hf_dir, vocab_size=vocab)
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    sd.pop("lm_head.weight", None)
    save_file(sd, str(hf_dir / "model.safetensors"))

    with AirColibriModel(
        str(hf_dir),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=torch.device("cpu"),
        weight_format="bf16",
        model_source=open_model_source(hf_dir),
    ) as eng:
        lg_pre, c_params = eng._forward_step(torch.tensor([[1, 2, 3]]), past_key_values=None, prefetch_next=True)
        lg_dec, c_params = eng._forward_step(torch.tensor([[4]]), past_key_values=c_params, prefetch_next=False)
        sess = ColibriKVPrefixSession()
        sess.record_turn_end(
            config=eng.config,
            all_ids_full=[1, 2, 3, 4],
            past_key_values=c_params,
            last_fed_h_post=None,
            engine=eng,
            mtp_head=None,
            captured={},
        )
        assert sess.tainted is True and sess.kv_prefix_reuse([1, 2, 3, 4, 5]) == 0
    assert (lg_pre.float() - ref_pre).abs().max().item() < 1e-2
    assert (lg_dec.float() - ref_dec).abs().max().item() < 1e-2

    # GGUF mamba (ssm_conv1d squeezed to 2D, ssm_a = -exp(A_log))
    gguf_dir = tmp_path / "mamba-gguf"
    gguf_dir.mkdir(parents=True)
    tok_meta, _ = _gguf_bpe_meta(vocab)
    meta = [
        ("general.architecture", 8, "mamba"),
        ("mamba.block_count", 4, 2),
        ("mamba.context_length", 4, 128),
        ("mamba.embedding_length", 4, 32),
        ("mamba.ssm.conv_kernel", 4, 4),
        ("mamba.ssm.state_size", 4, 16),
        ("mamba.ssm.inner_size", 4, 64),
        ("mamba.ssm.time_step_rank", 4, int(cfg.time_step_rank)),
        ("mamba.attention.layer_norm_rms_epsilon", 6, float(cfg.layer_norm_epsilon)),
        *tok_meta,
    ]
    gt = [
        ("token_embd.weight", sd["backbone.embeddings.weight"]),
        ("output_norm.weight", sd["backbone.norm_f.weight"]),
    ]
    for i in range(2):
        p = f"backbone.layers.{i}."
        gt.extend(
            [
                (f"blk.{i}.attn_norm.weight", sd[p + "norm.weight"]),
                (f"blk.{i}.ssm_in.weight", sd[p + "mixer.in_proj.weight"]),
                (f"blk.{i}.ssm_conv1d.weight", sd[p + "mixer.conv1d.weight"].squeeze(1)),
                (f"blk.{i}.ssm_conv1d.bias", sd[p + "mixer.conv1d.bias"]),
                (f"blk.{i}.ssm_x.weight", sd[p + "mixer.x_proj.weight"]),
                (f"blk.{i}.ssm_dt.weight", sd[p + "mixer.dt_proj.weight"]),
                (f"blk.{i}.ssm_dt.bias", sd[p + "mixer.dt_proj.bias"]),
                (f"blk.{i}.ssm_a", -torch.exp(sd[p + "mixer.A_log"].float())),
                (f"blk.{i}.ssm_d", sd[p + "mixer.D"]),
                (f"blk.{i}.ssm_out.weight", sd[p + "mixer.out_proj.weight"]),
            ]
        )
    _serialize_gguf_v3(gguf_dir / "mamba.gguf", meta, gt)

    with AirColibriModel(
        str(gguf_dir),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=torch.device("cpu"),
        weight_format="bf16",
        model_source=open_model_source(gguf_dir),
    ) as eng_g:
        gg_pre, gp = eng_g._forward_step(torch.tensor([[1, 2, 3]]), past_key_values=None, prefetch_next=True)
        gg_dec, _ = eng_g._forward_step(torch.tensor([[4]]), past_key_values=gp, prefetch_next=False)
    assert (gg_pre.float() - ref_pre).abs().max().item() < 1e-2
    assert (gg_dec.float() - ref_dec).abs().max().item() < 1e-2


def test_vlm_gemma3_unwrapping_embed_scale_and_gguf_parity(tmp_path):
    from transformers import AutoModelForCausalLM, Gemma3TextConfig
    from src.aircolibri_integrated import AirColibriModel

    torch.manual_seed(19)
    vocab = 259
    tcfg = Gemma3TextConfig(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref = AutoModelForCausalLM.from_config(tcfg, dtype=torch.bfloat16).to(dev).eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn_like(p) * 0.04)
            else:
                p.copy_(torch.randn_like(p) * 0.05)
        r_pre = ref(torch.tensor([[1, 3, 4]], device=dev), use_cache=True)
        ref_pre = r_pre.logits[:, -1, :].float().cpu()

    # Save as multimodal Gemma3 VLM checkpoint with vision_tower + multi_modal_projector
    vlm_dir = tmp_path / "gemma3-vlm"
    vlm_dir.mkdir(parents=True)
    vlm_cfg = {
        "model_type": "gemma3",
        "text_config": tcfg.to_dict(),
        "vision_config": {"model_type": "siglip_vision_model", "hidden_size": 16, "num_hidden_layers": 1},
    }
    (vlm_dir / "config.json").write_text(json.dumps(vlm_cfg), encoding="utf-8")
    _write_tiny_hf_tokenizer(vlm_dir, vocab_size=vocab)
    sd_cpu = {k: v.detach().cpu() for k, v in ref.state_dict().items()}
    vlm_sd = {}
    for k, v in sd_cpu.items():
        if k == "lm_head.weight":
            continue  # tied
        vlm_sd[f"language_model.{k}"] = v
    vlm_sd["vision_tower.vision_model.embeddings.patch_embedding.weight"] = torch.randn(16, 3, 4, 4, dtype=torch.bfloat16)
    vlm_sd["multi_modal_projector.mm_input_projection_weight"] = torch.randn(16, 32, dtype=torch.bfloat16)
    save_file(vlm_sd, str(vlm_dir / "model.safetensors"))

    with AirColibriModel(
        str(vlm_dir),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=dev,
        weight_format="bf16",
        model_source=open_model_source(vlm_dir),
    ) as eng:
        lg_pre, _ = eng._forward_step(
            torch.tensor([[1, 3, 4]]),
            past_key_values=None,
            prefetch_next=False,
            cache_position=torch.arange(0, 3),
        )
    assert (lg_pre.float().cpu() - ref_pre).abs().max().item() < 1e-2
    assert int(lg_pre.argmax()) == int(ref_pre.argmax())


def test_variable_shape_gemma4_and_coexisting_falcon_h1_ring_streaming(tmp_path):
    from transformers import AutoModelForCausalLM, FalconH1Config, Gemma4TextConfig
    from src.aircolibri_integrated import AirColibriModel

    torch.manual_seed(23)
    # 1) Gemma4: sliding_attention (head_dim=8) vs full_attention (global_head_dim=16) -> different q_proj shapes!
    g4_cfg = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        global_num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    g4_ref = AutoModelForCausalLM.from_config(g4_cfg, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for p in g4_ref.parameters():
            p.copy_(torch.randn_like(p) * 0.04 if p.ndim >= 2 else torch.ones_like(p) * 0.2)
        g4_exp = g4_ref(torch.tensor([[1, 2, 3, 4]])).logits[:, -1, :].float()

    g4_dir = tmp_path / "gemma4-varshape"
    g4_dir.mkdir(parents=True)
    g4_cfg.save_pretrained(g4_dir)
    _write_tiny_hf_tokenizer(g4_dir, vocab_size=64)
    g4_sd = {k: v.detach().clone() for k, v in g4_ref.state_dict().items()}
    assert g4_sd["model.layers.0.self_attn.q_proj.weight"].shape != g4_sd["model.layers.1.self_attn.q_proj.weight"].shape
    save_file(g4_sd, str(g4_dir / "model.safetensors"))

    with AirColibriModel(
        str(g4_dir),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=torch.device("cpu"),
        weight_format="bf16",
        model_source=open_model_source(g4_dir),
    ) as eng_g4:
        g4_got, _ = eng_g4._forward_step(
            torch.tensor([[1, 2, 3, 4]]),
            past_key_values=None,
            prefetch_next=False,
            cache_position=torch.arange(0, 4),
        )
    assert (g4_got.float() - g4_exp).abs().max().item() < 1e-2
    assert int(g4_got.argmax()) == int(g4_exp.argmax())

    # 2) FalconH1: BOTH mamba.* AND self_attn.* coexist in the SAME layer!
    fh_cfg = FalconH1Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        mamba_d_ssm=32,
        mamba_n_heads=4,
        mamba_headdim=8,
        mamba_n_groups=1,
        mamba_d_state=16,
        mamba_d_conv=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    fh_ref = AutoModelForCausalLM.from_config(fh_cfg, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for name, p in fh_ref.named_parameters():
            if "A_log" in name:
                p.copy_(torch.ones_like(p))
            elif p.ndim >= 2:
                p.copy_(torch.randn_like(p) * 0.04)
            else:
                p.copy_(torch.ones_like(p) * 0.5)
        fh_exp = fh_ref(torch.tensor([[1, 2, 3]])).logits[:, -1, :].float()

    fh_dir = tmp_path / "falcon-h1"
    fh_dir.mkdir(parents=True)
    fh_cfg.save_pretrained(fh_dir)
    _write_tiny_hf_tokenizer(fh_dir, vocab_size=64)
    save_file({k: v.detach().clone() for k, v in fh_ref.state_dict().items()}, str(fh_dir / "model.safetensors"))

    with AirColibriModel(
        str(fh_dir),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=torch.device("cpu"),
        weight_format="bf16",
        model_source=open_model_source(fh_dir),
    ) as eng_fh:
        assert eng_fh.weight_manager.pinned_pool.layout.overlay_prefixes == ()
        fh_got, _ = eng_fh._forward_step(
            torch.tensor([[1, 2, 3]]),
            past_key_values=None,
            prefetch_next=False,
            cache_position=torch.arange(0, 3),
        )
    assert (fh_got.float() - fh_exp).abs().max().item() < 1e-2
    assert int(fh_got.argmax()) == int(fh_exp.argmax())


def test_hf_moe_all_layouts_and_routing_buffers_exact_parity(tmp_path):
    from transformers import GraniteMoeConfig
    from transformers import AutoModelForCausalLM, DeepseekV3Config, GptOssConfig, Llama4TextConfig, MixtralConfig
    from src.aircolibri_integrated import AirColibriModel

    torch.manual_seed(31)

    def _verify_moe(name: str, cfg, transform_sd=None, init_extra=None):
        ref = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for p in ref.parameters():
                p.copy_(torch.randn_like(p) * 0.05 if p.ndim >= 2 else torch.ones_like(p) * 0.3)
            if init_extra is not None:
                init_extra(ref)
            exp = ref(torch.tensor([[1, 2, 3, 4]])).logits[:, -1, :].float()
        sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
        if transform_sd is not None:
            sd = transform_sd(sd)
        d = tmp_path / name
        d.mkdir(parents=True)
        cfg.save_pretrained(d)
        _write_tiny_hf_tokenizer(d, vocab_size=64)
        save_file(sd, str(d / "model.safetensors"))
        with AirColibriModel(
            str(d),
            pipeline_mode="triple_ring_prefetch",
            vram_resident_layers=0,
            num_slots=2,
            device=torch.device("cpu"),
            weight_format="bf16",
            model_source=open_model_source(d),
            expert_cache_bytes=64 * 1024,
        ) as eng:
            # Confirm routed experts never bloated ring slots
            assert all(".experts." not in k for k in eng.weight_manager.pinned_pool.layout.specs)
            got, _ = eng._forward_step(
                torch.tensor([[1, 2, 3, 4]]),
                past_key_values=None,
                prefetch_next=False,
                cache_position=torch.arange(0, 4),
            )
        diff = (got.float() - exp).abs().max().item()
        assert diff < 2e-2, f"{name} max logit diff={diff}"
        assert int(got.argmax()) == int(exp.argmax()), f"{name} argmax mismatch"

    # a) Mixtral stored in Hub legacy per-expert w1/w2/w3 under .block_sparse_moe.
    def _to_hub_mixtral(sd):
        out = {}
        for k, v in sd.items():
            if ".mlp.gate.weight" in k:
                out[k.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")] = v
            elif k.endswith(".mlp.experts.gate_up_proj"):
                base = k[: -len(".mlp.experts.gate_up_proj")] + ".block_sparse_moe.experts"
                inter = v.shape[1] // 2
                for e in range(v.shape[0]):
                    out[f"{base}.{e}.w1.weight"] = v[e, :inter].contiguous()
                    out[f"{base}.{e}.w3.weight"] = v[e, inter:].contiguous()
            elif k.endswith(".mlp.experts.down_proj"):
                base = k[: -len(".mlp.experts.down_proj")] + ".block_sparse_moe.experts"
                for e in range(v.shape[0]):
                    out[f"{base}.{e}.w2.weight"] = v[e].contiguous()
            else:
                out[k] = v
        return out

    _verify_moe(
        "mixtral-hub",
        MixtralConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            pad_token_id=0,
        ),
        transform_sd=_to_hub_mixtral,
    )

    # Granite checkpoints use the old input/output_linear and router.layer names.
    # All three must map before expert stripping or resident loading.
    def _to_hub_granite(sd):
        return {k.replace("block_sparse_moe.experts.gate_up_proj", "block_sparse_moe.input_linear.weight")
                 .replace("block_sparse_moe.experts.down_proj", "block_sparse_moe.output_linear.weight")
                 .replace("block_sparse_moe.router.weight", "block_sparse_moe.router.layer.weight"): v
                for k, v in sd.items()}

    _verify_moe("granite-hub", GraniteMoeConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                              num_local_experts=4, num_experts_per_tok=2, pad_token_id=0),
                transform_sd=_to_hub_granite)

    # b) Llama4 transposed 3D packed experts [E, H, 2*I] + input-weighted sigmoid routing
    _verify_moe(
        "llama4-3d-transposed",
        Llama4TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            intermediate_size_mlp=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_local_experts=4,
            num_experts_per_tok=2,
            interleave_moe_layer_step=1,
            pad_token_id=0,
        ),
    )

    # c) GptOss transposed biased 3D packed experts + interleaved clamped GLU
    _verify_moe(
        "gptoss-3d-biased",
        GptOssConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_local_experts=4,
            num_experts_per_tok=2,
            layer_types=["sliding_attention", "full_attention"],
            pad_token_id=0,
        ),
    )

    # d) DeepseekV3 MLA + routed experts + non-zero e_score_correction_bias buffer
    def _init_dsv3_bias(model):
        for m in model.modules():
            if hasattr(m, "e_score_correction_bias"):
                m.e_score_correction_bias.copy_(torch.tensor([0.8, -0.5, 0.3, -0.6], dtype=m.e_score_correction_bias.dtype))

    _verify_moe(
        "deepseek-v3-mla",
        DeepseekV3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            n_group=2,
            topk_group=1,
            first_k_dense_replace=0,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            pad_token_id=0,
        ),
        init_extra=_init_dsv3_bias,
    )


def test_gguf_llama_rope_permute_split_shards_and_chat_fallback(tmp_path):
    from transformers import AutoModelForCausalLM, LlamaConfig
    from aircolibri_cli.gguf_arch_registry import find_gguf_files
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine, ColibriKVPrefixSession
    from src.aircolibri_integrated import AirColibriModel

    torch.manual_seed(41)
    vocab = 259
    n_heads = 4
    n_kv = 2
    head_dim = 8
    hidden = 32
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=vocab - 1,
        pad_token_id=0,
    )
    ref = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p) * 0.05 if p.ndim >= 2 else torch.ones_like(p) * 0.8)
        r_pre = ref(torch.tensor([[1, 5, 9, 12]]), use_cache=True)
        ref_pre = r_pre.logits[:, -1, :].float()
        r_dec = ref(torch.tensor([[15]]), past_key_values=r_pre.past_key_values, use_cache=True)
        ref_dec = r_dec.logits[:, -1, :].float()

    def _to_gguf_interleaved_rope(w: torch.Tensor, heads: int) -> torch.Tensor:
        # Inverse of LlamaRopeReversePermute: HF [heads, 2, hd//2, in] -> GGUF [heads, hd//2, 2, in]
        hd = (w.shape[0] // heads)
        return w.view(heads, 2, hd // 2, w.shape[1]).transpose(1, 2).reshape_as(w).contiguous()

    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    gguf_dir = tmp_path / "llama-split-gguf"
    gguf_dir.mkdir(parents=True)
    # Place an mmproj vision file alongside split shards; it must be ignored automatically.
    (gguf_dir / "mmproj-model-f16.gguf").write_bytes(b"not-a-text-gguf")

    tok_meta, _ = _gguf_bpe_meta(vocab, with_template=False, with_merges=False)
    meta_shard1 = [
        ("general.architecture", 8, "llama"),
        ("llama.block_count", 4, 2),
        ("llama.context_length", 4, 128),
        ("llama.embedding_length", 4, hidden),
        ("llama.feed_forward_length", 4, 64),
        ("llama.attention.head_count", 4, n_heads),
        ("llama.attention.head_count_kv", 4, n_kv),
        ("llama.attention.layer_norm_rms_epsilon", 6, 1e-5),
        ("llama.rope.freq_base", 6, 10000.0),
        ("llama.rope.dimension_count", 4, head_dim),
        *tok_meta,
    ]
    shard1_tensors = [
        ("token_embd.weight", sd["model.embed_tokens.weight"]),
        ("blk.0.attn_norm.weight", sd["model.layers.0.input_layernorm.weight"]),
        ("blk.0.attn_q.weight", _to_gguf_interleaved_rope(sd["model.layers.0.self_attn.q_proj.weight"], n_heads)),
        ("blk.0.attn_k.weight", _to_gguf_interleaved_rope(sd["model.layers.0.self_attn.k_proj.weight"], n_kv)),
        ("blk.0.attn_v.weight", sd["model.layers.0.self_attn.v_proj.weight"]),
        ("blk.0.attn_output.weight", sd["model.layers.0.self_attn.o_proj.weight"]),
        ("blk.0.ffn_norm.weight", sd["model.layers.0.post_attention_layernorm.weight"]),
        ("blk.0.ffn_gate.weight", sd["model.layers.0.mlp.gate_proj.weight"]),
        ("blk.0.ffn_up.weight", sd["model.layers.0.mlp.up_proj.weight"]),
        ("blk.0.ffn_down.weight", sd["model.layers.0.mlp.down_proj.weight"]),
    ]
    shard2_tensors = [
        ("output_norm.weight", sd["model.norm.weight"]),
        ("output.weight", sd["lm_head.weight"]),
        ("blk.1.attn_norm.weight", sd["model.layers.1.input_layernorm.weight"]),
        ("blk.1.attn_q.weight", _to_gguf_interleaved_rope(sd["model.layers.1.self_attn.q_proj.weight"], n_heads)),
        ("blk.1.attn_k.weight", _to_gguf_interleaved_rope(sd["model.layers.1.self_attn.k_proj.weight"], n_kv)),
        ("blk.1.attn_v.weight", sd["model.layers.1.self_attn.v_proj.weight"]),
        ("blk.1.attn_output.weight", sd["model.layers.1.self_attn.o_proj.weight"]),
        ("blk.1.ffn_norm.weight", sd["model.layers.1.post_attention_layernorm.weight"]),
        ("blk.1.ffn_gate.weight", sd["model.layers.1.mlp.gate_proj.weight"]),
        ("blk.1.ffn_up.weight", sd["model.layers.1.mlp.up_proj.weight"]),
        ("blk.1.ffn_down.weight", sd["model.layers.1.mlp.down_proj.weight"]),
    ]
    _serialize_gguf_v3(gguf_dir / "tiny-llama-00001-of-00002.gguf", meta_shard1, shard1_tensors)
    _serialize_gguf_v3(
        gguf_dir / "tiny-llama-00002-of-00002.gguf",
        [("general.architecture", 8, "llama")],
        shard2_tensors,
    )

    found = [p.name for p in find_gguf_files(gguf_dir)]
    assert found == ["tiny-llama-00001-of-00002.gguf", "tiny-llama-00002-of-00002.gguf"]

    with AirColibriModel(
        str(gguf_dir),
        pipeline_mode="triple_ring_prefetch",
        vram_resident_layers=0,
        num_slots=2,
        device=torch.device("cpu"),
        weight_format="bf16",
        model_source=open_model_source(gguf_dir),
    ) as eng:
        lg_pre, past = eng._forward_step(
            torch.tensor([[1, 5, 9, 12]]),
            past_key_values=None,
            prefetch_next=True,
            cache_position=torch.arange(0, 4),
        )
        lg_dec, _ = eng._forward_step(
            torch.tensor([[15]]),
            past_key_values=past,
            prefetch_next=False,
            cache_position=torch.tensor([4]),
        )
        # Verify template-less prompt fallback on AirColibriUnifiedEngine._prepare_prompt_and_cache
        assert getattr(eng.tokenizer, "chat_template", None) in (None, "")
        dummy = object.__new__(AirColibriUnifiedEngine)
        from types import SimpleNamespace
        dummy.plan = SimpleNamespace(max_context_tokens=4096)
        dummy.engine = eng
        dummy.session = ColibriKVPrefixSession()
        dummy._session_messages = []
        dummy.mtp_head = None
        dummy.device = torch.device("cpu")
        p_ids, reused, p_kv, _ = dummy._prepare_prompt_and_cache(
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
            enable_thinking=False,
            use_session_cache=False,
        )
        assert len(p_ids) > 4 and reused == 0 and p_kv is None

    assert (lg_pre.float() - ref_pre).abs().max().item() < 1e-2
    assert (lg_dec.float() - ref_dec).abs().max().item() < 1e-2
    assert int(lg_pre.argmax()) == int(ref_pre.argmax())
    assert int(lg_dec.argmax()) == int(ref_dec.argmax())


