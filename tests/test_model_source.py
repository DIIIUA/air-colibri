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
    with AirColibriUnifiedEngine(model_dir=gguf_dir, vram_mode="normal", verbose=False) as ueng:
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
    ref_conv_fn = inspect.unwrap(getattr(q35m.causal_conv1d_fn, "__defaults__", (None,))[0] or q35m.causal_conv1d_fn)
    ref_conv_upd = inspect.unwrap(
        getattr(q35m.causal_conv1d_update, "__defaults__", (None, None, None))[2] or q35m.causal_conv1d_update
    )
    ref_rec = inspect.unwrap(
        getattr(q35m.torch_recurrent_gated_delta_rule, "__defaults__", (None, False, False, None))[-1]
        or q35m.torch_recurrent_gated_delta_rule
    )
    ref_chk = inspect.unwrap(
        getattr(q35m.torch_chunk_gated_delta_rule, "__defaults__", (64, None, False, False, None))[-1]
        or q35m.torch_chunk_gated_delta_rule
    )

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

