"""Numerical and shared-mechanism checks for the universal packed runtime."""
import copy
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from src.quantized_projection import PackedLinear, pack_weight, unpack_weight

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("bits", [4, 6])
def test_packing_is_independent_of_cpu_or_gpu_placement(bits):
    torch.manual_seed(518)
    # FP32 source checkpoints expose rounding differences hidden by BF16 inputs.
    weight = torch.randn(2048, 768, dtype=torch.float32) * .02
    cpu = pack_weight(weight, bits, device="cpu")
    gpu = pack_weight(weight, bits, device="cuda")
    for key in cpu:
        assert torch.equal(cpu[key], gpu[key].cpu()), key


@pytest.mark.parametrize("bits", [4, 6])
@pytest.mark.parametrize("group", [16, 32, 64])
def test_dense_kernel_matches_independent_affine_reference(bits, group):
    torch.manual_seed(721)
    weight = torch.randn(137, 256, device="cuda", dtype=torch.bfloat16) * .1
    packed = pack_weight(weight, bits, group)
    # Independent scalar-layout decoding: catches nibble order, high-bit layout,
    # scale and zero application errors rather than comparing two kernel calls.
    cols = torch.arange(256, device="cuda")
    q = ((packed["packed_lo"][:, cols//2].int() >> ((cols%2)*4)) & 15)
    if bits == 6:
        q |= (((packed["packed_hi"][:, cols//4].int() >> ((cols%4)*2)) & 3) << 4)
    ref_w = q.float() * packed["qscale"][:, cols//group].float() + packed["qzero"][:, cols//group].float()
    assert torch.equal(ref_w, unpack_weight(packed, group, torch.float32))
    mod = PackedLinear(256, 137, bits, bias=True, group_size=group)
    for key, value in packed.items():
        setattr(mod, key, value)
    mod.bias = torch.nn.Parameter(torch.randn(137, device="cuda", dtype=torch.bfloat16)*.01, requires_grad=False)
    for rows in (1, 4, 8, 11):
        x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        actual = mod(x)
        if rows <= 8:
            expected = F.linear(x.float(), ref_w, mod.bias.float()).to(torch.bfloat16)
            torch.testing.assert_close(actual, expected, rtol=.008, atol=.016)
        else:
            expected = F.linear(x, ref_w.to(x.dtype), mod.bias)
            assert torch.equal(actual, expected)
    assert mod.fused_calls == 3 and mod.tiled_calls == 1


class FakeExpertSource:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.uniform_shapes = self.has_gate = True
        self.I = self.inter_dim = 128
        self.H = self.hidden_dim = 128
        self._spans = {(layer, i): None for layer in range(2) for i in range(8)}
        self.weights = torch.randn(2, 8, 3, 128, 128, device=self.device, dtype=torch.bfloat16) * .04
        self.reads = []

    def load_batch(self, layer, ids, device, dtype, allow_evict):
        self.reads.extend((layer, i) for i in ids)
        w = self.weights[layer, ids].to(dtype)
        return w[:, 0], w[:, 1], w[:, 2]

    def stats(self):
        return {"reads": len(self.reads)}

    def close(self):
        self.weights = None


@pytest.mark.parametrize("bits", [4, 6])
def test_routed_kernel_cache_eviction_prefill_and_math_guard(bits):
    from src.quantized_experts import QuantizedExpertCache
    from aircolibri_cli.model_source import _fast_moe_dispatch
    torch.manual_seed(67)
    source = FakeExpertSource()
    # Deliberately eight slots for sixteen experts: force eviction and cross-layer keys.
    expert_bytes = 3*128*128*(bits/8 + 4/64)
    budget = int(8*6*128*128 + 32*1024**2 + 8*expert_bytes)
    cache = QuantizedExpertCache(source, bits, budget)
    module = torch.nn.Module()
    module.act_fn = torch.nn.SiLU()
    cache.configure_layer(0, module)
    cache.configure_layer(1, module)
    bad = torch.nn.Module()
    bad.act_fn = torch.nn.GELU()
    cache.configure_layer(2, bad)
    assert cache.enabled_layers == {0, 1} and cache.num_slots == 8
    x = torch.randn(1, 128, device=source.device, dtype=torch.bfloat16)
    idx = torch.tensor([[1, 3, 6]], device=source.device)
    weights = torch.tensor([[.2, .3, .5]], device=source.device, dtype=torch.bfloat16)
    actual = cache.try_fused(0, x, idx, weights, 8)
    reads = len(source.reads)
    again = cache.try_fused(0, x, idx, weights, 8)
    assert torch.equal(actual, again) and len(source.reads) == reads
    g, u, d = cache.load_batch(0, [1, 3, 6], source.device, torch.float32, True)
    mid = F.silu(torch.matmul(g, x[0].float())) * torch.matmul(u, x[0].float())
    ref = (torch.bmm(d, mid.unsqueeze(-1)).squeeze(-1) * weights[0].float().unsqueeze(-1)).sum(0, keepdim=True).to(x.dtype)
    torch.testing.assert_close(actual, ref, rtol=.008, atol=.004)
    prefill = _fast_moe_dispatch(module, x.repeat(11, 1), idx.repeat(11, 1), weights.repeat(11, 1), 8,
                               lambda ids, dev, dt, ev: cache.load_batch(0, ids, dev, dt, ev))
    assert F.cosine_similarity(prefill[0].float(), actual[0].float(), dim=0) > .999
    saved_keys = list(cache._lru)
    cache.load_batch(1, list(range(8)), source.device, x.dtype, False)
    assert set(saved_keys) <= set(cache._lru)  # long-prefill misses cannot evict hot experts
    cache.load_batch(1, list(range(8)), source.device, x.dtype, True)
    assert all(layer == 1 for layer, _ in cache._lru)
    restored = cache.try_fused(0, x, idx, weights, 8)
    assert torch.equal(restored, actual)
    assert cache.stats()["conversions"] > 8 and cache.stats()["fused_calls"] == 3
    cache.close()


@pytest.mark.parametrize("bits", [4, 6])
def test_unified_ring_experts_session_cancel_and_recovery(tmp_path, monkeypatch, bits):
    from dataclasses import replace
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, OlmoeConfig
    from test_model_source import _write_tiny_hf_tokenizer
    from aircolibri_cli import unified_runtime as runtime
    from aircolibri_cli.model_source import open_model_source
    from src.aircolibri_integrated import AirColibriModel
    from src.quantized_projection import projection_stats
    torch.manual_seed(17)
    folder = tmp_path / "olmoe"
    folder.mkdir()
    cfg = OlmoeConfig(vocab_size=259, hidden_size=128, intermediate_size=128,
                      num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=4,
                      num_experts=8, num_experts_per_tok=2, tie_word_embeddings=False, eos_token_id=0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()
    cfg.save_pretrained(folder)
    _write_tiny_hf_tokenizer(folder, vocab_size=259)
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    tok_path = folder / "tokenizer.json"
    tok_data = json.loads(tok_path.read_text(encoding="utf-8"))
    tok_data["model"]["vocab"] = {"<eos>": 0, **{c: i+1 for i, c in enumerate(bytes_to_unicode().values())}}
    tok_path.write_text(json.dumps(tok_data), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(folder / "model.safetensors"))
    del model
    resident_logits = []
    with AirColibriModel(str(folder), weight_format=f"q{bits}", universal_quantization=True,
                         vram_resident_layers=5, model_source=open_model_source(folder),
                         expert_cache_bytes=64*1024**2) as resident:
        past = None
        for tokens in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]):
            logits, past = resident._forward_step(torch.tensor([tokens]), past_key_values=past, prefetch_next=True)
            resident_logits.append(logits.clone())
        assert projection_stats(resident.model)["fused_calls"] > 0
    original_plan = runtime.plan_for_checkpoint

    def force_ring(*args, **kwargs):
        profile, plan, mtp = original_plan(*args, **kwargs)
        return profile, replace(plan, full_residency=False, resident_layers=1, ring_slots=3), mtp

    monkeypatch.setattr(runtime, "plan_for_checkpoint", force_ring)
    with runtime.AirColibriUnifiedEngine(model_dir=folder, weight_format=f"q{bits}", verbose=False,
                                        max_context_tokens=256, prefill_chunk_tokens=16) as engine:
        assert engine.engine.num_slots == 3
        assert engine.engine.tail_to_head_prefetch
        past = None
        for j, tokens in enumerate(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13])):
            logits, past = engine.engine._forward_step(torch.tensor([tokens]), past_key_values=past, prefetch_next=True)
            torch.testing.assert_close(logits, resident_logits[j], rtol=0, atol=0)
        assert engine.engine.weight_manager.ram_cache is not None
        stats = engine.expert_cache_stats()
        assert stats["enabled_layers"] == list(range(5))
        assert stats["fused_calls"] > 0 and stats["prefill_batches"] > 0 and stats["hits"] > 0
        assert not any(".experts." in k for k in engine.engine.weight_manager.pinned_pool.layout.specs)
        # Random weights can emit incomplete UTF-8 bytes, whose text round trip
        # correctly disables reuse. Keep this cache test on a lossless ASCII token.
        ascii_id = engine.tokenizer.encode("a", add_special_tokens=False)[0]

        def ascii_logits(module, inputs, output):
            out = torch.full_like(output, -float("inf"))
            out[..., ascii_id] = 0
            return out

        engine.engine.model.get_output_embeddings().register_forward_hook(ascii_logits)
        msgs = [{"role": "user", "content": "hi"}]
        done = [e for e in engine.stream_chat(msgs, max_new_tokens=3, enable_thinking=False,
                                             use_session_cache=True, stop_on_eos=False) if e["type"] == "done"][0]
        msgs += [{"role": "assistant", "content": done["answer_text"]}, {"role": "user", "content": "ok"}]
        _, reused, parked, _ = engine._prepare_prompt_and_cache(msgs, False, True)
        assert reused > 0 and parked is not None
        checks = [0]

        def cancel():
            checks[0] += 1
            return checks[0] > 3

        with pytest.raises(runtime.GenerationCancelled):
            list(engine.stream_chat(msgs, max_new_tokens=8, cancel_check=cancel, use_session_cache=False))
        assert all(slot is None for slot in engine.engine.active_slot_per_layer)
        after = [e for e in engine.stream_chat(msgs, max_new_tokens=3, enable_thinking=False,
                                              use_session_cache=False) if e["type"] == "done"]
        assert after and engine.generations_cancelled == 1
