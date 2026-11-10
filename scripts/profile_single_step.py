"""Deep profiling of a single generation step in AirColibri on DeepSeek-V4-9B.

Measures:
1. H2D / RAM lookup / set_module_tensor_to_device overhead
2. input_layernorm compute
3. Attention / DeltaNet compute (linear_attn vs self_attn)
4. post_attention_layernorm compute
5. MLP compute
6. _post_hook / meta deallocation overhead
7. LM Head projection & sampling
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from accelerate.utils import set_module_tensor_to_device
from src.aircolibri_integrated import AirColibriModel

MODEL_DIR = r"C:\air-colibri\models\deepseek-v4-9b-colibri"


def profile_decode_step():
    print("=" * 85)
    print("AirColibri Deep Step Profiler: DeepSeek-V4-9B (32 layers)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 85)

    engine = AirColibriModel(
        model_dir=MODEL_DIR,
        vram_resident_layers=4,
        dtype=torch.bfloat16
    )

    prompt = "The capital of France is"
    tokenizer = engine.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(engine.device)

    # 1. Warmup / Prefill step
    print("\n[Profiler] Running Prefill step (warming up caches)...")
    with torch.no_grad():
        outputs = engine.model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()

    curr_input = torch.tensor([[next_token]], device=engine.device)

    # 2. Detailed Profiling of Decode Step 1
    print("\n[Profiler] Starting detailed micro-profiling of Decode Step 1...")

    layer_stats = []
    
    # We will attach sub-hooks to each layer's components to measure exact CUDA and CPU times
    for idx, layer in enumerate(engine.model.model.layers):
        stat = {
            "layer_idx": idx,
            "layer_type": "full_attention" if idx % 4 == 3 else "linear_attention",
            "is_vram_resident": idx < engine.vram_resident_layers,
            "pre_hook_ms": 0.0,
            "input_norm_ms": 0.0,
            "attn_ms": 0.0,
            "post_norm_ms": 0.0,
            "mlp_ms": 0.0,
            "post_hook_ms": 0.0,
            "total_layer_ms": 0.0
        }
        layer_stats.append(stat)

        # Instrument pre-hook
        orig_pre = layer._forward_pre_hooks
        # Instrument submodules
        def make_timer(stat_dict, key):
            def _pre(m, a):
                torch.cuda.synchronize()
                m._t0 = time.perf_counter()
            def _post(m, a, o):
                torch.cuda.synchronize()
                stat_dict[key] = (time.perf_counter() - m._t0) * 1000.0
            return _pre, _post

        pre_n1, post_n1 = make_timer(stat, "input_norm_ms")
        layer.input_layernorm.register_forward_pre_hook(pre_n1)
        layer.input_layernorm.register_forward_hook(post_n1)

        attn_module = layer.self_attn if idx % 4 == 3 else layer.linear_attn
        pre_att, post_att = make_timer(stat, "attn_ms")
        attn_module.register_forward_pre_hook(pre_att)
        attn_module.register_forward_hook(post_att)

        pre_n2, post_n2 = make_timer(stat, "post_norm_ms")
        layer.post_attention_layernorm.register_forward_pre_hook(pre_n2)
        layer.post_attention_layernorm.register_forward_hook(post_n2)

        pre_mlp, post_mlp = make_timer(stat, "mlp_ms")
        layer.mlp.register_forward_pre_hook(pre_mlp)
        layer.mlp.register_forward_hook(post_mlp)

    # Also wrap layer pre_hook and post_hook to measure data transfer vs dealloc
    for idx, layer in enumerate(engine.model.model.layers):
        stat = layer_stats[idx]
        
        # We wrap the layer itself
        orig_fwd = layer.forward
        def make_layer_fwd(orig_f, st):
            def _wrapped(*args, **kwargs):
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                out = orig_f(*args, **kwargs)
                torch.cuda.synchronize()
                st["total_layer_ms"] = (time.perf_counter() - t_start) * 1000.0
                return out
            return _wrapped
        layer.forward = make_layer_fwd(orig_fwd, stat)

    # Now run the single decode step
    torch.cuda.synchronize()
    t0_step = time.perf_counter()

    with torch.no_grad():
        outputs = engine.model(
            input_ids=curr_input,
            past_key_values=past_key_values,
            use_cache=True
        )
        torch.cuda.synchronize()
        t_lm_head_start = time.perf_counter()
        logits = outputs.logits[:, -1, :]
        tok_out = torch.argmax(logits, dim=-1).item()
        torch.cuda.synchronize()
        lm_head_ms = (time.perf_counter() - t_lm_head_start) * 1000.0

    total_step_ms = (time.perf_counter() - t0_step) * 1000.0

    print(f"\n[Profiler] Step complete: Total Time = {total_step_ms:.1f} ms | Token = {tok_out} ('{tokenizer.decode([tok_out])}')")
    print(f"[Profiler] LM Head + Argmax Time = {lm_head_ms:.1f} ms")

    # Aggregate statistics
    total_attn_ms = sum(s["attn_ms"] for s in layer_stats)
    total_linear_attn_ms = sum(s["attn_ms"] for s in layer_stats if s["layer_type"] == "linear_attention")
    total_full_attn_ms = sum(s["attn_ms"] for s in layer_stats if s["layer_type"] == "full_attention")
    total_mlp_ms = sum(s["mlp_ms"] for s in layer_stats)
    total_norm_ms = sum(s["input_norm_ms"] + s["post_norm_ms"] for s in layer_stats)
    total_compute_ms = total_attn_ms + total_mlp_ms + total_norm_ms
    
    # Pre-hook transfer and hook overhead is difference between total_layer_ms and compute
    for s in layer_stats:
        s["hook_and_transfer_overhead_ms"] = max(0.0, s["total_layer_ms"] - (s["input_norm_ms"] + s["attn_ms"] + s["post_norm_ms"] + s["mlp_ms"]))
    total_hook_overhead_ms = sum(s["hook_and_transfer_overhead_ms"] for s in layer_stats)

    report = {
        "total_step_ms": total_step_ms,
        "lm_head_ms": lm_head_ms,
        "summary": {
            "linear_attention_24_layers_ms": total_linear_attn_ms,
            "linear_attention_pct": (total_linear_attn_ms / total_step_ms) * 100.0,
            "full_attention_8_layers_ms": total_full_attn_ms,
            "full_attention_pct": (total_full_attn_ms / total_step_ms) * 100.0,
            "total_mlp_32_layers_ms": total_mlp_ms,
            "total_mlp_pct": (total_mlp_ms / total_step_ms) * 100.0,
            "total_norm_32_layers_ms": total_norm_ms,
            "total_norm_pct": (total_norm_ms / total_step_ms) * 100.0,
            "hook_transfer_meta_overhead_ms": total_hook_overhead_ms,
            "hook_transfer_meta_pct": (total_hook_overhead_ms / total_step_ms) * 100.0,
            "pure_compute_total_ms": total_compute_ms,
            "pure_compute_pct": (total_compute_ms / total_step_ms) * 100.0
        },
        "layers": layer_stats
    }

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
    out_json = os.path.join(os.path.dirname(__file__), "..", "data", "step_profiling_breakdown.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 85)
    print("PROFILING BREAKDOWN SUMMARY:")
    print(f"Total Decode Step Time:             {total_step_ms:9.1f} ms (100.0%)")
    print(f"  1. Gated DeltaNet (24 layers):    {total_linear_attn_ms:9.1f} ms ({report['summary']['linear_attention_pct']:5.1f}%)")
    print(f"  2. MLP Forward (32 layers):       {total_mlp_ms:9.1f} ms ({report['summary']['total_mlp_pct']:5.1f}%)")
    print(f"  3. Hook + Transfer + Meta Overh.: {total_hook_overhead_ms:9.1f} ms ({report['summary']['hook_transfer_meta_pct']:5.1f}%)")
    print(f"  4. Full Attention (8 layers):     {total_full_attn_ms:9.1f} ms ({report['summary']['full_attention_pct']:5.1f}%)")
    print(f"  5. RMSNorm (32 layers):           {total_norm_ms:9.1f} ms ({report['summary']['total_norm_pct']:5.1f}%)")
    print(f"  6. LM Head & Sampling:            {lm_head_ms:9.1f} ms ({lm_head_ms/total_step_ms*100.0:5.1f}%)")
    print("=" * 85)
    print(f"Full per-layer JSON report written to: {out_json}")


if __name__ == "__main__":
    profile_decode_step()
