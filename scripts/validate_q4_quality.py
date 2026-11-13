"""Bounded, resumable-by-phase Qwen Q4/BF16 quality experiment.

Reference and candidate run in separate processes on the same fixed text.
An observer captures all logits from the existing lm_head, without replacing
model mathematics. This diagnostic is not a standard language-quality benchmark.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
import torch
import transformers

from src.aircolibri_integrated import AirColibriModel
from src.cli import default_airllm_source
from src.quality_metrics import compare_logits, summarize
from src.resource_plan import make_plan, GIB


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ResourceUnavailable(RuntimeError):
    """Startup refused before allocating model resources."""


def identity(model, prompts, manifest, airllm):
    return dict(
        runtime_sources={p.name:sha(p) for p in sorted((ROOT / "src").glob("*.py"))},
        checkpoint_metadata={p.name:sha(p) for p in (model/"config.json", model/"tokenizer.json")},
        checkpoint_files={p.name:dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                          for p in sorted(model.glob("model-*.safetensors"))},
        trusted_layer_manifest_sha256=sha(manifest), corpus_sha256=sha(prompts),
        airllm_reader_sha256=sha(Path(airllm)/"air_llm/airllm/utils.py"),
        torch=torch.__version__, transformers=transformers.__version__,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["reference", "q4", "bf16-check"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--model", type=Path, default=ROOT/"models/deepseek-v4-9b-colibri")
    p.add_argument("--prompts", type=Path, default=ROOT/"tests/fixtures/quality_prompts_v1.json")
    p.add_argument("--steps", type=int, default=5, help="Number of teacher-forced chunks")
    p.add_argument("--chunk-tokens", type=int, default=16)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--group-size", type=int, choices=[16,32,64], default=64)
    p.add_argument("--airllm", default=default_airllm_source())
    p.add_argument("--manifest", type=Path, default=ROOT/"data/integration-20261112/trusted-checkpoint-manifest.json")
    p.add_argument("--q4-cache", type=Path, default=ROOT/"data/q4-g64-cache")
    p.add_argument("--ram-gib", type=float, default=4)
    p.add_argument("--vram-gib", type=float, default=5.5)
    a = p.parse_args()
    if min(a.steps, a.chunk_tokens, a.batch) < 1:
        p.error("Steps, prompt length and batch must be positive")
    if a.mode != "reference" and a.reference is None:
        p.error("Candidate requires --reference")
    # Never overwrite evidence from a preceding attempt.
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="STARTED", mode=a.mode, requested_steps=a.steps,
                  group_size=a.group_size if a.mode=="q4" else None,
                  quantize_lm_head=False, seed=1234, rows=[], steps=[],
                  protocol="fixed-text-chunks-v2",
                  method="Same original text tokens in BF16 and Q4; all original lm_head logits; greedy argmax and tie-inclusive top-3; no padding or generated/EOS repetition",
                  checkpoint_hash_scope="Historical trusted layer SHA manifest plus current metadata/stat identity; no fresh full weight rehash")
    dest = a.output/"result.json"
    save_json(dest, result)
    engine = state = logits = None
    handles = []
    started = time.perf_counter()
    exit_code = 1
    try:
        result["available_ram_before"] = psutil.virtual_memory().available
        if not torch.cuda.is_available():
            raise ResourceUnavailable("CUDA required")
        if result["available_ram_before"] < 2.8*GIB:
            raise ResourceUnavailable("Startup requires at least 2.8 GiB available RAM")
        ident = identity(a.model, a.prompts, a.manifest, a.airllm)
        result["identity"] = ident
        result["runner_sha256"] = sha(__file__)
        reference = None
        if a.mode != "reference":
            reference = json.loads((a.reference/"result.json").read_text(encoding="utf-8"))
            if reference["status"] != "COMPLETED" or reference["mode"] != "reference":
                raise ValueError("Reference must be a complete BF16 run")
            if reference.get("protocol") != result["protocol"]:
                raise ValueError("Reference protocol differs")
            if ident != reference["identity"]:
                raise ValueError("Reference identity differs from current runtime/checkpoint/corpus")
            if reference["runner_sha256"] != result["runner_sha256"]:
                raise ValueError("Reference runner source differs")
            a.steps, a.batch, a.chunk_tokens = reference["requested_steps"], reference["batch"], reference["chunk_tokens"]
            result["requested_steps"] = a.steps
            result["reference_sha256"] = sha(a.reference/"result.json")
        config = json.loads((a.model/"config.json").read_text(encoding="utf-8"))
        plan = make_plan(a.model, "q4" if a.mode=="q4" else "bf16", a.group_size,
                         1, a.ram_gib, a.vram_gib, 256, "off", a.chunk_tokens*a.steps)
        # Add explicit batched recurrent/KV/logit reserves to the B=1 planner.
        linear_layers = config["layer_types"].count("linear_attention")
        full_layers = config["layer_types"].count("full_attention")
        recurrent = linear_layers * config["linear_num_value_heads"] * config["linear_key_head_dim"] * config["linear_value_head_dim"] * 4
        kv = full_layers * 2 * config["num_key_value_heads"] * config["head_dim"] * a.chunk_tokens*a.steps * 2
        prefill_logits = a.chunk_tokens * config["vocab_size"] * 4
        batch_reserve = a.batch*(recurrent+kv+prefill_logits)
        plan.gpu_state_scratch_reserve_bytes += batch_reserve
        plan.vram_estimate_bytes += batch_reserve
        result["plan"] = plan.as_dict()
        try:
            plan.validate(available_vram=torch.cuda.mem_get_info()[0])
        except MemoryError as exc:
            raise ResourceUnavailable(str(exc)) from exc
        result["batch"], result["chunk_tokens"] = a.batch, a.chunk_tokens
        torch.set_num_threads(4)
        torch.manual_seed(1234)
        torch.cuda.set_per_process_memory_fraction(a.vram_gib*GIB/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        engine = AirColibriModel(str(a.model), pipeline_mode="sequential",
            vram_resident_layers=0, max_ram_layers=2, max_ram_bytes=256*2**20,
            cache_policy="lru", control_mode="enforce", airllm_source=a.airllm,
            weight_format="q4" if a.mode=="q4" else "bf16", q4_group_size=a.group_size,
            q4_cache_dir=str(a.q4_cache) if a.mode=="q4" else None,
            checkpoint_manifest=str(a.manifest) if a.mode=="q4" else None)
        result["initialization_seconds"] = time.perf_counter()-started
        corpus = json.loads(a.prompts.read_text(encoding="utf-8"))
        prompts = corpus["prompts"][:a.batch]
        if len(prompts) != a.batch:
            raise ValueError("Corpus smaller than requested batch")
        total_tokens = a.steps*a.chunk_tokens+1
        encoded = [engine.tokenizer.encode(row["text"], add_special_tokens=False)[:total_tokens] for row in prompts]
        if any(len(row) != total_tokens for row in encoded):
            raise ValueError("A prefix is shorter than requested token window; no padding is allowed")
        result["input_ids"] = encoded
        result["prompt_ids"] = [row["id"] for row in prompts]
        if reference and encoded != reference["input_ids"]:
            raise ValueError("Tokenizer output differs from reference")
        process = psutil.Process()
        def guard(_module, _args):
            if psutil.virtual_memory().available < 512*2**20:
                raise MemoryError("RAM pressure at layer boundary")
            if process.memory_info().rss > plan.ram_budget_bytes:
                raise MemoryError("Process RSS exceeds RAM budget")
        for layer in engine.model.model.layers:
            handles.append(layer.register_forward_pre_hook(guard, prepend=True))
        # A local reference to the last layer would retain its arena views even
        # after engine.close(), making the cleanup measurement misleading.
        del layer
        all_ids = torch.tensor(encoded)
        eos = config.get("eos_token_id")
        eos_ids = eos if isinstance(eos, list) else [eos] if eos is not None else []
        if any(token in eos_ids for row in encoded for token in row):
            raise ValueError("Fixed corpus must not contain EOS tokens")
        captured = []
        def capture_head(_module, _args, output):
            captured.append(output.detach().cpu())
        handles.append(engine.model.lm_head.register_forward_hook(capture_head))
        bitwise = True
        max_abs = 0.0
        for step in range(a.steps):
            t = time.perf_counter()
            offset = step*a.chunk_tokens
            ids = all_ids[:, offset:offset+a.chunk_tokens]
            targets = all_ids[:, offset+1:offset+a.chunk_tokens+1].flatten()
            with torch.no_grad():
                logits, state = engine._forward_step(ids, state, prefetch_next=False)
            torch.cuda.synchronize()
            if len(captured) != 1 or captured[0].shape[:2] != (a.batch, a.chunk_tokens):
                raise AssertionError("Expected exactly one full lm_head output per chunk")
            full_logits = captured.pop()
            if not torch.equal(full_logits[:, -1], logits.detach().cpu()):
                raise AssertionError("Observed lm_head last position differs from runtime output")
            cpu = full_logits.reshape(-1, full_logits.shape[-1]).contiguous()
            if not torch.isfinite(cpu).all():
                raise ValueError("Nonfinite model logits")
            if reference is None:
                path = a.output/f"step-{step:03d}.pt"
                torch.save(dict(logits=cpu, targets=targets), path)
                ref_logits = cpu
                record = dict(file=path.name, sha256=sha(path), active_positions=cpu.shape[0])
            else:
                expected = reference["steps"][step]
                path = a.reference/expected["file"]
                if sha(path) != expected["sha256"]:
                    raise ValueError("Reference tensor artifact changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if not torch.equal(targets, saved["targets"]):
                    raise ValueError("Reference target tokens differ")
                ref_logits = saved["logits"]
                bitwise = bitwise and torch.equal(saved["logits"], cpu)
                max_abs = max(max_abs, (saved["logits"].float()-cpu.float()).abs().max().item())
                record = dict(active_positions=cpu.shape[0])
            rows = compare_logits(ref_logits, cpu)
            ref_nll = torch.nn.functional.cross_entropy(ref_logits.float(), targets, reduction="none")
            candidate_nll = torch.nn.functional.cross_entropy(cpu.float(), targets, reduction="none")
            for idx, row in enumerate(rows):
                row.update(prompt_id=prompts[idx//a.chunk_tokens]["id"], step=step,
                           position=offset+idx%a.chunk_tokens, target_id=int(targets[idx]),
                           bf16_nll=float(ref_nll[idx]), candidate_nll=float(candidate_nll[idx]),
                           bf16_token=engine.tokenizer.decode([row["bf16_id"]]),
                           candidate_token=engine.tokenizer.decode([row["candidate_id"]]))
            result["rows"].extend(rows)
            record.update(step=step, seconds=time.perf_counter()-t,
                          available_ram=psutil.virtual_memory().available, rss=process.memory_info().rss,
                          cuda_allocated=torch.cuda.memory_allocated())
            result["steps"].append(record)
            result["metrics"] = summarize(result["rows"])
            result["metrics"]["bf16_mean_nll"] = sum(r["bf16_nll"] for r in result["rows"])/len(result["rows"])
            result["metrics"]["candidate_mean_nll"] = sum(r["candidate_nll"] for r in result["rows"])/len(result["rows"])
            print(f"{a.mode} {step+1}/{a.steps}: {record['seconds']:.2f}s; positions={result['metrics']['positions']}; confident={result['metrics']['confident_count']}; flips={result['metrics']['confident_flips']}; RAM={record['available_ram']/GIB:.2f} GiB", flush=True)
            save_json(dest, result)
            del cpu, full_logits, ref_logits
            if reference:
                del saved
        result.update(status="COMPLETED", all_logits_bitwise_equal=bitwise if reference else None,
            max_abs_logit_difference=max_abs if reference else None,
            loads=engine.pipeline.total_loads, computes=engine.pipeline.total_computes,
            quantization_calls=engine.weight_manager.quantization_calls,
            prepacked_cache=engine.weight_manager.prepacked_cache.stats if engine.weight_manager.prepacked_cache else None,
            lease_event_counts=dict(engine.ack_adapter.event_counts),
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            cache_peak_bytes=engine.weight_manager.ram_cache.peak_bytes,
            airllm_reader_calls=engine.weight_reader.calls)
        if result["loads"] != a.steps*config["num_hidden_layers"] or result["computes"] != result["loads"]:
            raise AssertionError("Layer transfer/compute count differs from expected")
        if a.mode=="bf16-check" and not bitwise:
            raise AssertionError("BF16 repeat is not bitwise equal")
        exit_code = 0 if a.mode != "q4" or result["metrics"]["gate"]=="PASS" else 2
    except ResourceUnavailable as exc:
        result.update(status="SAFE_SKIP", error=str(exc))
        exit_code = 3
    except KeyboardInterrupt:
        result.update(status="CANCELLED", error="Interrupted")
        exit_code = 130
    except Exception as exc:
        result.update(status="ERROR", error=str(exc), traceback=traceback.format_exc())
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        state = logits = None
        if engine:
            engine.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result["elapsed_seconds"] = time.perf_counter()-started
        result["cuda_allocated_after_close"] = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        save_json(dest, result)
    print(json.dumps({k:result.get(k) for k in ("status","mode","metrics","error")}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
