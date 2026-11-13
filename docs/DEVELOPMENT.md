# 🚧 Developer setup and validation

This guide describes a source-based experimental setup. It is not an installer
or a claim that a fresh machine has been certified. Windows x64 / RTX 3060 Ti
is the tested full-model environment; Linux CI exercises synthetic CPU contracts.

## Environment

Use Python 3.12 and a compatible CUDA driver for the tested Torch build.
Installed package versions are recorded in `requirements-runtime.lock.txt`;
this is a direct-dependency snapshot, not a transitive environment lock.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.lock.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

Keep upstream checkouts at paths of your choice. Reuse existing checkouts
instead of duplicating them. Tested source revisions:

| Upstream | Revision |
|---|---|
| [AirLLM](https://github.com/lyogavin/airllm) | `e496ed3c5c8f614df943b9401cf17e156ef7aca0` |
| [Colibrì](https://github.com/JustVugg/colibri) | `0a5504a1dc64f86fcd0fe7db298b0448c730e7e6` |

`--airllm-source` points to the repository root containing
`air_llm/airllm/utils.py`, not an arbitrary pip version. It overrides
`AIRCOLIBRI_AIRLLM_SOURCE`, then the optional ignored `aircolibri.local.json`
(`{"airllm_source":"D:/path/to/airllm"}`), then `./upstream_airllm`.
The adapter verifies the actual imported reader's location.

## Checkpoint layouts

**OLMoE:** the original local OLMoE-1B-7B-0125-Instruct BF16 checkpoint with
config, tokenizer, `model.safetensors.index.json` and its referenced shards.
There is no automatic download and no private weight artifact in this repo.

**Qwen:** the tested Qwen3.5-9B text-only `qwen3_5_text` layout with
`model-00000.safetensors` … `model-00031.safetensors`,
`model-globals.safetensors`, config and tokenizer. Layer keys retain the
`model.layers.N.` prefix. Globals contain embedding, final norm and lm_head.
This is a prepared split checkpoint, not arbitrary Hugging Face shards;
automatic conversion of arbitrary Qwen layouts is not provided here.

Choose `--model-dir` explicitly. The lab-compatible default Qwen directory
has an old DeepSeek name; the runtime checks the architecture in config.

## Native admission policy

Build the small bridge with installed MSVC Build Tools and `vswhere`:

```powershell
.\.venv\Scripts\python.exe scripts/build_native_tier.py --colibri-source "D:\path\to\colibri"
```

This compiles a thin shim against the original `c/tier.h`, not a replacement
implementation or the full CUDA engine. The output DLL and SHA/ABI provenance
manifest live in `data/native`. Keep both together. No binary is distributed.
`--policy off` does not require the bridge when GPU retention is also disabled;
RAM `shadow`/`apply` and any nonzero `--gpu-expert-slots` require it.

## Run

```powershell
.\.venv\Scripts\python.exe aircolibri.py doctor --airllm-source "D:\path\to\airllm"
.\.venv\Scripts\python.exe aircolibri.py plan --model olmoe --model-dir "D:\models\OLMoE-1B-7B-0125-Instruct"
.\.venv\Scripts\python.exe aircolibri.py run --model olmoe --model-dir "D:\models\OLMoE-1B-7B-0125-Instruct" --airllm-source "D:\path\to\airllm" --policy apply --max-new-tokens 8 --report data/my-run.json
```

Policy modes: `off` baseline, `shadow` decisions without applying native
admission, `apply` actual native admission. Dense additionally accepts
`--control off/observe/enforce` and sequential/double/triple ring pipelines.
MoE currently requires sequential/enforce. Raw prompts are used; no chat
template is added automatically.

OLMoE defaults to `--moe-placement streamed --moe-slot-layout compact`: two
role-specific slots from one existing GPU Ring hold the current Dense trunk
and current expert, with the same pinned pool and CUDA lease ACKs. On the tested
checkpoint these slots reserve 34 and 12 MiB. Layer parameters return to meta
after their consumers. `--moe-slot-layout union` selects the historical layout
where every slot reserves the full 46 MiB union of both resource templates.
Embedding, final norm and output head stay resident. `--moe-placement resident`
selects the previous Dense-resident/expert-copy path for comparisons. The
low-level `AirColibriOLMoE` API retains its previous default; pass
`stream_dense=True` explicitly outside the CLI.

Optional `--gpu-expert-slots 1|2` adds dedicated expert slots to this same Ring.
The default is `0`. It requires streamed OLMoE and the native tier library, even
when RAM `--policy` is off. GPU admission uses a separate policy instance.
`--gpu-cache-mode shadow` evolves a virtual resident set while streaming every
expert; `apply` reuses retained tensors and skips their host supplier and DMA.
Both modes reserve the extra GPU and pinned slots for a matched memory layout.
Each compact expert slot reserves 12 MiB; two slots add **24 MiB GPU plus
24 MiB pinned RAM**. The complete four-slot arena is 70 MiB on each side,
compared with 184 MiB in union mode. Startup plans use the same layout builder
as runtime allocations. This is not an automatic cache-size or global policy.

```powershell
.\.venv\Scripts\python.exe aircolibri.py run --model olmoe --policy apply --cache-mib 64 --ram-gib 4 --vram-gib 1.2 --gpu-expert-slots 2 --gpu-cache-mode apply --max-new-tokens 2 --report data/gpu-cache-run.json
```

Supply your model/source paths as above. `tests/test_gpu_expert_residency.py`
covers skipped host reads/DMA, shadow behavior, stale/active leases, malformed
admission proposals, completion ACKs and cross-stream eviction dependencies.

Windows OLMoE defaults to safetensors `pread` after native crashes were localized
to mmap storage construction. `--moe-read-backend auto|pread|mmap` controls this;
auto uses mmap on other platforms. The original AirLLM function is bound to a
private backend configuration, without changing upstream source or globals.

The startup plan estimates cache, pinned/transient weights and runtime/state/scratch
space; it is not a live reservation ledger. Startup requires at least 2.8 GiB
available RAM, plus the plan's requirements and Windows commit headroom.
Torch allocator VRAM fraction is
process-local; RAM is checked at layer boundaries, not constrained by an OS
hard quota. Results distinguish estimates from measured allocations.

## Q4 remains experimental

Q4 requires a source-specific trusted SHA manifest and a compatible prepared
cache. The local trusted manifest and generated cache are not distributed.
Cache identity includes source SHA, tensor layout, group size, compute dtype
and quantizer source SHA. Cache misses can quantize weights, so first-use
work differs from a verified prepacked-cache run. Do not treat the published
cache-hit timings as fresh-conversion timings.

The cache implementation supports a disk quota, but the main manager and CLI
currently use its unlimited default. Cache disk limits are not enforced by
`--ram-gib` or `--cache-mib`; those refer to memory budgets.

Use BF16 unless specifically investigating Q4. Off/on Q4 parity does not
satisfy the strict Q4-versus-BF16 quality gate.

The [fixed-text quality protocol](QUALITY_EVALUATION.md) describes bounded
teacher forcing, reusable reference artifacts and the unchanged acceptance gate.

## Tests and research probes

Synthetic contracts (small tensors, no model download):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_integration_contract.py tests/test_olmoe_integration.py -q
```

CUDA/native cases skip when prerequisites are absent. The full suite includes
local real-checkpoint tests and old machine-specific fixtures; it is not
expected to work on a bare checkout. Heavy test files should run in separate
processes when retained allocator memory reduces later residency.
The published full-suite failure and isolated rerun are described in the report.

`scripts/verify_integration.py` accepts explicit Dense model/source paths.
`verify_moe_integration.py`, matrix/lifecycle/release scripts and earlier lab
scripts use the documented local fixture layout and may need path adaptation.
They are research probes, not general model-management tools.
Raw logits and local outputs remain under ignored `data` paths;
selected portable summaries are published under `docs/evidence`.

Shared-resource tests are in `tests/test_shared_resources.py`. The independent
reference in `scripts/reference_olmoe.py` reads safetensors directly and stages
all experts of one layer in original HF modules; it imports no AirColibri
reader, cache or Ring. It is a verification harness, not a production loader.
Run `scripts/verify_shared_moe.py` in separate processes with `--mode reference`,
`resident` and `streamed`, unique `--case` names, explicit model/source paths and
an adequate `--vram-gib`. `--native` applies to candidate modes. Compare completed
artifacts using `scripts/compare_shared_moe.py`; a partial run cannot pass.
Use Python `-X faulthandler` and preserve logs when investigating native faults.
For GPU cache comparisons, use streamed candidates with zero slots, then
`--gpu-expert-slots 2 --gpu-cache-mode shadow` and `apply`, keeping all other
inputs and budgets identical. With hits, `loads + resident_acquires == computes`;
the old equality between loads and computes no longer applies.
`--workload extended` uses one 128-token prefill and 16 fixed teacher-forced
decode steps (144 tokens final context), with a separate independent reference
run on the same inputs. It is a longer-context probe, not a soak or generation
quality test. Reports include per-step state hashes, allocations and cache
counters. Comparisons reject bytewise logit differences and known source
identity mismatches; missing historical source metadata remains explicit.

The Ring and pinned pool still each use one allocation. Their `slot_layouts`,
`slot_offsets` and `slot_strides` describe actual storage. `slot_stride` is null
for nonuniform slot sizes; do not infer arena bytes by multiplying one stride.
The original uniform constructor path and Qwen attention overlays are preserved.
