# Resident / streaming decode, RTX 3060 Ti 8 GB

2026-11-18 development snapshot, after the CUDA/NVML startup-accounting correction.
Raw numeric data: [streaming_decode_results_20261118.json](streaming_decode_results_20261118.json).
This is a new Llama 3.2 1B comparison; the historical Qwen3.5-9B ~31 tok/s figure
is a separate experiment and was not remeasured.

## Protocol

- Windows x64, RTX 3060 Ti 8192 MiB, driver 617.42, unchanged 200 W limit;
  Ryzen 5 5600X, six PyTorch CPU threads, Torch 2.6.0+cu124, Transformers 5.22.2.
- Source checkpoint: `01_gguf_Llama-3.2-1B-Instruct-Q4_K_M`; runtime Q6 in all cases.
  MTP, vocabulary pruning and thought anchor off. Greedy decoding, EOS stopping off.
- Same sky-explanation prompt, fresh session each time, context reserve 2048,
  prefill chunk 64. Eight unscored warmup tokens before each block.
- Order **A B C C B A**: all 16 decoder layers resident, 8 resident + 8 streamed,
  all 16 through a three-slot GPU ring. Two measured 64-token requests per block:
  four samples per layout and 24 measured requests across both pressure settings.
  Decode rates exclude prefill and its first generated token: 63 decode tokens/request.
- Repeat the complete order with a separate idle process holding exactly 2 GiB
  of allocated CUDA memory before planning. It does no concurrent inference.
- Streamed packed layers can occupy all 16 host-cache entries within **896 MiB**.
  The measured full-streaming cache peaked at **754.125 MiB**. This isolates warm
  RAM-to-GPU layer streaming. It is a controlled placement, not automatic selection
  of these splits by the planner; the small checkpoint still fits under both pressures.
- OS caches were not flushed; normal desktop applications remained active.
  No inference/logit observers or per-layer KV copies ran inside measured requests.

## Results

Median decode tok/s, with min–max of four samples:

| Persistent resident decoder layers | No added pressure | Separate 2 GiB allocation |
|---|---:|---:|
| 16/16 | **30.55** (26.33–33.64) | **31.64** (25.86–34.81) |
| 8/16, others streamed | **11.98** (11.65–12.40) | **11.92** (11.73–12.05) |
| 0/16, all through ring | **7.34** (7.25–7.62) | **7.34** (6.72–7.45) |

At no added pressure, half streaming retained 39% of resident decode speed;
all streaming retained 24%. This is approximately **61% / 76% lower**, respectively.
The pressure and no-pressure resident ranges overlap; this is not evidence that
occupying more VRAM speeds up inference. A universal fixed 50% penalty is not supported.

All 24 output text hashes were identical. Every request generated its full 64 tokens.
There were **zero host layer-cache misses** during measured requests. Logical
process reads were 166,320 bytes/request in every placement; the embedding path
reads selected rows on demand. These counters do not measure physical HDD traffic.
Cold disk rereads/packing and models larger than the available host cache require
separate measurements. Weight file size/mtime remained unchanged.

## Startup memory accounting finding

The first paired run exposed a real discrepancy on this Windows/driver combination:
CUDA's free counter stayed around 6.90 GiB while a separate process held 2 GiB;
`nvidia-smi` reflected the additional occupied memory. That first run is retained
locally as diagnostic evidence and is not the table above.

`aircolibri_cli/gpu_memory.py` now queries the matching GPU by CUDA UUID through
NVML and uses **min(CUDA free, NVML free)**. GPU-first startup, progressive GPU
preparation and `doctor` share this helper. Status reports both counters and the
source; optional NVML failure retains the CUDA reading with an explicit diagnostic.
The device match uses UUID because CUDA-visible ordinals can be remapped.

In the corrected run the median effective startup free reading changed from
**6.542 GiB to 4.160 GiB**. The plan received that effective reading in every case.
The live two-process test also checks that another 2 GiB allocation reduces this
process's usable budget by more than 1.5 GiB. Startup accounting is verified here;
continuous global replanning after later surprise allocations is not implemented.

NVIDIA documents its [NVML memory queries and WDDM accounting caveats](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).
Its [CUDA memory-query reference](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html)
also explains that an OS free-memory reading is not an allocation guarantee.
The guard, KV/SSM reserve and workspaces remain necessary.

## Reproduction and evidence

Run `benchmarks/streaming_decode.py --models 01_gguf_Llama-3.2-1B-Instruct-Q4_K_M --output PATH`
from a fresh output directory. Pressure-control files are retained; a new directory
prevents reuse of stale readiness signals. The holder is stopped in `finally`.
The current harness also matches the pressure process by its reported CUDA-worker
PID, accounting for the extra Windows virtualenv launcher process.

Measured engine LF hash: `adf24260d9010f1b035cd1881d3106c90c62b4af840f57fece4ca76f67b48c8c`.
The local `streaming_reply_2/receipt.json` verifies sample counts, equal hashes,
cache misses, budget inputs and the pressure delta. `measured_source.zip` preserves
the exact measured source; later CLI packaging changes are separate.
Per-process `nvidia-smi` used-memory fields were N/A on this WDDM host; global
memory readings and the holder's actual allocated-byte count are recorded.
