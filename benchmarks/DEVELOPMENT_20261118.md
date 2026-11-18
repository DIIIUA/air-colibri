# Development validation — 18 November 2026

This is a historical development record between v1.1.0 and v1.2.0. Its individual
results are not a fresh release-wide rerun or a universal-support claim. See
[final release validation](RELEASE_1_2_0.md) for the later census and regression suite.
Raw reports are in `D:\Air-Colibri Storage\development_20261118`.

## Active scope

The current roster is `model_roster_20261118.json`: 46 checkpoints. DeepSeek and OLMoE
were deliberately removed by the user for disk space and are excluded from this run.
The 25.6 GB uncensored Qwen3.6-35B remains included. No checkpoints are downloaded or modified.
The historical 48-checkpoint matrix and release evidence remain dated records.

## Probability retention: implementation and meaning

The first next-token distribution is conditional on a particular prefix and output position.
It is not a general representation of the task's meaning. Subsequent output distributions
are computed from the evolving token sequence and model state; they are not normally updates
of the previous output-probability vector. Layer activations, attention probabilities, expert
router scores and output-token probabilities have different meanings and coordinates.
Trained weights remain fixed during ordinary inference; changing activations or router scores
does not mean that the weights are being retrained or that the entire context has been discarded.

The current thought buffer retains the full original FP32 log-probability distribution and
a separate contrastive lexical prior. It applies a bounded logit adjustment. It does **not**
establish semantic memory, improved reasoning or preservation of all original conditions.
Keeping every prior output-probability vector would not by itself establish those properties.
KV/recurrent state and the original context serve a different function.

The concrete v1.1.0 failure was observed on Granite: the anchor changed a raw `To` prefix
for Tokyo to `N`, leading to `N/A`. Its old adjustments were 1.375 and 1.625 respectively
on raw logits 18.0 and 17.75. The current implementation:

- bounds pairwise log-odds changes to 0.125 and keeps the adjusted logits in FP32;
- preserves the original distribution while allowing the working field to follow corrected input;
- keeps snapshots/rollback independent of optional direct recurrent-state mixing, now off by default;
- carries the original distribution through draft and synthesis stages;
- treats lexical alignment as a diagnostic, never as verification or a reason to retry with stronger bias;
- aggregates telemetry without several GPU-to-CPU scalar synchronizations per token.

Paired acceptance evidence combines completed cases from `anchor_acceptance_q4.json`,
`anchor_acceptance_q4_granite.json` and `anchor_acceptance_q6.json`. These are separate runs,
with source hashes in each report; the first was interrupted at Granite construction by the
subsequently fixed Windows whole-file mapping problem.

Across 48 paired short-answer cases: expected-answer substring presence was 44/48 in each
mode; exact normalized answers were 39/48 plain and 40/48 anchored. There were no new losses
of a previously correct control answer. Original-prior influence was nonzero in 34 cases.
Six anchored two-turn conversations retained both original-distribution and original-prior
hashes. The corrected marker was returned in five of six configurations in both modes;
Granite Q4 failed in both, while Granite Q6 passed. These checks are deliberately narrow.

## Common memory and positioned I/O

Dynamic memory still uses one fixed-address GPU allocation with elastic A/current-cache,
B/next-preparation and empty C/standby roles. Real CUDA completion events protect reuse.
Pinned expert staging now occupies two expert slots instead of mirroring the whole GPU cache:
1.6875 MiB instead of roughly 596.5 MiB in the controlled Granite fixture.

Allocator changes cache tensor views/layouts, avoid constructing all eviction candidates,
and preserve recently used experts when borrowing contiguous pages for a dense payload.
Router observations share one current-chunk CPU copy with dispatch. The producer is woken
for missing preparation work; routine cache hits and reader-release notifications no longer
need to schedule an idle producer thread. No estimated processing rate or fixed rotation
interval is introduced.

HF globals and layer metadata now use the positioned reader instead of mapping an entire
safetensors shard. This fixes a reproduced Windows error 1455 during Granite construction.
Global matrix staging tiles and physical reads respect the configured read bound.

The first full regression run exposed eight staging-size assertions (8 KiB versus a 4 KiB
test bound); numerical ring/reference results agreed. That failure was retained and fixed,
without relaxing the assertions. Final regression and performance results are recorded below
when their runs finish.

## Method and acceptance boundaries

At the user's request, the mirror-equality and Logic/Strategy co-generation sections of the
local Crossroad reference are applied as a method of checking the work. The check runs in both
directions: intended property -> implemented mechanism -> observable result, and observed
result -> what it actually establishes about the intended property. A candidate explanation
must survive a distinguishing experiment before becoming a validated claim.

This changes the interpretation of the buffer result: eliminating a demonstrated regression
is not evidence that a lexical prior retains global meaning. A useful semantic-memory candidate
would need tests of original conditions, explicit corrections, distractors and long continuations,
with a model-state/context mechanism and a control without that mechanism.

Likewise, warmup predicts which immutable weights to have ready. It must not suppress actual
router-selected experts or change the mathematical result. A wrong prediction costs loading
time. Dynamic memory must be judged by matched outputs, actual overlap, resource bounds and
latency together. Current combined-memory support remains HF Granite/OLMoE Q4/Q6; the roster
census is a separate ordinary-runtime startup/decode/state-continuation test.

## Final regression and timing results

`full_regression_v3.xml`: **180 passed, 10 skipped**, 214.83 seconds. Seven skips require
the removed compatible native Qwen3.5-9B fixture; three require absent native Colibri weights.
These are unavailable fixtures, not successful executions. The original eight failing
staging assertions pass unchanged, along with the new no-whole-shard-map and producer tests.

Controlled Granite 1B MoE Q4 placement: 18 resident and six streamed dense layers, initial
700-expert cache budget, three logical roles, MTP off. Each fresh engine has an untimed
48-token warm-up, two measured 48-token requests and separate short answer probes.
Plain placement order is ring/dynamic/dynamic/ring; anchored placements follow separately.

| Placement | Measured requests | Median decode tok/s | Range |
|---|---:|---:|---:|
| Ring | 4 | 13.96 | 13.46–14.40 |
| Dynamic | 4 | 11.38 | 10.57–11.95 |
| Ring + bounded anchor | 2 | 12.93 | 11.95–13.91 |
| Dynamic + bounded anchor | 2 | 9.24 | 8.90–9.57 |

Files: `after3_*_round*.json`, `after3_ring_anchor.json`, `after3_dynamic_anchor.json`.
This is not a cold-HDD benchmark. Background load, clocks and OS caches are not held constant.
The dynamic policy still costs time on this small fixture: a speed advantage is **not** shown.
Its unanchored median is about 18.5% below the matched ring control. Earlier iterations and
profiles are retained; their differently timed results are not an isolated causal speedup proof.

All three memory roles have recorded reads; C owns zero payload bytes and dense occupancy
never exceeds current plus next. The captured preparation transfers completed before the end
of their recorded generation steps. This timing alone does not prove that every transfer was
needed or ready before its target layer. Both facts are distinct from answer/placement parity.

## Current roster census

At this historical stage the census was paused after entries **01–07 all passed**.
The controller and worker stopped between checkpoints; no checkpoint was interrupted.
The remaining 39 entries had no result for that iteration. The separate final
1.2.0 census is recorded in `MODEL_SUPPORT.md`; those runs are not mixed into this record.

Protocol for all 46 entries: Q6 ordinary runtime, no downloads, prefill chunks of one token,
context budget 512, two generated tokens and one continuation token after the short `Hi`
prompt. Each also receives a raw forward/finite-logit check. This compact protocol is a
startup/state-lifecycle check; its short timing is not a representative generation-speed result.
The controller reuses imports between models and restarts a worker after CUDA cleanup faults
or excessive retained allocations. Failed and time-bounded attempts remain in the report.
