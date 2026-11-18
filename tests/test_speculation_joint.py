"""Speculation boundaries through the same bounded executor and working memory."""
import copy
from dataclasses import replace

import pytest
import torch

from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine, ColibriSpecGate
from test_general_speculation import collect, fixed_proposals


@pytest.mark.parametrize("case", ["budget", "eos", "cancel"])
def test_verification_boundaries(tmp_path, monkeypatch, case):
    from test_cpu_independence import tiny_checkpoint
    from aircolibri_cli.unified_runtime import GenerationCancelled
    tiny_checkpoint(tmp_path)
    monkeypatch.setattr(ColibriSpecGate, "pick_depth", lambda self, source, probs: len(probs))
    prompt = [3, 7, 4, 9]
    with AirColibriUnifiedEngine(tmp_path, device="cpu", speculation="off", verbose=False) as base:
        expected, _ = collect(base, prompt)
    with AirColibriUnifiedEngine(tmp_path, device="cpu", speculation="lookup", verbose=False) as u:
        fixed_proposals(u.speculator, expected, len(prompt), "none")
        if case == "budget":
            u.speculator.state_budget_bytes = 1
            actual, result = collect(u, prompt)
            assert actual == expected
            assert result["spec_gate"]["state_budget_plain_steps"] > 0
            assert result["spec_gate"].get("verified_batches", 0) == 0
        elif case == "eos":
            # Make the first accepted draft a terminal token, including the case
            # of an EOS id recurring elsewhere in the deterministic trajectory.
            u.tokenizer.eos_token_id = expected[1]
            events = list(u._stream_token_ids(prompt, 9, True, 0, None))
            actual = [t for e in events if e["kind"] == "tokens" for t in e["new_ids"]]
            end = expected.index(expected[1])+1
            assert actual == expected[:end]
            assert events[-1]["consumed_tokens"] == len(prompt)+end-1
        else:
            stream = u._stream_token_ids(prompt, 9, False, 0, None,
                                         cancel_check=lambda: requested[0])
            requested = [False]
            next(stream)
            requested[0] = True
            with pytest.raises(GenerationCancelled):
                next(stream)
            assert u.engine._step_cancel_check is None
            assert u.speculator.rowwise.rows == 0
            assert u.engine.route_exclusion.capture is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["granitemoe", "olmoe"])
@pytest.mark.parametrize("bits", [4, 6])
@torch.no_grad()
def test_verified_routes_anchor_dynamic_three_regions_match_plain(tmp_path, monkeypatch, kind, bits):
    from test_probability_warmup import make_model
    from aircolibri_cli import unified_runtime as runtime
    from aircolibri_cli.thought_buffer import InitialProbabilityAnchor
    make_model(tmp_path, kind)
    original = runtime.plan_for_checkpoint
    def ring(*args, **kwargs):
        profile, plan, mtp = original(*args, **kwargs)
        return profile, replace(plan, full_residency=False, resident_layers=1, ring_slots=3), mtp
    monkeypatch.setattr(runtime, "plan_for_checkpoint", ring)
    monkeypatch.setattr(ColibriSpecGate, "pick_depth", lambda self, source, probs: len(probs))
    prompt = [1, 4, 6, 8]
    reference = None
    for mode in ("off", "lookup"):
        with AirColibriUnifiedEngine(tmp_path, progressive_start=False, weight_format=f"q{bits}",
                                      speculation=mode, verbose=False) as u:
            warm = u.engine.enable_probability_warmup(region_experts=4, gpu_budget_bytes=2**20,
                host_cache_bytes=2**20, unified_memory=True, dynamic_regions=True)
            anchor = InitialProbabilityAnchor()
            u._active_thought_anchor = anchor
            if reference is not None:
                fixed_proposals(u.speculator, reference[0], len(prompt), "second")
            actual, state = collect(u, prompt, count=15)
            warm.prefetch.drain()  # correctness synchronization, not timed per-token scheduling
            fingerprint = anchor.initial_probability_fingerprint()
            initial_map = copy.deepcopy(warm.region_map.initial)
            if reference is None:
                reference = (actual, copy.deepcopy(state["past"]), fingerprint, initial_map)
            else:
                assert actual == reference[0]
                assert fingerprint == reference[2]
                assert initial_map == reference[3]
                for a, b in zip(state["past"].layers, reference[1].layers):
                    torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
                    torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)
                assert state["spec_gate"]["verified_batches"] > 0
                assert u._region_controller.stats()["discarded_route_rows"] > 0
            memory = u.engine.model_source._expert_loader.shared_memory
            assert memory.gpu_bytes <= memory.budget_bytes
            assert all(n > 0 for n in memory.area_reads)
            assert memory.stats()["roles"]["C"]["reserved_bytes"] == 0
            assert anchor.telemetry.initial_rows_active > 0
            consumed = warm.prefetch.stats()["bank_consumptions"]
            assert all(n > 0 for n in consumed), warm.prefetch.stats()
        assert not warm.prefetch.thread.is_alive()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_speculative_cpu_work_continues_into_gpu_ring_without_prefix_replay(tmp_path, monkeypatch):
    import threading
    from test_probability_warmup import make_model
    from aircolibri_cli import unified_runtime as runtime
    from aircolibri_cli.thought_buffer import InitialProbabilityAnchor
    make_model(tmp_path, "granitemoe")
    release = threading.Event()
    original = runtime.plan_for_checkpoint
    def delayed_plan(*args, **kwargs):
        assert release.wait(30)
        profile, plan, mtp = original(*args, **kwargs)
        return profile, replace(plan, resident_layers=2, full_residency=False, ring_slots=3), mtp
    monkeypatch.setattr(runtime, "plan_for_checkpoint", delayed_plan)
    monkeypatch.setattr(ColibriSpecGate, "pick_depth", lambda self, source, probs: len(probs))
    u = runtime.AirColibriUnifiedEngine(tmp_path, weight_format="q6", speculation="lookup", verbose=False)
    try:
        pending = u.enable_unified_memory(region_experts=2, host_cache_bytes=2**20)
        anchor = InitialProbabilityAnchor()
        u._active_thought_anchor = anchor
        calls = [0]
        def propose(history, count, native_ready):
            calls[0] += 1
            if calls[0] == 3:
                release.set()
                assert u.engine.progressive.backend_ready.wait(30), u.engine.progressive.stats()
            return [5, 7][:count], [.95]*min(count, 2), ColibriSpecGate.SRC_LOOKUP
        u.speculator.propose = propose
        actual, state = collect(u, [3, 7, 4], count=32)
        assert len(actual) == 32
        progressive = u.engine.progressive
        assert progressive.complete, progressive.stats()
        assert progressive.cpu_steps > 0 and progressive.gpu_layers > 0
        assert state["spec_gate"]["verified_batches"] > 0
        assert state["consumed_tokens"] == 34
        assert pending.active is u.engine.probability_warmup
        assert pending.region_map.stats()["initial_sha256"] == pending.active.region_map.stats()["initial_sha256"]
        assert anchor.telemetry.initial_rows_active > 0
        reference = copy.deepcopy(state["past"])
        a, _ = u.engine._forward_step(torch.tensor([[19]]), state["past"])
        b, _ = u.engine._forward_step(torch.tensor([[19]]), reference)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert all(n > 0 for n in pending.active.stats()["shared_memory"]["area_reads"])
    finally:
        release.set()
        u.close()
