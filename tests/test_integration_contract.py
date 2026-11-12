"""Regression tests for real ownership, admission and cache boundaries."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.bounded_memory_manager import BoundedPinnedBufferPool, BoundedRAMCache, LazyGlobalsMap
from src.control_contract import TraceContract, ContractViolation
from src.gpu_ring_pipeline import GPURingPipeline
from src.lease_ack_bridge import ExistingRingAcknowledgementAdapter, BackendLeaseViolation
from src.prepacked_q4_cache import PrepackedLayerCache, CheckpointChangedError
from src.upstream_bridge import NativeTierPolicy, AirLLMReader

ROOT = Path(__file__).resolve().parents[1]


def test_zero_and_byte_cache_budgets():
    c = BoundedRAMCache(0, "lru")
    c.put(0, {"w": torch.ones(4)})
    assert c.current_bytes == 0
    c = BoundedRAMCache(3, "lru", max_bytes=24)
    c.put(0, {"w": torch.ones(4)})
    c.put(1, {"w": torch.ones(4)})
    assert list(c.cache) == [1] and c.current_bytes == 16
    c.put(2, {"w": torch.ones(10)})
    assert list(c.cache) == [1]


def test_pinned_rejects_overlay_before_any_write():
    template = {"linear_attn.w": ((8,), torch.float32), "self_attn.w": ((8,), torch.float32)}
    with BoundedPinnedBufferPool(template, num_slots=1, dtype=torch.float32) as pool:
        pool.copy_into_slot(0, {"linear_attn.w": torch.ones(8)})
        before = pool.arena.clone()
        with pytest.raises(ValueError):
            pool.copy_into_slot(0, {"linear_attn.w": torch.zeros(8), "self_attn.w": torch.zeros(8)})
        assert torch.equal(before, pool.arena)
        with pytest.raises(ValueError):
            pool.copy_into_slot(0, {"linear_attn.w": torch.zeros(1)})
        assert torch.equal(before, pool.arena)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lease_rejects_active_reuse_and_wrong_layer(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with GPURingPipeline({"w": torch.zeros(8)}, num_slots=1, device=torch.device(device), dtype=torch.float32) as pipe:
        with ExistingRingAcknowledgementAdapter(pipe) as ack:
            slot = pipe.get_slot_for_loading(0)
            with pytest.raises(BackendLeaseViolation):
                pipe.acquire_slot_for_compute(0)
            pipe.load_tensors_async(slot, {"w": torch.ones(8)})
            with pytest.raises(BackendLeaseViolation):
                pipe.acquire_slot_for_compute(1)
            pipe.acquire_slot_for_compute(0)
            with pytest.raises(BackendLeaseViolation):
                pipe.get_slot_for_loading(1)
            pipe.release_slot_after_compute(slot)
            ack.wait_for_free(0)
            assert ack.snapshot()["leases"][0]["phase"] == "FREE"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_reuse_fence_on_nondefault_stream():
    device = torch.device("cuda")
    with GPURingPipeline({"w": torch.zeros(64)}, num_slots=1, device=device, dtype=torch.float32) as pipe:
        with ExistingRingAcknowledgementAdapter(pipe) as ack:
            slot = pipe.get_slot_for_loading(0)
            pipe.load_tensors_async(slot, {"w": torch.ones(64)})
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                pipe.acquire_slot_for_compute(0)
                torch.cuda._sleep(20_000_000)
                first = slot.tensors["w"].sum()
                pipe.release_slot_after_compute(slot)
            assert ack.leases[0].phase == "RELEASE_QUEUED"
            slot = pipe.get_slot_for_loading(1)
            pipe.load_tensors_async(slot, {"w": torch.full((64,), 2.)})
            pipe.acquire_slot_for_compute(1)
            second = slot.tensors["w"].sum()
            pipe.release_slot_after_compute(slot)
            torch.cuda.synchronize()
            assert first.item() == 64 and second.item() == 128


def test_bounded_trace_keeps_lifetime_counts_and_grammar():
    trace = TraceContract(1, max_events=2)
    for _ in range(10):
        trace.start_step(); trace.enter_layer(0); trace.leave_layer(0); trace.finish_step()
    assert len(trace.events) == 2 and trace.snapshot()["event_counts"]["step_end"] == 10
    with pytest.raises(ContractViolation):
        trace.enter_layer(0)


def test_q4_cache_integrity_identity_and_no_rebuild(tmp_path):
    source = tmp_path / "model-00000.safetensors"
    save_file({"model.layers.0.norm.weight": torch.ones(64)}, str(source))
    st = source.stat()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(format="aircolibri-checkpoint-sha256-v1", layers={"0":dict(
        filename=source.name, size_bytes=st.st_size, mtime_ns=st.st_mtime_ns,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tensors={"model.layers.0.norm.weight":dict(shape=[64],dtype="F32")})})))
    c = PrepackedLayerCache(tmp_path, tmp_path / "cache", manifest, dtype=torch.float32, group_size=16)
    builds = []
    def build():
        builds.append(1)
        return {"norm.weight":torch.ones(64)}
    c.get_or_build(0, build); c.get_or_build(0, build)
    assert len(builds) == 1 and c.stats["hits"] == 1
    path, _, _ = c.cache_coordinates(0)
    with path.open("r+b") as f:
        f.seek(-1, 2); f.write(b"\xff")
    c.get_or_build(0, build)
    assert len(builds) == 2 and c.stats["corruption"] >= 1
    source.write_bytes(source.read_bytes() + b"x")
    with pytest.raises(CheckpointChangedError):
        c.get_or_build(0, build)


def test_native_policy_retains_hot_weights_and_admits_after_margin():
    dll = ROOT / "data/native/colibri_tier.dll"
    if not dll.exists():
        pytest.skip("Build native tier bridge first")
    policy = NativeTierPolicy(dll, 4, 1)
    cache = BoundedRAMCache(1, "lru", max_bytes=32)
    cache.on_access, cache.admission = policy.record_access, policy.admit
    cache.on_placement = policy.acknowledge
    cache.get(0); cache.put(0, {"w":torch.ones(4)})
    for _ in range(8):
        cache.get(0)
    cache.get(1); cache.put(1, {"w":torch.ones(4)})
    assert list(cache.cache) == [0] and cache.rejected_admissions == 1
    for _ in range(20):
        cache.get(1)
    cache.put(1, {"w":torch.ones(4)})
    assert list(cache.cache) == [1] and cache.evictions == 1
    assert policy.lib.ac_tier_promote(0xffffffff, 0xffffffff) == 0
    assert policy.stats["backend_admissions"] == 2
    assert policy.stats["backend_rejections"] == 1


def test_rejected_replacement_is_atomic():
    cache = BoundedRAMCache(2, "static_pinned", max_bytes=32)
    cache.put(0, {"w":torch.ones(4)})
    cache.put(1, {"w":torch.ones(4)})
    old = cache.cache[0]
    cache.put(0, {"w":torch.ones(8)})
    assert cache.cache[0] is old and cache.current_bytes == 32
    assert cache.placement_events[-1]["status"] == "rejected"


def test_lazy_globals_can_unmap_and_reopen(tmp_path):
    path = tmp_path / "globals.safetensors"
    save_file({"emb":torch.arange(64).reshape(8, 8).float()}, str(path))
    g = LazyGlobalsMap(str(path), torch.float32)
    expected = g.lookup_rows("emb", torch.tensor([[2]]))
    g.release_handle()
    assert g._handle is None
    assert torch.equal(expected, g.lookup_rows("emb", torch.tensor([[2]])))
    g.close()
