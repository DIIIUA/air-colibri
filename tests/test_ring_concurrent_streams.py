"""Independent consumers must keep their own completion fences."""
import pytest
import torch

from src.bounded_memory_manager import BoundedPinnedBufferPool
from src.gpu_ring_pipeline import GPURingPipeline
from src.lease_ack_bridge import ExistingRingAcknowledgementAdapter


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_first_dma_waits_for_arena_initialization_on_another_stream():
    constructor = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    initialized = torch.cuda.Event()
    host = {"w": torch.ones(4096, pin_memory=True)}
    with torch.cuda.stream(constructor):
        torch.cuda._sleep(150_000_000)
        pipe = GPURingPipeline({"w": torch.zeros(4096)}, num_slots=1,
                              device=torch.device("cuda"), dtype=torch.float32)
        initialized.record(constructor)
    with pipe:
        slot = pipe.get_slot_for_loading(0)
        pipe.load_tensors_async(slot, host)
        with torch.cuda.stream(consumer):
            consumer.wait_event(initialized)
            pipe.acquire_slot_for_compute(0)
            actual = slot.tensors["w"].sum()
            pipe.release_slot_after_compute(slot)
        torch.cuda.synchronize()
        assert actual.item() == 4096


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_two_consumers_release_against_their_own_streams(monkeypatch):
    count = 4096
    host = [torch.full((count,), float(value), pin_memory=True) for value in (1, 2, 3)]
    consumers = [torch.cuda.Stream() for _ in range(3)]
    with GPURingPipeline({"w": torch.zeros(count)}, num_slots=2,
                         device=torch.device("cuda"), dtype=torch.float32) as pipe:
        with ExistingRingAcknowledgementAdapter(pipe):
            first_slot = pipe.get_slot_for_loading(0)
            first_event = first_slot.compute_done_event
            first_release_streams = []
            original_record = torch.cuda.Event.record
            def record_event(event, stream=None):
                if event is first_event:
                    first_release_streams.append((stream or torch.cuda.current_stream()).cuda_stream)
                return original_record(event, stream)
            monkeypatch.setattr(torch.cuda.Event, "record", record_event)
            pipe.load_tensors_async(first_slot, {"w": host[0]})
            other_slot = pipe.get_slot_for_loading(1)
            pipe.load_tensors_async(other_slot, {"w": host[1]})
            with torch.cuda.stream(consumers[0]):
                pipe.acquire_slot_for_compute(0)
                torch.cuda._sleep(150_000_000)
                first = first_slot.tensors["w"].sum()
            with torch.cuda.stream(consumers[1]):
                pipe.acquire_slot_for_compute(1)
                other = other_slot.tensors["w"].sum()
                pipe.release_slot_after_compute(other_slot)
            # Release the first consumer only after another stream was acquired.
            with torch.cuda.stream(consumers[0]):
                pipe.release_slot_after_compute(first_slot)
            replacement = pipe.get_slot_for_loading(2)
            pipe.load_tensors_async(replacement, {"w": host[2]})
            with torch.cuda.stream(consumers[2]):
                pipe.acquire_slot_for_compute(2)
                third = replacement.tensors["w"].sum()
                pipe.release_slot_after_compute(replacement)
            torch.cuda.synchronize()
            assert first.item() == count, "First consumer read weights replaced before its completion"
            assert other.item() == 2*count
            assert third.item() == 3*count
            # Scheduling can accidentally serialize the data race on WDDM.
            # Inspect the real CUDA event's owner as well as output values.
            assert first_release_streams[0] == consumers[0].cuda_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_1024_reuses_keep_values_and_bounded_event_history(slots):
    count = 4096
    template = {"w": torch.zeros(count)}
    streams = [torch.cuda.Stream() for _ in range(3)]
    pending, expected = [], []
    with BoundedPinnedBufferPool(template, num_slots=slots, dtype=torch.float32) as pool:
        with GPURingPipeline(template, num_slots=slots, device=torch.device("cuda"),
                             dtype=torch.float32) as pipe:
            with ExistingRingAcknowledgementAdapter(pipe) as ack:
                for cycle in range(1024):
                    value = float(cycle % 17 + 1)
                    sid = cycle % slots
                    host = pool.copy_into_slot(sid, {"w": torch.full((count,), value)})
                    slot = pipe.get_slot_for_loading(cycle)
                    pipe.load_tensors_async(slot, host, pinned_pool=pool)
                    # Host reuse must wait for its own DMA, independently of compute.
                    pool.copy_into_slot(sid, {"w": torch.full((count,), -999.)})
                    with torch.cuda.stream(streams[cycle % len(streams)]):
                        pipe.acquire_slot_for_compute(cycle)
                        if cycle % 7 == 0:
                            torch.cuda._sleep(250_000)
                        pending.append(slot.tensors["w"].sum())
                        expected.append(value * count)
                        pipe.release_slot_after_compute(slot)
                    if len(pending) == 128:
                        torch.cuda.synchronize()
                        assert [v.item() for v in pending] == expected
                        pending.clear()
                        expected.clear()
                for sid in range(slots):
                    ack.wait_for_free(sid)
                assert pipe.total_loads == pipe.total_computes == 1024
                assert ack.event_counts["slot_reserved"] == 1024
                assert sum(ack.event_counts.values()) > 4096
                assert len(ack.events) <= 4096 and ack.violations == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_close_drains_an_inflight_tail_prefetch():
    host = {"w": torch.ones(4096, pin_memory=True)}
    pipe = GPURingPipeline({"w": torch.zeros(4096)}, num_slots=2,
                          device=torch.device("cuda"), dtype=torch.float32,
                          tail_to_head_prefetch=True)
    with ExistingRingAcknowledgementAdapter(pipe) as ack:
        with torch.cuda.stream(pipe.transfer_stream):
            torch.cuda._sleep(150_000_000)
        slot = pipe.prefetch_tail_to_head(0, host)
        event = slot.transfer_done_event
        assert not event.query(), "Test did not observe an in-flight prefetch"
        assert ack.leases[0].phase == "DMA_QUEUED"
        pipe.close()
        assert event.query(), "Close returned before the prefetch finished"
        assert pipe.arena is None and pipe.slots == []
        pipe.close()
