"""Cancellation must release physical buffers and their ownership ledger together."""
import pytest
import torch
from src.gpu_ring_pipeline import GPURingPipeline, SlotState
from src.lease_ack_bridge import ExistingRingAcknowledgementAdapter, BackendLeaseViolation


@pytest.mark.parametrize("phase", ["reserved", "dma", "compute", "released"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cancel_discards_abandoned_work_then_reuses_slot(phase, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = {"w": torch.arange(64, dtype=torch.float32).reshape(8, 8)}
    if device == "cuda":
        source = {k: v.pin_memory() for k, v in source.items()}
    pipe = GPURingPipeline(source, num_slots=3, device=torch.device(device), dtype=torch.float32)
    adapter = ExistingRingAcknowledgementAdapter(pipe).install()
    try:
        slot = pipe.get_slot_for_loading(0)
        if phase != "reserved":
            pipe.load_tensors_async(slot, source)
        if phase in {"compute", "released"}:
            pipe.acquire_slot_for_compute(0)
            slot.active_tensors["w"].mul_(2)
        if phase == "released":
            pipe.release_slot_after_compute(slot)
        else:
            with pytest.raises(BackendLeaseViolation, match="overwrite active"):
                pipe.get_slot_for_loading(3)
        generation = adapter.leases[0].generation
        pipe.reset_in_flight()
        assert all(s.state == SlotState.FREE and not s.loaded_keys for s in pipe.slots)
        assert all(adapter.poll(i)["phase"] == "FREE" for i in range(3))
        assert adapter.leases[0].generation > generation
        slot = pipe.get_slot_for_loading(3)
        pipe.load_tensors_async(slot, source)
        pipe.acquire_slot_for_compute(3)
        assert torch.equal(slot.active_tensors["w"].cpu(), source["w"])
        pipe.release_slot_after_compute(slot)
        pipe.reset_in_flight()
    finally:
        adapter.close()
        pipe.close()


def test_failed_backend_reset_does_not_clear_ownership():
    pipe = GPURingPipeline({"w": torch.ones(4)}, device=torch.device("cpu"))
    adapter = ExistingRingAcknowledgementAdapter(pipe).install()
    try:
        pipe.get_slot_for_loading(0)
        def failed_reset():
            raise RuntimeError("fence failed")
        adapter.originals["reset_in_flight"] = failed_reset
        with pytest.raises(RuntimeError, match="fence failed"):
            pipe.reset_in_flight()
        assert adapter.leases[0].phase == "LOADING_RESERVED"
    finally:
        adapter.close()
        pipe.close()
