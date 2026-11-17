"""Actual elastic ownership, standby, strided kernels and reader fences."""
import pytest
import torch

from src.dynamic_working_memory import DynamicWorkingMemory
from src.quantized_experts import QuantizedExpertCache
from test_unified_working_memory import make_memory

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('bits',[4,6])
@torch.no_grad()
def test_dynamic_pages_reused_between_dense_and_experts_without_copy(bits):
    _,cache,ring,_=make_memory(bits,unify=False)
    memory=DynamicWorkingMemory(ring,cache,24*cache.expert_bytes,min_region_experts=4)
    reference=QuantizedExpertCache(cache.base,bits,cache.temporary_reserve_bytes+8*cache.expert_bytes)
    reference.enabled_layers={0,1,2,3}
    try:
        ptr=memory.gpu.data_ptr()
        assert memory.stats()['roles']['C']['reserved_bytes']==0
        # Fill the whole shared pool, then borrow cached pages for real dense DMA.
        for i in range(memory.num_slots):
            with memory.lease(i//96,[i%96],cache._load_packed):pass
        memory.begin_layer(0)
        payload={'weight':torch.randn(128,128,dtype=torch.bfloat16).pin_memory()}
        a=ring.get_slot_for_loading(0)
        ring.load_tensors_async(a,payload)
        ring.acquire_slot_for_compute(0)
        observed=a.active_tensors['weight'].clone()
        b=ring.get_slot_for_loading(1)
        ring.load_tensors_async(b,payload)
        stats=memory.stats()
        assert stats['dense_blocks']==2 and stats['dense_reclaims_expert']>0
        assert stats['roles']['A']['used_bytes']>stats['roles']['B']['used_bytes']
        assert stats['roles']['C']['used_bytes']==0
        assert not ring.slots[2].tensors
        assert all(v.untyped_storage().data_ptr()==ptr for s in (a,b) for v in s.tensors.values())
        ring.release_slot_after_compute(a)
        memory.begin_layer(1)
        ring.acquire_slot_for_compute(1)
        ring.release_slot_after_compute(b)
        torch.cuda.synchronize()
        assert torch.equal(observed.cpu(),payload['weight'])
        assert memory.stats()['dense_blocks']==0 and all(not s.tensors for s in ring.slots)
        # Returned pages can hold experts again. One batched kernel reads AOS strides.
        for layer in range(3):
            memory.begin_layer(layer)
            ids=torch.tensor([[0,1,2,3,4,5,6,7]],device='cuda')
            weights=torch.softmax(torch.randn(1,8,device='cuda'),-1).to(torch.bfloat16)
            hidden=torch.randn(1,128,device='cuda',dtype=torch.bfloat16)
            assert torch.equal(reference.try_fused(layer,hidden,ids,weights,96),
                               cache.try_fused(layer,hidden,ids,weights,96))
        assert all(n>0 for n in memory.area_reads)
        assert memory.gpu.data_ptr()==ptr and memory.gpu_bytes<=memory.budget_bytes
    finally:
        reference.close()
        ring.close()
        cache.close()


def test_standby_cannot_be_filled_and_reset_releases_cancelled_dense():
    _,cache,ring,_=make_memory(4,unify=False)
    memory=DynamicWorkingMemory(ring,cache,24*cache.expert_bytes)
    payload={'weight':torch.ones(128,128,dtype=torch.bfloat16).pin_memory()}
    try:
        for layer in (0,1):
            slot=ring.get_slot_for_loading(layer)
            ring.load_tensors_async(slot,payload)
        slot=ring.get_slot_for_loading(2)
        with pytest.raises(RuntimeError,match='Standby must stay empty'):
            ring.load_tensors_async(slot,payload)
        ring.reset_in_flight()
        assert not memory.dense_blocks and all(not s.tensors for s in ring.slots)
        assert memory.stats()['roles']['C']['reserved_bytes']==0
    finally:
        ring.close()
        cache.close()


@torch.no_grad()
def test_retired_dense_bytes_wait_for_gpu_before_expert_overwrite():
    _,cache,ring,_=make_memory(4,unify=False)
    memory=DynamicWorkingMemory(ring,cache,24*cache.expert_bytes)
    slow=torch.cuda.Stream()
    payload={'weight':torch.ones(128,128,dtype=torch.bfloat16).pin_memory()}
    try:
        slot=ring.get_slot_for_loading(0)
        ring.load_tensors_async(slot,payload)
        with torch.cuda.stream(slow):
            ring.acquire_slot_for_compute(0)
            torch.cuda._sleep(200_000_000)
            observed=slot.active_tensors['weight'].clone()
            ring.release_slot_after_compute(slot)
        stats=memory.stats()
        assert stats['retired_waiting_cuda_bytes']>0
        assert stats['roles']['C']['reserved_bytes']==0
        # Demand may enqueue a reuse, but its stream must wait for the reader.
        with memory.lease(0,[0],cache._load_packed):pass
        torch.cuda.synchronize()
        assert torch.equal(observed.cpu(),payload['weight'])
        assert memory.stats()['retired_waiting_cuda_bytes']==0
    finally:
        slow.synchronize()
        ring.close()
        cache.close()
