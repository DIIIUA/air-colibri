"""Shared allocation, direct experts, real reader fences and bounded admission."""
from contextlib import ExitStack
from types import SimpleNamespace
import threading

import pytest
import torch

from src.gpu_ring_pipeline import GPURingPipeline
from src.quantized_experts import QuantizedExpertCache
from src.quantized_projection import pack_weight
from src.unified_working_memory import UnifiedWorkingMemory

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def make_memory(bits, unify=True):
    torch.manual_seed(78)
    weights=torch.randn(96,3,128,128,dtype=torch.bfloat16)*.04
    def load(layer,ids,device,dtype,allow_evict):
        block=weights[torch.tensor(ids)]
        return tuple(block[:,i].to(device=device,dtype=dtype) for i in range(3))
    base=SimpleNamespace(device=torch.device('cuda:0'),uniform_shapes=True,has_gate=True,
        inter_dim=128,hidden_dim=128,_spans=list(range(96)),load_batch=load,
        close=lambda:None,stats=lambda:{})
    size=int(3*128*128*(bits/8+4/64))
    cache=QuantizedExpertCache(base,bits,8*6*128*128+32*2**20+8*size)
    cache.enabled_layers={0,1,2,3}
    ring=GPURingPipeline({'weight':((128,128),torch.bfloat16)},device=base.device)
    memory=UnifiedWorkingMemory(ring,cache,24*size) if unify else None
    return weights,cache,ring,memory


@pytest.mark.parametrize('bits',[4,6])
@torch.no_grad()
def test_one_allocation_and_direct_routed_parity(bits):
    w,cache,ring,memory=make_memory(bits)
    try:
        assert memory.gpu_bytes<=memory.budget_bytes
        ptr=memory.gpu.untyped_storage().data_ptr()
        assert all(v.untyped_storage().data_ptr()==ptr for s in ring.slots for v in s.tensors.values())
        assert all(v.untyped_storage().data_ptr()==ptr for b in memory.banks for d in b for v in d.values())
        stream=torch.cuda.Stream()
        memory.set_desired({(i,i) for i in range(3)})
        for i in range(3):
            packed=(pack_weight(torch.cat((w[i,0],w[i,1])),bits),pack_weight(w[i,2],bits))
            status,record=memory.prepare((i,i),packed,stream,memory.epoch)
            assert status=='published' and record['bank']==i
        stream.synchronize()
        for i in range(3):
            def no_load(*args):raise AssertionError('Prepared expert was loaded again')
            with memory.lease(i,[i],no_load) as entries:
                assert entries[0]['bank']==i
                assert entries[0]['gu']['packed_lo'].untyped_storage().data_ptr()==ptr
        # Different areas feed one routed result without BF16 intermediate sums.
        reference=QuantizedExpertCache(cache.base,bits,cache.temporary_reserve_bytes+8*cache.expert_bytes)
        reference.enabled_layers={0,1,2,3}
        for layer in (0,1,2):
            ids=torch.tensor([[0,1,2,3,4,5,6,7]],device='cuda')
            routes=torch.softmax(torch.randn(1,8,device='cuda'),-1).to(torch.bfloat16)
            hidden=torch.randn(1,128,device='cuda',dtype=torch.bfloat16)
            assert torch.equal(reference.try_fused(layer,hidden,ids,routes,96),
                               cache.try_fused(layer,hidden,ids,routes,96))
        assert memory.stats()['warm_bank_to_cache_copies']==0
        assert all(n>0 for n in memory.area_reads)
        reference.close()
    finally:
        ring.close()
        cache.close()


def test_budget_rejection_preserves_original_storage():
    _,cache,ring,_=make_memory(4,unify=False)
    try:
        ptr=ring.arena.data_ptr()
        old_gu=cache.gu
        with pytest.raises(ValueError,match='cannot fit'):
            UnifiedWorkingMemory(ring,cache,0,min_region_experts=8)
        assert ring.arena.data_ptr()==ptr and cache.gu is old_gu
        assert ring.working_memory is None and cache.shared_memory is None
    finally:
        ring.close()
        cache.close()


def test_unified_producer_reset_inflight_error_and_exact_fallback():
    from src.unified_prefetch import UnifiedExpertPrefetch
    w,cache,ring,memory=make_memory(4)
    entered,release=threading.Event(),threading.Event()
    def load(layer,eid):
        if layer==99:
            raise OSError('injected optional read failure')
        if eid==0:
            entered.set()
            assert release.wait(10)
        return tuple(w[eid])
    worker=UnifiedExpertPrefetch(memory,cache,3,load,lambda:None,2**20)
    try:
        worker.submit(0,[0,1],lane=0,region=0,step=1)
        assert entered.wait(10)
        worker.reset()
        worker.submit(1,[2],lane=1,region=1,step=2)
        release.set()
        worker.drain()
        assert (0,0) not in memory.directory and (1,2) in memory.directory
        assert worker.counters['stale']==1
        worker.submit(99,[3],lane=2,region=99,step=3)
        worker.drain()
        assert worker.counters['errors']==1 and (99,3) not in memory.directory
        with memory.lease(0,[3],cache._load_packed) as entries:
            expected=pack_weight(torch.cat((w[3,0],w[3,1])),4)
            assert all(torch.equal(entries[0]['gu'][k].cpu(),v) for k,v in expected.items())
    finally:
        release.set()
        worker.close()
        ring.close()
        cache.close()
    assert not worker.thread.is_alive()


@torch.no_grad()
def test_prefetch_cannot_reuse_a_live_reader_and_refinement_keeps_common():
    w,cache,ring,memory=make_memory(4)
    slow=torch.cuda.Stream()
    transfer=torch.cuda.Stream()
    try:
        # Fill all records; no unrelated free space may hide an unsafe reuse.
        for eid in range(memory.num_slots):
            with memory.lease(0,[eid],cache._load_packed):pass
        torch.cuda.synchronize()
        original=memory.records[memory.directory[(0,0)]]
        generation=original['generation']
        snapshot=memory._views(original)[0]['packed_lo'].clone()
        memory.set_desired({(1,90)})
        with ExitStack() as stack:
            for eid in range(1,memory.num_slots):
                stack.enter_context(memory.lease(0,[eid],cache._load_packed))
            with torch.cuda.stream(slow):
                with memory.lease(0,[0],cache._load_packed) as refs:
                    torch.cuda._sleep(300_000_000)
                    observed=refs[0]['gu']['packed_lo'].clone()
            packed=(pack_weight(torch.cat((w[90,0],w[90,1])),4),pack_weight(w[90,2],4))
            status,_=memory.prepare((1,90),packed,transfer,memory.epoch)
            assert status is None and original['generation']==generation
            slow.synchronize()
            assert torch.equal(observed,snapshot)
            status,record=memory.prepare((1,90),packed,transfer,memory.epoch)
            assert status=='published' and record is original
        transfer.synchronize()
        memory.set_desired({(1,90),(2,91)})
        before=record['generation']
        assert memory.prepare((1,90),packed,transfer,memory.epoch)[0]=='retained'
        assert record['generation']==before
        memory.reset_task()
        assert memory.prepare((2,91),packed,transfer,memory.epoch-1)[0]=='stale'
    finally:
        slow.synchronize()
        transfer.synchronize()
        ring.close()
        cache.close()
