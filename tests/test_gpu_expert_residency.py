"""Cache hit/eviction behavior and actual cross-stream consumer ordering."""
from collections import Counter
from pathlib import Path

import pytest
import torch

from src.shared_resources import SharedWeightResources
from src.lease_ack_bridge import BackendLeaseViolation
from src.upstream_bridge import NativeTierPolicy


class TestAdmission:
    __test__ = False
    def __init__(self, capacity, replace=True):
        self.capacity,self.replace,self.mode = capacity,replace,"apply"
        self.stats=Counter()
        self.provenance={"kind":"synthetic-test-policy"}
    def record_access(self,key):self.stats["accesses"]+=1
    def admit(self,key,residents):
        if len(residents)<self.capacity:return True,None
        return (True,residents[0]) if self.replace else (False,None)
    def acknowledge(self,event):
        self.stats["backend_acks"]+=1
        self.stats["backend_admissions"]+=event["status"]=="applied"


def backend(device,capacity=2,mode="apply",replace=True,count=16,compact=False):
    templates={"expert.w":((count,),torch.float32)}
    if compact:templates["dense.w"]=((3*2**19,),torch.float32)
    r=SharedWeightResources(templates,device,torch.float32,num_slots=2+capacity,
                           slot_layout="compact" if compact else "union")
    r.enable_expert_residency(TestAdmission(capacity,replace),mode)
    return r


@pytest.mark.parametrize("device",["cpu","cuda"])
@pytest.mark.parametrize("mode",["shadow","apply"])
def test_hits_skip_supplier_and_dma_with_atomic_evictions(device,mode):
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    r=backend(device,mode=mode)
    supplied=[]
    try:
        outputs=[]
        for key in (1,2,1,3,1):
            def supplier():
                supplied.append(key)
                return {"expert.w":torch.full((16,),float(key))}
            lease,views,nbytes=r.residency.acquire(key,("expert",0,key),supplier)
            outputs.append(views["expert.w"].sum())
            r.release(lease)
        if device=="cuda":torch.cuda.synchronize()
        assert [o.item() for o in outputs]==[16,32,16,48,16]
        s=r.snapshot()
        assert s["computes"]==5
        if mode=="apply":
            assert supplied==[1,2,3,1]
            assert s["loads"]==4 and s["resident_acquires"]==1
            assert s["logical_h2d_bytes"]["expert"]==4*64
            assert s["expert_residency"]["stats"]["admissions_confirmed"]==4
            assert s["expert_residency"]["stats"]["evictions"]==2
            assert s["expert_residency"]["stats"]["avoided_h2d_bytes"]==64
        else:
            assert supplied==[1,2,1,3,1]
            assert s["loads"]==5 and s["resident_acquires"]==0
            assert s["expert_residency"]["stats"]["shadow_hits"]==1
            assert s["expert_residency"]["native_policy"]["backend_admissions"]==0
            assert s["expert_residency"]["resident_experts"]==[]
            assert set(r.contents)=={1}
    finally:
        r.close()
    assert r.residency.entries=={} and r.ring.arena is None


def test_live_victim_stale_release_and_failed_read_do_not_destroy_resident():
    r=backend("cpu",capacity=1)
    try:
        lease,views,_=r.residency.acquire(1,("expert",0,1),lambda:{"expert.w":torch.ones(16)})
        def must_not_read():raise AssertionError("Unexpected host read")
        with pytest.raises(RuntimeError,match="active"):
            r.residency.acquire(2,("expert",0,2),must_not_read)
        r.release(lease)
        def broken_read():raise OSError("read failed")
        with pytest.raises(OSError,match="read failed"):
            r.residency.acquire(2,("expert",0,2),broken_read)
        hit,views,size=r.residency.acquire(1,("expert",0,1),must_not_read)
        assert size==0 and torch.equal(views["expert.w"],torch.ones(16))
        with pytest.raises(RuntimeError,match="stale"):
            r.release(lease)
        r.release(hit)
        with pytest.raises(BackendLeaseViolation):
            r.ring.acquire_resident_for_compute(8,999)
    finally:r.close()


@pytest.mark.parametrize("victim", [None, 999, 2])
def test_invalid_policy_proposal_cannot_read_or_evict(victim, monkeypatch):
    r=backend("cpu",capacity=1)
    try:
        lease,_,_=r.residency.acquire(1,("expert",0,1),lambda:{"expert.w":torch.ones(16)})
        r.release(lease)
        monkeypatch.setattr(r.residency.policy,"admit",lambda *_:(True,victim))
        def must_not_read():pytest.fail("Invalid proposal reached the weight reader")
        with pytest.raises(ValueError,match="victim|eviction"):
            r.residency.acquire(2,("expert",0,2),must_not_read)
        lease,views,size=r.residency.acquire(1,("expert",0,1),must_not_read)
        assert size==0 and torch.equal(views["expert.w"],torch.ones(16))
        r.release(lease)
        assert r.ring.total_loads==1
    finally:r.close()


def test_admission_ack_requires_observed_dma_completion(monkeypatch):
    r=backend("cpu",capacity=1)
    try:
        original=r.ack.poll
        def pending(slot):
            return dict(original(slot),dma_done_confirmed=False)
        monkeypatch.setattr(r.ack,"poll",pending)
        lease,_,_=r.residency.acquire(1,("expert",0,1),lambda:{"expert.w":torch.ones(16)})
        assert r.residency.stats["admissions_submitted"]==1
        assert r.residency.stats["admissions_confirmed"]==0
        assert r.residency.policy.stats["backend_admissions"]==0
        assert r.residency.events[-1]["status"]=="submitted"
        monkeypatch.setattr(r.ack,"poll",original)
        r.residency.confirm()
        r.residency.confirm()
        assert r.residency.stats["admissions_confirmed"]==1
        assert r.residency.policy.stats["backend_admissions"]==1
        assert r.residency.events[-1]["dma_physically_complete"] is True
        r.release(lease)
    finally:r.close()


@pytest.mark.parametrize("device",["cpu","cuda"])
@pytest.mark.parametrize("compact",[False,True])
def test_real_native_policy_rejects_then_physically_evicts(device,compact):
    dll=Path(__file__).resolve().parents[1]/"data/native/colibri_tier.dll"
    if not dll.exists():pytest.skip("Build native tier bridge first")
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    templates={"expert.w":((16,),torch.float32)}
    if compact:templates["dense.w"]=((3*2**19,),torch.float32)
    r=SharedWeightResources(templates,device,torch.float32,num_slots=3,slot_layout="compact" if compact else "union")
    policy=NativeTierPolicy(dll,4,1)
    r.enable_expert_residency(policy,"apply")
    try:
        outputs=[]
        keys=[0]*9+[1]*22
        for key in keys:
            lease,views,_=r.residency.acquire(key,("expert",0,key),
                lambda:{"expert.w":torch.full((16,),float(key+1))})
            outputs.append(views["expert.w"].sum())
            r.release(lease)
        if device=="cuda":torch.cuda.synchronize()
        assert [v.item() for v in outputs]==[16*(k+1) for k in keys]
        s=r.snapshot()
        assert s["expert_residency"]["resident_experts"]==[1]
        assert s["expert_residency"]["stats"]["evictions"]==1
        assert policy.stats["backend_admissions"]==2
        assert policy.stats["backend_rejections"]>0 and policy.stats["native_calls"]>0
        assert s["loads"]+s["resident_acquires"]==s["computes"]==len(keys)
        assert s["active_slots"]==s["violations"]==0
    finally:r.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
@pytest.mark.parametrize("compact",[False,True])
def test_cached_read_chains_consumers_before_eviction(monkeypatch,compact):
    r=backend("cuda",capacity=1,count=4096,compact=compact)
    streams=[torch.cuda.Stream() for _ in range(3)]
    waits=[]
    event=r.ring.slots[2].compute_done_event
    original=torch.cuda.Stream.wait_event
    def wait_event(stream,e):
        if e is event:waits.append(stream.cuda_stream)
        return original(stream,e)
    monkeypatch.setattr(torch.cuda.Stream,"wait_event",wait_event)
    try:
        with torch.cuda.stream(streams[0]):
            a,av,_=r.residency.acquire(1,("expert",0,1),lambda:{"expert.w":torch.ones(4096)})
            torch.cuda._sleep(150_000_000)
            out_a=av["expert.w"].sum()
            r.release(a)
        with torch.cuda.stream(streams[1]):
            b,bv,size=r.residency.acquire(1,("expert",0,1),lambda:pytest.fail("Hit must not read host weights"))
            out_b=bv["expert.w"].sum()
            r.release(b)
        with torch.cuda.stream(streams[2]):
            c,cv,_=r.residency.acquire(2,("expert",0,2),lambda:{"expert.w":torch.full((4096,),2.)})
            out_c=cv["expert.w"].sum()
            r.release(c)
        torch.cuda.synchronize()
        assert [out_a.item(),out_b.item(),out_c.item()]==[4096,4096,8192]
        assert size==0 and r.ring.total_loads==2 and r.ring.total_resident_acquires==1
        assert streams[1].cuda_stream in waits, "Cached reader did not wait for the prior consumer"
        assert r.ring.transfer_stream.cuda_stream in waits, "Eviction did not wait for the last consumer"
    finally:r.close()
