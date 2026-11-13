"""Disjoint role-specific views in the original single pinned/GPU arenas."""
import pytest
import torch

from src.bounded_memory_manager import BoundedPinnedBufferPool, SlotArenaLayout
from src.gpu_ring_pipeline import GPURingPipeline
from src.shared_resources import SharedWeightResources


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_compact_slots_share_storage_without_cross_role_writes(device):
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    mib=2**20
    templates={"dense.w":((3*mib//2,),torch.float32),"expert.w":((mib//2,),torch.float32)}
    r=SharedWeightResources(templates,device,torch.float32,num_slots=4,slot_layout="compact")
    try:
        assert r.ring.slot_strides==(6*mib,2*mib,2*mib,2*mib)
        assert r.ring.slot_offsets==r.pool.slot_offsets==(0,6*mib,8*mib,10*mib)
        assert r.ring.slot_stride is None
        assert r.ring.arena.nbytes==r.pool.arena.nbytes==12*mib
        for i,slot in enumerate(r.ring.slots):
            for value in slot.tensors.values():
                assert value.untyped_storage().data_ptr()==r.ring.arena.data_ptr()
                assert value.data_ptr()==r.ring.arena.data_ptr()+r.ring.slot_offsets[i]
        dense,d=r.acquire(0,("dense",0),{"dense.w":torch.full((3*mib//2,),7.)})
        results=[]
        for sid in (1,2,3,1,2,3):
            lease,e=r.acquire(sid,("expert",0,sid),{"expert.w":torch.full((mib//2,),float(sid))})
            results.append((e["expert.w"].sum(),d["dense.w"].sum()))
            r.release(lease)
        before=r.pool.arena.clone()
        loads=r.ring.total_loads
        with pytest.raises(ValueError,match="Unknown pinned"):
            r.acquire(1,("expert",0,9),{"expert.w":torch.zeros(mib//2),"dense.w":torch.zeros(3*mib//2)})
        assert torch.equal(before,r.pool.arena) and r.ring.total_loads==loads
        r.release(dense)
        if device=="cuda":torch.cuda.synchronize()
        assert [(a.item(),b.item()) for a,b in results]==[(sid*mib//2,7*3*mib//2) for sid in (1,2,3,1,2,3)]
        assert r.snapshot()["active_slots"]==r.snapshot()["violations"]==0
    finally:r.close()


@pytest.mark.parametrize("factory",[BoundedPinnedBufferPool,GPURingPipeline])
def test_invalid_geometry_is_rejected_before_binding(factory):
    layout=SlotArenaLayout.from_templates({"w":(16,)},dtype=torch.float32)
    wrong=SlotArenaLayout.from_templates({"w":(16,)},dtype=torch.bfloat16)
    with pytest.raises(ValueError,match="count"):
        factory(num_slots=2,dtype=torch.float32,arena_layout=layout,slot_layouts=[layout])
    with pytest.raises(ValueError,match="dtype"):
        factory(num_slots=2,dtype=torch.float32,arena_layout=layout,slot_layouts=[layout,wrong])
    with pytest.raises(ValueError,match="outside"):
        layout.bind_slot_views(torch.empty(layout.slot_stride,dtype=torch.uint8),0,base_offset_bytes=256)
