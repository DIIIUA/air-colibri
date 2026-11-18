"""Joint anchor/three-bank checks, with independent logits, KV and prior ablations."""
import copy
from types import SimpleNamespace
import pytest
import torch

from aircolibri_cli.thought_buffer import InitialProbabilityAnchor, AnchorStepTelemetry, ColibriThoughtBuffer
from test_probability_warmup import make_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('kind',['granitemoe','olmoe'])
@pytest.mark.parametrize('bits',[4,6])
@pytest.mark.parametrize('unified',[False,True,'dynamic'])
@torch.no_grad()
def test_anchor_probes_continuation_and_three_banks_match_anchor_only(tmp_path,kind,bits,unified):
    from src.aircolibri_integrated import AirColibriModel
    from aircolibri_cli.model_source import open_model_source
    make_model(tmp_path,kind)
    expert_bytes=int(3*128*128*(bits/8+4/64))
    budget=8*6*128*128+32*1024**2+8*expert_bytes
    kwargs=dict(weight_format=f'q{bits}',universal_quantization=True,
                vram_resident_layers=1,num_slots=3,expert_cache_bytes=budget)
    refs=[]
    for joint in (False,True):
        with AirColibriModel(str(tmp_path),model_source=open_model_source(tmp_path),**kwargs) as engine:
            warm=engine.enable_probability_warmup(region_experts=4,gpu_budget_bytes=2**20,
                                                  unified_memory=bool(unified),dynamic_regions=unified=='dynamic') if joint else None
            u=SimpleNamespace(engine=engine,tokenizer=engine.tokenizer,fast_qwen=False,
                              captured={},_last_prompt_opened_think=True,_last_prompt_chatml_think=True)
            def prepare(*args,**kwargs):
                u._last_prompt_opened_think=False
                u._last_prompt_chatml_think=False
                return [1,2,3],0,None,0.0
            u._prepare_prompt_and_cache=prepare
            anchor=InitialProbabilityAnchor()
            past=None
            original=None
            initial_map=None
            next_id=1
            affected=[]
            for step in range(36):
                prefill=step in (0,8)
                tokens=[1,4,6,8] if step==0 else ([5,9,11] if step==8 else [next_id])
                if joint:tokens=refs[step][0]
                raw,past=engine._forward_step(torch.tensor([tokens]),past,prefill=prefill)
                kv=[(l.keys.clone(),l.values.clone()) for l in past.layers]
                if prefill:
                    if warm:
                        before=(copy.deepcopy(warm.region_map.stats()),dict(warm.submitted_versions),
                                warm.step,warm.prefetch.epoch,warm.counters['map_submissions'])
                    out=anchor.on_prefill_end(u,past,raw)
                    assert u._last_prompt_opened_think and u._last_prompt_chatml_think
                    assert all(torch.equal(l.keys,k) and torch.equal(l.values,v) for l,(k,v) in zip(past.layers,kv))
                    if warm:
                        after=(warm.region_map.stats(),dict(warm.submitted_versions),
                               warm.step,warm.prefetch.epoch,warm.counters['map_submissions'])
                        assert before==after
                        assert engine.model_source._expert_loader.warmup is warm.prefetch
                        assert warm._auxiliary_depth==0
                        warm.prefetch.drain()  # deterministic correctness test; live trials do not do this
                else:
                    out=anchor.transform_logits(raw,past,step)
                    anchor.on_step_committed(past,1)
                if original is None:original=anchor.phi0.clone()
                assert torch.equal(original,anchor.phi0)
                if warm:
                    current=dict(warm.region_map.initial)
                    if initial_map is None:initial_map=current
                    assert initial_map==current
                # Remove only the original channel, leaving current working memory intact.
                ablated=copy.copy(anchor)
                ablated.telemetry=AnchorStepTelemetry()
                ablated.phi0=torch.zeros_like(anchor.phi0)
                without=ablated.transform_logits(raw,past,step)
                affected.append(float((out.float().softmax(-1)-without.float().softmax(-1)).abs().sum()))
                next_id=int(out.argmax(-1).item())
                saved=(tokens,raw.cpu(),out.cpu(),[(k.cpu(),v.cpu()) for k,v in kv],anchor.phi0.cpu(),anchor.phi_buf.cpu())
                if joint:
                    ref=refs[step]
                    assert torch.equal(saved[1],ref[1]) and torch.equal(saved[2],ref[2])
                    assert all(torch.equal(a,c) and torch.equal(b,d) for (a,b),(c,d) in zip(saved[3],ref[3]))
                    assert torch.equal(saved[4],ref[4]) and torch.equal(saved[5],ref[5])
                else:refs.append(saved)
            assert any(v>0 for v in affected[:8]) and any(v>0 for v in affected[32:])
            assert anchor.telemetry.initial_rows_active>0 and anchor.telemetry.phi0_chosen_late
            assert anchor.snap0.kv_lengths=={i:4 for i in range(5)}
            if warm:
                if unified:
                    memory=engine.model_source._expert_loader.shared_memory
                    assert memory.stats()['warm_bank_to_cache_copies']==0
                    assert memory.gpu_bytes<=memory.budget_bytes
                    assert all(n>0 for n in memory.area_reads)
                    assert all(c['load']>0 and c['acquire']>0 and c['release']>0 for c in memory.area_dense)
                    assert memory.stats()['aux_leases']>0
                assert warm.counters['auxiliary_forwards']==4  # two neutral + two task probes
                assert warm.counters['prefill_chunks']==2
                assert all(n>0 for n in warm.prefetch.stats()['bank_consumptions'])
                assert warm.counters['route_checks']==34*len(warm.targets)
                if unified=='dynamic':
                    assert memory.stats()['roles']['C']['reserved_bytes']==0
                    assert memory.stats()['peak_dense_blocks']==2
                with pytest.raises(RuntimeError,match='scope failure'):
                    with warm.auxiliary_forwards():
                        with warm.auxiliary_forwards():
                            assert warm.cache.warmup is None
                            raise RuntimeError('scope failure')
                assert warm._auxiliary_depth==0 and warm.cache.warmup is warm.prefetch
                engine.reset_after_cancel()
                assert not warm.region_map.initial
        if warm:assert not warm.prefetch.thread.is_alive()


@pytest.mark.parametrize('bits',[4,6])
@pytest.mark.parametrize('unified',[False,True,'dynamic'])
def test_public_thought_buffer_entrypoints_with_warmup(tmp_path,monkeypatch,bits,unified):
    from dataclasses import replace
    from aircolibri_cli import unified_runtime as runtime
    make_model(tmp_path,'granitemoe')
    original=runtime.plan_for_checkpoint
    def ring(*args,**kwargs):
        profile,plan,mtp=original(*args,**kwargs)
        expert_bytes=int(3*128*128*(bits/8+4/64))
        cache_budget=8*6*128*128+32*2**20+8*expert_bytes
        return profile,replace(plan,full_residency=False,resident_layers=1,ring_slots=3,
                               budget_bytes=plan.weights_bytes+256*2**20+cache_budget),mtp
    monkeypatch.setattr(runtime,'plan_for_checkpoint',ring)
    setup=InitialProbabilityAnchor.on_prefill_end
    checked=[]
    def observe(anchor,u,past,logits):
        warm=u.engine.probability_warmup
        before=(copy.deepcopy(warm.region_map.stats()),warm.step,warm.prefetch.epoch)
        out=setup(anchor,u,past,logits)
        assert before==(warm.region_map.stats(),warm.step,warm.prefetch.epoch)
        checked.append(anchor)
        return out
    monkeypatch.setattr(InitialProbabilityAnchor,'on_prefill_end',observe)
    with runtime.AirColibriUnifiedEngine(cpu_first=False, model_dir=tmp_path,weight_format=f'q{bits}',enable_mtp=False,
                                        verbose=False,max_context_tokens=2048,prefill_chunk_tokens=32) as u:
        warm=u.engine.enable_probability_warmup(region_experts=4,gpu_budget_bytes=2**20,
            unified_memory=bool(unified),dynamic_regions=unified=='dynamic')
        buffer=ColibriThoughtBuffer(u)
        result=buffer.ask_anchored([{'role':'user','content':'Remember number 7.'}],max_new_tokens=8)
        assert result['anchor_telemetry']['initial_rows_active']>0
        assert result['metrics']['generated_tokens']>0
        result=buffer.ask_wave_buffered([{'role':'user','content':'Remember number 7.'}],
                                       buffer_tokens=4,answer_tokens=6,min_buffer_alignment=100)
        assert not result['retried_drifted_wave']
        assert not result['buffer_verified'] and result['alignment_is_diagnostic_only']
        assert result['synth_telemetry']['initial_rows_active']>0
        assert result['initial_phi0_sha256']==result['wave1_initial_phi0_sha256']==result['synth_initial_phi0_sha256']
        assert result['initial_probability_sha256'] is not None
        assert result['initial_probability_sha256']==result['wave1_initial_probability_sha256']==result['synth_initial_probability_sha256']
        assert len(checked)>=3 and all(a.phi0 is not None for a in checked)
        assert all(n>0 for n in warm.prefetch.stats()['bank_consumptions'])
        assert warm._auxiliary_depth==0 and u._active_thought_anchor is None
        u.reset_session()
        assert not warm.region_map.initial
    assert not warm.prefetch.thread.is_alive()


@pytest.mark.parametrize('bits',[4,6])
@torch.no_grad()
def test_unified_cancel_mid_forward_and_restart(tmp_path,bits):
    from aircolibri_cli.model_source import open_model_source
    from aircolibri_cli.unified_runtime import GenerationCancelled
    from src.aircolibri_integrated import AirColibriModel
    from src.gpu_ring_pipeline import SlotState
    make_model(tmp_path,'granitemoe')
    expert_bytes=int(3*128*128*(bits/8+4/64))
    budget=8*6*128*128+32*2**20+44*expert_bytes
    with AirColibriModel(str(tmp_path),model_source=open_model_source(tmp_path),
            weight_format=f'q{bits}',universal_quantization=True,vram_resident_layers=1,
            num_slots=3,expert_cache_bytes=budget) as engine:
        warm=engine.enable_unified_memory(region_experts=4)
        memory=warm.cache.shared_memory
        tokens=torch.tensor([[1,4,6,8]])
        expected,kv=engine._forward_step(tokens,None,prefill=True)
        expected_kv=[(l.keys.clone(),l.values.clone()) for l in kv.layers]
        expected=expected.clone()
        engine.reset_after_cancel()
        count=0
        def cancel():
            nonlocal count
            count+=1
            if count==3:raise GenerationCancelled('cancel at layer boundary')
        engine._step_cancel_check=cancel
        with pytest.raises(GenerationCancelled,match='cancel at layer'):
            engine._forward_step(tokens,None,prefill=True)
        engine.reset_after_cancel()
        warm.prefetch.drain()
        assert not warm.region_map.initial and not memory.desired
        assert not any(memory.dense_active)
        assert all(s.state==SlotState.FREE for s in engine.pipeline.slots)
        assert memory.stats()['active_leases']==0
        actual,kv=engine._forward_step(tokens,None,prefill=True)
        assert torch.equal(actual,expected)
        assert all(torch.equal(l.keys,k) and torch.equal(l.values,v)
                   for l,(k,v) in zip(kv.layers,expected_kv))
    assert not warm.prefetch.thread.is_alive()
