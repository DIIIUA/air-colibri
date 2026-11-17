"""Real transfers and exact fallback; predictions never remove actual work."""
import threading
import pytest
import torch
from src.probability_warmup import PrefillRegionMap, ThreeBankExpertPrefetch
from src.quantized_projection import pack_weight, packed_specs


def test_prefill_map_is_retained_until_persistent_route_change():
    plan=PrefillRegionMap({0:8},4,drift_patience=3,recent_window=4)
    assert plan.seed(0,[9,8,7,6,0,0,0,0])
    original=plan.stats()['initial_sha256']
    region=plan.regions[0]
    for _ in range(12):assert not plan.observe(0,[0,1])
    assert not plan.observe(0,[6,7])  # one miss is not a trajectory change
    assert not plan.observe(0,[0,1])
    assert plan.regions[0] is region and plan.revisions[0]==1
    assert not plan.observe(0,[6,7])
    assert not plan.observe(0,[6,7])
    assert plan.observe(0,[6,7])
    assert {6,7}.issubset(plan.regions[0]) and plan.revisions[0]==2
    assert plan.stats()['initial_sha256']==original
    plan.seed(0,[0,0,0,0,5,6,7,8])  # later user input preserves the first map
    assert plan.stats()['initial_sha256']==original


def test_initial_map_contributes_beyond_breaking_recent_ties():
    plan = PrefillRegionMap({0: 4}, 2)
    plan.seed(0, [100, 0, 0, 0])
    fingerprint = plan.stats()['initial_sha256']
    plan.seed(0, [0, 8, 1, 1])
    assert set(plan.regions[0]) == {0, 1}
    assert plan.stats()['initial_sha256'] == fingerprint


@pytest.mark.parametrize('bits',[4,6])
def test_three_banks_exact_weights_wrong_forecasts_and_reset(bits):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(32)
    w=torch.randn(8,3,128,128,dtype=torch.bfloat16)*.04
    entered,release=threading.Event(),threading.Event()
    block=[False]
    def load(layer,eid):
        if block[0]:
            entered.set()
            assert release.wait(5)
        return tuple(w[eid])
    worker=ThreeBankExpertPrefetch(128,128,bits,4,load,lambda:None,'cuda:0',host_cache_bytes=2**20)
    dst=tuple({k:torch.empty(sh,dtype=dt,device='cuda') for k,(sh,dt) in packed_specs('weight',shape,bits).items()}
              for shape in ((256,128),(128,128)))
    try:
        for i in range(6):
            worker.submit(0,[i%8,(i+1)%8])
            worker.drain()
            assert worker.try_copy(0,i%8,dst)
            expected=(pack_weight(torch.cat((w[i%8,0],w[i%8,1])),bits),pack_weight(w[i%8,2],bits))
            assert all(torch.equal(d[k].cpu(),e[k]) for d,e in zip(dst,expected) for k in d)
            assert not worker.try_copy(99,i%8,dst)
        assert all(x>0 for x in worker.stats()['bank_rotations'])
        # Invalidate an actively prepared request and replace a queued request.
        block[0]=True
        worker.submit(1,[0])
        assert entered.wait(5)
        worker.submit(1,[1])
        worker.submit(1,[2])
        worker.reset()
        release.set()
        worker.drain()
        assert not worker.try_copy(1,0,dst)
        assert worker.stats()['stale']>=1 and worker.stats()['superseded']>=1
        assert worker.stats()['host_cache_bytes']<=worker.stats()['host_cache_cap']
        block[0]=False
        worker.submit(0,[7])
        worker.drain()
        assert worker.try_copy(0,7,dst)
    finally:
        release.set()
        worker.close()
    assert not worker.thread.is_alive()


def test_optional_source_failure_does_not_publish_partial_bank():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    def fail(layer,eid):raise OSError('injected source failure')
    worker=ThreeBankExpertPrefetch(128,128,4,2,fail,lambda:None,'cuda:0')
    try:
        worker.submit(0,[0,1])
        worker.drain()
        assert worker.stats()['errors']==1 and worker.stats()['published']==0
        assert not worker.try_copy(0,0,({},{}))
    finally:worker.close()


def test_bank_reuse_waits_for_actual_cuda_reader():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    w=torch.randn(4,3,128,128,dtype=torch.bfloat16)
    worker=ThreeBankExpertPrefetch(128,128,4,1,lambda layer,eid:tuple(w[eid]),lambda:None,'cuda:0')
    dst=tuple({k:torch.empty(sh,dtype=dt,device='cuda') for k,(sh,dt) in packed_specs('weight',shape,4).items()}
              for shape in ((256,128),(128,128)))
    consumer=torch.cuda.Stream()
    try:
        for eid in range(3):
            worker.submit(0,[eid])
            worker.drain()
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(300_000_000)
            assert worker.try_copy(0,0,dst)
        worker.submit(0,[3])
        worker.drain()
        # Another bank can accept new work while bank 0 still has a reader.
        assert worker.trace[-1]['bank']!=0
        consumer.synchronize()
        expected=pack_weight(torch.cat((w[0,0],w[0,1])),4)
        assert all(torch.equal(dst[0][k].cpu(),v) for k,v in expected.items())
        assert worker.stats()['published']==4
    finally:
        consumer.synchronize()
        worker.close()


def make_model(folder,kind):
    import json
    from transformers import AutoConfig,AutoModelForCausalLM
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    from safetensors.torch import save_file
    from test_model_source import _write_tiny_hf_tokenizer
    cfg=AutoConfig.for_model(kind,vocab_size=259,hidden_size=128,intermediate_size=128,
        num_hidden_layers=5,num_attention_heads=4,num_key_value_heads=4,
        num_experts=8,num_local_experts=8,num_experts_per_tok=2,
        tie_word_embeddings=False,eos_token_id=0)
    torch.manual_seed(234)
    model=AutoModelForCausalLM.from_config(cfg,dtype=torch.bfloat16).eval()
    cfg.save_pretrained(folder)
    _write_tiny_hf_tokenizer(folder,vocab_size=259)
    tokenizer_path=folder/'tokenizer.json'
    tokenizer=json.loads(tokenizer_path.read_text(encoding='utf-8'))
    tokenizer['model']['vocab']={'<eos>':0,**{c:i+1 for i,c in enumerate(bytes_to_unicode().values())}}
    tokenizer_path.write_text(json.dumps(tokenizer),encoding='utf-8')
    save_file({k:v.contiguous() for k,v in model.state_dict().items()},str(folder/'model.safetensors'))


@pytest.mark.parametrize('kind',['granitemoe','olmoe'])
@pytest.mark.parametrize('bits',[4,6])
@torch.no_grad()
def test_predictive_region_ring_logits_kv_and_cancel(tmp_path,kind,bits):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from src.aircolibri_integrated import AirColibriModel
    from aircolibri_cli.model_source import open_model_source
    make_model(tmp_path,kind)
    # Exactly eight cache slots force experts out between layers.
    expert_bytes=int(3*128*128*(bits/8+4/64))
    budget=8*6*128*128+32*1024**2+8*expert_bytes
    kwargs=dict(weight_format=f'q{bits}',universal_quantization=True,
                vram_resident_layers=1,num_slots=3,expert_cache_bytes=budget)
    expected=[]
    steps=([1,4,6,8],[7],[9],[11],[13])
    with AirColibriModel(str(tmp_path),model_source=open_model_source(tmp_path),**kwargs) as base:
        past=None
        for tokens in steps:
            logits,past=base._forward_step(torch.tensor([tokens]),past_key_values=past)
            expected.append((logits.cpu(),[(l.keys.cpu(),l.values.cpu()) for l in past.layers]))
    del past,logits
    with AirColibriModel(str(tmp_path),model_source=open_model_source(tmp_path),**kwargs) as engine:
        warm=engine.enable_probability_warmup(region_experts=4,gpu_budget_bytes=2**20,host_cache_bytes=2**20)
        counts={'head':0,'attention':0}
        def head(*args):counts['head']+=1
        def attention(*args):counts['attention']+=1
        head_hook=engine.model.get_output_embeddings().register_forward_hook(head)
        attn_hook=engine.model.model.layers[0].self_attn.register_forward_hook(attention)
        past=None
        initial=None
        for tokens,(ref,state) in zip(steps,expected):
            logits,past=engine._forward_step(torch.tensor([tokens]),past_key_values=past)
            assert torch.equal(logits.cpu(),ref)
            assert all(torch.equal(l.keys.cpu(),k) and torch.equal(l.values.cpu(),v)
                       for l,(k,v) in zip(past.layers,state))
            warm.prefetch.drain()  # correctness test only; live timing never drains per token
            current=warm.region_map.stats()['initial_sha256']
            if initial is None:initial=current
            assert current==initial
        assert engine.model_source._expert_loader.warm_hits>0
        assert counts=={'head':len(steps),'attention':len(steps)}
        head_hook.remove(); attn_hook.remove()
        assert len(warm.region_map.initial)==5
        assert warm.stats()['policy']=='prefill_map_with_route_guard'
        assert warm.stats()['route_checks']==3*(len(steps)-1)
        assert warm.stats()['extra_lm_head_calls']==0
        assert all(x>0 for x in warm.prefetch.stats()['bank_rotations'])
        assert all(x>0 for x in warm.prefetch.stats()['bank_consumptions'])
        engine.reset_after_cancel()
        assert not warm.region_map.regions and not warm.region_map.initial
        assert not warm.prefetch.try_copy(0,0,({},{}))
        logits,past=engine._forward_step(torch.tensor([steps[0]]),past_key_values=None)
        assert torch.equal(logits.cpu(),expected[0][0])
    assert not warm.prefetch.thread.is_alive()


def test_unified_cancel_and_session_reset_clear_pending_predictions(tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from dataclasses import replace
    from aircolibri_cli import unified_runtime as runtime
    make_model(tmp_path,'olmoe')
    original=runtime.plan_for_checkpoint
    def ring(*args,**kwargs):
        profile,plan,mtp=original(*args,**kwargs)
        return profile,replace(plan,full_residency=False,resident_layers=1,ring_slots=3),mtp
    monkeypatch.setattr(runtime,'plan_for_checkpoint',ring)
    with runtime.AirColibriUnifiedEngine(model_dir=tmp_path,weight_format='q6',enable_mtp=False,
                                        verbose=False,max_context_tokens=128,prefill_chunk_tokens=8) as engine:
        warm=engine.engine.enable_probability_warmup(region_experts=4,gpu_budget_bytes=2**20)
        engine.engine._forward_step(torch.tensor([[1,2,3]]))
        before=warm.prefetch.epoch
        calls=[0]
        def cancel():
            calls[0]+=1
            return calls[0]>3
        with pytest.raises(runtime.GenerationCancelled):
            list(engine.stream_chat([{'role':'user','content':'hi'}],max_new_tokens=8,cancel_check=cancel))
        assert warm.prefetch.epoch>before and not warm.region_map.regions
        assert all(slot is None for slot in engine.engine.active_slot_per_layer)
        result=list(engine.stream_chat([{'role':'user','content':'hi'}],max_new_tokens=2,stop_on_eos=False))
        assert any(ev['type']=='done' for ev in result)
        engine.reset_session()
        assert not warm.region_map.regions and not warm.region_map.initial
        warm.prefetch.drain()
        assert not warm.prefetch.try_copy(0,0,({},{}))
        assert 'probability_warmup' in engine.status_snapshot()
    assert warm.prefetch.stats()['gpu_bytes']==0 and not warm.prefetch.thread.is_alive()


@pytest.mark.parametrize('bits',[4,6])
def test_refinement_preserves_published_common_and_abandons_obsolete_inflight(bits):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(71)
    w=torch.randn(7,3,128,128,dtype=torch.bfloat16)
    entered,release=threading.Event(),threading.Event()
    loaded=[]
    def load(layer,eid):
        loaded.append(eid)
        if eid==1:
            entered.set()
            assert release.wait(5)
        return tuple(w[eid])
    worker=ThreeBankExpertPrefetch(128,128,bits,3,load,lambda:None,'cuda:0')
    dst=tuple({k:torch.empty(sh,dtype=dt,device='cuda') for k,(sh,dt) in packed_specs('weight',shape,bits).items()}
              for shape in ((256,128),(128,128)))
    try:
        worker.submit(0,[0,1,2],region=10,depth=0)
        assert entered.wait(5)
        # First expert is consumable while the rest of the region is unfinished.
        assert worker.try_copy(0,0,dst)
        original_ready=worker.banks[0]['keys'][(0,0)]['ready']
        worker.submit(0,[0,1,3],region=10,depth=1)
        worker.submit(0,[0,3,4],region=10,depth=2)
        release.set()
        worker.drain()
        assert loaded==[0,1,3,4]  # 2 cancelled before packing; 1 before DMA
        assert not worker.try_copy(0,1,dst)
        assert worker.banks[0]['keys'][(0,0)]['ready'] is original_ready
        assert worker.stats()['obsolete_after_pack']==1
        assert worker.stats()['retained_on_refinement']==2
        for eid in (0,3,4):
            assert worker.try_copy(0,eid,dst)
            expected=(pack_weight(torch.cat((w[eid,0],w[eid,1])),bits),pack_weight(w[eid,2],bits))
            assert all(torch.equal(d[k].cpu(),e[k]) for d,e in zip(dst,expected) for k in d)
        for region,eid in ((11,5),(12,6)):
            worker.submit(0,[eid],region=region,depth=0)
            worker.drain()
        assert all(x>0 for x in worker.stats()['bank_rotations'])
        # The three consecutive regions coexist, including the oldest common part.
        assert all(worker.try_copy(0,eid,dst) for eid in (0,5,6))
    finally:
        release.set()
        worker.close()


def test_busy_three_bank_ring_keeps_latest_refinement_until_real_reader_finishes():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    w=torch.randn(5,3,128,128,dtype=torch.bfloat16)
    loaded=[]
    def load(layer,eid):
        loaded.append(eid)
        return tuple(w[eid])
    worker=ThreeBankExpertPrefetch(128,128,4,1,load,lambda:None,'cuda:0')
    def destination():
        return tuple({k:torch.empty(sh,dtype=dt,device='cuda') for k,(sh,dt) in packed_specs('weight',shape,4).items()}
                     for shape in ((256,128),(128,128)))
    dst=[destination() for _ in range(3)]
    consumer=torch.cuda.Stream()
    try:
        for eid in range(3):
            worker.submit(0,[eid],region=eid)
            worker.drain()
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(500_000_000)
            for eid in range(3):assert worker.try_copy(0,eid,dst[eid])
        worker.submit(0,[3],region=3,depth=0)
        with worker.cv:
            assert worker.cv.wait_for(lambda:worker.counters['bank_busy']>0,5)
        worker.submit(0,[4],region=3,depth=1)
        worker.drain()
        consumer.synchronize()
        assert loaded==[0,1,2,4]
        for eid in range(3):
            expected=pack_weight(torch.cat((w[eid,0],w[eid,1])),4)
            assert all(torch.equal(dst[eid][0][k].cpu(),v) for k,v in expected.items())
        assert worker.try_copy(0,4,dst[0])
    finally:
        consumer.synchronize()
        worker.close()


def test_failed_later_expert_keeps_completed_prefix_usable():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    w=torch.randn(3,128,128,dtype=torch.bfloat16)
    def load(layer,eid):
        if eid:raise OSError('second expert unavailable')
        return tuple(w)
    worker=ThreeBankExpertPrefetch(128,128,4,2,load,lambda:None,'cuda:0')
    try:
        worker.submit(0,[0,1],region=1)
        worker.drain()
        assert worker.stats()['errors']==1 and worker.stats()['published']==1
        assert worker.try_copy(0,0,({},{}))
        assert not worker.try_copy(0,1,({},{}))
    finally:worker.close()


@pytest.mark.parametrize('bits',[4,6])
def test_three_independent_maps_do_not_supersede_each_other(bits):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    w=torch.randn(3,3,3,128,128,dtype=torch.bfloat16)
    entered,release=threading.Event(),threading.Event()
    def load(layer,eid):
        if layer==0 and eid==0:
            entered.set()
            assert release.wait(5)
        return tuple(w[layer,eid])
    worker=ThreeBankExpertPrefetch(128,128,bits,2,load,lambda:None,'cuda:0')
    dst=tuple({k:torch.empty(sh,dtype=dt,device='cuda') for k,(sh,dt) in packed_specs('weight',shape,bits).items()}
              for shape in ((256,128),(128,128)))
    try:
        worker.submit(0,[0,1],region=0,lane=0,step=1)
        assert entered.wait(5)
        worker.submit(1,[0,1],region=1,lane=1,step=1)
        worker.submit(2,[0,1],region=2,lane=2,step=1)
        worker.submit(1,[0,2],region=1,lane=1,step=1)
        release.set()
        worker.drain()
        for layer,ids in ((0,[0,1]),(1,[0,2]),(2,[0,1])):
            for eid in ids:
                assert worker.try_copy(layer,eid,dst)
                expected=pack_weight(torch.cat((w[layer,eid,0],w[layer,eid,1])),bits)
                assert all(torch.equal(dst[0][k].cpu(),v) for k,v in expected.items())
        published=worker.stats()['published']
        # Reusing the same area does not retransmit its weights.
        for layer,ids in ((0,[0,1]),(1,[0,2]),(2,[0,1])):
            worker.submit(layer,ids,region=layer,lane=layer,step=2)
        worker.drain()
        assert worker.stats()['published']==published==6
        assert all(x>0 for x in worker.stats()['bank_consumptions'])
    finally:
        release.set()
        worker.close()


def test_chunked_prefill_including_one_token_tail_builds_one_initial_map(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from src.aircolibri_integrated import AirColibriModel
    from aircolibri_cli.model_source import open_model_source,PreadSafetensorsReader
    make_model(tmp_path,'olmoe')
    expert_bytes=int(3*128*128*(6/8+4/64))
    budget=8*6*128*128+32*1024**2+8*expert_bytes
    reader=PreadSafetensorsReader()
    try:
        with AirColibriModel(str(tmp_path),model_source=open_model_source(tmp_path),weight_reader=reader,
                            weight_format='q6',universal_quantization=True,vram_resident_layers=1,
                            num_slots=3,expert_cache_bytes=budget) as engine:
            warm=engine.enable_probability_warmup(region_experts=4,gpu_budget_bytes=2**20)
            logits,past=engine._forward_step(torch.tensor([[1,2,3]]),prefill=True,prefill_end=False)
            assert not warm.region_map.initial and warm.stats()['map_submissions']==0
            logits,past=engine._forward_step(torch.tensor([[4]]),past,prefill=True,prefill_end=True)
            assert len(warm.region_map.initial)==5
            assert all(sum(row)==4*2 for row in warm.region_map.initial.values())
            initial=warm.region_map.stats()['initial_sha256']
            engine._forward_step(torch.tensor([[5]]),past,prefill=False)
            assert warm.region_map.stats()['initial_sha256']==initial
            warm.prefetch.drain()
            assert warm.stats()['map_submissions']==3
    finally:reader.close()


def test_map_can_be_prepared_later_without_rebuilding_unchanged_area(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from src.aircolibri_integrated import AirColibriModel
    from aircolibri_cli.model_source import open_model_source,PreadSafetensorsReader
    make_model(tmp_path,'granitemoe')
    expert_bytes=int(3*128*128*(4/8+4/64))
    budget=8*6*128*128+32*1024**2+8*expert_bytes
    reader=PreadSafetensorsReader()
    try:
        with AirColibriModel(str(tmp_path),model_source=open_model_source(tmp_path),weight_reader=reader,
                            weight_format='q4',universal_quantization=True,vram_resident_layers=1,
                            num_slots=3,expert_cache_bytes=budget) as engine:
            warm=engine.enable_probability_warmup(region_experts=4,gpu_budget_bytes=2**20)
            _,past=engine._forward_step(torch.tensor([[1,2,3]]),prefetch_next=False,prefill=True)
            original=warm.region_map.stats()['initial_sha256']
            assert warm.stats()['map_submissions']==0
            _,past=engine._forward_step(torch.tensor([[4]]),past,prefetch_next=True,prefill=False)
            warm.prefetch.drain()
            assert warm.stats()['map_submissions']==3
            sent=warm.prefetch.stats()['published']
            _,past=engine._forward_step(torch.tensor([[5]]),past,prefetch_next=True,prefill=False)
            warm.prefetch.drain()
            assert warm.stats()['map_submissions']==3
            assert warm.prefetch.stats()['published']==sent
            assert warm.region_map.stats()['initial_sha256']==original
    finally:reader.close()
