"""Paired short-answer checks; a passing probe is not a reasoning certification."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')

CASES = [
    ('japan','Name the capital of Japan. Answer with one word.','tokyo'),
    ('france','Name the capital of France. Answer with one word.','paris'),
    ('germany','Name the capital of Germany. Answer with one word.','berlin'),
    ('addition','What is 17 + 25? Answer with only the number.','42'),
    ('multiplication','What is 7 times 8? Answer with only the number.','56'),
    ('comparison','Which is larger, 7 or 9? Answer with only the larger number.','9'),
    ('extract','The box is green. What color is the box? Answer with one word.','green'),
    ('correction','First I said the box was red. Correction: the box is blue. What color is it now? Answer with one word.','blue'),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',default=['01','05','12','28'])
    parser.add_argument('--weight-format',default='q4',choices=('q4','q6'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine
    from aircolibri_cli.thought_buffer import InitialProbabilityAnchor
    torch.set_num_threads(6)
    digest=hashlib.sha256()
    for directory in ('aircolibri_cli','src'):
        for file in sorted((ROOT/directory).rglob('*')):
            if file.suffix in ('.py','.cu','.h','.c'):
                digest.update(file.relative_to(ROOT).as_posix().encode())
                digest.update(file.read_bytes().replace(b'\r\n',b'\n'))
    report=dict(scope='Greedy paired short answers, exact normalized answer and substring checks reported separately.',
                engine_lf_sha256=digest.hexdigest(),cases=[],conversations=[])
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    def generate(engine,messages,anchor,session=False):
        engine._active_thought_anchor=anchor
        done=None
        t0=time.perf_counter()
        for event in engine.stream_chat(messages,max_new_tokens=24,enable_thinking=False,use_session_cache=session):
            if event['type']=='done':done=event
        if done is None:raise RuntimeError('Missing completion')
        result=dict(answer=done['answer_text'],metrics=done['metrics'],wall_s=time.perf_counter()-t0)
        if anchor is not None:
            result.update(anchor=anchor.telemetry.summary(),initial_phi_sha256=anchor.initial_fingerprint(),
                          initial_probabilities_sha256=anchor.initial_probability_fingerprint())
        engine._active_thought_anchor=None
        return result
    for prefix in args.models:
        matches=list((ROOT/'models').glob(prefix+'_*'))
        if len(matches)!=1:raise ValueError(f'Ambiguous/missing model prefix: {prefix}')
        model=matches[0]
        with AirColibriUnifiedEngine(model_dir=model,weight_format=args.weight_format,enable_mtp=False,
                verbose=False,max_context_tokens=2048,prefill_chunk_tokens=64) as engine:
            for name,prompt,expected in CASES:
                row=dict(model=model.name,case=name,prompt=prompt,expected=expected)
                for mode in ('plain','anchored'):
                    engine.reset_session()
                    anchor=InitialProbabilityAnchor() if mode=='anchored' else None
                    result=generate(engine,[{'role':'user','content':prompt}],anchor)
                    normalized=result['answer'].strip().strip('.!').lower()
                    result.update(exact=normalized==expected,contains_expected=expected in normalized)
                    row[mode]=result
                report['cases'].append(row);save()
                print(model.name,name,row['plain']['answer'],'=>',row['anchored']['answer'],flush=True)
            for mode in ('plain','anchored'):
                engine.reset_session()
                anchor=InitialProbabilityAnchor() if mode=='anchored' else None
                messages=[{'role':'user','content':'Remember my marker PX-481. Reply only OK.'}]
                first=generate(engine,messages,anchor,True)
                messages += [{'role':'assistant','content':first['answer']},
                             {'role':'user','content':'Correction: use marker ZE-92 instead. What is my marker? Reply only with the marker.'}]
                second=generate(engine,messages,anchor,True)
                row=dict(model=model.name,mode=mode,first=first,second=second,
                         corrected_marker='ZE-92' in second['answer'])
                if anchor is not None:
                    row['original_probabilities_retained']=first['initial_probabilities_sha256']==second['initial_probabilities_sha256']
                    row['original_prior_retained']=first['initial_phi_sha256']==second['initial_phi_sha256']
                    assert row['original_probabilities_retained'] and row['original_prior_retained']
                report['conversations'].append(row);save()
        gc.collect();torch.cuda.empty_cache()
    report['regressions']=[dict(model=r['model'],case=r['case']) for r in report['cases']
                          if r['plain']['contains_expected'] and not r['anchored']['contains_expected']]
    report['exact_regressions']=[dict(model=r['model'],case=r['case']) for r in report['cases']
                               if r['plain']['exact'] and not r['anchored']['exact']]
    report['completed']=True;save()


if __name__=='__main__':main()
