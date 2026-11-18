"""Run the agreed checkpoint list in isolated processes, retaining failed attempts.

This is a startup/decode census, not an answer-quality or throughput benchmark.
An optional engine root pins a previous source archive while development continues.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19: raise
            time.sleep(0.01)


def read(path):
    # Windows readers/atomic replacements can briefly hold incompatible handles.
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            if attempt == 19: raise
            time.sleep(0.01)


def worker(args):
    sys.path.insert(0, str(args.engine_root))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    folder = args.models / args.worker
    destination = args.output / (args.worker + '.json')
    report = dict(model=args.worker, status='running', stage='imports', engine_root=str(args.engine_root),
                  options=dict(prefill_chunk=args.prefill_chunk, chat_tokens=args.tokens,
                               continuation_tokens=args.continuation_tokens, prompt=args.prompt,
                               context_tokens=512, mtp=False, thought_buffer=False, dynamic_memory=False))
    started = time.perf_counter()
    engine = None
    fingerprint = hashlib.sha256()
    for directory in ('src', 'aircolibri_cli'):
        for path in sorted((args.engine_root / directory).rglob('*')):
            if path.suffix in ('.py', '.cu', '.h', '.c'):
                fingerprint.update(path.relative_to(args.engine_root).as_posix().encode())
                fingerprint.update(path.read_bytes().replace(b'\r\n', b'\n'))
    report['engine_lf_sha256'] = fingerprint.hexdigest()
    before = {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns)
              for p in folder.rglob('*') if p.is_file()}
    report['model_bytes'] = sum(s[0] for s in before.values())
    write(destination, report)
    try:
        import torch
        from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine, forward_step_all_logits
        torch.set_num_threads(6)
        report.update(stage='load', weight_format=args.weight_format)
        write(destination, report)
        engine = AirColibriUnifiedEngine(model_dir=folder, aircolibri_root=args.engine_root,
            weight_format=args.weight_format, enable_mtp=False, allow_download=False,
            max_context_tokens=512, prefill_chunk_tokens=args.prefill_chunk, verbose=False)
        report.update(stage='raw_forward', loaded=True, plan=engine.plan.as_dict())
        write(destination, report)
        ids = engine.tokenizer('Hello', return_tensors='pt').input_ids
        with torch.inference_mode():
            if engine.fast_qwen:
                logits, cache = forward_step_all_logits(engine.engine, ids,
                    past_key_values=None, prefetch_next=False, logits_to_keep=1)
            else:
                logits, cache = engine.engine._forward_step(ids, past_key_values=None, prefetch_next=False)
            report['raw_logits_finite'] = bool(torch.isfinite(logits).all())
            if not report['raw_logits_finite']:
                raise ValueError('Nonfinite raw logits')
            del logits, cache
        report['stage'] = 'chat_decode'
        write(destination, report)
        messages = [{'role': 'user', 'content': args.prompt}]
        result = None
        for event in engine.stream_chat(messages, max_new_tokens=args.tokens,
                enable_thinking=False, use_session_cache=True, stop_on_eos=False):
            if event['type'] == 'done':
                result = event
        if result is None:
            raise RuntimeError('No completion event')
        report['answer'] = result['answer_text']
        report['metrics'] = result['metrics']
        report['stage'] = 'session_continuation'
        write(destination, report)
        messages += [{'role': 'assistant', 'content': result['answer_text']},
                     {'role': 'user', 'content': 'Continue.'}]
        result = None
        for event in engine.stream_chat(messages, max_new_tokens=args.continuation_tokens,
                enable_thinking=False, use_session_cache=True, stop_on_eos=False):
            if event['type'] == 'done':
                result = event
        if result is None:
            raise RuntimeError('No continuation completion event')
        report.update(status='passed', stage='complete', continuation=result['answer_text'],
                      continuation_metrics=result['metrics'])
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                report.update(status='failed', cleanup_error=repr(exc))
        engine = None
        gc.collect()
        if 'torch' in locals() and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                report['cuda_allocated_after_close_mib'] = torch.cuda.memory_allocated()/2**20
            except Exception as exc:
                report['cuda_cleanup_error'] = repr(exc)
        after = {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in folder.rglob('*') if p.is_file()}
        report['source_files_unchanged'] = before == after
        report['elapsed_s'] = round(time.perf_counter() - started, 3)
        write(destination, report)
    return 0 if report['status'] == 'passed' else 1


def kill_worker(process):
    import psutil
    try:
        parent = psutil.Process(process.pid)
        for child in reversed(parent.children(recursive=True)):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait()


def worker_batch(args):
    names = read(args.worker_batch)
    for name in names:
        if args.pause_file is not None and args.pause_file.exists():
            write(args.output/'worker_state.json', dict(phase='paused'))
            while args.pause_file.exists():
                time.sleep(0.5)
        write(args.output/'worker_state.json', dict(phase='running', model=name, started=time.monotonic()))
        args.worker = name
        worker(args)
        result = read(args.output/(name+'.json'))
        print(name+': '+result['status'], flush=True)
        # Reset the process after a damaged CUDA context or surviving allocations.
        if result.get('cuda_cleanup_error') or result.get('cuda_allocated_after_close_mib',0) > 128:
            return 17
    write(args.output/'worker_state.json', dict(phase='complete'))
    return 0


def persistent_census(args, entries):
    """Amortize HDD import scanning while retaining per-model watchdog/reset."""
    names = [entry['model'] for entry in entries]
    def results():
        collected = []
        for name in names:
            path = args.output/(name+'.json')
            if path.exists():
                result = read(path)
                if result.get('status') != 'running':
                    collected.append(result)
        return collected
    while True:
        done = {r['model'] for r in results()}
        remaining = [name for name in names if name not in done]
        if not remaining:
            break
        plan = args.output/'worker_plan.json'
        write(plan, remaining)
        write(args.output/'worker_state.json', dict(phase='starting'))
        command = [sys.executable,str(Path(__file__).resolve()),'--worker-batch',str(plan),
            '--engine-root',str(args.engine_root),'--models',str(args.models),'--output',str(args.output),
            '--weight-format',args.weight_format,'--tokens',str(args.tokens),
            '--prefill-chunk',str(args.prefill_chunk),'--continuation-tokens',str(args.continuation_tokens),
            '--prompt',args.prompt]
        if args.pause_file:
            command += ['--pause-file',str(args.pause_file)]
        with (args.output/'batch_worker.log').open('a',encoding='utf-8') as log:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    state = read(args.output/'worker_state.json')
                    if state['phase'] == 'paused':
                        started = time.monotonic()
                    elif state['phase'] != 'complete' and time.monotonic()-state.get('started',started) > args.timeout:
                        kill_worker(process)
                        name = state.get('model',remaining[0])
                        path = args.output/(name+'.json')
                        report = read(path) if path.exists() else dict(model=name)
                        report.update(status='bounded_stop',stage=report.get('stage','imports'))
                        write(path,report)
                        break
                    now = results()
                    if len(now) != len(done):
                        done = {r['model'] for r in now}
                        write(args.output/'summary.json',[{k:r[k] for k in ('model','status','stage','elapsed_s','error') if k in r} for r in now])
                        print(f'{len(done)}/{len(names)} completed attempts',flush=True)
                    time.sleep(0.5)
            except BaseException:
                kill_worker(process)
                raise
        state = read(args.output/'worker_state.json')
        if state.get('phase') == 'running':
            path = args.output/(state['model']+'.json')
            report = read(path) if path.exists() else dict(model=state['model'])
            if report.get('status') in (None,'running'):
                report.update(status='process_error',exit_code=process.returncode)
                write(path,report)
    write(args.output/'summary.json',[{k:r[k] for k in ('model','status','stage','elapsed_s','error') if k in r} for r in results()])
    return 0


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-root', type=Path, default=root)
    parser.add_argument('--models', type=Path, default=root/'models')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, default=root/'benchmarks/model_matrix_20261116.json')
    parser.add_argument('--weight-format', default='q6', choices=('q4', 'q6', 'bf16', 'auto'))
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--prefill-chunk', type=int, default=32)
    parser.add_argument('--continuation-tokens', type=int, default=2)
    parser.add_argument('--prompt', default='Hello. Say briefly what you can help with.')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--pause-file', type=Path,
                        help='Pause between workers while this file exists; never interrupts a model')
    parser.add_argument('--worker')
    parser.add_argument('--worker-batch', type=Path)
    parser.add_argument('--reuse-process', action='store_true',
                        help='Reuse imports across checkpoints; resume completed reports and reset on CUDA failure')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    if args.worker_batch:
        return worker_batch(args)
    import psutil
    entries = json.loads(args.matrix.read_text(encoding='utf-8'))['models']
    if args.reuse_process:
        return persistent_census(args,entries)
    summary = []
    for entry in entries:
        while args.pause_file is not None and args.pause_file.exists():
            time.sleep(0.25)
        name = entry['model']
        report_path = args.output/(name+'.json')
        if not (args.models/name).is_dir():
            report = dict(model=name, status='not_present', historical_short_cycle=entry['short_cycle_completed'])
            write(report_path, report)
        else:
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', name,
                '--engine-root', str(args.engine_root), '--models', str(args.models),
                '--output', str(args.output), '--weight-format', args.weight_format, '--tokens', str(args.tokens),
                '--prefill-chunk',str(args.prefill_chunk),'--continuation-tokens',str(args.continuation_tokens),
                '--prompt',args.prompt]
            with (args.output/(name+'.log')).open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=args.timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    parent = psutil.Process(process.pid)
                    for child in reversed(parent.children(recursive=True)):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    process.kill()
                    process.wait()
                    code = -1
                report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {'model': name}
                if report.get('status') == 'running' or 'status' not in report:
                    report.update(status='bounded_stop' if code == -1 else 'process_error', exit_code=code)
                    write(report_path, report)
        summary.append({k: report[k] for k in ('model', 'status', 'stage', 'elapsed_s', 'error') if k in report})
        write(args.output/'summary.json', summary)
        print(f"{len(summary)}/{len(entries)} {name}: {report['status']}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
