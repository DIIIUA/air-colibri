"""Serial, fresh-process policy A/B with alternating order and durable status."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--steps",type=int,default=8)
    p.add_argument("--prefix",default="moe-bench")
    a=p.parse_args()
    if min(a.repeats,a.steps)<1 or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in a.prefix):
        p.error("Invalid repeat, step or prefix")
    out=ROOT/"data/integration-20261112"
    rows=[]
    snapshot={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/"src").glob('*.py')}
    for rep in range(1,a.repeats+1):
        for mode in (["off","native"] if rep%2 else ["native","off"]):
            # Abort on code drift so timings never mix implementations silently.
            if any(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=sha for f,sha in snapshot.items()):
                raise RuntimeError("Runtime source changed during A/B matrix")
            name=f"{a.prefix}-{mode}-{rep}"
            cmd=[sys.executable,"-X","faulthandler","-u",str(ROOT/"scripts/verify_moe_integration.py"),"--case",name,"--steps",str(a.steps)]
            if mode=="native": cmd.append("--native")
            started=time.perf_counter()
            with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                process=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            path=out/(name+'.json')
            data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            row=dict(case=name,returncode=process.returncode,status=data.get('status','NO_RESULT'),wall_seconds=time.perf_counter()-started)
            if process.returncode:
                row['status']='PROCESS_FAILED'
            rows.append(row)
            (out/(a.prefix+'-matrix.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
            if process.returncode or data.get('status')!='PASS':
                raise SystemExit(1)


if __name__=='__main__': main()
