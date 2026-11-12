"""Fresh-process checks after failure-path hardening and bounded diagnostics."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/integration-20261112"
jobs=[
    ("tiered-isolated",["-c","import torch,pytest; torch.set_num_threads(4); raise SystemExit(pytest.main(['tests/test_tiered_cache.py','-q','--junitxml=data/integration-20261112/tiered-isolated.xml']))"]),
    ("final-contract",["-c","import torch,pytest; torch.set_num_threads(4); raise SystemExit(pytest.main(['tests/test_integration_contract.py','tests/test_olmoe_integration.py','tests/test_resource_plan.py','tests/test_gpu_ring_pipeline.py','tests/test_memory_pool.py','-q','--junitxml=data/integration-20261112/final-contract.xml']))"]),
    ("cli-final-q4",["aircolibri.py","run","--weight-format","q4","--pipeline","triple_ring_prefetch","--policy","apply","--max-new-tokens","2","--report",str(OUT/"cli-final-q4.json")]),
    ("cli-final-moe",["aircolibri.py","run","--model","olmoe","--policy","apply","--max-new-tokens","2","--report",str(OUT/"cli-final-moe.json")]),
]
rows=[]
for name,args in jobs:
    with (OUT/(name+".log")).open("w",encoding="utf-8") as log:
        r=subprocess.run([sys.executable,"-X","faulthandler","-u",*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    row=dict(case=name,returncode=r.returncode)
    rows.append(row)
    (OUT/"final-checks.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print(json.dumps(row),flush=True)
    if r.returncode: raise SystemExit(r.returncode)
