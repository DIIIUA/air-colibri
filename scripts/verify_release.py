"""Serial CLI and real-model acceptance probes; preserve every result/log."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/integration-20261112"


def main():
    jobs=[
        ("doctor",["aircolibri.py","doctor"],0),
        ("plan-dense",["aircolibri.py","plan","--model","qwen9b"],0),
        ("plan-moe",["aircolibri.py","plan","--model","olmoe","--policy","apply"],0),
        ("reject-budget",["aircolibri.py","run","--ram-gib","0.1","--report",str(OUT/"cli-reject-budget.json")],1),
        ("reject-moe-ring",["aircolibri.py","plan","--model","olmoe","--pipeline","triple_ring_prefetch"],2),
        ("cli-q4-ring",["aircolibri.py","run","--weight-format","q4","--pipeline","triple_ring_prefetch","--policy","apply","--max-new-tokens","2","--report",str(OUT/"cli-q4-ring.json")],0),
        ("cli-moe",["aircolibri.py","run","--model","olmoe","--policy","apply","--max-new-tokens","2","--report",str(OUT/"cli-moe.json")],0),
        ("bf16-off",["scripts/verify_integration.py","--case","bf16-off","--mode","off","--weight-format","bf16","--cache-mib","1024"],0),
        ("bf16-native",["scripts/verify_integration.py","--case","bf16-native","--native","--weight-format","bf16","--cache-mib","1024"],0),
    ]
    prompt="Русский текст: привет, мир! English: explain a loop. Python: for i in range(3): print(i)"
    for mode,extra in [("off",[]),("native",["--native"]),("shadow",["--shadow"])]:
        name="moe-multitoken-"+mode
        jobs.append((name,["scripts/verify_moe_integration.py","--case",name,"--prompt",prompt,*extra],0))
    results=[]
    for name,args,expected in jobs:
        start=time.perf_counter()
        with (OUT/(name+"-release.log")).open("w",encoding="utf-8") as log:
            r=subprocess.run([sys.executable,"-X","faulthandler","-u",*args],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,timeout=600)
        row=dict(case=name,returncode=r.returncode,expected_returncode=expected,seconds=time.perf_counter()-start)
        result=OUT/(name+".json")
        if result.exists():
            row["status"]=json.loads(result.read_text(encoding="utf-8-sig")).get("status")
        row["accepted"]=r.returncode==expected and (expected!=0 or row.get("status","PASS")=="PASS")
        results.append(row)
        (OUT/"release-validation.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
        print(json.dumps(row),flush=True)
        if not row["accepted"]:
            raise SystemExit(1)


if __name__=="__main__": main()
