from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
repo = "allenai/OLMoE-1B-7B-0125-Instruct"
revision = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"
dest = Path(r"C:\air-colibri\models\OLMoE-1B-7B-0125-Instruct")
names = ["README.md","config.json","generation_config.json","model-00001-of-00003.safetensors","model-00002-of-00003.safetensors","model-00003-of-00003.safetensors","model.safetensors.index.json","special_tokens_map.json","tokenizer.json","tokenizer_config.json"]
print("START",repo,"DEST",dest,flush=True)
snapshot_download(repo_id=repo,revision=revision,local_dir=str(dest),allow_patterns=names,max_workers=3,token=False)
print("TRANSFER_FINISHED",flush=True)
info=HfApi().model_info(repo,revision=revision,files_metadata=True,token=False)
sizes={f.rfilename:f.size for f in info.siblings}
failed=[]
for name in names:
    p=dest/name
    size=p.stat().st_size if p.is_file() else -1
    print("VERIFY",name,size,"expected",sizes.get(name),flush=True)
    if size<0 or (sizes.get(name) is not None and size!=sizes.get(name)):failed.append(name)
print("VERIFIED" if not failed else "FAILED "+str(failed),flush=True)
if failed:raise SystemExit(1)