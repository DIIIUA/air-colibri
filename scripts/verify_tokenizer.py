"""Check the adapter preserves checkpoint tokenizer IDs, including Unicode/code."""
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"models/deepseek-v4-9b-colibri"
texts=["Hello", "The capital of France is", "Привет, мир! Ёжик съел яблоко.",
       "def hello(name: str):\n    return f'Hello, {name}!'\n", "1234567890 1.23e-4", "中文 日本語 한국어", "🙂\t a\n\n", "e\u0301 é ① — «кавычки»"]
raw=Tokenizer.from_file(str(MODEL/"tokenizer.json"))
adapter=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,fix_mistral_regex=False)
rows=[]
for text in texts:
    expected=raw.encode(text,add_special_tokens=False).ids
    actual=adapter.encode(text,add_special_tokens=False)
    assert actual==expected, text
    rows.append(dict(text=text,ids=actual,exact=True))
result=dict(status="PASS",scope="preserve supplied tokenizer.json, not certify external corrected tokenizer",
    tokenizer_sha256=hashlib.sha256((MODEL/"tokenizer.json").read_bytes()).hexdigest(),samples=rows)
(ROOT/"data/integration-20261112/tokenizer-preservation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(dict(status="PASS",samples=len(rows))))
