"""Normalize public Hugging Face references without importing an ML runtime."""
from dataclasses import dataclass
import re
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class HFReference:
    repo_id: str
    filename: str | None = None
    revision: str | None = None


def parse_hf_reference(value, *, filename=None, revision=None):
    value = str(value).strip()
    selected = None
    url_revision = None
    if "://" in value:
        url = urlsplit(value)
        if url.scheme not in ("http", "https") or url.hostname not in ("huggingface.co", "www.huggingface.co"):
            raise ValueError("Use a Hugging Face model URL or owner/repository ID")
        if url.username or url.password:
            raise ValueError("Use a public Hugging Face URL without embedded credentials")
        parts = [unquote(p) for p in url.path.strip("/").split("/")]
        if len(parts) < 2:
            raise ValueError("A Hugging Face model URL needs owner/repository")
        repo = "/".join(parts[:2])
        if len(parts) > 2:
            if len(parts) < 4 or parts[2] not in ("tree", "blob", "resolve"):
                raise ValueError("Use a model page, revision page or model-file URL")
            url_revision = parts[3]
            if parts[2] in ("blob", "resolve"):
                leaf = "/".join(parts[4:])
                # HF shard links identify the complete HF checkpoint. A GGUF
                # file link selects one quantization instead of every variant.
                selected = leaf if leaf.lower().endswith(".gguf") else None
            elif len(parts) > 4:
                raise ValueError("Use the model revision page; select a GGUF with --file")
    else:
        repo, separator, selected = value.partition("::")
        selected = selected if separator else None
    if not re.fullmatch(r"[A-Za-z0-9][\w.\-]*/[\w.\-]+", repo):
        raise ValueError("Expected a Hugging Face owner/repository ID")
    if filename and selected and filename != selected:
        raise ValueError("The URL and --file select different files")
    return HFReference(repo, filename or selected, revision or url_revision)
