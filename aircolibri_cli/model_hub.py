"""Managed downloads keep their original files and a small local source receipt."""
import hashlib
import json
from pathlib import Path

from .model_reference import parse_hf_reference
from .model_source import hf_snapshot, ModelNotFoundError


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def register_model(root, folder):
    registry = Path(root)/"data"/"model-paths.json"
    paths = json.loads(registry.read_text("utf-8")) if registry.is_file() else []
    if not isinstance(paths, list) or not all(isinstance(p,str) for p in paths):
        raise ModelNotFoundError("Invalid local model-path registry")
    paths = sorted(set(paths+[str(Path(folder).resolve())]))
    save_json(registry, paths)


def download_model(reference, root, *, directory=None, filename=None, revision=None):
    ref = parse_hf_reference(reference, filename=filename, revision=revision)
    name = ref.repo_id.replace("/", "__")[:80]
    if ref.filename:
        name += "--"+hashlib.sha256(ref.filename.encode()).hexdigest()[:8]
    requested_revision = ref.revision or "main"
    if requested_revision != "main":
        name += "--"+hashlib.sha256(requested_revision.encode()).hexdigest()[:8]
    target = Path(directory).expanduser().resolve() if directory else Path(root)/"models"/name
    marker = target/"aircolibri-source.json"
    identity = dict(repo_id=ref.repo_id, filename=ref.filename)
    if marker.is_file():
        old = json.loads(marker.read_text("utf-8"))
        if any(old.get(k) != v for k,v in identity.items()):
            raise ModelNotFoundError("The destination belongs to a different model/quantization; choose another directory")
        if (old.get("requested_revision") or "main") != requested_revision:
            raise ModelNotFoundError("The destination belongs to another revision; choose a new directory")
    elif target.exists() and any(target.iterdir()):
        raise ModelNotFoundError("The download destination is not empty. Choose an empty directory or load its existing model with --model")
    target.mkdir(parents=True, exist_ok=True)
    receipt = dict(**identity, requested_revision=requested_revision, status="downloading", runtime_subdir=".",
                   adaptation="on_demand", preparation=None)
    save_json(marker, receipt)
    try:
        model = hf_snapshot(ref.repo_id, True, local_dir=target, filename=ref.filename, revision=ref.revision)
        receipt.update(status="downloaded", runtime_subdir=model.resolve().relative_to(target.resolve()).as_posix())
        save_json(marker, receipt)
        register_model(root, target)
        return model.resolve(), marker, receipt
    except Exception as exc:
        receipt.update(status="download_failed", error_type=type(exc).__name__)
        save_json(marker, receipt)
        # Network failures should be concise at the CLI; keep completed files
        # for the Hub's normal resumable download path.
        if isinstance(exc, (OSError, ValueError)):
            raise
        raise ModelNotFoundError(f"Hugging Face download failed ({type(exc).__name__}); completed files were retained for retry") from exc
