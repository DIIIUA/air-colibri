"""Model discovery and weight sources for AirColibri.

Two on-disk formats are supported:

* ``colibri`` — the per-layer container (``config.hf.json``, ``model-00000.safetensors`` …
  ``model-globals.safetensors``, optional ``model-mtp.safetensors``).
* ``hf`` — a standard Hugging Face download (``config.json`` + ``model.safetensors.index.json``
  + ``model-00001-of-0000N.safetensors``, or a single ``model.safetensors``). Shards are read
  directly; no converter and no second copy of the weights on disk are needed.

The module also provides ``PreadSafetensorsReader``: a dependency-free reader that fetches
tensor byte ranges with positioned reads instead of ``mmap`` (Windows ``mmap`` inflates the
process working set and was the source of ``0xc0000005`` crashes).
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import threading
from typing import Any, Iterable

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
SUPPORTED_MODEL_TYPES = ("qwen3_5_text", "qwen3_5")

_LAYER_RE = re.compile(r"^(?:model\.language_model\.|model\.)layers\.(\d+)\.(.+)$")
_GLOBAL_ALIASES = {
    "model.embed_tokens.weight": ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"),
    "model.norm.weight": ("model.language_model.norm.weight", "model.norm.weight"),
    "lm_head.weight": ("lm_head.weight", "model.language_model.lm_head.weight"),
}


class ModelNotFoundError(FileNotFoundError):
    pass


class UnsupportedModelError(ValueError):
    pass


# --------------------------------------------------------------------------------------
# Format detection and configuration
# --------------------------------------------------------------------------------------


def detect_format(model_dir: str | Path) -> str | None:
    """Returns ``"colibri"``, ``"hf"`` or ``None`` for an unrecognised folder."""
    d = Path(model_dir)
    if (d / "config.hf.json").is_file() and (d / "model-00000.safetensors").is_file():
        return "colibri"
    if (d / "config.json").is_file() and (
        (d / "model.safetensors.index.json").is_file() or (d / "model.safetensors").is_file()
    ):
        return "hf"
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_text_config_dict(model_dir: str | Path) -> dict[str, Any]:
    """Returns the text-model config dict (``text_config`` for multimodal Qwen3.5 checkpoints)."""
    d = Path(model_dir)
    cfg_path = d / "config.hf.json" if (d / "config.hf.json").is_file() else d / "config.json"
    cfg = _read_json(cfg_path)
    if cfg.get("model_type") == "qwen3_5" and isinstance(cfg.get("text_config"), dict):
        return cfg["text_config"]
    return cfg


def check_supported(model_dir: str | Path) -> None:
    """Raises ``UnsupportedModelError`` unless the folder holds a Qwen3.5 hybrid (GatedDeltaNet) model."""
    fmt = detect_format(model_dir)
    if fmt is None:
        raise UnsupportedModelError(
            f"{model_dir}: no Colibri container or Hugging Face safetensors checkpoint found"
        )
    try:
        cfg = raw_text_config_dict(model_dir)
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedModelError(f"{model_dir}: unreadable config ({exc})") from exc
    model_type = str(cfg.get("model_type", ""))
    layer_types = cfg.get("layer_types") or []
    if model_type not in SUPPORTED_MODEL_TYPES and model_type != "qwen3_5_text":
        raise UnsupportedModelError(
            f"{model_dir}: model_type={model_type!r} is not supported (expected a Qwen3.5 hybrid model)"
        )
    if "linear_attention" not in layer_types:
        raise UnsupportedModelError(f"{model_dir}: config has no GatedDeltaNet (linear_attention) layers")


def is_supported(model_dir: str | Path) -> bool:
    try:
        check_supported(model_dir)
        return True
    except UnsupportedModelError:
        return False


def load_text_config(model_dir: str | Path):
    """Loads a ``qwen3_5_text`` transformers config for either format."""
    from transformers import AutoConfig

    d = Path(model_dir)
    if detect_format(d) == "colibri":
        return AutoConfig.from_pretrained(str(d), config_file="config.hf.json")
    raw = raw_text_config_dict(d)
    if raw.get("model_type") in ("qwen3_5", "qwen3_5_text"):
        # Multimodal checkpoints nest the text model under `text_config`; build the plain text config
        # exactly as a Colibri container's `config.hf.json` would be loaded.
        kwargs = {k: v for k, v in raw.items() if k not in ("model_type", "transformers_version")}
        cfg = AutoConfig.for_model("qwen3_5_text", **kwargs)
    else:
        cfg = AutoConfig.from_pretrained(str(d))
    if getattr(cfg, "model_type", "") != "qwen3_5_text":
        raise UnsupportedModelError(f"{model_dir}: expected qwen3_5_text config, got {cfg.model_type!r}")
    return cfg


# --------------------------------------------------------------------------------------
# Model folder resolution
# --------------------------------------------------------------------------------------

_PREFERRED_NAMES = ("Qwen3.5-9B", "qwen3.5-9b", "deepseek-v4-9b-colibri", "deepseek-v4-9b")


def find_local_models(root: str | Path) -> list[Path]:
    """Lists supported model folders under ``<root>/models`` (unsupported ones such as OLMoE are skipped)."""
    models_dir = Path(root) / "models"
    if not models_dir.is_dir():
        return []
    found = [p for p in sorted(models_dir.iterdir()) if p.is_dir() and is_supported(p)]

    def _rank(p: Path) -> tuple[int, str]:
        name = p.name.lower()
        for i, pref in enumerate(_PREFERRED_NAMES):
            if pref.lower() in name:
                return (i, name)
        return (len(_PREFERRED_NAMES), name)

    return sorted(found, key=_rank)


def _looks_like_repo_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][\w.\-]*/[\w.\-]+", value)) and not Path(value).exists()


def hf_snapshot(repo_id: str, allow_download: bool) -> Path:
    """Returns a local snapshot of ``repo_id`` (downloads it only when ``allow_download``)."""
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "tokenizer*"]
    try:
        return Path(snapshot_download(repo_id, allow_patterns=patterns, local_files_only=True))
    except Exception:
        if not allow_download:
            raise
    print(f"[AirColibri] Downloading {repo_id} from Hugging Face (one-time, ~19 GB for a 9B model)...", flush=True)
    return Path(snapshot_download(repo_id, allow_patterns=patterns))


def resolve_model_dir(model: str | Path | None, root: str | Path, allow_download: bool = False) -> Path:
    """Resolves ``--model`` (folder or Hugging Face repo id) or auto-detects a supported local model.

    Order: explicit value → ``AIRCOLIBRI_MODEL`` → supported folders in ``<root>/models`` (OLMoE and
    other unsupported checkpoints are never picked) → a cached ``Qwen/Qwen3.5-9B`` snapshot.
    """
    value = str(model) if model else os.environ.get("AIRCOLIBRI_MODEL")
    if value:
        p = Path(value).expanduser()
        if p.exists():
            check_supported(p)
            return p.resolve()
        if _looks_like_repo_id(value):
            snap = hf_snapshot(value, allow_download=allow_download)
            check_supported(snap)
            return snap
        raise ModelNotFoundError(f"Model folder not found: {value}")

    local = find_local_models(root)
    if local:
        return local[0].resolve()
    try:
        snap = hf_snapshot(DEFAULT_MODEL_ID, allow_download=False)
        check_supported(snap)
        return snap
    except Exception:
        pass
    raise ModelNotFoundError(
        f"No supported model found in {Path(root) / 'models'}. Pass `--model <folder>` or "
        f"`--model {DEFAULT_MODEL_ID}` to download the recommended model from Hugging Face."
    )


# --------------------------------------------------------------------------------------
# Positioned-read safetensors access (no mmap, no external checkout)
# --------------------------------------------------------------------------------------

_ST_DTYPES = {
    "BF16": ("bfloat16", 2),
    "F16": ("float16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "I64": ("int64", 8),
    "I32": ("int32", 4),
    "I16": ("int16", 2),
    "I8": ("int8", 1),
    "U8": ("uint8", 1),
    "BOOL": ("bool", 1),
}


def read_safetensors_header(path: str | Path) -> tuple[dict[str, Any], int]:
    """Returns ``(header, data_offset)`` of a safetensors file."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        (n,) = struct.unpack("<Q", head)
        if n <= 0 or n > 256 * 1024 * 1024:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        header = json.loads(f.read(n).decode("utf-8"))
    return header, 8 + n


class PreadSafetensorsReader:
    """Reads named tensors with positioned file reads; keeps a small LRU of open file handles."""

    def __init__(self, max_handles: int = 8):
        self.max_handles = max(1, int(max_handles))
        self._handles: "OrderedDict[str, Any]" = OrderedDict()
        self._headers: dict[str, tuple[dict[str, Any], int]] = {}
        self._lock = threading.Lock()

    def _header(self, path: str) -> tuple[dict[str, Any], int]:
        hdr = self._headers.get(path)
        if hdr is None:
            hdr = read_safetensors_header(path)
            self._headers[path] = hdr
        return hdr

    def _file(self, path: str):
        f = self._handles.get(path)
        if f is None:
            f = open(path, "rb", buffering=0)
            self._handles[path] = f
            while len(self._handles) > self.max_handles:
                _, old = self._handles.popitem(last=False)
                old.close()
        else:
            self._handles.move_to_end(path)
        return f

    def read_tensor(self, path: str | Path, key: str):
        import torch

        path = str(path)
        with self._lock:
            header, base = self._header(path)
            meta = header.get(key)
            if meta is None:
                raise KeyError(f"{key} not in {path}")
            dtype_name, _ = _ST_DTYPES[meta["dtype"]]
            start, end = meta["data_offsets"]
            buf = bytearray(end - start)
            f = self._file(path)
            if hasattr(os, "pread"):
                view = memoryview(buf)
                done = 0
                while done < len(buf):
                    chunk = os.pread(f.fileno(), len(buf) - done, base + start + done)
                    if not chunk:
                        raise OSError(f"{path}: unexpected EOF while reading {key}")
                    view[done : done + len(chunk)] = chunk
                    done += len(chunk)
            else:
                f.seek(base + start)
                got = f.readinto(buf)
                if got != len(buf):
                    raise OSError(f"{path}: unexpected EOF while reading {key}")
        t = torch.frombuffer(buf, dtype=getattr(torch, dtype_name)) if buf else torch.empty(0, dtype=getattr(torch, dtype_name))
        return t.reshape(meta["shape"])

    def read_entries(self, entries: dict[str, tuple[str, str]]) -> dict[str, Any]:
        """Reads ``{name: (file, stored_key)}`` into ``{name: tensor}``."""
        return {name: self.read_tensor(path, key) for name, (path, key) in entries.items()}

    def __call__(self, model_dir: str, stem: str, names: Iterable[str]) -> dict[str, Any]:
        """``weight_reader`` protocol used by the engine: reads ``<model_dir>/<stem>.safetensors``."""
        path = str(Path(model_dir) / f"{stem}.safetensors")
        return {k: self.read_tensor(path, k) for k in names}

    def close(self) -> None:
        with self._lock:
            for f in self._handles.values():
                try:
                    f.close()
                except OSError:
                    pass
            self._handles.clear()
            self._headers.clear()


# --------------------------------------------------------------------------------------
# Hugging Face sharded checkpoints
# --------------------------------------------------------------------------------------


class HFShardedSource:
    """Maps a standard Hugging Face checkpoint onto the engine's canonical tensor names.

    Canonical names: ``model.layers.{i}.<param>``, ``model.embed_tokens.weight``,
    ``model.norm.weight``, ``lm_head.weight`` and ``mtp.*``.
    """

    kind = "hf"

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).resolve()
        if detect_format(self.model_dir) != "hf":
            raise UnsupportedModelError(f"{self.model_dir}: not a Hugging Face safetensors checkpoint")
        index = self.model_dir / "model.safetensors.index.json"
        if index.is_file():
            weight_map = _read_json(index)["weight_map"]
        else:
            header, _ = read_safetensors_header(self.model_dir / "model.safetensors")
            weight_map = {k: "model.safetensors" for k in header if k != "__metadata__"}
        # Separate MTP files shipped next to the shards (e.g. ``model-mtp.safetensors``).
        for extra in sorted(self.model_dir.glob("*mtp*.safetensors")):
            header, _ = read_safetensors_header(extra)
            for k in header:
                if k != "__metadata__" and k not in weight_map:
                    weight_map[k] = extra.name

        self._stored: dict[str, tuple[str, str]] = {}
        self._layers: dict[int, dict[str, tuple[str, str]]] = {}
        self._mtp: dict[str, tuple[str, str]] = {}
        for stored_key, filename in weight_map.items():
            entry = (str(self.model_dir / filename), stored_key)
            m = _LAYER_RE.match(stored_key)
            if m and not stored_key.startswith("mtp."):
                idx = int(m.group(1))
                self._layers.setdefault(idx, {})[f"model.layers.{idx}.{m.group(2)}"] = entry
                continue
            if stored_key.startswith("mtp."):
                self._mtp[stored_key] = entry
                continue
            self._stored[stored_key] = entry

        self._globals: dict[str, tuple[str, str]] = {}
        for canonical, aliases in _GLOBAL_ALIASES.items():
            for alias in aliases:
                if alias in self._stored:
                    self._globals[canonical] = self._stored[alias]
                    break
        cfg = raw_text_config_dict(self.model_dir)
        if "lm_head.weight" not in self._globals and "model.embed_tokens.weight" in self._globals:
            if cfg.get("tie_word_embeddings", False):
                self._globals["lm_head.weight"] = self._globals["model.embed_tokens.weight"]
        missing = [k for k in _GLOBAL_ALIASES if k not in self._globals]
        if missing:
            raise UnsupportedModelError(f"{self.model_dir}: checkpoint misses global tensors {missing}")
        self.num_layers = int(cfg.get("num_hidden_layers", len(self._layers)))
        absent = [i for i in range(self.num_layers) if i not in self._layers]
        if absent:
            raise UnsupportedModelError(f"{self.model_dir}: checkpoint misses decoder layers {absent[:8]}")
        self._header_cache: dict[str, dict[str, Any]] = {}

    # -- tensor lookup ------------------------------------------------------------------

    def layer_entries(self, layer_idx: int) -> dict[str, tuple[str, str]]:
        return dict(self._layers[int(layer_idx)])

    def globals_entries(self) -> dict[str, tuple[str, str]]:
        return dict(self._globals)

    def mtp_entries(self) -> dict[str, tuple[str, str]]:
        return dict(self._mtp)

    def has_mtp(self) -> bool:
        return "mtp.fc.weight" in self._mtp

    def _tensor_meta(self, path: str, key: str) -> dict[str, Any]:
        hdr = self._header_cache.get(path)
        if hdr is None:
            hdr, _ = read_safetensors_header(path)
            self._header_cache[path] = hdr
        return hdr[key]

    def tensor_shape(self, entry: tuple[str, str]) -> tuple[int, ...]:
        return tuple(self._tensor_meta(*entry)["shape"])

    def layer_identity(self, layer_idx: int) -> dict[str, Any]:
        """Cache identity of one decoder layer: tensor names, dtypes, shapes, byte ranges and file stamps.

        Cheap to compute (headers only) and changes whenever a shard is replaced or edited.
        """
        tensors: dict[str, Any] = {}
        files: dict[str, Any] = {}
        for name, (path, key) in sorted(self.layer_entries(layer_idx).items()):
            meta = self._tensor_meta(path, key)
            tensors[name] = {"shape": list(meta["shape"]), "dtype": meta["dtype"],
                             "file": os.path.basename(path), "offsets": list(meta["data_offsets"])}
            if path not in files:
                st = os.stat(path)
                files[os.path.basename(path)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        digest = hashlib.sha256(json.dumps({"tensors": tensors, "files": files}, sort_keys=True).encode()).hexdigest()
        return {"sha256": digest, "tensors": {k: {"shape": v["shape"]} for k, v in tensors.items()}}


def open_model_source(model_dir: str | Path):
    """Returns an ``HFShardedSource`` for Hugging Face checkpoints, ``None`` for Colibri containers."""
    fmt = detect_format(model_dir)
    if fmt == "hf":
        return HFShardedSource(model_dir)
    if fmt == "colibri":
        return None
    raise UnsupportedModelError(f"{model_dir}: unrecognised model folder")
