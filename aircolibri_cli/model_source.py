"""Model discovery and weight sources for AirColibri.

Three on-disk formats are supported:

* ``colibri`` — the per-layer container (``config.hf.json``, ``model-00000.safetensors`` …
  ``model-globals.safetensors``, optional ``model-mtp.safetensors``).
* ``hf`` — a standard Hugging Face download (``config.json`` + ``model.safetensors.index.json``
  + ``model-00001-of-0000N.safetensors``, or a single ``model.safetensors``). Shards are read
  directly; no converter and no second copy of the weights on disk are needed.
* ``gguf`` — a folder containing one ``.gguf`` checkpoint of a supported ``transformers`` GGUF
  architecture (such as ``qwen35moe`` / ``qwen35``). Dense trunk tensors are unpacked into the
  BF16 ring while routed MoE experts stay in the ``.gguf`` file and are read by positioned byte
  slices on demand (with an optional budgeted VRAM hot-expert slot arena), without ``mmap`` and
  without writing sidecar files into the model folder.

The module also provides ``PreadSafetensorsReader``: a dependency-free reader that fetches
tensor byte ranges with positioned reads instead of ``mmap`` (Windows ``mmap`` inflates the
process working set and was the source of ``0xc0000005`` crashes).
"""
from __future__ import annotations

import collections.abc
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import struct
import threading
import types
from typing import Any, Iterable

from aircolibri_cli.gguf_reader import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_F64,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_I64,
    GGML_TYPE_MXFP4,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_K,
    GGUFError,
    GGUFReader,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
SUPPORTED_MODEL_TYPES = ("qwen3_5_text", "qwen3_5")

# aircolibri_cli/kernels.py specializes the fused Q4 kernels to this geometry.
# A Qwen3.5 hybrid whose config differs stays on the BF16 ring instead of a wrong kernel.
FAST_KERNEL_FIELDS = {
    "hidden_size": 4096,
    "head_dim": 256,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}

_LAYER_RE = re.compile(r"^(?:model\.language_model\.|model\.)layers\.(\d+)\.(.+)$")
_GLOBAL_ALIASES = {
    "model.embed_tokens.weight": ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"),
    "model.norm.weight": (
        "model.language_model.norm.weight",
        "model.norm.weight",
        "model.embedding_norm.weight",
    ),
    "lm_head.weight": ("lm_head.weight", "model.language_model.lm_head.weight"),
}
_MISSING = object()
_MAX_EXPERT_DEQUANT_BATCH = 8
_IO_WORKERS = 4


class ModelNotFoundError(FileNotFoundError):
    pass


class UnsupportedModelError(ValueError):
    pass


# --------------------------------------------------------------------------------------
# Format detection and configuration
# --------------------------------------------------------------------------------------


def detect_format(model_dir: str | Path) -> str | None:
    """Returns ``"colibri"``, ``"hf"``, ``"gguf"`` or ``None`` for an unrecognised folder."""
    d = Path(model_dir)
    if (d / "config.hf.json").is_file() and (d / "model-00000.safetensors").is_file():
        return "colibri"
    if (d / "config.json").is_file() and (
        (d / "model.safetensors.index.json").is_file() or (d / "model.safetensors").is_file()
    ):
        return "hf"
    if any(p.is_file() for p in d.glob("*.gguf")):
        return "gguf"
    return None


def find_gguf_file(model_dir: str | Path) -> Path:
    """Returns the single ``.gguf`` file inside ``model_dir`` or raises ``UnsupportedModelError``."""
    d = Path(model_dir)
    found = sorted(p for p in d.glob("*.gguf") if p.is_file())
    if not found:
        raise UnsupportedModelError(f"{d}: no .gguf file found")
    if len(found) > 1:
        names = ", ".join(p.name for p in found[:4])
        raise UnsupportedModelError(f"{d}: expected a single .gguf file, found {len(found)} ({names})")
    return found[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_GGUF_HEADER_CACHE: "OrderedDict[tuple[str, int, int], tuple[dict[str, Any], list[dict[str, Any]]]]" = OrderedDict()
_GGUF_HEADER_LOCK = threading.Lock()


def _read_gguf_header_cached(gguf_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reads GGUF metadata and tensor descriptors with pure positioned I/O (zero mmap), cached by stat."""
    resolved = gguf_path.resolve()
    try:
        st = os.stat(resolved)
    except OSError as exc:
        raise UnsupportedModelError(f"{gguf_path}: {exc}") from exc
    key = (str(resolved), int(st.st_size), int(st.st_mtime_ns))
    with _GGUF_HEADER_LOCK:
        hit = _GGUF_HEADER_CACHE.get(key)
        if hit is not None:
            _GGUF_HEADER_CACHE.move_to_end(key)
            return hit

    try:
        with GGUFReader(str(resolved)) as reader:
            meta = dict(reader.metadata)
            raw_tensors = list(reader.tensors)
    except (GGUFError, OSError) as exc:
        raise UnsupportedModelError(f"{gguf_path}: {exc}") from exc

    with _GGUF_HEADER_LOCK:
        while len(_GGUF_HEADER_CACHE) >= 2:
            _GGUF_HEADER_CACHE.popitem(last=False)
        _GGUF_HEADER_CACHE[key] = (meta, raw_tensors)
    return meta, raw_tensors


def _validate_gguf_header(gguf_path: Path) -> str:
    """Validates GGUF binary header with pure positioned I/O (no mmap) and checks arch support."""
    from transformers.integrations.gguf.utils import is_gguf_arch_supported

    meta, _ = _read_gguf_header_cached(gguf_path)
    arch = str(meta.get("general.architecture") or "")
    if not arch:
        raise UnsupportedModelError(f"{gguf_path}: GGUF metadata misses 'general.architecture'")
    if not is_gguf_arch_supported(arch):
        raise UnsupportedModelError(f"{gguf_path}: GGUF architecture {arch!r} is not supported")
    return arch


def raw_text_config_dict(model_dir: str | Path) -> dict[str, Any]:
    """Returns the text-model config dict (``text_config`` for multimodal checkpoints)."""
    d = Path(model_dir)
    if detect_format(d) == "gguf":
        cfg_obj = load_runtime_config(d)
        text_cfg = cfg_obj.get_text_config() if hasattr(cfg_obj, "get_text_config") else cfg_obj
        return text_cfg.to_dict()
    cfg_path = d / "config.hf.json" if (d / "config.hf.json").is_file() else d / "config.json"
    cfg = _read_json(cfg_path)
    if isinstance(cfg.get("text_config"), dict):
        return cfg["text_config"]
    return cfg


def fast_kernel_mismatches(cfg: dict[str, Any]) -> list[str]:
    """Fields the fused Q4 kernels assume. Empty means the fast path is safe."""
    bad: list[str] = []
    for key, expect in FAST_KERNEL_FIELDS.items():
        got = cfg.get(key)
        if got is None or int(got) != int(expect):
            bad.append(f"{key}={got!r}, kernels expect {expect}")
    return bad


def runtime_profile(model_dir: str | Path) -> dict[str, Any]:
    """What this checkpoint can run, decided from its config rather than its folder name.

    Q4 and MTP turn on only when the model is a dense Qwen3.5 GatedDeltaNet hybrid whose
    shapes match the fused kernels. Everything else that still has a decoder stack —
    including GGUF checkpoints and MoE models — runs BF16 on the same ring, with MTP off.
    """
    fmt = detect_format(model_dir)
    cfg = raw_text_config_dict(model_dir)
    has_deltanet = "linear_attention" in (cfg.get("layer_types") or [])
    hybrid = is_qwen35_hybrid(model_dir)
    mismatches = fast_kernel_mismatches(cfg) if has_deltanet else []
    num_experts = int(cfg.get("num_experts") or 0)
    fast = bool(hybrid and not mismatches and num_experts == 0 and fmt != "gguf")

    why: list[str] = []
    if not has_deltanet:
        why.append("no Qwen3.5 GatedDeltaNet layers")
    why.extend(mismatches)
    if num_experts > 0:
        why.append(f"num_experts={num_experts}, dense MLP required")
    if fmt == "gguf" and not why:
        why.append("GGUF block weights run on the BF16 ring")

    out: dict[str, Any] = {
        "fast_qwen": fast,
        "backend": "qwen35-q4" if fast else "hf-bf16",
        "weight_format": "q4" if fast else "bf16",
        "model_type": str(cfg.get("model_type", "")),
        "num_hidden_layers": int(cfg.get("num_hidden_layers") or 0),
        "hidden_size": int(cfg.get("hidden_size") or 0),
        "why_not_fast": [] if fast else why,
    }
    if num_experts > 0:
        out["num_experts"] = num_experts
        out["num_experts_per_tok"] = int(cfg.get("num_experts_per_tok") or 0)
    return out


def is_qwen35_hybrid(model_dir: str | Path) -> bool:
    """True for the fast path: dense safetensors/Colibri Qwen3.5 text with GatedDeltaNet layers."""
    if detect_format(model_dir) == "gguf":
        return False
    try:
        cfg = raw_text_config_dict(model_dir)
    except (OSError, json.JSONDecodeError, UnsupportedModelError):
        return False
    if cfg.get("num_experts"):
        return False
    model_type = str(cfg.get("model_type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES and model_type != "qwen3_5_text":
        return False
    return "linear_attention" in (cfg.get("layer_types") or [])


def _require_qwen35_hybrid(model_dir: str | Path, cfg: dict[str, Any]) -> None:
    model_type = str(cfg.get("model_type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES and model_type != "qwen3_5_text":
        raise UnsupportedModelError(
            f"{model_dir}: model_type={model_type!r} is not a Qwen3.5 hybrid model"
        )
    if "linear_attention" not in (cfg.get("layer_types") or []):
        raise UnsupportedModelError(f"{model_dir}: config has no GatedDeltaNet (linear_attention) layers")


def check_supported(model_dir: str | Path) -> None:
    """Raises ``UnsupportedModelError`` unless the folder can be run.

    Qwen3.5 hybrid checkpoints use the fast Q4/MTP engine when geometry matches. Supported GGUF
    checkpoints (such as ``qwen35moe`` / ``qwen35``) and standard Hugging Face causal LMs
    (OLMoE, Llama, Mistral, Qwen2, Gemma, Phi, …) run on the BF16 ring.
    A Colibrì container that is not Qwen3.5 is rejected.
    """
    fmt = detect_format(model_dir)
    if fmt is None:
        raise UnsupportedModelError(
            f"{model_dir}: no Colibri container, Hugging Face safetensors checkpoint, or GGUF file found"
        )
    if fmt == "gguf":
        src = GGUFModelSource(model_dir)
        src.close()
        return
    try:
        cfg = raw_text_config_dict(model_dir)
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedModelError(f"{model_dir}: unreadable config ({exc})") from exc
    if fmt == "colibri":
        _require_qwen35_hybrid(model_dir, cfg)
        return
    if is_qwen35_hybrid(model_dir):
        _require_qwen35_hybrid(model_dir, cfg)
        HFShardedSource(model_dir).close()
        return
    if "num_hidden_layers" not in cfg or "hidden_size" not in cfg:
        raise UnsupportedModelError(f"{model_dir}: config has no decoder shape (num_hidden_layers, hidden_size)")
    # Validates shard index, decoder layers and the embed/norm/lm_head edges.
    HFShardedSource(model_dir).close()


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
    cfg_dict = raw_text_config_dict(d)
    _require_qwen35_hybrid(d, cfg_dict)
    if detect_format(d) == "colibri":
        return AutoConfig.from_pretrained(str(d), config_file="config.hf.json")
    if cfg_dict.get("model_type") in ("qwen3_5", "qwen3_5_text"):
        # Multimodal checkpoints nest the text model under `text_config`; build the plain text config
        # exactly as a Colibri container's `config.hf.json` would be loaded.
        kwargs = {k: v for k, v in cfg_dict.items() if k not in ("model_type", "transformers_version")}
        cfg = AutoConfig.for_model("qwen3_5_text", **kwargs)
    else:
        cfg = AutoConfig.from_pretrained(str(d))
    if getattr(cfg, "model_type", "") != "qwen3_5_text":
        raise UnsupportedModelError(f"{model_dir}: expected qwen3_5_text config, got {cfg.model_type!r}")
    return cfg


def load_runtime_config(model_dir: str | Path):
    """Config for the engine: Qwen3.5 text config, zero-mmap GGUF-derived config, or checkpoint causal config."""
    d = Path(model_dir)
    if detect_format(d) == "gguf":
        gguf_path = find_gguf_file(d)
        _validate_gguf_header(gguf_path)
        from transformers import AutoConfig
        from transformers.integrations.gguf import get_gguf_config

        try:
            meta, raw_tensors = _read_gguf_header_cached(gguf_path)
            cfg_meta = {
                k: (len(v) if isinstance(v, list) and (not v or isinstance(v[0], str)) else v)
                for k, v in meta.items()
            }
            tensor_names = tuple(t["name"] for t in raw_tensors)
            cfg_dict = get_gguf_config(cfg_meta, tensor_names)
            return AutoConfig.for_model(**cfg_dict)
        except Exception as exc:
            raise UnsupportedModelError(f"{d}: invalid GGUF config ({exc})") from exc
    if is_qwen35_hybrid(d):
        return load_text_config(d)
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(d))


def load_runtime_tokenizer(model_dir: str | Path, config: Any = None):
    """Loads the checkpoint tokenizer without calling ``mmap`` / ``np.memmap`` on GGUF files."""
    d = Path(model_dir)
    if detect_format(d) == "gguf":
        gguf_path = find_gguf_file(d)
        _validate_gguf_header(gguf_path)
        from transformers import PreTrainedTokenizerFast
        from transformers.integrations.gguf import (
            GGUF_TOKENIZER_MAPPING,
            convert_gguf_tokenizer,
        )
        from transformers.integrations.gguf.gguf_tokenizer_mapping import _SPECIAL_TOKENS
        from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

        cfg = config if config is not None else load_runtime_config(d)
        meta, _ = _read_gguf_header_cached(gguf_path)
        arch = str(meta["general.architecture"])
        grouped = {
            group: {name: meta[f"tokenizer.{key}"] for key, name in renames.items() if f"tokenizer.{key}" in meta}
            for group, renames in GGUF_TOKENIZER_MAPPING.items()
        }
        tok_dict, tok_cfg = grouped["tokenizer"], grouped["tokenizer_config"]
        for name, key in _SPECIAL_TOKENS.items():
            tid = meta.get(f"tokenizer.ggml.{key}")
            tok_cfg[name] = tok_dict["tokens"][tid] if tid is not None else None
        fast_tok, add_kw = convert_gguf_tokenizer(arch, tok_dict)
        mapped = TOKENIZER_MAPPING.get(type(cfg), None)
        tok_cls = (mapped[1] or mapped[0]) if isinstance(mapped, (tuple, list)) else mapped
        tok_cls = tok_cls or PreTrainedTokenizerFast
        return tok_cls(tokenizer_object=fast_tok, **tok_cfg, **add_kw)

    from transformers import AutoTokenizer

    tok_kwargs: dict[str, Any] = {"fix_mistral_regex": False, "local_files_only": True}
    try:
        return AutoTokenizer.from_pretrained(str(d), **tok_kwargs)
    except TypeError:
        tok_kwargs.pop("fix_mistral_regex", None)
        return AutoTokenizer.from_pretrained(str(d), **tok_kwargs)


# --------------------------------------------------------------------------------------
# Model folder resolution
# --------------------------------------------------------------------------------------

_PREFERRED_NAMES = ("Qwen3.5-9B", "qwen3.5-9b", "deepseek-v4-9b-colibri", "deepseek-v4-9b")


def find_local_models(root: str | Path) -> list[Path]:
    """Lists runnable model folders under ``<root>/models``.

    Qwen3.5 names stay first, so a neighbouring OLMoE or Qwen3.6 GGUF folder is never
    the default while a Qwen3.5-9B checkpoint is present. Broken folders are skipped.
    """
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

    Order: explicit value → ``AIRCOLIBRI_MODEL`` → runnable folders in ``<root>/models``
    (Qwen3.5 first) → a cached ``Qwen/Qwen3.5-9B`` snapshot.
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
# Positioned-read safetensors access & parallel NVMe pinned slice pool (no mmap)
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


def _pread_or_seek_into(f: Any, abs_offset: int, view: memoryview, path: str) -> None:
    total = len(view)
    if total == 0:
        return
    if hasattr(os, "pread"):
        done = 0
        fd = f.fileno()
        while done < total:
            chunk = os.pread(fd, total - done, abs_offset + done)
            if not chunk:
                raise OSError(f"{path}: unexpected EOF at offset {abs_offset + done}")
            view[done : done + len(chunk)] = chunk
            done += len(chunk)
    else:
        f.seek(abs_offset)
        done = 0
        while done < total:
            got = f.readinto(view[done:])
            if not got:
                raise OSError(f"{path}: unexpected EOF at offset {abs_offset + done}")
            done += got


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

    def tensor_span(self, path: str | Path, key: str) -> tuple[str, int, int, str, tuple[int, ...]]:
        """Returns ``(path_str, abs_offset, nbytes, torch_dtype_name, shape)``."""
        p = str(path)
        with self._lock:
            header, base = self._header(p)
        meta = header.get(key)
        if meta is None:
            raise KeyError(f"{key} not in {p}")
        dtype_name, _ = _ST_DTYPES[meta["dtype"]]
        start, end = meta["data_offsets"]
        return p, int(base + start), int(end - start), dtype_name, tuple(int(x) for x in meta["shape"])

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
            _pread_or_seek_into(f, base + start, memoryview(buf), path)
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


def _canonical_device(device: Any):
    import torch

    d = torch.device(device) if device is not None else torch.device("cpu")
    if d.type == "cuda" and d.index is None:
        return torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0)
    return d


class _ParallelSliceStagingPool:
    """Double-buffered pinned host staging + 4-worker NVMe positioned-read pool (zero mmap, zero races)."""

    def __init__(self, slot_bytes: int, device: Any, workers: int = _IO_WORKERS):
        import torch

        self.device = _canonical_device(device)
        self.use_cuda = self.device.type == "cuda"
        self.workers = max(1, int(workers))
        self.slot_bytes = max(4096, int(slot_bytes))
        self._slots = [
            torch.empty(self.slot_bytes, dtype=torch.uint8, pin_memory=self.use_cuda)
            for _ in range(2 if self.use_cuda else 1)
        ]
        self._views = [memoryview(s.numpy()) for s in self._slots]
        self._events = [torch.cuda.Event() if self.use_cuda else None for _ in self._slots]
        self._in_flight = [False] * len(self._slots)
        self._cursor = 0

        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._handle_queues: dict[str, queue.SimpleQueue] = {}
        self._all_handles: list[Any] = []

    def _borrow_handle(self, path: str):
        with self._lock:
            q = self._handle_queues.get(path)
            if q is None:
                q = queue.SimpleQueue()
                self._handle_queues[path] = q
        try:
            return q.get_nowait(), q
        except queue.Empty:
            f = open(path, "rb", buffering=0)
            with self._lock:
                self._all_handles.append(f)
            return f, q

    def _read_one(self, task: tuple[str, int, int, int], view: memoryview) -> None:
        path, abs_offset, dst_offset, length = task
        f, q = self._borrow_handle(path)
        try:
            _pread_or_seek_into(f, abs_offset, view[dst_offset : dst_offset + length], path)
        finally:
            q.put(f)

    def _ensure_capacity(self, needed_bytes: int) -> None:
        import torch

        if needed_bytes <= self.slot_bytes:
            return
        if self.use_cuda:
            for i, in_f in enumerate(self._in_flight):
                if in_f and self._events[i] is not None:
                    self._events[i].synchronize()
                    self._in_flight[i] = False
        self.slot_bytes = max(needed_bytes, self.slot_bytes * 2)
        self._slots = [
            torch.empty(self.slot_bytes, dtype=torch.uint8, pin_memory=self.use_cuda)
            for _ in range(len(self._slots))
        ]
        self._views = [memoryview(s.numpy()) for s in self._slots]

    @staticmethod
    def _coalesce_tasks(
        tasks: list[tuple[str, int, int, int]], max_chunk_bytes: int = 8 * 1024 * 1024
    ) -> list[tuple[str, int, int, int]]:
        if len(tasks) <= 1:
            return tasks
        merged: list[tuple[str, int, int, int]] = []
        cur_p, cur_off, cur_dst, cur_n = tasks[0]
        for p, off, dst, n in tasks[1:]:
            if (
                p == cur_p
                and off == cur_off + cur_n
                and dst == cur_dst + cur_n
                and cur_n + n <= max_chunk_bytes
            ):
                cur_n += n
            else:
                merged.append((cur_p, cur_off, cur_dst, cur_n))
                cur_p, cur_off, cur_dst, cur_n = p, off, dst, n
        merged.append((cur_p, cur_off, cur_dst, cur_n))
        return merged

    def fetch_to_device(self, tasks: list[tuple[str, int, int, int]], total_bytes: int):
        """Executes ``[(path, abs_offset, dst_offset, nbytes), ...]`` into a pinned slot and copies to device."""
        import torch

        if total_bytes <= 0 or not tasks:
            return torch.empty(0, dtype=torch.uint8, device=self.device)
        tasks = self._coalesce_tasks(tasks)
        self._ensure_capacity(total_bytes)
        idx = self._cursor
        self._cursor = (self._cursor + 1) % len(self._slots)

        if self.use_cuda and self._in_flight[idx] and self._events[idx] is not None:
            self._events[idx].synchronize()
            self._in_flight[idx] = False

        view = self._views[idx]
        if len(tasks) == 1 or self.workers == 1:
            for t in tasks:
                self._read_one(t, view)
        else:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="aircolibri-moe-io")
            list(self._pool.map(lambda t: self._read_one(t, view), tasks))

        src_u8 = self._slots[idx][:total_bytes]
        if self.use_cuda:
            dst_u8 = src_u8.to(device=self.device, non_blocking=True)
            self._events[idx].record(torch.cuda.current_stream(self.device))
            self._in_flight[idx] = True
            return dst_u8
        return src_u8.clone()

    def close(self) -> None:
        if self.use_cuda:
            for i, in_f in enumerate(self._in_flight):
                if in_f and self._events[i] is not None:
                    try:
                        self._events[i].synchronize()
                    except Exception:
                        pass
                    self._in_flight[i] = False
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        with self._lock:
            for f in self._all_handles:
                try:
                    f.close()
                except OSError:
                    pass
            self._all_handles.clear()
            self._handle_queues.clear()
        self._views.clear()
        self._slots.clear()


# --------------------------------------------------------------------------------------
# GGUF block dequantization (extended ggml types + NVRTC BF16 CUDA fast path)
# --------------------------------------------------------------------------------------

_EXTRA_GGML_BLOCKS = {
    GGML_TYPE_Q5_0: ("Q5_0", 32, 22),
    GGML_TYPE_Q5_1: ("Q5_1", 32, 24),
    GGML_TYPE_Q8_K: ("Q8_K", 256, 292),
    GGML_TYPE_MXFP4: ("MXFP4", 32, 17),
}
_MXFP4_DOUBLED_LEVELS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0)

_NVRTC_GGUF_CU = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float val) {
    unsigned int bits = __float_as_uint(val);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) return 0x7fc0u;
    unsigned int lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (unsigned short)(bits >> 16);
}
__device__ __forceinline__ float bf16_round(float val) {
    return bf16_to_f32(f32_to_bf16_rne(val));
}
__device__ __forceinline__ float fp16_to_f32(unsigned short h) {
    unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    unsigned int exp  = (h >> 10) & 0x1fu;
    unsigned int mant = h & 0x03ffu;
    if (exp == 0u) {
        if (mant == 0u) return __uint_as_float(sign);
        while ((mant & 0x0400u) == 0u) { mant <<= 1; exp -= 1u; }
        exp += 1u; mant &= ~0x0400u;
    } else if (exp == 31u) {
        return __uint_as_float(sign | 0x7f800000u | (mant << 13));
    }
    exp = exp + (127u - 15u);
    return __uint_as_float(sign | (exp << 23) | (mant << 13));
}
__device__ __forceinline__ unsigned short rd_u16(const unsigned char* p) {
    return (unsigned short)p[0] | ((unsigned short)p[1] << 8);
}

extern "C" __global__ void deq_q8_0_bf16(const unsigned char* __restrict__ src, unsigned short* __restrict__ dst, int nb) {
    int b = blockIdx.x;
    if (b >= nb) return;
    int t = threadIdx.x;
    const unsigned char* p = src + (long long)b * 34;
    float d = bf16_round(fp16_to_f32(rd_u16(p)));
    int q = (int)((signed char)p[2 + t]);
    dst[(long long)b * 32 + t] = f32_to_bf16_rne(d * (float)q);
}

extern "C" __global__ void deq_q4_0_bf16(const unsigned char* __restrict__ src, unsigned short* __restrict__ dst, int nb) {
    int b = blockIdx.x;
    if (b >= nb) return;
    int t = threadIdx.x;
    const unsigned char* p = src + (long long)b * 18;
    float d = bf16_round(fp16_to_f32(rd_u16(p)));
    unsigned char byte = p[2 + (t & 15)];
    int nib = (t < 16) ? (byte & 0xF) : (byte >> 4);
    dst[(long long)b * 32 + t] = f32_to_bf16_rne(d * (float)(nib - 8));
}

extern "C" __global__ void deq_q4_k_bf16(const unsigned char* __restrict__ src, unsigned short* __restrict__ dst, int nb) {
    int b = blockIdx.x;
    if (b >= nb) return;
    int t = threadIdx.x;
    const unsigned char* p = src + (long long)b * 144;
    float d = fp16_to_f32(rd_u16(p));
    float dmin = fp16_to_f32(rd_u16(p + 2));
    const unsigned char* sc = p + 4;
    int sub = t >> 5;
    int in_sub = t & 31;
    unsigned int s_val, m_val;
    if (sub < 4) {
        s_val = sc[sub] & 63u;
        m_val = sc[sub + 4] & 63u;
    } else {
        s_val = (sc[sub + 4] & 0xFu) | ((sc[sub - 4] >> 6) << 4);
        m_val = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4);
    }
    int chunk = sub >> 1;
    int shift = (sub & 1) * 4;
    int q = (p[16 + chunk * 32 + in_sub] >> shift) & 0xF;
    float dl = bf16_round(d * (float)s_val);
    float ml = bf16_round(dmin * (float)m_val);
    dst[(long long)b * 256 + t] = f32_to_bf16_rne(bf16_round(dl * (float)q) - ml);
}

extern "C" __global__ void deq_q5_k_bf16(const unsigned char* __restrict__ src, unsigned short* __restrict__ dst, int nb) {
    int b = blockIdx.x;
    if (b >= nb) return;
    int t = threadIdx.x;
    const unsigned char* p = src + (long long)b * 176;
    float d = fp16_to_f32(rd_u16(p));
    float dmin = fp16_to_f32(rd_u16(p + 2));
    const unsigned char* sc = p + 4;
    int sub = t >> 5;
    int in_sub = t & 31;
    unsigned int s_val, m_val;
    if (sub < 4) {
        s_val = sc[sub] & 63u;
        m_val = sc[sub + 4] & 63u;
    } else {
        s_val = (sc[sub + 4] & 0xFu) | ((sc[sub - 4] >> 6) << 4);
        m_val = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4);
    }
    int chunk = sub >> 1;
    int shift = (sub & 1) * 4;
    int low = (p[48 + chunk * 32 + in_sub] >> shift) & 0xF;
    int high = (p[16 + in_sub] >> sub) & 1;
    int q = low | (high << 4);
    float dl = bf16_round(d * (float)s_val);
    float ml = bf16_round(dmin * (float)m_val);
    dst[(long long)b * 256 + t] = f32_to_bf16_rne(bf16_round(dl * (float)q) - ml);
}

extern "C" __global__ void deq_q6_k_bf16(const unsigned char* __restrict__ src, unsigned short* __restrict__ dst, int nb) {
    int b = blockIdx.x;
    if (b >= nb) return;
    int t = threadIdx.x;
    const unsigned char* p = src + (long long)b * 210;
    float d = fp16_to_f32(rd_u16(p + 208));
    int scale_idx = t >> 4;
    int half_blk = t >> 7;
    int quad = (t & 127) >> 5;
    int in32 = t & 31;
    unsigned char ql_b = p[half_blk * 64 + (quad & 1) * 32 + in32];
    int low = (ql_b >> ((quad >> 1) * 4)) & 0xF;
    unsigned char qh_b = p[128 + half_blk * 32 + in32];
    int high = (qh_b >> (quad * 2)) & 3;
    int q = (low | (high << 4)) - 32;
    float sc = bf16_round(d * (float)((signed char)p[192 + scale_idx]));
    dst[(long long)b * 256 + t] = f32_to_bf16_rne(sc * (float)q);
}
"""

_NVRTC_GGUF_SPECS = {
    GGML_TYPE_Q8_0: (b"deq_q8_0_bf16", 34, 32),
    GGML_TYPE_Q4_0: (b"deq_q4_0_bf16", 18, 32),
    GGML_TYPE_Q4_K: (b"deq_q4_k_bf16", 144, 256),
    GGML_TYPE_Q5_K: (b"deq_q5_k_bf16", 176, 256),
    GGML_TYPE_Q6_K: (b"deq_q6_k_bf16", 210, 256),
}
_NVRTC_GGUF_LOCK = threading.Lock()
_NVRTC_GGUF_MODS: dict[int, tuple[Any, dict[int, tuple[ctypes.c_void_p, int, int]]] | None] = {}


def _get_nvrtc_gguf_kernels(device: Any):
    import torch

    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    dev_idx = 0 if dev.index is None else int(dev.index)
    if dev_idx in _NVRTC_GGUF_MODS:
        return _NVRTC_GGUF_MODS[dev_idx]
    with _NVRTC_GGUF_LOCK:
        if dev_idx in _NVRTC_GGUF_MODS:
            return _NVRTC_GGUF_MODS[dev_idx]
        try:
            from src.colibri_nvrtc import PTX_CACHE_DIR, _load_driver_and_nvrtc

            nvcuda, nvrtc = _load_driver_and_nvrtc()
            with torch.cuda.device(dev_idx):
                _ = torch.empty(1, device=f"cuda:{dev_idx}")
                major, minor = torch.cuda.get_device_capability(dev_idx)
                src_bytes = _NVRTC_GGUF_CU.encode("utf-8")
                digest = hashlib.sha256(src_bytes).hexdigest()[:16]
                PTX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file = PTX_CACHE_DIR / f"gguf_dequant_sm{major}{minor}_{digest}.ptx"
                if cache_file.is_file():
                    raw_ptx = cache_file.read_bytes()
                    if not raw_ptx.endswith(b"\x00"):
                        raw_ptx += b"\x00"
                else:
                    prog = ctypes.c_void_p()
                    if nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src_bytes, b"gguf_deq.cu", 0, None, None) != 0:
                        raise RuntimeError("nvrtcCreateProgram failed")
                    try:
                        opts = (ctypes.c_char_p * 1)(f"--gpu-architecture=compute_{major}{minor}".encode("utf-8"))
                        if nvrtc.nvrtcCompileProgram(prog, 1, opts) != 0:
                            raise RuntimeError("nvrtcCompileProgram failed")
                        sz = ctypes.c_size_t()
                        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(sz))
                        ptx_buf = ctypes.create_string_buffer(sz.value)
                        nvrtc.nvrtcGetPTX(prog, ptx_buf)
                        raw_ptx = ptx_buf.raw
                    finally:
                        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
                    tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
                    tmp.write_bytes(raw_ptx)
                    os.replace(tmp, cache_file)

                mod = ctypes.c_void_p()
                ptx_c = ctypes.create_string_buffer(raw_ptx)
                if nvcuda.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(ptx_c, ctypes.c_void_p)) != 0:
                    raise RuntimeError("cuModuleLoadData failed")
                fns: dict[int, tuple[ctypes.c_void_p, int, int]] = {}
                for gt, (kname, bb, be) in _NVRTC_GGUF_SPECS.items():
                    fn = ctypes.c_void_p()
                    if nvcuda.cuModuleGetFunction(ctypes.byref(fn), mod, kname) != 0:
                        raise RuntimeError(f"cuModuleGetFunction({kname!r}) failed")
                    fns[gt] = (fn, bb, be)
                _NVRTC_GGUF_MODS[dev_idx] = (nvcuda, fns)
        except Exception:
            _NVRTC_GGUF_MODS[dev_idx] = None
        return _NVRTC_GGUF_MODS[dev_idx]


def dequantize_gguf_tensor(u8: Any, ggml_type: int, dtype: Any = None):
    """Dequantizes flat ``uint8`` GGUF block bytes into flat values of ``dtype``."""
    import torch
    from transformers.integrations.gguf.dequant import GGML_BLOCK, dequantize

    if dtype is None:
        dtype = torch.float32
    if u8.numel() == 0:
        return torch.empty(0, dtype=dtype, device=u8.device)

    if u8.is_cuda and dtype == torch.bfloat16 and ggml_type in _NVRTC_GGUF_SPECS:
        kern_pack = _get_nvrtc_gguf_kernels(u8.device)
        if kern_pack is not None:
            nvcuda, fns = kern_pack
            fn, block_bytes, block_elems = fns[ggml_type]
            if not u8.is_contiguous():
                u8 = u8.contiguous()
            nb = u8.numel() // block_bytes
            out = torch.empty(nb * block_elems, dtype=torch.bfloat16, device=u8.device)
            stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(u8.device).cuda_stream)
            c_src = ctypes.c_uint64(u8.data_ptr())
            c_dst = ctypes.c_uint64(out.data_ptr())
            c_nb = ctypes.c_int(nb)
            params = (ctypes.c_void_p * 3)(
                ctypes.cast(ctypes.byref(c_src), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_dst), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(c_nb), ctypes.c_void_p),
            )
            rc = nvcuda.cuLaunchKernel(fn, nb, 1, 1, block_elems, 1, 1, 0, stream_ptr, params, None)
            if rc == 0:
                return out

    if ggml_type in GGML_BLOCK:
        return dequantize(u8, ggml_type, dtype=dtype)

    if ggml_type == GGML_TYPE_Q5_0:
        blocks = u8.reshape(-1, 22)
        d = blocks[:, 0:2].view(torch.float16).to(dtype)
        qh = blocks[:, 2:6].unsqueeze(-1) >> torch.arange(8, device=u8.device, dtype=torch.uint8)
        high = (qh.reshape(-1, 32) & 1) << 4
        qs = blocks[:, 6:22]
        low = torch.cat([qs & 0xF, qs >> 4], dim=1)
        q = (low | high).to(torch.int8) - 16
        return (d * q).reshape(-1)

    if ggml_type == GGML_TYPE_Q5_1:
        blocks = u8.reshape(-1, 24)
        d = blocks[:, 0:2].view(torch.float16).to(dtype)
        m = blocks[:, 2:4].view(torch.float16).to(dtype)
        qh = blocks[:, 4:8].unsqueeze(-1) >> torch.arange(8, device=u8.device, dtype=torch.uint8)
        high = (qh.reshape(-1, 32) & 1) << 4
        qs = blocks[:, 8:24]
        low = torch.cat([qs & 0xF, qs >> 4], dim=1)
        q = low | high
        return (d * q + m).reshape(-1)

    if ggml_type == GGML_TYPE_Q8_K:
        blocks = u8.reshape(-1, 292)
        d = blocks[:, 0:4].contiguous().view(torch.float32)
        qs = blocks[:, 4:260].view(torch.int8)
        return (d * qs.float()).to(dtype).reshape(-1)

    if ggml_type == GGML_TYPE_MXFP4:
        blocks = u8.reshape(-1, 17)
        exp_u8 = blocks[:, 0:1].to(torch.int32)
        bits = torch.where(exp_u8 < 2, 0x00200000 << exp_u8, (exp_u8 - 1) << 23)
        d = bits.view(torch.float32)
        qs = blocks[:, 1:17]
        nibs = torch.cat([qs & 0xF, qs >> 4], dim=1).long()
        levels = torch.tensor(_MXFP4_DOUBLED_LEVELS, device=u8.device, dtype=torch.float32)
        return (d * levels[nibs]).to(dtype).reshape(-1)

    raise UnsupportedModelError(f"Unsupported ggml block type {ggml_type}")


def _fast_moe_dispatch(
    self_exp: Any,
    hidden_states: Any,
    top_k_index: Any,
    top_k_weights: Any,
    num_exp: int,
    load_batch_fn: Any,
):
    """Sync-free MoE router dispatch: batched GEMV/BMM for decode (T=1) and grouped batches for prefill."""
    import torch
    import torch.nn.functional as F

    T, D = hidden_states.shape
    if T == 0:
        return torch.zeros_like(hidden_states)

    if T == 1:
        # Single-token decode: one D2H read of K integers, zero one_hot/torch.where syncs, 3 batched cuBLAS calls.
        raw_k = top_k_index[0].tolist()
        valid_pos: list[int] = []
        exp_ids: list[int] = []
        for pos, eid in enumerate(raw_k):
            e_int = int(eid)
            if 0 <= e_int < num_exp:
                valid_pos.append(pos)
                exp_ids.append(e_int)
        if not exp_ids:
            return torch.zeros_like(hidden_states)

        comp_dtype = hidden_states.dtype
        w = top_k_weights[0, valid_pos].to(dtype=comp_dtype)
        x = hidden_states[0]
        if len(exp_ids) <= _MAX_EXPERT_DEQUANT_BATCH:
            gate_w, up_w, down_w = load_batch_fn(exp_ids, hidden_states.device, comp_dtype, True)
            act = self_exp.act_fn(torch.matmul(gate_w, x)) * torch.matmul(up_w, x)
            del gate_w, up_w
            out_k = torch.bmm(down_w, act.unsqueeze(-1)).squeeze(-1)
            del down_w
            return torch.matmul(w, out_k).unsqueeze(0).to(dtype=hidden_states.dtype)

        acc = torch.zeros((1, D), dtype=comp_dtype, device=hidden_states.device)
        for b0 in range(0, len(exp_ids), _MAX_EXPERT_DEQUANT_BATCH):
            sub_ids = exp_ids[b0 : b0 + _MAX_EXPERT_DEQUANT_BATCH]
            sub_w = w[b0 : b0 + _MAX_EXPERT_DEQUANT_BATCH]
            gate_w, up_w, down_w = load_batch_fn(sub_ids, hidden_states.device, comp_dtype, True)
            act = self_exp.act_fn(torch.matmul(gate_w, x)) * torch.matmul(up_w, x)
            del gate_w, up_w
            out_k = torch.bmm(down_w, act.unsqueeze(-1)).squeeze(-1)
            del down_w
            acc = acc + torch.matmul(sub_w, out_k).unsqueeze(0)
        return acc.to(dtype=hidden_states.dtype)

    # Multi-token prefill: group token/slot indices on CPU in one transfer (zero per-expert GPU syncs).
    idx_rows = top_k_index.detach().to(device="cpu").tolist()
    by_expert: dict[int, tuple[list[int], list[int]]] = {}
    for tok_i, row in enumerate(idx_rows):
        for k_pos, eid in enumerate(row):
            e_int = int(eid)
            if 0 <= e_int < num_exp:
                bucket = by_expert.get(e_int)
                if bucket is None:
                    bucket = ([], [])
                    by_expert[e_int] = bucket
                bucket[0].append(tok_i)
                bucket[1].append(k_pos)

    if not by_expert:
        return torch.zeros_like(hidden_states)

    hit_list = sorted(by_expert)
    dev = hidden_states.device
    comp_dtype = hidden_states.dtype
    is_short_step = T <= 4
    final_hidden_states = torch.zeros_like(hidden_states)

    flat_tok: list[int] = []
    flat_pos: list[int] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for exp_id in hit_list:
        tok_list, pos_list = by_expert[exp_id]
        nxt = cursor + len(tok_list)
        spans.append((cursor, nxt))
        flat_tok.extend(tok_list)
        flat_pos.extend(pos_list)
        cursor = nxt

    all_tok_gpu = torch.tensor(flat_tok, dtype=torch.long, device=dev)
    all_pos_gpu = torch.tensor(flat_pos, dtype=torch.long, device=dev)
    all_w_gpu = top_k_weights[all_tok_gpu, all_pos_gpu, None].to(dtype=comp_dtype)

    for b0 in range(0, len(hit_list), _MAX_EXPERT_DEQUANT_BATCH):
        batch_ids = hit_list[b0 : b0 + _MAX_EXPERT_DEQUANT_BATCH]
        gate_w, up_w, down_w = load_batch_fn(batch_ids, dev, comp_dtype, is_short_step)
        for local_i in range(len(batch_ids)):
            s0, s1 = spans[b0 + local_i]
            token_idx = all_tok_gpu[s0:s1]
            current_state = hidden_states.index_select(0, token_idx)
            act = self_exp.act_fn(F.linear(current_state, gate_w[local_i])) * F.linear(current_state, up_w[local_i])
            current_hidden = F.linear(act, down_w[local_i]) * all_w_gpu[s0:s1]
            final_hidden_states.index_add_(0, token_idx, current_hidden.to(dtype=final_hidden_states.dtype))
        del gate_w, up_w, down_w

    return final_hidden_states


# --------------------------------------------------------------------------------------
# Hugging Face sharded checkpoints
# --------------------------------------------------------------------------------------

_HF_EXPERT_LEAF_RE = re.compile(
    r"^(mlp\.experts|block_sparse_moe\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj|w1|w3|w2)\.weight$"
)
_HF_EXPERT_ROLE = {
    "gate_proj": "gate",
    "w1": "gate",
    "up_proj": "up",
    "w3": "up",
    "down_proj": "down",
    "w2": "down",
}


class _HFExpertCacheAndLoader:
    """Contiguous single-arena VRAM LRU cache + 4-worker pinned loader for per-expert HF MoE shards."""

    def __init__(
        self,
        reader: PreadSafetensorsReader,
        routed_experts: dict[int, dict[int, dict[str, tuple[str, str]]]],
        device: Any,
        dtype: Any,
        budget_bytes: int,
    ):
        import torch

        self.device = _canonical_device(device)
        self.dtype = dtype
        self.elem_size = torch.empty((), dtype=dtype).element_size()
        self.hits = 0
        self.misses = 0

        # Pre-index byte spans (path, abs_offset, nbytes, raw_torch_dtype) for every expert once.
        self._spans: dict[tuple[int, int], tuple[tuple[str, int, int], tuple[str, int, int], tuple[str, int, int], Any]] = {}
        inter_dim = 0
        hidden_dim = 0
        uniform_shapes = True
        max_expert_raw_bytes = 0
        for l_idx, by_id in routed_experts.items():
            for exp_id, parts in by_id.items():
                gp, goff, gn, gdt, gsh = reader.tensor_span(*parts["gate"])
                up, uoff, un, udt, ush = reader.tensor_span(*parts["up"])
                dp, doff, dn, ddt, dsh = reader.tensor_span(*parts["down"])
                if inter_dim == 0 and hidden_dim == 0:
                    inter_dim, hidden_dim = int(gsh[0]), int(gsh[1])
                if gsh != (inter_dim, hidden_dim) or ush != (inter_dim, hidden_dim) or dsh != (hidden_dim, inter_dim):
                    uniform_shapes = False
                if not (gdt == udt == ddt):
                    uniform_shapes = False
                raw_dt = getattr(torch, gdt)
                self._spans[(l_idx, exp_id)] = ((gp, goff, gn), (up, uoff, un), (dp, doff, dn), raw_dt)
                max_expert_raw_bytes = max(max_expert_raw_bytes, gn + un + dn)

        self.inter_dim = inter_dim
        self.hidden_dim = hidden_dim
        self._staging = _ParallelSliceStagingPool(
            slot_bytes=max(4096, _MAX_EXPERT_DEQUANT_BATCH * max_expert_raw_bytes),
            device=self.device,
        )

        expert_vram_bytes = 3 * inter_dim * hidden_dim * self.elem_size
        total_experts = len(self._spans)
        if self.device.type == "cuda" and uniform_shapes and expert_vram_bytes > 0 and budget_bytes >= expert_vram_bytes * _MAX_EXPERT_DEQUANT_BATCH:
            num_slots = min(total_experts, int(budget_bytes) // expert_vram_bytes)
        else:
            num_slots = 0

        self.num_slots = num_slots
        self.arena = (
            torch.empty((num_slots, 3, inter_dim, hidden_dim), dtype=dtype, device=self.device)
            if num_slots > 0
            else None
        )
        self._lru: "OrderedDict[tuple[int, int], int]" = OrderedDict()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        arena_bytes = int(self.arena.numel() * self.arena.element_size()) if self.arena is not None else 0
        return {
            "slots": self.num_slots,
            "used_slots": len(self._lru),
            "arena_mib": round(arena_bytes / (1024 ** 2), 1),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(100.0 * self.hits / max(1, total), 1) if total > 0 else 0.0,
        }

    def _fetch_raw_experts(self, layer_idx: int, exp_ids: list[int], dtype: Any):
        """Fetches ``exp_ids`` in parallel into ``(K, 3, I, D)`` on ``self.device``."""
        import torch

        k = len(exp_ids)
        I, D = self.inter_dim, self.hidden_dim
        tasks: list[tuple[str, int, int, int]] = []
        cursor = 0
        raw_dt = self.dtype
        for eid in exp_ids:
            g_sp, u_sp, d_sp, raw_dt = self._spans[(layer_idx, eid)]
            for path, off, nbytes in (g_sp, u_sp, d_sp):
                tasks.append((path, off, cursor, nbytes))
                cursor += nbytes
        u8 = self._staging.fetch_to_device(tasks, cursor)
        vals = u8.view(raw_dt).to(dtype=dtype)
        # Stored per expert as [gate (I, D), up (I, D), down (D, I)] — all three have I * D elements!
        return vals.reshape(k, 3, I, D)

    def load_batch(self, layer_idx: int, batch_ids: list[int], dev: Any, comp_dtype: Any, allow_evict: bool):
        import torch

        dev = _canonical_device(dev)
        I, D = self.inter_dim, self.hidden_dim
        k = len(batch_ids)
        if self.arena is None or dev != self.device:
            self.misses += k
            packed = self._fetch_raw_experts(layer_idx, batch_ids, comp_dtype).to(device=dev)
            return packed[:, 0], packed[:, 1], packed[:, 2].reshape(k, D, I)

        # Touch hits first so none of this batch's experts can be chosen as LRU victims.
        miss_ids: list[int] = []
        seen_miss: set[int] = set()
        for eid in batch_ids:
            key = (layer_idx, eid)
            if key in self._lru:
                self.hits += 1
                if allow_evict:
                    self._lru.move_to_end(key)
            else:
                self.misses += 1
                if eid not in seen_miss:
                    seen_miss.add(eid)
                    miss_ids.append(eid)

        if not miss_ids:
            slots = torch.tensor([self._lru[(layer_idx, eid)] for eid in batch_ids], dtype=torch.long, device=dev)
            packed = self.arena.index_select(0, slots).to(dtype=comp_dtype)
            return packed[:, 0], packed[:, 1], packed[:, 2].reshape(k, D, I)

        can_admit = (allow_evict and self.num_slots >= len(batch_ids)) or (
            len(self._lru) + len(miss_ids) <= self.num_slots
        )
        miss_packed = self._fetch_raw_experts(layer_idx, miss_ids, self.dtype)

        if can_admit:
            target_slots: list[int] = []
            for eid in miss_ids:
                if len(self._lru) < self.num_slots:
                    s_idx = len(self._lru)
                else:
                    _, s_idx = self._lru.popitem(last=False)
                self._lru[(layer_idx, eid)] = s_idx
                target_slots.append(s_idx)
            slot_t = torch.tensor(target_slots, dtype=torch.long, device=dev)
            self.arena.index_copy_(0, slot_t, miss_packed)
            all_slots = torch.tensor([self._lru[(layer_idx, eid)] for eid in batch_ids], dtype=torch.long, device=dev)
            packed = self.arena.index_select(0, all_slots).to(dtype=comp_dtype)
            return packed[:, 0], packed[:, 1], packed[:, 2].reshape(k, D, I)

        # Prefill anti-thrashing path when arena is full: combine arena hits + transient misses.
        miss_pos = {eid: idx for idx, eid in enumerate(miss_ids)}
        out = torch.empty((k, 3, I, D), dtype=comp_dtype, device=dev)
        for i, eid in enumerate(batch_ids):
            key = (layer_idx, eid)
            if key in self._lru:
                out[i].copy_(self.arena[self._lru[key]])
            else:
                out[i].copy_(miss_packed[miss_pos[eid]])
        return out[:, 0], out[:, 1], out[:, 2].reshape(k, D, I)

    def close(self) -> None:
        self._staging.close()
        self._lru.clear()
        self.arena = None


class HFShardedSource:
    """Maps a standard Hugging Face checkpoint onto the engine's canonical tensor names.

    Canonical names: ``model.layers.{i}.<param>``, ``model.embed_tokens.weight``,
    ``model.norm.weight``, ``lm_head.weight`` and ``mtp.*``. Per-expert MoE weights
    (such as ``OLMoE`` / ``Qwen3Moe`` / ``Mixtral``) stay on disk and are fetched with
    positioned reads (and cached in a budgeted VRAM slot arena) for only the routed experts.
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
        self._routed_experts: dict[int, dict[int, dict[str, tuple[str, str]]]] = {}
        self._routed_expert_container: dict[int, str] = {}
        self._mtp: dict[str, tuple[str, str]] = {}
        for stored_key, filename in weight_map.items():
            entry = (str(self.model_dir / filename), stored_key)
            m = _LAYER_RE.match(stored_key)
            if m and not stored_key.startswith("mtp."):
                idx = int(m.group(1))
                leaf = m.group(2)
                em = _HF_EXPERT_LEAF_RE.match(leaf)
                if em:
                    exp_path, exp_id_s, proj_name = em.groups()
                    self._routed_expert_container[idx] = exp_path
                    self._routed_experts.setdefault(idx, {}).setdefault(int(exp_id_s), {})[
                        _HF_EXPERT_ROLE[proj_name]
                    ] = entry
                    continue
                self._layers.setdefault(idx, {})[f"model.layers.{idx}.{leaf}"] = entry
                continue
            if stored_key.startswith("mtp."):
                self._mtp[stored_key] = entry
                continue
            self._stored[stored_key] = entry

        for idx, exp_by_id in self._routed_experts.items():
            for exp_id, parts in exp_by_id.items():
                if set(parts) != {"gate", "up", "down"}:
                    raise UnsupportedModelError(
                        f"{self.model_dir}: layer {idx} expert {exp_id} misses projections ({sorted(parts)})"
                    )

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
        self._pread_reader = PreadSafetensorsReader(max_handles=8)
        self._expert_loader: _HFExpertCacheAndLoader | None = None

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
        """Cache identity of one decoder layer: tensor names, dtypes, shapes, byte ranges and file stamps."""
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

    def expert_cache_stats(self) -> dict[str, Any] | None:
        return self._expert_loader.stats() if self._expert_loader is not None else None

    def prepare_model(self, model: Any, device: Any = None, dtype: Any = None, expert_cache_bytes: int = 0) -> None:
        """Strips meta routed-expert parameters on HF MoE models and binds positioned-read expert routing."""
        import torch

        if not self._routed_experts:
            return
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
        if layers is None:
            raise UnsupportedModelError(f"{type(model).__name__} has no model.layers stack for HF MoE preparation")

        if self._expert_loader is not None:
            self._expert_loader.close()
        default_dtype = dtype or torch.bfloat16
        loader = _HFExpertCacheAndLoader(
            self._pread_reader,
            self._routed_experts,
            device=device,
            dtype=default_dtype,
            budget_bytes=int(expert_cache_bytes),
        )
        self._expert_loader = loader

        for layer_idx, exp_by_id in sorted(self._routed_experts.items()):
            exp_path = self._routed_expert_container[layer_idx]
            experts_mod = layers[layer_idx].get_submodule(exp_path)
            experts_mod._parameters.pop("gate_up_proj", None)
            experts_mod._parameters.pop("down_proj", None)
            for child in experts_mod.modules():
                if child is not experts_mod:
                    child._parameters.clear()

            num_exp_layer = len(exp_by_id)

            def _hf_sliced_experts_forward(
                self_exp,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
                _layer_idx=layer_idx,
                _num_exp=num_exp_layer,
            ) -> torch.Tensor:
                n_e = int(getattr(self_exp, "num_experts", _num_exp))
                return _fast_moe_dispatch(
                    self_exp,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    n_e,
                    lambda ids, dev, dt, evict: loader.load_batch(_layer_idx, ids, dev, dt, evict),
                )

            experts_mod.forward = types.MethodType(_hf_sliced_experts_forward, experts_mod)

    def close(self) -> None:
        if self._expert_loader is not None:
            self._expert_loader.close()
            self._expert_loader = None
        self._pread_reader.close()


# --------------------------------------------------------------------------------------
# Direct GGUF weight source (BF16 dense trunk ring + on-demand sliced routed experts)
# --------------------------------------------------------------------------------------


class _ShapeOnlySlice:
    """Lightweight slice descriptor exposing ``get_shape()`` without touching tensor bytes."""

    __slots__ = ("_shape",)

    def __init__(self, shape: tuple[int, ...]):
        self._shape = tuple(shape)

    def get_shape(self) -> tuple[int, ...]:
        return self._shape


class _GGUFRowSlice:
    """Chunked 2D row reader for large global GGUF matrices (such as ``lm_head.weight``)."""

    def __init__(self, source: "GGUFModelSource", gguf_name: str, dtype: Any):
        self._source = source
        self._gguf_name = gguf_name
        self._dtype = dtype
        self._shape = source.post_shape(gguf_name)

    def get_shape(self) -> tuple[int, ...]:
        return self._shape

    def __getitem__(self, item: slice):
        rows = self._shape[0]
        if isinstance(item, slice):
            start, stop, step = item.indices(rows)
            if step != 1:
                raise ValueError("GGUF row slices require unit step")
            return self._source.read_matrix_row_range(self._gguf_name, start, stop, dtype=self._dtype)
        raise TypeError(f"Unsupported GGUF slice index: {item!r}")


class _GGUFLayerView:
    """``safe_open``-compatible view of one GGUF decoder layer's dense trunk."""

    def __init__(self, source: "GGUFModelSource", layer_idx: int, dtype: Any):
        self._source = source
        self._entries = source.layer_entries(layer_idx)
        self._dtype = dtype

    def keys(self) -> list[str]:
        return list(self._entries)

    def get_slice(self, key: str) -> _ShapeOnlySlice:
        return _ShapeOnlySlice(self._source.tensor_shape(self._entries[key]))

    def get_tensor(self, key: str, device: Any = None):
        _, gguf_name = self._entries[key]
        return self._source.read_converted_tensor(gguf_name, dtype=self._dtype, device=device)


class _GGUFGlobalsMap(collections.abc.MutableMapping):
    """Lazy globals mapping over GGUF ``token_embd``, ``output_norm`` and ``output``."""

    def __init__(self, source: "GGUFModelSource", dtype: Any):
        self._source = source
        self._dtype = dtype
        self._entries = source.globals_entries()
        self._keys: dict[str, None] = dict.fromkeys(self._entries)
        self._materialized: dict[str, Any] = {}

    def release_handle(self) -> None:
        self._source.release_handle()

    def get_tensor_slice(self, key: str) -> _GGUFRowSlice:
        if key not in self._keys:
            raise KeyError(key)
        _, gguf_name = self._entries[key]
        return _GGUFRowSlice(self._source, gguf_name, dtype=self._dtype)

    def lookup_rows(self, key: str, indices: Any, dtype: Any = None):
        if key not in self._keys:
            raise KeyError(key)
        target_dtype = dtype or self._dtype
        if key in self._materialized:
            return self._materialized[key][indices].to(dtype=target_dtype)
        _, gguf_name = self._entries[key]
        return self._source.lookup_matrix_rows(gguf_name, indices, dtype=target_dtype)

    def _read_full(self, key: str):
        _, gguf_name = self._entries[key]
        return self._source.read_converted_tensor(gguf_name, dtype=self._dtype)

    def __getitem__(self, key: str):
        if key not in self._keys:
            raise KeyError(key)
        if key not in self._materialized:
            self._materialized[key] = self._read_full(key)
        return self._materialized[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._keys[key] = None
        self._materialized[key] = value

    def __delitem__(self, key: str) -> None:
        if key not in self._keys:
            raise KeyError(key)
        del self._keys[key]
        self._materialized.pop(key, None)

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        if key not in self._keys:
            if default is _MISSING:
                raise KeyError(key)
            return default
        tensor = self._materialized.pop(key) if key in self._materialized else self._read_full(key)
        self._keys.pop(key, None)
        return tensor

    def __iter__(self):
        return iter(list(self._keys))

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def clear(self) -> None:
        self._materialized.clear()
        self._keys.clear()
        self._source.release_handle()

    def close(self) -> None:
        self.clear()


def _apply_shape_ops(torch_shape: tuple[int, ...], operations: Iterable[Any]) -> tuple[int, ...]:
    """Computes the target PyTorch shape after 1-to-1 GGUF ``ConversionOps`` (without reading bytes)."""
    from transformers.integrations.gguf.gguf_conversion_mapping import Unsqueeze

    shape = tuple(torch_shape)
    for op in operations:
        if isinstance(op, Unsqueeze):
            if op.expected_ndim is not None and len(shape) >= op.expected_ndim:
                continue
            pos = op.dim if op.dim >= 0 else len(shape) + 1 + op.dim
            shape = shape[:pos] + (1,) + shape[pos:]
    return shape


class _GGUFExpertCacheAndLoader:
    """Contiguous ``uint8`` VRAM arena + 4-worker pinned NVMe reader + NVRTC dequantizer for GGUF MoE."""

    def __init__(
        self,
        gguf_path: Path,
        routed_experts: dict[int, dict[str, dict[str, Any]]],
        source: "GGUFModelSource",
        device: Any,
        budget_bytes: int,
    ):
        import torch

        self.gguf_str = str(gguf_path)
        self.source = source
        self.device = _canonical_device(device)
        self.hits = 0
        self.misses = 0

        self._layer_meta: dict[int, dict[str, Any]] = {}
        max_expert_bytes = 0
        total_experts = 0
        for l_idx, spec in routed_experts.items():
            g_inf, u_inf, d_inf = spec["gate"], spec["up"], spec["down"]
            n_exp = int(g_inf["torch_shape"][0])
            bg = int(g_inf["nbytes"]) // n_exp
            bu = int(u_inf["nbytes"]) // n_exp
            bd = int(d_inf["nbytes"]) // n_exp
            tot = bg + bu + bd
            max_expert_bytes = max(max_expert_bytes, tot)
            total_experts += n_exp
            self._layer_meta[l_idx] = {
                "num_experts": n_exp,
                "bg": bg,
                "bu": bu,
                "bd": bd,
                "tot": tot,
                "gate": g_inf,
                "up": u_inf,
                "down": d_inf,
            }

        self.max_expert_bytes = max_expert_bytes
        self._staging = _ParallelSliceStagingPool(
            slot_bytes=max(4096, _MAX_EXPERT_DEQUANT_BATCH * max_expert_bytes),
            device=self.device,
        )

        if self.device.type == "cuda" and max_expert_bytes > 0 and budget_bytes >= max_expert_bytes * _MAX_EXPERT_DEQUANT_BATCH:
            num_slots = min(total_experts, int(budget_bytes) // max_expert_bytes)
        else:
            num_slots = 0

        self.num_slots = num_slots
        self.arena = (
            torch.empty((num_slots, max_expert_bytes), dtype=torch.uint8, device=self.device)
            if num_slots > 0
            else None
        )
        self._lru: "OrderedDict[tuple[int, int], int]" = OrderedDict()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        arena_bytes = int(self.arena.numel()) if self.arena is not None else 0
        return {
            "slots": self.num_slots,
            "used_slots": len(self._lru),
            "arena_mib": round(arena_bytes / (1024 ** 2), 1),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(100.0 * self.hits / max(1, total), 1) if total > 0 else 0.0,
        }

    def _fetch_grouped_u8(self, meta: dict[str, Any], exp_ids: list[int]):
        """Reads ``[all_gates | all_ups | all_downs]`` so consecutive experts coalesce into sequential NVMe reads."""
        k = len(exp_ids)
        bg, bu, bd, tot = meta["bg"], meta["bu"], meta["bd"], meta["tot"]
        g_base = int(meta["gate"]["abs_offset"])
        u_base = int(meta["up"]["abs_offset"])
        d_base = int(meta["down"]["abs_offset"])
        u_dst0 = k * bg
        d_dst0 = k * (bg + bu)
        g_tasks = [(self.gguf_str, g_base + eid * bg, idx * bg, bg) for idx, eid in enumerate(exp_ids)]
        u_tasks = [(self.gguf_str, u_base + eid * bu, u_dst0 + idx * bu, bu) for idx, eid in enumerate(exp_ids)]
        d_tasks = [(self.gguf_str, d_base + eid * bd, d_dst0 + idx * bd, bd) for idx, eid in enumerate(exp_ids)]
        u8 = self._staging.fetch_to_device(g_tasks + u_tasks + d_tasks, k * tot)
        return (
            u8[:u_dst0].reshape(k, bg),
            u8[u_dst0:d_dst0].reshape(k, bu),
            u8[d_dst0 : k * tot].reshape(k, bd),
        )

    def _dequant_split(self, g_raw: Any, u_raw: Any, d_raw: Any, k: int, meta: dict[str, Any], dtype: Any):
        g_inf, u_inf, d_inf = meta["gate"], meta["up"], meta["down"]
        _, g1, g2 = g_inf["torch_shape"]
        _, u1, u2 = u_inf["torch_shape"]
        _, d1, d2 = d_inf["torch_shape"]
        gate_w = self.source._decode_u8_tensor(g_raw, int(g_inf["ggml_type"]), (k, g1, g2), dtype=dtype)
        up_w = self.source._decode_u8_tensor(u_raw, int(u_inf["ggml_type"]), (k, u1, u2), dtype=dtype)
        down_w = self.source._decode_u8_tensor(d_raw, int(d_inf["ggml_type"]), (k, d1, d2), dtype=dtype)
        return gate_w, up_w, down_w

    def _dequant_packed(self, packed_u8: Any, meta: dict[str, Any], dtype: Any):
        k = packed_u8.shape[0]
        bg, bu, bd = meta["bg"], meta["bu"], meta["bd"]
        g_raw = packed_u8[:, :bg].contiguous().reshape(-1)
        u_raw = packed_u8[:, bg : bg + bu].contiguous().reshape(-1)
        d_raw = packed_u8[:, bg + bu : bg + bu + bd].contiguous().reshape(-1)
        return self._dequant_split(g_raw, u_raw, d_raw, k, meta, dtype)

    def load_batch(self, layer_idx: int, batch_ids: list[int], dev: Any, comp_dtype: Any, allow_evict: bool):
        import torch

        dev = _canonical_device(dev)
        meta = self._layer_meta[layer_idx]
        tot = meta["tot"]
        k = len(batch_ids)
        if self.arena is None or dev != self.device:
            self.misses += k
            g2, u2, d2 = self._fetch_grouped_u8(meta, batch_ids)
            return self._dequant_split(
                g2.to(device=dev).reshape(-1),
                u2.to(device=dev).reshape(-1),
                d2.to(device=dev).reshape(-1),
                k,
                meta,
                comp_dtype,
            )

        miss_ids: list[int] = []
        seen_miss: set[int] = set()
        for eid in batch_ids:
            key = (layer_idx, eid)
            if key in self._lru:
                self.hits += 1
                if allow_evict:
                    self._lru.move_to_end(key)
            else:
                self.misses += 1
                if eid not in seen_miss:
                    seen_miss.add(eid)
                    miss_ids.append(eid)

        if not miss_ids:
            slots = torch.tensor([self._lru[(layer_idx, eid)] for eid in batch_ids], dtype=torch.long, device=dev)
            packed = self.arena.index_select(0, slots)[:, :tot]
            return self._dequant_packed(packed, meta, comp_dtype)

        can_admit = (allow_evict and self.num_slots >= len(batch_ids)) or (
            len(self._lru) + len(miss_ids) <= self.num_slots
        )
        mg, mu, md = self._fetch_grouped_u8(meta, miss_ids)

        if not can_admit and miss_ids == batch_ids:
            return self._dequant_split(mg.reshape(-1), mu.reshape(-1), md.reshape(-1), k, meta, comp_dtype)

        miss_packed = torch.cat([mg, mu, md], dim=1)

        if can_admit:
            target_slots: list[int] = []
            for eid in miss_ids:
                if len(self._lru) < self.num_slots:
                    s_idx = len(self._lru)
                else:
                    _, s_idx = self._lru.popitem(last=False)
                self._lru[(layer_idx, eid)] = s_idx
                target_slots.append(s_idx)
            slot_t = torch.tensor(target_slots, dtype=torch.long, device=dev)
            self.arena[:, :tot].index_copy_(0, slot_t, miss_packed)
            all_slots = torch.tensor([self._lru[(layer_idx, eid)] for eid in batch_ids], dtype=torch.long, device=dev)
            packed = self.arena.index_select(0, all_slots)[:, :tot]
            return self._dequant_packed(packed, meta, comp_dtype)

        miss_pos = {eid: idx for idx, eid in enumerate(miss_ids)}
        packed = torch.empty((k, tot), dtype=torch.uint8, device=dev)
        for i, eid in enumerate(batch_ids):
            key = (layer_idx, eid)
            if key in self._lru:
                packed[i].copy_(self.arena[self._lru[key], :tot])
            else:
                packed[i].copy_(miss_packed[miss_pos[eid]])
        return self._dequant_packed(packed, meta, comp_dtype)

    def close(self) -> None:
        self._staging.close()
        self._lru.clear()
        self.arena = None


class GGUFModelSource:
    """Positioned-read GGUF weight source for the unified BF16 ring engine.

    * Dense layer trunk tensors are mapped through ``transformers`` GGUF conversion ops into
      canonical parameter names and shapes for ``StableWeightManager`` / ``GPURingPipeline``.
    * Routed MoE expert tensors (``blk.{i}.ffn_{gate,up,down}_exps.weight``) stay excluded from
      ring slots; ``prepare_model`` strips the meta expert parameters and binds an on-demand
      forward pass that reads and NVRTC-dequantizes only the routed experts' byte slices (backed
      by a single-allocation VRAM hot-expert slot cache within the spare VRAM budget).
    * Never calls ``mmap`` and never writes files into ``model_dir``.
    """

    kind = "gguf"

    def __init__(self, model_dir: str | Path):
        from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
        from transformers.integrations.gguf.dequant import GGML_BLOCK
        from transformers.integrations.gguf.utils import get_gguf_conversion_mapping

        self.model_dir = Path(model_dir).resolve()
        if detect_format(self.model_dir) != "gguf":
            raise UnsupportedModelError(f"{self.model_dir}: not a GGUF model folder")
        self.gguf_path = find_gguf_file(self.model_dir)
        self.gguf_file = self.gguf_path.name
        self.arch = _validate_gguf_header(self.gguf_path)
        meta, raw_tensors = _read_gguf_header_cached(self.gguf_path)
        self.metadata = meta

        for t in raw_tensors:
            gt = int(t["ggml_type"])
            if self._raw_torch_dtype(gt) is None and gt not in GGML_BLOCK and gt not in _EXTRA_GGML_BLOCKS:
                raise UnsupportedModelError(
                    f"{self.gguf_path}: tensor {t['name']!r} uses unsupported ggml quantization {t['type_name']} ({gt})"
                )

        cfg_obj = load_runtime_config(self.model_dir)
        text_cfg = cfg_obj.get_text_config() if hasattr(cfg_obj, "get_text_config") else cfg_obj
        cfg_dict = text_cfg.to_dict()

        mapping = deepcopy(get_gguf_conversion_mapping(self.arch, text_cfg))
        renamings = [e for e in mapping if isinstance(e, WeightRenaming)]
        converters = [e for e in mapping if isinstance(e, WeightConverter)]
        pattern_to_converter = {pat: conv for conv in converters for pat in conv.source_patterns}

        self._tensor_info: dict[str, dict[str, Any]] = {}
        self._canonical_by_gguf: dict[str, str] = {}
        self._converter_by_gguf: dict[str, tuple[Any, str] | None] = {}
        self._post_shapes: dict[str, tuple[int, ...]] = {}
        self._layers: dict[int, dict[str, tuple[str, str]]] = {}
        self._routed_experts: dict[int, dict[str, dict[str, Any]]] = {}
        self._stored: dict[str, tuple[str, str]] = {}
        self._mtp: dict[str, tuple[str, str]] = {}

        gguf_str = str(self.gguf_path)
        for t in raw_tensors:
            gguf_name = t["name"]
            torch_shape = tuple(int(d) for d in reversed(t["dims"]))
            info = dict(t)
            info["torch_shape"] = torch_shape
            self._tensor_info[gguf_name] = info

            canonical, src_pat = rename_source_key(gguf_name, renamings, converters)
            conv = pattern_to_converter.get(src_pat) if src_pat is not None else None
            self._canonical_by_gguf[gguf_name] = canonical
            self._converter_by_gguf[gguf_name] = (conv, src_pat) if conv is not None else None

            m = _LAYER_RE.match(canonical)
            if m and not canonical.startswith("mtp."):
                idx = int(m.group(1))
                leaf = m.group(2)
                if leaf == "mlp.experts.gate_up_proj":
                    slot = "gate" if "gate_exps" in gguf_name else "up"
                    self._routed_experts.setdefault(idx, {})[slot] = info
                    continue
                if leaf == "mlp.experts.down_proj":
                    self._routed_experts.setdefault(idx, {})["down"] = info
                    continue
                ops = getattr(conv, "operations", ()) if conv is not None else ()
                self._post_shapes[gguf_name] = _apply_shape_ops(torch_shape, ops)
                self._layers.setdefault(idx, {})[f"model.layers.{idx}.{leaf}"] = (gguf_str, gguf_name)
                continue

            ops = getattr(conv, "operations", ()) if conv is not None else ()
            self._post_shapes[gguf_name] = _apply_shape_ops(torch_shape, ops)
            self._stored[canonical] = (gguf_str, gguf_name)

        for idx, exp_parts in self._routed_experts.items():
            if set(exp_parts) != {"gate", "up", "down"}:
                raise UnsupportedModelError(
                    f"{self.gguf_path}: layer {idx} has incomplete routed expert tensors {sorted(exp_parts)}"
                )

        self._globals: dict[str, tuple[str, str]] = {}
        for canonical, aliases in _GLOBAL_ALIASES.items():
            for alias in aliases:
                if alias in self._stored:
                    self._globals[canonical] = self._stored[alias]
                    break
        if "lm_head.weight" not in self._globals and "model.embed_tokens.weight" in self._globals:
            if cfg_dict.get("tie_word_embeddings", True):
                self._globals["lm_head.weight"] = self._globals["model.embed_tokens.weight"]
        missing = [k for k in _GLOBAL_ALIASES if k not in self._globals]
        if missing:
            raise UnsupportedModelError(f"{self.gguf_path}: GGUF misses global tensors {missing}")

        self.num_layers = int(cfg_dict.get("num_hidden_layers") or len(self._layers))
        absent = [i for i in range(self.num_layers) if i not in self._layers]
        if absent:
            raise UnsupportedModelError(f"{self.gguf_path}: GGUF misses decoder layers {absent[:8]}")

        self._io_lock = threading.Lock()
        self._file_handle: Any = None
        self._preferred_device: Any = None
        self._expert_loader: _GGUFExpertCacheAndLoader | None = None

    # -- file I/O (positioned reads, zero mmap) ----------------------------------------

    def _ensure_file(self):
        if self._file_handle is None or self._file_handle.closed:
            self._file_handle = open(self.gguf_path, "rb", buffering=0)
        return self._file_handle

    def _read_into(self, abs_offset: int, buf: bytearray | memoryview) -> None:
        view = buf if isinstance(buf, memoryview) else memoryview(buf)
        if len(view) == 0:
            return
        with self._io_lock:
            f = self._ensure_file()
            _pread_or_seek_into(f, abs_offset, view, str(self.gguf_path))

    def release_handle(self) -> None:
        with self._io_lock:
            if self._file_handle is not None:
                try:
                    self._file_handle.close()
                except OSError:
                    pass
                self._file_handle = None

    def close(self) -> None:
        if self._expert_loader is not None:
            self._expert_loader.close()
            self._expert_loader = None
        self.release_handle()

    # -- model_source contract ---------------------------------------------------------

    def layer_entries(self, layer_idx: int) -> dict[str, tuple[str, str]]:
        return dict(self._layers[int(layer_idx)])

    def globals_entries(self) -> dict[str, tuple[str, str]]:
        return dict(self._globals)

    def mtp_entries(self) -> dict[str, tuple[str, str]]:
        return dict(self._mtp)

    def has_mtp(self) -> bool:
        return False

    def post_shape(self, gguf_name: str) -> tuple[int, ...]:
        return self._post_shapes[gguf_name]

    def tensor_shape(self, entry: tuple[str, str]) -> tuple[int, ...]:
        _, gguf_name = entry
        return self._post_shapes[gguf_name]

    def expert_cache_stats(self) -> dict[str, Any] | None:
        return self._expert_loader.stats() if self._expert_loader is not None else None

    def layer_identity(self, layer_idx: int) -> dict[str, Any]:
        tensors: dict[str, Any] = {}
        for name, (_, gguf_name) in sorted(self.layer_entries(layer_idx).items()):
            info = self._tensor_info[gguf_name]
            tensors[name] = {
                "shape": list(self._post_shapes[gguf_name]),
                "ggml_type": int(info["ggml_type"]),
                "offset": int(info["abs_offset"]),
                "nbytes": int(info["nbytes"]),
            }
        st = os.stat(self.gguf_path)
        payload = {"file": self.gguf_file, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "tensors": tensors}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return {"sha256": digest, "tensors": {k: {"shape": v["shape"]} for k, v in tensors.items()}}

    def open_layer(self, layer_idx: int, dtype: Any = None) -> _GGUFLayerView:
        import torch

        return _GGUFLayerView(self, layer_idx, dtype=dtype or torch.bfloat16)

    def open_globals(self, dtype: Any = None) -> _GGUFGlobalsMap:
        import torch

        return _GGUFGlobalsMap(self, dtype=dtype or torch.bfloat16)

    # -- tensor unpacking & conversion -------------------------------------------------

    @staticmethod
    def _raw_torch_dtype(ggml_type: int):
        import torch

        return {
            GGML_TYPE_F32: torch.float32,
            GGML_TYPE_F16: torch.float16,
            GGML_TYPE_BF16: torch.bfloat16,
            GGML_TYPE_F64: torch.float64,
            GGML_TYPE_I8: torch.int8,
            GGML_TYPE_I16: torch.int16,
            GGML_TYPE_I32: torch.int32,
            GGML_TYPE_I64: torch.int64,
        }.get(ggml_type)

    def _decode_u8_tensor(self, u8: Any, ggml_type: int, shape: tuple[int, ...], dtype: Any):
        raw_dt = self._raw_torch_dtype(ggml_type)
        if raw_dt is not None:
            return u8.view(raw_dt).reshape(shape).to(dtype=dtype)
        return dequantize_gguf_tensor(u8, ggml_type, dtype=dtype).reshape(shape)

    def _decode_bytes(
        self,
        buf: bytearray,
        ggml_type: int,
        shape: tuple[int, ...],
        dtype: Any,
        device: Any = None,
    ):
        import torch

        raw_dt = self._raw_torch_dtype(ggml_type)
        target_dev = torch.device(device) if device is not None else torch.device("cpu")
        if raw_dt is not None:
            t = torch.frombuffer(buf, dtype=raw_dt) if buf else torch.empty(0, dtype=raw_dt)
            return t.reshape(shape).to(device=target_dev, dtype=dtype)
        u8 = torch.frombuffer(buf, dtype=torch.uint8) if buf else torch.empty(0, dtype=torch.uint8)
        if target_dev.type == "cuda":
            u8 = u8.to(device=target_dev)
        return dequantize_gguf_tensor(u8, ggml_type, dtype=dtype).reshape(shape)

    def read_converted_tensor(self, gguf_name: str, dtype: Any = None, device: Any = None):
        import torch
        from transformers.integrations.gguf.gguf_conversion_mapping import LogNegate, SubtractOne

        target_dtype = dtype or torch.bfloat16
        info = self._tensor_info[gguf_name]
        conv_entry = self._converter_by_gguf.get(gguf_name)
        ops = getattr(conv_entry[0], "operations", ()) if conv_entry is not None else ()
        needs_fp32 = any(isinstance(op, (SubtractOne, LogNegate)) for op in ops)
        work_dtype = torch.float32 if needs_fp32 else target_dtype

        buf = bytearray(int(info["nbytes"]))
        self._read_into(int(info["abs_offset"]), buf)
        tensor = self._decode_bytes(buf, int(info["ggml_type"]), info["torch_shape"], dtype=work_dtype, device=device)

        if conv_entry is not None:
            conv, src_pat = conv_entry
            canonical = self._canonical_by_gguf[gguf_name]
            cur = {src_pat: tensor}
            for op in conv.operations:
                cur = op.convert(
                    cur,
                    source_patterns=conv.source_patterns,
                    target_patterns=conv.target_patterns,
                    full_layer_name=canonical,
                )
            tensor = next(iter(cur.values()))

        if tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        if device is not None and tensor.device != torch.device(device):
            tensor = tensor.to(device=device)
        return tensor.contiguous()

    def _row_byte_stride(self, info: dict[str, Any]) -> int:
        from transformers.integrations.gguf.dequant import GGML_BLOCK, row_bytes

        shape = info["torch_shape"]
        if len(shape) != 2:
            raise ValueError(f"Row slicing requires a 2D tensor, got {shape}")
        gt = int(info["ggml_type"])
        raw_dt = self._raw_torch_dtype(gt)
        if raw_dt is not None:
            return int(shape[1]) * raw_dt.itemsize
        if gt in GGML_BLOCK:
            return row_bytes(gt, int(shape[1]))
        if gt in _EXTRA_GGML_BLOCKS:
            _, block_elems, block_bytes = _EXTRA_GGML_BLOCKS[gt]
            return (int(shape[1]) // block_elems) * block_bytes
        raise UnsupportedModelError(f"Unsupported row-slice ggml type {gt}")

    def read_matrix_row_range(self, gguf_name: str, r0: int, r1: int, dtype: Any = None):
        import torch

        target_dtype = dtype or torch.bfloat16
        info = self._tensor_info[gguf_name]
        rows, cols = info["torch_shape"]
        r0 = max(0, min(int(r0), rows))
        r1 = max(r0, min(int(r1), rows))
        count = r1 - r0
        dev = self._preferred_device or torch.device("cpu")
        if count == 0:
            return torch.empty((0, cols), dtype=target_dtype, device=dev)
        stride = self._row_byte_stride(info)
        buf = bytearray(count * stride)
        self._read_into(int(info["abs_offset"]) + r0 * stride, buf)
        return self._decode_bytes(buf, int(info["ggml_type"]), (count, cols), dtype=target_dtype, device=dev)

    def lookup_matrix_rows(self, gguf_name: str, indices: Any, dtype: Any = None):
        import torch

        target_dtype = dtype or torch.bfloat16
        info = self._tensor_info[gguf_name]
        _, cols = info["torch_shape"]
        flat_cpu = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if flat_cpu.numel() == 0:
            return torch.empty((*indices.shape, cols), dtype=target_dtype)

        uniq_ids, inverse = torch.unique(flat_cpu, sorted=True, return_inverse=True)
        uniq_list = uniq_ids.tolist()
        stride = self._row_byte_stride(info)
        base = int(info["abs_offset"])
        buf = bytearray(len(uniq_list) * stride)
        view = memoryview(buf)

        # Coalesce contiguous token IDs into single positioned reads.
        pos = 0
        n = len(uniq_list)
        while pos < n:
            end = pos + 1
            while end < n and uniq_list[end] == uniq_list[end - 1] + 1:
                end += 1
            run_len = end - pos
            self._read_into(base + int(uniq_list[pos]) * stride, view[pos * stride : (pos + run_len) * stride])
            pos = end

        dev = self._preferred_device or torch.device("cpu")
        uniq_rows = self._decode_bytes(buf, int(info["ggml_type"]), (n, cols), dtype=target_dtype, device=dev)
        gathered = uniq_rows.index_select(0, inverse.to(device=uniq_rows.device))
        return gathered.reshape(*indices.shape, cols)

    # -- on-demand sliced routed MoE experts -------------------------------------------

    def _read_expert_batch(self, info: dict[str, Any], expert_ids: list[int], dtype: Any, device: Any):
        num_experts, d1, d2 = info["torch_shape"]
        exp_bytes = int(info["nbytes"]) // int(num_experts)
        base = int(info["abs_offset"])
        k = len(expert_ids)
        buf = bytearray(k * exp_bytes)
        view = memoryview(buf)

        pos = 0
        while pos < k:
            end = pos + 1
            while end < k and expert_ids[end] == expert_ids[end - 1] + 1:
                end += 1
            run_len = end - pos
            self._read_into(
                base + int(expert_ids[pos]) * exp_bytes,
                view[pos * exp_bytes : (pos + run_len) * exp_bytes],
            )
            pos = end

        return self._decode_bytes(buf, int(info["ggml_type"]), (k, d1, d2), dtype=dtype, device=device)

    def prepare_model(self, model: Any, device: Any = None, dtype: Any = None, expert_cache_bytes: int = 0) -> None:
        """Strips meta routed-expert parameters and binds sliced GGUF expert execution."""
        import torch

        if device is not None:
            self._preferred_device = torch.device(device)
        if not self._routed_experts:
            return

        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
        if layers is None:
            raise UnsupportedModelError(f"{type(model).__name__} has no model.layers stack for GGUF MoE preparation")

        if self._expert_loader is not None:
            self._expert_loader.close()
        loader = _GGUFExpertCacheAndLoader(
            self.gguf_path,
            self._routed_experts,
            source=self,
            device=device,
            budget_bytes=int(expert_cache_bytes),
        )
        self._expert_loader = loader

        for layer_idx, exp_spec in sorted(self._routed_experts.items()):
            layer_mod = layers[layer_idx]
            mlp_mod = getattr(layer_mod, "mlp", None)
            experts_mod = getattr(mlp_mod, "experts", None) if mlp_mod is not None else None
            if experts_mod is None:
                raise UnsupportedModelError(f"Layer {layer_idx} has no mlp.experts module for GGUF routed experts")

            # Remove meta parameters so neither _load_resident_layer nor ring hooks expect them in slots.
            experts_mod._parameters.pop("gate_up_proj", None)
            experts_mod._parameters.pop("down_proj", None)

            num_exp_layer = int(exp_spec["gate"]["torch_shape"][0])

            def _sliced_experts_forward(
                self_exp,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
                _layer_idx=layer_idx,
                _num_exp=num_exp_layer,
            ) -> torch.Tensor:
                n_e = int(getattr(self_exp, "num_experts", _num_exp))
                return _fast_moe_dispatch(
                    self_exp,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    n_e,
                    lambda ids, dev, dt, evict: loader.load_batch(_layer_idx, ids, dev, dt, evict),
                )

            experts_mod.forward = types.MethodType(_sliced_experts_forward, experts_mod)


def open_model_source(model_dir: str | Path):
    """Returns ``HFShardedSource`` or ``GGUFModelSource``; ``None`` for Colibri containers."""
    fmt = detect_format(model_dir)
    if fmt == "hf":
        return HFShardedSource(model_dir)
    if fmt == "gguf":
        return GGUFModelSource(model_dir)
    if fmt == "colibri":
        return None
    raise UnsupportedModelError(f"{model_dir}: unrecognised model folder")
