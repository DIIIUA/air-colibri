"""Persistent Q4 layer cache, integrated from the preserved experiments.
Cache reads trust an EXTERNAL checkpoint manifest, but re-check size/mtime.
Cache misses verify the original file SHA256 before converting the weights.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import contextmanager
from uuid import uuid4

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from src import quantization_q4 as q4

FORMAT = 'aircolibri-prepacked-q4-layer-v1'
MANIFEST_FORMAT = 'aircolibri-checkpoint-sha256-v1'

class CheckpointChangedError(ValueError):
    pass

class CacheValidationError(ValueError):
    pass

def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def sha_tensor(t):
    if t.device.type != 'cpu':
        raise ValueError('Cache accepts CPU tensors only')
    a = t.detach().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(a).cast('B')).hexdigest()

def sha_object(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()

@contextmanager
def exclusive_lock(path, timeout=30.0):
    """Release file lock on process crash (Windows msvcrt; POSIX fcntl)."""
    f = open(path, 'a+b')
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            if os.fstat(f.fileno()).st_size == 0:
                f.write(b'0')
                f.flush()
        else:
            import fcntl
        started = time.monotonic()
        while not acquired:
            try:
                f.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (OSError, BlockingIOError):
                if time.monotonic() - started > timeout:
                    raise TimeoutError('Q4 layer lock timeout')
                time.sleep(.03)
        yield
    finally:
        if acquired:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()

class PrepackedLayerCache:
    def __init__(self, model_dir, cache_dir, manifest_path, dtype=torch.bfloat16,
                 group_size=64, max_bytes=0, reader=None):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if self.manifest.get('format') != MANIFEST_FORMAT:
            raise ValueError('Unsupported checkpoint manifest format')
        self.dtype = dtype
        self.group_size = int(group_size)
        self.max_bytes = int(max_bytes)
        self.reader = reader
        if self.group_size not in (16, 32, 64) or self.max_bytes < 0:
            raise ValueError('Invalid Q4 cache parameters')
        self.quantizer_sha = sha_file(q4.__file__)
        self.stats = dict(hits=0, misses=0, builds=0,
                          corruption=0, writes=0, quota_bypass=0)

    def source_entry(self, idx):
        idx = int(idx)
        ent = self.manifest['layers'][str(idx)]
        filename = 'model-%05d.safetensors' % idx
        if ent['filename'] != filename:
            raise ValueError('Wrong layer filename in manifest')
        path = self.model_dir / filename
        st = path.stat()
        if st.st_size != ent['size_bytes'] or st.st_mtime_ns != ent['mtime_ns']:
            raise CheckpointChangedError('Source file differs from trusted manifest')
        if len(ent.get('sha256', '')) != 64:
            raise ValueError('Malformed checkpoint SHA256')
        return path, ent

    def specs(self, idx, ent):
        prefix = 'model.layers.%d.' % idx
        result = {}
        for name, meta in ent['tensors'].items():
            if not name.startswith(prefix):
                raise ValueError('Foreign tensor key in manifest')
            short = name[len(prefix):]
            shape = tuple(meta['shape'])
            if q4.is_q4_quantizable(short, shape, self.group_size):
                result.update(q4.q4_tensor_specs(short, shape,
                    self.group_size, self.dtype))
            else:
                result[short] = (shape, self.dtype)
        if not result:
            raise ValueError('Manifest has no layer tensors')
        return result

    def cache_coordinates(self, idx):
        _, ent = self.source_entry(idx)
        specs = self.specs(idx, ent)
        meta = dict(format=FORMAT, layer=str(idx),
            source_sha=ent['sha256'], quantizer_sha=self.quantizer_sha,
            group_size=str(self.group_size), dtype=str(self.dtype),
            specs_sha=sha_object({k:(list(s),str(t)) for k,(s,t) in specs.items()}))
        name = 'layer-%05d-%s.safetensors' % (idx, sha_object(meta)[:24])
        return self.cache_dir / name, meta, specs

    def _load(self, path, meta, specs):
        if not path.is_file():
            return None
        result = {}
        with safe_open(str(path), framework='pt', device='cpu') as f:
            md = f.metadata() or {}
            if any(md.get(k) != v for k,v in meta.items()):
                raise CacheValidationError('Cache identity mismatch')
            if set(f.keys()) != set(specs):
                raise CacheValidationError('Cached tensor keys mismatch')
            payload = self.reader(str(path.parent), path.stem, list(specs)) if self.reader else None
            for k, (shape, dtype) in specs.items():
                t = payload[k] if payload is not None else f.get_tensor(k)
                if tuple(t.shape) != shape or t.dtype != dtype:
                    raise CacheValidationError('Cached shape/dtype mismatch')
                if md.get('sha256:' + k) != sha_tensor(t):
                    raise CacheValidationError('Cached payload checksum mismatch')
                result[k] = t
        return result

    def _try(self, path, meta, specs):
        try:
            result = self._load(path, meta, specs)
        except (OSError, RuntimeError, ValueError):
            self.stats['corruption'] += 1
            return None
        if result is not None:
            self.stats['hits'] += 1
        return result

    def get_or_build(self, idx, builder):
        idx = int(idx)
        source, ent = self.source_entry(idx)
        path, meta, specs = self.cache_coordinates(idx)
        data = self._try(path, meta, specs)
        if data is not None:
            return data
        self.stats['misses'] += 1
        with exclusive_lock(path.with_suffix('.lock')):
            self.source_entry(idx)
            data = self._try(path, meta, specs)
            if data is not None:
                return data
            if sha_file(source) != ent['sha256']:
                raise CheckpointChangedError('Original checksum mismatch')
            data = builder()
            if set(data) != set(specs):
                raise CacheValidationError('Producer keys mismatch')
            for k, (shape, dtype) in specs.items():
                t = data[k]
                if tuple(t.shape) != shape or t.dtype != dtype or t.device.type != 'cpu':
                    raise CacheValidationError('Producer tensor shape/dtype/device mismatch')
            self.source_entry(idx)
            self.stats['builds'] += 1
            payload = {k:v.contiguous() for k,v in data.items()}
            md = dict(meta)
            md.update({'sha256:'+k:sha_tensor(v) for k,v in payload.items()})
            tmp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
            try:
                save_file(payload, str(tmp), metadata=md)
                # Quota must be serialized ACROSS different layer keys, not
                # merely inside each per-layer lock. Otherwise concurrent
                # writers can each see stale free space and oversubscribe.
                with exclusive_lock(self.cache_dir / '.global-budget.lock'):
                    existing = sum(p.stat().st_size for p in self.cache_dir.glob('*.safetensors'))
                    if self.max_bytes and existing + tmp.stat().st_size > self.max_bytes:
                        self.stats['quota_bypass'] += 1
                        return data
                    os.replace(tmp, path)
                    self.stats['writes'] += 1
            finally:
                tmp.unlink(missing_ok=True)
            return data
