"""Lightweight CLI contracts; listing/help must not import Torch or start CUDA."""
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VRAM_MODES = ("normal", "turbo")


def find_aircolibri_root(explicit_root=None):
    candidates = [Path(explicit_root)] if explicit_root else []
    if os.environ.get("AIRCOLIBRI_ROOT"):
        candidates.append(Path(os.environ["AIRCOLIBRI_ROOT"]))
    candidates.append(PROJECT_ROOT)
    for candidate in candidates:
        if (candidate/"src"/"aircolibri_integrated.py").exists():
            resolved = candidate.resolve()
            if str(resolved) not in sys.path:
                sys.path.insert(0, str(resolved))
            return resolved
    raise FileNotFoundError("Could not locate the AirColibri project root. Pass --aircolibri-root or set AIRCOLIBRI_ROOT.")


def checkpoint_manifest_path(root):
    return root/"data"/"trusted-checkpoint-manifest.json"


class ContextBudgetExceeded(ValueError):
    """Request exceeds the context capacity reserved by the current VRAM plan."""


class GenerationCancelled(RuntimeError):
    _is_cooperative_cancel = True

    def __init__(self, phase, reason="cancelled"):
        super().__init__(f"generation cancelled during {phase}: {reason}")
        self.phase, self.reason = phase, reason
