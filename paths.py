"""Artifact locations. Override with SPECTRAL_BIAS_ROOT if needed."""

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_SCRATCH = Path("/pscratch/sd/c/cainslie/spectral_bias")


def _root() -> Path:
    override = os.environ.get("SPECTRAL_BIAS_ROOT")
    if override:
        return Path(override)
    if _DEFAULT_SCRATCH.parent.exists():
        return _DEFAULT_SCRATCH
    return SCRIPT_DIR


ROOT = _root()
SAVE_ROOT = ROOT / "saved_model_dicts"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
