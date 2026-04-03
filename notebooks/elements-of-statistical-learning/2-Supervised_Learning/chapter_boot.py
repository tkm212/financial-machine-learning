"""Locate and load ``helpers.py`` in this folder (for notebooks)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_helpers():
    here = Path(__file__).resolve().parent
    h = here / "helpers.py"
    spec = importlib.util.spec_from_file_location("esl_ch2_helpers", h)
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("spec.loader")
    spec.loader.exec_module(mod)
    return mod
