# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""ReGRPO reference package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

__version__ = "0.1.0"


def try_import_mat() -> ModuleType | None:
    """Return the optional MAT-Agent module, or None when unavailable.

    MAT-Agent is an external ReAct tool-use harness used only for paper-style
    GTA/GAIA evaluation and deployment; it is not bundled in this training
    release. When a ``MAT-Agent/`` checkout is placed next to this package it
    is added to ``sys.path`` and imported lazily. Import failures are
    intentionally swallowed (e.g. transformers >= 5 removed the legacy
    ``transformers.agents`` module MAT-Agent relies on) so the training core
    stays importable on its own.
    """

    root = Path(__file__).resolve().parents[1]
    mat_path = root / "MAT-Agent"
    if mat_path.is_dir():
        mat_path_str = str(mat_path)
        if mat_path_str not in sys.path:
            sys.path.insert(0, mat_path_str)
    try:
        return importlib.import_module("tongagent")
    except Exception:
        return None
