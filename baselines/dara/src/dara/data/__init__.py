"""Data files for DARA."""
from __future__ import annotations

from pathlib import Path

from monty.serialization import loadfn

cwd = Path(__file__).parent.resolve()

COMMON_GASES: list[str] = loadfn(cwd / "common_gases.json")
