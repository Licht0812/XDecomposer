"""Run output helpers."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict


_STAMP_RE = re.compile(r"(\d{8}_\d{6})$")


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_timestamp(path: str) -> str | None:
    leaf = os.path.basename(os.path.normpath(path))
    match = _STAMP_RE.search(leaf)
    return match.group(1) if match else None


def ensure_timestamp_dir(path: str) -> tuple[str, str]:
    stamp = extract_timestamp(path)
    if stamp:
        os.makedirs(path, exist_ok=True)
        return path, stamp

    stamp = current_timestamp()
    stamped_path = os.path.join(path, stamp)
    os.makedirs(stamped_path, exist_ok=True)
    return stamped_path, stamp


def with_run_timestamp(payload: Dict[str, Any], stamp: str) -> Dict[str, Any]:
    data = dict(payload)
    data["run_timestamp"] = stamp
    return data
