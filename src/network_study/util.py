from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dumps(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def stable_hash(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    elif isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = json_dumps(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json_dumps(value, pretty=pretty))
        fh.write("\n")


def append_jsonl(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json_dumps(value))
        fh.write("\n")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def get_path(obj: Any, paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        current = obj
        ok = True
        for segment in path.split("."):
            if isinstance(current, dict) and segment in current:
                current = current[segment]
            else:
                ok = False
                break
        if ok and current not in (None, ""):
            return current
    return default


def first_present(obj: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_timestamp(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        # Moltbook sometimes exposes millisecond timestamps.
        seconds = value / 1000.0 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc).isoformat()
    except ValueError:
        return str(value)


def slugify(value: str, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return text or fallback


def gini(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0
    n = len(vals)
    weighted = sum((idx + 1) * val for idx, val in enumerate(vals))
    return (2 * weighted / (n * total)) - ((n + 1) / n)


def top_share(values: Iterable[float], fraction: float) -> float:
    vals = sorted((float(v) for v in values if math.isfinite(float(v))), reverse=True)
    if not vals:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0
    k = max(1, math.ceil(len(vals) * fraction))
    return sum(vals[:k]) / total
