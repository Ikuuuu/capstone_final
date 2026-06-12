"""YAML 설정 파일 로더 (defaults 상속 지원)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """딕셔너리 깊은 병합 (override 우선)."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    """defaults 상속을 처리하면서 YAML 설정 로드."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    defaults = cfg.pop("defaults", []) if isinstance(cfg, dict) else []
    merged: Dict[str, Any] = {}
    for d in defaults:
        if isinstance(d, str):
            parent_path = path.parent / f"{d}.yaml"
            merged = _merge(merged, load_config(parent_path))
        elif isinstance(d, dict):
            for k, v in d.items():
                parent_path = path.parent / f"{v}.yaml"
                merged = _merge(merged, load_config(parent_path))

    return _merge(merged, cfg)


def get(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """'a.b.c' 형식의 키로 중첩 설정 접근."""
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
