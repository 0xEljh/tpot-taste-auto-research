"""PEFT adapter metadata helpers shared by training, CLI, and serving."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_BASE = "Qwen/Qwen2.5-3B-Instruct"


def adapter_config_path(adapter_path: Path | str) -> Path:
    return Path(adapter_path) / "adapter_config.json"


def read_adapter_config(adapter_path: Path | str) -> dict[str, Any]:
    cfg = adapter_config_path(adapter_path)
    return json.loads(cfg.read_text())


def adapter_recorded_base(adapter_path: Path | str) -> str | None:
    cfg = adapter_config_path(adapter_path)
    if not cfg.exists():
        return None
    base = read_adapter_config(adapter_path).get("base_model_name_or_path")
    return str(base) if base else None


def resolve_adapter_base(adapter_path: Path | str, fallback: str = DEFAULT_BASE) -> str:
    """Resolve an adapter's recorded base, falling back only for legacy adapters."""
    return adapter_recorded_base(adapter_path) or fallback


def require_matching_adapter_base(adapter_path: Path | str, requested_base: str) -> str:
    """Fail fast when a caller tries to load an adapter on the wrong base."""
    recorded = adapter_recorded_base(adapter_path)
    if recorded and recorded != requested_base:
        raise ValueError(
            f"requested base {requested_base!r} does not match adapter base {recorded!r} "
            f"from {adapter_config_path(adapter_path)}"
        )
    return requested_base
