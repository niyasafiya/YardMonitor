"""Load and validate config.yaml."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG: Dict[str, Any] | None = None
_CONFIG_PATH = Path(os.getenv("YM_CONFIG", "config.yaml"))


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration from YAML file. Cached after first call."""
    global _CONFIG
    if _CONFIG is not None and path is None:
        return _CONFIG

    cfg_path = Path(path) if path else _CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. "
            f"Set YM_CONFIG env var or place config.yaml in project root."
        )

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Make sure required keys exist
    for required in ("system", "sources", "models", "thresholds", "gate", "storage", "web"):
        if required not in cfg:
            raise ValueError(f"config.yaml missing required section: {required}")

    _CONFIG = cfg
    return cfg


def get(*keys: str, default: Any = None) -> Any:
    """Dotted-style lookup: get('thresholds', 'vehicle_conf')."""
    cfg = load_config()
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
