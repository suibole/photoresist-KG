
"""Configuration loading helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml


LOGGER = logging.getLogger(__name__)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file and return its dictionary."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {cfg_path} must be a mapping, got {type(config)!r}")
    LOGGER.debug("Loaded configuration from %s", cfg_path)
    return config


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create the directory if missing and return it as Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
