"""
Configuration Loading Utility for OrchestraNet.

Loads YAML config files and merges with CLI arguments.
Supports nested config access via dot notation.
"""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """
    Hierarchical configuration container with dot-notation access.

    Usage:
        cfg = Config.from_yaml("configs/base_config.yaml")
        cfg.training.batch_size  # 16
        cfg.backbone.name  # "mobilenetv4_hybrid"

        # Override from CLI args
        cfg.merge_args(args)
    """

    def __init__(self, data: dict | None = None):
        self._data = data or {}
        for k, v in self._data.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file."""
        path = Path(path)
        if not path.exists():
            print(f"⚠️  Config file not found: {path}, using defaults")
            return cls({})
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create config from a dict."""
        return cls(d)

    def merge_args(self, args) -> "Config":
        """
        Merge argparse namespace into config. CLI args take precedence.
        Only overrides if the arg value is not None.
        """
        if args is None:
            return self
        for k, v in vars(args).items():
            if v is not None:
                setattr(self, k, v)
                self._data[k] = v
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by dot-notation key (e.g. 'training.batch_size')."""
        keys = key.split(".")
        obj = self
        for k in keys:
            if isinstance(obj, Config):
                obj = getattr(obj, k, None)
            elif isinstance(obj, dict):
                obj = obj.get(k, None)
            else:
                return default
            if obj is None:
                return default
        return obj

    def to_dict(self) -> dict:
        """Convert back to a flat dict."""
        result = {}
        for k, v in self._data.items():
            if isinstance(v, dict):
                result[k] = Config(v).to_dict()
            else:
                result[k] = v
        return result

    def __repr__(self) -> str:
        return f"Config({self._data})"

    def __contains__(self, key: str) -> bool:
        return key in self._data
