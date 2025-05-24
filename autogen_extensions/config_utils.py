import json
from typing import Any, Dict
import os

ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON config file and return as a dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {path}: {e}")


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save a dict as a JSON config file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save config to {path}: {e}")
