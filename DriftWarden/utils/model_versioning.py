"""
Track deployed model versions for before/after demo storytelling.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

VERSION_PATH = Path(__file__).resolve().parent.parent / "models" / "version_registry.json"


def _load() -> dict:
    if not VERSION_PATH.exists():
        return {
            "current": "v1.0",
            "history": [],
        }
    with open(VERSION_PATH) as f:
        return json.load(f)


def _save(data: dict) -> None:
    VERSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VERSION_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_current_version() -> str:
    return _load().get("current", "v1.0")


def register_version(f1: float, promoted: bool, note: str = "") -> dict:
    """
    Bump version on promotion (v1.2 -> v1.3 style).
    Returns old and new version labels for UI.
    """
    data = _load()
    old_version = data.get("current", "v1.0")

    if promoted:
        # Parse v1.2 -> increment minor
        try:
            major, minor = old_version.lstrip("v").split(".")
            new_version = f"v{major}.{int(minor) + 1}"
        except ValueError:
            new_version = f"{old_version}-next"
        data["current"] = new_version
    else:
        new_version = old_version

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": new_version if promoted else old_version,
        "previous": old_version,
        "f1": round(f1, 4),
        "promoted": promoted,
        "note": note,
    }
    data.setdefault("history", []).append(entry)
    _save(data)

    return {
        "old_version": old_version,
        "new_version": new_version,
        "promoted": promoted,
    }
