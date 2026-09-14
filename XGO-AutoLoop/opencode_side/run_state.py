"""Read/write the OpenCode-side run status file.

Path: ``<paths.state>/robot_run_status.json``.  It always exposes at least:
run_id, phase, remote_pid, heartbeat_age, started_at, last_update, exit_code.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

FIELDS = (
    "run_id",
    "phase",
    "remote_pid",
    "heartbeat_age",
    "started_at",
    "last_update",
    "exit_code",
)


def status_path(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["state"]) / "robot_run_status.json"


def read_status(config: Dict[str, Any]) -> Dict[str, Any]:
    path = status_path(config)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(config: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
    path = status_path(config)
    current: Dict[str, Any] = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(updates)
    current["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for field in FIELDS:
        current.setdefault(field, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current
