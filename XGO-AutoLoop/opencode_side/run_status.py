"""Show the active run: local/remote PIDs, state machine status and log tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dog_ssh import connect, load_config, remote_cmdline, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Show active robot run status")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tail", type=int, default=15)
    args = parser.parse_args()

    config = load_config(args.config)
    active_path = Path(config["paths"]["state"]) / "active_run.json"
    if not active_path.is_file():
        print("No active run (state/active_run.json missing).")
        return 0
    record = json.loads(active_path.read_text(encoding="utf-8"))
    print(json.dumps(record, ensure_ascii=False, indent=2))

    client = connect(config["robot"]["ssh_target"])
    pid = record.get("remote_pid", "")
    pid_file = record.get("pid_file", "")
    pid_from_file = run(client, "cat '%s' 2>/dev/null" % pid_file)[0].strip()
    if pid_from_file.isdigit():
        pid = pid_from_file
    command_line = remote_cmdline(client, pid)
    alive = bool(command_line) and bool(pid_file) and pid_file in command_line
    print("remote pid alive:", alive)
    print("remote pid command:", command_line or "<gone or unavailable>")
    print("--- log tail ---")
    print(run(client, "tail -n %d '%s' 2>&1" % (args.tail, record["remote_log"]))[0])
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
