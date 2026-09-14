"""Stop the active robot run immediately (execution layer only).

Order mandated by the operator:
1. create this run's ``stop.request`` (the control loop checks it every frame);
2. wait up to 1 s for a graceful exit;
3. if still alive, send SIGINT ONLY to this run's ``robot.pid``;
   ``killall python3`` is forbidden;
4. wait for the background supervisor to download artifacts and ingest
   ``termination=opencode_stop`` (rating 2 is forced by the controller).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dog_ssh import connect, load_config, remote_cmdline, run
from run_state import read_status, write_status


def _this_run_alive(client, pid: str, pid_file: str) -> bool:
    """Confirm both process existence and this run's unique PID-file token."""

    command_line = remote_cmdline(client, pid)
    return bool(command_line) and pid_file in command_line


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop the active robot run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--wait-seconds", type=float, default=120.0)
    args = parser.parse_args()

    config = load_config(args.config)
    active_path = Path(config["paths"]["state"]) / "active_run.json"
    if not active_path.is_file():
        print("No active run recorded (state/active_run.json missing).")
        return 1
    record = json.loads(active_path.read_text(encoding="utf-8"))

    write_status(config, phase="stopping")
    client = connect(config["robot"]["ssh_target"])

    # 1. per-run stop flag
    run(client, "touch '%s'" % record["stop_request_file"])
    print("stop.request created:", record["stop_request_file"])

    # robot.pid is authoritative.  The launch PID is only a fallback, although
    # the detached exec launch normally makes both values identical.
    pid_from_file = run(
        client, "cat '%s' 2>/dev/null" % record["pid_file"]
    )[0].strip()
    target_pid = pid_from_file if pid_from_file.isdigit() else record["remote_pid"]

    # 2. wait up to 1 s for the control loop to stop on its own
    deadline = time.time() + 1.0
    while time.time() < deadline and _this_run_alive(
        client, target_pid, record["pid_file"]
    ):
        time.sleep(0.1)

    # 3. escalate to SIGINT on THIS run's robot.pid only
    if _this_run_alive(client, target_pid, record["pid_file"]):
        run(client, "kill -INT %s 2>/dev/null" % target_pid)
        print("sent SIGINT to this run's pid", target_pid)
    client.close()

    # 4. wait for the supervisor to download + ingest
    deadline = time.time() + args.wait_seconds
    last = None
    while time.time() < deadline:
        status = read_status(config)
        if status.get("phase") in ("stopped", "ingested_stop", "ended", "awaiting_rating"):
            last = status
            break
        time.sleep(1.0)
    print(json.dumps(last or read_status(config), ensure_ascii=False, indent=2))
    if last is None:
        print("Supervisor did not finish; active_run.json retained for safe retry.")
        return 2
    active_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
