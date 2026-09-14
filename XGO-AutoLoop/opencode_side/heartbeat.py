"""Local supervisor heartbeat sender.

Runs on the OpenCode computer and refreshes a file on the robot every second.
The robot control loop stops the motors automatically if this heartbeat goes
stale for ``robot.heartbeat_timeout_s`` seconds, so a crashed/frozen supervisor
or a lost network connection cannot leave the robot driving.

This process *is* the recorded local supervisor: its PID is stored in
``state/active_run.json`` so ``stop_run.py`` can terminate it.
"""

from __future__ import annotations

import argparse
import time

from dog_ssh import connect, load_config, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the robot heartbeat file")
    parser.add_argument("--config", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config)
    target = config["robot"]["ssh_target"]
    heartbeat = config["robot"].get(
        "heartbeat_file", "/home/pi/xgo-autoloop/supervisor.heartbeat"
    )

    client = None
    while True:
        try:
            if client is None:
                client = connect(target)
            run(
                client,
                "mkdir -p $(dirname %s); touch %s" % (heartbeat, heartbeat),
            )
        except Exception:
            # Connection lost.  Stop refreshing so the robot's watchdog fires.
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass
            client = None
        time.sleep(max(0.2, float(args.interval)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
