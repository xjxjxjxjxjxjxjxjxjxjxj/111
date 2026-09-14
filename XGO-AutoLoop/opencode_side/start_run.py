"""Start a robot run in the background and return immediately.

Execution layer only.  It never changes vision/decision code and never waits for
the robot process: OpenCode's prompt is usable within a couple of seconds.

Steps:
1. controller ``prepare-run`` (only the exact word 启动 is accepted);
2. atomic deploy of ``current_project`` to the robot;
3. create the per-run runtime dir, drop any stale stop.request, seed heartbeat;
4. launch the robot control program with ``nohup`` under the runtime dir;
5. launch the local ``supervisor.py`` (0.5 s heartbeat, PID watch, download);
6. persist run metadata and status, then return.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import subprocess
import sys
import time
from pathlib import Path

from dog_ssh import (
    build_detached_python_launch,
    connect,
    deploy_project,
    ensure_remote_dir,
    free_camera,
    load_config,
    run,
)
from run_state import write_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Start a background robot run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--trigger", default="启动")
    parser.add_argument("--mode", default="run", choices=("run", "observe"))
    parser.add_argument("--supervisor-interval", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["paths"]["root"])
    controller = root / "automation" / "workflow_controller.py"

    # 1. prepare-run (controller enforces the exact 启动 trigger)
    completed = subprocess.run(
        [sys.executable, str(controller), "prepare-run", "--config", args.config,
         "--trigger", args.trigger],
        capture_output=True, text=True, encoding="utf-8",
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout + completed.stderr)
        return completed.returncode
    request = json.loads(completed.stdout)
    run_id = request["run_id"]

    robot = config["robot"]
    target = robot["ssh_target"]
    project = robot["remote_project_dir"]
    backup = robot.get("remote_backup_dir", "/home/pi/xgo-autoloop/previous")
    tmp = robot.get("remote_tmp_dir", "/home/pi/xgo-autoloop/current.tmp")
    base = "/home/pi/xgo-autoloop"
    runtime_dir = "%s/runtime/%s" % (base, run_id)
    stop_file = "%s/stop.request" % runtime_dir
    heartbeat = "%s/heartbeat" % runtime_dir
    pid_file = "%s/robot.pid" % runtime_dir
    remote_log = "%s/robot.log" % runtime_dir
    hb_timeout = float(robot.get("heartbeat_timeout_s", 2.0))
    remote_python = robot.get("python", "python3")
    entry = robot["entry_point"]
    max_seconds = float(robot.get("maximum_run_seconds", 300))
    record_fps = float(config.get("recording", {}).get("fps", 20.0))
    entry_args = robot.get("entry_args", ["--mode", "run", "--start-delay", "5"])
    start_delay = "5"
    if "--start-delay" in entry_args:
        start_delay = str(entry_args[entry_args.index("--start-delay") + 1])

    client = connect(target)
    sftp = client.open_sftp()
    uploaded = deploy_project(client, sftp, request["current_project"], project, backup, tmp)
    ensure_remote_dir(sftp, runtime_dir)
    # Remote paths are POSIX.  Building their parent with pathlib.Path on Windows
    # rewrites separators to backslashes and mkdir then targets "/\home\...".
    ensure_remote_dir(sftp, posixpath.dirname(request["remote_video_partial"]))
    sftp.close()
    print("DEPLOYED %d files to %s" % (uploaded, project))

    # Free the camera from boot/leftover programs before the run opens it.
    free_camera(client)
    # 3. fresh runtime dir state for THIS run only
    run(client, "rm -f '%s' '%s'; touch '%s'" % (stop_file, pid_file, heartbeat))

    # 4. Launch a single, fully detached process.  The helper deliberately puts
    # ``nohup sh -c 'cd ... && exec python ...'`` in the background.  This avoids
    # the shell-precedence bug where ``cd && python &`` kept Paramiko waiting
    # until Python exited and therefore started the supervisor too late.
    launch = build_detached_python_launch(
        project,
        remote_python,
        entry,
        [
            "--mode", args.mode,
            "--start-delay", start_delay,
            "--max-seconds", str(max_seconds),
            "--record-video", request["remote_video_partial"],
            "--telemetry", request["remote_telemetry_partial"],
            "--record-fps", str(record_fps),
            "--stop-request", stop_file,
            "--heartbeat", heartbeat,
            "--heartbeat-timeout-s", str(hb_timeout),
            "--pid-file", pid_file,
        ],
        remote_log,
    )
    out, err, launch_code = run(client, launch)
    if launch_code != 0:
        raise RuntimeError("Remote launch failed: %s" % (err.strip() or out.strip()))
    remote_pid = out.strip().splitlines()[-1].strip() if out.strip() else ""
    if not remote_pid.isdigit():
        raise RuntimeError("Remote launch did not return a numeric PID: %r" % remote_pid)
    print("REMOTE PID:", remote_pid)

    # 5. launch the local supervisor detached (heartbeat + watch + download + ingest)
    supervisor_script = str(Path(__file__).resolve().parent / "supervisor.py")
    sup_args = [
        sys.executable, supervisor_script, "--config", args.config,
        "--run-id", run_id, "--remote-pid", remote_pid, "--pid-file", pid_file,
        "--heartbeat", heartbeat, "--stop-request", stop_file,
        "--runtime-dir", runtime_dir,
        "--remote-video-partial", request["remote_video_partial"],
        "--remote-telemetry-partial", request["remote_telemetry_partial"],
        "--remote-log", remote_log, "--download-dir", request["download_dir"],
        "--interval", str(args.supervisor_interval), "--ingest-on-stop",
    ]
    os.makedirs(request["download_dir"], exist_ok=True)
    sup_log = open(os.path.join(request["download_dir"], "supervisor.log"), "w", encoding="utf-8")
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    sup = subprocess.Popen(sup_args, creationflags=flags, close_fds=True,
                           stdout=sup_log, stderr=subprocess.STDOUT)

    record = {
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "supervisor_pid": sup.pid,
        "remote_pid": remote_pid,
        "mode": args.mode,
        "runtime_dir": runtime_dir,
        "heartbeat_file": heartbeat,
        "stop_request_file": stop_file,
        "pid_file": pid_file,
        "remote_log": remote_log,
        "remote_video_partial": request["remote_video_partial"],
        "remote_video_final": request["remote_video_final"],
        "remote_telemetry_partial": request["remote_telemetry_partial"],
        "remote_telemetry_final": request["remote_telemetry_final"],
        "download_dir": request["download_dir"],
    }
    (Path(config["paths"]["state"]) / "active_run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_status(
        config, run_id=run_id, phase="running", remote_pid=remote_pid,
        heartbeat_age=0.0, started_at=record["started_at"], exit_code=None,
    )

    print("RUN STARTED (background) mode=%s id=%s" % (args.mode, run_id))
    print("  supervisor pid:", sup.pid, " remote pid:", remote_pid)
    print("  runtime dir:", runtime_dir)
    print("OpenCode 输入界面已恢复。要停止请输入：停止")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
