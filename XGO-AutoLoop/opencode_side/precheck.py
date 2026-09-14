"""Supervisor pre-check with the motors DISABLED (observe mode).

Verifies the execution layer only:
  a. the launch command returns at once (OpenCode prompt not blocked);
  b. the heartbeat file keeps refreshing;
  c. stop.request terminates this run's process;
  d. the remote PID belongs to this run only;
  e. the video and log paths are writable.

It does NOT call prepare-run and does NOT create / move the robot (observe mode
never instantiates the XGO motor object).
"""

from __future__ import annotations

import argparse
import json
import os
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
    remote_cmdline,
    run,
)
from run_state import read_status

BASE = "/home/pi/xgo-autoloop"


def main() -> int:
    parser = argparse.ArgumentParser(description="Motor-disabled supervisor pre-check")
    parser.add_argument("--config", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.config)
    robot = config["robot"]
    target = robot["ssh_target"]
    project = robot["remote_project_dir"]
    run_id = "precheck_%s" % time.strftime("%Y%m%d-%H%M%S")
    runtime = "%s/runtime/%s" % (BASE, run_id)
    heartbeat = "%s/heartbeat" % runtime
    stop_file = "%s/stop.request" % runtime
    pid_file = "%s/robot.pid" % runtime
    remote_log = "%s/robot.log" % runtime
    video = "%s/precheck.partial.mp4" % runtime
    telemetry = "%s/precheck.partial.jsonl" % runtime
    download_dir = Path(config["paths"]["root"]) / "downloads" / run_id
    results = {}

    client = connect(target)
    sftp = client.open_sftp()
    uploaded = deploy_project(
        client, sftp, config["paths"]["current_project"], project,
        robot.get("remote_backup_dir", "%s/previous" % BASE),
        robot.get("remote_tmp_dir", "%s/current.tmp" % BASE),
    )
    ensure_remote_dir(sftp, runtime)
    sftp.close()
    results["deployed_files"] = uploaded
    # Free the camera so observe mode can open it (no motors involved) and clear
    # any leaked loop from a previous pre-check.  Never killall python3.
    free_camera(client)
    run(client, "rm -f '%s' '%s'; touch '%s'" % (stop_file, pid_file, heartbeat))

    launch = build_detached_python_launch(
        project,
        robot.get("python", "python3"),
        "sign_line_closed_loop.py",
        [
            "--mode", "observe", "--start-delay", "0", "--max-seconds", "60",
            "--record-video", video, "--telemetry", telemetry,
            "--record-fps", "20", "--stop-request", stop_file,
            "--heartbeat", heartbeat, "--heartbeat-timeout-s", "2",
            "--pid-file", pid_file,
        ],
        remote_log,
    )

    t0 = time.time()
    out, err, launch_code = run(client, launch)
    return_seconds = time.time() - t0
    remote_pid = out.strip().splitlines()[-1].strip() if out.strip() else ""
    results["launch_return_seconds"] = round(return_seconds, 2)
    results["launch_nonblocking"] = (
        launch_code == 0 and remote_pid.isdigit() and return_seconds < 2.0
    )
    results["launch_error"] = err.strip() or None

    # start the local supervisor detached (heartbeat + watch + download)
    download_dir.mkdir(parents=True, exist_ok=True)
    sup_script = str(Path(__file__).resolve().parent / "supervisor.py")
    sup_log = open(str(download_dir / "supervisor.log"), "w", encoding="utf-8")
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    sup = subprocess.Popen(
        [sys.executable, sup_script, "--config", args.config, "--run-id", run_id,
         "--remote-pid", remote_pid, "--pid-file", pid_file,
         "--heartbeat", heartbeat, "--stop-request", stop_file,
         "--runtime-dir", runtime, "--remote-video-partial", video,
         "--remote-telemetry-partial", telemetry, "--remote-log", remote_log,
         "--download-dir", str(download_dir), "--interval", str(args.interval)],
        creationflags=flags, close_fds=True,
        stdout=sup_log, stderr=subprocess.STDOUT,
    )
    results["supervisor_pid"] = sup.pid

    # (b) heartbeat keeps refreshing
    time.sleep(2.0)
    m1 = run(client, "stat -c '%%Y' '%s' 2>/dev/null" % heartbeat)[0].strip()
    time.sleep(1.5)
    m2 = run(client, "stat -c '%%Y' '%s' 2>/dev/null" % heartbeat)[0].strip()
    results["heartbeat_mtime_1"] = m1
    results["heartbeat_mtime_2"] = m2
    results["heartbeat_refreshing"] = bool(m1) and bool(m2) and m2 > m1

    # (d) PID belongs to this run only
    pid_from_file = ""
    for _ in range(10):
        pid_from_file = run(client, "cat '%s' 2>/dev/null" % pid_file)[0].strip()
        if pid_from_file:
            break
        time.sleep(0.3)
    target_pid = pid_from_file or remote_pid
    args_line = remote_cmdline(client, target_pid)
    proc_count = run(
        client, "ps -ef | grep sign_line_closed_loop | grep -v grep | wc -l"
    )[0].strip()
    # robot.pid was cleared before launch and is written only by this run's
    # program, so a live PID that matches plus exactly one control process means
    # the PID belongs to this run.
    # The run-specific --pid-file token binds this PID to this exact launch and
    # also rejects a recycled PID that happens to belong to another process.
    pid_alive = bool(args_line) and pid_file in args_line
    results["robot_pid"] = target_pid
    results["pid_args"] = args_line
    results["control_process_count"] = proc_count
    results["pid_is_this_run"] = pid_alive and proc_count == "1"

    # (e) video and log paths writable
    time.sleep(2.0)
    size = run(client, "stat -c '%%s' '%s' 2>/dev/null" % video)[0].strip()
    log_size = run(client, "stat -c '%%s' '%s' 2>/dev/null" % remote_log)[0].strip()
    results["video_bytes"] = size
    results["log_bytes"] = log_size
    results["paths_writable"] = bool(size) and int(size or 0) > 0 and bool(log_size) and int(log_size or 0) > 0

    # (c) stop.request terminates this run's process
    run(client, "touch '%s'" % stop_file)
    t_stop = time.time()
    stopped = False
    while time.time() - t_stop < 5.0:
        alive = bool(remote_cmdline(client, pid_from_file or remote_pid))
        if not alive:
            stopped = True
            break
        time.sleep(0.2)
    results["stop_exit_seconds"] = round(time.time() - t_stop, 2)
    results["stop_request_terminates"] = stopped

    # wait for the supervisor to finish downloading
    deadline = time.time() + 40
    while time.time() < deadline and sup.poll() is None:
        time.sleep(1.0)
    status = read_status(config)
    results["status_after"] = status
    local_video = download_dir / ("%s.mp4" % run_id)
    local_partial = download_dir / ("%s.partial.mp4" % run_id)
    results["local_video_verified"] = local_video.is_file() and local_video.stat().st_size > 0
    results["downloaded_video"] = str(local_video if local_video.is_file() else local_partial)

    if sup.poll() is None:
        sup.terminate()
    client.close()

    passed = all([
        results["launch_nonblocking"],
        results["heartbeat_refreshing"],
        results["pid_is_this_run"],
        results["paths_writable"],
        results["stop_request_terminates"],
        results["local_video_verified"],
    ])
    results["passed"] = passed
    report_path = Path(config["paths"]["state"]) / "precheck_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in results.items() if k != "status_after"},
                     ensure_ascii=False, indent=2))
    print("PRECHECK", "PASS" if passed else "FAIL", "->", report_path)
    if not passed:
        return 1
    print("后台启动、停止通道、心跳和录像均已就绪，准备好后请输入启动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
