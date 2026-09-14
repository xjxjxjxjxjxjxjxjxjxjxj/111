"""Background local supervisor for one robot run.

Responsibilities (execution layer only; never touches vision code):
* refresh the remote heartbeat every ``--interval`` seconds (default 0.5);
* watch the remote robot PID that belongs to this run only;
* keep ``state/robot_run_status.json`` up to date;
* when the robot process ends, download video/telemetry/log, verify SHA-256,
  rename the ``.partial`` recording to its final name and delete the robot copy;
* for an OpenCode stop, optionally run the controller ``ingest`` (rating 2).

It never waits on the robot in the foreground: OpenCode starts it detached and
returns immediately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from dog_ssh import connect, load_config, run
from run_state import write_status


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_process_state(client, pid: str, expected_token: str) -> tuple[str, str]:
    """Return ``(state, cmdline)`` for the run-specific remote process.

    States are ``alive``, ``gone``, ``mismatch`` and ``unknown``.  A transient
    SSH/query failure is *unknown*, never dead, so it cannot make the supervisor
    stop its heartbeat and trigger the robot watchdog.  The run-specific PID
    file argument also protects against PID reuse.
    """

    if not re.fullmatch(r"[1-9][0-9]*", str(pid or "")):
        return "gone", ""
    command = (
        "if [ -r /proc/{pid}/cmdline ]; then "
        "tr '\\000' ' ' < /proc/{pid}/cmdline; else exit 3; fi"
    ).format(pid=pid)
    try:
        out, _, code = run(client, command)
    except Exception:
        return "unknown", ""
    if code == 3:
        return "gone", ""
    if code != 0:
        return "unknown", ""
    cmdline = out.strip()
    if expected_token and expected_token not in cmdline:
        return "mismatch", cmdline
    return "alive", cmdline


def _start_heartbeat_writer(config, args):
    """Start a dedicated SSH heartbeat loop and return its stop event/thread."""

    hb_stop = threading.Event()

    def heartbeat_writer() -> None:
        hb_client = None
        try:
            while not hb_stop.is_set():
                try:
                    if hb_client is None:
                        hb_client = connect(config["robot"]["ssh_target"])
                        hb_client.get_transport().set_keepalive(1)
                    _, error, code = run(
                        hb_client,
                        "mkdir -p '%s'; touch '%s'"
                        % (args.runtime_dir, args.heartbeat),
                    )
                    if code != 0:
                        raise RuntimeError(error.strip() or "heartbeat touch failed")
                except Exception:
                    try:
                        if hb_client is not None:
                            hb_client.close()
                    except Exception:
                        pass
                    hb_client = None
                hb_stop.wait(max(0.2, args.interval))
        finally:
            if hb_client is not None:
                try:
                    hb_client.close()
                except Exception:
                    pass

    thread = threading.Thread(target=heartbeat_writer, daemon=True)
    thread.start()
    return hb_stop, thread


def _download_and_finalize(config, client, record, download_dir: Path) -> dict:
    download_dir.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    result = {"video": None, "telemetry": None, "log": None, "verified": False}
    run_id = record["run_id"]
    try:
        final_video = download_dir / ("%s.mp4" % run_id)
        partial_video = download_dir / ("%s.partial.mp4" % run_id)
        telemetry = download_dir / ("%s.jsonl" % run_id)
        log = download_dir / "terminal.log"

        for remote, local in (
            (record["remote_video_partial"], str(partial_video) + ".downloading"),
            (record["remote_telemetry_partial"], str(telemetry) + ".downloading"),
            (record["remote_log"], str(log) + ".downloading"),
        ):
            try:
                sftp.get(remote, local)
                os.replace(local, local[:-len(".downloading")])
            except FileNotFoundError:
                print("missing remote artifact:", remote, flush=True)
            except Exception as exc:  # pragma: no cover - network dependent
                print("download failed:", remote, exc, flush=True)

        if not partial_video.is_file():
            result["telemetry"] = str(telemetry) if telemetry.is_file() else None
            result["log"] = str(log) if log.is_file() else None
            return result

        size_a = partial_video.stat().st_size
        time.sleep(1)
        size_b = partial_video.stat().st_size
        remote_sha = run(client, "sha256sum '%s' | cut -d' ' -f1" % record["remote_video_partial"])[0].strip()
        if size_a > 0 and size_a == size_b and _sha256(str(partial_video)) == remote_sha:
            if final_video.exists():
                final_video.unlink()
            os.replace(str(partial_video), str(final_video))
            run(client, "rm -f '%s' '%s'" % (record["remote_video_partial"], record["remote_telemetry_partial"]))
            result.update(video=str(final_video), verified=True)
        else:
            result.update(video=str(partial_video), verified=False)
        result["telemetry"] = str(telemetry)
        result["log"] = str(log)
    finally:
        sftp.close()
    return result


def _read_outcome(telemetry_path: str) -> tuple[str | None, int | None]:
    outcome = None
    exit_code = None
    try:
        with open(telemetry_path, "r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") == "run_finished":
                    outcome = record.get("outcome")
                    exit_code = record.get("exit_code")
    except OSError:
        pass
    return outcome, exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-run local supervisor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remote-pid", default="")
    parser.add_argument("--pid-file", default="")
    parser.add_argument("--heartbeat", required=True)
    parser.add_argument("--stop-request", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--remote-video-partial", required=True)
    parser.add_argument("--remote-telemetry-partial", required=True)
    parser.add_argument("--remote-log", required=True)
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--ingest-on-stop", action="store_true")
    parser.add_argument("--comment", default="OpenCode收到用户停止命令")
    args = parser.parse_args()

    config = load_config(args.config)
    record = {
        "run_id": args.run_id,
        "remote_video_partial": args.remote_video_partial,
        "remote_telemetry_partial": args.remote_telemetry_partial,
        "remote_log": args.remote_log,
    }
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    client = connect(config["robot"]["ssh_target"])

    # Refresh once synchronously, then start the dedicated writer *before* PID
    # resolution.  SSH handshakes or a slow camera can no longer consume the
    # robot's entire startup grace while no heartbeat is being sent.
    _, heartbeat_error, heartbeat_code = run(
        client,
        "mkdir -p '%s'; touch '%s'" % (args.runtime_dir, args.heartbeat),
    )
    if heartbeat_code != 0:
        write_status(
            config,
            run_id=args.run_id,
            phase="ended",
            remote_pid=args.remote_pid,
            heartbeat_age=None,
            started_at=started_at,
            exit_code=4,
            outcome="supervisor_heartbeat_start_failed",
        )
        client.close()
        print("heartbeat start failed: %s" % heartbeat_error.strip(), flush=True)
        return 4
    hb_stop, hb_thread = _start_heartbeat_writer(config, args)

    # Resolve this run's PID from robot.pid (written by the control program),
    # falling back to the PID returned by the detached ``exec`` launch.  Wait for
    # the authoritative file even when a fallback exists; the previous code
    # exited this loop immediately and never actually preferred robot.pid.
    fallback_pid = args.remote_pid if str(args.remote_pid).isdigit() else ""
    remote_pid = ""
    for _ in range(50):
        if args.pid_file:
            candidate = run(client, "cat '%s' 2>/dev/null" % args.pid_file)[0].strip()
            if candidate.isdigit():
                remote_pid = candidate
                break
        time.sleep(0.2)
    if not remote_pid:
        remote_pid = fallback_pid
    if not remote_pid:
        hb_stop.set()
        hb_thread.join(timeout=3.0)
        write_status(
            config,
            run_id=args.run_id,
            phase="ended",
            remote_pid="",
            heartbeat_age=None,
            started_at=started_at,
            exit_code=5,
            outcome="run_pid_unavailable",
        )
        client.close()
        print("no run-specific remote PID became available", flush=True)
        return 5
    print("supervisor watching pid=%s" % remote_pid, flush=True)

    write_status(
        config, run_id=args.run_id, phase="running", remote_pid=remote_pid,
        heartbeat_age=0.0, started_at=started_at, exit_code=None,
    )

    watch_started = time.time()
    misses = 0
    unknowns = 0
    print("watch loop started for pid=%s" % remote_pid, flush=True)
    while True:
        now = time.time()
        try:
            hb_mtime = float(
                run(client, "stat -c %%Y '%s' 2>/dev/null" % args.heartbeat)[0].strip() or 0
            )
            server_now = float(run(client, "date +%%s")[0].strip() or 0)
            heartbeat_age = round(server_now - hb_mtime, 3) if hb_mtime else None
        except Exception:
            heartbeat_age = None
        process_state, command_line = _remote_process_state(
            client, remote_pid, args.pid_file
        )
        if process_state == "alive":
            misses = 0
            unknowns = 0
        elif process_state in ("gone", "mismatch"):
            misses += 1
            unknowns = 0
            print(
                "pid %s state=%s (miss %d, elapsed %.1fs) cmd=%r"
                % (remote_pid, process_state, misses, now - watch_started, command_line),
                flush=True,
            )
        else:
            # Unknown means the probe failed, not that the process died.  Keep
            # heartbeats alive and reconnect the watch channel after repeated
            # unknown results.
            unknowns += 1
            print("pid %s state=unknown (probe %d)" % (remote_pid, unknowns), flush=True)
            if unknowns >= 3:
                try:
                    client.close()
                except Exception:
                    pass
                try:
                    client = connect(config["robot"]["ssh_target"])
                    unknowns = 0
                except Exception:
                    pass
        write_status(config, phase="running", remote_pid=remote_pid,
                     heartbeat_age=heartbeat_age)
        # Six confirmed /proc misses are required.  Unknown SSH failures never
        # count, so the supervisor cannot create its own watchdog failure.
        if misses >= 6 and now - watch_started >= 3.0:
            print("pid %s confirmed gone after %d misses" % (remote_pid, misses), flush=True)
            break
        time.sleep(max(0.1, args.interval))

    # Stop the heartbeat writer; the remote read loop then exits on its timeout.
    hb_stop.set()
    hb_thread.join(timeout=3.0)

    stop_requested = run(
        client, "test -f '%s' && echo YES || echo NO" % args.stop_request
    )[0].strip() == "YES"

    result = {"video": None, "telemetry": None, "log": None, "verified": False}
    exit_code = None
    outcome = None
    if not args.no_download:
        write_status(config, phase="downloading")
        result = _download_and_finalize(config, client, record, Path(args.download_dir))
        if result.get("telemetry"):
            outcome, exit_code = _read_outcome(result["telemetry"])
    client.close()

    if stop_requested:
        phase = "stopped"
    elif outcome in ("goal_complete_black", "goal_complete_yellow") or exit_code == 0:
        phase = "awaiting_rating"
    else:
        phase = "ended"
    write_status(config, phase=phase, exit_code=exit_code, outcome=outcome,
                 video=result.get("video"), telemetry=result.get("telemetry"),
                 log=result.get("log"), verified=result.get("verified"))
    print("SUPERVISOR DONE run=%s phase=%s outcome=%s exit=%s verified=%s"
          % (args.run_id, phase, outcome, exit_code, result.get("verified")))

    if stop_requested and args.ingest_on_stop and result.get("video"):
        controller = Path(config["paths"]["root"]) / "automation" / "workflow_controller.py"
        subprocess.run(
            [sys.executable, str(controller), "ingest", "--config", args.config,
             "--video", result["video"], "--telemetry", result["telemetry"],
             "--terminal-log", result["log"], "--termination", "opencode_stop",
             "--sign-color", "auto", "--comment", args.comment],
            capture_output=True, text=True, encoding="utf-8",
        )
        write_status(config, phase="ingested_stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
