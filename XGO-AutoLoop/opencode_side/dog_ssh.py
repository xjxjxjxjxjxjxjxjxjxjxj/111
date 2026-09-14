"""OpenCode-side SSH/SFTP helpers for the robot supervisor.

These scripts live under ``XGO-AutoLoop/`` which ``.gitignore`` excludes runtime
data from while the execution layer itself is versioned.  No password is ever
committed: the SSH password is read from the ``DOG_PASSWORD`` environment
variable and must be set before running the workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import paramiko

# No default lab password lives in source control.  DOG_PASSWORD must be set in
# the environment; scripts fail closed instead of falling back to a literal.
DEFAULT_PASSWORD = os.environ.get("DOG_PASSWORD", "")
SKIP_DIRS = {"__pycache__", ".git"}
SKIP_EXT = {".pyc"}


def _resolve_password(password: str | None = None) -> str:
    """Resolve the SSH password from the argument or DOG_PASSWORD, else fail."""

    resolved = password or DEFAULT_PASSWORD
    if not resolved:
        raise RuntimeError(
            "SSH password is not set; export DOG_PASSWORD before running the workflow."
        )
    return resolved

# Camera owners are authorized by their resolved script path, not by a loose
# basename. 摄像头占用者必须匹配解析后的完整路径，不能只匹配同名脚本。
CURRENT_PROJECT = "/home/pi/xgo-autoloop/current"
CURRENT_CAMERA_SCRIPTS = frozenset(
    {
        "sign_line_closed_loop.py",
        "autonomous_senior_v4.py",
        "ball_closed_loop.py",
        "ball_grab_test.py",
        "ball_observe.py",
        "cup_vision.py",
    }
)
LEGACY_CAMERA_SCRIPTS = frozenset(
    {
        "/home/pi/main.py",
        "/home/pi/text.py",
        "/home/pi/text3.py",
        "/home/pi/seven1.py",
    }
)
AUTOSTART_ROOT = "/home/pi/RaspberryPi-CM4-main"
AUTOSTART_MAIN = AUTOSTART_ROOT + "/main.py"
AUTOSTART_CAMERA_APP = AUTOSTART_ROOT + "/app/app_dogzilla.py"


def _command_tokens(command_line: str) -> List[str]:
    """Split a proc command line defensively; malformed text is never trusted."""

    try:
        return shlex.split(command_line)
    except ValueError:
        return []


def _script_path(command_line: str, cwd: str) -> str:
    """Resolve the first Python script token against the process working dir."""

    for token in _command_tokens(command_line):
        if token.endswith(".py"):
            if not token.startswith("/"):
                token = posixpath.join(cwd, token)
            return posixpath.normpath(token)
    return ""


def _command_basename(command_line: str) -> str:
    tokens = _command_tokens(command_line)
    return posixpath.basename(tokens[0]) if tokens else ""


def _is_trusted_standalone_script(script: str) -> bool:
    """Accept only exact legacy paths or named scripts under current/."""

    if script in LEGACY_CAMERA_SCRIPTS:
        return True
    return (
        posixpath.dirname(script) == CURRENT_PROJECT
        and posixpath.basename(script) in CURRENT_CAMERA_SCRIPTS
    )


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def ssh_parts(target: str) -> Tuple[str, str]:
    user, host = target.split("@", 1)
    return user, host


def connect(target: str, password: str | None = None, timeout: int = 25) -> paramiko.SSHClient:
    user, host = ssh_parts(target)
    resolved = _resolve_password(password)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        port=22,
        username=user,
        password=resolved,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
    )
    return client


def run(client: paramiko.SSHClient, command: str) -> Tuple[str, str, int]:
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return out, err, code


def _sudo(command: str, password: str) -> str:
    """Build a non-interactive sudo command without assuming passwordless sudo."""

    return "printf '%s\\n' {pw} | sudo -S -p '' {command}".format(
        pw=shlex.quote(password), command=command
    )


def remote_cmdline(client: paramiko.SSHClient, pid: str) -> str:
    """Return one remote process command line, or an empty string if it is gone."""

    if not re.fullmatch(r"[1-9][0-9]*", str(pid or "")):
        return ""
    out, _, code = run(
        client,
        "if [ -r /proc/{pid}/cmdline ]; then "
        "tr '\\000' ' ' < /proc/{pid}/cmdline; else exit 3; fi".format(pid=pid),
    )
    return out.strip() if code == 0 else ""


def _root_process_info(
    client: paramiko.SSHClient, pid: str, password: str
) -> Dict[str, str] | None:
    """Read security-relevant proc fields through narrowly scoped sudo calls.

    The boot service is root-owned, so an ordinary SSH account may not be able
    to read every proc entry. No process is signalled unless command line, cwd
    and parent PID were all read successfully. 开机服务属于 root，因此读取不完整
    时必须按未知进程处理并停止预检。
    """

    if not re.fullmatch(r"[1-9][0-9]*", str(pid or "")):
        return None
    proc = "/proc/%s" % pid
    cmdline_command = "tr '\\000' ' ' < %s/cmdline" % proc
    cmdline, _, cmdline_code = run(
        client, _sudo("sh -c %s" % shlex.quote(cmdline_command), password)
    )
    cwd, _, cwd_code = run(
        client, _sudo("readlink -f %s/cwd" % proc, password)
    )
    ppid, _, ppid_code = run(
        client, _sudo("ps -o ppid= -p %s" % pid, password)
    )
    stat_line, _, stat_code = run(
        client, _sudo("cat %s/stat" % proc, password)
    )
    command_line = cmdline.strip()
    working_dir = cwd.strip()
    parent_pid = ppid.strip()
    # Field 22 is process starttime. Parse after the last ')' because the comm
    # field itself may contain spaces or parentheses. 第22字段用于防止 PID 复用。
    stat_tail = stat_line.strip().rsplit(")", 1)
    stat_fields = stat_tail[1].strip().split() if len(stat_tail) == 2 else []
    starttime = stat_fields[19] if len(stat_fields) > 19 else ""
    if (
        cmdline_code != 0
        or cwd_code != 0
        or ppid_code != 0
        or stat_code != 0
        or not command_line
        or not working_dir.startswith("/")
        or not re.fullmatch(r"[0-9]+", parent_pid)
        or not re.fullmatch(r"[0-9]+", starttime)
    ):
        return None
    return {
        "pid": str(pid),
        "ppid": parent_pid,
        "cwd": posixpath.normpath(working_dir),
        "cmdline": command_line,
        "script": _script_path(command_line, working_dir),
        "starttime": starttime,
    }


def _verified_camera_kill_chain(
    client: paramiko.SSHClient,
    owner: Dict[str, str],
    password: str,
) -> Tuple[List[Dict[str, str]], str]:
    """Return a verified top-down kill chain, or a refusal explanation.

    The CM4 image starts the camera app as main.py -> sh -> app_dogzilla.py.
    Killing only the leaf is unreliable because its supervisor can retain or
    recreate it. The chain is authorized only when exact paths, cwd values and
    parent PIDs all agree with the known image layout. 只有完整父子链吻合才授权。
    """

    if _is_trusted_standalone_script(owner["script"]):
        return [owner], ""
    if owner["script"] != AUTOSTART_CAMERA_APP or owner["cwd"] != AUTOSTART_ROOT:
        return [], "camera owner path/cwd is not allow-listed"

    shell = _root_process_info(client, owner["ppid"], password)
    if shell is None:
        return [], "cannot inspect app_dogzilla parent"
    if (
        _command_basename(shell["cmdline"]) not in {"sh", "dash", "bash"}
        or shell["cwd"] != AUTOSTART_ROOT
        or shell["script"] != AUTOSTART_CAMERA_APP
    ):
        return [], "app_dogzilla parent is not the expected launch shell"

    main = _root_process_info(client, shell["ppid"], password)
    if main is None:
        return [], "cannot inspect launch-shell parent"
    if main["cwd"] != AUTOSTART_ROOT or main["script"] != AUTOSTART_MAIN:
        return [], "launch-shell parent is not the expected main.py"
    return [main, shell, owner], ""


def _root_process_unchanged(
    client: paramiko.SSHClient, expected: Dict[str, str], password: str
) -> bool:
    """Revalidate identity immediately before signalling to prevent PID reuse.

    信号发送前再次核对 cmdline/cwd/ppid/script；若 PID 已复用则拒绝误杀。
    """

    actual = _root_process_info(client, expected["pid"], password)
    if actual is None:
        return False
    # PPID may legitimately change after the verified parent receives SIGINT;
    # starttime still proves that the PID has not been recycled.
    fields = ("pid", "cwd", "cmdline", "script", "starttime")
    return all(actual[field] == expected[field] for field in fields)


def build_detached_python_launch(
    project: str,
    python: str,
    entry: str,
    arguments: List[str],
    log_path: str,
) -> str:
    """Build a truly detached launch whose ``$!`` becomes the Python PID.

    ``cd PROJECT && python ... &`` backgrounds the complete AND-list on some
    remote shells.  That helper subshell can keep Paramiko's SSH channel open
    until Python exits, which delays supervisor startup and creates a false
    heartbeat failure.  Here the detached shell redirects every standard file
    descriptor and ``exec`` replaces itself with Python, so ``$!`` is also the
    final controller PID and the SSH command returns immediately.
    """

    argv = [python, "-u", entry] + [str(item) for item in arguments]
    inner = "cd {project} && exec env PYTHONPATH=/home/pi {argv}".format(
        project=shlex.quote(project), argv=" ".join(shlex.quote(item) for item in argv)
    )
    return (
        "nohup sh -c {inner} </dev/null >{log} 2>&1 & "
        "printf '%s\\n' \"$!\""
    ).format(inner=shlex.quote(inner), log=shlex.quote(log_path))


def free_camera(
    client: paramiko.SSHClient,
    password: str | None = None,
    device: str = "/dev/video0",
) -> List[Dict[str, str]]:
    """Release /dev/video0 from an exactly verified XGO process or service.

    Every fuser owner is inspected before the first signal. A standalone XGO
    process is accepted only at an exact approved path. The CM4 boot app is
    accepted only with its complete main.py -> shell -> app_dogzilla.py
    ancestry. Unknown or partially readable owners fail closed without any
    signal. Broad pkill/killall commands are intentionally forbidden. 任一占用者
    身份不明时，所有进程均保持不动并让预检失败。
    """

    password = _resolve_password(password)
    quoted_device = shlex.quote(device)
    out, err, code = run(
        client,
        _sudo("fuser %s 2>/dev/null" % quoted_device, password),
    )
    if code == 1:  # fuser: no process currently owns the device
        return []
    if code != 0:
        raise RuntimeError(
            "Cannot inspect camera owner with fuser (exit %d): %s"
            % (code, (err or out).strip())
        )

    pids = sorted(set(re.findall(r"\b[1-9][0-9]*\b", out)), key=int)
    owners: List[Dict[str, str]] = []
    unknown: List[Dict[str, str]] = []
    kill_chain: List[Dict[str, str]] = []
    for pid in pids:
        owner = _root_process_info(client, pid, password)
        if owner is None:
            unknown.append({"pid": pid, "reason": "cannot read complete proc identity"})
            continue
        owners.append(owner)
        verified, reason = _verified_camera_kill_chain(client, owner, password)
        if not verified:
            unknown.append(
                {"pid": pid, "cmdline": owner["cmdline"], "reason": reason}
            )
        else:
            kill_chain.extend(verified)

    # Inspect all owners before stopping any of them.  If even one is unknown,
    # leave the machine untouched and require a human decision.
    if unknown:
        raise RuntimeError("Refusing to stop unknown camera owner(s): %s" % unknown)

    # Preserve top-down order and signal every verified PID at most once. PID 1
    # is excluded defensively even though it cannot match the verified chain.
    targets: List[Dict[str, str]] = []
    seen = set()
    for process in kill_chain:
        pid = process["pid"]
        if pid != "1" and pid not in seen:
            seen.add(pid)
            targets.append(process)

    for process in targets:
        pid = process["pid"]
        if _root_process_unchanged(client, process, password):
            run(client, _sudo("kill -INT %s 2>/dev/null || true" % pid, password))
    time.sleep(0.5)
    for process in targets:
        pid = process["pid"]
        if _root_process_unchanged(client, process, password):
            run(client, _sudo("kill -TERM %s 2>/dev/null || true" % pid, password))
    time.sleep(0.5)
    for process in targets:
        pid = process["pid"]
        if _root_process_unchanged(client, process, password):
            run(client, _sudo("kill -KILL %s 2>/dev/null || true" % pid, password))

    out, err, code = run(
        client,
        _sudo("fuser %s 2>/dev/null" % quoted_device, password),
    )
    if code not in (0, 1):
        raise RuntimeError(
            "Cannot verify camera release with fuser (exit %d): %s"
            % (code, (err or out).strip())
        )
    remaining = re.findall(r"\b[1-9][0-9]*\b", out) if code == 0 else []
    if remaining:
        raise RuntimeError("Camera is still owned by PID(s): %s" % remaining)
    return owners


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_remote_dir(sftp: paramiko.SFTPClient, path: str) -> None:
    current = ""
    for part in path.strip("/").split("/"):
        current += "/" + part
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)


def deploy_project(
    client: paramiko.SSHClient,
    sftp: paramiko.SFTPClient,
    local_root: str,
    remote_dir: str,
    backup_dir: str,
    tmp_dir: str,
) -> int:
    """Atomic deploy: upload to tmp, verify SHA-256, back up previous, replace."""

    run(client, "rm -rf %s" % tmp_dir)
    ensure_remote_dir(sftp, tmp_dir)
    uploaded: List[Tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(local_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = os.path.relpath(dirpath, local_root)
        rdir = tmp_dir if rel == "." else posixpath.join(tmp_dir, rel.replace("\\", "/"))
        ensure_remote_dir(sftp, rdir)
        for filename in filenames:
            if os.path.splitext(filename)[1] in SKIP_EXT:
                continue
            local_file = os.path.join(dirpath, filename)
            remote_file = posixpath.join(rdir, filename)
            sftp.put(local_file, remote_file)
            uploaded.append((local_file, remote_file))

    mismatches = []
    for local_file, remote_file in uploaded:
        remote_sha = run(client, "sha256sum '%s' | cut -d' ' -f1" % remote_file)[0].strip()
        if sha256_file(local_file) != remote_sha:
            mismatches.append(remote_file)
    if mismatches:
        raise RuntimeError("deploy hash mismatch: %s" % mismatches[:5])

    run(
        client,
        "if [ -d '%s' ]; then rm -rf '%s'; mv '%s' '%s'; fi; mv '%s' '%s'"
        % (remote_dir, backup_dir, remote_dir, backup_dir, tmp_dir, remote_dir),
    )
    return len(uploaded)
