#!/usr/bin/env python3
"""State controller for the OpenCode -> XGO -> Codex improvement loop.

The controller never connects to the robot and never runs GitHub commands.
Those are explicitly OpenCode responsibilities.  It creates auditable handoff
manifests, invokes Codex for both completed and problematic runs, validates the next
``dogN`` release, and deletes a video only after the report and tests succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


CST = timezone(timedelta(hours=8))
VERSION_RE = re.compile(r"^dog(?P<number>\d+)(?P<suffix>.*)$", re.IGNORECASE)
INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def now_cst() -> datetime:
    return datetime.now(CST)


def timestamp(dt: Optional[datetime] = None) -> str:
    return (dt or now_cst()).strftime("%Y%m%d-%H%M%S")


def iso_time(dt: Optional[datetime] = None) -> str:
    return (dt or now_cst()).isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_note(note: str, limit: int = 42) -> str:
    note = INVALID_FILENAME.sub("_", note.strip())
    note = re.sub(r"\s+", "_", note)
    note = note.strip("._ ") or "未填写备注"
    return note[:limit]


def infer_run_evidence(
    telemetry: Optional[Path], terminal_log: Optional[Path]
) -> Tuple[Optional[str], Optional[float]]:
    """Infer the acted-on sign colour and elapsed time from auditable run files.

    A terminal ``TRIGGER`` line is stronger evidence than raw per-frame labels.
    Telemetry is still used as a fallback and supplies the last elapsed time.
    """

    terminal_color: Optional[str] = None
    if terminal_log is not None and terminal_log.is_file():
        text = terminal_log.read_text(encoding="utf-8", errors="replace")
        triggers = re.findall(r"\bTRIGGER\s+(YELLOW|BLACK)\b", text, re.IGNORECASE)
        completions = re.findall(
            r"\b(YELLOW|BLACK)\s+COMPLETE\b", text, re.IGNORECASE
        )
        evidence = completions or triggers
        if evidence:
            terminal_color = evidence[-1].lower()

    counts = {"yellow": 0, "black": 0}
    elapsed_seconds: Optional[float] = None
    if telemetry is not None and telemetry.is_file():
        with telemetry.open("r", encoding="utf-8", errors="replace") as stream:
            for raw_line in stream:
                try:
                    record = json.loads(raw_line)
                except (json.JSONDecodeError, TypeError):
                    continue
                label = str(record.get("sign_label") or "").lower()
                if label in counts:
                    counts[label] += 1
                try:
                    elapsed_seconds = max(
                        elapsed_seconds or 0.0, float(record["elapsed_s"])
                    )
                except (KeyError, TypeError, ValueError):
                    pass

    telemetry_color = max(counts, key=counts.get) if max(counts.values()) > 0 else None
    return terminal_color or telemetry_color, elapsed_seconds


def resolve_codex_executable(configured: str) -> Path:
    """Resolve Codex CLI without assuming npm or the caller's PATH.

    Codex Desktop keeps its bundled CLI below LocalAppData in a directory whose
    build hash can change after an app update.  ``auto`` therefore checks PATH
    first and then selects the newest bundled executable instead of hardcoding
    one temporary build directory.
    """

    configured = os.path.expandvars(os.path.expanduser(configured.strip()))
    if configured and configured.lower() not in {"auto", "codex", "codex.exe"}:
        explicit = Path(configured).resolve()
        if explicit.is_file():
            return explicit
        found_explicit = shutil.which(configured)
        if found_explicit:
            return Path(found_explicit).resolve()
        raise FileNotFoundError("Configured Codex executable does not exist: %s" % configured)

    for command_name in ("codex", "codex.exe", "codex.cmd"):
        found = shutil.which(command_name)
        if found:
            return Path(found).resolve()

    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        local_root = Path(local_app_data)
        candidates.extend((local_root / "OpenAI" / "Codex" / "bin").glob("*/codex.exe"))
        candidates.append(local_root / "Microsoft" / "WindowsApps" / "codex.exe")
    candidates = [item for item in candidates if item.is_file()]
    if candidates:
        return max(candidates, key=lambda item: item.stat().st_mtime).resolve()

    raise FileNotFoundError(
        "Codex CLI not found in PATH or the Codex Desktop bundle. "
        "C:\\Users\\<name>\\.codex is a data/config directory, not the executable."
    )


def version_parts(version: str) -> Tuple[int, str]:
    match = VERSION_RE.match(version)
    if not match:
        raise ValueError("VERSION.json version must begin with dogN: %s" % version)
    return int(match.group("number")), match.group("suffix")


def next_version(version: str) -> str:
    number, suffix = version_parts(version)
    return "dog%d%s" % (number + 1, suffix)


class Workflow:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = read_json(self.config_path)
        self.paths = {key: Path(value).resolve() for key, value in self.config["paths"].items()}
        self.root = self.paths["root"]
        self.state_path = self.paths["state"] / "workflow_state.json"
        self.ready_path = self.paths["state"] / "READY_FOR_OPENCODE.json"

    def initialize(self) -> Dict[str, Any]:
        for key in ("root", "video_inbox", "reports", "releases", "work", "logs", "state"):
            self.paths[key].mkdir(parents=True, exist_ok=True)
        (self.root / "downloads").mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            state = {
                "status": "WAITING_FOR_START",
                "created_at": iso_time(),
                "next_poll_not_before_epoch": time.time(),
                "message": "准备好后在OpenCode中输入：启动",
            }
            atomic_write_json(self.state_path, state)
        return read_json(self.state_path)

    def state(self) -> Dict[str, Any]:
        return self.initialize()

    def write_state(self, **updates: Any) -> Dict[str, Any]:
        state = self.state()
        state.update(updates)
        state["updated_at"] = iso_time()
        atomic_write_json(self.state_path, state)
        return state

    def current_version(self) -> str:
        version_file = self.paths["current_project"] / "VERSION.json"
        if not version_file.exists():
            raise RuntimeError("Current project VERSION.json missing: %s" % version_file)
        return str(read_json(version_file)["version"])

    def doctor(self) -> Dict[str, Any]:
        """Verify that the configured Codex command can actually start."""

        configured = str(self.config["codex"].get("executable", "auto"))
        executable = resolve_codex_executable(configured)
        completed = subprocess.run(
            [str(executable), "--version"],
            cwd=str(self.root),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Codex executable was found but --version failed: %s"
                % (completed.stderr.strip() or completed.stdout.strip())
            )
        return {
            "status": "ok",
            "configured": configured,
            "resolved_executable": str(executable),
            "version": completed.stdout.strip(),
            "warning": completed.stderr.strip() or None,
        }

    def prepare_run(self, trigger: str) -> Dict[str, Any]:
        if trigger.strip() != "启动":
            raise RuntimeError("Only the exact user command 启动 can begin a robot run")
        state = self.state()
        if state["status"] not in {"WAITING_FOR_START", "READY_TO_RUN"}:
            raise RuntimeError("Workflow is not ready to run: %s" % state["status"])

        version = self.current_version()
        run_id = "%s_%s" % (version.split("_", 1)[0], timestamp())
        extension = str(self.config["recording"].get("extension", ".mp4"))
        if extension not in {".mp4", ".avi"}:
            raise ValueError("recording.extension must be .mp4 or .avi")
        remote_dir = str(self.config["robot"]["remote_recording_dir"]).rstrip("/")
        # 把 .partial 放在真正的扩展名前，确保 OpenCV 能按 .mp4/.avi 选择编码器。
        # OpenCode 仅在下载和哈希校验完成后，才将本地文件改成最终名称并生成 ready 清单。
        remote_video = "%s/%s.partial%s" % (remote_dir, run_id, extension)
        remote_video_final = "%s/%s%s" % (remote_dir, run_id, extension)
        remote_telemetry = "%s/%s.partial.jsonl" % (remote_dir, run_id)
        remote_telemetry_final = "%s/%s.jsonl" % (remote_dir, run_id)
        download_dir = self.root / "downloads" / run_id
        download_dir.mkdir(parents=True, exist_ok=False)
        request = {
            "run_id": run_id,
            "version": version,
            "created_at": iso_time(),
            "current_project": str(self.paths["current_project"]),
            "download_dir": str(download_dir),
            "remote_video_partial": remote_video,
            "remote_video_final": remote_video_final,
            "remote_telemetry_partial": remote_telemetry,
            "remote_telemetry_final": remote_telemetry_final,
            "robot": self.config["robot"],
            "recording": self.config["recording"],
        }
        request_path = self.paths["state"] / ("run_request_%s.json" % run_id)
        atomic_write_json(request_path, request)
        self.write_state(
            status="RUN_REQUESTED",
            active_run_id=run_id,
            run_request=str(request_path),
            message="OpenCode应部署、启动、下载录像，然后询问评价",
        )
        return request

    def ingest(
        self,
        video: Path,
        telemetry: Optional[Path],
        terminal_log: Optional[Path],
        rating: Optional[int],
        comment: str,
        sign_color: str = "auto",
        termination_reason: str = "goal_complete",
        elapsed_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        allowed_terminations = {
            "goal_complete",
            "normal_exit_incomplete",
            "timeout",
            "opencode_stop",
            "manual_stop",
            "safety_stop",
            "process_crash",
            "killed",
        }
        if termination_reason not in allowed_terminations:
            raise ValueError("unsupported termination reason: %s" % termination_reason)
        # Only a state-machine completion marker counts as a normal finish.
        # In the full course, stopping after StartZoneTracker confirms complete
        # entry is goal_complete. A user command entered in OpenCode is always
        # opencode_stop and therefore rating 2.
        automatically_failed = termination_reason != "goal_complete"
        if automatically_failed:
            # A forced/abnormal ending can never be recorded as a completion,
            # even when a stale UI selection accidentally supplied rating=1.
            rating = 2
        if rating not in (1, 2):
            raise ValueError("rating must be 1 (完赛) or 2 (有问题)")
        state = self.state()
        if state["status"] != "RUN_REQUESTED":
            raise RuntimeError("No active run is waiting for ingestion")
        video = video.resolve()
        if not video.is_file() or video.stat().st_size <= 0:
            raise RuntimeError("Downloaded video is missing or empty: %s" % video)

        telemetry = telemetry.resolve() if telemetry is not None else None
        terminal_log = terminal_log.resolve() if terminal_log is not None else None
        inferred_color, inferred_elapsed = infer_run_evidence(telemetry, terminal_log)
        if sign_color == "auto":
            sign_color = inferred_color or "unknown"
        if sign_color not in {"yellow", "black", "unknown"}:
            raise ValueError("sign_color must be auto, yellow, black, or unknown")
        if elapsed_seconds is None:
            elapsed_seconds = inferred_elapsed
        if elapsed_seconds is not None and elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative")
        if rating == 1 and sign_color not in {"yellow", "black"}:
            raise RuntimeError("Completed runs must identify the acted-on sign as yellow or black")
        if rating == 1 and elapsed_seconds is None:
            raise RuntimeError("Completed runs must include or infer total elapsed seconds")

        run_id = str(state["active_run_id"])
        version = self.current_version()
        sign_label = {"yellow": "黄牌", "black": "黑牌", "unknown": "路牌未知"}[
            sign_color
        ]
        label = (
            "评价1_完赛_%s" % sign_label
            if rating == 1
            else "评价2_%s_%s" % (sign_label, safe_note(comment or termination_reason))
        )
        target_video = self.paths["video_inbox"] / (
            "%s_%s%s" % (run_id, label, video.suffix.lower())
        )
        if target_video.exists():
            raise FileExistsError(target_video)
        shutil.copy2(video, target_video)
        if target_video.stat().st_size != video.stat().st_size:
            target_video.unlink(missing_ok=True)
            raise RuntimeError("Video size verification failed after copy")

        copied_telemetry = self._copy_optional(telemetry, run_id, ".jsonl")
        copied_terminal = self._copy_optional(terminal_log, run_id, ".terminal.log")
        metadata = {
            "run_id": run_id,
            "version": version,
            "rating": rating,
            "rating_label": "完赛" if rating == 1 else "有问题",
            "rating_was_automatic": automatically_failed,
            "sign_color": sign_color,
            "sign_color_label": sign_label,
            "termination_reason": termination_reason,
            "elapsed_seconds": elapsed_seconds,
            "comment": comment.strip(),
            "submitted_at": iso_time(),
            "video_path": str(target_video),
            "video_sha256": sha256(target_video),
            "video_bytes": target_video.stat().st_size,
            "telemetry_path": str(copied_telemetry) if copied_telemetry else None,
            "terminal_log_path": str(copied_terminal) if copied_terminal else None,
            "original_download_path": str(video),
        }
        manifest_path = target_video.with_suffix(target_video.suffix + ".ready.json")
        atomic_write_json(manifest_path, metadata)

        metadata["codex_goal"] = (
            "optimize_speed_while_preserving_completion"
            if rating == 1
            else "diagnose_and_fix_reported_problem"
        )
        atomic_write_json(manifest_path, metadata)
        self.write_state(
            status="WAITING_FOR_CODEX",
            rating_manifest=str(manifest_path),
            message=(
                "Codex将分析完赛耗时并生成下一dog提速候选"
                if rating == 1
                else "Codex将读取视频、写明原因并生成下一dog修复版"
            ),
        )
        return metadata

    def _copy_optional(self, source: Optional[Path], run_id: str, suffix: str) -> Optional[Path]:
        if source is None:
            return None
        source = source.resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise RuntimeError("Optional run artifact is missing or empty: %s" % source)
        target = self.paths["video_inbox"] / (run_id + suffix)
        shutil.copy2(source, target)
        return target

    def next_review_manifest(self) -> Optional[Path]:
        candidates = sorted(
            self.paths["video_inbox"].glob("*.ready.json"),
            key=lambda item: item.stat().st_mtime,
        )
        for path in candidates:
            data = read_json(path)
            if int(data.get("rating", 0)) in (1, 2) and not data.get("processed_at"):
                return path
        return None

    def _codex_prompt(
        self,
        manifest_path: Path,
        work_project: Path,
        report_path: Path,
        result_path: Path,
        old_version: str,
        new_version: str,
    ) -> str:
        extractor = self.root / "automation" / "extract_video_evidence.py"
        metadata = read_json(manifest_path)
        rating_instruction = (
            "本轮评价为1完赛。把完赛事实作为强约束，先分段统计耗时并找出可验证的速度瓶颈；只做保守提速，不能删掉多帧确认、动作等待、丢线停车或其他安全条件。"
            if int(metadata["rating"]) == 1
            else "本轮评价为2有问题。结合用户备注、视频时间点、遥测和代码写出直接原因与根因，并做最小修复。"
        )
        return f"""你是机械狗代码改进者Codex。只处理本轮文件，不执行SSH、部署或GitHub操作。

输入评价清单：{manifest_path}
当前代码的隔离工作副本：{work_project}
旧版本：{old_version}
要求新版本：{new_version}
诊断报告：{report_path}
机器可读结果：{result_path}
视频抽帧工具：{extractor}
本轮目标：{rating_instruction}

必须完成：
1. 读取评价JSON、视频、遥测和终端日志。先用抽帧工具生成接触表，再按异常附近时间点补看关键帧；不得只凭用户一句话猜原因。
2. 在报告中写：用户评价、视频SHA-256、关键时间点、观察证据、判断出的原因；评价1还要写分段耗时、提速瓶颈和预计收益，评价2要写直接原因与根因。
3. 只修改隔离工作副本。保留用户已有修改，写清中文或英文注释供其他AI读取；不得删除安全锁来让测试表面通过。
4. 运行现有全部测试，并为本次故障增加回归测试。测试失败就保留视频并在结果JSON写status=failed。
5. 成功时把工作副本VERSION.json的version改为精确的{new_version}，is_current_latest=true；不得跳号或复用旧号。
6. 成功时写入UTF-8 JSON：status=success、old_version、new_version、work_project、report_path、tests_passed=true、diagnosis_summary。失败时写status=failed及reason。
7. 不删除视频；控制器只有在验证报告、版本和测试结果后才会删除。
"""

    def process_one_with_codex(self) -> Optional[Dict[str, Any]]:
        manifest_path = self.next_review_manifest()
        if manifest_path is None:
            return None
        metadata = read_json(manifest_path)
        old_version = str(metadata["version"])
        if old_version != self.current_version():
            raise RuntimeError(
                "Video version %s does not match current code %s"
                % (old_version, self.current_version())
            )
        new_version = next_version(old_version)
        work_project = self.paths["work"] / new_version
        if work_project.exists():
            raise FileExistsError(
                "Work directory already exists; inspect it before retrying: %s" % work_project
            )
        shutil.copytree(
            self.paths["current_project"],
            work_project,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
        )
        report_dir = self.paths["reports"] / str(metadata["run_id"])
        report_dir.mkdir(parents=True, exist_ok=False)
        report_path = report_dir / "diagnosis.md"
        result_path = self.paths["state"] / ("codex_result_%s.json" % metadata["run_id"])
        prompt_path = self.paths["state"] / ("codex_prompt_%s.md" % metadata["run_id"])
        last_message = self.paths["logs"] / ("codex_%s.last.txt" % metadata["run_id"])
        prompt = self._codex_prompt(
            manifest_path, work_project, report_path, result_path, old_version, new_version
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        self.write_state(
            status="CODEX_RUNNING",
            codex_prompt=str(prompt_path),
            codex_result=str(result_path),
            message="Codex正在分析视频和修改隔离副本",
        )

        codex = str(
            resolve_codex_executable(
                str(self.config["codex"].get("executable", "auto"))
            )
        )
        sandbox = str(self.config["codex"].get("sandbox", "workspace-write"))
        approval = str(self.config["codex"].get("approval_policy", "never"))
        command = [
            codex,
            "-a",
            approval,
            "exec",
            "-C",
            str(self.root),
            "--skip-git-repo-check",
            "-s",
            sandbox,
            "-o",
            str(last_message),
            prompt,
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0 or not result_path.exists():
            self.write_state(
                status="CODEX_FAILED",
                message="Codex命令失败或未生成结果；视频已保留",
                codex_exit_code=completed.returncode,
            )
            return {"status": "failed", "reason": "codex invocation failed"}
        result = read_json(result_path)
        if result.get("status") != "success":
            self.write_state(
                status="CODEX_FAILED",
                message="Codex报告失败；视频已保留",
                codex_failure=result,
            )
            return result
        self._validate_codex_success(result, work_project, report_path, old_version, new_version)

        release_path = self.paths["releases"] / new_version
        if release_path.exists():
            raise FileExistsError(release_path)
        shutil.move(str(work_project), str(release_path))
        final_report = report_path
        telemetry = metadata.get("telemetry_path")
        terminal = metadata.get("terminal_log_path")
        for artifact in (telemetry, terminal):
            if artifact and Path(artifact).is_file():
                shutil.copy2(Path(artifact), report_dir / Path(artifact).name)

        # Deletion is intentionally last.  If anything above fails, evidence stays.
        video_path = Path(str(metadata["video_path"])).resolve()
        inbox = self.paths["video_inbox"]
        if bool(self.config["recording"].get("delete_local_video_after_codex_success", True)):
            if inbox != video_path and inbox not in video_path.parents:
                raise RuntimeError("Refusing to delete video outside configured inbox")
            video_path.unlink()
        metadata["processed_at"] = iso_time()
        metadata["video_deleted_after_success"] = not video_path.exists()
        metadata["diagnosis_report"] = str(final_report)
        atomic_write_json(manifest_path, metadata)

        ready = {
            "action": "deploy_and_push",
            "old_version": old_version,
            "new_version": new_version,
            "run_id": metadata["run_id"],
            "release_path": str(release_path),
            "diagnosis_report": str(final_report),
            "diagnosis_summary": result.get("diagnosis_summary", ""),
            "suggested_commit": "%s: %s" % (
                new_version,
                result.get(
                    "diagnosis_summary",
                    "根据实测视频提速" if int(metadata["rating"]) == 1 else "根据实测视频修复",
                ),
            ),
            "run_rating": int(metadata["rating"]),
            "run_sign_color": metadata.get("sign_color"),
            "run_sign_color_label": metadata.get("sign_color_label"),
            "run_elapsed_seconds": metadata.get("elapsed_seconds"),
            "termination_reason": metadata.get("termination_reason"),
            "completed_version": old_version if int(metadata["rating"]) == 1 else None,
            "completion_time_cst": metadata["submitted_at"] if int(metadata["rating"]) == 1 else None,
            "suggested_completion_tag": (
                "%s-complete-%s" % (old_version.split("_", 1)[0], timestamp())
                if int(metadata["rating"]) == 1
                else None
            ),
            "new_release_status": (
                "speed_candidate_pending_field_test"
                if int(metadata["rating"]) == 1
                else "fix_candidate_pending_field_test"
            ),
            "created_at": iso_time(),
        }
        atomic_write_json(self.ready_path, ready)
        self.write_state(
            status="READY_TO_DEPLOY",
            active_release=str(release_path),
            message="OpenCode应部署新版本、执行GitHub流程，然后ack-deploy",
        )
        return ready

    @staticmethod
    def _validate_codex_success(
        result: Dict[str, Any],
        work_project: Path,
        report_path: Path,
        old_version: str,
        new_version: str,
    ) -> None:
        if result.get("old_version") != old_version or result.get("new_version") != new_version:
            raise RuntimeError("Codex result version mismatch")
        if not bool(result.get("tests_passed")):
            raise RuntimeError("Codex did not confirm passing tests")
        if not report_path.is_file() or report_path.stat().st_size == 0:
            raise RuntimeError("Codex diagnosis report missing")
        actual_version = str(read_json(work_project / "VERSION.json")["version"])
        if actual_version != new_version:
            raise RuntimeError("Work project VERSION.json was not incremented correctly")

    def acknowledge_deploy(self, release: Path, github_commit: str) -> Dict[str, Any]:
        state = self.state()
        if state["status"] != "READY_TO_DEPLOY":
            raise RuntimeError("No Codex release is waiting for deployment")
        release = release.resolve()
        if str(release) != str(Path(state["active_release"]).resolve()):
            raise RuntimeError("Acknowledged release does not match active release")
        current = self.current_version()
        expected = str(read_json(release / "VERSION.json")["version"])
        if current != expected:
            raise RuntimeError(
                "OpenCode must first replace current_project with release; got %s expected %s"
                % (current, expected)
            )
        delay = int(self.config["codex"].get("initial_wait_seconds", 300))
        state = self.write_state(
            status="WAITING_FOR_START",
            deployed_version=expected,
            github_commit=github_commit,
            next_poll_not_before_epoch=time.time() + delay,
            message="新版本已部署；准备好后在OpenCode中输入：启动",
        )
        self.ready_path.unlink(missing_ok=True)
        return state

def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(name: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True, type=Path)
        return sub

    common("init")
    common("doctor")
    common("status")
    prepare = common("prepare-run")
    prepare.add_argument("--trigger", required=True)
    ingest = common("ingest")
    ingest.add_argument("--video", required=True, type=Path)
    ingest.add_argument("--telemetry", type=Path)
    ingest.add_argument("--terminal-log", type=Path)
    ingest.add_argument("--rating", type=int, choices=(1, 2))
    ingest.add_argument("--comment", default="")
    ingest.add_argument(
        "--sign-color", choices=("auto", "yellow", "black", "unknown"), default="auto"
    )
    ingest.add_argument(
        "--termination",
        required=True,
        choices=(
            "goal_complete",
            "normal_exit_incomplete",
            "timeout",
            "opencode_stop",
            "manual_stop",
            "safety_stop",
            "process_crash",
            "killed",
        ),
    )
    ingest.add_argument("--elapsed-seconds", type=float)
    watch = common("watch")
    watch.add_argument("--once", action="store_true")
    acknowledge = common("ack-deploy")
    acknowledge.add_argument("--release", required=True, type=Path)
    acknowledge.add_argument("--github-commit", required=True)
    return parser


def run_watch(workflow: Workflow, once: bool) -> int:
    lock_path = workflow.paths["state"] / "codex_watcher.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Codex watcher is already running: %s" % lock_path)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.close(descriptor)
    poll_seconds = int(workflow.config["codex"].get("poll_seconds", 100))
    try:
        while True:
            state = workflow.state()
            wait_until = float(state.get("next_poll_not_before_epoch", 0.0))
            if time.time() < wait_until:
                wait_s = min(poll_seconds, max(1, int(wait_until - time.time())))
                if once:
                    print_json({"status": "waiting_initial_delay", "seconds": wait_s})
                    return 0
                time.sleep(wait_s)
                continue
            if state.get("status") == "WAITING_FOR_CODEX":
                result = workflow.process_one_with_codex()
                if result is not None:
                    print_json(result)
            elif once:
                print_json({"status": state.get("status"), "message": "no rated video ready"})
                return 0
            if once:
                return 0
            time.sleep(poll_seconds)
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> int:
    args = build_parser().parse_args()
    workflow = Workflow(args.config)
    if args.command == "init":
        print_json(workflow.initialize())
    elif args.command == "doctor":
        print_json(workflow.doctor())
    elif args.command == "status":
        print_json(workflow.state())
    elif args.command == "prepare-run":
        print_json(workflow.prepare_run(args.trigger))
    elif args.command == "ingest":
        print_json(
            workflow.ingest(
                args.video,
                args.telemetry,
                args.terminal_log,
                args.rating,
                args.comment,
                args.sign_color,
                args.termination,
                args.elapsed_seconds,
            )
        )
    elif args.command == "watch":
        return run_watch(workflow, args.once)
    elif args.command == "ack-deploy":
        print_json(workflow.acknowledge_deploy(args.release, args.github_commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
