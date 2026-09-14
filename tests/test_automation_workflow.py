import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from automation.workflow_controller import (
    Workflow,
    next_version,
    resolve_codex_executable,
    safe_note,
)
from run_recording import RunRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TEMP_ROOT = PROJECT_ROOT / "tests" / "_workflow_tmp"


@contextmanager
def test_directory():
    path = TEST_TEMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class WorkflowControllerTests(unittest.TestCase):
    def make_workflow(self, root: Path) -> Workflow:
        current = root / "current" / "dog11_closed_loop"
        current.mkdir(parents=True)
        (current / "VERSION.json").write_text(
            json.dumps({"version": "dog11_closed_loop"}), encoding="utf-8"
        )
        config = {
            "paths": {
                "root": str(root),
                "current_project": str(current),
                "video_inbox": str(root / "video_inbox"),
                "reports": str(root / "reports"),
                "releases": str(root / "releases"),
                "work": str(root / "work"),
                "logs": str(root / "logs"),
                "state": str(root / "state"),
            },
            "robot": {
                "remote_recording_dir": "/tmp/xgo",
                "entry_point": "sign_line_closed_loop.py",
            },
            "recording": {
                "extension": ".mp4",
                "delete_local_video_after_codex_success": True,
            },
            "codex": {"initial_wait_seconds": 300, "poll_seconds": 100},
        }
        config_path = root / "workflow_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        workflow = Workflow(config_path)
        workflow.initialize()
        return workflow

    def test_dog_version_increments_exactly_once(self):
        self.assertEqual(next_version("dog11_closed_loop"), "dog12_closed_loop")

    def test_filename_note_removes_windows_reserved_characters(self):
        value = safe_note('路牌:提前/报警? "测试"')
        self.assertNotRegex(value, r'[<>:"/\\|?*]')

    def test_explicit_codex_executable_is_resolved_without_path(self):
        with test_directory() as root:
            executable = root / "codex-test.exe"
            executable.write_bytes(b"placeholder")
            self.assertEqual(resolve_codex_executable(str(executable)), executable.resolve())

    def test_only_exact_start_command_creates_run_request(self):
        with test_directory() as directory:
            workflow = self.make_workflow(directory)
            with self.assertRaises(RuntimeError):
                workflow.prepare_run("开始")
            request = workflow.prepare_run("启动")
            self.assertEqual(request["version"], "dog11_closed_loop")
            self.assertTrue(request["remote_video_partial"].endswith(".partial.mp4"))
            self.assertTrue(request["remote_video_final"].endswith(".mp4"))
            self.assertNotIn(".partial", Path(request["remote_video_final"]).name)

    def test_rating_one_still_routes_to_codex_for_speed_optimization(self):
        with test_directory() as root:
            workflow = self.make_workflow(root)
            workflow.prepare_run("启动")
            video = root / "download.mp4"
            video.write_bytes(b"not-empty-test-video")
            telemetry = root / "run.jsonl"
            telemetry.write_text(
                '{"elapsed_s": 42.5, "sign_label": "yellow"}\n', encoding="utf-8"
            )
            metadata = workflow.ingest(video, telemetry, None, 1, "完赛但需要更快")
            self.assertEqual(metadata["codex_goal"], "optimize_speed_while_preserving_completion")
            self.assertEqual(metadata["sign_color"], "yellow")
            self.assertEqual(metadata["sign_color_label"], "黄牌")
            self.assertEqual(metadata["elapsed_seconds"], 42.5)
            self.assertIn("完赛_黄牌", Path(metadata["video_path"]).name)
            self.assertEqual(workflow.state()["status"], "WAITING_FOR_CODEX")

    def test_rating_two_routes_to_codex_for_problem_fix(self):
        with test_directory() as root:
            workflow = self.make_workflow(root)
            workflow.prepare_run("启动")
            video = root / "download.mp4"
            video.write_bytes(b"not-empty-test-video")
            metadata = workflow.ingest(video, None, None, 2, "黑牌识别错误")
            self.assertEqual(metadata["codex_goal"], "diagnose_and_fix_reported_problem")
            self.assertEqual(workflow.state()["status"], "WAITING_FOR_CODEX")

    def test_forced_termination_automatically_becomes_rating_two(self):
        with test_directory() as root:
            workflow = self.make_workflow(root)
            workflow.prepare_run("启动")
            video = root / "download.mp4"
            video.write_bytes(b"not-empty-test-video")
            metadata = workflow.ingest(
                video,
                None,
                None,
                1,
                "用户原先误点了完赛",
                sign_color="black",
                termination_reason="opencode_stop",
                elapsed_seconds=12.0,
            )
            self.assertEqual(metadata["rating"], 2)
            self.assertTrue(metadata["rating_was_automatic"])
            self.assertEqual(metadata["termination_reason"], "opencode_stop")

    def test_visual_goal_completion_stop_is_not_forced_to_rating_two(self):
        with test_directory() as root:
            workflow = self.make_workflow(root)
            workflow.prepare_run("启动")
            video = root / "download.mp4"
            video.write_bytes(b"not-empty-test-video")
            metadata = workflow.ingest(
                video,
                None,
                None,
                1,
                "状态机确认进入出发区后自行停车",
                sign_color="yellow",
                termination_reason="goal_complete",
                elapsed_seconds=88.0,
            )
            self.assertEqual(metadata["rating"], 1)
            self.assertFalse(metadata["rating_was_automatic"])

    def test_completed_run_without_sign_colour_is_rejected(self):
        with test_directory() as root:
            workflow = self.make_workflow(root)
            workflow.prepare_run("启动")
            video = root / "download.mp4"
            video.write_bytes(b"not-empty-test-video")
            with self.assertRaises(RuntimeError):
                workflow.ingest(
                    video,
                    None,
                    None,
                    1,
                    "完赛",
                    sign_color="unknown",
                    elapsed_seconds=30.0,
                )


class RunRecorderTests(unittest.TestCase):
    def test_telemetry_is_utf8_jsonl(self):
        with test_directory() as directory:
            telemetry = directory / "run.jsonl"
            recorder = RunRecorder(None, telemetry, 20.0, (320, 240))
            recorder.write_frame(np.zeros((240, 320, 3), dtype=np.uint8))
            recorder.write_telemetry({"state": "巡线", "distance_cm": 20.0})
            recorder.close()
            record = json.loads(telemetry.read_text(encoding="utf-8"))
            self.assertEqual(record["state"], "巡线")


if __name__ == "__main__":
    unittest.main()
