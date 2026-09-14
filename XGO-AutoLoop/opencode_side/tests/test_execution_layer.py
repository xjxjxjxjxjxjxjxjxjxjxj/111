from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock


EXECUTION_DIR = Path(__file__).resolve().parents[1]
if str(EXECUTION_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTION_DIR))

import dog_ssh  # noqa: E402
import supervisor  # noqa: E402


class DetachedLaunchTests(unittest.TestCase):
    def test_launch_backgrounds_exec_shell_not_cd_and_list(self):
        command = dog_ssh.build_detached_python_launch(
            "/home/pi/xgo-autoloop/current",
            "python3",
            "sign_line_closed_loop.py",
            ["--mode", "observe", "--pid-file", "/tmp/run/robot.pid"],
            "/tmp/run/robot.log",
        )

        self.assertTrue(command.startswith("nohup sh -c "))
        self.assertIn("exec env PYTHONPATH=/home/pi", command)
        self.assertIn("</dev/null", command)
        self.assertIn("2>&1 & printf", command)
        self.assertNotIn("PYTHONPATH=/home/pi nohup", command)


class RemoteProcessStateTests(unittest.TestCase):
    def test_run_specific_cmdline_is_alive(self):
        token = "/runtime/run-1/robot.pid"
        cmdline = "python3 -u sign_line_closed_loop.py --pid-file %s" % token
        with mock.patch.object(supervisor, "run", return_value=(cmdline, "", 0)):
            state, actual = supervisor._remote_process_state(object(), "123", token)
        self.assertEqual(state, "alive")
        self.assertEqual(actual, cmdline)

    def test_recycled_pid_is_mismatch(self):
        with mock.patch.object(
            supervisor, "run", return_value=("python3 unrelated.py", "", 0)
        ):
            state, _ = supervisor._remote_process_state(
                object(), "123", "/runtime/run-1/robot.pid"
            )
        self.assertEqual(state, "mismatch")

    def test_query_error_is_unknown_not_dead(self):
        with mock.patch.object(supervisor, "run", side_effect=OSError("ssh reset")):
            state, _ = supervisor._remote_process_state(
                object(), "123", "/runtime/run-1/robot.pid"
            )
        self.assertEqual(state, "unknown")

    def test_missing_proc_entry_is_gone(self):
        with mock.patch.object(supervisor, "run", return_value=("", "", 3)):
            state, _ = supervisor._remote_process_state(
                object(), "123", "/runtime/run-1/robot.pid"
            )
        self.assertEqual(state, "gone")


class CameraReleaseTests(unittest.TestCase):
    @staticmethod
    def _proc_reply(command, process_table):
        """Return mocked sudo /proc data used by camera safety tests."""

        for pid, info in process_table.items():
            marker = "/proc/%s" % pid
            if marker in command and "cmdline" in command:
                return (info["cmdline"], "", 0)
            if marker in command and "readlink -f" in command:
                return (info["cwd"], "", 0)
            if "ps -o ppid= -p %s" % pid in command:
                return (str(info["ppid"]), "", 0)
            if marker in command and command.rstrip().endswith("/stat"):
                fields = ["S", str(info["ppid"])] + ["0"] * 17 + [
                    str(info["starttime"])
                ]
                return ("%s (python3) %s" % (pid, " ".join(fields)), "", 0)
        return None

    def test_seven1_camera_owner_is_stopped_by_exact_pid(self):
        commands = []
        fuser_calls = 0
        alive = {"4321"}
        process_table = {
            "4321": {
                "cmdline": "python3 /home/pi/seven1.py",
                "cwd": "/home/pi",
                "ppid": "1",
                "starttime": "1001",
            }
        }

        def fake_run(_client, command):
            nonlocal fuser_calls
            commands.append(command)
            if "fuser /dev/video0" in command:
                fuser_calls += 1
                return (("4321\n", "", 0) if fuser_calls == 1 else ("", "", 1))
            reply = self._proc_reply(command, process_table)
            if reply is not None:
                return reply
            if "test -d /proc/4321" in command:
                return ("", "", 0 if "4321" in alive else 1)
            if "kill -INT 4321" in command:
                alive.discard("4321")
            return ("", "", 0)

        with mock.patch.object(dog_ssh, "run", side_effect=fake_run), mock.patch.object(
            dog_ssh.time, "sleep", return_value=None
        ):
            owners = dog_ssh.free_camera(object(), password="pw")

        self.assertEqual(owners[0]["pid"], "4321")
        self.assertTrue(any("kill -INT 4321" in item for item in commands))
        self.assertFalse(any("pkill" in item or "killall" in item for item in commands))

    def test_unknown_camera_owner_is_never_killed(self):
        commands = []
        process_table = {
            "9876": {
                "cmdline": "python3 /opt/unrelated_camera_service.py",
                "cwd": "/opt",
                "ppid": "1",
                "starttime": "1002",
            }
        }

        def fake_run(_client, command):
            commands.append(command)
            if "fuser /dev/video0" in command:
                return ("9876\n", "", 0)
            reply = self._proc_reply(command, process_table)
            if reply is not None:
                return reply
            return ("", "", 0)

        with mock.patch.object(dog_ssh, "run", side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, "unknown camera owner"):
                dog_ssh.free_camera(object(), password="pw")

        self.assertFalse(any("kill -" in item for item in commands))

    def test_cm4_main_child_camera_chain_is_stopped_top_down(self):
        commands = []
        fuser_calls = 0
        alive = {"664", "1159", "1160"}
        process_table = {
            "664": {
                "cmdline": "python3 main.py",
                "cwd": "/home/pi/RaspberryPi-CM4-main",
                "ppid": "1",
                "starttime": "2001",
            },
            "1159": {
                "cmdline": "sh -c python3 app/app_dogzilla.py",
                "cwd": "/home/pi/RaspberryPi-CM4-main",
                "ppid": "664",
                "starttime": "2002",
            },
            "1160": {
                "cmdline": "python3 app/app_dogzilla.py",
                "cwd": "/home/pi/RaspberryPi-CM4-main",
                "ppid": "1159",
                "starttime": "2003",
            },
        }

        def fake_run(_client, command):
            nonlocal fuser_calls
            commands.append(command)
            if "fuser /dev/video0" in command:
                fuser_calls += 1
                return (("1160\n", "", 0) if fuser_calls == 1 else ("", "", 1))
            reply = self._proc_reply(command, process_table)
            if reply is not None:
                return reply
            for pid in process_table:
                if "test -d /proc/%s" % pid in command:
                    return ("", "", 0 if pid in alive else 1)
                if "kill -INT %s" % pid in command:
                    alive.discard(pid)
                    return ("", "", 0)
            return ("", "", 0)

        with mock.patch.object(dog_ssh, "run", side_effect=fake_run), mock.patch.object(
            dog_ssh.time, "sleep", return_value=None
        ):
            owners = dog_ssh.free_camera(object(), password="pw")

        self.assertEqual([item["pid"] for item in owners], ["1160"])
        int_commands = [item for item in commands if "kill -INT" in item]
        self.assertEqual(len(int_commands), 3)
        self.assertIn("kill -INT 664", int_commands[0])
        self.assertIn("kill -INT 1159", int_commands[1])
        self.assertIn("kill -INT 1160", int_commands[2])
        self.assertFalse(any("pkill" in item or "killall" in item for item in commands))

    def test_same_app_basename_at_wrong_path_is_never_killed(self):
        commands = []
        process_table = {
            "2222": {
                "cmdline": "python3 /tmp/app/app_dogzilla.py",
                "cwd": "/tmp",
                "ppid": "1",
                "starttime": "3001",
            }
        }

        def fake_run(_client, command):
            commands.append(command)
            if "fuser /dev/video0" in command:
                return ("2222\n", "", 0)
            reply = self._proc_reply(command, process_table)
            if reply is not None:
                return reply
            return ("", "", 0)

        with mock.patch.object(dog_ssh, "run", side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, "unknown camera owner"):
                dog_ssh.free_camera(object(), password="pw")

        self.assertFalse(any("kill -" in item for item in commands))


class StartRunRemotePathTests(unittest.TestCase):
    def test_remote_video_parent_uses_posix_dirname(self):
        source = (EXECUTION_DIR / "start_run.py").read_text(encoding="utf-8")
        self.assertIn('posixpath.dirname(request["remote_video_partial"])', source)
        self.assertNotIn('Path(request["remote_video_partial"])', source)


class OrderingRegressionTests(unittest.TestCase):
    def test_supervisor_starts_heartbeat_before_pid_resolution(self):
        source = inspect.getsource(supervisor.main)
        heartbeat = source.index("_start_heartbeat_writer")
        pid_resolution = source.index("fallback_pid =")
        self.assertLess(heartbeat, pid_resolution)

    def test_controller_publishes_pid_before_opening_camera(self):
        controller = (
            EXECUTION_DIR.parent
            / "current"
            / "dog11_closed_loop"
            / "sign_line_closed_loop.py"
        )
        source = controller.read_text(encoding="utf-8")
        run_source = source[source.index("def run(args:") : source.index("def build_parser")]
        self.assertLess(run_source.index("pid_path.write_text"), run_source.index("open_camera("))


if __name__ == "__main__":
    unittest.main()
