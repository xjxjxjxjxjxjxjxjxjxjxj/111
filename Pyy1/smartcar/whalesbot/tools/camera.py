#!/usr/bin/python3
# -*- coding: utf-8 -*-

import threading
from multiprocessing import Process
import time
import math
import re
import shutil
import subprocess

import cv2
import numpy as np
import platform
import os, sys


try:
    from .log_wrap import logger
except ImportError:
    from log_wrap import logger


# ============================================================
# 摄像头物理 USB 路径映射
# ============================================================
#
# 注意：
# 这里的 0 和 2 仍然是你原来程序中的“摄像头编号”。
#
# 你的 YAML 不需要修改：
#
# camera:
#   front: 0
#   side: 2
#
# 映射关系：
#
#   程序编号 0
#       ↓
#   USB 物理端口 2.1
#       ↓
#   /dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0
#       ↓
#   当前对应 /dev/video0
#
#   程序编号 2
#       ↓
#   USB 物理端口 2.4
#       ↓
#   /dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0
#       ↓
#   当前对应 /dev/video2
#
# 使用物理路径以后，即使 /dev/video0、/dev/video2
# 因为摄像头重新插拔导致编号变化，只要 USB 物理端口
# 不变，程序仍然可以找到对应摄像头。
#
# ============================================================

PHYSICAL_CAMERA_PATHS = {
    0: "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0",
    2: "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0",
}


class _V4L2Controls:
    """Small, fail-closed wrapper around v4l2-ctl.

    This helper is constructed only when the per-camera AE switch is 1.  That
    property is important: switch 0 must not even probe an exposure control.
    """

    CTRL_RE = re.compile(
        r"^\s*([A-Za-z0-9_]+)\s+0x[0-9a-fA-F]+\s+\(([^)]+)\)\s*:\s*(.*)$"
    )
    MENU_RE = re.compile(r"^\s*(-?\d+)\s*:\s*(.+?)\s*$")
    KV_RE = re.compile(r"(min|max|step|default|value)=(-?\d+)")

    def __init__(self, device, log_prefix="AE"):
        self.device = device
        self.prefix = log_prefix
        self.exe = shutil.which("v4l2-ctl")
        self.info = {}
        self.last_error = ""

    def _run(self, *args, **kwargs):
        timeout = float(kwargs.get("timeout", 1.0))
        if not self.exe:
            self.last_error = "v4l2-ctl not found"
            return False, self.last_error

        try:
            process = subprocess.run(
                [self.exe, "-d", self.device] + list(args),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=timeout,
                check=False,
            )

            message = (process.stdout or "") + (process.stderr or "")
            self.last_error = (
                "" if process.returncode == 0 else message.strip()
            )

            return process.returncode == 0, message.strip()

        except Exception as exc:
            self.last_error = repr(exc)
            return False, repr(exc)

    def refresh(self):
        ok, output = self._run(
            "--list-ctrls-menus",
            timeout=2.0
        )

        if not ok:
            return False

        info = {}
        current = None

        for line in output.splitlines():

            match = self.CTRL_RE.match(line)

            if match:
                current = match.group(1)

                meta = {
                    "type": match.group(2).strip().lower(),
                    "menu": {},
                }

                for key, value in self.KV_RE.findall(match.group(3)):
                    meta[key] = int(value)

                info[current] = meta
                continue

            match = self.MENU_RE.match(line)

            if (
                current
                and match
                and "menu" in info[current]["type"]
            ):
                info[current]["menu"][
                    int(match.group(1))
                ] = match.group(2).strip()

        self.info = info

        return True

    def has(self, name):
        return name in self.info

    def menu_value(self, name, required_words):
        words = tuple(
            str(word).lower()
            for word in required_words
        )

        menu = self.info.get(
            name,
            {}
        ).get(
            "menu",
            {}
        )

        for raw_value, label in menu.items():

            lowered = label.lower()

            if all(
                word in lowered
                for word in words
            ):
                return raw_value

        return None

    def get(self, name, timeout=1.0):
        ok, output = self._run(
            "--get-ctrl={}".format(name),
            timeout=timeout
        )

        if not ok:
            return None

        match = re.search(
            r":\s*(-?\d+)\b",
            output
        )

        return (
            int(match.group(1))
            if match
            else None
        )

    def set(
        self,
        name,
        value,
        verify=False,
        timeout=1.0
    ):
        value = int(value)

        ok, _ = self._run(
            "--set-ctrl={}={}".format(
                name,
                value
            ),
            timeout=timeout
        )

        if not ok:
            return False

        return (
            (not verify)
            or self.get(
                name,
                timeout=timeout
            ) == value
        )

    def quantize(
        self,
        name,
        value,
        low=None,
        high=None
    ):
        meta = self.info[name]

        lower = (
            meta.get("min", int(value))
            if low is None
            else int(low)
        )

        upper = (
            meta.get("max", int(value))
            if high is None
            else int(high)
        )

        step = max(
            1,
            int(
                meta.get(
                    "step",
                    1
                )
            )
        )

        value = max(
            lower,
            min(
                upper,
                int(round(value))
            )
        )

        quantized = (
            lower
            + int(
                round(
                    float(value - lower)
                    / step
                )
            )
            * step
        )

        return max(
            lower,
            min(
                upper,
                quantized
            )
        )


class _SoftwareAE:
    """Low-frequency software auto exposure for one independent camera."""

    def __init__(
        self,
        device,
        cfg,
        name="camera"
    ):
        self.name = name
        self.cfg = dict(cfg or {})

        self.ctl = _V4L2Controls(
            device,
            "AE-{}".format(name)
        )

        self.enabled = False
        self.needs_restore = False
        self.restore_state = "not_needed"
        self.start_error = ""
        self.snapshot = {}

        self.auto_name = None
        self.exp_name = None
        self.priority_name = None
        self.manual_value = None

        self.exp = None
        self.gain = None

        self.exp_min = None
        self.exp_max = None

        self.gain_min = None
        self.gain_max = None

        self.fail_count = 0
        self.write_count = 0

        self.ema = None
        self.last_eval_seq = -10 ** 9

        self.hold = 0
        self.last_dir = 0
        self.outside_count = 0

        self.last_log = 0.0

    def _find_driver_auto_value(self):
        if not self.auto_name:
            return None

        for words in (
            ("aperture", "priority"),
            ("auto", "mode"),
            ("auto",)
        ):
            value = self.ctl.menu_value(
                self.auto_name,
                words
            )

            if value is not None:
                return value

        return None

    def _restore(self):

        self.restore_state = "failed"

        try:
            all_ok = True

            old_auto = self.snapshot.get(
                self.auto_name
            )

            if old_auto is None:
                all_ok = False

            elif self.manual_value is None:
                all_ok = False

            else:

                ok = self.ctl.set(
                    self.auto_name,
                    self.manual_value,
                    verify=True,
                    timeout=1.0
                )

                all_ok = ok and all_ok

            for name in (
                self.exp_name,
                "gain"
            ):

                old_value = self.snapshot.get(
                    name
                )

                if (
                    name
                    and old_value is not None
                ):

                    if not self.ctl.has(name):
                        all_ok = False

                    else:

                        ok = self.ctl.set(
                            name,
                            old_value,
                            verify=False,
                            timeout=1.0
                        )

                        all_ok = ok and all_ok

            for name in (
                "power_line_frequency",
                self.priority_name
            ):

                if not name:
                    continue

                old_value = self.snapshot.get(
                    name
                )

                if old_value is not None:

                    if not self.ctl.has(name):
                        all_ok = False

                    else:

                        ok = self.ctl.set(
                            name,
                            old_value,
                            verify=False,
                            timeout=1.0
                        )

                        all_ok = ok and all_ok

            auto_ok = (
                old_auto is not None
                and self.ctl.set(
                    self.auto_name,
                    old_auto,
                    verify=True,
                    timeout=1.0
                )
            )

            if auto_ok and all_ok:

                self.needs_restore = False
                self.restore_state = "original"

                return True

            fallback_auto = (
                self._find_driver_auto_value()
            )

            if (
                fallback_auto is not None
                and self.ctl.set(
                    self.auto_name,
                    fallback_auto,
                    verify=True,
                    timeout=1.0
                )
            ):

                self.needs_restore = True
                self.restore_state = "fallback_auto"

                return False

        except Exception:
            pass

        self.needs_restore = True
        self.restore_state = "failed"

        return False

    def _fail_start(self, reason):

        self.enabled = False
        self.start_error = str(reason)

        if self.needs_restore:
            self._restore()

        print(
            "[AE][{}] OFF: {}; restore={}".format(
                self.name,
                self.start_error,
                self.restore_state
            )
        )

        return False

    def start(self):

        try:
            return self._start_impl()

        except Exception as exc:

            return self._fail_start(
                "start exception: {!r}".format(exc)
            )

    def _start_impl(self):

        roi = self.cfg.get(
            "roi",
            [
                0.08,
                0.08,
                0.92,
                0.92
            ]
        )

        if not isinstance(
            roi,
            (list, tuple)
        ) or len(roi) != 4:

            raise ValueError(
                "roi must contain four normalized numbers"
            )

        x0, y0, x1, y1 = [
            float(value)
            for value in roi
        ]

        if not (
            0.0 <= x0 < x1 <= 1.0
            and
            0.0 <= y0 < y1 <= 1.0
        ):

            raise ValueError(
                "invalid roi order/range"
            )

        alpha = float(
            self.cfg.get(
                "ema_alpha",
                0.20
            )
        )

        if not (
            0.0 < alpha <= 1.0
        ):

            raise ValueError(
                "ema_alpha must be in (0, 1]"
            )

        target = float(
            self.cfg.get(
                "target_luma",
                115
            )
        )

        deadband = float(
            self.cfg.get(
                "deadband",
                8
            )
        )

        if (
            not math.isfinite(target)
            or not (
                1.0 <= target <= 254.0
            )
        ):

            raise ValueError(
                "target_luma must be finite and in [1, 254]"
            )

        if (
            not math.isfinite(deadband)
            or not (
                0.0 <= deadband < 128.0
            )
        ):

            raise ValueError(
                "deadband must be finite and in [0, 128)"
            )

        for key, default in (
            ("highlight_ratio", 0.15),
            ("shadow_ratio", 0.35)
        ):

            value = float(
                self.cfg.get(
                    key,
                    default
                )
            )

            if (
                not math.isfinite(value)
                or not (
                    0.0 <= value <= 1.0
                )
            ):

                raise ValueError(
                    "{} must be in [0, 1]".format(key)
                )

        runtime_timeout = float(
            self.cfg.get(
                "runtime_control_timeout",
                0.12
            )
        )

        if (
            not math.isfinite(runtime_timeout)
            or not (
                0.05 <= runtime_timeout <= 0.50
            )
        ):

            raise ValueError(
                "runtime_control_timeout must be in [0.05, 0.50]"
            )

        for key in (
            "adjust_every_frames",
            "outside_need",
            "settle_frames"
        ):

            if int(
                self.cfg.get(
                    key,
                    1
                )
            ) < 1:

                raise ValueError(
                    "{} must be >= 1".format(key)
                )

        if not self.ctl.refresh():

            print(
                "[AE][{}] OFF: v4l2 controls unavailable".format(
                    self.name
                )
            )

            return False

        self.exp_name = next(
            (
                name
                for name in (
                    "exposure_absolute",
                    "exposure_time_absolute"
                )
                if self.ctl.has(name)
            ),
            None
        )

        self.auto_name = next(
            (
                name
                for name in (
                    "exposure_auto",
                    "auto_exposure"
                )
                if self.ctl.has(name)
            ),
            None
        )

        self.priority_name = next(
            (
                name
                for name in (
                    "exposure_auto_priority",
                    "exposure_dynamic_framerate"
                )
                if self.ctl.has(name)
            ),
            None
        )

        manual_value = (
            self.ctl.menu_value(
                self.auto_name,
                ("manual",)
            )
            if self.auto_name
            else None
        )

        self.manual_value = manual_value

        adjust_gain = int(
            self.cfg.get(
                "adjust_gain",
                0
            )
        ) == 1

        if (
            self.exp_name is None
            or manual_value is None
        ):

            print(
                "[AE][{}] OFF: manual/exposure control unsupported".format(
                    self.name
                )
            )

            return False

        max_override = self.cfg.get(
            "exposure_max_override"
        )

        if max_override is None:

            print(
                "[AE][{}] OFF: exposure_max_override is required".format(
                    self.name
                )
            )

            return False

        names = (
            self.auto_name,
            self.exp_name,
            "gain",
            "power_line_frequency",
            self.priority_name
        )

        self.snapshot = {
            name: self.ctl.get(name)
            for name in names
            if name and self.ctl.has(name)
        }

        if self.snapshot.get(
            self.auto_name
        ) is None:

            print(
                "[AE][{}] OFF: cannot snapshot {}".format(
                    self.name,
                    self.auto_name
                )
            )

            return False

        if self.snapshot.get(
            self.exp_name
        ) is None:

            print(
                "[AE][{}] OFF: cannot snapshot original exposure".format(
                    self.name
                )
            )

            return False

        if (
            adjust_gain
            and self.snapshot.get("gain") is None
        ):

            print(
                "[AE][{}] OFF: cannot snapshot original gain".format(
                    self.name
                )
            )

            return False

        self.needs_restore = True

        if not self.ctl.set(
            self.auto_name,
            manual_value,
            verify=True,
            timeout=1.0
        ):

            return self._fail_start(
                "cannot enter manual exposure"
            )

        if int(
            self.cfg.get(
                "set_50hz",
                1
            )
        ) == 1:

            hz50 = self.ctl.menu_value(
                "power_line_frequency",
                ("50", "hz")
            )

            old_power_line = self.snapshot.get(
                "power_line_frequency"
            )

            if (
                hz50 is not None
                and old_power_line is not None
            ):

                if not self.ctl.set(
                    "power_line_frequency",
                    hz50,
                    verify=True,
                    timeout=1.0
                ):

                    print(
                        "[AE][{}] WARN: cannot verify 50 Hz control".format(
                            self.name
                        )
                    )

            elif hz50 is not None:

                print(
                    "[AE][{}] WARN: skip 50 Hz; original value unreadable".format(
                        self.name
                    )
                )

        if self.priority_name:

            if self.snapshot.get(
                self.priority_name
            ) is None:

                print(
                    "[AE][{}] WARN: skip dynamic frame control; original value unreadable".format(
                        self.name
                    )
                )

            elif not self.ctl.set(
                self.priority_name,
                0,
                verify=True,
                timeout=1.0
            ):

                print(
                    "[AE][{}] WARN: cannot verify {}=0".format(
                        self.name,
                        self.priority_name
                    )
                )

        self.exp = self.ctl.get(
            self.exp_name
        )

        if self.exp is None:

            return self._fail_start(
                "cannot read current exposure"
            )

        exp_meta = self.ctl.info[
            self.exp_name
        ]

        self.exp_min = int(
            exp_meta["min"]
        )

        self.exp_max = min(
            int(exp_meta["max"]),
            int(max_override)
        )

        if self.exp_max < self.exp_min:

            return self._fail_start(
                "invalid exposure range"
            )

        self.exp = self.ctl.quantize(
            self.exp_name,
            self.exp,
            self.exp_min,
            self.exp_max
        )

        if not self.ctl.set(
            self.exp_name,
            self.exp,
            verify=True,
            timeout=1.0
        ):

            return self._fail_start(
                "cannot initialize exposure"
            )

        if self.ctl.has("gain"):

            self.gain = self.ctl.get(
                "gain"
            )

            gain_meta = self.ctl.info[
                "gain"
            ]

            self.gain_min = int(
                gain_meta.get(
                    "min",
                    0
                )
            )

            self.gain_max = int(
                gain_meta.get(
                    "max",
                    self.gain or 0
                )
            )

            gain_override = self.cfg.get(
                "gain_max_override"
            )

            if gain_override is not None:

                self.gain_max = min(
                    self.gain_max,
                    int(gain_override)
                )

        if adjust_gain:

            if self.gain is None:

                return self._fail_start(
                    "adjust_gain=1 but gain is unreadable"
                )

            if self.cfg.get(
                "gain_max_override"
            ) is None:

                return self._fail_start(
                    "adjust_gain=1 requires gain_max_override"
                )

            if self.gain_max < self.gain_min:

                return self._fail_start(
                    "invalid gain range"
                )

            self.gain = self.ctl.quantize(
                "gain",
                self.gain,
                self.gain_min,
                self.gain_max
            )

            if not self.ctl.set(
                "gain",
                self.gain,
                verify=True,
                timeout=1.0
            ):

                return self._fail_start(
                    "cannot initialize gain"
                )

        self.enabled = True

        print(
            "[AE][{}] ON exp={} range={}..{} gain={}".format(
                self.name,
                self.exp,
                self.exp_min,
                self.exp_max,
                self.gain
            )
        )

        return True

    def stop_with_fallback(
        self,
        reason
    ):

        self.enabled = False

        restored = True

        if self.needs_restore:
            restored = self._restore()

        print(
            "[AE][{}] disabled; restore={}; reason={}".format(
                self.name,
                self.restore_state,
                reason
            )
        )

        return restored

    def disable_runtime(
        self,
        reason
    ):

        self.enabled = False
        self.needs_restore = True

        fallback_state = "failed"

        try:

            fallback_auto = self.snapshot.get(
                self.auto_name
            )

            if (
                fallback_auto is None
                or fallback_auto == self.manual_value
            ):

                fallback_auto = (
                    self._find_driver_auto_value()
                )

            timeout = float(
                self.cfg.get(
                    "runtime_control_timeout",
                    0.12
                )
            )

            timeout = max(
                0.05,
                min(
                    0.25,
                    timeout
                )
            )

            if (
                fallback_auto is not None
                and self.ctl.set(
                    self.auto_name,
                    fallback_auto,
                    verify=False,
                    timeout=timeout
                )
            ):

                fallback_state = "driver_auto"
                self.restore_state = "fallback_auto"

        except Exception:

            fallback_state = "failed"

        print(
            "[AE][{}] runtime disabled; fallback={}; "
            "full_restore=deferred; reason={}".format(
                self.name,
                fallback_state,
                reason
            )
        )

    def restore_before_release(self):

        if not self.needs_restore:
            return True

        return self._restore()

    def _measure(
        self,
        frame
    ):

        height, width = frame.shape[:2]

        x0, y0, x1, y1 = self.cfg.get(
            "roi",
            [
                0.08,
                0.08,
                0.92,
                0.92
            ]
        )

        px0 = max(
            0,
            min(
                width,
                int(
                    width * float(x0)
                )
            )
        )

        px1 = max(
            0,
            min(
                width,
                int(
                    width * float(x1)
                )
            )
        )

        py0 = max(
            0,
            min(
                height,
                int(
                    height * float(y0)
                )
            )
        )

        py1 = max(
            0,
            min(
                height,
                int(
                    height * float(y1)
                )
            )
        )

        roi = frame[
            py0:py1,
            px0:px1
        ]

        if roi.size == 0:
            return None

        value_channel = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )[:, :, 2]

        scale = min(
            1.0,
            160.0 / max(
                1,
                value_channel.shape[1]
            )
        )

        if scale < 1.0:

            value_channel = cv2.resize(
                value_channel,
                (
                    max(
                        8,
                        int(
                            value_channel.shape[1]
                            * scale
                        )
                    ),
                    max(
                        8,
                        int(
                            value_channel.shape[0]
                            * scale
                        )
                    )
                ),
                interpolation=cv2.INTER_AREA
            )

        pixels = (
            value_channel
            .reshape(-1)
            .astype(np.float32)
        )

        p10, p50, p90, p98 = np.percentile(
            pixels,
            [
                10,
                50,
                90,
                98
            ]
        )

        trimmed = pixels[
            (pixels >= p10)
            &
            (pixels <= p90)
        ]

        trimmed_mean = (
            float(trimmed.mean())
            if trimmed.size
            else float(p50)
        )

        score = (
            0.65 * float(p50)
            +
            0.35 * trimmed_mean
        )

        return (
            score,
            float(p90),
            float(p98),
            float(np.mean(pixels >= 250)),
            float(np.mean(pixels <= 8))
        )

    def _write(
        self,
        name,
        value,
        low,
        high
    ):

        value = self.ctl.quantize(
            name,
            value,
            low,
            high
        )

        timeout = float(
            self.cfg.get(
                "runtime_control_timeout",
                0.12
            )
        )

        timeout = max(
            0.05,
            min(
                0.50,
                timeout
            )
        )

        if self.ctl.set(
            name,
            value,
            verify=False,
            timeout=timeout
        ):

            self.fail_count = 0
            self.write_count += 1

            if name == self.exp_name:
                self.exp = value

            elif name == "gain":
                self.gain = value

            if self.write_count % 10 == 0:

                actual = self.ctl.get(
                    name,
                    timeout=timeout
                )

                if actual is not None:

                    if name == self.exp_name:
                        self.exp = actual

                    elif name == "gain":
                        self.gain = actual

            return True

        self.fail_count += 1

        self.disable_runtime(
            "{} runtime set failed: {}".format(
                name,
                self.ctl.last_error
            )
        )

        return False

    def _adjust(
        self,
        direction,
        error
    ):

        deadband = float(
            self.cfg.get(
                "deadband",
                8
            )
        )

        severity = min(
            1.0,
            max(
                0.0,
                (
                    abs(error)
                    - deadband
                )
                / 64.0
            )
        )

        ratio = (
            0.04
            +
            0.08 * severity
        )

        adjust_gain = int(
            self.cfg.get(
                "adjust_gain",
                0
            )
        ) == 1

        if direction > 0:

            if self.exp < self.exp_max:

                step = int(
                    self.ctl.info[
                        self.exp_name
                    ].get(
                        "step",
                        1
                    )
                )

                delta = max(
                    step,
                    int(
                        round(
                            max(
                                1,
                                self.exp
                            )
                            * ratio
                        )
                    )
                )

                ok = self._write(
                    self.exp_name,
                    self.exp + delta,
                    self.exp_min,
                    self.exp_max
                )

                return (
                    ok,
                    "exp_up"
                )

            if (
                adjust_gain
                and self.gain is not None
                and self.gain < self.gain_max
            ):

                step = int(
                    self.ctl.info[
                        "gain"
                    ].get(
                        "step",
                        1
                    )
                )

                delta = max(
                    step,
                    int(
                        round(
                            max(
                                8,
                                self.gain
                            )
                            * ratio
                        )
                    )
                )

                ok = self._write(
                    "gain",
                    self.gain + delta,
                    self.gain_min,
                    self.gain_max
                )

                return (
                    ok,
                    "gain_up"
                )

        else:

            if (
                adjust_gain
                and self.gain is not None
                and self.gain > self.gain_min
            ):

                step = int(
                    self.ctl.info[
                        "gain"
                    ].get(
                        "step",
                        1
                    )
                )

                delta = max(
                    step,
                    int(
                        round(
                            max(
                                8,
                                self.gain
                            )
                            * ratio
                        )
                    )
                )

                ok = self._write(
                    "gain",
                    self.gain - delta,
                    self.gain_min,
                    self.gain_max
                )

                return (
                    ok,
                    "gain_down"
                )

            if self.exp > self.exp_min:

                step = int(
                    self.ctl.info[
                        self.exp_name
                    ].get(
                        "step",
                        1
                    )
                )

                delta = max(
                    step,
                    int(
                        round(
                            max(
                                1,
                                self.exp
                            )
                            * ratio
                        )
                    )
                )

                ok = self._write(
                    self.exp_name,
                    self.exp - delta,
                    self.exp_min,
                    self.exp_max
                )

                return (
                    ok,
                    "exp_down"
                )

        return (
            False,
            "at_limit"
        )

    def on_frame(
        self,
        frame,
        frame_seq
    ):

        if not self.enabled:
            return

        if self.hold > 0:
            self.hold -= 1
            return

        period = max(
            1,
            int(
                self.cfg.get(
                    "adjust_every_frames",
                    10
                )
            )
        )

        if (
            frame_seq
            - self.last_eval_seq
            < period
        ):
            return

        self.last_eval_seq = frame_seq

        measured = self._measure(
            frame
        )

        if measured is None:
            return

        (
            score,
            p90,
            p98,
            highlight_ratio,
            shadow_ratio
        ) = measured

        alpha = float(
            self.cfg.get(
                "ema_alpha",
                0.20
            )
        )

        self.ema = (
            score
            if self.ema is None
            else (
                alpha * score
                +
                (1.0 - alpha)
                * self.ema
            )
        )

        target = float(
            self.cfg.get(
                "target_luma",
                115
            )
        )

        deadband = float(
            self.cfg.get(
                "deadband",
                8
            )
        )

        error = (
            target
            - self.ema
        )

        if (
            self.ema
            > target + deadband
            and p98 >= 248
            and highlight_ratio
            > float(
                self.cfg.get(
                    "highlight_ratio",
                    0.15
                )
            )
        ):

            error = min(
                error,
                -deadband - 1
            )

        elif (
            shadow_ratio
            > float(
                self.cfg.get(
                    "shadow_ratio",
                    0.35
                )
            )
            and p90
            < target - deadband
        ):

            error = max(
                error,
                deadband + 1
            )

        action = "deadband"

        if abs(error) <= deadband:

            self.outside_count = 0
            self.last_dir = 0

        else:

            direction = (
                1
                if error > 0
                else -1
            )

            if direction == self.last_dir:
                self.outside_count += 1
            else:
                self.outside_count = 1

            self.last_dir = direction
            action = "waiting"

            if (
                self.outside_count
                >= int(
                    self.cfg.get(
                        "outside_need",
                        2
                    )
                )
            ):

                ok, action = self._adjust(
                    direction,
                    error
                )

                self.outside_count = 0

                if ok:

                    self.hold = int(
                        self.cfg.get(
                            "settle_frames",
                            6
                        )
                    )

        now = time.monotonic()

        if (
            now - self.last_log
            >= 1.0
        ):

            print(
                "[AE][{}] v={:.1f}/{:.1f} target={:.0f} "
                "p90={:.0f} clip={:.1%}/{:.1%} "
                "exp={} gain={} {}".format(
                    self.name,
                    score,
                    self.ema,
                    target,
                    p90,
                    highlight_ratio,
                    shadow_ratio,
                    self.exp,
                    self.gain,
                    action
                )
            )

            self.last_log = now

    def reapply_after_reconnect(self):

        try:

            if not self.enabled:
                return False

            if not self.ctl.refresh():

                return self._reconnect_fail(
                    "controls missing after reconnect"
                )

            self.auto_name = next(
                (
                    name
                    for name in (
                        "exposure_auto",
                        "auto_exposure"
                    )
                    if self.ctl.has(name)
                ),
                None
            )

            self.priority_name = next(
                (
                    name
                    for name in (
                        "exposure_auto_priority",
                        "exposure_dynamic_framerate"
                    )
                    if self.ctl.has(name)
                ),
                None
            )

            manual_value = (
                self.ctl.menu_value(
                    self.auto_name,
                    ("manual",)
                )
                if self.auto_name
                else None
            )

            self.manual_value = manual_value

            if (
                manual_value is None
                or not self.ctl.set(
                    self.auto_name,
                    manual_value,
                    verify=True,
                    timeout=1.0
                )
            ):

                return self._reconnect_fail(
                    "manual mode failed after reconnect"
                )

            if int(
                self.cfg.get(
                    "set_50hz",
                    1
                )
            ) == 1:

                hz50 = self.ctl.menu_value(
                    "power_line_frequency",
                    ("50", "hz")
                )

                old_power_line = self.snapshot.get(
                    "power_line_frequency"
                )

                if (
                    hz50 is not None
                    and old_power_line is not None
                    and not self.ctl.set(
                        "power_line_frequency",
                        hz50,
                        verify=True,
                        timeout=1.0
                    )
                ):

                    print(
                        "[AE][{}] WARN: 50 Hz reapply failed".format(
                            self.name
                        )
                    )

            if (
                self.priority_name
                and self.snapshot.get(
                    self.priority_name
                ) is not None
            ):

                if not self.ctl.set(
                    self.priority_name,
                    0,
                    verify=True,
                    timeout=1.0
                ):

                    print(
                        "[AE][{}] WARN: {} reapply failed".format(
                            self.name,
                            self.priority_name
                        )
                    )

            if not self.ctl.set(
                self.exp_name,
                self.exp,
                verify=False,
                timeout=1.0
            ):

                return self._reconnect_fail(
                    "exposure restore failed"
                )

            if (
                int(
                    self.cfg.get(
                        "adjust_gain",
                        0
                    )
                ) == 1
                and self.gain is not None
                and self.ctl.has("gain")
                and not self.ctl.set(
                    "gain",
                    self.gain,
                    verify=False,
                    timeout=1.0
                )
            ):

                return self._reconnect_fail(
                    "gain restore failed"
                )

            self.ema = None
            self.hold = int(
                self.cfg.get(
                    "settle_frames",
                    6
                )
            )

            self.last_eval_seq = -10 ** 9
            self.outside_count = 0
            self.last_dir = 0

            return True

        except Exception as exc:

            return self._reconnect_fail(
                "reapply exception: {!r}".format(exc)
            )

    def _reconnect_fail(
        self,
        reason
    ):

        try:

            self.disable_runtime(
                reason
            )

        except Exception:

            self.enabled = False
            self.needs_restore = True
            self.restore_state = "failed"

        return False


class Camera:

    def __init__(
        self,
        index=0,
        width=640,
        height=480,
        auto_exposure=0,
        ae_cfg=None,
        camera_name="camera"
    ):

        # ========================================================
        # 原来的三个位置参数保持不变
        #
        # Camera(0)
        # Camera(2)
        #
        # 仍然可以正常使用。
        # ========================================================

        self.width = width
        self.height = height
        self.index = index
        self.camera_name = camera_name

        self._frame_lock = threading.Lock()
        self._cap_lock = threading.RLock()

        self.cap = None

        self.src = index

        self.frame = None
        self.frame_seq = 0
        self.capture_time = 0.0

        self.pause_flag = False
        self.stop_flag = False

        self.flag_thread = False
        self.cap_thread = None

        self._ae = None
        self._last_reopen_log = 0.0

        if not self.init():
            raise RuntimeError(
                "camera initialization stopped before device opened"
            )

        self._apply_capture_settings()

        # ========================================================
        # 自动曝光
        # ========================================================

        try:

            ae_switch = int(
                auto_exposure
            )

        except (
            TypeError,
            ValueError
        ):

            logger.error(
                "invalid auto_exposure={!r}; use 0".format(
                    auto_exposure
                )
            )

            ae_switch = 0

        if ae_switch not in (
            0,
            1
        ):

            logger.error(
                "auto_exposure must be 0/1; use 0"
            )

            ae_switch = 0

        if ae_switch == 1:

            candidate = None

            try:

                candidate = _SoftwareAE(
                    self.src,
                    ae_cfg or {},
                    camera_name
                )

                started = bool(
                    candidate.start()
                )

                if (
                    started
                    or candidate.needs_restore
                ):

                    self._ae = candidate

            except Exception as exc:

                logger.error(
                    "[AE][{}] start exception: {}".format(
                        camera_name,
                        exc
                    )
                )

                if candidate is not None:

                    try:

                        candidate.stop_with_fallback(
                            "constructor/start exception: {}".format(
                                exc
                            )
                        )

                    except Exception as restore_exc:

                        candidate.needs_restore = True
                        candidate.restore_state = "failed"

                        logger.error(
                            "[AE][{}] RESTORE FAILED: {}".format(
                                camera_name,
                                restore_exc
                            )
                        )

                    if candidate.needs_restore:
                        self._ae = candidate

        self.start_back_thread()

    # ============================================================
    # 打开摄像头
    #
    # 这里是本次修改的核心。
    #
    # Windows：
    #     继续使用原来的 index。
    #
    # Linux / Jetson：
    #
    #     index=0
    #         ↓
    #     USB 2.1 物理路径
    #
    #     index=2
    #         ↓
    #     USB 2.4 物理路径
    #
    # ============================================================

    def _open_capture_once(self):

        cap = None

        try:

            # ====================================================
            # Windows
            # ====================================================

            if platform.system() == "Windows":

                self.src = self.index

                cap = cv2.VideoCapture(
                    self.src,
                    cv2.CAP_DSHOW
                )

            # ====================================================
            # Linux / Jetson
            # ====================================================

            else:

                # ------------------------------------------------
                # 根据原来的摄像头编号查找物理 USB 路径
                # ------------------------------------------------

                if self.index not in PHYSICAL_CAMERA_PATHS:

                    logger.error(
                        "camera index {} has no physical path mapping; "
                        "valid mappings={}".format(
                            self.index,
                            list(
                                PHYSICAL_CAMERA_PATHS.keys()
                            )
                        )
                    )

                    return None

                self.src = PHYSICAL_CAMERA_PATHS[
                    self.index
                ]

                # ------------------------------------------------
                # 检查物理路径
                # ------------------------------------------------

                if not os.path.exists(
                    self.src
                ):

                    logger.error(
                        "camera physical path not found: {}".format(
                            self.src
                        )
                    )

                    return None

                # ------------------------------------------------
                # 直接通过物理 USB 路径打开
                # ------------------------------------------------

                cap = cv2.VideoCapture(
                    self.src
                )

            # ====================================================
            # 检查摄像头是否打开成功
            # ====================================================

            if (
                cap is None
                or not cap.isOpened()
            ):

                try:

                    if cap is not None:
                        cap.release()

                except Exception:
                    pass

                logger.error(
                    "camera {} open failed".format(
                        self.src
                    )
                )

                return None

            logger.info(
                "camera opened successfully: "
                "index={} path={}".format(
                    self.index,
                    self.src
                )
            )

            return cap

        except Exception as exc:

            logger.error(
                "camera {} open exception: {}".format(
                    self.src,
                    exc
                )
            )

            try:

                if cap is not None:
                    cap.release()

            except Exception:
                pass

            return None

    def init(self):

        """Open the camera, retaining the old wait-until-present behaviour."""

        last_open_log = 0.0

        while (
            self.cap is None
            and not self.stop_flag
        ):

            cap = self._open_capture_once()

            if cap is not None:

                with self._cap_lock:

                    self.cap = cap

                return True

            now = time.monotonic()

            if (
                now - last_open_log
                >= 2.0
            ):

                logger.error(
                    "camera {} not ready; retrying".format(
                        self.index
                    )
                )

                last_open_log = now

            time.sleep(0.5)

        return self.cap is not None

    def _apply_capture_settings(self):

        with self._cap_lock:

            if self.cap is None:
                return False

            results = (

                self.cap.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(
                        *"MJPG"
                    )
                ),

                self.cap.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1
                ),

                self.cap.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    self.width
                ),

                self.cap.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    self.height
                ),
            )

            try:

                actual_width = self.cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )

                actual_height = self.cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )

                actual_fourcc = int(
                    self.cap.get(
                        cv2.CAP_PROP_FOURCC
                    )
                )

                backend = (
                    self.cap.getBackendName()
                    if hasattr(
                        self.cap,
                        "getBackendName"
                    )
                    else "unknown"
                )

            except Exception:

                actual_width = -1
                actual_height = -1
                actual_fourcc = -1
                backend = "unknown"

        if not all(results):

            logger.warning(
                "camera {}: one or more capture settings rejected".format(
                    self.src
                )
            )

        logger.info(
            "camera {} opened: {}x{} fourcc={} backend={}".format(
                self.src,
                int(actual_width),
                int(actual_height),
                actual_fourcc,
                backend
            )
        )

        return True

    def start_back_thread(self):

        if not self.flag_thread:

            self.cap_thread = threading.Thread(
                target=self.update,
                args=()
            )

            self.cap_thread.daemon = True

            self.flag_thread = True

            self.cap_thread.start()

        time.sleep(0.5)

    def update(self):

        try:

            while not self.stop_flag:

                if self.pause_flag:

                    time.sleep(0.01)
                    continue

                # =================================================
                # 摄像头读取
                # =================================================

                try:

                    with self._cap_lock:

                        cap = self.cap

                    if cap is None:

                        ret, frame = (
                            False,
                            None
                        )

                    else:

                        ret, frame = cap.read()

                except Exception as exc:

                    logger.error(
                        "camera read exception {}: {}".format(
                            self.src,
                            exc
                        )
                    )

                    self._reopen_camera(
                        "read exception"
                    )

                    continue

                if self.stop_flag:
                    break

                if (
                    not ret
                    or frame is None
                ):

                    self._reopen_camera(
                        "read returned false"
                    )

                    continue

                capture_time = time.monotonic()

                with self._frame_lock:

                    self.frame = frame

                    self.frame_seq += 1

                    frame_seq = self.frame_seq

                    self.capture_time = capture_time

                # =================================================
                # 自动曝光
                # =================================================

                ae = self._ae

                if (
                    ae is not None
                    and ae.enabled
                ):

                    try:

                        ae.on_frame(
                            frame,
                            frame_seq
                        )

                    except Exception as exc:

                        logger.error(
                            "[AE][{}] frame exception: {}".format(
                                ae.name,
                                exc
                            )
                        )

                        try:

                            ae.disable_runtime(
                                "on_frame exception: {}".format(
                                    exc
                                )
                            )

                        except Exception as disable_exc:

                            ae.enabled = False
                            ae.needs_restore = True
                            ae.restore_state = "failed"

                            logger.error(
                                "[AE][{}] disable exception: {}".format(
                                    ae.name,
                                    disable_exc
                                )
                            )

        finally:

            self.flag_thread = False

    def _reopen_camera(
        self,
        reason
    ):

        if self.stop_flag:
            return False

        # ========================================================
        # 不返回断开前的旧画面
        # ========================================================

        with self._frame_lock:

            self.frame = None
            self.capture_time = 0.0

        opened = False

        old_cap = None
        new_cap = None

        try:

            # ====================================================
            # 分离旧摄像头
            # ====================================================

            with self._cap_lock:

                old_cap = self.cap
                self.cap = None

            try:

                if old_cap is not None:
                    old_cap.release()

            except Exception:
                pass

            if self.stop_flag:
                return False

            # ====================================================
            # 重新通过物理路径打开
            # ====================================================

            new_cap = self._open_capture_once()

            if new_cap is not None:

                with self._cap_lock:

                    if self.stop_flag:

                        new_cap.release()

                        return False

                    self.cap = new_cap

                opened = bool(
                    self._apply_capture_settings()
                )

        except Exception as exc:

            logger.error(
                "camera reopen {} failed: {}".format(
                    self.src,
                    exc
                )
            )

            opened = False

        # ========================================================
        # 打开失败
        # ========================================================

        if not opened:

            failed_cap = None

            with self._cap_lock:

                failed_cap = self.cap
                self.cap = None

            try:

                if failed_cap is not None:
                    failed_cap.release()

            except Exception:
                pass

            now = time.monotonic()

            if (
                now - self._last_reopen_log
                >= 2.0
            ):

                logger.error(
                    "camera {} offline; retrying".format(
                        self.src
                    )
                )

                self._last_reopen_log = now

            for _ in range(10):

                if self.stop_flag:
                    return False

                time.sleep(0.05)

            return False

        # ========================================================
        # 摄像头重新连接后重新配置 AE
        # ========================================================

        ae = self._ae

        if ae is not None:

            try:

                if ae.enabled:

                    if not ae.reapply_after_reconnect():

                        logger.error(
                            "[AE][{}] disabled after reconnect".format(
                                ae.name
                            )
                        )

                elif ae.needs_restore:

                    logger.warning(
                        "[AE][{}] remains disabled; "
                        "restore deferred to close".format(
                            ae.name
                        )
                    )

            except Exception as exc:

                logger.error(
                    "[AE] reconnect handling failed: {}".format(
                        exc
                    )
                )

                try:

                    ae.stop_with_fallback(
                        "reconnect exception: {}".format(
                            exc
                        )
                    )

                except Exception:

                    ae.enabled = False
                    ae.needs_restore = True
                    ae.restore_state = "failed"

        logger.warning(
            "camera {} reopened: {}".format(
                self.src,
                reason
            )
        )

        return True

    def set_size(
        self,
        width,
        height
    ):

        self.width = width
        self.height = height

        with self._cap_lock:

            if self.cap is None:
                return False

            result_width = self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.width
            )

            result_height = self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.height
            )

        return bool(
            result_width
            and result_height
        )

    def read(
        self,
        timeout=3.0
    ):

        frame, _, _ = self.read_with_meta(
            timeout
        )

        return frame

    def read_with_meta(
        self,
        timeout=3.0
    ):

        """Return one atomic snapshot of frame, sequence and capture time."""

        try:

            timeout = max(
                0.0,
                float(timeout)
            )

        except (
            TypeError,
            ValueError
        ):

            timeout = 3.0

        # timeout=0 仍应返回已经发布的最新帧，
        # 保持非阻塞读取语义。

        with self._frame_lock:

            if self.frame is not None:

                return (
                    self.frame,
                    self.frame_seq,
                    self.capture_time
                )

        deadline = (
            time.monotonic()
            + timeout
        )

        while (
            time.monotonic()
            < deadline
        ):

            with self._frame_lock:

                if self.frame is not None:

                    return (
                        self.frame,
                        self.frame_seq,
                        self.capture_time
                    )

            time.sleep(0.005)

        return (
            None,
            0,
            0.0
        )

    def close(self):

        self.stop_flag = True
        self.pause_flag = False

        thread = getattr(
            self,
            "cap_thread",
            None
        )

        if (
            thread is not None
            and thread is not threading.current_thread()
        ):

            try:

                thread.join(
                    timeout=3.0
                )

            except Exception as exc:

                logger.error(
                    "camera thread join failed: {}".format(
                        exc
                    )
                )

        thread_alive = bool(
            thread is not None
            and thread.is_alive()
        )

        cap_released = False

        if thread_alive:

            acquired = False

            try:

                acquired = (
                    self._cap_lock.acquire(
                        timeout=1.0
                    )
                )

                if acquired:

                    cap = self.cap
                    self.cap = None

                    if cap is not None:
                        cap.release()

                    cap_released = True

            except Exception as exc:

                logger.error(
                    "camera forced release failed: {}".format(
                        exc
                    )
                )

            finally:

                if acquired:
                    self._cap_lock.release()

            try:

                thread.join(
                    timeout=1.0
                )

            except Exception:
                pass

            thread_alive = thread.is_alive()

        # ========================================================
        # 恢复自动曝光
        # ========================================================

        ae = getattr(
            self,
            "_ae",
            None
        )

        if (
            not thread_alive
            and ae is not None
        ):

            try:

                if not ae.restore_before_release():

                    logger.error(
                        "[AE][{}] RESTORE FAILED state={}".format(
                            ae.name,
                            ae.restore_state
                        )
                    )

            except Exception as exc:

                logger.error(
                    "[AE] close restore exception: {}".format(
                        exc
                    )
                )

        elif (
            thread_alive
            and ae is not None
            and ae.needs_restore
        ):

            logger.error(
                "[AE][{}] restore skipped: "
                "capture thread still alive".format(
                    ae.name
                )
            )

        # ========================================================
        # 最终释放摄像头
        # ========================================================

        if not cap_released:

            acquired = False

            try:

                acquired = (
                    self._cap_lock.acquire(
                        timeout=1.0
                    )
                )

                if acquired:

                    cap = self.cap
                    self.cap = None

                    if cap is not None:
                        cap.release()

            except Exception as exc:

                logger.error(
                    "camera release failed: {}".format(
                        exc
                    )
                )

            finally:

                if acquired:
                    self._cap_lock.release()

        if thread_alive:

            logger.error(
                "camera thread did not stop; OS cleanup required"
            )

        logger.info(
            "{} close".format(
                self.src
            )
        )


def main():

    # ============================================================
    # 你的原始摄像头编号完全保持不变
    #
    # 0 -> USB 2.1 物理摄像头
    # 2 -> USB 2.4 物理摄像头
    # ============================================================

    camera = Camera(
        2,
        640,
        480
    )

    while True:

        try:

            img = camera.read()

            if img is None:
                continue

            cv2.imshow(
                "img",
                img
            )

            key = cv2.waitKey(1)

            if key == ord("q"):

                time.sleep(0.1)

                break

        except Exception as e:

            logger.error(e)

    camera.close()

    logger.info(
        "over"
    )

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()