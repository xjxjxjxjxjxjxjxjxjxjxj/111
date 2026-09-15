#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely collect first-round exposure evidence from the two SP2812 cameras.

Stop the car program before running this script.  The script snapshots the
actual V4L2 controls, enters manual exposure, captures metrics/images for each
candidate, and verifies that the original controls were restored on every
handled exit path (including Ctrl+C, SIGTERM, and SIGHUP).

SIGKILL, power loss, or a kernel/USB driver call that never returns cannot be
recovered by any Python finally block.  Always read the final control snapshot.
"""

import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import cv2
import numpy as np


AUTO_NAME = "auto_exposure"
EXPOSURE_NAME = "exposure_time_absolute"
DYNAMIC_FPS_NAME = "exposure_dynamic_framerate"
POWER_LINE_NAME = "power_line_frequency"
CONTROL_NAMES = (
    AUTO_NAME,
    EXPOSURE_NAME,
    DYNAMIC_FPS_NAME,
    POWER_LINE_NAME,
)

# These are intentionally conservative first-round TEST limits, not final
# competition settings.  Do not silently raise them to the driver max (5000).
DEVICE_PROFILES = {
    "/dev/video0": {
        "role": "front_task",
        "roi": [0.08, 0.08, 0.92, 0.92],
        "target_luma": 120.0,
        "default_values": "80,120,157,200,250,300",
        "safe_test_max": 300,
    },
    "/dev/video2": {
        "role": "side_lane",
        "roi": [0.05, 0.55, 0.95, 0.90],
        "target_luma": 105.0,
        "default_values": "60,80,100,120,157,180,200",
        "safe_test_max": 200,
    },
}


class TerminationRequested(Exception):
    def __init__(self, signum):
        Exception.__init__(self, "signal {}".format(signum))
        self.signum = int(signum)


def run_v4l2(executable, device, argument, timeout=2.0):
    process = subprocess.run(
        [executable, "-d", device, argument],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=timeout,
        check=False,
    )
    message = ((process.stdout or "") + (process.stderr or "")).strip()
    if process.returncode != 0:
        raise RuntimeError("{} failed: {}".format(argument, message))
    return message


def get_control(executable, device, name):
    output = run_v4l2(
        executable, device, "--get-ctrl={}".format(name), timeout=1.0
    )
    # match = re.search(r":\s*(-?\d+)\s*$", output)
    match = re.search(r":\s*(-?\d+)\b", output)
    if not match:
        raise RuntimeError("cannot parse {} from {!r}".format(name, output))
    return int(match.group(1))


def set_control(executable, device, name, value, verify=True):
    value = int(value)
    run_v4l2(
        executable,
        device,
        "--set-ctrl={}={}".format(name, value),
        timeout=1.0,
    )
    if verify:
        actual = get_control(executable, device, name)
        if actual != value:
            raise RuntimeError(
                "{} requested {}, read back {}".format(name, value, actual)
            )


def snapshot_controls(executable, device):
    return {
        name: get_control(executable, device, name)
        for name in CONTROL_NAMES
    }


def parse_values(raw_values, profile):
    values = []
    for item in raw_values.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if not 1 <= value <= int(profile["safe_test_max"]):
            raise ValueError(
                "{} is outside the first-round safe test range 1..{} for this "
                "camera; do not use the driver max 5000".format(
                    value, profile["safe_test_max"]
                )
            )
        values.append(value)
    if not values:
        raise ValueError("at least one exposure value is required")
    return values


def measure_frames(capture, count):
    frames = []
    timestamps = []
    deadline = time.monotonic() + max(4.0, count * 0.25)
    while len(frames) < count and time.monotonic() < deadline:
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)
            timestamps.append(time.monotonic())
    if len(frames) < max(3, count // 2):
        raise RuntimeError(
            "too few frames: got {} of {}; camera may be occupied".format(
                len(frames), count
            )
        )
    return frames, timestamps


def crop_normalized(frame, roi):
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = [float(value) for value in roi]
    px0 = max(0, min(width, int(width * x0)))
    px1 = max(0, min(width, int(width * x1)))
    py0 = max(0, min(height, int(height * y0)))
    py1 = max(0, min(height, int(height * y1)))
    cropped = frame[py0:py1, px0:px1]
    if cropped.size == 0:
        raise RuntimeError("configured ROI is empty")
    return cropped


def frame_metrics(frames, timestamps, profile):
    scores = []
    medians = []
    p90_values = []
    p98_values = []
    clip_ratios = []
    dark_ratios = []
    sharpness_values = []
    full_gray_medians = []

    for frame in frames:
        cropped = crop_normalized(frame, profile["roi"])
        value_channel = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)[:, :, 2]
        scale = min(1.0, 160.0 / max(1, value_channel.shape[1]))
        if scale < 1.0:
            size = (
                max(8, int(value_channel.shape[1] * scale)),
                max(8, int(value_channel.shape[0] * scale)),
            )
            value_channel = cv2.resize(
                value_channel, size, interpolation=cv2.INTER_AREA
            )

        pixels = value_channel.reshape(-1).astype(np.float32)
        p10, p50, p90, p98 = np.percentile(pixels, [10, 50, 90, 98])
        trimmed = pixels[(pixels >= p10) & (pixels <= p90)]
        trimmed_mean = float(trimmed.mean()) if trimmed.size else float(p50)
        scores.append(0.65 * float(p50) + 0.35 * trimmed_mean)
        medians.append(float(p50))
        p90_values.append(float(p90))
        p98_values.append(float(p98))
        clip_ratios.append(float(np.mean(pixels >= 250)))
        dark_ratios.append(float(np.mean(pixels <= 8)))

        gray_roi = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.resize(
            gray_roi,
            (value_channel.shape[1], value_channel.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        sharpness_values.append(
            float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
        )
        full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full_gray_medians.append(float(np.median(full_gray)))

    fps = 0.0
    if len(timestamps) >= 2 and timestamps[-1] > timestamps[0]:
        fps = float(len(timestamps) - 1) / (timestamps[-1] - timestamps[0])

    score = float(np.median(scores))
    target = float(profile["target_luma"])
    return {
        "target": target,
        "ae_score": score,
        # Same sign as the production controller: positive means too dark and
        # asks for more exposure; negative means too bright.
        "control_error": target - score,
        "v_median": float(np.median(medians)),
        "p90": float(np.median(p90_values)),
        "p98": float(np.median(p98_values)),
        "clip_ratio": float(np.median(clip_ratios)),
        "dark_ratio": float(np.median(dark_ratios)),
        "sharpness": float(np.median(sharpness_values)),
        "full_gray_median": float(np.median(full_gray_medians)),
        "fps": fps,
    }


def write_controls(path, controls):
    with open(path, "w", newline="") as output:
        for name in CONTROL_NAMES:
            output.write("{}={}\n".format(name, controls.get(name, "ERROR")))


def restore_controls(executable, device, snapshot):
    last_failures = []
    for attempt in range(1, 3):
        failures = []
        try:
            # Exposure is inactive in auto mode.  Always enter manual first,
            # restore the saved values, and restore the original mode last.
            set_control(executable, device, AUTO_NAME, 1, verify=True)
        except Exception as exc:
            failures.append("{}: {}".format(AUTO_NAME, exc))

        for name in (EXPOSURE_NAME, DYNAMIC_FPS_NAME, POWER_LINE_NAME):
            try:
                set_control(
                    executable, device, name, snapshot[name], verify=True
                )
            except Exception as exc:
                failures.append("{}: {}".format(name, exc))

        try:
            set_control(
                executable,
                device,
                AUTO_NAME,
                snapshot[AUTO_NAME],
                verify=True,
            )
        except Exception as exc:
            failures.append("{} final: {}".format(AUTO_NAME, exc))

        if not failures:
            return []
        last_failures = failures
        if attempt < 2:
            time.sleep(0.2)
    return last_failures


def compare_final_controls(snapshot, after):
    """Compare controls whose final value is expected to remain stable.

    In Aperture Priority mode the driver is allowed to change the absolute
    exposure immediately, so exposure is compared only when the original mode
    was Manual.  Exposure was already read back while manual inside
    restore_controls().
    """
    names = [AUTO_NAME, DYNAMIC_FPS_NAME, POWER_LINE_NAME]
    if snapshot.get(AUTO_NAME) == 1:
        names.append(EXPOSURE_NAME)
    failures = []
    for name in names:
        if after.get(name) != snapshot.get(name):
            failures.append(
                "final {} mismatch: before={} after={}".format(
                    name, snapshot.get(name), after.get(name)
                )
            )
    return failures


def print_manual_restore(device, snapshot):
    print("MANUAL RESTORE REQUIRED. Run these commands:", file=sys.stderr)
    print(
        "v4l2-ctl -d {} --set-ctrl={}={}".format(device, AUTO_NAME, 1),
        file=sys.stderr,
    )
    print(
        "v4l2-ctl -d {} --set-ctrl={}={},{}={},{}={}".format(
            device,
            EXPOSURE_NAME,
            snapshot[EXPOSURE_NAME],
            DYNAMIC_FPS_NAME,
            snapshot[DYNAMIC_FPS_NAME],
            POWER_LINE_NAME,
            snapshot[POWER_LINE_NAME],
        ),
        file=sys.stderr,
    )
    print(
        "v4l2-ctl -d {} --set-ctrl={}={}".format(
            device, AUTO_NAME, snapshot[AUTO_NAME]
        ),
        file=sys.stderr,
    )


def fourcc_text(raw_value):
    raw_value = int(raw_value)
    return "".join(chr((raw_value >> (8 * index)) & 0xFF) for index in range(4))


def write_csv(path, rows):
    fieldnames = (
        "requested",
        "actual",
        "target",
        "ae_score",
        "control_error",
        "v_median",
        "p90",
        "p98",
        "clip_ratio",
        "dark_ratio",
        "sharpness",
        "full_gray_median",
        "fps",
        "width",
        "height",
        "fourcc",
        "driver_fps",
    )
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", required=True, choices=tuple(sorted(DEVICE_PROFILES))
    )
    parser.add_argument(
        "--values",
        default=None,
        help="comma-separated exposure_time_absolute candidates",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    executable = shutil.which("v4l2-ctl")
    if not executable:
        print("ERROR: v4l2-ctl not found", file=sys.stderr)
        return 2

    profile = DEVICE_PROFILES[args.device]
    raw_values = args.values or profile["default_values"]
    values = parse_values(raw_values, profile)
    if args.frames < 5:
        raise ValueError("--frames must be >= 5")

    camera_tag = os.path.basename(args.device)
    output_root = args.output or os.path.join("ae_test", camera_tag)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, run_tag)
    os.makedirs(output_dir, exist_ok=False)

    snapshot = snapshot_controls(executable, args.device)
    write_controls(os.path.join(output_dir, "controls_before.txt"), snapshot)
    print("original controls: {}".format(snapshot))
    print(
        "profile: role={} target={} ROI={} safe_test_max={}".format(
            profile["role"],
            profile["target_luma"],
            profile["roi"],
            profile["safe_test_max"],
        )
    )

    capture = cv2.VideoCapture(args.device)
    if capture is None or not capture.isOpened():
        print(
            "ERROR: cannot open {}; stop the car program first".format(args.device),
            file=sys.stderr,
        )
        return 3

    setting_results = {}
    setting_results["fourcc"] = capture.set(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
    )
    buffer_property = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
    if buffer_property is not None:
        setting_results["buffer"] = capture.set(buffer_property, 1)
    setting_results["width"] = capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    setting_results["height"] = capture.set(
        cv2.CAP_PROP_FRAME_HEIGHT, args.height
    )
    read_timeout_property = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if read_timeout_property is not None:
        setting_results["read_timeout"] = capture.set(
            read_timeout_property, 1000
        )

    format_info = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fourcc": fourcc_text(capture.get(cv2.CAP_PROP_FOURCC)),
        "driver_fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    print("capture set results: {}".format(setting_results))
    print("capture readback: {}".format(format_info))

    cleanup_state = {"active": False, "pending": []}

    def guarded_signal_handler(signum, _frame):
        if cleanup_state["active"]:
            cleanup_state["pending"].append(int(signum))
            print(
                "signal {} noted; cleanup will not be interrupted".format(signum),
                file=sys.stderr,
            )
            return
        raise TerminationRequested(signum)

    previous_handlers = {}
    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is not None:
            previous_handlers[signal_value] = signal.getsignal(signal_value)
            signal.signal(signal_value, guarded_signal_handler)

    rows = []
    exit_code = 0
    restore_failures = []
    controls_touched = False
    try:
        # Detect common camera-occupied/disconnected/format failures before
        # changing any V4L2 exposure control.  This is not a kernel-level hard
        # watchdog, but it closes the common failure path safely.
        print("preflight: reading 5 frames before changing exposure controls")
        measure_frames(capture, 5)
        controls_touched = True
        set_control(executable, args.device, AUTO_NAME, 1, verify=True)
        set_control(
            executable, args.device, DYNAMIC_FPS_NAME, 0, verify=True
        )
        set_control(executable, args.device, POWER_LINE_NAME, 1, verify=True)

        for exposure in values:
            set_control(
                executable,
                args.device,
                EXPOSURE_NAME,
                exposure,
                verify=True,
            )
            actual_exposure = get_control(
                executable, args.device, EXPOSURE_NAME
            )
            time.sleep(0.35)

            # Discard buffered transition frames before measuring.
            measure_frames(capture, 8)
            frames, timestamps = measure_frames(
                capture, max(5, args.frames)
            )
            metrics = frame_metrics(frames, timestamps, profile)

            image_paths = []
            for image_index, frame in enumerate(frames[-3:], start=1):
                image_path = os.path.join(
                    output_dir,
                    "exp_{:04d}_actual_{:04d}_{}.png".format(
                        exposure, actual_exposure, image_index
                    ),
                )
                if not cv2.imwrite(image_path, frame):
                    raise RuntimeError("failed to save {}".format(image_path))
                image_paths.append(image_path)

            row = {"requested": exposure, "actual": actual_exposure}
            row.update(metrics)
            row.update(format_info)
            rows.append(row)
            print(
                "exp={requested} actual={actual} score={ae_score:.1f} "
                "target={target:.0f} control_error={control_error:+.1f} "
                "p98={p98:.1f} clip={clip_ratio:.1%} "
                "dark={dark_ratio:.1%} sharp={sharpness:.1f} "
                "fps={fps:.1f} -> {path}".format(
                    path=image_paths[0], **row
                )
            )
    except TerminationRequested as exc:
        print("received signal {}; restoring controls".format(exc.signum))
        exit_code = 128 + exc.signum
    except KeyboardInterrupt:
        # Defensive fallback if an environment replaces the SIGINT handler.
        print("interrupted by Ctrl+C; restoring controls")
        exit_code = 130
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        exit_code = 4
    finally:
        cleanup_state["active"] = True
        try:
            if controls_touched:
                restore_failures = restore_controls(
                    executable, args.device, snapshot
                )
        finally:
            try:
                capture.release()
            except Exception as exc:
                print("RELEASE WARN: {}".format(exc), file=sys.stderr)

        try:
            after = snapshot_controls(executable, args.device)
            write_controls(
                os.path.join(output_dir, "controls_after.txt"), after
            )
            print("final controls: {}".format(after))
            restore_failures.extend(
                compare_final_controls(snapshot, after)
            )
            if (
                snapshot.get(AUTO_NAME) != 1
                and after.get(EXPOSURE_NAME) != snapshot.get(EXPOSURE_NAME)
            ):
                print(
                    "NOTE: absolute exposure changed after automatic mode was "
                    "restored; this is expected and is not a restore failure"
                )
        except Exception as exc:
            restore_failures.append("final snapshot: {}".format(exc))

        for signal_value, previous_handler in previous_handlers.items():
            try:
                signal.signal(signal_value, previous_handler)
            except Exception:
                pass

        if restore_failures:
            print(
                "RESTORE ERROR: {}".format("; ".join(restore_failures)),
                file=sys.stderr,
            )
            print_manual_restore(args.device, snapshot)
            if exit_code == 0:
                exit_code = 5
        else:
            print("original control mode/settings restored and verified")

        if cleanup_state["pending"]:
            print(
                "signals received during cleanup: {}".format(
                    cleanup_state["pending"]
                ),
                file=sys.stderr,
            )
            if exit_code == 0:
                exit_code = 128 + int(cleanup_state["pending"][0])

    if rows:
        csv_path = os.path.join(output_dir, "summary.csv")
        write_csv(csv_path, rows)
        latest_csv = os.path.join(output_root, "latest_summary.csv")
        shutil.copyfile(csv_path, latest_csv)
        print("summary: {}".format(csv_path))
        print("latest summary: {}".format(latest_csv))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
