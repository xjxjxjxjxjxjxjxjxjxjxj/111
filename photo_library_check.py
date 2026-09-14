#!/usr/bin/env python3
"""Run the complete recognition-only photo library as an offline regression."""

import json
from pathlib import Path

import cv2

from ball_vision import BallVision, load_ball_config
from sign_vision import SignVision, load_sign_config


ROOT = Path(__file__).resolve().parent


def main() -> int:
    with (ROOT / "recognition_library.json").open("r", encoding="utf-8") as stream:
        library = json.load(stream)
    ball_vision = BallVision(load_ball_config(ROOT / "ball_config.json"))
    sign_vision = SignVision(load_sign_config(ROOT / "sign_config.json"))
    failures = []

    for sample in library["samples"]:
        path = ROOT / "test_images" / sample["file"]
        frame = cv2.imread(str(path))
        if frame is None:
            failures.append("missing image: %s" % sample["file"])
            continue
        actual_balls = sorted(
            name
            for name, result in ball_vision.detect_all(frame).items()
            if result is not None
        )
        expected_balls = sorted(sample["expected_valid_balls"])
        yellow = sign_vision.detect_yellow(frame) is not None
        ok = actual_balls == expected_balls and yellow == sample["expected_yellow_evidence"]
        print(
            ("PASS" if ok else "FAIL"),
            sample["file"],
            "balls=", actual_balls,
            "yellow_evidence=", yellow,
        )
        if not ok:
            failures.append(sample["file"])

    print("识别样本总数:", len(library["samples"]), "失败:", len(failures))
    if failures:
        print("失败项:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
