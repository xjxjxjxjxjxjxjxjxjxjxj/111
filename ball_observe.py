#!/usr/bin/env python3
"""Camera-only ball preview.  This file never creates an XGO motor object."""

import argparse
import time
from pathlib import Path

import cv2

from ball_vision import BallVision, load_ball_config


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Preview ball recognition only")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--color", choices=("all", "blue", "green", "red"), default="all"
    )
    args = parser.parse_args()

    config = load_ball_config(ROOT / "ball_config.json")
    detector = BallVision(config)
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["camera"]["width"]))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["camera"]["height"]))
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError("摄像头打开失败")

    colors = ("blue", "green", "red") if args.color == "all" else (args.color,)
    draw_colors = {"blue": (255, 80, 20), "green": (30, 255, 30), "red": (30, 30, 255)}
    last_print = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue
            frame = detector.prepare_frame(frame)
            results = {name: detector.detect(frame, name) for name in colors}
            now = time.monotonic()
            if now - last_print >= 0.5:
                summary = []
                for name, item in results.items():
                    if item is None:
                        summary.append(name + "=none")
                    else:
                        summary.append(
                            "%s=(%.1f,%.1f,r%.1f,c%.2f)"
                            % (name, item.x, item.y, item.radius, item.confidence)
                        )
                print(" ".join(summary))
                last_print = now

            for name, item in results.items():
                if item is None:
                    continue
                color = draw_colors[name]
                center = (int(round(item.x)), int(round(item.y)))
                cv2.circle(frame, center, int(round(item.radius)), color, 2)
                cv2.putText(
                    frame,
                    "%s r=%.1f" % (name, item.radius),
                    (max(0, center[0] - 35), max(15, center[1] - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow("XGO ball observe (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
