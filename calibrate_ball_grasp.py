#!/usr/bin/env python3
"""Set the ball grasp window from actual successful-grasp camera photos."""

import argparse
import json
import statistics
from pathlib import Path

import cv2

from ball_vision import BallVision, load_ball_config
from camera_geometry import PinholeRangeModel, load_geometry_config
from image_io import read_image


ROOT = Path(__file__).resolve().parent


def image_paths(folder: Path):
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        yield from folder.glob(suffix)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the closed-loop target from successful grasp poses"
    )
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--color", choices=("blue", "green", "red"), required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "ball_config.json")
    parser.add_argument(
        "--geometry", type=Path, default=ROOT / "camera_geometry.json"
    )
    args = parser.parse_args()

    config = load_ball_config(args.config)
    vision = BallVision(config)
    range_model = PinholeRangeModel.from_config(
        load_geometry_config(args.geometry)
    )
    range_model.require_calibrated()

    centers = []
    distances = []
    paths = sorted(set(image_paths(args.image_dir)))
    if not paths:
        raise RuntimeError("抓取标定目录中没有 JPG/JPEG/PNG 图片")
    for path in paths:
        frame = read_image(path)
        if frame is None:
            print("SKIP unreadable:", path.name)
            continue
        detection = vision.detect(frame, args.color)
        if detection is None:
            print("MISS:", path.name)
            continue
        distance_cm = range_model.ball_distance_cm(2.0 * detection.radius)
        centers.append(detection.x)
        distances.append(distance_cm)
        print(
            "OK:", path.name,
            "x=%.2f" % detection.x,
            "distance=%.2fcm" % distance_cm,
        )

    minimum_samples = 2
    if len(centers) < minimum_samples:
        raise RuntimeError(
            "有效成功抓取样本不足：至少需要 %d 张，当前 %d 张"
            % (minimum_samples, len(centers))
        )
    target_x = statistics.median(centers)
    target_distance = statistics.median(distances)

    print(
        "GRASP TARGET x=%.2fpx distance=%.2fcm samples=%d"
        % (target_x, target_distance, len(centers))
    )
    if args.write:
        alignment = config["alignment"]
        alignment["calibrated_for_grasp"] = True
        alignment["parameter_source"] = (
            "each color uses only its own successful-grasp photos"
        )
        profiles = alignment.setdefault("profiles", {})
        profiles[args.color] = {
            "target_x_px": round(target_x, 4),
            "target_distance_cm": round(target_distance, 4),
            "successful_x_range_px": [round(min(centers), 4), round(max(centers), 4)],
            "successful_distance_range_cm": [
                round(min(distances), 4), round(max(distances), 4)
            ],
            "x_tolerance_px": round(max(4.0, (max(centers) - min(centers)) / 2 + 1.0), 4),
            "distance_tolerance_cm": round(
                max(0.8, (max(distances) - min(distances)) / 2 + 0.2), 4
            ),
            "sample_count": len(centers),
        }
        with args.config.open("w", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print("WROTE:", args.config)
    else:
        print("未写入配置；确认结果后增加 --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
