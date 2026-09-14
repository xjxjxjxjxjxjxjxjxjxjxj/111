#!/usr/bin/env python3
"""Fit and verify the sign range model from measured XGO-camera photos.

Example (PowerShell)::

    python calibrate_sign_distance.py `
      --sample calibration_images/sign_yellow_20cm 20 yellow `
      --sample calibration_images/sign_yellow_30cm 30 yellow `
      --validate calibration_images/sign_black_20cm 20 black `
      --validate calibration_images/sign_black_30cm 30 black --write

Unknown-distance recognition-library pictures are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from camera_geometry import fit_range_model, load_geometry_config
from image_io import read_image
from sign_line_closed_loop import VisionProcessor, load_config


ROOT = Path(__file__).resolve().parent


def image_paths(folder: Path) -> Iterable[Path]:
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        yield from folder.glob(suffix)


def measure_group(
    vision: VisionProcessor, folder: Path, expected_color: str
) -> List[float]:
    """Return apparent plate diameters for one labelled calibration group."""
    paths = sorted(set(image_paths(folder)))
    if not paths:
        raise RuntimeError("标定目录中没有 JPG/JPEG/PNG 图片: %s" % folder)
    diameters: List[float] = []
    for path in paths:
        frame = read_image(path)
        detection = None if frame is None else vision.analyze(frame).sign
        if detection is None:
            print("MISS:", path.name)
            continue
        if detection.label != expected_color:
            print(
                "WRONG COLOR:", path.name,
                "expected=", expected_color,
                "actual=", detection.label,
            )
            continue
        diameter = 2.0 * detection.radius
        diameters.append(diameter)
        print("OK:", path.name, "diameter_px=%.2f" % diameter)
    if len(diameters) < 3:
        raise RuntimeError(
            "每个距离/颜色组至少需要3张有效照片，当前只有%d张: %s"
            % (len(diameters), folder)
        )
    return diameters


def parse_groups(raw_groups: List[List[str]], option: str):
    parsed = []
    for folder_text, distance_text, color in raw_groups:
        if color not in ("yellow", "black"):
            raise ValueError("%s color must be yellow or black" % option)
        folder = Path(folder_text).resolve()
        distance = float(distance_text)
        if distance <= 0:
            raise ValueError("%s distance must be positive" % option)
        parsed.append((folder, distance, color))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit physical distance from 50 mm sign photos at 2+ distances"
    )
    parser.add_argument(
        "--sample", nargs=3, action="append", required=True,
        metavar=("DIR", "DISTANCE_CM", "COLOR"),
        help="repeat for each fitting group; COLOR is yellow or black",
    )
    parser.add_argument(
        "--validate", nargs=3, action="append", default=[],
        metavar=("DIR", "DISTANCE_CM", "COLOR"),
        help="optional groups checked but not used to fit the model",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "line_sign_config.json"
    )
    parser.add_argument(
        "--geometry", type=Path, default=ROOT / "camera_geometry.json"
    )
    args = parser.parse_args()

    fitting_groups = parse_groups(args.sample, "--sample")
    validation_groups = parse_groups(args.validate, "--validate")
    if len({distance for _, distance, _ in fitting_groups}) < 2:
        parser.error("--sample must include at least two distinct measured distances")

    vision = VisionProcessor(load_config(args.config))
    geometry = load_geometry_config(args.geometry)
    plate_cm = float(geometry["objects"]["information_sign_diameter_cm"])
    fit_samples: List[Tuple[float, float]] = []
    group_report: List[Dict[str, object]] = []
    for folder, distance, color in fitting_groups:
        diameters = measure_group(vision, folder, color)
        fit_samples.extend((distance, diameter) for diameter in diameters)
        group_report.append(
            {
                "role": "fit",
                "folder": folder.name,
                "distance_cm": distance,
                "color": color,
                "diameters_px": [round(value, 4) for value in diameters],
            }
        )

    focal_px, offset_cm, residuals = fit_range_model(fit_samples, plate_cm)
    rmse = math.sqrt(statistics.fmean(value * value for value in residuals))
    print(
        "FIT focal=%.4fpx camera_to_front=%.4fcm rmse=%.4fcm"
        % (focal_px, offset_cm, rmse)
    )

    for folder, distance, color in validation_groups:
        diameters = measure_group(vision, folder, color)
        errors = [
            focal_px * plate_cm / diameter - offset_cm - distance
            for diameter in diameters
        ]
        print(
            "VALIDATE", color, "at", distance, "cm:",
            "median_error=%.2fcm" % statistics.median(errors),
            "range=[%.2f, %.2f]cm" % (min(errors), max(errors)),
        )
        group_report.append(
            {
                "role": "validation",
                "folder": folder.name,
                "distance_cm": distance,
                "color": color,
                "diameters_px": [round(value, 4) for value in diameters],
                "errors_cm": [round(value, 4) for value in errors],
            }
        )

    if args.write:
        geometry["calibrated"] = True
        geometry["focal_length_px"] = round(focal_px, 4)
        geometry["camera_to_front_cm"] = round(offset_cm, 4)
        geometry["calibration"] = {
            "source": "measured XGO-camera sign photos; recognition library excluded",
            "method": (
                "least-squares fit of clearance=focal*(5cm/diameter_px)"
                "-camera_to_front_offset"
            ),
            "groups": group_report,
            "sample_count": len(fit_samples),
            "fit_rmse_cm": round(rmse, 4),
        }
        with args.geometry.open("w", encoding="utf-8") as stream:
            json.dump(geometry, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print("WROTE:", args.geometry)
    else:
        print("未写入配置；确认结果后增加 --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
