#!/usr/bin/env python3
"""Shared monocular distance model for the 50 mm signs and 28 mm balls.

The competition rules provide the real object diameters.  A measured camera
focal length in pixels is still required.  It is obtained from XGO-camera
images taken at a known distance; unknown-distance recognition photos are never
accepted as calibration input.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


class CalibrationRequiredError(RuntimeError):
    """Raised when a physical-distance decision is requested before calibration."""


@dataclass(frozen=True)
class PinholeRangeModel:
    """Estimate range from the apparent diameter of a known-size round object."""

    calibrated: bool
    focal_length_px: Optional[float]
    principal_x_px: float
    sign_diameter_cm: float
    ball_diameter_cm: float
    # The rule's 20 cm is the clearance from the front of the dog to the
    # object, while the optical equation measures from the camera.  The fitted
    # offset keeps those two reference points from being silently mixed.
    camera_to_front_cm: float = 0.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PinholeRangeModel":
        focal = config.get("focal_length_px")
        return cls(
            calibrated=bool(config.get("calibrated", False)),
            focal_length_px=None if focal is None else float(focal),
            principal_x_px=float(config["principal_x_px"]),
            sign_diameter_cm=float(
                config["objects"]["information_sign_diameter_cm"]
            ),
            ball_diameter_cm=float(config["objects"]["ball_diameter_cm"]),
            camera_to_front_cm=float(config.get("camera_to_front_cm", 0.0)),
        )

    def require_calibrated(self) -> None:
        if (
            not self.calibrated
            or self.focal_length_px is None
            or not math.isfinite(self.focal_length_px)
            or self.focal_length_px <= 0
        ):
            raise CalibrationRequiredError(
                "相机距离模型尚未标定：请先使用机械狗相机拍摄已知距离的路牌"
            )

    def distance_cm(self, apparent_diameter_px: float, object_diameter_cm: float) -> float:
        """Return camera-to-object distance using the pinhole diameter equation."""
        self.require_calibrated()
        if not math.isfinite(apparent_diameter_px) or apparent_diameter_px <= 0:
            raise ValueError("apparent_diameter_px must be positive")
        if not math.isfinite(object_diameter_cm) or object_diameter_cm <= 0:
            raise ValueError("object_diameter_cm must be positive")
        camera_range = (
            float(self.focal_length_px) * object_diameter_cm / apparent_diameter_px
        )
        return max(0.0, camera_range - self.camera_to_front_cm)

    def sign_distance_cm(self, apparent_diameter_px: float) -> float:
        return self.distance_cm(apparent_diameter_px, self.sign_diameter_cm)

    def ball_distance_cm(self, apparent_diameter_px: float) -> float:
        return self.distance_cm(apparent_diameter_px, self.ball_diameter_cm)

    def expected_diameter_px(self, distance_cm: float, object_diameter_cm: float) -> float:
        self.require_calibrated()
        camera_range = float(distance_cm) + self.camera_to_front_cm
        if camera_range <= 0:
            raise ValueError("distance_cm plus camera offset must be positive")
        return float(self.focal_length_px) * object_diameter_cm / camera_range


def fit_range_model(
    samples: Sequence[Tuple[float, float]], object_diameter_cm: float
) -> Tuple[float, float, Tuple[float, ...]]:
    """Fit focal pixels and the camera-to-front offset from 2+ distances.

    Every sample is ``(front_clearance_cm, apparent_diameter_px)``.  Rewriting
    the pinhole equation gives a straight line::

        clearance = focal_px * (object_diameter_cm / diameter_px) - offset

    A two-distance fit is important here: a single 20 cm photo cannot
    distinguish the focal scale from the camera's mounting setback.
    """
    if object_diameter_cm <= 0:
        raise ValueError("object_diameter_cm must be positive")
    valid = [
        (float(distance), float(diameter))
        for distance, diameter in samples
        if math.isfinite(float(distance))
        and math.isfinite(float(diameter))
        and float(distance) > 0
        and float(diameter) > 0
    ]
    if len(valid) < 2:
        raise ValueError("At least two valid distance samples are required")

    xs = [object_diameter_cm / diameter for _, diameter in valid]
    ys = [distance for distance, _ in valid]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 1e-12:
        raise ValueError("Calibration must contain at least two distinct distances")
    focal_px = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / denominator
    intercept = mean_y - focal_px * mean_x
    camera_to_front_cm = -intercept
    if focal_px <= 0 or camera_to_front_cm < 0:
        raise ValueError(
            "Calibration produced non-physical focal length or camera offset"
        )
    residuals = tuple(
        focal_px * x - camera_to_front_cm - y for x, y in zip(xs, ys)
    )
    return focal_px, camera_to_front_cm, residuals


def focal_length_from_reference(
    apparent_diameters_px: Iterable[float],
    reference_distance_cm: float,
    object_diameter_cm: float,
) -> float:
    """Calculate a robust effective focal length from known-distance samples."""
    diameters = [
        float(value)
        for value in apparent_diameters_px
        if math.isfinite(float(value)) and float(value) > 0
    ]
    if not diameters:
        raise ValueError("No valid apparent-diameter samples")
    if reference_distance_cm <= 0 or object_diameter_cm <= 0:
        raise ValueError("Physical distances and diameters must be positive")
    median_diameter = statistics.median(diameters)
    return median_diameter * float(reference_distance_cm) / float(object_diameter_cm)


def load_geometry_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"model", "calibrated", "focal_length_px", "principal_x_px", "objects"}
    missing = required - set(config)
    if missing:
        raise ValueError("Camera geometry fields missing: %s" % sorted(missing))
    return config
