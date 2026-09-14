#!/usr/bin/env python3
"""Physical-distance closed-loop controller for ball approach.

The controller never compares a ball radius with a hard-coded grab threshold.
It converts the detected 28 mm ball diameter to centimetres using the shared
camera model, then returns exactly one short correction.  The caller must stop,
wait for the quadruped to settle, reobserve, and decide again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ball_vision import BallDetection
from camera_geometry import CalibrationRequiredError, PinholeRangeModel


@dataclass(frozen=True)
class BallApproachDecision:
    action: str
    x_error_px: float
    distance_cm: float
    distance_error_cm: float


class BallClosedLoopController:
    """Choose one bounded movement from the latest stable ball observation."""

    def __init__(self, config: Dict[str, Any], range_model: PinholeRangeModel):
        self.config = config
        self.alignment = config["alignment"]
        self.motion = config["motion"]
        self.range_model = range_model

    @property
    def ready(self) -> bool:
        return (
            bool(self.alignment.get("calibrated_for_grasp", False))
            and bool(self.alignment.get("profiles"))
            and self.range_model.calibrated
            and self.range_model.focal_length_px is not None
        )

    def require_ready(self) -> None:
        self.range_model.require_calibrated()
        if not bool(self.alignment.get("calibrated_for_grasp", False)):
            raise CalibrationRequiredError(
                "抓球闭环尚未标定：需要成功抓取位置的机械狗相机照片"
            )
        profiles = self.alignment.get("profiles", {})
        missing = [
            color
            for color in ("blue", "green", "red")
            if color not in profiles
            or profiles[color].get("target_x_px") is None
            or profiles[color].get("target_distance_cm") is None
        ]
        if missing:
            raise CalibrationRequiredError(
                "缺少分颜色抓取标定: %s" % ", ".join(missing)
            )

    def decide(self, detection: BallDetection) -> BallApproachDecision:
        """Use horizontal error first, then physical centimetre range."""
        self.require_ready()
        profiles = self.alignment["profiles"]
        if detection.label not in profiles:
            raise CalibrationRequiredError(
                "没有%s球的成功抓取标定" % detection.label
            )
        profile = profiles[detection.label]
        target_x = float(profile["target_x_px"])
        target_distance = float(profile["target_distance_cm"])
        x_error = float(detection.x) - target_x
        distance_cm = self.range_model.ball_distance_cm(
            2.0 * float(detection.radius)
        )
        distance_error = distance_cm - target_distance

        if abs(x_error) > float(profile["x_tolerance_px"]):
            action = "move_left" if x_error < 0 else "move_right"
        elif distance_error > float(profile["distance_tolerance_cm"]):
            action = "move_forward"
        elif distance_error < -float(profile["distance_tolerance_cm"]):
            action = "move_backward"
        else:
            action = "aligned"
        return BallApproachDecision(action, x_error, distance_cm, distance_error)

    def motion_for_decision(
        self, decision: BallApproachDecision
    ) -> Optional[Tuple[str, float, float]]:
        """Convert one decision to one bounded pulse; the next step must reobserve."""
        cfg = self.motion
        if decision.action == "aligned":
            return None
        if decision.action in ("move_left", "move_right"):
            magnitude = max(
                float(cfg["lateral_min_command"]),
                min(
                    float(cfg["lateral_max_command"]),
                    4.0 + abs(decision.x_error_px) * 0.10,
                ),
            )
            command = magnitude if decision.action == "move_left" else -magnitude
            duration = max(
                float(cfg["lateral_min_s"]),
                min(
                    float(cfg["lateral_max_s"]),
                    abs(decision.x_error_px)
                    * float(cfg["lateral_seconds_per_pixel"]),
                ),
            )
            return "y", command, duration

        command = (
            float(cfg["forward_command"])
            if decision.action == "move_forward"
            else float(cfg["backward_command"])
        )
        duration = min(
            float(cfg["longitudinal_max_s"]),
            float(cfg["longitudinal_base_s"])
            + abs(decision.distance_error_cm)
            * float(cfg["longitudinal_seconds_per_cm"]),
        )
        return "x", command, duration
