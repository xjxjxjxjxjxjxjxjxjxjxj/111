#!/usr/bin/env python3
"""Closed-loop vision for aligning the gripper over either task cup.

The cup mouth is a 70 mm circle in the rules, but appears as an ellipse in the
pitched XGO camera.  The controller therefore uses the observed ellipse centre
and major axis from real pre-release photos.  It issues one bounded correction
at a time; callers must settle and reobserve before the next command.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CupDetection:
    center_x: float
    center_y: float
    major_axis_px: float
    minor_axis_px: float
    confidence: float


@dataclass(frozen=True)
class CupApproachDecision:
    action: str
    x_error_px: float
    major_axis_error_px: float


class CupVision:
    """Detect the white cup rim as a horizontal ellipse in the lower frame."""

    def __init__(self, ball_config: Dict[str, Any]):
        self.camera_cfg = ball_config["camera"]
        self.config = ball_config["cup_alignment"]

    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        width = int(self.camera_cfg["width"])
        height = int(self.camera_cfg["height"])
        if frame.shape[1] != width or frame.shape[0] != height:
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return frame

    @staticmethod
    def _candidate_from_contour(contour: np.ndarray) -> Optional[CupDetection]:
        if len(contour) < 30:
            return None
        (center_x, center_y), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major = float(max(axis_a, axis_b))
        minor = float(min(axis_a, axis_b))
        ratio = major / max(minor, 1.0)
        _, _, bound_width, _ = cv2.boundingRect(contour)
        if not (50.0 <= major <= 160.0 and 18.0 <= minor <= 80.0):
            return None
        if not (1.5 <= ratio <= 4.0):
            return None
        if not (60.0 <= center_x <= 260.0 and 145.0 <= center_y <= 210.0):
            return None
        # Partial rim arcs are accepted, but a nearly vertical scratch is not.
        if bound_width < 0.25 * major:
            return None

        shape_score = max(0.0, 1.0 - abs(ratio - 2.30) / 1.50)
        y_score = max(0.0, 1.0 - abs(center_y - 190.0) / 55.0)
        size_score = max(0.0, 1.0 - abs(major - 95.0) / 80.0)
        confidence = min(0.98, 0.40 + 0.30 * shape_score + 0.15 * y_score + 0.15 * size_score)
        return CupDetection(center_x, center_y, major, minor, confidence)

    def detect(self, frame: np.ndarray) -> Optional[CupDetection]:
        """Return the most cup-like ellipse across three exposure thresholds."""
        prepared = self.prepare_frame(frame)
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        candidates: List[CupDetection] = []
        for low, high in ((18, 55), (28, 85), (38, 115)):
            edges = cv2.Canny(blurred, low, high)
            roi_edges = np.zeros_like(edges)
            roi_edges[115:220, 40:280] = edges[115:220, 40:280]
            contours = cv2.findContours(
                roi_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )[-2]
            for contour in contours:
                candidate = self._candidate_from_contour(contour)
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            return None

        target_x = float(self.config["target_x_px"])
        target_y = float(self.config["target_y_px"])
        target_major = float(self.config["target_major_axis_px"])
        # At the release stage the cup is already in the gripper's working
        # neighbourhood.  Shape confidence dominates; target proximity breaks
        # ties between a true rim and unrelated curved field markings.
        return max(
            candidates,
            key=lambda item: (
                item.confidence
                - 0.0015 * abs(item.center_x - target_x)
                - 0.0010 * abs(item.center_y - target_y)
                - 0.0010 * abs(item.major_axis_px - target_major)
            ),
        )

    @staticmethod
    def stable_detection(
        detections: Iterable[CupDetection], minimum: int = 4
    ) -> Optional[CupDetection]:
        values = list(detections)
        if len(values) < minimum:
            return None
        median_x = statistics.median(item.center_x for item in values)
        median_y = statistics.median(item.center_y for item in values)
        median_major = statistics.median(item.major_axis_px for item in values)
        stable = [
            item
            for item in values
            if math.hypot(item.center_x - median_x, item.center_y - median_y) <= 12.0
            and abs(item.major_axis_px - median_major) <= 14.0
        ]
        if len(stable) < minimum:
            return None
        return CupDetection(
            statistics.median(item.center_x for item in stable),
            statistics.median(item.center_y for item in stable),
            statistics.median(item.major_axis_px for item in stable),
            statistics.median(item.minor_axis_px for item in stable),
            statistics.median(item.confidence for item in stable),
        )

    def sample_from_camera(self, xgo_edu: Any) -> Optional[CupDetection]:
        xgo_edu.open_camera()
        camera = getattr(xgo_edu, "cap", None)
        if camera is None:
            return None
        observations: List[CupDetection] = []
        for _ in range(int(self.config["sampling_frames"]) * 3):
            ok, frame = camera.read()
            if ok and frame is not None:
                detected = self.detect(frame)
                if detected is not None:
                    observations.append(detected)
                    if len(observations) >= int(self.config["sampling_frames"]):
                        break
            time.sleep(float(self.config["sampling_interval_s"]))
        return self.stable_detection(
            observations, int(self.config["minimum_valid_frames"])
        )


class CupClosedLoopController:
    """Map the latest cup observation to one short lateral/forward pulse."""

    def __init__(self, ball_config: Dict[str, Any]):
        self.config = ball_config["cup_alignment"]
        self.motion = ball_config["motion"]

    @property
    def ready(self) -> bool:
        return bool(self.config.get("pre_release_calibrated", False))

    def decide(self, cup: CupDetection) -> CupApproachDecision:
        if not self.ready:
            raise RuntimeError("杯口释放姿态尚未标定")
        x_error = cup.center_x - float(self.config["target_x_px"])
        size_error = cup.major_axis_px - float(self.config["target_major_axis_px"])
        if abs(x_error) > float(self.config["x_tolerance_px"]):
            action = "move_left" if x_error < 0 else "move_right"
        elif size_error < -float(self.config["major_axis_tolerance_px"]):
            action = "move_forward"
        elif size_error > float(self.config["major_axis_tolerance_px"]):
            action = "move_backward"
        else:
            action = "aligned"
        return CupApproachDecision(action, x_error, size_error)

    def motion_for_decision(
        self, decision: CupApproachDecision
    ) -> Optional[Tuple[str, float, float]]:
        if decision.action == "aligned":
            return None
        if decision.action in ("move_left", "move_right"):
            command = 6.0 if decision.action == "move_left" else -6.0
            duration = min(0.45, max(0.22, abs(decision.x_error_px) * 0.010))
            return "y", command, duration
        command = 5.0 if decision.action == "move_forward" else -4.0
        duration = min(
            0.40,
            max(0.20, abs(decision.major_axis_error_px) * 0.012),
        )
        return "x", command, duration
