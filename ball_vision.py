#!/usr/bin/env python3
"""Ball recognition and separately gated alignment helpers for XGO.

This module deliberately does not import XGO hardware libraries.  It can be
tested on a PC, while ``sample_from_camera`` accepts the senior project's
existing XGOEDU object and reuses its camera.  OpenCV frames are handled as BGR;
do not swap the red and blue channels before converting to HSV.

The supplied photos calibrate colour/shape recognition only.  They must not be
used to infer a grasp point, a grasp distance, or the sign-warning distance.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class BallDetection:
    """One filtered ball observation in a 320 x 240 camera frame."""

    label: str
    x: float
    y: float
    radius: float
    area: float
    circularity: float
    confidence: float


@dataclass(frozen=True)
class RemovalCheck:
    """Post-grasp visual evidence that the ball left its source position."""

    removed: bool
    valid_frames: int
    source_hits: int


@dataclass(frozen=True)
class BallPositionHint:
    """Last visual observation of one ball; never a blind-motion command."""

    color: str
    x: float
    y: float
    radius: float
    distance_cm: float
    observed_at: float


def load_ball_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "camera",
        "ball_roi",
        "colors",
        "geometry",
        "sampling",
        "alignment",
        "motion",
        "grasp_verification",
        "target_memory",
        "cup_alignment",
    }
    missing = required - set(config)
    if missing:
        raise ValueError("Ball config sections missing: %s" % sorted(missing))
    return config


class BallVision:
    """Detect red, green or blue balls using HSV plus geometric filtering."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.camera_cfg = config["camera"]
        self.geometry_cfg = config["geometry"]
        self.sampling_cfg = config["sampling"]
        self.verification_cfg = config["grasp_verification"]

    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        width = int(self.camera_cfg["width"])
        height = int(self.camera_cfg["height"])
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return frame

    def _roi_bounds(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        left, top, right, bottom = (float(x) for x in self.config["ball_roi"])
        return (
            max(0, min(width - 1, int(round(left * width)))),
            max(0, min(height - 1, int(round(top * height)))),
            max(1, min(width, int(round(right * width)))),
            max(1, min(height, int(round(bottom * height)))),
        )

    def color_mask(self, frame: np.ndarray, color_name: str) -> np.ndarray:
        """Build one clean mask; red uses both ends of OpenCV's hue scale."""
        if color_name not in self.config["colors"]:
            raise ValueError("Unsupported ball color: %s" % color_name)

        prepared = self.prepare_frame(frame)
        blurred = cv2.GaussianBlur(prepared, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.config["colors"][color_name]["hsv_ranges"]:
            current = cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )
            mask = cv2.bitwise_or(mask, current)

        x1, y1, x2, y2 = self._roi_bounds(prepared)
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, small, iterations=1)
        return cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, large, iterations=1)

    def detect(self, frame: np.ndarray, color_name: str) -> Optional[BallDetection]:
        """Return the strongest ball-like component for one requested colour."""
        prepared = self.prepare_frame(frame)
        mask = self.color_mask(prepared, color_name)
        cfg = self.geometry_cfg
        best: Optional[BallDetection] = None
        best_score = -1.0

        contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(cfg["min_area_px"]):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect = bw / max(float(bh), 1.0)
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
            fill_ratio = area / max(math.pi * radius * radius, 1.0)
            if not float(cfg["min_radius_px"]) <= radius <= float(cfg["max_radius_px"]):
                continue
            if not float(cfg["min_aspect"]) <= aspect <= float(cfg["max_aspect"]):
                continue
            if circularity < float(cfg["min_circularity"]):
                continue
            if fill_ratio < float(cfg["min_fill_ratio"]):
                continue

            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])
            confidence = max(
                0.0,
                min(1.0, 0.10 + 0.52 * circularity + 0.38 * min(fill_ratio, 1.0)),
            )
            candidate = BallDetection(
                color_name,
                center_x,
                center_y,
                float(radius),
                area,
                circularity,
                confidence,
            )
            score = area * confidence
            if score > best_score:
                best = candidate
                best_score = score
        return best

    def detect_all(self, frame: np.ndarray) -> Dict[str, Optional[BallDetection]]:
        return {
            color_name: self.detect(frame, color_name)
            for color_name in ("blue", "green", "red")
        }

    @staticmethod
    def nearest_detection(
        detections: Dict[str, Optional[BallDetection]], range_model: Any
    ) -> Optional[BallDetection]:
        """Choose the physically nearest currently visible ball.

        All three rule balls have the same 28 mm diameter, but converting to
        centimetres makes the selection rule explicit and keeps it consistent
        with the later closed-loop approach.  The function does not use the
        other balls' expected red/green/blue layout.
        """
        visible = [item for item in detections.values() if item is not None]
        if not visible:
            return None
        return min(
            visible,
            key=lambda item: range_model.ball_distance_cm(2.0 * item.radius),
        )

    @staticmethod
    def _median_detection(
        detections: Iterable[BallDetection], color_name: str
    ) -> Optional[BallDetection]:
        values = list(detections)
        if not values:
            return None
        return BallDetection(
            color_name,
            statistics.median(x.x for x in values),
            statistics.median(x.y for x in values),
            statistics.median(x.radius for x in values),
            statistics.median(x.area for x in values),
            statistics.median(x.circularity for x in values),
            statistics.median(x.confidence for x in values),
        )

    def stable_detection(
        self, detections: Sequence[BallDetection], color_name: str
    ) -> Optional[BallDetection]:
        """Reject one-frame hits and retain a stable cluster around the median."""
        cfg = self.sampling_cfg
        if len(detections) < int(cfg["min_valid_frames"]):
            return None
        median = self._median_detection(detections, color_name)
        if median is None:
            return None
        stable = [
            item
            for item in detections
            if math.hypot(item.x - median.x, item.y - median.y)
            <= float(cfg["max_center_jitter_px"])
            and abs(item.radius - median.radius)
            <= float(cfg["max_radius_jitter_px"])
        ]
        if len(stable) < int(cfg["min_valid_frames"]):
            return None
        return self._median_detection(stable, color_name)

    def sample_from_camera(self, xgo_edu: Any, color_name: str) -> Optional[BallDetection]:
        """Reuse XGOEDU's camera and aggregate several frames without reopening it."""
        xgo_edu.open_camera()
        camera = getattr(xgo_edu, "cap", None)
        if camera is None:
            return None

        cfg = self.sampling_cfg
        deadline = time.monotonic() + float(cfg["timeout_s"])
        requested = int(cfg["frames"])
        attempts = 0
        detections: List[BallDetection] = []
        while time.monotonic() < deadline and attempts < requested * 3:
            attempts += 1
            ok, frame = camera.read()
            if ok and frame is not None:
                detection = self.detect(frame, color_name)
                if detection is not None:
                    detections.append(detection)
                    if len(detections) >= requested:
                        break
            time.sleep(float(cfg["frame_interval_s"]))
        return self.stable_detection(detections, color_name)

    def sample_all_from_camera(
        self, xgo_edu: Any
    ) -> Dict[str, Optional[BallDetection]]:
        """Take one shared frame batch and stabilize all three ball colours."""
        xgo_edu.open_camera()
        camera = getattr(xgo_edu, "cap", None)
        empty = {color: None for color in ("blue", "green", "red")}
        if camera is None:
            return empty
        cfg = self.sampling_cfg
        requested = int(cfg["frames"])
        deadline = time.monotonic() + float(cfg["timeout_s"])
        samples: Dict[str, List[BallDetection]] = {
            color: [] for color in ("blue", "green", "red")
        }
        attempts = 0
        while time.monotonic() < deadline and attempts < requested * 3:
            attempts += 1
            ok, frame = camera.read()
            if ok and frame is not None:
                for color, detection in self.detect_all(frame).items():
                    if detection is not None:
                        samples[color].append(detection)
                if all(len(values) >= requested for values in samples.values()):
                    break
            time.sleep(float(cfg["frame_interval_s"]))
        return {
            color: self.stable_detection(values, color)
            for color, values in samples.items()
        }

    def confirm_source_removed_from_camera(
        self,
        xgo_edu: Any,
        color_name: str,
        source_detection: BallDetection,
    ) -> RemovalCheck:
        """Verify a grasp by checking that the ball left its former image region.

        A camera read failure is never treated as success.  The check allows a
        held ball to remain visible elsewhere in the frame; only repeated hits
        near the pre-grasp source centre count as a failed pickup.
        """
        xgo_edu.open_camera()
        camera = getattr(xgo_edu, "cap", None)
        if camera is None:
            return RemovalCheck(False, 0, 0)

        cfg = self.verification_cfg
        requested = int(cfg["frames"])
        valid_frames = 0
        source_hits = 0
        attempts = 0
        while valid_frames < requested and attempts < requested * 3:
            attempts += 1
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(float(cfg["frame_interval_s"]))
                continue
            valid_frames += 1
            detection = self.detect(frame, color_name)
            if detection is not None:
                separation = math.hypot(
                    detection.x - source_detection.x,
                    detection.y - source_detection.y,
                )
                if separation <= float(cfg["source_neighborhood_px"]):
                    source_hits += 1
            time.sleep(float(cfg["frame_interval_s"]))

        removed = (
            valid_frames >= int(cfg["required_valid_frames"])
            and source_hits <= int(cfg["maximum_source_hits"])
        )
        return RemovalCheck(removed, valid_frames, source_hits)


class BallSceneMemory:
    """Lock one ball and retain the other two as stale-able search hints.

    A hint records where a ball was last observed.  It may guide a future
    camera scan, but every locomotion correction still requires a fresh visual
    detection because another ball can roll when the dog touches the field.
    """

    def __init__(self, config: Dict[str, Any], range_model: Any):
        self.config = config["target_memory"]
        self.range_model = range_model
        self.locked_color: Optional[str] = None
        self.hints: Dict[str, BallPositionHint] = {}
        self.completed_colors = set()

    def update(
        self, detections: Dict[str, Optional[BallDetection]], now: Optional[float] = None
    ) -> Optional[str]:
        """Refresh all visible positions and lock the nearest unfinished ball."""
        observed_at = time.monotonic() if now is None else float(now)
        for color, detection in detections.items():
            if detection is None or color in self.completed_colors:
                continue
            distance = self.range_model.ball_distance_cm(2.0 * detection.radius)
            self.hints[color] = BallPositionHint(
                color,
                detection.x,
                detection.y,
                detection.radius,
                distance,
                observed_at,
            )

        if self.locked_color in self.completed_colors:
            self.locked_color = None
        if self.locked_color is None:
            unfinished = {
                color: detection
                for color, detection in detections.items()
                if detection is not None and color not in self.completed_colors
            }
            nearest = BallVision.nearest_detection(unfinished, self.range_model)
            self.locked_color = None if nearest is None else nearest.label
        self.discard_stale(observed_at)
        return self.locked_color

    def mark_completed(self, color: str) -> None:
        self.completed_colors.add(color)
        self.hints.pop(color, None)
        if self.locked_color == color:
            self.locked_color = None

    def discard_stale(self, now: Optional[float] = None) -> None:
        current = time.monotonic() if now is None else float(now)
        max_age = float(self.config["maximum_hint_age_s"])
        self.hints = {
            color: hint
            for color, hint in self.hints.items()
            if current - hint.observed_at <= max_age
        }
