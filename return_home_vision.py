#!/usr/bin/env python3
"""Closed-loop visual evidence for the autonomous return to the start zone.

This module deliberately separates two different black-line problems:

1. ``ReturnGuideVision`` follows the original guide line on the way home.  It
   rejects dark components attached to the right image edge, because the right
   competition-field boundary is also a long black line.  Once a guide is
   locked, a new observation must stay close to the previous guide position;
   losing the guide makes the caller stop instead of switching to a boundary.
   A broad curved-line component in the floor region is used only as supporting
   confidence, never as permission to move by itself.
2. ``StartZoneTracker`` confirms entry into the rectangular start zone.  It
   first requires two transverse black boundary bands, then requires *zero*
   transverse bands across the full camera width for several frames.  One line
   fragment remaining at the left or right side therefore prevents completion,
   even when the robot is offset and the line appears diagonal.

The supplied return photographs are calibration/regression evidence only.  The
complete competition program remains disabled until cup-release verification
and the full return motion state machine have been tested on hardware.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "return_home_config.json"


def load_return_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load and minimally validate the auditable return-vision thresholds."""
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"camera", "dark_gray_max", "guide", "start_zone"}
    missing = required - set(config)
    if missing:
        raise ValueError("Return config sections missing: %s" % sorted(missing))
    return config


@dataclass(frozen=True)
class ReturnGuideDetection:
    """One accepted guide-line observation in the return camera view."""

    x_at_control_row: float
    normalized_error: float
    axis_angle_deg: float
    area: float
    confidence: float
    arc_reference_visible: bool
    touches_right_edge: bool


@dataclass(frozen=True)
class HomeBoundaryObservation:
    """Transverse start-zone boundary bands visible in one camera frame."""

    band_y: Tuple[float, ...]
    frame_width: int
    frame_height: int

    @property
    def band_count(self) -> int:
        return len(self.band_y)

    @property
    def all_transverse_bands_gone(self) -> bool:
        return self.band_count == 0


class ReturnGuideVision:
    """Select the return guide while refusing the right field boundary."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.camera_cfg = config["camera"]
        self.guide_cfg = config["guide"]

    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        width = int(self.camera_cfg["width"])
        height = int(self.camera_cfg["height"])
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if bool(self.camera_cfg.get("rotate_180", False)):
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def dark_mask(self, frame: np.ndarray) -> np.ndarray:
        prepared = self.prepare_frame(frame)
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.inRange(gray, 0, int(self.config["dark_gray_max"]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    @staticmethod
    def _axis_angle(contour: np.ndarray) -> Tuple[float, Tuple[float, float, float, float]]:
        vx, vy, x0, y0 = (float(value) for value in cv2.fitLine(
            contour, cv2.DIST_L2, 0, 0.01, 0.01
        ).reshape(-1))
        angle = math.degrees(math.atan2(vy, vx))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        return angle, (vx, vy, x0, y0)

    def _arc_reference_visible(
        self, contours: Iterable[np.ndarray], width: int, height: int
    ) -> bool:
        """Return supporting evidence for the broad lower circular arc.

        The cue is intentionally weak: a wide, shallow dark component in the
        floor ROI raises confidence but never selects a guide or commands motion.
        This avoids turning furniture or the rectangular boundary into a route.
        """
        min_width = float(self.guide_cfg["arc_min_width_ratio"]) * width
        max_height = float(self.guide_cfg["arc_max_height_ratio"]) * height
        for contour in contours:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if box_w >= min_width and 4 <= box_h <= max_height:
                aspect = box_w / max(float(box_h), 1.0)
                if aspect >= 2.2:
                    return True
        return False

    def detect(
        self, frame: np.ndarray, previous_x: Optional[float] = None
    ) -> Optional[ReturnGuideDetection]:
        """Detect the guide; return ``None`` rather than follow unsafe evidence."""
        prepared = self.prepare_frame(frame)
        height, width = prepared.shape[:2]
        mask = self.dark_mask(prepared)
        roi_top = int(round(float(self.guide_cfg["floor_roi_top"]) * height))
        floor_mask = mask[roi_top:, :]
        contours = list(cv2.findContours(
            floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2])
        arc_visible = self._arc_reference_visible(contours, width, height)
        if (
            previous_x is None
            and bool(self.guide_cfg.get("require_arc_for_first_lock", False))
            and not arc_visible
        ):
            # The first lock is the dangerous moment: without the course arc as
            # context, a lone right-side boundary must not start locomotion.
            return None

        candidates: List[Tuple[float, ReturnGuideDetection]] = []
        control_y = 0.88 * height - roi_top
        min_span = float(self.guide_cfg["min_vertical_span_ratio"]) * height
        right_limit = float(self.guide_cfg["right_edge_reject_ratio"]) * width
        target_x = float(self.guide_cfg["target_x_ratio"]) * width

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(self.guide_cfg["min_area_px"]):
                continue
            bx, by, box_w, box_h = cv2.boundingRect(contour)
            if box_h < min_span:
                continue
            touches_right = bx + box_w >= right_limit
            # The known right-side field boundary enters/leaves through the
            # image's right edge.  It is never allowed to become the guide.
            if touches_right:
                continue

            angle, (vx, vy, x0, y0) = self._axis_angle(contour)
            if abs(angle) < float(self.guide_cfg["min_axis_angle_deg"]):
                continue
            if abs(vy) < 1e-6:
                continue
            x_at_control = x0 + (control_y - y0) * vx / vy
            x_at_control = max(0.0, min(float(width - 1), x_at_control))
            error = (x_at_control - target_x) / max(width / 2.0, 1.0)

            if previous_x is not None:
                maximum_jump = float(self.guide_cfg["max_lock_jump_ratio"]) * width
                if abs(x_at_control - previous_x) > maximum_jump:
                    continue
                proximity = 1.0 - abs(x_at_control - previous_x) / maximum_jump
            else:
                proximity = 1.0 - min(abs(x_at_control - target_x) / (width / 2.0), 1.0)

            span_score = min(box_h / max(height * 0.45, 1.0), 1.0)
            area_score = min(area / max(width * height * 0.025, 1.0), 1.0)
            confidence = min(
                1.0,
                0.48 * proximity
                + 0.30 * span_score
                + 0.14 * area_score
                + (0.08 if arc_visible else 0.0),
            )
            detection = ReturnGuideDetection(
                x_at_control_row=float(x_at_control),
                normalized_error=float(error),
                axis_angle_deg=float(angle),
                area=area,
                confidence=float(confidence),
                arc_reference_visible=arc_visible,
                touches_right_edge=False,
            )
            candidates.append((confidence, detection))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]


class ReturnGuideTracker:
    """Require stable acquisition and prohibit jumps to another black line."""

    def __init__(self, vision: ReturnGuideVision):
        self.vision = vision
        self.locked_x: Optional[float] = None
        self.pending: List[ReturnGuideDetection] = []
        self.confirm_frames = int(vision.guide_cfg["confirm_frames"])

    def update(self, frame: np.ndarray) -> Optional[ReturnGuideDetection]:
        detection = self.vision.detect(frame, previous_x=self.locked_x)
        if detection is None:
            self.pending.clear()
            return None
        if self.locked_x is not None:
            self.locked_x = detection.x_at_control_row
            return detection
        self.pending.append(detection)
        if len(self.pending) > self.confirm_frames:
            self.pending.pop(0)
        if len(self.pending) < self.confirm_frames:
            return None
        positions = [item.x_at_control_row for item in self.pending]
        max_spread = float(self.vision.guide_cfg["max_lock_jump_ratio"]) * int(
            self.vision.camera_cfg["width"]
        )
        if max(positions) - min(positions) > max_spread:
            return None
        self.locked_x = float(np.median(positions))
        return detection

    def reset(self) -> None:
        self.locked_x = None
        self.pending.clear()


class StartZoneVision:
    """Detect long transverse black bands across the full camera width."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.camera_cfg = config["camera"]
        self.zone_cfg = config["start_zone"]
        self.guide_vision = ReturnGuideVision(config)

    @staticmethod
    def _canonical_angle(x1: int, y1: int, x2: int, y2: int) -> float:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        return angle

    @staticmethod
    def _group_positions(values: Sequence[float], tolerance: float) -> Tuple[float, ...]:
        if not values:
            return ()
        groups: List[List[float]] = []
        for value in sorted(values):
            if not groups or value - float(np.mean(groups[-1])) > tolerance:
                groups.append([value])
            else:
                groups[-1].append(value)
        return tuple(float(np.median(group)) for group in groups)

    def detect(self, frame: np.ndarray) -> HomeBoundaryObservation:
        prepared = self.guide_vision.prepare_frame(frame)
        height, width = prepared.shape[:2]
        mask = self.guide_vision.dark_mask(prepared)
        roi_top = int(round(float(self.zone_cfg["transverse_roi_top"]) * height))
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray[roi_top:, :], 35, 110)
        min_length = int(round(float(self.zone_cfg["min_segment_length_ratio"]) * width))
        max_gap = int(round(float(self.zone_cfg["max_line_gap_ratio"]) * width))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(24, min_length // 2),
            minLineLength=min_length,
            maxLineGap=max_gap,
        )

        y_at_center: List[float] = []
        max_angle = float(self.zone_cfg["max_abs_angle_deg"])
        if lines is not None:
            for x1, local_y1, x2, local_y2 in np.asarray(lines).reshape(-1, 4):
                y1 = int(local_y1) + roi_top
                y2 = int(local_y2) + roi_top
                angle = self._canonical_angle(int(x1), y1, int(x2), y2)
                if abs(angle) > max_angle or x1 == x2:
                    continue
                # Extrapolating partial/slanted segments to image center makes a
                # line at either side count; completion never uses a center crop.
                center_y = y1 + (width / 2.0 - x1) * (y2 - y1) / (x2 - x1)
                if roi_top <= center_y <= height + 0.08 * height:
                    y_at_center.append(float(center_y))

        bottom_rows = max(
            2, int(round(float(self.zone_cfg["bottom_band_height_ratio"]) * height))
        )
        bottom_dark = float(np.mean(mask[-bottom_rows:, :] > 0))
        if bottom_dark >= float(self.zone_cfg["bottom_band_min_dark_fraction"]):
            y_at_center.append(float(height - 1))

        tolerance = float(self.zone_cfg["group_y_tolerance_ratio"]) * height
        groups = self._group_positions(y_at_center, tolerance)
        return HomeBoundaryObservation(groups, width, height)


class StartZoneTracker:
    """Declare home only after two seen boundaries have both fully disappeared."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vision = StartZoneVision(config)
        zone = config["start_zone"]
        self.minimum_seen_bands = int(zone["minimum_seen_bands"])
        self.seen_confirm_frames = int(zone["seen_confirm_frames"])
        self.clear_confirm_frames = int(zone["clear_confirm_frames"])
        self.seen_streak = 0
        self.clear_streak = 0
        self.boundary_pair_armed = False

    def update_observation(self, observation: HomeBoundaryObservation) -> bool:
        if not self.boundary_pair_armed:
            if observation.band_count >= self.minimum_seen_bands:
                self.seen_streak += 1
            else:
                self.seen_streak = 0
            if self.seen_streak >= self.seen_confirm_frames:
                self.boundary_pair_armed = True

        if self.boundary_pair_armed and observation.all_transverse_bands_gone:
            self.clear_streak += 1
        else:
            # A remaining fragment on either side resets the completion timer.
            self.clear_streak = 0
        return self.boundary_pair_armed and self.clear_streak >= self.clear_confirm_frames

    def update(self, frame: np.ndarray) -> Tuple[HomeBoundaryObservation, bool]:
        observation = self.vision.detect(frame)
        return observation, self.update_observation(observation)

    def reset(self) -> None:
        self.seen_streak = 0
        self.clear_streak = 0
        self.boundary_pair_armed = False
