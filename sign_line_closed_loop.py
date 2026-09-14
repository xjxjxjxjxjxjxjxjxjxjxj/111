#!/usr/bin/env python3
"""XGO autonomous line following and yellow/black sign handling.

The competition sign is a 50 mm circular plate placed on the black guide line:
yellow means danger (alarm and bypass), black means obstacle (alarm and knock it
down).  Run in ``observe`` mode before enabling the motors.

Rule-to-code mapping for future maintainers and AI agents:

1. FOLLOW: use camera feedback to follow the approximately 20 mm black line.
2. APPROACH: convert the detected 50 mm plate diameter to centimetres with the
   calibrated camera model.  Forward speed is recalculated on every frame.
3. CONFIRM: stop at 20 cm only after N stable distance and colour observations.
4. BLACK_ADVANCE: alarm, continue closed-loop line following while the arm waves
   up/down for exactly five seconds, then stop and exit.
5. YELLOW_RECOVER: alarm, bypass without contact, reacquire and center the black
   line for N frames, stop, exit.

This program intentionally ends after handling the one random information sign.
It does not implement the later ball-placement part of the autonomous stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from camera_geometry import (
    CalibrationRequiredError,
    PinholeRangeModel,
    load_geometry_config,
)
from run_recording import RunRecorder


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "line_sign_config.json"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def limit_reacquire_command(
    error: float,
    follower_forward: float,
    follower_turn: float,
    action_cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """Bound the yellow reacquire command so the dog aligns instead of spinning.

    A far-off guide line previously saturated the turn term (up to the 48 deg/s
    cap) while still creeping forward, so the dog circled in place and never
    reacquired the line (field report 2026-09-14 dog18: 原地/小幅打转).  When the
    error is large the command now holds position and turns in place with a
    capped yaw; once roughly aligned it resumes a bounded forward follow. 偏离较大
    时限幅原地转向对齐，避免满舵打转。
    """

    align_error = float(action_cfg.get("yellow_align_error", 0.35))
    max_seek_turn = float(action_cfg.get("yellow_reacquire_max_turn", 20.0))
    forward = min(follower_forward, float(action_cfg["yellow_reacquire_speed"]))
    # The recovery ceiling applies in both alignment and forward-follow modes.
    # Otherwise one derivative spike just inside ``yellow_align_error`` can
    # still send the original 48-degree command. 回线全过程限制转向峰值。
    turn = clamp(follower_turn, -max_seek_turn, max_seek_turn)
    if abs(error) > align_error:
        forward = 0.0
        # Use a bounded proportional alignment command rather than the normal
        # PID derivative while stationary. This removes sign-flipping spikes
        # when the detector changes segments. 原地对齐不用巡线微分项。
        align_kp = float(action_cfg.get("yellow_align_kp", 24.0))
        turn = clamp(-align_kp * error, -max_seek_turn, max_seek_turn)
    return forward, turn


def reacquire_search_turn(
    last_error: Optional[float], action_cfg: Dict[str, Any]
) -> float:
    """Return a slow deterministic yaw while the guide line is not visible.

    A non-zero last error supplies the search side. If the line was centered
    before the open-loop detour, its expected side follows the configured
    bypass direction (a right bypass leaves the guide to the left). This avoids
    the old zero-error deadlock without adding forward motion. 丢线时只低速转向
    搜索；绕行前误差接近零时按绕行方向推断搜索侧，绝不盲目前进。
    """

    max_turn = abs(float(action_cfg.get("yellow_reacquire_max_turn", 20.0)))
    search_turn = min(
        max_turn, abs(float(action_cfg.get("yellow_search_turn", 10.0)))
    )
    direction_deadband = abs(
        float(action_cfg.get("yellow_search_error_deadband", 0.05))
    )
    if last_error is not None and abs(last_error) > direction_deadband:
        return math.copysign(search_turn, -last_error)
    bypass_direction = str(action_cfg.get("bypass_direction", "right")).lower()
    return search_turn if bypass_direction == "right" else -search_turn


def gate_initial_reacquire_error(
    error: Optional[float],
    already_acquired: bool,
    action_cfg: Dict[str, Any],
) -> Tuple[Optional[float], bool]:
    """Reject far-edge markings until a plausible guide-line lock is acquired.

    dog18 first saw the circular boundary at error ~= -1.0 after its fixed
    bypass. Treating that boundary as the guide immediately caused a saturated
    turn. Before the first plausible lock, only a line within the configured
    acquisition window may steer the dog; after lock, temporal tracking governs
    continuity. 首次回线前拒绝画面边缘圆弧，避免把赛区圆环当成引导直线。
    """

    if error is None:
        return None, already_acquired
    if already_acquired:
        return error, True
    maximum = float(action_cfg.get("yellow_acquire_error_max", 0.65))
    if abs(error) > maximum:
        return None, False
    return error, True


@dataclass(frozen=True)
class Detection:
    label: str
    center: Tuple[int, int]
    radius: float
    area: float
    confidence: float


@dataclass
class FrameAnalysis:
    line_error: Optional[float]
    line_coverage: float
    sign: Optional[Detection]
    debug_frame: Optional[np.ndarray] = None


class VisionProcessor:
    """Detect the guide line and a circular yellow/black sign in one frame."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.camera_cfg = config["camera"]
        self.line_cfg = config["line"]
        self.sign_cfg = config["sign"]
        # Temporal anchor used only to choose between current-frame candidates.
        # It never turns a missing line into a detection, so the existing
        # line-lost safety stop remains authoritative. 跨帧锚点只用于候选消歧，
        # 丢线帧仍返回 None，绝不绕过丢线停车安全锁。
        self._last_line_center: Optional[float] = None
        self._filtered_line_center: Optional[float] = None
        self._line_tracking_misses = 0

    def reset_line_tracking(self) -> None:
        """Forget line history at a deliberate manoeuvre boundary / 动作切换清空历史。"""

        self._last_line_center = None
        self._filtered_line_center = None
        self._line_tracking_misses = 0

    @staticmethod
    def _narrow_runs(
        dark_fraction: np.ndarray,
        dark_threshold: float,
        min_width: int,
        max_width: int,
    ) -> List[Tuple[int, int]]:
        """Return contiguous narrow dark runs from one horizontal scan band."""

        runs: List[Tuple[int, int]] = []
        run_start: Optional[int] = None
        for column, fraction in enumerate(dark_fraction):
            if fraction >= dark_threshold and run_start is None:
                run_start = column
            elif fraction < dark_threshold and run_start is not None:
                runs.append((run_start, column - 1))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(dark_fraction) - 1))
        return [
            (start, end)
            for start, end in runs
            if min_width <= (end - start + 1) <= max_width
        ]

    @staticmethod
    def _roi_rect(
        shape: Sequence[int], normalized: Sequence[float]
    ) -> Tuple[int, int, int, int]:
        height, width = shape[:2]
        x1 = int(clamp(float(normalized[0]), 0.0, 1.0) * width)
        y1 = int(clamp(float(normalized[1]), 0.0, 1.0) * height)
        x2 = int(clamp(float(normalized[2]), 0.0, 1.0) * width)
        y2 = int(clamp(float(normalized[3]), 0.0, 1.0) * height)
        return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)

    @staticmethod
    def _contours(mask: np.ndarray) -> Sequence[np.ndarray]:
        return cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        width = int(self.camera_cfg["width"])
        height = int(self.camera_cfg["height"])
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if bool(self.camera_cfg.get("rotate_180", False)):
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def _detect_line(
        self, frame: np.ndarray
    ) -> Tuple[Optional[float], float, np.ndarray, Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = self._roi_rect(frame.shape, self.line_cfg["roi"])
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(
            gray,
            int(self.line_cfg["gray_max"]),
            255,
            cv2.THRESH_BINARY_INV,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        roi_area = float(mask.shape[0] * mask.shape[1])
        height, width = mask.shape[:2]
        # dog15/dog18 evidence: whole-ROI blobs merged the guide with shadows,
        # while a single horizontal scan could jump between the straight guide
        # and the circular field marking. Sample several mid/lower bands, require
        # agreement, then prefer the track continuous with the previous frame.
        # 多采样带投票并结合上一帧位置选段，避免在直线、圆弧和阴影间跳变。
        band_h = int(self.line_cfg.get("scan_band_px", 18))
        band_h = max(4, min(height, band_h))
        center_ratio = float(self.line_cfg.get("scan_band_offset_ratio", 0.66))
        configured_offsets = self.line_cfg.get(
            "scan_band_offsets_ratio", [center_ratio]
        )
        offsets = sorted(
            {
                float(clamp(float(value), 0.0, 1.0))
                for value in configured_offsets
            }
        )
        if not offsets:
            offsets = [center_ratio]
        dark_threshold = float(self.line_cfg.get("scan_col_dark", 0.55))
        min_width = int(self.line_cfg.get("scan_min_width_px", 4))
        max_width = int(float(self.line_cfg.get("scan_max_width_ratio", 0.22)) * width)
        max_width = max(min_width + 1, max_width)
        runs_by_band: List[List[Tuple[int, int]]] = []
        for offset in offsets:
            center_row = int(offset * height)
            band_top = max(0, min(height - band_h, center_row - band_h // 2))
            dark_fraction = (mask[band_top : band_top + band_h, :] > 0).mean(axis=0)
            runs_by_band.append(
                self._narrow_runs(
                    dark_fraction, dark_threshold, min_width, max_width
                )
            )

        min_votes = max(
            1,
            min(
                len(offsets),
                int(self.line_cfg.get("scan_min_band_votes", 1)),
            ),
        )
        max_band_shift = float(
            self.line_cfg.get("scan_max_band_shift_ratio", 0.16)
        ) * width
        max_frame_jump = float(
            self.line_cfg.get("scan_max_frame_jump_ratio", 0.24)
        ) * width
        target_offset = float(self.line_cfg.get("line_target_offset", 0.0))
        target_center = width / 2.0 * (1.0 + target_offset)

        tracks: List[Tuple[int, float, float, float, float]] = []
        for anchor_band, runs in enumerate(runs_by_band):
            for start, end in runs:
                anchor_center = (start + end) / 2.0
                members: List[Tuple[int, float, int]] = [
                    (anchor_band, anchor_center, end - start + 1)
                ]
                for other_band, other_runs in enumerate(runs_by_band):
                    if other_band == anchor_band or not other_runs:
                        continue
                    nearest = min(
                        other_runs,
                        key=lambda run: abs((run[0] + run[1]) / 2.0 - anchor_center),
                    )
                    other_center = (nearest[0] + nearest[1]) / 2.0
                    if abs(other_center - anchor_center) <= max_band_shift:
                        members.append(
                            (other_band, other_center, nearest[1] - nearest[0] + 1)
                        )
                if len(members) < min_votes:
                    continue
                # The median represents the cross-band track without letting
                # one circular-arc intersection own the steering point.
                # 中位数融合多带中心，单个圆弧交点不能主导转向。
                raw_center = float(np.median([value for _, value, _ in members]))
                run_width = float(np.median([value for _, _, value in members]))
                spread = max(value for _, value, _ in members) - min(
                    value for _, value, _ in members
                )
                temporal_distance = (
                    abs(raw_center - self._last_line_center)
                    if self._last_line_center is not None
                    else abs(raw_center - target_center)
                )
                if (
                    self._last_line_center is not None
                    and temporal_distance > max_frame_jump
                ):
                    continue
                tracks.append(
                    (
                        len(members),
                        temporal_distance,
                        spread,
                        raw_center,
                        run_width,
                    )
                )

        if not tracks:
            self._line_tracking_misses += 1
            if self._line_tracking_misses >= int(
                self.line_cfg.get("line_memory_miss_frames", 5)
            ):
                self._last_line_center = None
                self._filtered_line_center = None
            return None, 0.0, mask, (x1, y1, x2, y2)

        # Balance band votes against temporal distance. Pure "most votes wins"
        # switched dog16 from its guide to a five-band border fragment; pure
        # nearest-neighbour tracking can latch onto noise. The weighted score
        # preserves both evidence sources. 投票数与跨帧连续性共同打分。
        continuity_scale = max(
            1.0,
            float(self.line_cfg.get("scan_continuity_scale_ratio", 0.10)) * width,
        )
        votes, _, _, raw_center, run_width = max(
            tracks,
            key=lambda item: (
                item[0]
                - item[1] / continuity_scale
                - item[2] / (2.0 * width),
                -item[1],
                -item[2],
            ),
        )
        del votes
        alpha = float(
            clamp(float(self.line_cfg.get("line_error_ema_alpha", 0.60)), 0.0, 1.0)
        )
        cx = raw_center
        if self._filtered_line_center is not None:
            cx = alpha * raw_center + (1.0 - alpha) * self._filtered_line_center
        # Candidate continuity follows the raw observation; the separate EMA
        # only damps the command. A lagging EMA must not reject a real moving
        # guide on the next frame. 原始中心负责建轨，平滑中心只负责控制。
        self._last_line_center = raw_center
        self._filtered_line_center = cx
        self._line_tracking_misses = 0
        normalized_error = (cx - width / 2.0) / (width / 2.0) - target_offset
        coverage = (run_width * band_h) / roi_area
        return (
            float(clamp(normalized_error, -1.0, 1.0)),
            float(coverage),
            mask,
            (x1, y1, x2, y2),
        )

    def _candidate_from_mask(
        self,
        mask: np.ndarray,
        offset: Tuple[int, int],
        label: str,
    ) -> Optional[Detection]:
        cfg = self.sign_cfg
        best: Optional[Detection] = None
        roi_height, roi_width = mask.shape[:2]
        edge_margin = int(cfg.get("edge_margin_px", 0))
        for contour in self._contours(mask):
            area = float(cv2.contourArea(contour))
            if area < float(cfg["min_area_px"]):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect = bw / max(float(bh), 1.0)
            (_, _), radius = cv2.minEnclosingCircle(contour)
            # A target touching an ROI edge is incomplete and its apparent radius
            # is unreliable for the competition's approximately-20-cm trigger.
            if (
                bx <= edge_margin
                or by <= edge_margin
                or bx + bw >= roi_width - edge_margin
                or by + bh >= roi_height - edge_margin
            ):
                continue
            if radius < float(cfg["min_radius_px"]):
                continue
            if not float(cfg["min_aspect"]) <= aspect <= float(cfg["max_aspect"]):
                continue
            if circularity < float(cfg["min_circularity"]):
                continue

            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            cx = int(round(moments["m10"] / moments["m00"])) + offset[0]
            cy = int(round(moments["m01"] / moments["m00"])) + offset[1]
            fill_ratio = area / max(math.pi * radius * radius, 1.0)
            if fill_ratio < float(cfg.get("min_fill_ratio", 0.0)):
                continue
            # Do not constrain the plaque to the guide line's apparent x position.
            # Perspective, camera mounting and course layout can place a valid sign
            # anywhere inside the configured upper ROI.
            confidence = clamp(
                0.12 + 0.50 * circularity + 0.38 * min(fill_ratio, 1.0),
                0.0,
                1.0,
            )
            if confidence < float(cfg.get("min_candidate_confidence", 0.0)):
                continue
            candidate = Detection(label, (cx, cy), float(radius), area, confidence)
            if best is None or candidate.radius > best.radius:
                best = candidate
        return best

    def _detect_sign(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Detection], np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = self._roi_rect(frame.shape, self.sign_cfg["roi"])
        roi = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_low = np.array(self.sign_cfg["yellow_hsv_low"], dtype=np.uint8)
        yellow_high = np.array(self.sign_cfg["yellow_hsv_high"], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)
        yellow_mask = cv2.medianBlur(yellow_mask, 5)
        yellow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        yellow_mask = cv2.morphologyEx(
            yellow_mask, cv2.MORPH_OPEN, yellow_kernel, iterations=1
        )
        yellow_mask = cv2.morphologyEx(
            yellow_mask, cv2.MORPH_CLOSE, yellow_kernel, iterations=2
        )
        yellow = self._candidate_from_mask(yellow_mask, (x1, y1), "yellow")

        # Use HSV brightness rather than a fixed grayscale threshold.  The limit
        # follows the current scene brightness but remains within calibrated
        # bounds, so a black plaque survives exposure changes without accepting
        # every ordinary mid-tone object as black.
        value = hsv[:, :, 2]
        scene_value = float(np.median(value))
        black_limit = int(
            clamp(
                scene_value * float(self.sign_cfg.get("black_value_ratio", 0.38)),
                float(self.sign_cfg.get("black_value_min", 0)),
                float(self.sign_cfg["black_value_max"]),
            )
        )
        black_mask = cv2.inRange(value, 0, black_limit)
        black_mask = cv2.medianBlur(black_mask, 5)
        # A wide, low opening removes the thin vertical pole.  It separates
        # the circular plaque from the guide line without erasing the plaque.
        open_size = int(self.sign_cfg["black_open_kernel"])
        if open_size % 2 == 0:
            open_size += 1
        black_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (open_size, 3)
        )
        black_mask = cv2.morphologyEx(
            black_mask, cv2.MORPH_OPEN, black_kernel, iterations=1
        )
        black_mask = cv2.morphologyEx(
            black_mask, cv2.MORPH_CLOSE, yellow_kernel, iterations=1
        )
        black = self._candidate_from_mask(black_mask, (x1, y1), "black")
        if black is None and bool(
            self.sign_cfg.get("black_hough", {}).get("enabled", False)
        ):
            black = self._black_hough_candidate(roi, (x1, y1))

        if yellow and black:
            # A shaded yellow plaque can satisfy both masks around its darkest
            # pixels.  Yellow is the non-contact hazard, so ambiguity is resolved
            # conservatively as yellow instead of risking a forbidden knock-down.
            detected = yellow
        else:
            detected = yellow or black
        return detected, yellow_mask, black_mask, (x1, y1, x2, y2)

    def _black_hough_candidate(
        self, roi: np.ndarray, offset: Tuple[int, int]
    ) -> Optional[Detection]:
        """Find a black plate whose mask merged into the dark wall/guide line.

        The black calibration photos show that threshold contours can join the
        circular plate to the horizontal background strip and its I-shaped
        support.  This fallback therefore asks for four independent cues:
        a circular edge, a dark interior, a brighter surrounding ring and a
        narrow dark support continuing below the circle.  It deliberately has
        no left/right-corridor restriction; the official sign may appear at
        any horizontal position inside the broad sign ROI.
        """
        cfg = self.sign_cfg["black_hough"]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.4)
        edges = cv2.Canny(blurred, 40, 100)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=float(cfg["min_center_distance_px"]),
            param1=float(cfg["param1"]),
            param2=float(cfg["param2"]),
            minRadius=int(cfg["min_radius_px"]),
            maxRadius=int(cfg["max_radius_px"]),
        )
        if circles is None:
            return None

        height, width = gray.shape
        yy, xx = np.ogrid[:height, :width]
        best: Optional[Detection] = None
        best_score = -math.inf
        for circle_x, circle_y, raw_radius in circles[0]:
            radius = float(raw_radius)
            margin = radius + float(self.sign_cfg.get("edge_margin_px", 0))
            if not (
                margin < circle_x < width - margin
                and margin < circle_y < height - margin
            ):
                continue

            radial = np.sqrt((xx - circle_x) ** 2 + (yy - circle_y) ** 2)
            inner = gray[radial <= 0.70 * radius]
            ring = gray[
                (radial >= 1.15 * radius) & (radial <= 1.65 * radius)
            ]
            rim = edges[
                (radial >= 0.85 * radius) & (radial <= 1.15 * radius)
            ]
            if inner.size == 0 or ring.size == 0 or rim.size == 0:
                continue
            inner_median = float(np.median(inner))
            ring_median = float(np.median(ring))
            contrast = ring_median - inner_median
            if inner_median > float(cfg["max_inner_median"]):
                continue
            if contrast < float(cfg["min_inner_ring_contrast"]):
                continue

            strip_half_width = max(2, int(round(0.18 * radius)))
            strip_x1 = max(0, int(round(circle_x)) - strip_half_width)
            strip_x2 = min(width, int(round(circle_x)) + strip_half_width + 1)
            strip_y1 = max(0, int(round(circle_y + 0.8 * radius)))
            strip_y2 = min(height, int(round(circle_y + 3.1 * radius)))
            stem = gray[strip_y1:strip_y2, strip_x1:strip_x2]
            if stem.size == 0:
                continue
            stem_rows = float(
                np.mean(np.any(stem < int(cfg["stem_gray_max"]), axis=1))
            )
            if stem_rows < float(cfg["min_stem_row_support"]):
                continue

            dark_fraction = float(np.mean(inner < int(cfg["max_inner_median"])))
            edge_fraction = float(np.mean(rim > 0))
            score = (
                2.2 * contrast
                + 50.0 * dark_fraction
                + 55.0 * edge_fraction
                + 120.0 * stem_rows
            )
            if score < float(cfg["min_score"]):
                continue

            # The lower semicircle is set against the light field and is much
            # less affected by the dark horizontal wall strip.  Its strongest
            # outward brightness jump gives a stable apparent plate radius.
            refined_radius = self._refine_black_radius(
                gray, float(circle_x), float(circle_y), radius
            )
            if refined_radius is None:
                continue
            center = (
                int(round(circle_x)) + offset[0],
                int(round(circle_y)) + offset[1],
            )
            confidence = clamp(0.55 + score / 1000.0, 0.0, 0.98)
            area = math.pi * refined_radius * refined_radius
            candidate = Detection(
                "black", center, refined_radius, area, confidence
            )
            # Hough can also propose a larger circle made from the wall edge
            # plus the real plate.  Prefer a proposal whose raw radius agrees
            # with the independent lower-arc measurement.
            selection_score = score - float(
                cfg["radius_refine_disagreement_penalty"]
            ) * abs(refined_radius - radius)
            if selection_score > best_score:
                best = candidate
                best_score = selection_score
        return best

    @staticmethod
    def _refine_black_radius(
        gray: np.ndarray, center_x: float, center_y: float, raw_radius: float
    ) -> Optional[float]:
        """Measure the black disk edge on lower side arcs, excluding the pole."""
        height, width = gray.shape
        start_radius = max(5, int(math.floor(raw_radius * 0.45)))
        end_radius = min(
            int(math.ceil(raw_radius * 1.55)),
            int(center_x) - 1,
            width - int(center_x) - 2,
            height - int(center_y) - 2,
        )
        if end_radius - start_radius < 4:
            return None
        angles = np.concatenate(
            (
                np.linspace(math.radians(15), math.radians(70), 70),
                np.linspace(math.radians(110), math.radians(165), 70),
            )
        )
        medians = []
        radii = list(range(start_radius, end_radius + 1))
        for radius in radii:
            xs = np.clip(
                np.rint(center_x + radius * np.cos(angles)).astype(int),
                0,
                width - 1,
            )
            ys = np.clip(
                np.rint(center_y + radius * np.sin(angles)).astype(int),
                0,
                height - 1,
            )
            medians.append(float(np.median(gray[ys, xs])))
        rises = [later - earlier for earlier, later in zip(medians, medians[1:])]
        valid = [
            index + 1
            for index, rise in enumerate(rises)
            if rise >= 18.0 and medians[index + 1] >= 45.0
        ]
        if not valid:
            return None
        boundary_index = max(valid, key=lambda index: rises[index - 1])
        return float(radii[boundary_index])

    def analyze(self, frame: np.ndarray, draw: bool = False) -> FrameAnalysis:
        frame = self._prepare_frame(frame)
        line_error, line_coverage, line_mask, line_rect = self._detect_line(frame)
        sign, yellow_mask, black_mask, sign_rect = self._detect_sign(frame)

        debug = None
        if draw:
            debug = frame.copy()
            lx1, ly1, lx2, ly2 = line_rect
            sx1, sy1, sx2, sy2 = sign_rect
            cv2.rectangle(debug, (lx1, ly1), (lx2, ly2), (255, 180, 0), 1)
            cv2.rectangle(debug, (sx1, sy1), (sx2, sy2), (180, 180, 180), 1)
            if line_error is not None:
                target_offset = float(self.line_cfg.get("line_target_offset", 0.0))
                observed_position = clamp(line_error + target_offset, -1.0, 1.0)
                line_x = lx1 + int(
                    round((lx2 - lx1) * (observed_position + 1.0) / 2.0)
                )
                cv2.line(debug, (line_x, ly1), (line_x, ly2), (255, 0, 0), 2)
                cv2.putText(
                    debug,
                    f"line error={line_error:+.2f}",
                    (5, debug.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    debug,
                    "LINE LOST",
                    (5, debug.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if sign:
                color = (0, 255, 255) if sign.label == "yellow" else (255, 255, 255)
                cv2.circle(
                    debug,
                    sign.center,
                    int(round(sign.radius)),
                    color,
                    2,
                )
                cv2.putText(
                    debug,
                    f"{sign.label} r={sign.radius:.1f}",
                    (max(0, sign.center[0] - 60), max(15, sign.center[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Small mask previews make on-site threshold tuning much faster.
            preview_w = 80
            preview_h = 55
            for index, mask in enumerate((yellow_mask, black_mask, line_mask)):
                preview = cv2.resize(mask, (preview_w, preview_h))
                preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                x0 = index * preview_w
                debug[0:preview_h, x0 : x0 + preview_w] = preview

        return FrameAnalysis(line_error, line_coverage, sign, debug)


class SignDistanceTracker:
    """Turn repeated sign observations into a physical 20 cm stop decision.

    Apparent radius is never compared with a magic trigger number.  It is first
    converted to centimetres through the shared pinhole model.  The same model
    is later reused by the ball controller with the ball's 28 mm diameter.
    """

    def __init__(
        self,
        sign_config: Dict[str, Any],
        distance_config: Dict[str, Any],
        range_model: PinholeRangeModel,
    ):
        self.confirm_frames = int(sign_config["confirm_frames"])
        self.min_confidence = float(
            sign_config.get("min_candidate_confidence", 0.0)
        )
        self.max_center_jitter = float(
            sign_config.get("max_center_jitter_px", math.inf)
        )
        self.max_radius_jitter = float(
            sign_config.get("max_radius_jitter_px", math.inf)
        )
        self.target_distance_cm = float(distance_config["target_distance_cm"])
        self.distance_tolerance_cm = float(
            distance_config["distance_tolerance_cm"]
        )
        self.distance_kp = float(distance_config["distance_kp"])
        self.min_forward = float(distance_config["min_forward_command"])
        self.max_forward = float(distance_config["max_forward_command"])
        self.max_reverse = float(distance_config["max_reverse_command"])
        self.maximum_valid_distance_cm = float(
            distance_config["maximum_valid_distance_cm"]
        )
        self.range_model = range_model
        self.history: Deque[Optional[Detection]] = deque(
            maxlen=self.confirm_frames
        )
        self.last_distance_cm: Optional[float] = None

    def distance_for(self, sign: Detection) -> float:
        """Estimate distance using the plate's full apparent diameter."""
        return self.range_model.sign_distance_cm(2.0 * float(sign.radius))

    def forward_command(self, distance_cm: float) -> float:
        """Proportional approach/reverse command recalculated after every frame."""
        error = float(distance_cm) - self.target_distance_cm
        if abs(error) <= self.distance_tolerance_cm:
            return 0.0
        raw = abs(error) * self.distance_kp
        if error > 0:
            return clamp(raw, self.min_forward, self.max_forward)
        return -clamp(raw, self.min_forward, self.max_reverse)

    def update(self, sign: Optional[Detection]) -> Optional[Detection]:
        """Return a sign only when colour, position and 20 cm range are stable."""
        if sign is not None and sign.confidence < self.min_confidence:
            sign = None
        self.history.append(sign)
        self.last_distance_cm = None
        if len(self.history) < self.confirm_frames or any(
            item is None for item in self.history
        ):
            return None

        detections = [item for item in self.history if item is not None]
        labels = {item.label for item in detections}
        if len(labels) != 1:
            return None
        center_x = sorted(item.center[0] for item in detections)[
            len(detections) // 2
        ]
        center_y = sorted(item.center[1] for item in detections)[
            len(detections) // 2
        ]
        if any(
            math.hypot(item.center[0] - center_x, item.center[1] - center_y)
            > self.max_center_jitter
            for item in detections
        ):
            return None
        radii = sorted(item.radius for item in detections)
        if radii[-1] - radii[0] > self.max_radius_jitter:
            return None

        median_radius = radii[len(radii) // 2]
        distance_cm = self.range_model.sign_distance_cm(2.0 * median_radius)
        if distance_cm > self.maximum_valid_distance_cm:
            return None
        self.last_distance_cm = distance_cm
        if (
            abs(distance_cm - self.target_distance_cm)
            > self.distance_tolerance_cm
        ):
            return None

        best = max(detections, key=lambda item: item.confidence)
        self.history.clear()
        return Detection(
            best.label,
            (center_x, center_y),
            median_radius,
            best.area,
            best.confidence,
        )


class TargetLossSafetyLock:
    """Arm lost-target stopping only after a credible near-sign acquisition.

    A single Hough candidate is not enough to prove that the robot is close to
    the sign.  Field run dog11_20260913-214512 produced sparse black candidates
    with strongly varying radii; latching on the first candidate made the robot
    stop after every short detection gap and prevented it from approaching.
    Requiring a short, geometrically consistent near-target streak preserves the
    stop-on-loss safety rule without letting an unconfirmed far candidate own it.
    """

    def __init__(
        self,
        sign_config: Dict[str, Any],
        distance_config: Dict[str, Any],
        range_model: PinholeRangeModel,
    ) -> None:
        self.confirm_frames = int(distance_config["target_lock_confirm_frames"])
        if self.confirm_frames < 2:
            raise ValueError("target_lock_confirm_frames must be at least 2")
        self.min_confidence = float(
            sign_config.get("min_candidate_confidence", 0.0)
        )
        self.max_center_jitter = float(
            sign_config.get("max_center_jitter_px", math.inf)
        )
        self.max_radius_jitter = float(
            sign_config.get("max_radius_jitter_px", math.inf)
        )
        self.maximum_lock_distance_cm = float(
            distance_config["target_distance_cm"]
        ) + float(distance_config["distance_tolerance_cm"])
        self.lost_stop_s = float(distance_config["target_lost_stop_s"])
        self.range_model = range_model
        self.candidates: Deque[Detection] = deque(maxlen=self.confirm_frames)
        self.engaged = False
        self.locked_label: Optional[str] = None
        self.last_seen: Optional[float] = None

    def update(self, sign: Optional[Detection], now: float) -> None:
        """Observe one frame and arm only on a stable near-target streak."""
        if sign is None or sign.confidence < self.min_confidence:
            self.candidates.clear()
            return

        if self.engaged:
            # Once safely acquired, keep the lock conservative: any credible
            # observation of that same physical sign refreshes the loss timer.
            if sign.label == self.locked_label:
                self.last_seen = now
            return

        distance_cm = self.range_model.sign_distance_cm(2.0 * sign.radius)
        if distance_cm > self.maximum_lock_distance_cm:
            self.candidates.clear()
            return

        self.candidates.append(sign)
        if len(self.candidates) < self.confirm_frames:
            return
        if len({item.label for item in self.candidates}) != 1:
            self.candidates.clear()
            return

        centers_x = sorted(item.center[0] for item in self.candidates)
        centers_y = sorted(item.center[1] for item in self.candidates)
        center_x = centers_x[len(centers_x) // 2]
        center_y = centers_y[len(centers_y) // 2]
        radii = sorted(item.radius for item in self.candidates)
        if any(
            math.hypot(item.center[0] - center_x, item.center[1] - center_y)
            > self.max_center_jitter
            for item in self.candidates
        ) or radii[-1] - radii[0] > self.max_radius_jitter:
            self.candidates.clear()
            return

        self.engaged = True
        self.locked_label = sign.label
        self.last_seen = now
        self.candidates.clear()

    def should_stop(self, now: float) -> bool:
        return (
            self.engaged
            and self.last_seen is not None
            and now - self.last_seen >= self.lost_stop_s
        )


class PIDLineFollower:
    """Keep the dog parallel to the guide line with speed reduced on larger errors.

    ``forward_speed`` is the normal straight-line command.  The September field
    test raised it from 9 to 12 while retaining the slow gait; this improves
    progress without changing the camera target or the left-of-line alignment.
    """

    def __init__(self, config: Dict[str, Any]):
        self.kp = float(config["kp"])
        self.ki = float(config["ki"])
        self.kd = float(config["kd"])
        self.max_turn = float(config["max_turn"])
        self.base_speed = float(config["forward_speed"])
        self.min_speed = float(config["min_forward_speed"])
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time: Optional[float] = None

    def reset(self) -> None:
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def command(self, error: float, now: float) -> Tuple[float, float]:
        if self.last_time is None:
            dt = 0.05
        else:
            dt = clamp(now - self.last_time, 0.01, 0.25)
        self.integral = clamp(self.integral + error * dt, -0.8, 0.8)
        derivative = (error - self.last_error) / dt
        # XGO positive yaw is left; an image target on the right needs right yaw.
        turn = -(self.kp * error + self.ki * self.integral + self.kd * derivative)
        turn = clamp(turn, -self.max_turn, self.max_turn)
        speed = max(self.min_speed, self.base_speed * (1.0 - 0.65 * abs(error)))
        self.last_error = error
        self.last_time = now
        return speed, turn


class BypassPlan:
    """Replay the yellow-sign detour every control frame instead of once.

    The XGO firmware only advances a few steps per ``move`` packet.  The former
    blocking ``_timed_move`` sent one packet and slept, so the dog stopped after
    a few steps and never cleared the sign (field report 2026-09-14: "有横移但
    没绕过去").  This plan keeps the same three phases (lateral out, forward
    past, lateral back) but is polled per frame so the command is re-issued and
    the per-frame stop/heartbeat safety checks keep running. 每个控制帧重发移动
    指令，避免走几步就停下，同时让停止/心跳检查继续生效。
    """

    def __init__(self, action_cfg: Dict[str, Any]) -> None:
        direction = str(action_cfg["bypass_direction"]).lower()
        sign = 1.0 if direction == "left" else -1.0
        lateral_speed = sign * float(action_cfg["bypass_lateral_speed"])
        self.phases: List[Tuple[str, float, float]] = [
            ("y", lateral_speed, float(action_cfg["bypass_lateral_s"])),
            (
                "x",
                float(action_cfg["bypass_forward_speed"]),
                float(action_cfg["bypass_forward_s"]),
            ),
            ("y", -lateral_speed, float(action_cfg["bypass_return_s"])),
        ]
        self.index = 0
        self.phase_start: Optional[float] = None

    @property
    def finished(self) -> bool:
        return self.index >= len(self.phases)

    def command(self, now: float) -> Optional[Tuple[str, float]]:
        """Return the current (axis, speed), advancing phases as time elapses."""
        if self.finished:
            return None
        if self.phase_start is None:
            self.phase_start = now
        axis, speed, duration = self.phases[self.index]
        while now - self.phase_start >= duration:
            self.index += 1
            self.phase_start += duration
            if self.finished:
                return None
            axis, speed, duration = self.phases[self.index]
        return axis, speed


class NullRobot:
    """Observe-mode robot: log commands without touching hardware."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.action_cfg = config["actions"]
        self.last_command: Optional[Tuple[Any, ...]] = None
        self.arm_wave_deadline: Optional[float] = None
        self.arm_wave_s = float(config["actions"]["arm_wave_s"])
        self.bypass_plan: Optional[BypassPlan] = None

    def follow(self, forward: float, turn: float) -> None:
        self.last_command = (forward, turn)

    def stop(self) -> None:
        self.last_command = (0.0, 0.0)

    def alarm(self) -> None:
        print("[observe] ALARM")

    def start_bypass(self) -> None:
        self.bypass_plan = BypassPlan(self.action_cfg)
        print("[observe] YELLOW -> bypass")

    def update_bypass(self, now: float) -> bool:
        assert self.bypass_plan is not None
        command = self.bypass_plan.command(now)
        if command is None:
            self.bypass_plan = None
            self.last_command = (0.0, 0.0)
            return True
        self.last_command = command
        return False

    def begin_knock_down(self) -> None:
        self.arm_wave_deadline = time.monotonic() + self.arm_wave_s
        print("[observe] BLACK -> follow line and wave arm for 5 seconds")

    def update_knock_down(self, now: float) -> bool:
        return self.arm_wave_deadline is not None and now >= self.arm_wave_deadline

    def close(self) -> None:
        self.stop()


class XGORobot:
    """Thin, safety-focused wrapper around the supplied XGO APIs."""

    def __init__(self, config: Dict[str, Any], model: str):
        try:
            from xgolib import XGO  # type: ignore
        except ImportError:
            try:
                from xgo_lib import XGO  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Cannot import xgolib. Install xgo-pythonlib or place xgo_lib.py "
                    "beside this script."
                ) from exc

        self.config = config
        self.action_cfg = config["actions"]
        self.dog = XGO(model)
        self.edu = None
        self.last_command: Optional[Tuple[Any, ...]] = None
        self.last_write = 0.0
        self.min_write_interval = float(config["motion"]["command_interval_s"])
        self.arm_wave_deadline: Optional[float] = None
        self.next_arm_command = 0.0
        self.arm_wave_high = True
        self.bypass_plan: Optional[BypassPlan] = None
        self.dog.pace(str(config["motion"]["pace"]))
        # 用户要求：每次启动后先收回机械臂，避免带着上次姿态出发。
        self.stow_arm()

        try:
            try:
                from xgoedu import XGOEDU  # type: ignore
            except ImportError:
                from xgo_edu import XGOEDU  # type: ignore
            self.edu = XGOEDU()
        except Exception as exc:
            print(f"[warning] XGOEDU unavailable; alarm will use terminal bell: {exc}")

    def stow_arm(self) -> None:
        """Bring the arm back to its stowed/initial pose.

        用户要求：启动后、抓球后、投放后都要收回机械臂。获准入口在启动与收尾
        调用本方法；完整自主程序执行抓球/投放后也应调用同一个方法。
        """
        try:
            self.dog.arm(
                float(self.action_cfg["arm_stow_x"]),
                float(self.action_cfg["arm_stow_z"]),
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"[warning] arm stow failed: {exc}")

    def follow(self, forward: float, turn: float) -> None:
        now = time.monotonic()
        command = (int(round(forward)), int(round(turn)))
        if command == self.last_command and now - self.last_write < 0.30:
            return
        if now - self.last_write < self.min_write_interval:
            return
        self.dog.move("x", command[0])
        self.dog.turn(command[1])
        self.last_command = command
        self.last_write = now

    def stop(self) -> None:
        self.dog.stop()
        self.last_command = (0, 0)
        self.last_write = time.monotonic()

    def alarm(self) -> None:
        self.stop()
        time.sleep(float(self.action_cfg["motion_pause_s"]))
        if self.edu is not None:
            try:
                self.edu.xgoSpeaker(str(self.action_cfg["alarm_file"]))
                time.sleep(float(self.action_cfg["motion_pause_s"]))
                return
            except Exception as exc:
                print(f"[warning] speaker failed: {exc}")
        print("\aALARM")
        time.sleep(float(self.action_cfg["motion_pause_s"]))

    def start_bypass(self) -> None:
        """Begin the non-blocking yellow detour; polled via update_bypass()."""
        self.bypass_plan = BypassPlan(self.action_cfg)

    def update_bypass(self, now: float) -> bool:
        """Re-issue the current detour command each frame; return True when done.

        The former implementation slept inside ``_timed_move`` after a single
        ``move`` packet, so the firmware stopped the dog after a few steps and
        the sign was never cleared.  Re-sending here mirrors ``follow`` and keeps
        the per-frame stop/heartbeat checks alive. 每帧重发，走完全程才结束。
        """
        assert self.bypass_plan is not None
        command = self.bypass_plan.command(now)
        if command is None:
            self.stop()
            self.bypass_plan = None
            return True
        axis, speed = command
        if command != self.last_command or now - self.last_write >= self.min_write_interval:
            self.dog.move(axis, speed)
            self.last_command = command
            self.last_write = now
        return False

    def begin_knock_down(self) -> None:
        """Start the non-blocking arm wave used for a black sign.

        The main loop keeps running the camera-based line follower.  Arm and
        locomotion commands therefore share one thread and one serial-command
        sequence, avoiding concurrent writes inside the supplied XGO library.
        """
        self.dog.claw(int(self.action_cfg["claw_open"]))
        now = time.monotonic()
        self.arm_wave_deadline = now + float(self.action_cfg["arm_wave_s"])
        self.next_arm_command = now
        self.arm_wave_high = True

    def update_knock_down(self, now: float) -> bool:
        """Send the next arm pose when due; return True after five seconds."""
        if self.arm_wave_deadline is None:
            raise RuntimeError("Arm wave was not started")
        if now >= self.arm_wave_deadline:
            self.stow_arm()
            self.arm_wave_deadline = None
            return True

        if now >= self.next_arm_command:
            arm_x = float(self.action_cfg["arm_x"])
            high_z = float(self.action_cfg["arm_high_z"])
            low_z = float(self.action_cfg["arm_low_z"])
            self.dog.arm(arm_x, high_z if self.arm_wave_high else low_z)
            self.arm_wave_high = not self.arm_wave_high
            interval = float(self.action_cfg["arm_half_period_s"])
            self.next_arm_command += interval
            if self.next_arm_command <= now:
                self.next_arm_command = now + interval
        return False

    def close(self) -> None:
        try:
            self.stop()
            self.stow_arm()
        except Exception as exc:
            print(f"[warning] shutdown command failed: {exc}")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "camera",
        "line",
        "sign",
        "distance_control",
        "course",
        "motion",
        "actions",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config sections missing: {sorted(missing)}")
    course = config["course"]
    if int(course["information_sign_count"]) != 1 or not bool(
        course["ignore_signs_after_first_action"]
    ):
        raise ValueError("正式规则要求单个随机黄/黑信息立牌，完成后不得二次触发")
    return config


def open_camera(config: Dict[str, Any], source: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(source)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["camera"]["width"]))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["camera"]["height"]))
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera index {source}")
    # Discard stale auto-exposure frames.
    for _ in range(5):
        camera.read()
    return camera


def describe(
    analysis: FrameAnalysis, range_model: PinholeRangeModel
) -> str:
    line = "lost" if analysis.line_error is None else f"{analysis.line_error:+.2f}"
    if analysis.sign is None:
        sign = "none"
    else:
        if range_model.calibrated:
            distance_text = (
                f" distance={range_model.sign_distance_cm(2.0 * analysis.sign.radius):.1f}cm"
            )
        else:
            distance_text = " distance=UNCALIBRATED"
        sign = (
            f"{analysis.sign.label} diameter={2.0 * analysis.sign.radius:.1f}px "
            f"conf={analysis.sign.confidence:.2f}{distance_text}"
        )
    return f"line={line} coverage={analysis.line_coverage:.3f} sign={sign}"


def run(args: argparse.Namespace) -> int:
    # Publish the real Python PID before camera/hardware initialization.  The
    # supervisor can now bind this exact run even when opening the camera fails
    # or takes unusually long.
    start = time.monotonic()
    if args.pid_file:
        pid_path = Path(args.pid_file)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()), encoding="ascii")
        print(f"PID {os.getpid()} written to {pid_path}")

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    geometry_path = config_path.parent / str(
        config["distance_control"]["geometry_config"]
    )
    range_model = PinholeRangeModel.from_config(
        load_geometry_config(geometry_path)
    )
    # Motor mode is unavailable until the XGO camera has a measured scale.
    # Observe mode remains useful and prints the raw apparent plate diameter.
    if args.mode == "run":
        range_model.require_calibrated()
    vision = VisionProcessor(config)
    follower = PIDLineFollower(config["motion"])
    tracker = SignDistanceTracker(
        config["sign"], config["distance_control"], range_model
    )
    target_loss_lock = TargetLossSafetyLock(
        config["sign"], config["distance_control"], range_model
    )
    robot: Any = NullRobot(config) if args.mode == "observe" else XGORobot(config, args.model)
    camera = open_camera(config, args.camera)
    video_path = Path(args.record_video).resolve() if args.record_video else None
    telemetry_path = (
        Path(args.telemetry).resolve()
        if args.telemetry
        else (video_path.with_suffix(".jsonl") if video_path is not None else None)
    )
    recorder = RunRecorder(
        video_path,
        telemetry_path,
        args.record_fps,
        (int(config["camera"]["width"]), int(config["camera"]["height"])),
    )
    last_print = 0.0
    line_lost_since: Optional[float] = None
    exit_code = 0
    run_outcome = "running"

    # During an action state, sign recognition cannot fire again.  BLACK_ADVANCE
    # keeps the visual line follower active while scheduling arm poses in this
    # same loop.  YELLOW_RECOVER ends only after stable line centering.
    black_advancing = False
    yellow_bypassing = False
    yellow_recovery = False
    yellow_recovery_started = 0.0
    yellow_centered_frames = 0
    yellow_last_error: Optional[float] = None
    yellow_line_acquired = False
    target_loss_stopped = False

    # --- OpenCode supervisor integration -------------------------------------
    # The launch side seeds the run-specific heartbeat and the supervisor starts
    # refreshing it before PID resolution.  Do not add start_delay here: after a
    # motor start delay, the very first control frame must verify a fresh
    # heartbeat instead of granting extra unmonitored driving time.
    heartbeat_grace_s = max(
        3.0, max(0.0, float(args.heartbeat_timeout_s)) + 1.0
    )

    if args.mode == "run" and args.start_delay > 0:
        print(f"Motors enabled in {args.start_delay:.1f} s. Press Ctrl+C to abort.")
        time.sleep(args.start_delay)

    calibration_state = (
        f"focal_length={range_model.focal_length_px:.2f}px"
        if range_model.calibrated and range_model.focal_length_px is not None
        else "UNCALIBRATED (observe only)"
    )
    print(
        f"mode={args.mode} target={tracker.target_distance_cm:.1f}cm "
        f"tolerance=±{tracker.distance_tolerance_cm:.1f}cm {calibration_state} "
        "(press q in the preview or Ctrl+C to stop)"
    )
    try:
        while args.max_seconds <= 0 or time.monotonic() - start < args.max_seconds:
            # Per-frame safety checks.  They MUST run on every control frame (not
            # only at some boundary) so an OpenCode "停止" stops the motors on the
            # very next frame.  They run before any motor command is issued.
            if args.stop_request and os.path.exists(args.stop_request):
                robot.stop()
                exit_code = 0
                run_outcome = "opencode_stop"
                print("STOP REQUESTED: stop.request detected; robot stopped safely.")
                break
            if args.heartbeat and float(args.heartbeat_timeout_s) > 0:
                if time.monotonic() - start >= heartbeat_grace_s:
                    try:
                        heartbeat_age = time.time() - os.path.getmtime(args.heartbeat)
                    except OSError:
                        heartbeat_age = None
                    if heartbeat_age is None or heartbeat_age > float(
                        args.heartbeat_timeout_s
                    ):
                        robot.stop()
                        exit_code = 3
                        run_outcome = "supervisor_heartbeat_lost"
                        print(
                            "SAFETY STOP: supervisor heartbeat missing or stale; "
                            "robot stopped."
                        )
                        break

            ok, frame = camera.read()
            if not ok or frame is None:
                robot.stop()
                print("[warning] empty camera frame")
                time.sleep(0.1)
                continue

            # The controller and recorder use the same prepared frame.  Never
            # start a second camera process alongside this program.
            prepared_frame = vision._prepare_frame(frame)
            recorder.write_frame(prepared_frame)
            analysis = vision.analyze(prepared_frame, draw=args.display)
            now = time.monotonic()
            if now - last_print >= float(args.print_interval):
                print(describe(analysis, range_model))
                last_print = now

            if black_advancing:
                if robot.update_knock_down(now):
                    robot.stop()
                    time.sleep(float(config["actions"]["motion_pause_s"]))
                    print(
                        "BLACK COMPLETE: followed the guide line while waving the "
                        "arm for five seconds; robot stopped and the test ended."
                    )
                    run_outcome = "goal_complete_black"
                    break

                if analysis.line_error is None:
                    if line_lost_since is None:
                        line_lost_since = now
                    if now - line_lost_since >= float(
                        config["motion"]["line_lost_stop_s"]
                    ):
                        robot.stop()
                else:
                    line_lost_since = None
                    forward, turn = follower.command(analysis.line_error, now)
                    forward = min(
                        forward, float(config["actions"]["black_follow_speed"])
                    )
                    robot.follow(forward, turn)
            elif yellow_bypassing:
                # The fixed detour deliberately moves away from the tracked
                # line. Do not let circular markings seen during that manoeuvre
                # become the temporal anchor for recovery. 绕行阶段每帧清空跟踪
                # 历史，防止圆弧被带入回线状态。
                vision.reset_line_tracking()
                # Non-blocking detour: the command is re-issued every frame so
                # the dog actually travels the whole lateral/forward/return path
                # (the old one-shot timed move stopped after a few steps). 逐帧
                # 重发移动指令，走完绕行路径再进入找线恢复。
                if robot.update_bypass(now):
                    yellow_bypassing = False
                    yellow_recovery = True
                    yellow_recovery_started = now
                    yellow_centered_frames = 0
                    follower.reset()
                    line_lost_since = None
                    print(
                        "YELLOW RECOVERY: bypass finished; looking for the guide "
                        "line and waiting for stable centering."
                    )
            elif yellow_recovery:
                recovery_cfg = config["actions"]
                recovery_age = now - yellow_recovery_started
                if recovery_age > float(recovery_cfg["yellow_reacquire_timeout_s"]):
                    robot.stop()
                    print(
                        "YELLOW FAILED: bypass finished but the guide line was not "
                        "reacquired before the safety timeout."
                    )
                    exit_code = 2
                    run_outcome = "yellow_reacquire_timeout"
                    break

                recovery_error, yellow_line_acquired = gate_initial_reacquire_error(
                    analysis.line_error, yellow_line_acquired, recovery_cfg
                )
                if analysis.line_error is not None and recovery_error is None:
                    # The far-edge candidate failed the initial gate. Forget it
                    # immediately so it cannot become the next-frame anchor.
                    # 未通过首次锁定门限的边缘候选不得写入跨帧记忆。
                    vision.reset_line_tracking()
                max_seek_turn = float(recovery_cfg.get("yellow_reacquire_max_turn", 20.0))
                if recovery_error is None:
                    # The guide line is out of view.  Rotate slowly toward the
                    # last known side to search instead of sitting still. 线不在
                    # 视野内时朝上次方向缓慢转向搜索。
                    yellow_centered_frames = 0
                    seek_turn = reacquire_search_turn(
                        yellow_last_error, recovery_cfg
                    )
                    robot.follow(
                        0.0, clamp(seek_turn, -max_seek_turn, max_seek_turn)
                    )
                else:
                    yellow_last_error = recovery_error
                    if abs(recovery_error) <= float(
                        recovery_cfg["yellow_center_error_max"]
                    ):
                        yellow_centered_frames += 1
                    else:
                        yellow_centered_frames = 0

                    required = int(recovery_cfg["yellow_center_frames"])
                    if yellow_centered_frames >= required:
                        robot.stop()
                        time.sleep(float(recovery_cfg["motion_pause_s"]))
                        print(
                            "YELLOW COMPLETE: bypassed the sign, returned to the "
                            "straight guide line, stopped, and ended the test."
                        )
                        run_outcome = "goal_complete_yellow"
                        break

                    follower_forward, follower_turn = follower.command(
                        recovery_error, now
                    )
                    forward, turn = limit_reacquire_command(
                        recovery_error,
                        follower_forward,
                        follower_turn,
                        recovery_cfg,
                    )
                    robot.follow(forward, turn)
            else:
                triggered = tracker.update(analysis.sign)
                target_loss_lock.update(analysis.sign, now)
                target_loss_stopped = False
                if triggered is not None:
                    robot.stop()
                    distance_cm = tracker.distance_for(triggered)
                    print(
                        f"TRIGGER {triggered.label.upper()} "
                        f"distance={distance_cm:.1f}cm "
                        f"diameter={2.0 * triggered.radius:.1f}px "
                        f"confidence={triggered.confidence:.2f}"
                    )
                    # Both colours share the measured 20 cm gate. The action is
                    # selected only after distance and colour are both stable.
                    robot.alarm()
                    if triggered.label == "yellow":
                        # Preserve the last trustworthy guide side before the
                        # open-loop detour. It supplies a bounded search direction
                        # if the returned line is initially out of view. 绕行前保存
                        # 可信方向，回线丢失时只按该方向低速搜索。
                        yellow_last_error = analysis.line_error
                        yellow_line_acquired = False
                        robot.start_bypass()
                        follower.reset()
                        line_lost_since = None
                        yellow_bypassing = True
                        print(
                            "YELLOW BYPASS: alarm done; circling the sign, then "
                            "reacquiring the guide line."
                        )
                    else:
                        robot.begin_knock_down()
                        follower.reset()
                        line_lost_since = None
                        black_advancing = True
                        print(
                            "BLACK ADVANCE: following the guide line while waving "
                            "the arm for five seconds."
                        )
                elif analysis.sign is not None and range_model.calibrated:
                    # Recalculate distance and command after every camera frame.
                    # No fixed movement duration is used to approach the prop.
                    distance_cm = tracker.distance_for(analysis.sign)
                    if distance_cm <= tracker.maximum_valid_distance_cm:
                        if analysis.line_error is None:
                            robot.stop()
                        else:
                            _, turn = follower.command(analysis.line_error, now)
                            forward = tracker.forward_command(distance_cm)
                            if forward == 0.0:
                                # Hold still while remaining frames confirm 20 cm.
                                robot.stop()
                            elif forward < 0:
                                # Reverse steering sign when backing away from an
                                # accidental too-close starting position.
                                robot.follow(forward, -turn)
                            else:
                                robot.follow(forward, turn)
                elif target_loss_lock.should_stop(now):
                    # Never advance blindly after a *confirmed near* prop is
                    # acquired. Sparse unconfirmed candidates do not arm this.
                    robot.stop()
                    target_loss_stopped = True
                elif analysis.line_error is not None:
                    line_lost_since = None
                    forward, turn = follower.command(analysis.line_error, now)
                    robot.follow(forward, turn)
                else:
                    if line_lost_since is None:
                        line_lost_since = now
                    if now - line_lost_since >= float(
                        config["motion"]["line_lost_stop_s"]
                    ):
                        robot.stop()

            if black_advancing:
                controller_state = "black_advance"
            elif yellow_bypassing:
                controller_state = "yellow_bypass"
            elif yellow_recovery:
                controller_state = "yellow_recovery"
            elif target_loss_stopped:
                controller_state = "target_lost_safe_stop"
            else:
                controller_state = "follow_or_approach"
            measured_distance = None
            if analysis.sign is not None and range_model.calibrated:
                measured_distance = range_model.sign_distance_cm(
                    2.0 * analysis.sign.radius
                )
            recorder.write_telemetry(
                {
                    "elapsed_s": round(now - start, 4),
                    "state": controller_state,
                    "line_error": analysis.line_error,
                    "line_coverage": analysis.line_coverage,
                    "sign_label": analysis.sign.label if analysis.sign else None,
                    "sign_confidence": (
                        analysis.sign.confidence if analysis.sign else None
                    ),
                    "sign_distance_cm": measured_distance,
                    "robot_command": getattr(robot, "last_command", None),
                    "target_loss_lock_engaged": target_loss_lock.engaged,
                    "yellow_line_acquired": yellow_line_acquired,
                }
            )

            if args.display and analysis.debug_frame is not None:
                cv2.imshow("XGO line and sign test", analysis.debug_frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    exit_code = 130
                    run_outcome = "manual_preview_stop"
                    break
        else:
            # Reaching the configured duration is an incomplete run, not a
            # successful program exit. OpenCode maps exit 124 to rating 2.
            if args.max_seconds > 0:
                robot.stop()
                exit_code = 124
                run_outcome = "maximum_duration_exceeded"
                print("RUN INCOMPLETE: maximum duration exceeded; robot stopped.")
    except KeyboardInterrupt:
        print("Interrupted; stopping the robot.")
        exit_code = 130
        run_outcome = "keyboard_interrupt"
    except Exception as exc:
        exit_code = 1
        run_outcome = "exception_%s" % type(exc).__name__
        print(f"RUN CRASHED: {type(exc).__name__}: {exc}")
        raise
    finally:
        # This final record lets OpenCode distinguish a visual goal-completion
        # stop from an operator/timeout stop. SIGKILL may skip it, which itself
        # is treated as rating 2 by the outer workflow.
        recorder.write_telemetry(
            {
                "event": "run_finished",
                "elapsed_s": round(time.monotonic() - start, 4),
                "outcome": run_outcome,
                "exit_code": exit_code,
            }
        )
        robot.close()
        camera.release()
        recorder.close()
        cv2.destroyAllWindows()
        if video_path is not None:
            print(f"VIDEO SAVED: {video_path} frames={recorder.frame_count}")
        if telemetry_path is not None:
            print(f"TELEMETRY SAVED: {telemetry_path}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Follow the black guide line and handle yellow/black signs."
    )
    parser.add_argument(
        "--mode",
        choices=("observe", "run"),
        default="observe",
        help="observe never creates an XGO motor object; run enables the robot",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="xgolite")
    parser.add_argument("--display", action="store_true")
    parser.add_argument(
        "--max-seconds", type=float, default=0.0, help="0 means run until stopped"
    )
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument("--print-interval", type=float, default=0.5)
    parser.add_argument(
        "--record-video",
        default="",
        help="save the exact control-camera stream; .mp4 uses mp4v, .avi uses MJPG",
    )
    parser.add_argument(
        "--telemetry",
        default="",
        help="optional JSONL path; defaults beside --record-video",
    )
    parser.add_argument("--record-fps", type=float, default=20.0)
    # OpenCode supervisor integration.  These are optional so offline tests and
    # observe runs are unaffected; when supplied they give OpenCode a way to
    # stop the loop immediately and a way to detect a dead supervisor.
    parser.add_argument(
        "--stop-request",
        default="",
        help="stop.flag path checked on every control frame; its presence means stop",
    )
    parser.add_argument(
        "--heartbeat",
        default="",
        help="local-supervisor heartbeat file; if its mtime is stale the robot stops",
    )
    parser.add_argument(
        "--heartbeat-timeout-s",
        type=float,
        default=2.0,
        help="seconds without a refreshed heartbeat before the safety stop fires",
    )
    parser.add_argument(
        "--pid-file",
        default="",
        help="write this process PID here so OpenCode can escalate to SIGINT",
    )
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
