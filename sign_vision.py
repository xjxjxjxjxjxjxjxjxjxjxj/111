#!/usr/bin/env python3
"""Offline-testable yellow sign evidence detector.

This module expands the recognition library without changing any physical
distance threshold.  A frontal yellow disk may be reported as ``yellow``.
Black is deliberately not inferred from an arbitrary frame merely because
yellow is absent: a side-on yellow sign can also contain too few yellow pixels.
The senior course may use "not yellow means black" only after it has separately
reached the known sign checkpoint.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SignDetection:
    """One frontal yellow-sign observation in a 320 x 240 frame."""

    label: str
    x: float
    y: float
    radius: float
    area: float
    circularity: float
    confidence: float


def load_sign_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"camera", "yellow_sign_roi", "yellow_hsv_ranges", "geometry", "policy"}
    missing = required - set(config)
    if missing:
        raise ValueError("Sign config sections missing: %s" % sorted(missing))
    return config


class SignVision:
    """Detect frontal yellow disks and return unknown for insufficient evidence."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        width = int(self.config["camera"]["width"])
        height = int(self.config["camera"]["height"])
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return frame

    def _roi_bounds(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        left, top, right, bottom = (
            float(value) for value in self.config["yellow_sign_roi"]
        )
        return (
            max(0, min(width - 1, int(round(left * width)))),
            max(0, min(height - 1, int(round(top * height)))),
            max(1, min(width, int(round(right * width)))),
            max(1, min(height, int(round(bottom * height)))),
        )

    def yellow_mask(self, frame: np.ndarray) -> np.ndarray:
        prepared = self.prepare_frame(frame)
        blurred = cv2.GaussianBlur(prepared, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.config["yellow_hsv_ranges"]:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(
                    hsv,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                ),
            )

        x1, y1, x2, y2 = self._roi_bounds(prepared)
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

    def detect_yellow(self, frame: np.ndarray) -> Optional[SignDetection]:
        """Return a frontal yellow disk; edge-on and weak evidence return None."""
        prepared = self.prepare_frame(frame)
        mask = self.yellow_mask(prepared)
        cfg = self.config["geometry"]
        best: Optional[SignDetection] = None
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
            _, _, width, height = cv2.boundingRect(contour)
            aspect = width / max(float(height), 1.0)
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

            confidence = max(
                0.0,
                min(1.0, 0.12 + 0.50 * circularity + 0.38 * min(fill_ratio, 1.0)),
            )
            candidate = SignDetection(
                "yellow", circle_x, circle_y, float(radius), area, circularity, confidence
            )
            score = area * confidence
            if score > best_score:
                best = candidate
                best_score = score
        return best

    def classify(self, frame: np.ndarray) -> str:
        """Return ``yellow`` or ``unknown``; never guess black from one frame."""
        return "yellow" if self.detect_yellow(frame) is not None else "unknown"
