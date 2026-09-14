#!/usr/bin/env python3
"""Write the control camera stream and synchronized JSONL telemetry.

The control program already owns the camera.  Recording those same frames here
avoids opening ``/dev/video0`` a second time, which would otherwise make either
the robot controller or a separate recorder fail on many Raspberry Pi cameras.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class RunRecorder:
    """Optional video/telemetry sink with explicit close and flush behavior."""

    def __init__(
        self,
        video_path: Optional[Path],
        telemetry_path: Optional[Path],
        fps: float,
        frame_size: Tuple[int, int],
    ) -> None:
        self.video_path = video_path
        self.telemetry_path = telemetry_path
        self.fps = float(fps)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.writer: Optional[cv2.VideoWriter] = None
        self.telemetry_stream = None
        self.frame_count = 0

        if self.fps <= 0:
            raise ValueError("Recording FPS must be positive")
        if telemetry_path is not None:
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            self.telemetry_stream = telemetry_path.open("w", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self.video_path is not None or self.telemetry_path is not None

    def _open_writer(self) -> None:
        if self.video_path is None or self.writer is not None:
            return
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self.video_path.suffix.lower()
        fourcc_text = "MJPG" if suffix == ".avi" else "mp4v"
        writer = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*fourcc_text),
            self.fps,
            self.frame_size,
        )
        if not writer.isOpened():
            raise RuntimeError(
                "Cannot open video writer for %s; try an .avi path if MP4 is unavailable"
                % self.video_path
            )
        self.writer = writer

    def write_frame(self, frame: np.ndarray) -> None:
        if self.video_path is None:
            return
        self._open_writer()
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        assert self.writer is not None
        self.writer.write(frame)
        self.frame_count += 1

    def write_telemetry(self, record: Dict[str, Any]) -> None:
        if self.telemetry_stream is None:
            return
        self.telemetry_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.frame_count % 20 == 0:
            self.telemetry_stream.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.telemetry_stream is not None:
            self.telemetry_stream.flush()
            self.telemetry_stream.close()
            self.telemetry_stream = None

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
