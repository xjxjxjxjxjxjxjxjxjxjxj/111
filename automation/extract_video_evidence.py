#!/usr/bin/env python3
"""Extract evenly spaced frames and a contact sheet for Codex video review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def extract(video: Path, output: Path, maximum_frames: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("Cannot open video: %s" % video)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Invalid video metadata for %s" % video)

    sample_count = min(maximum_frames, frame_count)
    indices = sorted(set(np.linspace(0, frame_count - 1, sample_count, dtype=int)))
    frames: List[Tuple[int, Path, np.ndarray]] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        timestamp = index / fps
        path = output / ("frame_%06d_%08.3fs.jpg" % (index, timestamp))
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError("Cannot write extracted frame: %s" % path)
        frames.append((index, path, frame))
    capture.release()
    if not frames:
        raise RuntimeError("No frames could be extracted from %s" % video)

    thumb_w = 320
    thumb_h = max(1, int(round(thumb_w * height / width)))
    columns = 4
    label_h = 28
    rows = int(math.ceil(len(frames) / columns))
    sheet = np.full((rows * (thumb_h + label_h), columns * thumb_w, 3), 235, np.uint8)
    for item_index, (frame_index, _, frame) in enumerate(frames):
        row, column = divmod(item_index, columns)
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        y = row * (thumb_h + label_h)
        x = column * thumb_w
        sheet[y : y + thumb_h, x : x + thumb_w] = thumb
        cv2.putText(
            sheet,
            "frame %d  %.2fs" % (frame_index, frame_index / fps),
            (x + 7, y + thumb_h + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    contact_sheet = output / "contact_sheet.jpg"
    if not cv2.imwrite(str(contact_sheet), sheet):
        raise RuntimeError("Cannot write contact sheet")

    manifest = {
        "video": str(video.resolve()),
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": frame_count / fps,
        "resolution": [width, height],
        "contact_sheet": str(contact_sheet.resolve()),
        "sample_frames": [str(path.resolve()) for _, path, _ in frames],
    }
    (output / "video_evidence.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-frames", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(extract(args.video, args.output, args.maximum_frames), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
