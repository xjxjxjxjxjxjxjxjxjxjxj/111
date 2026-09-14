#!/usr/bin/env python3
"""OpenCV image I/O helpers that support Chinese paths on Windows."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def read_image(path: Path) -> Optional[np.ndarray]:
    """Read an image without relying on OpenCV's Windows path decoding."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
