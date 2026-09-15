# -*- coding: utf-8 -*-
"""公共颜色软距离模块。

用于国赛双链路颜色鲁棒化旁路验证：把 HSV 环形色距、严格种子
等基础运算集中在此处，避免复制到五个颜色函数中。

所有函数只接受已经读取好的 BGR/HSV 图像，禁止在内部调用 cap.read()。
"""
import cv2
import numpy as np


def hsv_distance(hsv, target_hsv, weights=(0.65, 0.25, 0.10), ignore_hue=False):
    """Return normalized float32 distance; lower means more similar."""
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    th, ts, tv = [float(x) for x in target_hsv]

    dh_raw = np.abs(h - th)
    dh = np.minimum(dh_raw, 180.0 - dh_raw) / 90.0
    ds = np.abs(s - ts) / 255.0
    dv = np.abs(v - tv) / 255.0

    if ignore_hue:
        hue_reliability = np.zeros_like(s, dtype=np.float32)
    else:
        # Low-saturation pixels have unstable Hue. S distance still rejects
        # unrelated white/gray pixels; these pixels must never become seeds.
        hue_reliability = np.clip(s / 50.0, 0.0, 1.0)

    wh, ws, wv = [float(x) for x in weights]
    return wh * dh * hue_reliability + ws * ds + wv * dv


def min_hsv_distance(hsv, hsv_samples, weights, ignore_hue=False):
    if not hsv_samples:
        raise ValueError("hsv_samples must not be empty")
    result = None
    for sample in hsv_samples:
        current = hsv_distance(hsv, sample, weights, ignore_hue=ignore_hue)
        result = current if result is None else np.minimum(result, current)
    return result


def normalized_roi_mask(shape, roi):
    h, w = shape[:2]
    x0, y0, x1, y1 = [float(x) for x in roi]
    x0 = max(0, min(w, int(round(x0 * w))))
    x1 = max(0, min(w, int(round(x1 * w))))
    y0 = max(0, min(h, int(round(y0 * h))))
    y1 = max(0, min(h, int(round(y1 * h))))
    mask = np.zeros((h, w), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def build_seed_mask(distance, seed_distance, roi_mask, seed_gate=None):
    seed = np.where(distance <= float(seed_distance), 255, 0).astype(np.uint8)
    seed = cv2.bitwise_and(seed, roi_mask)
    # seed_gate is a target-specific S/V validity mask.
    # For saturated colors this prevents low-S white highlights from becoming
    # seeds. Gray targets instead use an upper-S gate and ignore Hue.
    if seed_gate is not None:
        seed = cv2.bitwise_and(seed, seed_gate)

    # Only remove isolated pixels. Do not erode the final control boundary.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel)
    return seed


def retain_seeded_components(seed_mask, min_area=20,
                             max_area=None, min_seed_pixels=3):
    binary = (seed_mask > 0).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    # 已删除宽松生长阶段：每个连通域的每个像素都是种子，
    # 因此组件内种子数 == 面积。min_seed_pixels 仅作为遗留下限（与 min_area 等价）。
    seed_counts = np.bincount(labels[binary > 0].ravel(), minlength=count)
    keep_lookup = np.zeros(count, dtype=np.uint8)
    kept = []
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        if max_area is not None and area > int(max_area):
            continue
        seed_count = int(seed_counts[label_id])
        if seed_count < int(min_seed_pixels):
            continue
        keep_lookup[label_id] = 1
        kept.append({
            "label": label_id,
            "area": area,
            "seed_count": seed_count,
            "center": (float(centers[label_id][0]), float(centers[label_id][1])),
        })
    output = np.where(keep_lookup[labels] > 0, 255, 0).astype(np.uint8)
    return output, labels, kept


def fill_internal_holes(component_mask):
    """Fill only enclosed holes; do not globally accept white highlights."""
    src = np.where(component_mask > 0, 255, 0).astype(np.uint8)
    padded = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(src, holes)
