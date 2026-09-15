# -*- coding: utf-8 -*-
"""颜色鲁棒化离线验证脚本（只读图片，不控制小车）。

用法：
    python3 scripts/test_color_robustness.py <图片文件夹或单张图片> [颜色]

    python3 scripts/test_color_robustness.py ./front_images
    python3 scripts/test_color_robustness.py ./front_images/a.jpg yellow

说明：
    - 只对已保存的图片做 old（现有 HSV 掩膜）与新（软色距种子生长）的中心对比；
    - 不打开机械臂、不写摄像头控件、不额外读相机；
    - 输出 CSV 汇总与并排调试图。
"""
from __future__ import print_function

import csv
import os
import sys
import time

import cv2
import numpy as np

# 允许从项目根目录导入公共颜色模块
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from smartcar.whalesbot.tools import color_region  # noqa: E402


# ---- 与 color_region 配套的种子门限（占位值，需用真实数据标定） ----
def _gates(hsv, color_name):
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    if color_name == "gray":
        seed = ((s <= 45) & (v >= 30) & (v <= 170)).astype(np.uint8) * 255
    else:
        seed = ((s >= 30) & (v >= 40)).astype(np.uint8) * 255
    return seed


# ---- 与 car_wrap_2026.py 一致的 seeded 中心（独立副本，供离线对比） ----
def seeded_center(image, color_name, target):
    h, w = image.shape[:2]
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    hsv_samples = target.get("hsv_samples") or []
    if not hsv_samples:
        return None, {"reason": "no_hsv_samples"}

    is_gray = (color_name == "gray")
    weights = (0.0, 0.55, 0.45) if is_gray else (0.65, 0.25, 0.10)
    distance = color_region.min_hsv_distance(hsv, hsv_samples, weights, ignore_hue=is_gray)

    seed_gate, grow_gate = _gates(hsv, color_name)
    roi = target.get("roi", [0.0, 0.0, 1.0, 1.0])
    roi_mask = color_region.normalized_roi_mask(hsv.shape, roi)

    seed, grow = color_region.build_seed_grow_masks(
        distance, 0.18, 0.36, roi_mask, seed_gate, grow_gate)
    component_mask, labels, kept = color_region.retain_seeded_components(
        seed, grow, min_area=80, max_area=int(0.45 * h * w), min_seed_pixels=8)

    if not kept:
        return None, {"reason": "no_component"}

    best = max(kept, key=lambda c: c["area"])
    single = np.where(labels == best["label"], 255, 0).astype(np.uint8)
    filled = color_region.fill_internal_holes(single)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"reason": "no_contour"}
    contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(contour)
    if m["m00"] <= 0:
        return None, {"reason": "empty_moments"}
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    shape = target.get("shape", "rectangle")
    if shape == "circle":
        (sx, sy), _r = cv2.minEnclosingCircle(contour)
    else:
        (sx, sy), _s, _a = cv2.minAreaRect(contour)
    gap = float(np.hypot(cx - sx, cy - sy))

    center = (float(cx), float(cy))
    norm = ((cx / w) * 2.0 - 1.0, (cy / h) * 2.0 - 1.0)
    quality = {"reason": "ok", "area": int(best["area"]),
               "seed_pixels": int(best["seed_count"]), "shape_gap_px": gap,
               "component_mask": filled}
    return norm, quality


# ---- 旧算法（与 get_*_center 一致的最大轮廓质心，作为基准） ----
OLD_RANGE = {
    "light_blue": ((95, 10, 50), (125, 255, 255)),
    "dark_blue": ((100, 20, 35), (120, 255, 255)),
    "yellow": ((22, 40, 70), (38, 255, 255)),
    "red": ((0, 30, 150), (180, 255, 255)),
    "gray": ((0, 0, 25), (180, 35, 180)),
}


def old_center(image, color_name):
    h, w = image.shape[:2]
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower, upper = OLD_RANGE[color_name]
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"reason": "no_contour"}
    largest = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m["m00"] < 50:
        return None, {"reason": "small_area"}
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return ((cx / w) * 2.0 - 1.0, (cy / h) * 2.0 - 1.0), {"reason": "ok"}


TARGETS = {
    "light_blue": {"hsv_samples": [[100, 31, 127], [100, 40, 150], [100, 62, 175]],
                   "roi": [0.0, 0.15, 1.0, 1.0], "shape": "rectangle"},
    "dark_blue": {"hsv_samples": [[105, 160, 90], [112, 190, 130], [118, 130, 170]],
                  "roi": [0.10, 0.20, 0.90, 0.80], "shape": "circle"},
    "yellow": {"hsv_samples": [[23, 160, 130], [30, 190, 180], [37, 150, 210]],
               "roi": [0.10, 0.20, 0.90, 0.80], "shape": "circle"},
    "red": {"hsv_samples": [[2, 190, 180], [178, 190, 180]],
            "roi": [0.0, 0.25, 1.0, 1.0], "shape": "circle"},
    "gray": {"hsv_samples": [[0, 15, 90], [0, 25, 130]],
             "roi": [0.0, 0.25, 1.0, 0.80], "shape": "rectangle"},
}


def collect_images(path):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            if f.lower().endswith(exts):
                out.append(os.path.join(root, f))
    return out


def main(argv):
    if len(argv) < 2:
        print("用法: python3 test_color_robustness.py <图片文件夹或单张图片> [颜色]")
        return 1

    path = argv[1]
    color_filter = argv[2] if len(argv) > 2 else None
    images = collect_images(path)
    if not images:
        print("[ERROR] 未找到图片: {}".format(path))
        return 1

    colors = [color_filter] if color_filter else list(TARGETS.keys())
    if color_filter and color_filter not in TARGETS:
        print("[ERROR] 未知颜色: {}，可选: {}".format(color_filter, list(TARGETS.keys())))
        return 1

    csv_path = os.path.join(_PROJECT, "scripts", "color_robustness_results.csv")
    debug_dir = os.path.join(_PROJECT, "scripts", "color_robustness_debug")
    if not os.path.isdir(debug_dir):
        os.makedirs(debug_dir)

    rows = []
    for img_path in images:
        image = cv2.imread(img_path)
        if image is None:
            print("[WARN] 无法读取: {}".format(img_path))
            continue
        for color in colors:
            t0 = time.time()
            old_r, old_q = old_center(image, color)
            old_ms = (time.time() - t0) * 1000.0

            t0 = time.time()
            new_r, new_q = seeded_center(image, color, TARGETS[color])
            new_ms = (time.time() - t0) * 1000.0

            o_ok = int(old_r is not None)
            n_ok = int(new_r is not None)
            o_x, o_y = old_r if old_r else (None, None)
            n_x, n_y = new_r if new_r else (None, None)

            diff_px = None
            if old_r and new_r:
                h, w = image.shape[:2]
                diff_px = ((n_x - o_x) * w / 2.0, (n_y - o_y) * h / 2.0)

            rows.append({
                "file": os.path.basename(img_path),
                "color": color,
                "old_ok": o_ok, "new_ok": n_ok,
                "old_x": o_x, "old_y": o_y,
                "new_x": n_x, "new_y": n_y,
                "diff_px": diff_px,
                "area": new_q.get("area"),
                "seed_pixels": new_q.get("seed_pixels"),
                "shape_gap_px": new_q.get("shape_gap_px"),
                "reason": new_q.get("reason"),
                "old_ms": round(old_ms, 3),
                "new_ms": round(new_ms, 3),
            })

            # 并排调试图：原图 | 旧 mask | 新 component
            debug = _make_debug_image(image, old_r, new_r, new_q, color)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_name = os.path.join(debug_dir, "{}__{}.jpg".format(base, color))
            cv2.imwrite(out_name, debug)

    _write_csv(csv_path, rows)
    print("[DONE] 共处理 {} 张图，结果写入 {}".format(len(rows), csv_path))
    return 0


def _make_debug_image(image, old_r, new_r, new_q, color):
    h, w = image.shape[:2]
    new_mask = new_q.get("component_mask") if isinstance(new_q, dict) else None
    new_mask_bgr = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR) if new_mask is not None else np.zeros_like(image)

    def _mark(img, r):
        if r is None:
            return img
        cx = int((r[0] + 1) / 2 * w)
        cy = int((r[1] + 1) / 2 * h)
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
        return img

    annotated = _mark(image.copy(), old_r)
    annotated = _mark(annotated, new_r)
    cv2.putText(annotated, color, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return np.hstack([annotated, new_mask_bgr])


def _write_csv(csv_path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
