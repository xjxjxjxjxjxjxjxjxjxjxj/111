# -*- coding: utf-8 -*-
"""从已知目标区域采样真实 HSV，用于标定 config_car.yml 的 hsv_samples。

为什么需要它：
    新算法（软色距+种子生长）依赖 hsv_samples 描述目标的真实颜色。
    而 config_car.yml 里现在的 hsv_samples 是占位值，与场地里的浅蓝水块
    实际颜色（低饱和、高亮）对不上，导致新算法找到的是别的东西。

用法：
    python3 scripts/calibrate_hsv.py <图片路径> [颜色]

    python3 scripts/calibrate_hsv.py ./front.jpg light_blue
    python3 scripts/calibrate_hsv.py ./front.jpg yellow

说明：
    - 用「旧算法」的 HSV 掩膜（已知可用）定位目标的最大轮廓；
    - 只在轮廓内部统计 H/S/V 的中位数与 10/90 分位；
    - 输出可直接粘贴进 config_car.yml 该颜色 targets 下的 hsv_samples。
"""
from __future__ import print_function

import os
import sys

import cv2
import numpy as np


# 与 car_wrap_2026.py 里各 get_*_center 的旧 inRange 范围保持一致，
# 只用于「定位」目标，不参与新算法。
def _old_mask(hsv, color):
    if color == "light_blue":
        m1 = cv2.inRange(hsv, (95, 10, 50), (125, 255, 255))   # 正常浅蓝+阴影
        m2 = cv2.inRange(hsv, (95, 3, 140), (125, 20, 255))     # 过曝白斑
        mask = cv2.bitwise_or(m1, m2)
    elif color == "dark_blue":
        m1 = cv2.inRange(hsv, (100, 20, 35), (120, 255, 255))
        m2 = cv2.inRange(hsv, (100, 3, 180), (120, 25, 255))
        mask = cv2.bitwise_or(m1, m2)
    elif color == "yellow":
        mask = cv2.inRange(hsv, (22, 40, 70), (38, 255, 255))
    elif color == "red":
        m1 = cv2.inRange(hsv, (0, 30, 150), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 30, 150), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    elif color == "gray":
        mask = cv2.inRange(hsv, (0, 0, 25), (180, 35, 180))
    else:
        raise ValueError("未知颜色: {}".format(color))

    mask[: hsv.shape[0] // 6, :] = 0   # 同旧算法：上 1/6 置零
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_big)
    return mask


def _circular_mean_deg(h):
    rad = np.deg2rad(h * 2.0)
    m = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
    return (np.rad2deg(m) / 2.0) % 180.0


def main(argv):
    if len(argv) < 2:
        print("用法: python3 calibrate_hsv.py <图片路径> [颜色=light_blue]")
        return 1
    path = argv[1]
    color = argv[2] if len(argv) > 2 else "light_blue"
    if color not in ("light_blue", "dark_blue", "yellow", "red", "gray"):
        print("[ERROR] 未知颜色: {}".format(color))
        return 1

    image = cv2.imread(path)
    if image is None:
        print("[ERROR] 无法读取: {}".format(path))
        return 1

    blur = cv2.GaussianBlur(image, (9, 9), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = _old_mask(hsv, color)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[WARN] 旧算法在此图找不到目标，请换一张目标清晰的图")
        return 1
    contour = max(contours, key=cv2.contourArea)

    # 只在最大轮廓内部采样（避免把隔板等一并统计进来）
    h, w = hsv.shape[:2]
    cmask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(cmask, [contour], -1, 255, -1)
    xs = cmask > 0

    hh = hsv[:, :, 0][xs].astype(np.float32)
    ss = hsv[:, :, 1][xs].astype(np.float32)
    vv = hsv[:, :, 2][xs].astype(np.float32)
    n = int(hh.size)
    if n < 20:
        print("[WARN] 目标轮廓太小（{} px），请换一张更近/更清晰的图".format(n))
        return 1

    def pct(a, q):
        return float(np.percentile(a, q))

    mean_h = _circular_mean_deg(hh) if color == "red" else float(np.median(hh))
    s10, s50, s90 = pct(ss, 10), np.median(ss), pct(ss, 90)
    v10, v50, v90 = pct(vv, 10), np.median(vv), pct(vv, 90)

    print("颜色: {} | 采样像素数: {}".format(color, n))
    print("H: median={:.0f}  p10={:.0f}  p90={:.0f}".format(np.median(hh), pct(hh, 10), pct(hh, 90)))
    print("S: median={:.0f}  p10={:.0f}  p90={:.0f}".format(s50, s10, s90))
    print("V: median={:.0f}  p10={:.0f}  p90={:.0f}".format(v50, v10, v90))
    print("")
    print("建议 hsv_samples（粘贴到 config_car.yml 的 {} 下，替换占位值）:".format(color))
    print("  hsv_samples: [[{:.0f}, {:.0f}, {:.0f}], [{:.0f}, {:.0f}, {:.0f}], [{:.0f}, {:.0f}, {:.0f}]]".format(
        mean_h, s10, v50,
        mean_h, s50, v10,
        mean_h, s90, v90))
    print("")
    print("说明：三行分别覆盖 [暗/正常/亮] 三种曝光，S 用 10/50/90 分位。")
    print("如果目标本身饱和度很低（S<30），还需把 hue 信任度调高，见 car_wrap 里的 hue_reliability。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
