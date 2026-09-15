#!/usr/bin/python3
"""
calibrate_exposure.py — 比赛现场光线诊断 + 自动曝光参数推荐

用法：
    python3 calibrate_exposure.py          # 检测 /dev/video2（侧摄像头，循迹用）
    python3 calibrate_exposure.py 0        # 检测 /dev/video0（前摄像头）
    python3 calibrate_exposure.py 2 1      # 检测 /dev/video2，实时预览

输出：
    1. 亮度统计（全局 + 分区）
    2. 过曝/欠曝区域比例
    3. 推荐的目标亮度和 CLAHE 参数
    4. 可选：实时预览（按 q 退出）
"""

import cv2
import numpy as np
import sys
import time


def analyze_frame(frame, name="frame"):
    """分析单帧的光线情况，返回诊断字典"""
    if frame is None:
        return None

    h, w = frame.shape[:2]

    # 转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # === 全局统计 ===
    mean_brightness = float(np.mean(gray))
    median_brightness = float(np.median(gray))
    std_brightness = float(np.std(gray))
    min_brightness = float(np.min(gray))
    max_brightness = float(np.max(gray))

    # === 过曝/欠曝比例 ===
    overexposed_ratio = float(np.sum(gray > 230) / gray.size)  # 过曝（>230）
    underexposed_ratio = float(np.sum(gray < 30) / gray.size)   # 欠曝（<30）
    highlight_ratio = float(np.sum(gray > 200) / gray.size)     # 高亮区

    # === 分区亮度（上中下三等分） ===
    h3 = h // 3
    top_mean = float(np.mean(gray[0:h3, :]))       # 上部（天空/远处）
    mid_mean = float(np.mean(gray[h3:2*h3, :]))    # 中部
    bot_mean = float(np.mean(gray[2*h3:h, :]))     # 下部（路面）

    # === 左右分区 ===
    w2 = w // 2
    left_mean = float(np.mean(gray[:, 0:w2]))
    right_mean = float(np.mean(gray[:, w2:]))

    # === 直方图 ===
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    # 找出峰值亮度（histogram peak）
    peak_brightness = float(np.argmax(hist))

    return {
        "name": name,
        "size": (w, h),
        "mean": mean_brightness,
        "median": median_brightness,
        "std": std_brightness,
        "min": min_brightness,
        "max": max_brightness,
        "peak": peak_brightness,
        "overexposed_pct": overexposed_ratio * 100,
        "underexposed_pct": underexposed_ratio * 100,
        "highlight_pct": highlight_ratio * 100,
        "top_mean": top_mean,
        "mid_mean": mid_mean,
        "bot_mean": bot_mean,
        "left_mean": left_mean,
        "right_mean": right_mean,
    }


def recommend_params(stats, hw_available):
    """根据光线分析结果推荐自动曝光参数"""
    median = stats["median"]
    over = stats["overexposed_pct"]
    under = stats["underexposed_pct"]
    std = stats["std"]
    top_mean = stats["top_mean"]
    bot_mean = stats["bot_mean"]

    # === 判断光线场景 ===
    if over > 15:
        scene = "强光/过曝 — 赛道反光严重"
    elif over > 5:
        scene = "偏亮 — 有局部过曝"
    elif under > 15:
        scene = "偏暗 — 光线不足"
    elif under > 5:
        scene = "略暗 — 局部欠曝"
    elif std > 60:
        scene = "高对比度 — 明暗交替（树影/隧道口）"
    else:
        scene = "正常 — 光线均匀"

    # === 推荐 target_brightness ===
    # 目标：让中位数落在 100~130 之间
    if median > 150:
        target = max(80, median - 50)   # 太亮，压暗
    elif median > 130:
        target = 105
    elif median < 60:
        target = min(140, median + 50)  # 太暗，提亮
    elif median < 90:
        target = 115
    else:
        target = 110                     # 正常

    target = int(np.clip(target, 80, 140))

    # === 推荐 clahe_clip ===
    if std > 65:
        clahe_clip = 1.5   # 高对比度场景：低限幅，防噪点放大
    elif std > 45:
        clahe_clip = 2.0
    else:
        clahe_clip = 2.5   # 均匀场景：可以稍强

    # === 推荐 clahe_grid ===
    if stats["size"][0] >= 640:
        clahe_grid = (8, 8)
    else:
        clahe_grid = (4, 4)

    # === 推荐 roi ===
    # 如果上部（天空）比下部（路面）亮很多，ROI 排除上部
    if top_mean - bot_mean > 40:
        roi_top = 0.40
        roi_bot = 0.95
    elif top_mean - bot_mean > 20:
        roi_top = 0.25
        roi_bot = 0.95
    else:
        roi_top = 0.0
        roi_bot = 1.0

    # === 推荐 Kp ===
    if std > 60:
        Kp = 0.20  # 高对比度场景：慢响应，防振荡
    else:
        Kp = 0.30

    return {
        "scene": scene,
        "target_brightness": target,
        "clahe_clip": clahe_clip,
        "clahe_grid": clahe_grid,
        "roi_top_ratio": roi_top,
        "roi_bottom_ratio": roi_bot,
        "Kp": Kp,
    }


def print_report(stats, rec, hw_available):
    """打印诊断报告"""
    print("\n" + "=" * 62)
    print("   🏎️  比赛现场光线诊断报告")
    print("=" * 62)
    print(f"  摄像头: /dev/video{stats['name']}  分辨率: {stats['size'][0]}x{stats['size'][1]}")
    print(f"  硬件曝光: {'✅ 可用' if hw_available else '❌ 不可用（纯软件方案）'}")
    print("-" * 62)
    print(f"  场景判断: {rec['scene']}")
    print("-" * 62)
    print(f"  【亮度统计】")
    print(f"    均值: {stats['mean']:5.1f}   中位数: {stats['median']:5.1f}   峰值: {stats['peak']:5.1f}")
    print(f"    标准差: {stats['std']:5.1f}   最小: {stats['min']:5.1f}   最大: {stats['max']:5.1f}")
    print(f"    上部(天空): {stats['top_mean']:5.1f}   中部: {stats['mid_mean']:5.1f}   下部(路面): {stats['bot_mean']:5.1f}")
    print(f"    左半区: {stats['left_mean']:5.1f}   右半区: {stats['right_mean']:5.1f}")
    print(f"  【极端像素比例】")
    print(f"    过曝 (>230): {stats['overexposed_pct']:5.1f}%")
    print(f"    高亮 (>200): {stats['highlight_pct']:5.1f}%")
    print(f"    欠曝 (<30):  {stats['underexposed_pct']:5.1f}%")
    print("-" * 62)
    print(f"  📋 推荐参数（复制到 car_wrap_2026.py / alltasks.py）")
    print("-" * 62)
    print(f"    target_brightness = {rec['target_brightness']}")
    print(f"    roi_top_ratio     = {rec['roi_top_ratio']:.2f}")
    print(f"    roi_bottom_ratio  = {rec['roi_bottom_ratio']:.2f}")
    print(f"    clahe_clip        = {rec['clahe_clip']}")
    print(f"    clahe_grid        = {rec['clahe_grid']}")
    print(f"    Kp                = {rec['Kp']}")
    print("=" * 62)

    # 一键更新命令
    print(f"\n  📋 一键更新 car_wrap_2026.py 参数：")
    print(f"  -------------------------------------------------")
    print(f'  sed -i \'s/target_brightness=[0-9]*/target_brightness={rec["target_brightness"]}/\' car_wrap_2026.py')
    print(f'  sed -i \'s/roi_top_ratio=[0-9.]*/roi_top_ratio={rec["roi_top_ratio"]:.2f}/\' car_wrap_2026.py')
    print(f'  sed -i \'s/roi_bottom_ratio=[0-9.]*/roi_bottom_ratio={rec["roi_bottom_ratio"]:.2f}/\' car_wrap_2026.py')
    print(f'  sed -i \'s/clahe_clip=[0-9.]*/clahe_clip={rec["clahe_clip"]}/\' auto_exposure.py')
    print("  -------------------------------------------------\n")


def live_preview(device_id, ae_controller=None):
    """实时预览 + 每 30 帧打印一次统计"""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"❌ 无法打开 /dev/video{device_id}")
        return

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n🔴 实时预览中 — 按 q 退出，按 s 保存当前帧分析\n")
    frame_cnt = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        original = frame.copy()
        frame_cnt += 1

        # 如果有 AE 控制器就 apply
        if ae_controller:
            frame = ae_controller.apply(frame)

        # 每 30 帧打印统计
        if frame_cnt % 30 == 0:
            stats = analyze_frame(original, str(device_id))
            if stats:
                print(f"[frame {frame_cnt}] "
                      f"median={stats['median']:.0f} "
                      f"over={stats['overexposed_pct']:.1f}% "
                      f"under={stats['underexposed_pct']:.1f}% "
                      f"top={stats['top_mean']:.0f} bot={stats['bot_mean']:.0f}")

        # 拼接显示：左=原始，右=AE处理后
        if ae_controller:
            if original.shape != frame.shape:
                original = cv2.resize(original, (frame.shape[1], frame.shape[0]))
            display = np.hstack([original, frame])
            cv2.putText(display, "Original", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, "AE", (original.shape[1] + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display = original

        cv2.imshow(f"/dev/video{device_id} - 光线诊断", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            stats = analyze_frame(original, str(device_id))
            if stats:
                hw_ok = _check_hardware(device_id)
                rec = recommend_params(stats, hw_ok)
                print_report(stats, rec, hw_ok)

    cap.release()
    cv2.destroyAllWindows()


def _check_hardware(device_id):
    """检测硬件曝光是否可用"""
    import subprocess
    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', f'/dev/video{device_id}', '-C', 'exposure_absolute'],
            capture_output=True, text=True, timeout=2
        )
        return 'exposure_absolute' in result.stdout
    except Exception:
        return False


def main():
    device_id = 2  # 默认侧面摄像头

    live = False
    if len(sys.argv) >= 2:
        try:
            device_id = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) >= 3:
        live = (sys.argv[2] == '1' or sys.argv[2].lower() == 'live')

    print(f"\n📷 打开 /dev/video{device_id} ...")

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"❌ 无法打开 /dev/video{device_id}，尝试 /dev/video0 ...")
        device_id = 0
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 所有摄像头都打不开，请检查连接")
            sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 跳过前几帧（摄像头自动曝光稳定）
    print("⏳ 等待摄像头稳定（跳过前 15 帧）...")
    for i in range(15):
        cap.read()
        time.sleep(0.05)

    # 采集 10 帧做统计分析
    print("📊 采集 10 帧进行分析...")
    all_stats = []
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            stats = analyze_frame(frame, str(device_id))
            if stats:
                all_stats.append(stats)
        time.sleep(0.08)

    if not all_stats:
        print("❌ 未能获取有效帧")
        cap.release()
        sys.exit(1)

    # 取中位数统计（抗干扰）
    medians = sorted(s["median"] for s in all_stats)
    overs = sorted(s["overexposed_pct"] for s in all_stats)
    # 选最接近中位数的那一帧做详细报告
    mid_idx = len(all_stats) // 2
    # 按 median 排序后取中间帧
    sorted_stats = sorted(all_stats, key=lambda s: s["median"])
    representative = sorted_stats[len(sorted_stats) // 2]

    # 但用所有帧的统计做综合
    representative["median"] = medians[len(medians) // 2]
    representative["overexposed_pct"] = overs[len(overs) // 2]
    representative["top_mean"] = np.mean([s["top_mean"] for s in all_stats])
    representative["bot_mean"] = np.mean([s["bot_mean"] for s in all_stats])
    representative["std"] = np.mean([s["std"] for s in all_stats])
    representative["highlight_pct"] = np.mean([s["highlight_pct"] for s in all_stats])
    representative["underexposed_pct"] = np.mean([s["underexposed_pct"] for s in all_stats])
    representative["name"] = str(device_id)

    # 检测硬件曝光
    hw_available = _check_hardware(device_id)

    # 推荐参数
    rec = recommend_params(representative, hw_available)

    # 打印报告
    print_report(representative, rec, hw_available)

    cap.release()

    # 实时预览
    if live:
        print("启动实时预览...")
        # 用推荐参数临时建一个 AE 控制器预览
        from auto_exposure import AutoExposureController
        cap2 = cv2.VideoCapture(device_id)
        ae = AutoExposureController(
            cap=cap2,
            device_id=device_id,
            target_brightness=rec["target_brightness"],
            roi_top_ratio=rec["roi_top_ratio"],
            roi_bottom_ratio=rec["roi_bottom_ratio"],
            enable_hardware=hw_available,
            enable_clahe=True,
            clahe_clip=rec["clahe_clip"],
            clahe_grid=rec["clahe_grid"],
            debug=True,
        )
        live_preview(device_id, ae)
    else:
        print("💡 加参数 '1' 可开启实时预览：python3 calibrate_exposure.py 2 1")


if __name__ == "__main__":
    main()
