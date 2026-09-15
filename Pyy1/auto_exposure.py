"""
auto_exposure.py
适用于 OpenCV USB 摄像头的自动曝光控制器
- 优先尝试硬件曝光控制（V4L2）
- 兜底使用 CLAHE + Gamma 校正（纯软件, 不影响AI推理性能）
- 支持 ROI 加权测光（巡线场景重点关注路面区域）
"""

import cv2
import numpy as np
import subprocess
import time


class AutoExposureController:
    """
    自动曝光控制器

    使用方式：
        ae = AutoExposureController(cap_side.cap, device_id=2)
        # 在推理循环中：
        frame = ae.apply(frame)

    原理：
        1. 统计当前帧 ROI 区域的平均亮度
        2. 使用 PI 控制器平滑调整目标亮度
        3. 硬件方案：调整 V4L2 曝光参数
        4. 软件兜底：CLAHE + 简单 Gamma 映射
    """

    def __init__(
        self,
        cap,                    # cv2.VideoCapture 对象
        device_id=2,            # /dev/videoX 的设备号
        target_brightness=115,  # 目标亮度 (0-255)，115 偏保守，适合强光场景
        roi_top_ratio=0.55,     # ROI 上边界（画面比例），巡线只看下半部路面
        roi_bottom_ratio=0.95,  # ROI 下边界
        enable_hardware=True,   # 是否尝试硬件曝光控制
        enable_clahe=True,      # 是否启用 CLAHE（兜底方案）
        clahe_clip=2.0,         # CLAHE 对比度限幅
        clahe_grid=(8, 8),      # CLAHE 网格大小
        debug=False,            # 是否打印调试信息
    ):
        self.cap = cap
        self.device_id = device_id
        self.target_brightness = target_brightness
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_ratio = roi_bottom_ratio
        self.enable_hardware = enable_hardware
        self.enable_clahe = enable_clahe
        self.debug = debug

        # ---- CLAHE 初始化（在 LAB 的 L 通道上操作, 保持色彩不变） ----
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=clahe_grid
        ) if enable_clahe else None

        # ---- PI 控制器状态（用于平滑调整曝光） ----
        self._integral = 0.0
        self._last_brightness = None
        self._Kp = 0.3          # 比例系数
        self._Ki = 0.05         # 积分系数
        self._I_limit = 30      # 积分限幅

        # ---- 硬件曝光参数 ----
        self._hardware_available = False
        self._exposure_value = None   # 当前曝光值（摄像头单位）
        self._exposure_min = 3        # 最小曝光值
        self._exposure_max = 2047     # 最大曝光值
        self._frame_count = 0

        if enable_hardware:
            self._init_hardware()

    # ================================================================
    # 硬件曝光控制（V4L2）
    # ================================================================

    def _init_hardware(self):
        """初始化硬件曝光控制"""
        try:
            # 先关闭自动曝光，切换到手动模式
            subprocess.run(
                ['v4l2-ctl', '-d', f'/dev/video{self.device_id}',
                 '-c', 'exposure_auto=1'],
                capture_output=True, timeout=2
            )
            # 读取当前曝光值
            result = subprocess.run(
                ['v4l2-ctl', '-d', f'/dev/video{self.device_id}',
                 '-C', 'exposure_absolute'],
                capture_output=True, text=True, timeout=2
            )
            if 'exposure_absolute' in result.stdout:
                self._exposure_value = int(result.stdout.split(':')[1].strip())
                self._hardware_available = True
                if self.debug:
                    print(f"[曝光] 硬件曝光控制已就绪, 当前值={self._exposure_value}")
            else:
                if self.debug:
                    print("[曝光] 硬件不支持 exposure_absolute，将使用软件方案")
        except Exception as e:
            if self.debug:
                print(f"[曝光] 硬件初始化失败: {e}，将使用软件方案")

    def _set_hardware_exposure(self, value):
        """设置硬件曝光值"""
        value = int(np.clip(value, self._exposure_min, self._exposure_max))
        if value == self._exposure_value:
            return
        try:
            subprocess.run(
                ['v4l2-ctl', '-d', f'/dev/video{self.device_id}',
                 '-c', f'exposure_absolute={value}'],
                capture_output=True, timeout=1
            )
            self._exposure_value = value
        except Exception:
            pass

    # ================================================================
    # 亮度统计（ROI 加权）
    # ================================================================

    def _measure_brightness(self, frame):
        """
        统计 ROI 区域的亮度。

        巡线场景：画面下半部分是路面（最重要），上半部分是远处背景。
        因此 ROI 设为画面下 40%~95%。
        """
        h, w = frame.shape[:2]
        y1 = int(h * self.roi_top_ratio)
        y2 = int(h * self.roi_bottom_ratio)

        # 只取 ROI 区域
        roi = frame[y1:y2, :]

        # 转换到灰度
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # 使用中位数代替均值，抗干扰（个别过曝像素不影响整体判断）
        brightness = float(np.median(gray))
        return brightness

    # ================================================================
    # 软件端图像增强（CLAHE）
    # ================================================================

    def _apply_clahe(self, frame):
        """
        在 LAB 色彩空间的 L 通道上做 CLAHE，保持颜色不变。
        这对 AI 模型非常友好——增强了局部对比度，但不改变颜色分布。
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ================================================================
    # 主接口
    # ================================================================

    def apply(self, frame):
        """
        对输入帧进行自动曝光处理。

        参数:
            frame: BGR 图像 (numpy array)
        返回:
            处理后的 BGR 图像
        """
        if frame is None:
            return None

        self._frame_count += 1

        # ---- 第一步：测量亮度 ----
        brightness = self._measure_brightness(frame)
        error = self.target_brightness - brightness

        # ---- 第二步：PI 控制 ----
        # 积分分离：误差太大时清零积分，防止积分饱和
        if abs(error) > 60:
            self._integral = 0.0
        else:
            self._integral += self._Ki * error
            self._integral = np.clip(self._integral, -self._I_limit, self._I_limit)

        correction = self._Kp * error + self._integral

        # ---- 第三步：硬件调整（如果有） ----
        if self._hardware_available and self._frame_count % 8 == 0:
            # 每 8 帧调一次，给硬件足够响应时间
            if abs(error) > 8:   # 误差 < 8 不调整，避免抖动
                new_exp = self._exposure_value + int(correction * 3)
                self._set_hardware_exposure(new_exp)

        # ---- 第四步：软件 CLAHE 兜底 ----
        if self.enable_clahe:
            frame = self._apply_clahe(frame)

        # ---- 第五步：极端情况 Gamma 补偿 ----
        # 如果亮度偏离目标太多（硬件调不过来），做简单的 Gamma 校正
        if not self._hardware_available and abs(error) > 30:
            gamma = 1.0 + error / 200.0  # 偏暗→gamma<1(提亮), 偏亮→gamma>1(压暗)
            gamma = np.clip(gamma, 0.6, 1.5)
            lut = np.array([(i / 255.0) ** gamma * 255.0
                           for i in range(256)], dtype=np.uint8)
            # 只在亮度通道上做 Gamma
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.LUT(l, lut)
            lab = cv2.merge([l, a, b])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # ---- 调试输出 ----
        if self.debug and self._frame_count % 15 == 0:
            hw_tag = f"exp={self._exposure_value}" if self._hardware_available else "SW-only"
            print(f"[曝光] frame={self._frame_count} brightness={brightness:.0f} "
                  f"error={error:+.0f} corr={correction:+.1f} {hw_tag}")

        self._last_brightness = brightness
        return frame

    def reset(self):
        """重置 PI 控制器状态（例如巡线重新开始时调用）"""
        self._integral = 0.0
        self._last_brightness = None