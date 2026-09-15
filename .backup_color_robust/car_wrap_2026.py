#!/usr/bin/python
# -*- coding: utf-8 -*-
from urllib import response
import base64
import psutil
from typing import Union
import time
import threading
import os
import platform
import signal
from smartcar import Camera, Streamer
import numpy as np

from smartcar.whalesbot.vehicle import (
    ArmController,
    ScreenShow,
    Key4Btn,
    Infrared,
    LedLight,
    MecanumDriver,
    Beep,
    BluetoothPad,
    ServoPwm,
)
from smartcar import PID
import difflib
import cv2
import math
from smartcar.paddlebaidu.infer_cs import ClintInterface, Bbox
from smartcar.paddlebaidu.ernie_bot import (
    ErnieBotWrap,
    ActionPrompt,
    HumAttrPrompt,
    ImagePrompt,
    OrderPrompt,
)
from smartcar.whalesbot.tools import CountRecord, get_yaml, IndexWrap
from smartcar.whalesbot.tools import color_region
import sys
from typing import List
import re

from smartcar.whalesbot.vehicle.base.controller_wrap import PoutD

# 添加上本地目录
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from smartcar import logger
from auto_exposure import AutoExposureController


def filter_chinese_letter(text: str) -> str:
    # 正则：汉字 \u4e00-\u9fff + 大小写字母 a-zA-Z
    res = re.findall(r"[\u4e00-\u9fffa-zA-Z]", text)
    return "".join(res)


def sellect_program(programs, order, win_order):
    """
    选择程序并生成显示字符串

    该函数用于生成程序选择菜单的显示字符串，突出显示当前选中的程序。

    参数:
        programs: 程序列表，包含所有可选择的程序
        order: 当前选中的程序索引
        win_order: 窗口起始索引

    返回:
        str: 生成的显示字符串，包含程序列表和当前选中的程序标记
    """
    dis_str = ""
    start_index = 0

    start_index = order - win_order
    for i, program in enumerate(programs):
        if i < start_index:
            continue

        now = str(program)
        if i == order:
            now = f">>{i + 1}.{now}"
        else:
            now = f"  {i + 1}.{now}"
        if len(now) >= 19:
            now = now[:19]
        else:
            now = now + "\n"
        dis_str += now
        if i - start_index == 4:
            break
    return dis_str


def kill_other_python():
    """
    终止其他Python进程

    该函数用于终止除当前进程外的其他Python进程，以避免进程冲突。

    注意:
        该函数会强制终止其他Python进程，请谨慎使用。
    """

    pid_me = os.getpid()
    # logger.info("my pid ", pid_me, type(pid_me))
    python_processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (
                "python" in proc.info["name"].lower()
                and len(proc.info["cmdline"]) > 1
                and len(proc.info["cmdline"][1]) < 30
            ):
                python_processes.append(proc.info)
        # 出现异常的时候捕获 不存在的异常，权限不足的异常， 僵尸进程
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    for process in python_processes:
        # logger.info(f"PID: {process['pid']}, Name: {process['name']}, Cmdline: {process['cmdline']}")
        # logger.info("this", process['pid'], type(process['pid']))
        if int(process["pid"]) != pid_me:
            os.kill(int(process["pid"]), signal.SIGKILL)
            time.sleep(0.3)


def limit(value, value_range):
    """
    限制值在指定范围内

    该函数用于将输入值限制在[-value_range, value_range]范围内。

    参数:
        value: 输入值
        value_range: 范围上限

    返回:
        float: 限制后的值
    """
    return max(min(value, value_range), 0 - value_range)


def angle_error_map(error, dead_zone=0.015, boost=2.5, soft_thresh=0.12, compress=0.55):
    """
    角度误差非线性能量映射：小误差放大，大误差压缩

    - 直道：角度误差很小（车身基本平行于赛道），放大后角度PID会更积极地修正微小偏移
    - 弯道：角度误差大，开方压缩防止PID输出过冲，避免提前拐弯

    参数:
        error:        原始角度误差
        dead_zone:    死区，|e|小于此值输出0，避免直道抖动
        boost:        小误差放大倍数（>1）
        soft_thresh:  分界阈值，小于此值为"小误差"，大于为"大误差"
        compress:     大误差压缩系数（越小压缩越狠，0.5~0.7推荐）
    """
    abs_e = abs(error)
    sign = 1 if error > 0 else -1

    if abs_e < dead_zone:
        return 0.0

    if abs_e < soft_thresh:
        # 小误差区间：线性放大
        return sign * (abs_e - dead_zone) * boost
    else:
        # 大误差区间：开方压缩
        small_part = (soft_thresh - dead_zone) * boost
        return sign * (small_part + math.sqrt(abs_e - soft_thresh) * compress)


# 两个pid集合成一个
class PidCal2:
    """
    PID控制器集合类

    包含三个PID控制器：
    - pid_y:        横向误差 → 横向速度（Mecanum 平移，通常不用）
    - pid_angle:     角度误差 → 角速度（主要方案：AI角度 + CV横偏的组合误差）
    - pid_lat_angle: 横向误差 → 角速度（备用方案：纯CV横偏，不依赖AI角度）
    """

    def __init__(self, cfg_pid_y, cfg_pid_angle, cfg_pid_lat_angle=None):
        """
        初始化PID控制器集合

        参数:
            cfg_pid_y:         y轴PID控制器的配置参数
            cfg_pid_angle:     角度PID控制器的配置参数
            cfg_pid_lat_angle: 纯横向误差→角速度的PID参数（可选，默认同cfg_pid_angle）
        """
        self.pid_y = PID(**cfg_pid_y)
        self.pid_angle = PID(**cfg_pid_angle)
        if cfg_pid_lat_angle is not None:
            self.pid_lat_angle = PID(**cfg_pid_lat_angle)
        else:
            self.pid_lat_angle = PID(**cfg_pid_angle)  # fallback

    def get_out(self, error_y, error_angle):
        """
        计算PID输出

        参数:
            error_y: y轴误差
            error_angle: 角度误差

        返回:
            tuple: (y轴PID输出, 角度PID输出)
        """
        pid_y_out = self.pid_y(error_y)
        pid_angle_out = self.pid_angle(error_angle)
        return pid_y_out, pid_angle_out


class LanePidCal:
    """
    车道PID控制器类

    该类用于车道保持控制，包含y轴和角度PID控制器。
    """

    def __init__(self, cfg_pid_y, cfg_pid_angle):
        """
        初始化车道PID控制器

        参数:
            cfg_pid_y: y轴PID控制器的配置参数
            cfg_pid_angle: 角度PID控制器的配置参数
        """
        # y_out_limit = 0.7
        # self.pid_y = PID(5, 0, 0)
        # self.pid_y.setpoint = 0
        # self.pid_y.output_limits = (-y_out_limit, y_out_limit)
        # print(cfg_pid_y)
        # print(cfg_pid_angle)
        self.pid_y = PID(**cfg_pid_y)
        # print(self.pid_y)

        angle_out_limit = 1.5
        self.pid_angle = PID(3, 0, 0)
        self.pid_angle.setpoint = 0
        self.pid_angle.output_limits = (-angle_out_limit, angle_out_limit)

    def get_out(self, error_y, error_angle):
        """
        计算PID输出

        参数:
            error_y: y轴误差
            error_angle: 角度误差

        返回:
            tuple: (y轴PID输出, 角度PID输出)
        """
        pid_y_out = self.pid_y(error_y)
        pid_angle_out = self.pid_angle(error_angle)
        return pid_y_out, pid_angle_out


class DetPidCal:
    """
    检测PID控制器类

    该类用于目标检测控制，包含y轴和角度PID控制器。
    """

    def __init__(self, cfg_pid_y=None, cfg_pid_angle=None):
        """
        初始化检测PID控制器

        参数:
            cfg_pid_y: y轴PID控制器的配置参数（可选）
            cfg_pid_angle: 角度PID控制器的配置参数（可选）
        """
        y_out_limit = 0.7
        self.pid_y = PID(0.3, 0, 0)
        self.pid_y.setpoint = 0
        self.pid_y.output_limits = (-y_out_limit, y_out_limit)

        angle_out_limit = 1.5
        self.pid_angle = PID(2, 0, 0)
        self.pid_angle.setpoint = 0
        self.pid_angle.output_limits = (-angle_out_limit, angle_out_limit)

    def get_out(self, error_y, error_angle):
        """
        计算PID输出

        参数:
            error_y: y轴误差
            error_angle: 角度误差

        返回:
            tuple: (y轴PID输出, 角度PID输出)
        """
        pid_y_out = self.pid_y(error_y)
        pid_angle_out = self.pid_angle(error_angle)
        return pid_y_out, pid_angle_out


class LocatePidCal:
    """
    定位PID控制器类

    该类用于位置定位控制，包含x轴和y轴PID控制器。
    """

    def __init__(self):
        """
        初始化定位PID控制器

        初始化x轴和y轴的PID控制器，设置默认参数和输出限制。
        """
        y_out_limit = 0.3
        self.pid_y = PID(0.5, 0, 0)
        self.pid_y.setpoint = 0
        self.pid_y.output_limits = (-y_out_limit, y_out_limit)

        x_out_limit = 0.3
        self.pid_x = PID(0.5, 0, 0)
        self.pid_x.setpoint = 0
        self.pid_x.output_limits = (-x_out_limit, x_out_limit)

    def set_target(self, x, y):
        """
        设置目标位置

        参数:
            x: x轴目标位置
            y: y轴目标位置
        """
        self.pid_y.setpoint = y
        self.pid_x.setpoint = x

    def get_out(self, error_x, error_y):
        """
        计算PID输出

        参数:
            error_x: x轴误差
            error_y: y轴误差

        返回:
            tuple: (x轴PID输出, y轴PID输出)
        """
        pid_y_out = self.pid_y(error_y)
        pid_x_out = self.pid_x(error_x)
        return pid_x_out, pid_y_out


class MyCar(MecanumDriver):
    """
    智能车控制类

    该类继承自MecanumDriver，实现了智能车的完整控制功能，包括传感器初始化、PID控制、摄像头控制、
    目标检测、车道保持等功能。
    """

    STOP_PARAM: bool = True

    def __init__(self):
        """
        初始化智能车

        初始化智能车的各个组件，包括底盘、传感器、摄像头、PID控制器等。
        """
        # 调用继承的初始化
        start_time = time.time()
        super(MyCar, self).__init__()
        logger.info("my car init ok {}".format(time.time() - start_time))
        # 显示
        self.display = ScreenShow()

        self.streamer = Streamer()
        self.arm = ArmController()

        # 获取自己文件所在的目录路径
        self.path_dir = os.path.abspath(os.path.dirname(__file__))
        self.yaml_path = os.path.join(self.path_dir, "config_car.yml")
        # 获取配置
        cfg = get_yaml(self.yaml_path)
        perf_cfg = cfg.get("performance", {})
        self.performance_cfg = perf_cfg if isinstance(perf_cfg, dict) else {}
        self._save_lane_debug_images = bool(
            self.performance_cfg.get("save_lane_debug_images", 0)
        )
        self._save_lane_photo = bool(
            self.performance_cfg.get("save_lane_photo", 0)
        )
        self._lane_photo_interval = max(1, int(
            self.performance_cfg.get("lane_photo_interval", 5)
        ))
        self._disable_left_turn = int(
            self.performance_cfg.get("disable_left_turn", 0)
        )
        self._darken_enabled = False  # 暗化开关，默认关闭

        # ====== 颜色鲁棒化旁路验证配置（默认 mode=0，保持旧行为） ======
        _vision_cfg = cfg.get("vision_color_robust", {})
        _vision_cfg = _vision_cfg if isinstance(_vision_cfg, dict) else {}
        _object_cfg = _vision_cfg.get("object", {})
        self._vision_color_object_cfg = _object_cfg if isinstance(_object_cfg, dict) else {}
        self._vision_color_object_mode = int(self._vision_color_object_cfg.get("mode", 0))
        self._vision_color_object_save_debug = bool(
            self._vision_color_object_cfg.get("save_debug", 0)
        )
        # 最近一次 seeded 中心（按颜色），用于上一帧中心约束
        self._last_seeded_center_by_color = {}
        # 颜色 A/B 性能统计
        self._last_color_ab_perf = {}
        # seeded 中间结果（仅在 save_debug=1 时暂存，供并排图保存）
        self._last_seeded_debug = None
        self._color_debug_save_counter = 0

        print("[初始化] 读取配置完成")
        # 根据配置设置sensor
        self.sensor_init(cfg)
        print("[初始化] 传感器初始化完成")

        self.car_pid_init(cfg)
        print("[初始化] PID 控制器初始化完成")
        self.ring = Beep()
        self.camera_init(cfg)
        print("[初始化] 摄像头初始化完成")

        # ====== 自动曝光控制器（强光场景，现场校准） ======
        self.ae_controller = AutoExposureController(
            cap=self.cap_side.cap,
            device_id=cfg["camera"]["side"],
            target_brightness=106,           # 现场校准值（中位数155，路面144）
            roi_top_ratio=0.40,              # 排除上部天空（亮度191），不影响模型输入
            roi_bottom_ratio=0.95,           # 路面区域（亮度148）
            enable_hardware=True,            # 尝试硬件控制
            enable_clahe=True,               # CLAHE 兜底
            clahe_clip=2.5,                  # 现场校准值
            debug=False,                     # 比赛时关闭调试
        )

        # paddle推理初始化
        self.paddle_infer_init()
        print("[初始化] Paddle 推理初始化完成")
        # 文心一言分析初始化
        self.ernie_bot_init()
        print("[初始化] 文心一言初始化完成")

        # 相关临时变量设置
        # 程序结束标志
        self._stop_flag = False
        # 按键线程结束标志
        self._end_flag = False
        self.thread_key = threading.Thread(target=self.key_thread_func)
        self.thread_key.daemon = True
        self.thread_key.start()
        print("[初始化] 按键线程已启动")

        self.beep()
        print("[初始化] MyCar 初始化流程结束")

    def beep(self):
        """
        发出蜂鸣音

        控制蜂鸣器发出一声蜂鸣音，并等待0.2秒。
        """
        self.ring.rings()
        time.sleep(0.2)

    def disable_left_turn_key(self, val):
        """
        动态开关：禁止左拐/右拐

        val=0 → 正常双向巡线
        val=1 → 禁止左拐（angle>0 截断为 -0.0006）
        val=2 → 禁止右拐（angle<0 截断为 +0.0006）

        用法：
            my_car.disable_left_turn_key(1)   # 禁止左拐
            my_car.lane_dis_offset(speed=0.2, dis_hold=3.5, mode=4)
            my_car.disable_left_turn_key(0)   # 恢复
        """
        self._disable_left_turn = int(val)

    def darken_key(self, val):
        """
        动态开关：图像暗化（Gamma 校正）

        val=0 → 正常亮度
        val=1 → 整体压暗（Gamma=1.4），适合强光过曝场景

        用法：
            my_car.darken_key(1)   # 压暗图像
            my_car.lane_dis_offset(speed=0.2, dis_hold=3.5, mode=4)
            my_car.darken_key(0)   # 恢复
        """
        self._darken_enabled = bool(val)

    def sensor_init(self, cfg):
        """
        初始化传感器

        根据配置初始化按键、灯光和红外传感器。

        参数:
            cfg: 配置字典，包含传感器的配置信息

        """
        cfg_sensor = cfg["io"]
        # print(cfg_sensor)
        self.key = Key4Btn(cfg_sensor["key"])
        # self.light = LedLight(cfg_sensor['light'])
        # self.left_sensor = Infrared(cfg_sensor['left_sensor'])
        # self.right_sensor = Infrared(cfg_sensor['right_sensor'])
        self.servo_1_angle_list = [-42, 165]
        self.servo_1_flag = 1  # 启动时默认 set_storage(True)
        self.servo_1 = ServoPwm(1, 180)
        self.servo_1.set_angle(self.servo_1_angle_list[self.servo_1_flag])
        self.blue_pad = BluetoothPad()
        self.shoot = PoutD(4)

    def set_storage(self, state=False):
        """
        设置储存仓的位置

        根据状态参数控制储存仓的开关。

        参数:
            state (bool): 储存仓状态。False 表示放下，True 表示收起。默认为 False。
        """
        flag = 1 if state else 0
        self.servo_1.set_angle(self.servo_1_angle_list[flag])

    def shooting(self):
        self.shoot.set(1)
        time.sleep(0.3)
        self.shoot.set(0)
        time.sleep(0.5)

    def car_pid_init(self, cfg):
        """
        初始化PID控制器

        根据配置初始化车道保持和目标检测的PID控制器。

        参数:
            cfg: 配置字典，包含PID控制器的配置信息
        """
        # ====== 巡线PID ======
        cfg_lane = cfg["lane_pid"]

        # [Mode4] 纯AI模型PID：先于死区注入创建，保留原始PID参数
        self.lane_pid_pure = PidCal2(**cfg_lane)

        # 角度：仅加极小的死区防直道噪声，不再用非线性能量映射
        # （小误差放大→提前转弯，大误差压缩→急弯转不动）
        def _angle_dead_zone(e):
            return 0.0 if abs(e) < 0.01 else e
        cfg_lane["cfg_pid_angle"]["error_map"] = _angle_dead_zone

        self.lane_pid = PidCal2(**cfg_lane)
        self.lane_pid_lat = PID(**cfg_lane.get("cfg_pid_lat_angle", cfg_lane["cfg_pid_angle"]))
        self.det_pid = PidCal2(**cfg["det_pid"])

    def camera_init(self, cfg):
        """
        初始化摄像头

        根据配置初始化前置摄像头和侧面摄像头。

        参数:
            cfg: 配置字典，包含摄像头的配置信息
        """
        camera_cfg = cfg["camera"]
        exposure_cfg = camera_cfg.get("exposure") or {}

        # 实际任务/水块相机：cap_front / video0
        self.cap_front = Camera(
            camera_cfg["front"],
            auto_exposure=camera_cfg.get("front_auto_exposure", 0),
            ae_cfg=exposure_cfg.get("front", {}),
            camera_name="front_task_video0",
        )

        # 实际循迹相机：cap_side / video2
        self.cap_side = Camera(
            camera_cfg["side"],
            auto_exposure=camera_cfg.get("side_auto_exposure", 0),
            ae_cfg=exposure_cfg.get("side", {}),
            camera_name="side_lane_video2",
        )

    def paddle_infer_init(self):
        """
        初始化Paddle推理

        初始化车道保持、前置方向识别、任务识别和OCR识别的推理接口。
        """
        # 前置巡线
        self.crusie = ClintInterface("lane")
        # 前置左右方向识别
        # self.front_det = ClintInterface('front')
        # 任务识别
        self.task_det = ClintInterface("task")
        # ocr识别
        self.ocr_rec = ClintInterface("ocr")
        # 识别为None
        self.last_det = None

    def ernie_bot_init(self):
        """
        初始化文心一言分析

        初始化、图像分析和订单分析的文心一言接口。
        """
        self.image_analysis = ErnieBotWrap()

        self.order_analysis = ErnieBotWrap()
        self.order_analysis.set_promt(str(OrderPrompt()))

    def animal_image_analysis(self, det=None, image=None):
        """裁剪动物并调用云端分析。

        det/image 可由刚完成的视觉对准传入。二者同时传入时不会再次运行
        task 检测，并保证检测框与裁剪图来自同一物理帧。旧调用不传参数时
        仍会自行检测，接口保持兼容。
        """
        perf_start = time.monotonic()
        detection_seconds = 0.0
        if det is None or image is None:
            det_start = time.monotonic()
            dets = self.get_detection_results(update_stream=False)
            detection_seconds = time.monotonic() - det_start
            if len(dets) <= 0:
                print("未检测到任何目标，无法裁剪")
                return None, None
            det = dets[0]
            image = getattr(self, "_last_det_img", None)

        if image is None:
            print("检测图像为空，无法裁剪")
            return None, None

        image = image.copy()
        cls_id, det_id, label, score, x_c, y_c, w, h = det

        # 将归一化坐标转换为像素坐标
        img_h, img_w = image.shape[:2]
        x_c = int((x_c + 1) / 2 * img_w)
        y_c = int((y_c + 1) / 2 * img_h)
        w = int(w * img_w / 2)
        h = int(h * img_h / 2)
        x1 = max(0, int(x_c - w / 2))
        y1 = max(0, int(y_c - h / 2))
        x2 = min(img_w, int(x_c + w / 2))
        y2 = min(img_h, int(y_c + h / 2))

        if x2 <= x1 or y2 <= y1:
            print("裁剪区域无效，跳过")
            return None, None
        cropped_img = image[y1:y2, x1:x2]

        encode_start = time.monotonic()
        ok, img_encoded = cv2.imencode(".jpg", cropped_img)
        encode_seconds = time.monotonic() - encode_start
        if not ok:
            print("动物裁剪图 JPEG 编码失败")
            return None, None
        # 转 base64 字符串
        base64_image = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

        cloud_start = time.monotonic()
        result, analysis = self.image_analysis.get_image_res(
            base64_image, mime_type="image/jpeg"
        )
        cloud_seconds = time.monotonic() - cloud_start
        self._last_animal_perf = {
            "detection_seconds": detection_seconds,
            "encode_seconds": encode_seconds,
            "cloud_seconds": cloud_seconds,
            "total_seconds": time.monotonic() - perf_start,
        }
        if bool(self.performance_cfg.get("timing_log", 1)):
            print(
                "[PERF][animal] detection={:.3f}s encode={:.3f}s "
                "cloud={:.3f}s total={:.3f}s".format(
                    detection_seconds,
                    encode_seconds,
                    cloud_seconds,
                    self._last_animal_perf["total_seconds"],
                )
            )
        print(f"image result: {result}  \nanalysis:{analysis}")
        return result, analysis

    @staticmethod
    def get_cfg(path):
        """
        获取配置文件

        读取并解析YAML配置文件，将端口号转换为整数类型。

        参数:
            path: 配置文件路径
        """
        from yaml import load, Loader

        # 把配置文件读取到内存
        with open(path, "r") as stream:
            yaml_dict = load(stream, Loader=Loader)
        port_list = yaml_dict["port_io"]
        # 转化为int
        for port in port_list:
            port["port"] = int(port["port"])
        # print(yaml_dict)

    # 延时函数
    def delay(self, time_hold):
        """
        延时函数

        延时指定的时间，期间会检查停止标志。

        参数:
            time_hold: 延时时间（秒）
        """
        start_time = time.time()
        while True:
            if self._stop_flag:
                return
            if time.time() - start_time > time_hold:
                break
            time.sleep(0.005)

    # 按键检测线程
    def key_thread_func(self):
        """
        按键检测线程

        持续检测按键状态，当检测到按键3时设置停止标志。
        """
        while True:
            if self._end_flag:
                return
            if self._stop_flag:
                # stop 状态可能稍后被业务代码清除，线程保留但不能空转占满 CPU。
                time.sleep(0.05)
                continue
            key_val = self.key.get_key()
            # print(key_val)
            if key_val == 3:
                self._stop_flag = True
            time.sleep(0.2)

    # 根据某个值获取列表中匹配的结果
    @staticmethod
    def get_list_by_val(list, index, val):
        """
        根据某个值获取列表中匹配的结果

        参数:
            list: 要搜索的列表
            index: 要匹配的索引位置
            val: 要匹配的值

        返回:
            匹配的元素，如果没有匹配的则返回None
        """
        for det in list:
            if det[index] == val:
                return det
        return None

    def move_base(self, sp, end_fuction, stop=STOP_PARAM):
        """
        基础移动方法

        设置车辆速度并持续移动，直到满足结束条件。

        参数:
            sp: 速度向量 [x, y, z]
            end_fuction: 结束条件函数，返回True时停止移动
            stop: 是否在结束后停止车辆，默认为STOP_PARAM
        """
        self.set_velocity(sp[0], sp[1], sp[2])
        while True:
            if self._stop_flag:
                return
            if end_fuction():
                break
            self.set_velocity(sp[0], sp[1], sp[2])
        if stop:
            self.set_velocity(0, 0, 0)

    #  高级移动，按着给定速度进行移动，直到满足条件
    # def move_advance(self, sp, value_h=None, value_l=None, times=1, sides=1, dis_out=0.2, stop=STOP_PARAM):
    #     """
    #     高级移动方法

    #     按照给定速度移动，直到满足传感器条件。

    #     参数:
    #         sp: 速度向量 [x, y, z]
    #         value_h: 传感器上限值，默认为1200
    #         value_l: 传感器下限值，默认为0
    #         times: 重复次数，默认为1
    #         sides: 传感器选择，1为左侧，-1为右侧
    #         dis_out: 距离限制，默认为0.2
    #         stop: 是否在结束后停止车辆，默认为STOP_PARAM
    #     """
    #     if value_h is None:
    #         value_h = 1200
    #     if value_l is None:
    #         value_l = 0
    #     # _sensor_usr = self.left_sensor
    #     # if sides == -1:
    #     #     _sensor_usr = self.right_sensor
    #     # 用于检测开始过渡部分的标记
    #     flag_start = False
    #     def end_fuction():
    #         nonlocal flag_start
    #         val_sensor = _sensor_usr.read()
    #         # print("val:", val_sensor)
    #         if val_sensor < value_h and val_sensor > value_l:
    #             return flag_start
    #         else:
    #             flag_start = True
    #             return False
    #     for i in range(times):
    #         self.move_base(sp, end_fuction, stop=False)
    #     if stop:
    #         self.stop()

    def move_time(self, sp, dur_time=1, stop=STOP_PARAM):
        """
        按时间移动

        以给定速度移动指定的时间。

        参数:
            sp: 速度向量 [x, y, z]
            dur_time: 移动时间（秒），默认为1
            stop: 是否在结束后停止车辆，默认为STOP_PARAM
        """
        self.set_velocity_for_duration(sp[0], sp[1], sp[2], dur_time)
        if stop:
            self.stop()

    def move_distance(self, sp, dis=0.1, stop=STOP_PARAM):
        """
        按距离移动

        以给定速度移动指定的距离。

        参数:
            sp: 速度向量 [x, y, z]
            dis: 移动距离，默认为0.1
            stop: 是否在结束后停止车辆，默认为STOP_PARAM
        """
        end_dis = self.get_distance() + dis

        def end_func():
            return self.get_distance() > end_dis

        self.move_base(sp, end_func, stop)

    # 计算两个坐标的距离
    def calculation_dis(self, pos_dst, pos_src):
        """
        计算两个坐标的距离

        计算两个二维坐标点之间的欧几里得距离。

        参数:
            pos_dst: 目标坐标 [x, y]
            pos_src: 源坐标 [x, y]

        返回:
            float: 两个坐标之间的距离
        """
        return math.sqrt(
            (pos_dst[0] - pos_src[0]) ** 2 + (pos_dst[1] - pos_src[1]) ** 2
        )

    def det2pose(self, det, w_r=0.06):
        """
        将检测结果转换为真实世界坐标

        根据检测结果和物体实际宽度，计算物体在真实世界中的坐标和距离。

        参数:
            det: 检测结果，包含 [x, y, w, h]（归一化坐标）
            w_r: 物体实际宽度（米），默认为0.06

        返回:
            tuple: (x坐标, y坐标, 距离)，单位为米
        """
        # r 真实  v 成像  f 焦点
        # rf 真实到焦点的距离  vf 相到焦点的距离
        vf_dis = 1.445
        x_v, y_v, w_v, h_v = det

        rf_dis = vf_dis * w_r / w_v
        x_r = x_v * rf_dis / vf_dis
        y_r = y_v * rf_dis / vf_dis
        return x_r, y_r, rf_dis

    def calibrate_blue_hsv(self, image, cx_norm, cy_norm, sample_size=20, margin=15):
        """
        根据模型检测到的蓝色方块中心，采样 20x20 区域的 HSV 值，
        动态计算该蓝色方块的颜色范围基准。

        参数:
            image:       BGR 图像
            cx_norm:     模型检测的中心 x 归一化坐标 (-1~1)
            cy_norm:     模型检测的中心 y 归一化坐标 (-1~1)
            sample_size: 采样区域边长（像素）
            margin:      HSV 上下浮动范围

        返回:
            (lower_blue, upper_blue) 或 (None, None) 采样失败
        """
        import cv2
        import numpy as np

        h, w = image.shape[:2]

        # 归一化坐标 → 像素坐标
        cx_px = int((cx_norm + 1) / 2 * w)
        cy_px = int((cy_norm + 1) / 2 * h)

        half = sample_size // 2
        x1 = max(0, cx_px - half)
        y1 = max(0, cy_px - half)
        x2 = min(w, cx_px + half)
        y2 = min(h, cy_px + half)

        if x2 - x1 < 5 or y2 - y1 < 5:
            print(f"[HSV标定] 采样区域太小，跳过")
            return None, None

        # 提取采样区域的 HSV
        roi = image[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 取每个通道的中位数作为基准
        h_vals = hsv_roi[:, :, 0].flatten()
        s_vals = hsv_roi[:, :, 1].flatten()
        v_vals = hsv_roi[:, :, 2].flatten()

        h_med = float(np.median(h_vals))
        s_med = float(np.median(s_vals))
        v_med = float(np.median(v_vals))

        lower_blue = (max(0,   int(h_med - margin)),
                       max(0,   int(s_med - margin)),
                       max(0,   int(v_med - margin)))
        upper_blue = (min(255, int(h_med + margin)),
                       min(255, int(s_med + margin)),
                       min(255, int(v_med + margin)))

        print(f"[HSV标定] 采样区域({x1},{y1})-({x2},{y2}) | "
              f"基准 H={h_med:.0f} S={s_med:.0f} V={v_med:.0f} | "
              f"范围 lower={lower_blue} upper={upper_blue}")
        return lower_blue, upper_blue

    def _get_color_center(self, color, save_debug, image=None):
        """内部调度：根据 color 参数调用对应的色域检测函数。
        返回 (cx, cy, found, color_tag)，color_tag 用于映射 label。
        image: 可选，预读的帧（保证dx/dy与保存图片同帧）"""
        if image is None:
            image = self.cap_front.read()  # 默认用侧摄读取
        if color == "red":
            cx, cy, ok = self.get_red_center(image=image, save_debug=save_debug)
            return cx, cy, ok, "red"
        elif color == "yellow":
            cx, cy, ok = self.get_yellow_center(image=image, save_debug=save_debug)
            return cx, cy, ok, "yellow"
        elif color == "dark_blue":
            cx, cy, ok = self.get_dark_blue_center(image=image, save_debug=save_debug)
            return cx, cy, ok, "dark_blue"
        elif color == "gray":
            cx, cy, ok = self.get_gray_center(image=image, save_debug=save_debug)
            return cx, cy, ok, "gray"
        elif color == "dark_blue+yellow":
            # 模式2: 同时检测深蓝和黄色
            cx_y, cy_y, ok_y = self.get_yellow_center(image=image, save_debug=save_debug)
            cx_b, cy_b, ok_b = self.get_dark_blue_center(image=image, save_debug=save_debug)
            print(f"[color_mode=2] 深蓝={'OK' if ok_b else '--'} ({cx_b:+.3f},{cy_b:+.3f})  "
                  f"黄色={'OK' if ok_y else '--'} ({cx_y:+.3f},{cy_y:+.3f})")
            if ok_b:
                return cx_b, cy_b, True, "dark_blue"   # 深蓝优先
            elif ok_y:
                return cx_y, cy_y, True, "yellow"
            return 0.0, 0.0, False, "unknown"
        else:
            cx, cy, ok = self.get_blue_center(image=image, save_debug=save_debug)  # 默认浅蓝
            return cx, cy, ok, "light_blue"

    def _get_color_center_direct(self, color, image=None, save_debug=False):
        """_get_color_center 的别名，显式传递预读帧，保证帧一致性。

        同时作为颜色鲁棒化旁路验证的分发点：
          mode=0 -> 完全关闭，行为与旧版本一致
          mode=1 -> 同帧计算 new 结果但始终返回 old，只记录差异
          mode=2 -> new 结果质量合格时使用，否则回退 old
        """
        # 同一物理帧供 old/new 共用，禁止 new 额外读相机
        if image is None:
            image = self.cap_front.read()
        old_result = self._get_color_center(color, save_debug=save_debug, image=image)

        mode = int(getattr(self, "_vision_color_object_mode", 0))
        if mode == 0:
            return old_result

        # 计算 seeded 新结果；任何异常都必须回退旧算法，不得传播到控制循环
        try:
            new_result = self._color_center_seeded_dispatch(color, image, save_debug)
        except Exception as exc:  # noqa: BLE001
            self._last_color_ab_perf = {"error": str(exc)}
            return old_result

        if new_result is None:
            return old_result

        n_cx, n_cy, n_ok, n_tag, n_quality = new_result

        # 记录 A/B（节流）
        img_shape = image.shape[:2] if image is not None else None
        self._log_color_ab(color, old_result, new_result, img_shape)

        # 保存并排调试图（仅 save_debug=1 且 mode 1/2，节流）
        if self._vision_color_object_save_debug and image is not None:
            self._save_seeded_debug_image(image, color, old_result, new_result)

        if mode == 1:
            # 旁路：始终由旧结果控制
            return old_result

        # mode == 2：新结果合格才启用
        if n_ok:
            center_px = n_quality.get("center_px") if isinstance(n_quality, dict) else None
            if center_px is not None:
                self._last_seeded_center_by_color[n_tag] = center_px
            return n_cx, n_cy, n_ok, n_tag
        return old_result

    def _prepare_color_frame(self, image):
        """同一帧只做一次高斯模糊和一次 BGR2HSV，供多个颜色共用。"""
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        return blur, hsv

    def _color_center_seeded_dispatch(self, color, image, save_debug):
        """把 color 映射到 seeded 检测。深蓝+黄色保持深蓝优先。"""
        if color == "dark_blue+yellow":
            b = self._get_color_center_seeded(image, "dark_blue",
                                              previous_center=self._last_seeded_center_by_color.get("dark_blue"),
                                              save_debug=save_debug)
            y = self._get_color_center_seeded(image, "yellow",
                                              previous_center=self._last_seeded_center_by_color.get("yellow"),
                                              save_debug=save_debug)
            if b[2]:
                return b
            if y[2]:
                return y
            return 0.0, 0.0, False, "unknown", {"reason": "none"}
        color_name = color if color in ("red", "yellow", "dark_blue", "gray") else "light_blue"
        return self._get_color_center_seeded(
            image, color_name,
            previous_center=self._last_seeded_center_by_color.get(color_name),
            save_debug=save_debug)

    def _get_color_center_seeded(self, image, color_name, expected_center=None,
                                 previous_center=None, save_debug=False):
        """基于软色距 + 严格种子的中心检测。

        返回 (cx_norm, cy_norm, ok, color_tag, quality)。
        严禁在内部再次读取相机；失败必须返回 ok=False 和明确 reason。
        """
        t0 = time.time()
        h, w = image.shape[:2]

        targets = self._vision_color_object_cfg.get("targets", {})
        targets = targets if isinstance(targets, dict) else {}
        target = targets.get(color_name)
        if not isinstance(target, dict):
            return 0.0, 0.0, False, color_name, {"reason": "no_config"}

        hsv_samples = target.get("hsv_samples") or []
        if not hsv_samples:
            return 0.0, 0.0, False, color_name, {"reason": "no_hsv_samples"}

        seed_distance = float(self._vision_color_object_cfg.get("seed_distance", 0.18))
        min_seed_pixels = int(self._vision_color_object_cfg.get("min_seed_pixels", 8))
        min_component_area = int(self._vision_color_object_cfg.get("min_component_area", 80))
        max_area_ratio = float(self._vision_color_object_cfg.get("max_component_area_ratio", 0.45))
        max_shape_gap = float(self._vision_color_object_cfg.get("max_shape_center_gap_px", 12))
        max_prev_gap = float(self._vision_color_object_cfg.get("max_previous_center_gap_px", 140))

        roi = target.get("roi", [0.0, 0.0, 1.0, 1.0])
        shape = target.get("shape", "rectangle")

        # 一次预处理
        _blur, hsv = self._prepare_color_frame(image)

        is_gray = (color_name == "gray")
        weights = (0.0, 0.55, 0.45) if is_gray else (0.65, 0.25, 0.10)

        # 软色距（灰色忽略 H）
        distance = color_region.min_hsv_distance(hsv, hsv_samples, weights, ignore_hue=is_gray)

        # 目标专用 S/V 门限：彩色排除低饱和白高光；灰色用 S 上限 + V 范围
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        if is_gray:
            seed_gate = ((s_ch <= 45) & (v_ch >= 30) & (v_ch <= 170)).astype(np.uint8) * 255
        else:
            min_seed_s = float(self._vision_color_object_cfg.get("min_seed_saturation", 10))
            seed_gate = ((s_ch >= min_seed_s) & (v_ch >= 40)).astype(np.uint8) * 255

        roi_mask = color_region.normalized_roi_mask(hsv.shape, roi)
        seed = color_region.build_seed_mask(
            distance, seed_distance, roi_mask, seed_gate)

        max_area = int(max_area_ratio * h * w)
        component_mask, labels, kept = color_region.retain_seeded_components(
            seed, min_area=min_component_area, max_area=max_area,
            min_seed_pixels=min_seed_pixels)

        # 暂存中间结果供并排图保存（仅在 save_debug=1 时）
        if self._vision_color_object_save_debug:
            self._last_seeded_debug = {
                "distance": distance,
                "seed": seed,
                "component_mask": component_mask,
                "filled": None,
                "new_center_px": None,
                "shape_center_px": None,
            }

        if not kept:
            return 0.0, 0.0, False, color_name, {"reason": "no_component"}

        # 候选选择：面积 + 期望/上一帧中心约束（期望中心尚未由控制层传入时退化为面积优先）
        best = None
        best_score = None
        for comp in kept:
            comp_cx, comp_cy = comp["center"]
            score = float(comp["area"])
            if expected_center is not None:
                ex, ey = expected_center
                score -= 0.5 * (abs(comp_cx - ex) + abs(comp_cy - ey))
            if previous_center is not None:
                px, py = previous_center
                gap = math.sqrt((comp_cx - px) ** 2 + (comp_cy - py) ** 2)
                if gap > max_prev_gap:
                    continue
                score -= 0.3 * gap
            if best_score is None or score > best_score:
                best_score = score
                best = comp

        if best is None:
            return 0.0, 0.0, False, color_name, {"reason": "no_candidate"}

        # 只填选中目标内部小洞
        single = np.where(labels == best["label"], 255, 0).astype(np.uint8)
        filled = color_region.fill_internal_holes(single)

        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.0, False, color_name, {"reason": "no_contour"}
        contour = max(contours, key=cv2.contourArea)

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            return 0.0, 0.0, False, color_name, {"reason": "empty_moments"}

        cx_px = float(moments["m10"] / moments["m00"])
        cy_px = float(moments["m01"] / moments["m00"])

        # 形状中心交叉验证：矩形用 minAreaRect，圆形用 minEnclosingCircle
        if shape == "circle":
            (shape_x, shape_y), _radius = cv2.minEnclosingCircle(contour)
        else:
            (shape_x, shape_y), _size, _ang = cv2.minAreaRect(contour)
        shape_gap = math.sqrt((cx_px - shape_x) ** 2 + (cy_px - shape_y) ** 2)
        if shape_gap > max_shape_gap:
            return 0.0, 0.0, False, color_name, {"reason": "shape_gap", "shape_gap_px": shape_gap}

        cx_norm = (cx_px / w) * 2.0 - 1.0
        cy_norm = (cy_px / h) * 2.0 - 1.0

        if self._last_seeded_debug is not None:
            self._last_seeded_debug["filled"] = filled
            self._last_seeded_debug["new_center_px"] = (cx_px, cy_px)
            self._last_seeded_debug["shape_center_px"] = (shape_x, shape_y)

        quality = {
            "reason": "ok",
            "area": int(best["area"]),
            "seed_pixels": int(best["seed_count"]),
            "shape_gap_px": shape_gap,
            "center_px": (cx_px, cy_px),
            "elapsed_ms": (time.time() - t0) * 1000.0,
        }
        return cx_norm, cy_norm, True, color_name, quality

    def _log_color_ab(self, color, old_result, new_result, img_shape=None):
        """节流记录 old/new 中心差异，mode=1/2 通用。"""
        log_every = max(1, int(self._vision_color_object_cfg.get("log_every_frames", 10)))
        counter = getattr(self, "_color_ab_log_counter", 0)
        counter += 1
        self._color_ab_log_counter = counter
        if counter % log_every != 0:
            return
        o_cx, o_cy, o_ok, o_tag = old_result
        n_cx, n_cy, n_ok, n_tag, n_quality = new_result
        q = n_quality if isinstance(n_quality, dict) else {}
        diff_px = None
        if o_ok and n_ok and img_shape is not None:
            img_h, img_w = img_shape[:2]
            diff_px = ((n_cx - o_cx) * img_w / 2.0, (n_cy - o_cy) * img_h / 2.0)
        print("[OBJECT_AB] color={} old_ok={} new_ok={} old=({:+.3f},{:+.3f}) "
              "new=({:+.3f},{:+.3f}) diff_px={} area={} reason={} ms={:.2f}".format(
                  color, int(o_ok), int(n_ok),
                  float(o_cx), float(o_cy), float(n_cx), float(n_cy),
                  ("({:.1f},{:.1f})".format(*diff_px) if diff_px else "None"),
                  q.get("area"), q.get("reason"), q.get("elapsed_ms", 0.0)))

    def _save_seeded_debug_image(self, image, color, old_result, new_result):
        """保存并排调试图：原图 + 软色距 + 严格种子 + 最终连通域。

        仅当 save_debug=1 且 mode 1/2 时被调用，节流避免填盘。
        """
        import os
        from datetime import datetime

        dbg = self._last_seeded_debug
        if not isinstance(dbg, dict):
            return

        # 节流：每 log_every_frames 帧保存一次
        log_every = max(1, int(self._vision_color_object_cfg.get("log_every_frames", 2)))
        self._color_debug_save_counter += 1
        if self._color_debug_save_counter % log_every != 0:
            return

        h, w = image.shape[:2]

        # 面板1：原图 + 旧中心(绿) + 新中心(红) + 形状中心(蓝)
        panel_orig = image.copy()
        o_cx, o_cy, o_ok, o_tag = old_result
        if o_ok:
            ox = int((o_cx + 1.0) / 2.0 * w)
            oy = int((o_cy + 1.0) / 2.0 * h)
            cv2.circle(panel_orig, (ox, oy), 8, (0, 255, 0), 2)
        nc = dbg.get("new_center_px")
        if nc is not None:
            cv2.circle(panel_orig, (int(nc[0]), int(nc[1])), 8, (0, 0, 255), 2)
        sc = dbg.get("shape_center_px")
        if sc is not None:
            cv2.circle(panel_orig, (int(sc[0]), int(sc[1])), 5, (255, 0, 0), 1)
        cv2.putText(panel_orig, color, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        def _mask2bgr(m, color_bgr=(255, 255, 255)):
            if m is None:
                return np.zeros((h, w, 3), dtype=np.uint8)
            g = np.where(m > 0, 255, 0).astype(np.uint8)
            gb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            if color_bgr != (255, 255, 255):
                gb[gb[:, :, 0] == 255] = color_bgr
            return gb

        # 面板2：软色距热力图
        dist = dbg.get("distance")
        if dist is not None:
            disp = np.clip(dist, 0.0, 1.0)
            disp = (disp * 255.0).astype(np.uint8)
            panel_dist = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
        else:
            panel_dist = np.zeros((h, w, 3), dtype=np.uint8)

        panel_seed = _mask2bgr(dbg.get("seed"), (0, 255, 255))       # 黄
        panel_filled = _mask2bgr(dbg.get("filled"), (0, 255, 0))     # 绿

        # 顶部标题条
        def _title(text):
            p = np.zeros((20, w, 3), dtype=np.uint8)
            cv2.putText(p, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return p

        panels = [
            ("orig", panel_orig),
            ("distance", panel_dist),
            ("seed", panel_seed),
            ("filled", panel_filled),
        ]
        rows = []
        for name, p in panels:
            rows.append(_title(name))
            rows.append(p)
        montage = np.vstack(rows)

        # 保存目录
        save_dir = os.path.join(self.path_dir, "color_robust_debug")
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            return
        ts = datetime.now().strftime("%H%M%S_%f")[:-3]
        color_safe = str(color).replace("/", "-").replace("+", "_")
        filename = os.path.join(save_dir, "{}_{}_{}.jpg".format(color_safe, ts, self._color_debug_save_counter))
        try:
            cv2.imwrite(filename, montage)
        except Exception:
            return


    def save_align_debug_img_hsv(self, delta_x, dx, label, dy=None,
                                  final=False, frame=0, stage="", out_x=0.0, out_y=0.0,
                                  delta_y=0.0, ball_no=""):
        """HSV模式专用：使用 _last_align_img（与dx/dy同帧）保存调试图。"""
        import os, cv2
        if not hasattr(self, '_last_align_img') or self._last_align_img is None:
            return None
        if dx is None:
            return None

        img = self._last_align_img.copy()
        img_h, img_w = img.shape[:2]

        center_x = img_w // 2
        delta_px = int((delta_x + 1) / 2 * img_w)
        dx_px = int((dx + 1) / 2 * img_w)
        error = delta_x - dx

        cv2.line(img, (center_x, 0), (center_x, img_h), (255, 0, 0), 2)
        cv2.line(img, (delta_px, 0), (delta_px, img_h), (0, 255, 0), 2)
        cv2.line(img, (dx_px, 0), (dx_px, img_h), (0, 0, 255), 2)

        center_y = img_h // 2
        if dy is not None and delta_y is not None:
            dy_px = int((dy + 1) / 2 * img_h)
            delta_y_px = int((delta_y + 1) / 2 * img_h)
            cv2.line(img, (0, center_y), (img_w, center_y), (255, 0, 0), 1)
            cv2.line(img, (0, delta_y_px), (img_w, delta_y_px), (0, 255, 0), 1)
            cv2.line(img, (0, dy_px), (img_w, dy_px), (0, 0, 255), 1)

        final_tag = "[FINAL]" if final else ""
        ball_str = f" [{ball_no}]" if ball_no else ""
        info_lines = [
            f"#{frame} {stage} | {final_tag}{ball_str}",
            f"err_x={error:+.4f} dx={dx:+.4f} | label={label}",
        ]
        for j, txt in enumerate(info_lines):
            cv2.putText(img, txt, (10, 30 + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        try:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "det_target_debug")
            os.makedirs(filepath, exist_ok=True)
            ball_tag = ball_no.replace("/", "-") if ball_no else ""
            prefix = f"hsv_{ball_tag}_{label}" if ball_tag else f"hsv_align_{label}"
            filename = f"{prefix}_FINAL.jpg" if final else f"{prefix}_f{frame}.jpg"
            cv2.imwrite(os.path.join(filepath, filename), img)
        except Exception:
            pass

    def get_blue_center(self, image=None, save_debug=False, save_dir="blue_center_debug", hsv_range=None):
        """
        HSV 色彩提取法：提取画面中的蓝色方块，计算其中心坐标。

        通过 HSV 掩码分离蓝色区域，形态学去噪后取最大轮廓的中心点，
        返回该中心在归一化图像坐标（-1~1）中的位置。

        参数:
            image:      输入图像（BGR），为 None 时自动读取前摄像头
            save_debug: 是否保存调试图片
            save_dir:   调试图片保存目录

        返回:
            (cx_norm, cy_norm, success)
            cx_norm: 中心 x 归一化坐标 (-1~1)，-1=最左，0=中心，1=最右
            cy_norm: 中心 y 归一化坐标 (-1~1)，-1=最上，0=中心，1=最下
            success: 是否成功找到蓝色方块
        """
        import cv2
        import os
        if image is None:
            image = self.cap_front.read()

        h, w = image.shape[:2]

        # ====== 0. 高斯模糊 → 强力平滑反光/阴影边界 ======
        image_blur = cv2.GaussianBlur(image, (9, 9), 0)

        # ====== 1. HSV 浅蓝双段掩码（正常浅蓝 + 曝光高亮 + 阴影暗区） ======
        hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
        if hsv_range is not None:
            lower_blue, upper_blue = hsv_range
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        else:
            # 段1：正常浅蓝 + 阴影暗区（S≥10 接纳弱饱和阴影, V≥50 覆盖暗区）
            lower_blue1 = (95, 10, 50)
            upper_blue1 = (125, 255, 255)
            mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
            # 段2：过曝白斑（S<15低饱和, V≥140高亮, H仍在浅蓝区间）
            lower_blue2 = (95, 3, 140)
            upper_blue2 = (125, 20, 255)
            mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
            mask_blue = cv2.bitwise_or(mask1, mask2)

        # ---- 图像上1/3强制置零，默认为非蓝色 ----
        mask_blue[: h // 6, :] = 0

        # ====== 2. 形态学去噪+反光填充 ======
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_big   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_small)   # 去噪点
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_big)    # 填充反光空洞

        # ====== 3. 找最大轮廓 → 中心点 ======
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if save_debug:
                self._save_blue_debug(image, mask_blue, None, 0, 0, 0, 0, 0, False, save_dir)
            return 0.0, 0.0, False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        M = cv2.moments(largest)
        if M["m00"] < 50:
            if save_debug:
                self._save_blue_debug(image, mask_blue, largest, area, 0, 0, 0, 0, False, save_dir)
            return 0.0, 0.0, False

        cx_px = int(M["m10"] / M["m00"])
        cy_px = int(M["m01"] / M["m00"])

        # ====== 4. 转归一化坐标 (-1 ~ 1) ======
        cx_norm = (cx_px / w) * 2.0 - 1.0   # 0→中心, -1→最左, 1→最右
        cy_norm = (cy_px / h) * 2.0 - 1.0   # 0→中心, -1→最上, 1→最下

        # ====== 5. 保存调试图片 ======
        if save_debug:
            self._save_blue_debug(image, mask_blue, largest, area, cx_px, cy_px, cx_norm, cy_norm, True, save_dir)

        return cx_norm, cy_norm, True

    def _save_blue_debug(self, image, mask, contour, area, cx_px, cy_px, cx_norm, cy_norm, found, save_dir):
        """保存蓝色方块检测的调试图片"""
        import cv2
        import os
        from datetime import datetime

        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]

        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2

        # 左：原图 + 标注
        debug = image.copy()
        if found:
            # 画轮廓
            cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
            # 画中心十字
            cv2.line(debug, (cx_px - 15, cy_px), (cx_px + 15, cy_px), (0, 0, 255), 2)
            cv2.line(debug, (cx_px, cy_px - 15), (cx_px, cy_px + 15), (0, 0, 255), 2)
            cv2.circle(debug, (cx_px, cy_px), 5, (0, 0, 255), -1)
        # 画图像中心十字（虚线）
        cv2.line(debug, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 1)
        cv2.line(debug, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 1)

        info_lines = [
            f"found={found}  area={area:.0f}",
            f"center_px=({cx_px},{cy_px})",
            f"norm=({cx_norm:+.3f},{cy_norm:+.3f})",
        ]
        for i, text in enumerate(info_lines):
            cv2.putText(debug, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # 右：mask 图 + 轮廓叠加
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if found:
            cv2.drawContours(mask_color, [contour], -1, (0, 255, 0), 2)
            cv2.circle(mask_color, (cx_px, cy_px), 5, (0, 0, 255), -1)

        combined = np.hstack([debug, mask_color])
        filename = os.path.join(save_dir, f"blue_{timestamp}_{'ok' if found else 'fail'}.jpg")
        cv2.imwrite(filename, combined)
        # print(f"[get_blue_center] 调试图已保存: {filename}")

    def get_red_center(self, image=None, save_debug=False, save_dir="red_center_debug", hsv_range=None):
        """
        HSV 色彩提取法：提取画面中的红色方块，计算其中心坐标。
        红色在 HSV 中跨越 0° 边界，需要两段掩码拼接。

        参数与返回格式同 get_blue_center。
        """
        import cv2
        import os
        if image is None:
            image = self.cap_front.read()

        h, w = image.shape[:2]

        # ====== 0. 高斯模糊 → 平滑反光 ======
        image_blur = cv2.GaussianBlur(image, (5, 5), 0)

        # ====== 1. HSV 红色掩码（两段拼接，跨越0°边界） ======
        hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
        if hsv_range is not None:
            lower_red, upper_red = hsv_range
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
        else:
            lower_red1 = (0, 30, 150)      # 红色段1: H=0~8
            upper_red1 = (8, 255, 255)
            lower_red2 = (170, 30, 150)    # 红色段2: H=170~180
            upper_red2 = (180, 255, 255)
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # ---- 图像上1/3强制置零 ----
        mask_red[: h // 3, :] = 0

        # ====== 2. 形态学去噪+反光填充 ======
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_big   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_small)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_big)

        # ====== 3. 找最大轮廓 → 中心点 ======
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if save_debug:
                self._save_hsv_debug(image, mask_red, None, 0, 0, 0, 0, 0, 0, False, save_dir, "red")
            return 0.0, 0.0, False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        M = cv2.moments(largest)
        if M["m00"] < 50:
            if save_debug:
                self._save_hsv_debug(image, mask_red, largest, area, 0, 0, 0, 0, 0, False, save_dir, "red")
            return 0.0, 0.0, False

        cx_px = int(M["m10"] / M["m00"])
        cy_px = int(M["m01"] / M["m00"])

        cx_norm = (cx_px / w) * 2.0 - 1.0
        cy_norm = (cy_px / h) * 2.0 - 1.0

        if save_debug:
            self._save_hsv_debug(image, mask_red, largest, area, cx_px, cy_px, cx_norm, cy_norm, True, save_dir, "red")

        return cx_norm, cy_norm, True

    def get_yellow_center(self, image=None, save_debug=False, save_dir="yellow_center_debug", hsv_range=None):
        """
        HSV 色彩提取法：提取画面中的黄色方块，计算其中心坐标。

        参数与返回格式同 get_blue_center。
        """
        import cv2
        import os
        if image is None:
            image = self.cap_front.read()

        h, w = image.shape[:2]

        # ====== 0. 高斯模糊 ======
        image_blur = cv2.GaussianBlur(image, (5, 5), 0)

        # ====== 1. HSV 黄色掩码 ======
        hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
        if hsv_range is not None:
            lower_yellow, upper_yellow = hsv_range
        else:
            lower_yellow = (22, 40, 70)     # S≥40强排白色卡槽(白S<15), V≥70纳暗区
            upper_yellow = (38, 255, 255)   # H≤38纯黄, V不封顶(高亮纳进来)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # ---- 上下1/4 + 左右1/8置零 ----
        mask_yellow[: h // 4, :] = 0
        mask_yellow[3 * h // 4:, :] = 0
        mask_yellow[:, : w // 8] = 0
        mask_yellow[:, 7 * w // 8:] = 0

        # ====== 2. 形态学去噪+填充 ======
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_big   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel_small)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel_big)

        # ====== 3. 找最大轮廓 → 面积过滤小噪点 ======
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if save_debug:
                self._save_hsv_debug(image, mask_yellow, None, 0, 0, 0, 0, 0, False, save_dir, "yellow", all_contours=[])
            return 0.0, 0.0, False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 200:   # 小面积噪点直接丢弃
            if save_debug:
                self._save_hsv_debug(image, mask_yellow, largest, area, 0, 0, 0, 0, False, save_dir, "yellow", all_contours=contours)
            return 0.0, 0.0, False

        M = cv2.moments(largest)
        cx_px = int(M["m10"] / M["m00"])
        cy_px = int(M["m01"] / M["m00"])

        cx_norm = (cx_px / w) * 2.0 - 1.0
        cy_norm = (cy_px / h) * 2.0 - 1.0

        if save_debug:
            self._save_hsv_debug(image, mask_yellow, largest, area, cx_px, cy_px, cx_norm, cy_norm, True, save_dir, "yellow", all_contours=contours)

        return cx_norm, cy_norm, True

    def get_dark_blue_center(self, image=None, save_debug=False, save_dir="dark_blue_center_debug", hsv_range=None):
        """
        HSV 色彩提取法：提取画面中的深蓝色方块（比浅蓝模式更深）。

        参数与返回格式同 get_blue_center。
        """
        import cv2
        import os
        if image is None:
            image = self.cap_front.read()

        h, w = image.shape[:2]

        # ====== 0. 高斯模糊 → 强力平滑反光 ======
        image_blur = cv2.GaussianBlur(image, (9, 9), 0)

        # ====== 1. HSV 深蓝双段掩码（正常蓝 + 反光高亮区） ======
        hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
        if hsv_range is not None:
            lower_blue, upper_blue = hsv_range
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        else:
            # 段1：正常深蓝（S≥20 纯色区, V≥35）
            lower_blue1 = (100, 20, 35)
            upper_blue1 = (120, 255, 255)
            mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
            # 段2：反光白斑（S<20低饱和, V≥180高亮, H仍在蓝色区间）
            lower_blue2 = (100, 3, 180)
            upper_blue2 = (120, 25, 255)
            mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
            mask_blue = cv2.bitwise_or(mask1, mask2)

        # ---- 上下1/4 + 左右1/8置零 ----
        mask_blue[: h // 4, :] = 0
        mask_blue[3 * h // 4:, :] = 0
        mask_blue[:, : w // 8] = 0
        mask_blue[:, 7 * w // 8:] = 0

        # ====== 2. 形态学去噪+反光填充 ======
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_big   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_small)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_big)

        # ====== 3. 找最大轮廓 → 面积过滤小噪点 ======
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if save_debug:
                self._save_hsv_debug(image, mask_blue, None, 0, 0, 0, 0, 0, False, save_dir, "dark_blue", all_contours=[])
            return 0.0, 0.0, False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 200:
            if save_debug:
                self._save_hsv_debug(image, mask_blue, largest, area, 0, 0, 0, 0, False, save_dir, "dark_blue", all_contours=contours)
            return 0.0, 0.0, False

        M = cv2.moments(largest)
        cx_px = int(M["m10"] / M["m00"])
        cy_px = int(M["m01"] / M["m00"])

        cx_norm = (cx_px / w) * 2.0 - 1.0
        cy_norm = (cy_px / h) * 2.0 - 1.0

        if save_debug:
            self._save_hsv_debug(image, mask_blue, largest, area, cx_px, cy_px, cx_norm, cy_norm, True, save_dir, "dark_blue", all_contours=contours)

        return cx_norm, cy_norm, True

    def get_gray_center(self, image=None, save_debug=False, save_dir="gray_center_debug", hsv_range=None):
        """
        HSV 色彩提取法：提取画面中的纯灰色方块。

        灰色特征：极低饱和度 S, 中等亮度 V, H 无意义。
        """
        import cv2
        import os
        if image is None:
            image = self.cap_front.read()

        h, w = image.shape[:2]

        image_blur = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)

        if hsv_range is not None:
            lower_gray, upper_gray = hsv_range
        else:
            lower_gray = (0, 0, 25)       # H不限, S≥0, V≥25 排纯黑
            upper_gray = (180, 35, 180)   # H不限, S≤35 排彩色, V≤180 排白+浅灰
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

        # ---- 上1/4 + 下1/4置零 ----
        mask_gray[: h // 4, :] = 0
        mask_gray[3 * h // 4:, :] = 0

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_big   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel_small)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel_big)

        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if save_debug:
                self._save_hsv_debug(image, mask_gray, None, 0, 0, 0, 0, 0, False, save_dir, "gray", all_contours=[])
            return 0.0, 0.0, False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 100:
            if save_debug:
                self._save_hsv_debug(image, mask_gray, largest, area, 0, 0, 0, 0, False, save_dir, "gray", all_contours=contours)
            return 0.0, 0.0, False

        M = cv2.moments(largest)
        cx_px = int(M["m10"] / M["m00"])
        cy_px = int(M["m01"] / M["m00"])

        cx_norm = (cx_px / w) * 2.0 - 1.0
        cy_norm = (cy_px / h) * 2.0 - 1.0

        if save_debug:
            self._save_hsv_debug(image, mask_gray, largest, area, cx_px, cy_px, cx_norm, cy_norm, True, save_dir, "gray", all_contours=contours)

        return cx_norm, cy_norm, True

    def _save_hsv_debug(self, image, mask, contour, area, cx_px, cy_px, cx_norm, cy_norm, found, save_dir, color_name, all_contours=None):
        """保存 HSV 方块检测的调试图片。all_contours: 全部轮廓，保留的绿框，丢弃的红框。"""
        import cv2
        import os
        from datetime import datetime

        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]

        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        MIN_AREA = 200

        debug = image.copy()

        # ---- 画所有轮廓：保留的绿框 + 面积，丢弃的红框 + 面积 ----
        if all_contours:
            for cnt in all_contours:
                a = cv2.contourArea(cnt)
                if cnt is contour and found:
                    cv2.drawContours(debug, [cnt], -1, (0, 255, 0), 2)   # 保留 → 绿
                else:
                    cv2.drawContours(debug, [cnt], -1, (0, 0, 255), 1)   # 丢弃 → 红
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        tx = int(M["m10"] / M["m00"])
                        ty = int(M["m01"] / M["m00"])
                        cv2.putText(debug, f"{a:.0f}", (tx - 15, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # ---- 球心：红十字 | 摄像头中心：蓝十字 | 误差线：黄线 ----
        err_x_px = center_x - cx_px    # 像素误差（正=球偏左）
        err_y_px = center_y - cy_px
        err_x_norm = -cx_norm          # 归一化误差（球心→中心，delta_x=0时）
        err_y_norm = -cy_norm

        # ---- 球心：红色实心 + 贯穿全图的十字虚线 ----
        if found:
            cv2.circle(debug, (cx_px, cy_px), 6, (0, 0, 255), -1)
            # 球心水平虚线（红色）
            for xx in range(0, w, 8):
                cv2.line(debug, (xx, cy_px), (min(xx+4, w), cy_px), (0, 0, 255), 1)
            # 球心垂直虚线（红色）
            for yy in range(0, h, 8):
                cv2.line(debug, (cx_px, yy), (cx_px, min(yy+4, h)), (0, 0, 255), 1)

        # ---- 摄像头中心：蓝色实心 + 贯穿全图的十字虚线 ----
        cv2.circle(debug, (center_x, center_y), 4, (255, 0, 0), -1)
        for xx in range(0, w, 8):
            cv2.line(debug, (xx, center_y), (min(xx+4, w), center_y), (255, 0, 0), 1)
        for yy in range(0, h, 8):
            cv2.line(debug, (center_x, yy), (center_x, min(yy+4, h)), (255, 0, 0), 1)

        # ---- 误差标注：Δx 在球心x轴下方, Δy 在球心y轴右侧 ----
        if found:
            # Δx 标注（球心 x 轴与摄像头中心 y 交点处）
            dx_label_x = (cx_px + center_x) // 2
            dx_label_y = cy_px + 22
            cv2.putText(debug, f"dx={err_x_px:+d}px ({err_x_norm:+.3f})",
                        (dx_label_x - 50, dx_label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Δy 标注（球心 y 轴与摄像头中心 x 交点处）
            dy_label_x = cx_px + 15
            dy_label_y = (cy_px + center_y) // 2
            cv2.putText(debug, f"dy={err_y_px:+d}px ({err_y_norm:+.3f})",
                        (dy_label_x, dy_label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        tag = getattr(self, '_debug_tag', '')
        tag_str = f" [{tag}]" if tag else ""
        info_lines = [
            f"BALL{tag_str} color={color_name} found={found} area={area:.0f}",
            f"ball_center=({cx_px},{cy_px})px  ({cx_norm:+.3f},{cy_norm:+.3f})norm",
            f"cam_center =({center_x},{center_y})px  (0.000,0.000)norm",
            f"error=({err_x_px:+d},{err_y_px:+d})px  ({err_x_norm:+.3f},{err_y_norm:+.3f})norm",
            f"contours={len(all_contours) if all_contours else 0}",
        ]
        # 右上角大号球号
        if tag:
            cv2.putText(debug, f"Ball {tag}", (w - 180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        for i, text in enumerate(info_lines):
            cv2.putText(debug, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if found:
            cv2.drawContours(mask_color, [contour], -1, (0, 255, 0), 2)
            cv2.circle(mask_color, (cx_px, cy_px), 5, (0, 0, 255), -1)
        if all_contours:
            for cnt in all_contours:
                if cnt is not contour:
                    cv2.drawContours(mask_color, [cnt], -1, (0, 0, 255), 1)

        try:
            combined = np.hstack([debug, mask_color])
        except Exception as e:
            print(f"[DEBUG] hstack error: {e}, debug.shape={debug.shape}, mask_color.shape={mask_color.shape}")
            combined = debug
        tag = getattr(self, '_debug_tag', '').replace('/', '-')
        tag_prefix = f"{tag}_" if tag else ""
        filename = os.path.join(save_dir, f"{tag_prefix}{color_name}_{timestamp}_{'ok' if found else 'fail'}.jpg")
        try:
            cv2.imwrite(filename, combined)
            print(f"[DEBUG] saved: {filename}")
        except Exception as e:
            print(f"[DEBUG] imwrite error: {e}")

    # 侧面摄像头进行位置定位
    def lane_det_location(
        self,
        speed,
        pts_tar=[[0, 70, "text_det", 0, 0, 0, 0.70, 0.70]],
        dis_out=0.05,
        side=1,
        time_out=2,
        det="task",
    ):
        """
        侧面摄像头进行位置定位

        使用侧面摄像头检测目标并进行位置定位，通过PID控制调整车辆位置。

        参数:
            speed: 移动速度
            pts_tar: 目标点列表，每个元素包含 [id, 宽度, 标签, 置信度, x, y, w, h]
            dis_out: 距离限制，默认为0.05
            side: 方向，1为正方向，-1为反方向
            time_out: 超时时间（秒），默认为2
            det: 检测类型，默认为'task'

        返回:
            int: 目标索引，如果超时或距离超出限制则返回False
        """
        end_time = time.time() + time_out
        infer = self.task_det
        loc_pid = get_yaml(self.yaml_path)["location_pid"]  # type: ignore
        pid_x = PID(**loc_pid["pid_x"])
        pid_x.output_limits = (-speed, speed)
        pid_y = PID(**loc_pid["pid_y"])
        pid_y.output_limits = (-0.15, 0.15)
        # pid_w = PID(1.0, 0, 0.00, setpoint=0, output_limits=(-0.15, 0.15))

        # 用于相同记录结果的计数类
        x_count = CountRecord(2)
        dis_count = CountRecord(2)

        out_x = 0
        out_y = 0

        # 此时设置相对初始位置
        # self.set_pos_relative()
        # self.dis_tra_st = self.get_distance()
        x_st, y_st, _ = self.get_odometry()
        find_tar = False
        tar = []
        for pt_tar in pts_tar:
            # id, 物体宽度，置信度, 归一化bbox[x_c, y_c, w, h]
            tar_id, tar_width, tar_label, tar_score, tar_bbox = (
                pt_tar[0],
                pt_tar[1],
                pt_tar[2],
                pt_tar[3],
                pt_tar[4:],
            )
            tar_width *= 0.001
            tar_x, tar_y, tar_dis = self.det2pose(tar_bbox, tar_width)
            tar.append([tar_id, tar_width, tar_x, tar_y, tar_dis])
        # logger.info("tar x:{} dis:{}".format(tar_x, tar_dis))
        tar_id, tar_width, tar_x, tar_y, tar_dis = tar[0]
        pid_x.setpoint = tar_x
        pid_y.setpoint = tar_dis
        tar_index = 0
        flag_location = False
        while True:
            if self._stop_flag:
                return
            if time.time() > end_time:
                logger.info("time out")
                self.set_velocity(0, 0, 0)
                return False
            _pos_x, _pos_y, _pos_omage = self.get_odometry()  # 用来计算距离

            if abs(_pos_x - x_st) > dis_out or abs(_pos_y - y_st) > dis_out:
                if not find_tar:
                    logger.info("task location dis out")
                    self.set_velocity(0, 0, 0)
                    return False
            img_side = self.cap_front.read()
            dets_ret = infer(img_side)

            img_side_show = img_side.copy()
            for det in dets_ret:
                det_cls_id, det_id, det_label, det_score, det_bbox = (
                    det[0],
                    det[1],
                    det[2],
                    det[3],
                    det[4:],
                )
                x_c, y_c, w, h = det_bbox
                # 将归一化坐标转换为像素坐标
                img_h, img_w = img_side.shape[:2]
                x_c = int((x_c + 1) / 2 * img_w)
                y_c = int((y_c + 1) / 2 * img_h)
                w = int(w * img_w / 2)
                h = int(h * img_h / 2)
                x1 = int(x_c - w / 2)
                y1 = int(y_c - h / 2)
                x2 = int(x_c + w / 2)
                y2 = int(y_c + h / 2)
                # 绘制矩形框
                cv2.rectangle(img_side_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制标签
                label_text = f"{det_label}:{det_score:.2f}"
                cv2.putText(
                    img_side_show,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            self.streamer.update_frame(img_side_show, "cam2")

            # dets_ret = self.mot_hum(img_side)
            # cv2.imshow("side", img_side)
            # cv2.waitKey(1)

            # 进行排序，此处排列按照自中心由近及远的顺序
            dets_ret.sort(key=lambda x: (x[4]) ** 2 + (x[5]) ** 2)
            print(dets_ret)
            # # 找到最近对应的类别，类别存在第一个位置
            # det = self.get_list_by_val(dets_ret, 2, tar_label)

            # 如果没有，就重新获取
            if len(dets_ret) > 0:
                det = dets_ret[0]
                # 结果分解
                det_id, obj_id, det_label, det_score, det_bbox = (
                    det[0],
                    det[1],
                    det[2],
                    det[3],
                    det[4:],
                )
                # if find_tar is False:
                # tar_index = 0
                # for tar_pt in tar:
                for index, tar_pt in enumerate(tar):
                    if det_id == tar_pt[0]:
                        tar_index = index
                        tar_id, tar_width, tar_x, tar_y, tar_dis = tar_pt
                        pid_x.setpoint = tar_x
                        pid_y.setpoint = tar_dis
                        find_tar = True
                        # print("find tar", tar_id)
                        break

                if det_id == tar_id:
                    _x, _y, _dis = self.det2pose(det_bbox, tar_width)
                    out_x = pid_x(_x) * side  # type: ignore
                    out_y = pid_y(_dis) * side  # pyright: ignore[reportOptionalOperand]
                    # out_y = pid_y(_dis)
                    # out_y = pid_w(bbox_error[2])
                    # 检测偏差值连续小于阈值时，跳出循环
                    # print(bbox_error)
                    # print("err x:{:.2}, dis:{:.2}, tar x:{:.2}, tar dis:{:.2}".format(_x, _dis, tar_x, tar_dis))
                    flag_x = x_count(abs(_x - tar_x) < 0.03)
                    flag_dis = dis_count(abs(_dis - tar_dis) < 0.03)
                    if flag_x:
                        out_x = 0
                    if flag_dis:
                        out_y = 0
                    if flag_x and flag_dis:
                        logger.info("location{} ok".format(tar_id))
                        # flag_location = True
                        # 停止
                        self.set_velocity(0, 0, 0)
                        return tar_index

                # print("error_x:{:.2}, error_y:{:.2}, out_x:{:.2}, out_y:{:2}".format(bbox_error[0], bbox_error[2], out_x, out_y))
            else:
                x_count(False)
                dis_count(False)
            self.set_velocity(out_x, out_y, 0)

    def lane_base(self, speed, end_fuction, stop=STOP_PARAM, y_offset=0.0, mode=1):
        """车道保持入口，按 mode 分发到对应的独立函数（各函数在下方，各自完整独立）"""
        if mode == 2:
            self._lane_base_mode2(speed, end_fuction, stop, y_offset)
        elif mode == 3:
            self._lane_base_mode3(speed, end_fuction, stop, y_offset)
        elif mode == 4:
            self._lane_base_mode4(speed, end_fuction, stop, y_offset)
        else:
            self._lane_base_mode1(speed, end_fuction, stop, y_offset)

    def _lane_base_mode1(self, speed, end_fuction, stop=STOP_PARAM, y_offset=0.0):
        """[Mode1] 组合误差 PID（AI角度+CV横偏）-- 完整独立副本，参数在函数内直接修改"""

        # ============================================================
        #  Mode1 独立 PID 参数 —— 直接在此修改！
        #  与其他模式完全独立，互不影响
        # ============================================================
        if not hasattr(self, '_mode1_pid_inited'):
            from smartcar.whalesbot.tools.tools_class import PID
            self._mode1_pid_angle = PID(
                Kp=2.0,              # ← 角度比例系数
                Ki=0.0,              # ← 角度积分系数
                Kd=0.3,              # ← 角度微分系数
                setpoint=0.0,
                output_limits=(-1, 1),
            )
            self._mode1_pid_lat = PID(
                Kp=35.0,             # ← 横向→角度 比例系数
                Ki=0.0,              # ← 横向→角度 积分系数
                Kd=6.0,              # ← 横向→角度 微分系数
                setpoint=0.0,
                output_limits=(-1.2, 1.2),
            )
            self._mode1_pid_inited = True
        # ============================================================

        last_print = time.time()
        while True:
            if self._stop_flag:
                return

            # error_y, error_angle = self.get_lane_results()
            error_y, error_angle = self.get_lane_results( scan_range=(0.6,0.8,30))

            # 横向偏移：y_offset>0 → 车偏右行驶（arc right of centerline）
            # error_y>0 表示车偏右 → 减掉 y_offset = 目标就是偏右 y_offset 米
            error_y_effective = error_y - y_offset

            # 横向偏差不直接用Y轴平移（Mecanum横向移动会抽搐），
            # 而是转成角度偏置，让车通过转向弧线走回目标线
            k_cross = -0.75  # 横向→角度转换系数
            y_speed = 0.0  # 关掉横向平移

            # # 大右拐时丢弃横向偏差：车走外沿弧线，"中线"偏移是假的，强行拉回反而切弯
            # DROP_YERR_RIGHT_THRESH = 0.50  # 右转角度超过此值时 error_y 置零
            # if error_angle < -DROP_YERR_RIGHT_THRESH:
            #     error_y_effective = 0.0

            # ====== 弯道速度策略（可调参数） ======
            # 通用弯道减速：角度越大越慢
            CORNER_DECEL_START = 0.6    # abs_ang 超过此值开始减速
            CORNER_DECEL_MAX   = 1.2    # abs_ang 达到此值时减速到最低
            CORNER_SPEED_MIN   = 0.85    # 最弯时的最低速度比例（相对 speed）
            # 右转大弯额外减速（渐进式）
            RIGHT_EXTRA_START  = 0.45   # 右转角度超过此值触发额外减速
            RIGHT_EXTRA_MAX    = 1.0   # 右转角度达到此值时额外减速最大
            RIGHT_EXTRA_MIN    = 0.75   # 右转额外减速的最低速度比例
            # ==========================================

            abs_ang = abs(error_angle)

            # 直道置信度（用于 k_cross 加权，保留原逻辑）
            straightness = max(0.5, min(1.0, 1.0 - (abs_ang - CORNER_DECEL_START) / 0.4))
            k_cross_eff = k_cross * straightness

            # # 方案A：组合误差（AI角度 + CV横偏）
            # combined = (-error_angle + k_cross_eff * error_y_effective)
            # combined = (-error_y_effective)
            # if error_angle < -0.1:
            #     if error_angle < -0.8 :
            #         combined = (-0.4 - error_y_effective)
            #     else:
            #         combined = (-0.5*error_angle - error_y_effective)
            # else:
            combined = (-error_y_effective)
            angle_speed = self._mode1_pid_angle(combined)

            # 方案B：纯横偏PID（不依赖AI角度）
            # LAT_I_CLEAR = 0.025       # abs(error_y) < 此值(m)时启动积分衰减
            # LAT_I_DECAY_STEP = 0.005 # 每帧积分衰减步长，越小越平滑
            # LAT_I_LIMIT = 0.01       # 积分项硬上限（绝对值），防止积分无限累积
            # if abs(error_y_effective) < LAT_I_CLEAR:
            #     i_val = self.lane_pid_lat._integral
            #     if i_val > 0:
            #         self.lane_pid_lat._integral = max(0.0, i_val - LAT_I_DECAY_STEP)
            #     elif i_val < 0:
            #         self.lane_pid_lat._integral = min(0.0, i_val + LAT_I_DECAY_STEP)
            # if error_y_effective < 0:
            #     error_y_effective_1 = -error_y_effective
            # else:
            #     error_y_effective_1 = error_y_effective
            # angle_speed = -self.lane_pid_lat(error_y_effective_1*error_y_effective)
            # if abs(self.lane_pid_lat._integral) > LAT_I_LIMIT:
            #     self.lane_pid_lat._integral = LAT_I_LIMIT if self.lane_pid_lat._integral > 0 else -LAT_I_LIMIT

            # 阶段三：弯道减速   减少行驶速度
            # 通用弯道减速：abs_ang 从 CORNER_DECEL_START 到 CORNER_DECEL_MAX
            # 速度平滑从 speed 降到 speed * CORNER_SPEED_MIN
            if abs_ang <= CORNER_DECEL_START:
                speed_cur = speed
            else:
                t_corner = min(1.0, (abs_ang - CORNER_DECEL_START) / (CORNER_DECEL_MAX - CORNER_DECEL_START))
                speed_cur = speed * (1.0 - t_corner * (1.0 - CORNER_SPEED_MIN))

            # 阶段四：右转大弯   额外平滑减速
            # 右转大弯额外平滑减速
            # error_angle 从 -RIGHT_EXTRA_START 到 -RIGHT_EXTRA_MAX
            # 速度系数从 1.0 平滑降到 RIGHT_EXTRA_MIN
            if error_angle < -RIGHT_EXTRA_START:
                t_right = min(1.0, (abs(error_angle) - RIGHT_EXTRA_START) / (RIGHT_EXTRA_MAX - RIGHT_EXTRA_START))
                speed_cur *= (1.0 - t_right * (1.0 - RIGHT_EXTRA_MIN))

            # 阶段五：速度输出
            # self.set_velocity(speed, y_speed, angle_speed)
            self.set_velocity(speed_cur, y_speed, angle_speed)
            # self.set_velocity(0, 0, 0)

            # 每 0.3 秒输出一次误差，并保存带标注的调试图
            now = time.time()
            if now - last_print > 0.1:
                print(f"[巡线-Mode1] y_err={error_y:+.4f}  error_angle={error_angle:.4f}  |  str={straightness:.2f}  spd={speed_cur:.2f}  ang={angle_speed:+.3f}  dist={self.get_distance():.2f}m")
                last_print = now

                # 保存调试图（带巡线信息标注）
                if not hasattr(self, '_lane_dbg_cnt'):
                    self._lane_dbg_cnt = 0
                self._lane_dbg_cnt += 1
                if self._save_lane_debug_images and self._lane_dbg_cnt % 1 == 0:
                    dbg_img = getattr(self, '_last_cv_debug_img', self.cap_side.read()).copy()
                    i_term = getattr(self._mode1_pid_lat, '_integral', 0.0)
                    lines = [
                        f"y_err:{error_y:+.4f}",
                        f"a_err:{error_angle:+.4f}",
                        f"straightness:{straightness:.2f}",
                        f"speed:{speed_cur:.2f}",
                        f"ang_spd:{angle_speed:+.3f}",
                        f"I_int:{i_term:+.5f}",
                        f"dist:{self.get_distance():.2f}m",
                    ]
                    for i, line in enumerate(lines):
                        y_pos = 30 + i * 25
                        cv2.putText(dbg_img, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(dbg_img, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    save_dir = os.path.join(self.path_dir, "lane_debug")
                    os.makedirs(save_dir, exist_ok=True)
                    img_name = f"mode1_{self._lane_dbg_cnt:04d}.jpg"
                    cv2.imwrite(os.path.join(save_dir, img_name), dbg_img)
                    print(f"  [已保存] {os.path.join(save_dir, img_name)}")

            if end_fuction():
                break
            time.sleep(0.02)  # 控制频率~30Hz，防止更新过快引发振荡
        if stop:
            self.stop()

    def _lane_base_mode2(self, speed, end_fuction, stop=STOP_PARAM, y_offset=0.0):
        """[Mode2] 纯AI角度 PID -- 完整独立副本，自由修改"""
        
        last_print = time.time()
        while True:
            if self._stop_flag:
                return

            error_y, error_angle = self.get_lane_results()

            # 横向偏移：y_offset>0 → 车偏右行驶（arc right of centerline）
            # error_y>0 表示车偏右 → 减掉 y_offset = 目标就是偏右 y_offset 米
            error_y_effective = error_y - y_offset

            # 横向偏差不直接用Y轴平移（Mecanum横向移动会抽搐），
            # 而是转成角度偏置，让车通过转向弧线走回目标线
            k_cross = -1.25  # 横向→角度转换系数
            y_speed = 0.0  # 关掉横向平移

            # 大右拐时丢弃横向偏差：车走外沿弧线，"中线"偏移是假的，强行拉回反而切弯
            DROP_YERR_RIGHT_THRESH = 0.50  # 右转角度超过此值时 error_y 置零
            if error_angle < -DROP_YERR_RIGHT_THRESH:
                error_y_effective = 0.0

            # ====== 弯道速度策略（可调参数） ======
            # 通用弯道减速：角度越大越慢
            CORNER_DECEL_START = 0.6    # abs_ang 超过此值开始减速
            CORNER_DECEL_MAX   = 1.2    # abs_ang 达到此值时减速到最低
            CORNER_SPEED_MIN   = 0.85    # 最弯时的最低速度比例（相对 speed）
            # 右转大弯额外减速（渐进式）
            RIGHT_EXTRA_START  = 0.40   # 右转角度超过此值触发额外减速
            RIGHT_EXTRA_MAX    = 1.0   # 右转角度达到此值时额外减速最大
            RIGHT_EXTRA_MIN    = 0.3   # 右转额外减速的最低速度比例
            # ==========================================

            abs_ang = abs(error_angle)

            # 直道置信度（用于 k_cross 加权，保留原逻辑）
            straightness = max(0.5, min(1.0, 1.0 - (abs_ang - CORNER_DECEL_START) / 0.4))
            k_cross_eff = k_cross * straightness

            # 方案A：组合误差（AI角度 + CV横偏）
            # combined = (-error_angle + k_cross_eff * error_y_effective)
            # angle_speed = self.lane_pid.pid_angle(combined)

            # 方案B：纯横偏PID（不依赖AI角度）
            LAT_I_CLEAR = 0.025       # abs(error_y) < 此值(m)时启动积分衰减
            LAT_I_DECAY_STEP = 0.005 # 每帧积分衰减步长，越小越平滑
            LAT_I_LIMIT = 0.01       # 积分项硬上限（绝对值），防止积分无限累积
            if abs(error_y_effective) < LAT_I_CLEAR:
                i_val = self.lane_pid_lat._integral
                if i_val > 0:
                    self.lane_pid_lat._integral = max(0.0, i_val - LAT_I_DECAY_STEP)
                elif i_val < 0:
                    self.lane_pid_lat._integral = min(0.0, i_val + LAT_I_DECAY_STEP)
            if error_y_effective < 0:
                error_y_effective_1 = -error_y_effective
            else:
                error_y_effective_1 = error_y_effective
            angle_speed = -self.lane_pid_lat(error_y_effective_1*error_y_effective)
            if abs(self.lane_pid_lat._integral) > LAT_I_LIMIT:
                self.lane_pid_lat._integral = LAT_I_LIMIT if self.lane_pid_lat._integral > 0 else -LAT_I_LIMIT

            # 阶段三：弯道减速   减少行驶速度
            # 通用弯道减速：abs_ang 从 CORNER_DECEL_START 到 CORNER_DECEL_MAX
            # 速度平滑从 speed 降到 speed * CORNER_SPEED_MIN
            if abs_ang <= CORNER_DECEL_START:
                speed_cur = speed
            else:
                t_corner = min(1.0, (abs_ang - CORNER_DECEL_START) / (CORNER_DECEL_MAX - CORNER_DECEL_START))
                speed_cur = speed * (1.0 - t_corner * (1.0 - CORNER_SPEED_MIN))

            # 阶段四：右转大弯   额外平滑减速
            # 右转大弯额外平滑减速
            # error_angle 从 -RIGHT_EXTRA_START 到 -RIGHT_EXTRA_MAX
            # 速度系数从 1.0 平滑降到 RIGHT_EXTRA_MIN
            if error_angle < -RIGHT_EXTRA_START:
                t_right = min(1.0, (abs(error_angle) - RIGHT_EXTRA_START) / (RIGHT_EXTRA_MAX - RIGHT_EXTRA_START))
                speed_cur *= (1.0 - t_right * (1.0 - RIGHT_EXTRA_MIN))

            # 阶段五：速度输出
            # self.set_velocity(speed, y_speed, angle_speed)
            self.set_velocity(speed_cur, y_speed, angle_speed)
            # self.set_velocity(0, 0, 0)

            # 每 0.3 秒输出一次误差，并保存带标注的调试图
            now = time.time()
            if now - last_print > 0.1:
                print(f"[巡线] y_err={error_y:+.4f}  error_angle={error_angle:.4f}  |  str={straightness:.2f}  spd={speed_cur:.2f}  ang={angle_speed:+.3f}  dist={self.get_distance():.2f}m")
                last_print = now

                # 保存调试图（带巡线信息标注）
                if not hasattr(self, '_lane_dbg_cnt'):
                    self._lane_dbg_cnt = 0
                self._lane_dbg_cnt += 1
                if self._save_lane_debug_images and self._lane_dbg_cnt % 1 == 0:
                    dbg_img = getattr(self, '_last_cv_debug_img', self.cap_side.read()).copy()
                    i_term = getattr(self.lane_pid_lat, '_integral', 0.0)
                    lines = [
                        f"y_err:{error_y:+.4f}",
                        f"a_err:{error_angle:+.4f}",
                        f"straightness:{straightness:.2f}",
                        f"speed:{speed_cur:.2f}",
                        f"ang_spd:{angle_speed:+.3f}",
                        f"I_int:{i_term:+.5f}",
                        f"dist:{self.get_distance():.2f}m",
                    ]
                    for i, line in enumerate(lines):
                        y_pos = 30 + i * 25
                        cv2.putText(dbg_img, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(dbg_img, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    save_dir = os.path.join(self.path_dir, "lane_debug")
                    os.makedirs(save_dir, exist_ok=True)
                    img_name = f"mode2_{self._lane_dbg_cnt:04d}.jpg"
                    cv2.imwrite(os.path.join(save_dir, img_name), dbg_img)
                    print(f"  [已保存] {os.path.join(save_dir, img_name)}")

            if end_fuction():
                break
            time.sleep(0.02)  # 控制频率~30Hz，防止更新过快引发振荡
        if stop:
            self.stop()

    def _lane_base_mode3(self, speed, end_fuction, stop=STOP_PARAM, y_offset=0.0):
        """[Mode3] 纯CV横偏 PID -- 完整独立副本，参数在函数内直接修改"""

        # ============================================================
        #  Mode3 独立 PID 参数 —— 直接在此修改！
        #  与其他模式完全独立，互不影响
        # ============================================================
        if not hasattr(self, '_mode3_pid_inited'):
            from smartcar.whalesbot.tools.tools_class import PID
            self._mode3_pid_angle = PID(
                Kp=3.0,              # ← 角度比例系数
                Ki=0.0,              # ← 角度积分系数
                Kd=0.0,              # ← 角度微分系数
                setpoint=0.0,
                output_limits=(-1.5, 1.5),
            )
            self._mode3_pid_lat = PID(
                Kp=35.0,             # ← 横向→角度 比例系数
                Ki=0.0,              # ← 横向→角度 积分系数
                Kd=6.0,              # ← 横向→角度 微分系数
                setpoint=0.0,
                output_limits=(-1.2, 1.2),
            )
            self._mode3_pid_inited = True
        # ============================================================

        last_print = time.time()
        while True:
            if self._stop_flag:
                return

            # error_y, error_angle = self.get_lane_results()
            error_y, error_angle = self.get_lane_results(cv_mode=2,scan_range=(0.75, 0.85, 30))
            
            # 横向偏移：y_offset>0 → 车偏右行驶（arc right of centerline）
            # error_y>0 表示车偏右 → 减掉 y_offset = 目标就是偏右 y_offset 米
            error_y_effective = error_y - y_offset

            # 横向偏差不直接用Y轴平移（Mecanum横向移动会抽搐），
            # 而是转成角度偏置，让车通过转向弧线走回目标线
            k_cross = -0.75  # 横向→角度转换系数
            y_speed = 0.0  # 关掉横向平移

            # # 大右拐时丢弃横向偏差：车走外沿弧线，"中线"偏移是假的，强行拉回反而切弯
            # DROP_YERR_RIGHT_THRESH = 0.50  # 右转角度超过此值时 error_y 置零
            # if error_angle < -DROP_YERR_RIGHT_THRESH:
            #     error_y_effective = 0.0

            # ====== 弯道速度策略（可调参数） ======
            # 通用弯道减速：角度越大越慢
            CORNER_DECEL_START = 0.6    # abs_ang 超过此值开始减速
            CORNER_DECEL_MAX   = 1.2    # abs_ang 达到此值时减速到最低
            CORNER_SPEED_MIN   = 0.85    # 最弯时的最低速度比例（相对 speed）
            # 右转大弯额外减速（渐进式）
            RIGHT_EXTRA_START  = 0.3   # 右转角度超过此值触发额外减速
            RIGHT_EXTRA_MAX    = 1.2   # 右转角度达到此值时额外减速最大
            RIGHT_EXTRA_MIN    = 0.8   # 右转额外减速的最低速度比例
            # ==========================================

            abs_ang = abs(error_angle)

            # 直道置信度（用于 k_cross 加权，保留原逻辑）
            straightness = max(0.5, min(1.0, 1.0 - (abs_ang - CORNER_DECEL_START) / 0.4))
            k_cross_eff = k_cross * straightness

            # # 方案A：组合误差（AI角度 + CV横偏）
            # combined = (-error_angle + k_cross_eff * error_y_effective)
            # combined = (-error_y_effective)
            
            if error_angle < -0.5:
                if error_y_effective > 0.1:
                    combined = (-0.7*error_angle)
                else :
                    combined = (-0.7*error_angle - error_y_effective)
            else:
                combined = (-error_y_effective)
            angle_speed = self._mode3_pid_angle(combined)

            # 方案B：纯横偏PID（不依赖AI角度）
            # LAT_I_CLEAR = 0.025       # abs(error_y) < 此值(m)时启动积分衰减
            # LAT_I_DECAY_STEP = 0.005 # 每帧积分衰减步长，越小越平滑
            # LAT_I_LIMIT = 0.01       # 积分项硬上限（绝对值），防止积分无限累积
            # if abs(error_y_effective) < LAT_I_CLEAR:
            #     i_val = self.lane_pid_lat._integral
            #     if i_val > 0:
            #         self.lane_pid_lat._integral = max(0.0, i_val - LAT_I_DECAY_STEP)
            #     elif i_val < 0:
            #         self.lane_pid_lat._integral = min(0.0, i_val + LAT_I_DECAY_STEP)
            # if error_y_effective < 0:
            #     error_y_effective_1 = -error_y_effective
            # else:
            #     error_y_effective_1 = error_y_effective
            # angle_speed = -self.lane_pid_lat(error_y_effective_1*error_y_effective)
            # if abs(self.lane_pid_lat._integral) > LAT_I_LIMIT:
            #     self.lane_pid_lat._integral = LAT_I_LIMIT if self.lane_pid_lat._integral > 0 else -LAT_I_LIMIT

            # 阶段三：弯道减速   减少行驶速度
            # 通用弯道减速：abs_ang 从 CORNER_DECEL_START 到 CORNER_DECEL_MAX
            # 速度平滑从 speed 降到 speed * CORNER_SPEED_MIN
            if abs_ang <= CORNER_DECEL_START:
                speed_cur = speed
            else:
                t_corner = min(1.0, (abs_ang - CORNER_DECEL_START) / (CORNER_DECEL_MAX - CORNER_DECEL_START))
                speed_cur = speed * (1.0 - t_corner * (1.0 - CORNER_SPEED_MIN))

            # 阶段四：右转大弯   额外平滑减速
            # 右转大弯额外平滑减速
            # error_angle 从 -RIGHT_EXTRA_START 到 -RIGHT_EXTRA_MAX
            # 速度系数从 1.0 平滑降到 RIGHT_EXTRA_MIN
            # if error_angle < -RIGHT_EXTRA_START:
            #     t_right = min(1.0, (abs(error_angle) - RIGHT_EXTRA_START) / (RIGHT_EXTRA_MAX - RIGHT_EXTRA_START))
            #     speed_cur *= (1.0 - t_right * (1.0 - RIGHT_EXTRA_MIN))

            # 阶段五：速度输出
            # self.set_velocity(speed, y_speed, angle_speed)
            self.set_velocity(speed, y_speed, angle_speed)
            # self.set_velocity(0, 0, 0)

            # 每 0.3 秒输出一次误差，并保存带标注的调试图
            now = time.time()
            if now - last_print > 0.1:
                print(f"[巡线] y_err={error_y:+.4f}  error_angle={error_angle:.4f}  |  str={straightness:.2f}  spd={speed_cur:.2f}  ang={angle_speed:+.3f}  dist={self.get_distance():.2f}m")
                last_print = now

                # 保存调试图（带巡线信息标注）
                if not hasattr(self, '_lane_dbg_cnt'):
                    self._lane_dbg_cnt = 0
                self._lane_dbg_cnt += 1
                if self._save_lane_debug_images and self._lane_dbg_cnt % 1 == 0:
                    dbg_img = getattr(self, '_last_cv_debug_img', self.cap_side.read()).copy()
                    i_term = getattr(self._mode3_pid_lat, '_integral', 0.0)
                    lines = [
                        f"y_err:{error_y:+.4f}",
                        f"a_err:{error_angle:+.4f}",
                        f"straightness:{straightness:.2f}",
                        f"speed:{speed_cur:.2f}",
                        f"ang_spd:{angle_speed:+.3f}",
                        f"I_int:{i_term:+.5f}",
                        f"dist:{self.get_distance():.2f}m",
                    ]
                    for i, line in enumerate(lines):
                        y_pos = 30 + i * 25
                        cv2.putText(dbg_img, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(dbg_img, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    # save_dir = os.path.join(os.path.dirname(__file__), "lane_debug")
                    # os.makedirs(save_dir, exist_ok=True)
                    # img_name = f"{self._lane_dbg_cnt:04d}.jpg"
                    save_dir = os.path.join(self.path_dir, "lane_debug")
                    os.makedirs(save_dir, exist_ok=True)
                    img_name = f"mode3_{self._lane_dbg_cnt:04d}.jpg"
                    cv2.imwrite(os.path.join(save_dir, img_name), dbg_img)
                    print(f"  [已保存] {os.path.join(save_dir, img_name)}")

            if end_fuction():
                break
            time.sleep(0.02)  # 控制频率~30Hz，防止更新过快引发振荡
        if stop:
            self.stop()

    # ============================================================
    # [Mode4] 纯AI模型巡线
    # 误差和角度全部来自AI模型(self.crusie)，使用无死区PID
    # 用法：my_car.lane_dis_offset(speed=0.3, dis_hold=99, mode=4)
    # ============================================================
    def _get_lane_pure_model(self):
        """纯AI模型获取巡线结果：误差+角度均来自模型，经PID输出速度"""
        image_original = self.cap_side.read().copy()

        # ====== 自动曝光处理（暂时关闭） ======
        # image = self.ae_controller.apply(image_original.copy())
        image = image_original.copy()

        res = self.crusie(image)
        error, angle = res[0], res[1]

        # ====== 禁止左拐/右拐开关 ======
        # val=1: 正值(左拐)→截断为 -0.0006
        # val=2: 负值(右拐)→截断为 +0.0006
        if self._disable_left_turn == 1 and angle > 0:
            angle = -0.0006
        elif self._disable_left_turn == 2 and angle < 0:
            angle = +0.0006

        y_speed, angle_speed = self.lane_pid_pure.get_out(-error, -angle)
        return error, angle, y_speed, angle_speed, image_original, image

    def _lane_base_mode4(self, speed, end_fuction, stop=STOP_PARAM, y_offset=0.0):
        """[Mode4] 纯AI模型PID -- 完整独立副本，自由修改"""
        last_print = time.time()
        frame_cnt = 0
        while True:
            if self._stop_flag:
                return

            error_y, error_angle, y_speed, angle_speed, image_original, image = \
                self._get_lane_pure_model()
            self.set_velocity(speed, y_speed, angle_speed)

            # 终端打印
            frame_cnt += 1
            now = time.time()
            if now - last_print > 0.1:
                print(f"[巡线-纯模型] frame:{frame_cnt} "
                      f"model_error:{error_y:+.4f} model_angle:{error_angle:+.4f} "
                      f"pid_y:{y_speed:+.4f} pid_a:{angle_speed:+.4f} "
                      f"dist:{self.get_distance():.2f}m")
                last_print = now

            # ====== 拍照：巡线时保存照片并标注模型输出误差 ======
            if self._save_lane_photo and frame_cnt % self._lane_photo_interval == 0:
                photo = image_original.copy()
                h_ph, w_ph = photo.shape[:2]
                # 顶部深色标题栏，避免文字与画面重叠看不清
                bar_h = 70
                header = np.zeros((bar_h, w_ph, 3), dtype=np.uint8)
                header[:] = (40, 40, 40)
                info = [
                    f"mode:4 | frame:{frame_cnt} | dist:{self.get_distance():.2f}m",
                    f"model_error:{error_y:+.4f}  model_angle:{error_angle:+.4f}",
                    f"pid_y:{y_speed:+.4f}  pid_a:{angle_speed:+.4f}",
                ]
                for i, line in enumerate(info):
                    cv2.putText(header, line, (10, 20 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 255, 0), 1, cv2.LINE_AA)
                photo = np.vstack([header, photo])
                photo_dir = os.path.join(self.path_dir, "lane_photo")
                os.makedirs(photo_dir, exist_ok=True)
                cv2.imwrite(os.path.join(photo_dir, f"photo_{frame_cnt:05d}.jpg"), photo)

            # 每 2 帧保存曝光前后对比图 + 误差信息
            if self._save_lane_debug_images and frame_cnt % 2 == 0:
                h, w = image.shape[:2]
                # 曝光前原图（如果尺寸不同则 resize）
                original = image_original.copy()
                if original.shape[:2] != (h, w):
                    original = cv2.resize(original, (w, h))
                # 左右拼接：左=曝光前，右=曝光后
                comparison = np.hstack([original, image])

                # 顶部标题栏
                bar_h = 35
                header = np.zeros((bar_h, w * 2, 3), dtype=np.uint8)
                header[:] = (40, 40, 40)

                info = [
                    f"mode:4 | frame:{frame_cnt} | dist:{self.get_distance():.2f}m",
                    f"error:{error_y:+.4f}  angle:{error_angle:+.4f}  pid_y:{y_speed:+.4f}  pid_a:{angle_speed:+.4f}",
                ]
                for i, line in enumerate(info):
                    cv2.putText(header, line, (10, 12 + i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)

                # 标签：左下角 "原始" / 右下角 "自动曝光"
                label_y = h - 10
                cv2.putText(comparison, "yuan shi", (10, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(comparison, "AE", (w + 10, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)

                # 中间分隔线
                cv2.line(comparison, (w, 0), (w, h), (0, 255, 255), 2)

                # 拼接标题栏 + 对比图
                output = np.vstack([header, comparison])

                save_dir = os.path.join(self.path_dir, "lane_debug")
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f"ae_cmp_{frame_cnt:05d}.jpg"), output)

            if end_fuction():
                break
        if stop:
            self.stop()


    # def lane_det_base(self, speed, end_fuction, stop=STOP_PARAM):
    #     """
    #     目标检测基础方法

    #     使用前置摄像头进行目标检测，根据检测结果调整车辆方向。

    #     参数:
    #         speed: 行驶速度
    #         end_fuction: 结束条件函数，接收距离参数，返回True时停止
    #         stop: 是否在结束后停止车辆，默认为STOP_PARAM
    #     """
    #     # 初始化速度和角度速度
    #     y_speed = 0
    #     angle_speed = 0
    #     w_r=0.06
    #     # 无限循环
    #     while True:
    #         # 读取前摄像头图像
    #         image = self.cap_front.read()
    #         self.streamer.update_frame(image,"cam1")
    #         dets_ret = self.front_det(image)
    #         # 此处检测简单不需要排序
    #         # dets_ret.sort(key=lambda x: x[4]**2 + (x[5])**2)
    #         if len(dets_ret)>0:
    #             det = dets_ret[0]
    #             det_cls, det_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
    #             _x, _y, _dis = self.det2pose(det_bbox, w_r)
    #             # error_y = det_bbox[0]
    #             # dis_x = 1 - det_bbox[1]
    #             if end_fuction(_dis):
    #                 break
    #             error_angle = _x /_dis
    #             y_speed, angle_speed = self.det_pid.get_out(_x, error_angle)
    #             # print("_x:{:.2}, _angle:{:.2}, y_vel:{:.2}, angle_vel:{:.2}, dis{:.2}".format(_x, error_angle, y_speed, angle_speed, _dis))
    #         self.set_velocity(speed, y_speed, angle_speed)
    #         # if end_fuction(0):
    #         #     break
    #     if stop:
    #         self.stop()

    # def lane_det_time(self, speed, time_dur, stop=STOP_PARAM):
    #     """
    #     目标检测定时方法

    #     使用前置摄像头进行目标检测，持续指定的时间。

    #     参数:
    #         speed: 行驶速度
    #         time_dur: 持续时间（秒）
    #         stop: 是否在结束后停止车辆，默认为STOP_PARAM
    #     """
    #     time_end = time.time() + time_dur
    #     end_fuction = lambda x: time.time() > time_end
    #     self.lane_det_base(speed, end_fuction, stop=stop)

    # def lane_det_dis2pt(self, speed, dis_end, stop=STOP_PARAM):
    #     """
    #     目标检测定距方法

    #     使用前置摄像头进行目标检测，直到与目标的距离小于指定值。

    #     参数:
    #         speed: 行驶速度
    #         dis_end: 目标距离阈值
    #         stop: 是否在结束后停止车辆，默认为STOP_PARAM
    #     """
    #     # lambda定义endfunction
    #     end_fuction = lambda x: x < dis_end and x != 0
    #     self.lane_det_base(speed, end_fuction, stop=stop)

    def lane_time(self, speed, time_dur, stop=STOP_PARAM, mode=1, y_offset=0.0):
        time_end = time.time() + time_dur
        def end_fuction():
            return time.time() > time_end
        self.lane_base(speed, end_fuction, stop=stop, mode=mode, y_offset=y_offset)

    def lane_dis(self, speed, dis_end, stop=STOP_PARAM, mode=1, y_offset=0.0):
        def end_fuction():
            return self.get_distance() > dis_end
        self.lane_base(speed, end_fuction, stop=stop, mode=mode, y_offset=y_offset)

    def lane_dis_offset(self, speed, dis_hold, stop=STOP_PARAM, mode=1, y_offset=0.0):
        dis_start = self.get_distance()
        dis_stop = dis_start + dis_hold
        print(f"[lane_dis_offset] 开始行驶, 起点距离: {dis_start:.3f}m, "
              f"目标行驶: {dis_hold:.3f}m, 终点距离: {dis_stop:.3f}m, mode={mode}")
        self.lane_dis(speed, dis_stop, stop=stop, mode=mode, y_offset=y_offset)
        dis_end = self.get_distance()
        dis_traveled = dis_end - dis_start
        print(f"[lane_dis_offset] 行驶结束, 终点距离: {dis_end:.3f}m, "
              f"实际行驶: {dis_traveled:.3f}m (目标: {dis_hold:.3f}m)")

    def align_lane_heading(self, duration=5.0):
        """
        原地旋转校准：只扫描摄像头底部3行近场车道线，
        用AI模型获取角度偏差，PID控制车体原地旋转对齐赛道中线。

        Args:
            duration: 最长校准时间(秒)，可调
        """
        # ====== 旋转对准PID参数（可调） ======
        ANGLE_THRESHOLD = 0.01     # 横向误差阈值(m)，低于此值视为已对准
        COUNT_HOLD = 15              # 连续对准帧数，达到后退出
        MAX_ANGLE_SPEED = 0.6        # 旋转速度上限(rad/s)，防止转太快
        SCAN_RANGE = (0.65,0.75,30) # 扫线范围(底部)，可调
        KP_ANGLE = -3.25               # 旋转PID：比例系数
        KI_ANGLE = -0.0               # 旋转PID：积分系数（原地旋转稳态误差小，通常不需要）
        KD_ANGLE = -0.5               # 旋转PID：微分系数（抑制过冲时可加入）
        DRIVE_SPEED = 0.1            # 直行速度(m/s)，0=原地旋转，设正值=边前进边调整角度
        # =================================

        # 初始化旋转PID
        pid_align = PID(Kp=KP_ANGLE, Ki=KI_ANGLE, Kd=KD_ANGLE,
                        setpoint=0.0, output_limits=(-MAX_ANGLE_SPEED, MAX_ANGLE_SPEED))

        print(f"\n[车道对准] 开始原地旋转校准, 最长{duration:.0f}秒")
        print(f"  扫线范围=底部{SCAN_RANGE[0]:.0%}~{SCAN_RANGE[1]:.0%}, "
              f"阈值={ANGLE_THRESHOLD:.5f}m, PID=({KP_ANGLE},{KI_ANGLE},{KD_ANGLE})")

        count_aligned = 0
        time.sleep(0.2)
        self.set_velocity(0, 0, 0)
        time.sleep(0.2)
        time_start = time.time()
        last_print = time_start
        frame = 0

        try:
            while time.time() - time_start < duration:
                frame += 1
                if self._stop_flag:
                    break

                # 读图（不裁剪，通过scan_range控制扫描范围）
                image = self.cap_side.read().copy()

                # CV扫线：只看底部近场，获取横向偏差
                error_lat = self._get_lane_error_cv(image, scan_range=SCAN_RANGE,center_bias=0.0)
                debug_img = (
                    self._last_cv_debug_img.copy()
                    if self._save_lane_debug_images else None
                )

                lat_abs = abs(error_lat)

                if lat_abs < ANGLE_THRESHOLD:
                    count_aligned += 1
                else:
                    count_aligned = 0

                # PID控制旋转：error_lat>0=偏右→需左转(负角速度), error_lat<0=偏左→需右转(正角速度)
                angle_speed = -pid_align(error_lat)

                if debug_img is not None:
                    # 在调试图上标注信息并保存。比赛配置默认关闭，避免逐帧写盘。
                    cv2.putText(debug_img, f"frame:{frame} lat:{error_lat:+.4f} yellow:{self._last_yellow_px}px",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(debug_img, f"speed:{angle_speed:+.3f} aligned:{count_aligned}/{COUNT_HOLD}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    debug_dir = os.path.join(self.path_dir, "lane_align_debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(f"{debug_dir}/frame_{frame:04d}.jpg", debug_img)

                if count_aligned >= COUNT_HOLD:
                    self.set_velocity(0, 0, 0)
                    print(f"  ✅ 对准完成! (frame={frame}, lat={error_lat:+.4f})")
                    break

                self.set_velocity(DRIVE_SPEED, 0, angle_speed)
                time.sleep(0.02)

                now = time.time()
                if now - last_print > 0.1:
                    print(f"  [frame {frame}] lat={error_lat:+.4f}, "
                          f"speed={angle_speed:+.3f}, aligned={count_aligned}/{COUNT_HOLD}, yellow={self._last_yellow_px}px")
                    last_print = now

        finally:
            time.sleep(0.02)
            self.set_velocity(0, 0, 0)
            elapsed = time.time() - time_start
            forward_dist = DRIVE_SPEED * elapsed
            print(f"[车道对准] 结束, 前进了 {forward_dist:.4f}m, 开始纯后退返回...")
            # self.move_for([-forward_dist, 0.0, 0.0], max_velocities=[0.15, 0.0, 0.0])
            self.move_distance([-0.1,0,0], dis=(forward_dist-0.04))
            print(f"[车道对准] 已退回 {forward_dist:.4f}m\n")

    # def lane_sensor(self, speed, value_h=None, value_l=None, dis_offset=0.0, times=1, sides=1, stop=STOP_PARAM):
    #     """
    #     车道保持传感器方法

    #     使用前置摄像头进行车道保持，直到传感器检测到指定范围的值。

    #     参数:
    #         speed: 行驶速度
    #         value_h: 传感器上限值，默认为1200
    #         value_l: 传感器下限值，默认为0
    #         dis_offset: 距离偏移量，默认为0.0
    #         times: 重复次数，默认为1
    #         sides: 传感器选择，1为左侧，-1为右侧
    #         stop: 是否在结束后停止车辆，默认为STOP_PARAM
    #     """
    #     if value_h is None:
    #         value_h = 1200
    #     if value_l is None:
    #         value_l = 0
    #     # _sensor_usr = self.left_sensor
    #     # if sides == -1:
    #     #     _sensor_usr = self.right_sensor
    #     # 用于检测开始过渡部分的标记
    #     flag_start = False
    #     def end_fuction():
    #         nonlocal flag_start
    #         # val_sensor = _sensor_usr.read()
    #         # print("val:", val_sensor)
    #         if val_sensor < value_h and val_sensor > value_l:
    #             return flag_start
    #         else:
    #             flag_start = True
    #             return False

    #     for i in range(times):
    #         self.lane_base(speed, end_fuction, stop=False)
    #     # 根据需要是否巡航
    #     self.lane_dis_offset(speed, dis_offset, stop=stop)

    # def get_card_side(self):
    #     """
    #     检测卡片左右指示

    #     使用前置摄像头检测卡片上的左右指示，返回相应的方向。

    #     返回:
    #         int: -1表示右转，1表示左转，0表示停止或未检测到
    #     """
    #     # 检测卡片左右指示
    #     count_side = CountRecord(3)
    #     while True:
    #         if self._stop_flag:
    #             return 0
    #         image = self.cap_front.read()
    #         dets_ret = self.front_det(image)
    #         if len(dets_ret) == 0:
    #             count_side(-1)
    #             continue
    #         det = dets_ret[0]
    #         det_cls, det_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
    #         # 联系检测超过3次
    #         if count_side(det_label):
    #             if det_label == 'turn_right':
    #                 return -1
    #             elif det_label == 'turn_left':
    #                 return 1
    def get_det_ocr(self, det, label="name", time_out=5.0):
        time_stop = time.time() + time_out
        # 简单滤波,三次检测到相同的值，认为稳定并返回
        text_count = CountRecord(3)
        text_out = None
        print(det)
        while True:
            if self._stop_flag:
                return text_out
            if time.time() > time_stop:
                return text_out
            img = self.cap_front.read()
            if det is not None:
                det_cls_id, det_id, det_label, det_score, det_bbox = (
                    det[0],
                    det[1],
                    det[2],
                    det[3],
                    det[4:],
                )
                if label is not None:
                    flag = det_label == label
                else:
                    flag = det_label == "order" or det_label == "name"
                if flag:
                    # x1, y1, w, h = det_bbox
                    # # print(img.shape)
                    # # print(x1, y1, w, h)
                    # x1 = img.shape[1] * (1+x1) / 2 - img.shape[1] * w / 4
                    # x2 = x1 + img.shape[1] * w / 2
                    # y1 = img.shape[0] * (1+y1) / 2 - img.shape[0] * h / 4
                    # y2 = y1 + img.shape[0] * h / 2
                    # x1 = 0 if x1 < 0 else int(x1)
                    # x2 = img.shape[1] if x2 > img.shape[1] else int(x2)
                    # y1 = 0 if y1 < 0 else int(y1)
                    # y2 = img.shape[0] if y2 > img.shape[0] else int(y2)
                    # # print(x1, x2, y1, y2)

                    # 将归一化坐标转换为像素坐标
                    x_c, y_c, w, h = det_bbox
                    w *= 1.2
                    h *= 1.2
                    img_h, img_w = img.shape[:2]
                    x_c = int((x_c + 1) / 2 * img_w)
                    y_c = int((y_c + 1) / 2 * img_h)
                    w = int(w * img_w / 2)
                    h = int(h * img_h / 2)
                    x1 = int(x_c - w / 2)
                    y1 = int(y_c - h / 2)
                    x2 = int(x_c + w / 2)
                    y2 = int(y_c + h / 2)

                    img_txt = img[y1:y2, x1:x2]

                    self.streamer.update_frame(img_txt, "cam1")
                    text = self.ocr_rec(img_txt)
                    print(f"当前检测文本: {text}")
                    text = "".join(re.findall(r"[\u4e00-\u9fffa-zA-Z]", text))
                    print(f"整理后文本: {text}")
                    if text_out is None:
                        text_out = text
                    else:
                        # 文本相似度比较
                        matcher = difflib.SequenceMatcher(None, text_out, text).ratio()
                        if text_count(matcher > 0.85):
                            return text_out
                        else:
                            text_out = text

    def get_ocr(self, label=None, time_out=2.0):
        """
        进行OCR识别

        使用任务摄像头获取图像，进行文本检测和OCR识别，返回识别结果。

        参数:
            time_out: 超时时间（秒），默认为3

        返回:
            str: 识别到的文本，如果超时或未检测到则返回None
        """
        time_stop = time.monotonic() + time_out
        perf_start = time.monotonic()
        detection_seconds = 0.0
        ocr_seconds = 0.0
        attempts = 0
        # 简单滤波：连续2次识别到相同文本即返回
        text_last = None
        text_count = 0

        def finish(value, reason):
            total_seconds = time.monotonic() - perf_start
            self._last_ocr_perf = {
                "attempts": attempts,
                "detection_seconds": detection_seconds,
                "ocr_seconds": ocr_seconds,
                "total_seconds": total_seconds,
                "reason": reason,
            }
            if bool(self.performance_cfg.get("timing_log", 1)):
                print(
                    "[PERF][ocr] label={} attempts={} detection={:.3f}s "
                    "ocr={:.3f}s total={:.3f}s reason={}".format(
                        label,
                        attempts,
                        detection_seconds,
                        ocr_seconds,
                        total_seconds,
                        reason,
                    )
                )
            return value

        while True:
            if self._stop_flag:
                return finish(text_last, "stopped")
            if time.monotonic() > time_stop:
                return finish(text_last, "timeout")

            attempts += 1
            det_start = time.monotonic()
            dets = self.get_detection_results(update_stream=False)
            detection_seconds += time.monotonic() - det_start

            # 必须使用 task 检测对应的同一帧。旧代码重新 read()，框和裁图
            # 可能跨帧，从而制造 OCR 抖动和无谓重试。
            img = getattr(self, "_last_det_img", None)
            if img is None:
                time.sleep(0.01)
                continue
            img = img.copy()
            if len(dets) > 0:
                for det in dets:
                    det_cls_id, det_id, det_label, det_score, det_bbox = (
                        det[0],
                        det[1],
                        det[2],
                        det[3],
                        det[4:],
                    )
                    if label is not None:
                        flag = det_label == label
                    else:
                        flag = det_label == "order" or det_label == "name"
                    if flag:
                        # 将归一化坐标转换为像素坐标
                        x_c, y_c, w, h = det_bbox
                        w *= 1.3
                        h *= 1.2
                        img_h, img_w = img.shape[:2]
                        x_c = int((x_c + 1) / 2 * img_w)
                        y_c = int((y_c + 1) / 2 * img_h)
                        w = int(w * img_w / 2)
                        h = int(h * img_h / 2)
                        # 左侧额外扩展，防止切掉行首字符（如编号"1"）
                        left_extra = int(w * 0.15)
                        x1 = max(0, int(x_c - w / 2) - left_extra)
                        y1 = max(0, int(y_c - h / 2))
                        x2 = min(img_w, int(x_c + w / 2))
                        y2 = min(img_h, int(y_c + h / 2))

                        if x1 >= x2 or y1 >= y2:
                            continue

                        img_txt = img[y1:y2, x1:x2]
                        self.streamer.update_frame(img_txt, "cam1")

                        ocr_start = time.monotonic()
                        text = self.ocr_rec(img_txt)
                        ocr_seconds += time.monotonic() - ocr_start
                        if text is None:
                            continue

                        if text_last is None:
                            text_last = text
                            text_count = 1
                        elif text == text_last:
                            text_count += 1
                            if text_count >= 2:
                                return finish(text, "stable_exact")
                        else:
                            # 文本变了，检查相似度
                            matcher = difflib.SequenceMatcher(
                                None, text_last, text
                            ).ratio()
                            if matcher > 0.85:
                                text_count += 1
                                if text_count >= 2:
                                    return finish(text_last, "stable_similar")
                            else:
                                text_last = text
                                text_count = 1
    
    def get_ocr2(self,label=None,time_out=3.0):
        """
        进行OCR识别,不进行滤波过滤
        参数:
            time_out: 超时时间（秒），默认为3

        返回:
            str: 识别到的文本，如果超时或未检测到则返回None
        """
        time_stop = time.time() + time_out
        while True:
            if self._stop_flag:
                return None
            if time.time() > time_stop:
                return None
            dets = self.get_detection_results()

            img = self.cap_front.read()
            if len(dets) > 0:
                for det in dets:
                    det_cls_id, det_id, det_label, det_score, det_bbox = (
                        det[0],
                        det[1],
                        det[2],
                        det[3],
                        det[4:],
                    )
                    if label is not None:
                        flag = det_label == label
                    else:
                        flag = det_label == "order" or det_label == "name"
                    if flag:
                        # 将归一化坐标转换为像素坐标
                        x_c, y_c, w, h = det_bbox
                        w *= 1.3
                        h *= 1.2
                        img_h, img_w = img.shape[:2]
                        x_c = int((x_c + 1) / 2 * img_w)
                        y_c = int((y_c + 1) / 2 * img_h)
                        w = int(w * img_w / 2)
                        h = int(h * img_h / 2)
                        # 左侧额外扩展，防止切掉行首字符（如编号"1"）
                        left_extra = int(w * 0.15)
                        x1 = max(0, int(x_c - w / 2) - left_extra)
                        y1 = max(0, int(y_c - h / 2))
                        x2 = min(img_w, int(x_c + w / 2))
                        y2 = min(img_h, int(y_c + h / 2))

                        if x1 >= x2 or y1 >= y2:
                            continue

                        img_txt = img[y1:y2, x1:x2]
                        self.streamer.update_frame(img_txt, "cam1")

                        text = self.ocr_rec(img_txt)
                        if text is not None:
                            return text
                    
    def yiyan_get_humattr(self, text):
        """
        获取人类属性分析

        使用文心一言分析文本中的人类属性信息。

        参数:
            text: 包含人类属性信息的文本

        返回:
            dict: 人类属性分析结果
        """
        return self.hum_analysis.get_res_json(text)

    def yiyan_get_actions(self, text):
        """
        获取动作分析

        使用文心一言分析文本中的动作信息。

        参数:
            text: 包含动作信息的文本

        返回:
            dict: 动作分析结果
        """
        return self.action_bot.get_res_json(text)

    def draw_detection_results(self, img, dets_ret):
        """
        将检测结果绘制在图像上

        Args:
            img: 原始图像
            dets_ret: 检测结果列表，每个元素包含 [cls_id, det_id, label, score, x_c, y_c, w, h]

        Returns:
            绘制了检测结果的图像
        """
        # 创建图像副本，避免修改原始图像
        img_show = img.copy()

        # 遍历每个检测结果
        for index, det in enumerate(dets_ret):
            # [cls_id:6 obj_id:0 label:water_l2 score:0.955 bbox:[309 334 399 431]]
            det_cls_id, det_id, det_label, det_score, det_bbox = (
                det[0],
                det[1],
                det[2],
                det[3],
                det[4:],
            )
            x_c, y_c, w, h = det_bbox

            # 将归一化坐标转换为像素坐标
            img_h, img_w = img.shape[:2]
            x_c = int((x_c + 1) / 2 * img_w)
            y_c = int((y_c + 1) / 2 * img_h)
            w = int(w * img_w / 2)
            h = int(h * img_h / 2)
            x1 = int(x_c - w / 2)
            y1 = int(y_c - h / 2)
            x2 = int(x_c + w / 2)
            y2 = int(y_c + h / 2)

            # 绘制矩形框
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # 绘制标签
            label_text = f"{index}-{det_label}:{det_score:.2f}"
            cv2.putText(
                img_show,
                label_text,
                (x1, y1),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return img_show

    def save_align_debug_img(self, delta_x, dx, label, dy=None, save_dir="water_block_debug",
                              p_term=0.0, i_term=0.0, d_term=0.0, final=False,
                              frame=0, stage="", direction="", out_x=0.0, out_y=0.0, delta_y=0.0,
                              ball_no=""):
        """
        保存视觉对准调试截图，标注误差、PID分量、帧号、阶段、移动方向。
        """
        import os, cv2
        import os
        import cv2
        import time as time_mod

        if not hasattr(self, '_last_det_img') or self._last_det_img is None:
            return None
        if dx is None:
            return None

        img = self._last_det_img.copy()  # 和模型推理用的是同一帧
        img_h, img_w = img.shape[:2]

        # 归一化坐标 [-1, 1] → 像素坐标 [0, img_w]
        center_x = img_w // 2                        # x=0 = 画面中心
        delta_px = int((delta_x + 1) / 2 * img_w)   # delta_x 目标位置
        dx_px = int((dx + 1) / 2 * img_w)           # 实际检测位置
        error = delta_x - dx     # error_x: 正值=目标偏左(车需右移)

        # --- 画面中心竖线（蓝色） ---
        cv2.line(img, (center_x, 0), (center_x, img_h), (255, 0, 0), 2)
        cv2.putText(img, "center(x=0)", (center_x + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # --- delta_x 目标竖线（绿色） ---
        cv2.line(img, (delta_px, 0), (delta_px, img_h), (0, 255, 0), 2)
        cv2.putText(img, f"target delta_x={delta_x:+.3f}", (delta_px + 5, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- 实际 dx 竖线（红色） ---
        cv2.line(img, (dx_px, 0), (dx_px, img_h), (0, 0, 255), 2)
        cv2.putText(img, f"actual dx={dx:+.4f}", (dx_px + 5, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ====== y轴：水平线 ======
        center_y = img_h // 2
        cv2.line(img, (0, center_y), (img_w, center_y), (255, 0, 0), 2)
        cv2.putText(img, "center(y=0)", (5, center_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if dy is not None and delta_y is not None:
            dy_px = int((dy + 1) / 2 * img_h)
            delta_y_px = int((delta_y + 1) / 2 * img_h)
            cv2.line(img, (0, delta_y_px), (img_w, delta_y_px), (0, 255, 0), 2)
            cv2.putText(img, f"target delta_y={delta_y:+.3f}", (5, delta_y_px - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.line(img, (0, dy_px), (img_w, dy_px), (0, 0, 255), 2)
            cv2.putText(img, f"actual dy={dy:+.4f}", (5, dy_px - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- 信息栏（黑色字体，画面上半部） ---
        err_y_str = f"err_y={dy-delta_y:+.4f}" if (dy is not None and delta_y is not None) else "err_y=N/A"
        final_tag = "[FINAL]" if final else ""
        ball_str = f" [{ball_no}]" if ball_no else ""
        info_lines = [
            f"#{frame} {stage} {direction} | {final_tag}{ball_str}",
            f"err_x={error:+.4f} {err_y_str} out_x={out_x:+.4f} out_y={out_y:+.4f} | label={label}",
            f"P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f}",
        ]
        y_base = 145   # 从画面中上部开始，不要太靠下
        for j, txt in enumerate(info_lines):
            cv2.putText(img, txt, (10, y_base + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # --- 保存 ---
        try:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_dir)
            os.makedirs(filepath, exist_ok=True)
            ball_tag = ball_no.replace("/", "-") if ball_no else ""
            prefix = f"ball{ball_tag}_{label}" if ball_tag else f"align_{label}"
            if final:
                filename = f"{prefix}_FINAL.jpg"
            else:
                filename = f"{prefix}_f{frame}.jpg"
            fullpath = os.path.join(filepath, filename)
            cv2.imwrite(fullpath, img)
        except Exception:
            pass

    def get_detection_results(
        self, sort_pos=(0, 0), limit_x=1, limit_y=1, update_stream=True
    ) -> List[list]:
        """
        获取检测结果,使用任务的目标检测对侧边摄像头图像进行检测，返回检测结果。

        返回:
            list: - 检测结果列表，每个元素包含 [cls_id, det_id, label, score, x_c, y_c, w, h]
        """
        perf_start = time.monotonic()
        raw_img, frame_seq, cap_time = self.cap_front.read_with_meta()
        if raw_img is None:
            return []
        self._last_det_img = raw_img.copy()
        self._last_det_seq = frame_seq
        self._last_det_time = cap_time
        image = raw_img.copy()
        infer_start = time.monotonic()
        det_task = self.task_det(image)
        infer_seconds = time.monotonic() - infer_start
        det_task = [det for det in det_task if abs(det[4]) <= limit_x]
        det_task = [det for det in det_task if abs(det[5]) <= limit_y]

        det_task.sort(
            key=lambda x: (x[4] - sort_pos[0]) ** 2 + (x[5] - sort_pos[1]) ** 2
        )  # 按照距离由近及远排序
        # 与 _last_det_img 同帧缓存，供动物裁图和 OCR 直接复用。
        self._last_det_results = [list(det) for det in det_task]
        if update_stream:
            image = self.draw_detection_results(image, det_task)
            self.streamer.update_frame(image, "cam2")
        self._last_detection_perf = {
            "frame_seq": frame_seq,
            "capture_time": cap_time,
            "infer_seconds": infer_seconds,
            "total_seconds": time.monotonic() - perf_start,
        }
        # print(det_task)
        return det_task

    # def _get_lane_error_cv(self, image):
    #     """
    #     传统CV扫线法：动态阈值求横向偏差（比Paddle更准）

    #     原理：取画面底部中央灰度作为赛道底色，分别向左、右扫找
    #     高亮黄边，三条扫描线加权平均得到赛道中线偏移。
    #     """
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     h, w = gray.shape
    #     gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #     center_x = w // 2
    #     scan_lines = [int(h * 0.75), int(h * 0.65), int(h * 0.55)]
    #     weights = [0.5, 0.3, 0.2]

    #     total_error = 0.0
    #     valid_weight_sum = 0.0

    #     # 动态阈值：画面底部中央灰度 + 跳变容差
    #     track_base_color = int(gray[int(h * 0.95), center_x])
    #     dynamic_thresh = track_base_color + 40

    #     # 调试用：在图像上描点
    #     debug_img = image.copy()
    #     cv2.line(debug_img, (center_x, 0), (center_x, h - 1), (0, 255, 255), 1)  # 中线
    #     cv2.circle(debug_img, (center_x, int(h * 0.95)), 5, (255, 0, 0), -1)     # 底色采样点

    #     print(f"[CV扫线] 底色:{track_base_color} 阈值:{dynamic_thresh} w:{w} h:{h}")

    #     for idx, y in enumerate(scan_lines):
    #         left_bound = 0
    #         right_bound = w - 1

    #         # 向左扫找黄边
    #         for x in range(center_x, 0, -1):
    #             if int(gray[y, x]) > dynamic_thresh:
    #                 left_bound = x
    #                 break

    #         # 向右扫找黄边
    #         for x in range(center_x, w - 1):
    #             if int(gray[y, x]) > dynamic_thresh:
    #                 right_bound = x
    #                 break

    #         line_center = (left_bound + right_bound) / 2.0
    #         error_px = line_center - center_x

    #         # 描点：扫描线 + 左右边界 + 中心
    #         cv2.line(debug_img, (0, y), (w - 1, y), (0, 255, 0), 1)  # 扫描线
    #         cv2.circle(debug_img, (left_bound, y), 6, (0, 0, 255), -1)   # 左边界 红
    #         cv2.circle(debug_img, (right_bound, y), 6, (255, 0, 0), -1)  # 右边界 蓝
    #         cv2.circle(debug_img, (int(line_center), y), 4, (0, 255, 255), -1)  # 算出的中心 黄

    #         print(f"  线{idx} y={y}: left={left_bound} right={right_bound} center={line_center:.1f} err={error_px:+.1f}px")

    #         total_error += error_px * weights[idx]
    #         valid_weight_sum += weights[idx]

    #     avg_error = total_error / valid_weight_sum if valid_weight_sum > 0 else 0.0
    #     error_y = float(avg_error / (w / 2.0))

    #     print(f"  → avg_err={avg_error:+.2f}px  y_err={error_y:+.4f}")

    #     # 序号递增保存调试图（参考collect_control.py模式）
    #     if not hasattr(self, '_cv_dbg_cnt'):
    #         self._cv_dbg_cnt = 0
    #     self._cv_dbg_cnt += 1
    #     if self._cv_dbg_cnt % 10 == 0:  # 每5帧保存一张
    #         save_dir = os.path.join(os.path.dirname(__file__), "cv_scan_debug")
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir, exist_ok=True)
    #         img_name = f"{self._cv_dbg_cnt:04d}.jpg"
    #         save_path = os.path.join(save_dir, img_name)
    #         cv2.imwrite(save_path, debug_img)
    #         print(f"  [已保存] {save_path}")

    #     return error_y

    def _get_lane_error_cv(self, image, scan_range=None, center_bias=0):
        """
        HSV彩色扫线法

        - HSV空间分离黄色边线，形态学去噪后从中心向两侧扫描
        - 找到黄色→用黄色；找不到→用V通道暗区跳变；都找不到→丢线=图像边界
        - 多条扫描线加权平均得到赛道中线偏移

        Args:
            image:      输入图像
            scan_range: 扫描范围 (start, end, lines)，默认 (0.30, 0.70, 6)
                        校准模式可传 (0.85, 0.95, 3) 只看底部近场
        """
        h, w = image.shape[:2]
        center_x = int(w // 2 + center_bias * (w / 2.0))

        # ====== 扫描线配置（可调参数） ======
        if scan_range is None:
            SCAN_START = 0.6   # 巡线默认：中远距离
            SCAN_END   = 0.85
            SCAN_LINES = 30
        else:
            SCAN_START, SCAN_END, SCAN_LINES = scan_range
        # ==================================

        scan_lines = [int(h * (SCAN_START + (SCAN_END - SCAN_START) * i / (SCAN_LINES - 1)))
                      for i in range(SCAN_LINES)]
        # 权重：近处略高但整体均匀，保证前瞻不丢
        weights = [0.15 + 0.035 * i for i in range(SCAN_LINES)]  # 远处0.15 → 近处0.325
        _w_sum = sum(weights)
        weights = [w / _w_sum for w in weights]

        # ====== 1. HSV橙-黄掩码（hue下限降到10覆盖橙色） ======
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = (10, 40, 40)    # H=10覆盖橙黄, S/V门槛降低
        upper_yellow = (45, 255, 255)  # H=45覆盖到绿黄边界
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 统计黄色像素，存到self供外部读取
        self._last_yellow_px = (mask_yellow > 0).sum()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        # ====== 2. V通道暗区后备 ======
        _, _, v_ch = cv2.split(hsv)
        _, track_bright = cv2.threshold(v_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 梯度强度阈值：只接受 V 通道跳变 > 此值的边界点
        # 赛道→边界的灰度跳变大（通常 30-80），赛道中心灰色区的弱跳变被过滤
        EDGE_GRADIENT_THRESHOLD = 35  # V通道相邻列灰度差值阈值，越小越敏感

        # ====== 3. 扫描（从下往上，种子传递） ======
        # 策略：底部扫描线从 image center 开始，每条线扫完后把实际中心点传给上一行作种子。
        # 弯道大时黄线靠近画面中心，种子传递可以让扫描线沿着车道曲率走，避免丢线。
        total_error = 0.0
        valid_weight_sum = 0.0
        self._last_scan_centers = []   # 记录每行扫线中心，供外部计算斜率

        # 窗口滤波参数：每条扫描线用上下各 N 行做多数投票，防止赛道黑点被误判为边界
        WINDOW_HALF = 2       # 半窗高（总窗口 = 2*HALF+1 = 5行）
        WINDOW_VOTES = 3      # 至少需要多少行满足条件才算"找到了"
        HALF_H = WINDOW_HALF

        debug_img = image.copy()
        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[mask_yellow > 0] = (0, 255, 255)
        debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)
        cv2.line(debug_img, (center_x, 0), (center_x, h - 1), (255, 0, 0), 1)

        # 底部第一条线的种子 = 图像中线
        seed_x = center_x

        # 从下往上扫：i = SCAN_LINES-1（底部） → 0（顶部）
        for i in range(SCAN_LINES - 1, -1, -1):
            y = scan_lines[i]
            found_l, found_r = False, False
            left_bound, right_bound = 0, w - 1  # 丢线=图像边界
            scan_start = int(seed_x)             # 用上一行的实际中心作起始点
            y0 = max(0, y - HALF_H)
            y1 = min(h - 1, y + HALF_H)

            # 向左扫：从种子位置开始
            for x in range(scan_start, 0, -1):
                col_yellow = mask_yellow[y0:y1+1, x]
                if np.count_nonzero(col_yellow) >= WINDOW_VOTES:
                    left_bound, found_l = x, True
                    break

            # 向右扫：从种子位置开始
            for x in range(scan_start, w - 1):
                col_yellow = mask_yellow[y0:y1+1, x]
                if np.count_nonzero(col_yellow) >= WINDOW_VOTES:
                    right_bound, found_r = x, True
                    break

            # 计算本行中线，作为上一行（更远的线）的种子
            line_center = (left_bound + right_bound) / 2.0
            # 至少一侧找到黄色才更新种子，全丢线时保持上一次的种子
            if found_l or found_r:
                seed_x = line_center

            error_px = line_center - center_x
            if found_l and found_r:
                w_eff = weights[i]       # 双侧完整权重
            elif found_l or found_r:
                w_eff = weights[i] * 0.5  # 单侧丢线，权重打折
            else:
                w_eff = 0.0                 # 全丢，不计入
            total_error += error_px * w_eff
            valid_weight_sum += w_eff

            # 描点
            color_l = (0, 0, 255) if found_l else (128, 128, 128)
            color_r = (255, 0, 0) if found_r else (128, 128, 128)
            cv2.line(debug_img, (0, y), (w - 1, y), (0, 255, 0), 1)
            cv2.circle(debug_img, (left_bound, y), 6, color_l, -1)
            cv2.circle(debug_img, (right_bound, y), 6, color_r, -1)
            cv2.circle(debug_img, (int(line_center), y), 4, (0, 255, 255), -1)
            self._last_scan_centers.append((y, line_center))

            tag_l = "✓" if found_l else "x"
            tag_r = "✓" if found_r else "x"
            # print(f"  线{idx} y={y}: L={left_bound}{tag_l} R={right_bound}{tag_r} c={line_center:.1f} e={error_px:+.1f}px w={valid_weight_sum:.1f}")

        avg_error = total_error / valid_weight_sum if valid_weight_sum > 0 else 0.0
        error_y = float(avg_error / (w / 2.0))
        # print(f"  → avg={avg_error:+.2f}px  y_err={error_y:+.4f}")

        # 保存带扫描线/描点的调试图，供 lane_base 使用
        self._last_cv_debug_img = debug_img

        return error_y

    def _get_lane_error_cv_1(self, image, scan_range=None):
        """[Mode2用] CV扫线副本，参数你来改 —— 其余逻辑同 _get_lane_error_cv"""
        h, w = image.shape[:2]
        center_x = w // 2

        if scan_range is None:
            SCAN_START = 0.6
            SCAN_END   = 0.85
            SCAN_LINES = 30
        else:
            SCAN_START, SCAN_END, SCAN_LINES = scan_range

        scan_lines = [int(h * (SCAN_START + (SCAN_END - SCAN_START) * i / (SCAN_LINES - 1)))
                      for i in range(SCAN_LINES)]
        weights = [0.15 + 0.035 * i for i in range(SCAN_LINES)]
        _w_sum = sum(weights)
        weights = [w / _w_sum for w in weights]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = (10, 40, 40)
        upper_yellow = (45, 255, 255)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        self._last_yellow_px = (mask_yellow > 0).sum()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        _, _, v_ch = cv2.split(hsv)
        _, track_bright = cv2.threshold(v_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        EDGE_GRADIENT_THRESHOLD = 35

        total_error = 0.0
        valid_weight_sum = 0.0
        self._last_scan_centers = []
        WINDOW_HALF = 2
        WINDOW_VOTES = 3
        HALF_H = WINDOW_HALF
        debug_img = image.copy()
        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[mask_yellow > 0] = (0, 255, 255)
        debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)
        cv2.line(debug_img, (center_x, 0), (center_x, h - 1), (255, 0, 0), 1)
        seed_x = center_x
        for i in range(SCAN_LINES - 1, -1, -1):
            y = scan_lines[i]
            found_l, found_r = False, False
            left_bound, right_bound = 0, w - 1
            scan_start = int(seed_x)
            y0 = max(0, y - HALF_H)
            y1 = min(h - 1, y + HALF_H)
            for x in range(scan_start, 0, -1):
                col_yellow = mask_yellow[y0:y1+1, x]
                if np.count_nonzero(col_yellow) >= WINDOW_VOTES:
                    left_bound, found_l = x, True
                    break
            for x in range(scan_start, w - 1):
                col_yellow = mask_yellow[y0:y1+1, x]
                if np.count_nonzero(col_yellow) >= WINDOW_VOTES:
                    right_bound, found_r = x, True
                    break
            line_center = (left_bound + right_bound) / 2.0
            if found_l or found_r:
                seed_x = line_center
            error_px = line_center - center_x
            if found_l and found_r:
                w_eff = weights[i]
            elif found_l or found_r:
                w_eff = weights[i] * 0.5
            else:
                w_eff = 0.0
            total_error += error_px * w_eff
            valid_weight_sum += w_eff
            color_l = (0, 0, 255) if found_l else (128, 128, 128)
            color_r = (255, 0, 0) if found_r else (128, 128, 128)
            cv2.line(debug_img, (0, y), (w - 1, y), (0, 255, 0), 1)
            cv2.circle(debug_img, (left_bound, y), 6, color_l, -1)
            cv2.circle(debug_img, (right_bound, y), 6, color_r, -1)
            cv2.circle(debug_img, (int(line_center), y), 4, (0, 255, 255), -1)
            self._last_scan_centers.append((y, line_center))
        avg_error = total_error / valid_weight_sum if valid_weight_sum > 0 else 0.0
        error_y = float(avg_error / (w / 2.0))
        self._last_cv_debug_img = debug_img
        return error_y

    def get_lane_results(self, image=None, cv_mode=1, scan_range=None):
        """获取车道检测结果。
        cv_mode: 1=默认CV扫线, 2=备用CV扫线(_get_lane_error_cv_1)
        scan_range: CV扫线范围 (start, end, lines)，如(0.6, 0.85, 6)，默认None使用内部默认值"""
        if image is None:
            image = self.cap_side.read().copy()
        # Paddle推理：取角度偏差（AI模型对角度判断更准）
        res = self.crusie(image)
        angle_raw = res[1] if res[1] is not None else 0.0

        # CV扫线：取横向偏差
        if cv_mode == 2:
            error_raw = self._get_lane_error_cv_1(image, scan_range=scan_range)
        else:
            error_raw = self._get_lane_error_cv(image, scan_range=scan_range)

        # EMA低通滤波y_err，消除CV噪声引起的车身抖动
        if not hasattr(self, '_y_err_filtered'):
            self._y_err_filtered = error_raw
            self._y_err_last = error_raw
        self._y_err_filtered = 0.4 * error_raw + 0.6 * self._y_err_filtered

        # 变化率限制：单次变化不超过0.1，防止跳变
        slew_max = 0.4
        delta = self._y_err_filtered - self._y_err_last
        if delta > slew_max:
            error = self._y_err_last + slew_max
        elif delta < -slew_max:
            error = self._y_err_last - slew_max
        else:
            error = self._y_err_filtered
        self._y_err_last = error

        # EMA低通滤波angle，防止Paddle输出跳变（0.001→0.44→1.09）
        if not hasattr(self, '_ang_filtered'):
            self._ang_filtered = angle_raw
            self._ang_last = angle_raw
        self._ang_filtered = 0.3 * angle_raw + 0.7 * self._ang_filtered

        # 非对称变化率限制：变大严/变小松，防跳变同时允许快速回正
        ANGLE_SLEW_UP   = 0.25  # 绝对值增大时每帧最大变化（严格防突变）
        ANGLE_SLEW_DOWN = 0.4  # 绝对值减小时每帧最大变化（放宽快速回正）
        delta_ang = self._ang_filtered - self._ang_last
        abs_now  = abs(self._ang_filtered)
        abs_last = abs(self._ang_last)
        limit = ANGLE_SLEW_UP if abs_now > abs_last else ANGLE_SLEW_DOWN

        if delta_ang > limit:
            angle = self._ang_last + limit
        elif delta_ang < -limit:
            angle = self._ang_last - limit
        else:
            angle = self._ang_filtered
        self._ang_last = angle

        # 绘制标签
        label_text = f"d_e:{error:+.4f} d_a:{angle:+.4f}r{angle_raw:+.3f}"

        cv2.putText(
            image,
            label_text,
            (20, 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label_text,
            (20, 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            1.0,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        self.streamer.update_frame(image, "cam1")
        return -error, angle

    def get_target_location(self, det):
        """
        通过传入的目标在图像的坐标，计算目标相对小车的偏移 x,y

        参数:
            det: 包含目标检测信息的列表，格式为 [cls_id, obj_id,label, score, x_c, y_c, w, h]
                - x_c: 目标在图像中的 x 坐标
                - y_c: 目标在图像中的 y 坐标
                - w: 目标的宽度
                - h: 目标的高度

        返回:
            tuple: 目标相对小车的坐标 (loc_x, loc_y)
                - loc_x: 目标相对小车的 x 坐标
                - loc_y: 目标相对小车的 y 坐标
        """
        # 摄像头图像在现实中实际的高和宽
        CAMERA_HEIGHT = 0.23
        CAMERA_WIDTH = 0.33
        # 机械臂x原点距离小车中心的距离
        ARM_OFFSET = 0.15

        # 获取机械臂的方向和长度
        arm_y = self.arm.x_pose_now + ARM_OFFSET
        side = self.arm.side
        length = 0

        # 根据机械臂方向调整长度
        if side == "RIGHT":
            length = -self.arm.arm_length
        elif side == "LEFT":
            length = self.arm.arm_length

        # 提取目标在图像中的坐标和尺寸
        x_c, y_c, w, h = det[4:]

        # 计算目标中心点在摄像头中的世界坐标
        x = CAMERA_WIDTH * (x_c + w / 2)
        y = CAMERA_HEIGHT * (y_c + h / 2)

        # 计算目标中心点在小车中的世界坐标
        loc_x = x
        loc_y = y + arm_y + length

        return loc_x, loc_y

    def move_to_detection_target_pidlib(
        self,
        # delta_x=-0.025,                        # 目标在画面中的水平期望位置（归一化坐标，0=画面中心）
        delta_x=-0.0,
        delta_y: Union[float, None] = -0.05,  # 目标在画面中的垂直期望位置（None=不控制y方向）
        label=None,                          # 指定目标标签，None=选距离画面中心最近的目标
        time_out=2.0,                        # 超时时间(秒)
        sort_pos=(0, 0),                     # 排序参考点，默认画面中心
        num=0,                               # 取排序后第几个目标，0=最近
        ball_no="",                          # 球编号（如"1/3"），仅用于调试图片显示
    ):
        """
        视觉伺服定位：检测目标 → PID控制车体前后移动 + 机械臂横向移动 → 对齐目标。

        核心链路：
          侧面摄像头 → PP-YOLOE模型检测目标 → 获取目标归一化坐标(dx, dy)
          → pid_x(dx) 输出前后速度 → set_velocity 控制车体前后移动
          → kp_y*(dy-delta_y) 控制机械臂左右平移
          → dx/dy 均连续N帧满足阈值 → 对齐成功

        返回: (cls_id, label, dy)  超时或中断返回 (-1, "None", None)
        """
        # ====== 1. 初始化计时器和计数器 ======
        time_stop = time.time() + time_out  # 超时时刻

        # ====== 2. PID和状态变量初始化 ======
        out_x = 0                           # 输出到车体的前后速度(m/s)
        out_y = 0                           # 输出到机械臂的横向速度(m/s)
        last_out_x = 0.0                    # 上一帧out_x，刹车用
        frame_cnt = 0                       # 检测到目标的帧数
        step = 0                            # 步骤计数器（仅用于终端日志）
        phase = "x"                         # 当前阶段: "x"=底盘前后对齐, "y"=机械臂横向对齐
        x_good_cnt = 0                      # X轴连续误差<0.03的帧计数，>=2切到Y阶段
        print(f"[STEP] 视觉对准(顺序版)开始 | delta_x={delta_x:.2f} delta_y={delta_y} label={label} arm={self.arm.side}")

        # 根据机械臂方向选择 PID 参数符号（左/右两侧，摄像头图像镜像关系）
        if self.arm.side == "RIGHT":
            kp_y = -0.2                     # 机械臂横向P（右臂取反）
            kp_x = -0.1                     # 车体前后P（右臂取反）
            ki_x = -0.005                   # 车体前后I（右臂取反）
        else:
            kp_y = 0.08
            kp_x = 0.06
            ki_x = 0.05


        # 初始化车体前后移动的 PID 控制器（setpoint = delta_x = 画面期望位置，输入=实际dx）
        pid_x = PID(kp_x, ki_x)
        pid_x.output_limits = (-0.02, 0.02) # 速度上下限(m/s)，太小了推不动车
        pid_x.setpoint = delta_x            # PID目标值 = delta_x（默认0=画面居中）

        # ====== 3. 视觉伺服主循环 ======
        while True:
            # 急停信号检测
            if self._stop_flag:
                self.set_velocity(0, 0, 0)  # 停底盘
                self.arm.x_speed(0)          # 停机械臂
                print(f"[STEP] 视觉对准中断 | 总帧数={frame_cnt}")
                return -1, "None", None

            # ---- 3a. 获取目标检测结果 ----
            # 调用 PP-YOLOE 模型，每帧实时推理（约20Hz）
            dets = self.get_detection_results(sort_pos=sort_pos)

            # 如果指定了label，只保留匹配的目标
            if label is not None:
                dets = [item for item in dets if item[2] == label]

            # ---- 3b. 检测到目标 → PID控制 ----
            if len(dets) > num:
                frame_cnt += 1               # 有效检测帧计数

                # 取第num个目标，提取归一化坐标
                # det 格式: [cls_id, det_id, label, score, x_c, y_c, w, h]
                # dx = x_c ∈ [-1,1]: 目标在画面水平位置，0=居中
                # dy = y_c ∈ [-1,1]: 目标在画面垂直位置，0=居中
                det = dets[num]
                dx, dy = det[4:6]

                # ---- 先算误差 ----
                err_x = dx - delta_x
                err_y = dy - delta_y if delta_y is not None else 0
                err_x_abs = abs(err_x)
                err_y_abs = abs(err_y) if delta_y is not None else 0

                # ========== 阶段机：先X后Y ==========
                if phase == "x":
                    # ---- X阶段：只调底盘前后，机械臂Y轴不动 ----
                    out_y = 0.0

                    # 两帧两阈值：第1帧<0.03刹车+停车，第2帧<0.02确认锁定
                    if x_good_cnt == 1:
                        x_good_cnt = 2 if err_x_abs < 0.04 else 0
                        out_x = 0.0               # 确认帧不动车
                    elif err_x_abs < 0.06:
                        x_good_cnt = 1
                        out_x = 0.0               # 第一帧刹车，立刻停车
                    else:
                        x_good_cnt = 0

                    if x_good_cnt >= 2:
                        # X锁定，切换到Y阶段
                        phase = "y"
                        x_good_cnt = 0
                        pid_x.reset()
                        out_x = 0.0
                        p_term = i_term = d_term = 0.0
                        step += 1
                        print(f"[STEP {step}] X轴锁定 | err_x={err_x:+.4f} → 进入Y轴调整阶段")
                    elif x_good_cnt == 1:
                        # 第1帧达标：反向短刹 + 停车
                        pid_x.reset()
                        pid_x._integral = 0.0
                        out_x = 0.0
                        self.set_velocity(-last_out_x * 0.8, 0, 0)  # 反向80%力刹一脚
                        time.sleep(0.05)
                        self.set_velocity(0, 0, 0)
                        p_term = i_term = d_term = 0.0
                    else:
                        # X阶段PID控制（底盘前后移动）
                        if err_x_abs > 0.15 or err_x_abs < 0.03:
                            pid_x._integral = 0.0    # 远到离谱或近了都清积分
                        error_sign = err_x
                        if error_sign > 0 and pid_x._integral > 0:
                            pid_x._integral = 0.0
                        if error_sign < 0 and pid_x._integral < 0:
                            pid_x._integral = 0.0

                        _i_before = pid_x._integral
                        out_x = -pid_x(dx)
                        p_term = getattr(pid_x, '_proportional', 0.0)
                        d_term = getattr(pid_x, '_derivative', 0.0)
                        _i_delta = pid_x._integral - _i_before
                        # out接近上限 且 误差较小时收紧积分步长，防止积分超调
                        I_STEP_MAX = 0.001 if (abs(out_x) >= 0.0095 and err_x_abs < 0.08) else 0.0035
                        if abs(_i_delta) > I_STEP_MAX:
                            pid_x._integral = _i_before + (I_STEP_MAX if _i_delta > 0 else -I_STEP_MAX)

                        I_LIMIT = 0.01
                        iv = pid_x._integral
                        if abs(iv) > I_LIMIT:
                            pid_x._integral = I_LIMIT if iv > 0 else -I_LIMIT

                        pid_raw = p_term + pid_x._integral + d_term
                        lo, hi = pid_x.output_limits
                        pid_clamped = max(lo, min(hi, pid_raw))
                        out_x = -pid_clamped
                        i_term = pid_x._integral

                        # ==== 输出 = 基速(克服静摩擦) + PID ====
                        BASE_SPEED = 0.0095
                        if err_x_abs < 0.03:
                            BASE_SPEED = 0.005   # 近目标用小步长
                        out_x = BASE_SPEED if err_x > 0 else (-BASE_SPEED if err_x < 0 else 0)
                        out_x += -(p_term + pid_x._integral + d_term)

                        if abs(out_x) > 0.0125:
                            out_x = 0.0125 if out_x > 0 else -0.0125
                        if out_x > 0 and out_x < BASE_SPEED: out_x = BASE_SPEED
                        if out_x < 0 and out_x > -BASE_SPEED: out_x = -BASE_SPEED

                    last_out_x = out_x   # 记录给下一帧刹车用

                else:  # phase == "y"
                    # ---- Y阶段：X锁死，只调机械臂Y轴 ----
                    out_x = 0.0
                    p_term = i_term = d_term = 0.0

                    if delta_y is not None:
                        out_y = kp_y * (dy - delta_y)
                    else:
                        out_y = 0.0

                    # X漂移过大→回到X阶段
                    if err_x_abs > 0.08:
                        phase = "x"
                        x_good_cnt = 0
                        step += 1
                        print(f"[STEP {step}] X漂移({err_x:+.4f}) → 回到X轴调整阶段")

                # 方向和阶段标记
                if out_x > 0:
                    _dir_cn = "↑前进"
                elif out_x < 0:
                    _dir_cn = "↓后退"
                else:
                    _dir_cn = "⊙锁定"
                stage_str = f"[{phase.upper()}]"

                # 首次检测到目标时打 STEP 日志
                if frame_cnt == 1:
                    step += 1
                    print(f"[STEP {step}] 首次检测 | err_x={err_x:+.4f} dx={dx:+.3f}")

                # ---- 3d. 对齐判定（Y阶段 Y误差<0.02 → 抓取） ----
                if phase == "y" and (delta_y is None or abs(dy - delta_y) < 0.02):
                    pid_x.reset()
                    self.set_velocity(0, 0, 0)
                    self.arm.x_speed(0)
                    step += 1
                    print(f"[STEP {step}] 对齐成功 → 抓取! | frame={frame_cnt} cls_id={det[0]} label={det[2]} dy={dy:+.4f}")
                    print(f"--- 视觉对准完成 ---")
                    # self.save_align_debug_img(delta_x, dx, det[2], dy,
                    #     save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                    #     final=True, frame=frame_cnt, stage=stage_str, direction="F" if out_x>0 else "B", out_x=out_x, delta_y=delta_y, ball_no=ball_no)
                    return det[0], det[2], dy

                # ---- 3e. 每帧日志和调试截图 ----
                print(f"  [视觉对准] #{frame_cnt} {stage_str} | err_x={err_x:+.4f} err_y={err_y:+.4f} dx={dx:+.3f} {_dir_cn} out={out_x:+.4f} out_y={out_y:+.4f} | P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f}")
                # self.save_align_debug_img(delta_x, dx, det[2], dy,
                #     save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                #     frame=frame_cnt, stage=stage_str, direction="F" if out_x>0 else "B", out_x=out_x, delta_y=delta_y, ball_no=ball_no)
            else:
                pass  # 没检测到目标，继续等

            # ---- 3f. 下发控制指令到硬件 ----
            self.set_velocity(out_x, 0, 0)  # 底盘：out_x 控制前后移动
            self.arm.x_speed(out_y)          # 机械臂：out_y 控制横向平移
            time.sleep(0.025)                 # 控制频率 ~20Hz

            # ---- 3g. 超时处理 ----
            if time.time() > time_stop:
                self.set_velocity(0, 0, 0)
                self.arm.x_speed(0)
                step += 1
                print(f"[STEP {step}] 超时 | 总帧={frame_cnt}")
                try:
                    # self.save_align_debug_img(delta_x, last_dx if 'last_dx' in dir() else dx, last_label if 'last_label' in dir() else label, dy if 'dy' in dir() else 0.0, save_dir="det_target_debug")
                    return det[0], det[2], dy
                except:
                    return (None, None, None)
                
    def move_to_detection_target2(
        self,
        delta_x=-0.0,
        # delta_x=0.02,
        delta_y: Union[float, None] = -0.05,
        label=None,
        time_out=1.0,
        sort_pos=(0, 0),
        num=0,
        ball_no="",
        x_thr=0.06,        # x对齐阈值
        y_thr=0.06,        # y对齐阈值
        lock_thr=0.04,     # x锁阈值（误差<此值停车）
        skip_y=False,      # 跳过y对齐(水塔用)
        save_images=True,  # 是否保存逐帧/最终调试图；默认保持旧行为
        success_beeps=3,   # 成功蜂鸣次数；慢任务可设0，其他调用不受影响
        update_stream=True,# 是否绘框并更新网页流
    ):
        # ====== 1. 初始化 ======
        time_stop = time.time() + time_out
        out_x, out_y = 0.0, 0.0
        frame_cnt, step = 0, 0
        x_locked = False
        _tag = f" [{ball_no}]" if ball_no else ""
        print(f"[STEP] 视觉对准开始{_tag} | delta_x={delta_x:.2f} delta_y={delta_y} label={label} arm={self.arm.side}")

        perf_start = time.monotonic()
        infer_count = 0
        infer_seconds = 0.0
        last_selected_det = None
        last_selected_img = None
        last_selected_seq = 0
        last_selected_time = 0.0
        try:
            success_beeps = max(0, int(success_beeps))
        except (TypeError, ValueError):
            success_beeps = 3

        def remember_selected(selected_det=None):
            # 发布循环中当场保存的框/图快照。尤其在“先检测到、后丢失并
            # 超时”时，不能把旧 det 与超时前最新的另一帧重新配对。
            source_det = last_selected_det
            if source_det is None and selected_det is not None:
                source_det = list(selected_det)
            self._last_selected_det = (
                list(source_det) if source_det is not None else None
            )
            self._last_selected_img = (
                last_selected_img.copy() if last_selected_img is not None else None
            )
            self._last_selected_seq = last_selected_seq
            self._last_selected_time = last_selected_time

        def finish_perf(reason):
            self._last_align_perf = {
                "reason": reason,
                "frames": frame_cnt,
                "infer_count": infer_count,
                "infer_seconds": infer_seconds,
                "total_seconds": time.monotonic() - perf_start,
            }
            if bool(self.performance_cfg.get("timing_log", 1)):
                print(
                    "[PERF][align] label={} reason={} frames={} infer_count={} "
                    "infer={:.3f}s total={:.3f}s".format(
                        label,
                        reason,
                        frame_cnt,
                        infer_count,
                        infer_seconds,
                        self._last_align_perf["total_seconds"],
                    )
                )

        # ====== 2. 手写PID参数（统一使用RIGHT臂参数） ======
        kp_y, kp_x, ki_x, kd_x = -0.0, 0.018, 0.0, 0.0

        integral_x  = 0.0      # Σ(ki * error * dt)
        last_err_x  = 0.0      # 上帧误差
        OUT_LIMIT   = 0.02     # out_x 限幅
        I_LIMIT     = 0.01     # |integral| 硬上限
        I_STEP_MAX  = 0.008    # |integral| 单帧增量上限
        DT          = 0.05     # 帧间隔(秒)
        CLEAR_I_ERR = 0.15     # |error|>此值,本帧积分清零
        LOCK_THRESH = lock_thr   # x锁阈值
        ALIGN_X_THR = x_thr    # x对齐判定阈值
        ALIGN_Y_THR = y_thr    # y对齐判定阈值
        # ===================================================

        while True:
            if self._stop_flag:
                self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                print(f"[STEP] 中断 | 总帧={frame_cnt}")
                finish_perf("stopped")
                return -1, "None", None

            infer_start = time.monotonic()
            dets = self.get_detection_results(
                sort_pos=sort_pos, update_stream=update_stream
            )
            infer_seconds += time.monotonic() - infer_start
            infer_count += 1
            if label is not None:
                dets = [d for d in dets if d[2] == label]

            if len(dets) > num:
                frame_cnt += 1
                out_x = 0.0          # 每帧清零，新算
                det = dets[num]
                dx, dy = det[4:6]
                # 检测框与其来源帧必须在这里原子式成对保存，供成功或超时
                # 返回后的动物裁图复用。
                last_selected_det = list(det)
                selected_img = getattr(self, "_last_det_img", None)
                last_selected_img = (
                    selected_img.copy() if selected_img is not None else None
                )
                last_selected_seq = getattr(self, "_last_det_seq", 0)
                last_selected_time = getattr(self, "_last_det_time", 0.0)
                # self.beep()


                err_abs = abs(dx - delta_x)
                error_x = delta_x - dx           # PID误差 (setpoint - input)
                self._last_err_x = error_x       # 供外部读取最终err_x

                # ---- x锁：OK后锁死，err漂出阈值就解锁 ----
                if x_locked and err_abs > LOCK_THRESH:
                    x_locked = False

                if x_locked:
                    p_term = i_term = d_term = 0.0
                    out_x = 0.0
                elif err_abs < LOCK_THRESH:
                    x_locked = True
                    integral_x = 0.0
                    p_term = i_term = d_term = 0.0
                    out_x = 0.0
                else:
                    # ==== 手动 P ====
                    p_term = kp_x * error_x

                    # ==== 手动 D ====
                    d_term = kd_x * (error_x - last_err_x) / DT
                    last_err_x = error_x

                    # ==== 手动 I（增量限幅+硬限幅） ====
                    # 防积分饱和：P+D 还推不动车时暂停积分，避免车动后过冲
                    _pd_sum = abs(p_term + d_term)
                    if _pd_sum < 0.095:
                        integral_x = 0.0
                    else:
                        i_delta = ki_x * error_x * DT
                        if i_delta > I_STEP_MAX:  i_delta = I_STEP_MAX
                        if i_delta < -I_STEP_MAX: i_delta = -I_STEP_MAX
                        integral_x += i_delta
                    if err_abs > CLEAR_I_ERR: integral_x = 0.0
                    if integral_x > I_LIMIT:  integral_x = I_LIMIT
                    if integral_x < -I_LIMIT: integral_x = -I_LIMIT
                    i_term = integral_x

                    # ==== 输出 = 基速(克服静摩擦) + PID ====
                    BASE_SPEED = 0.0085
                    out_x = BASE_SPEED if error_x > 0 else (-BASE_SPEED if error_x < 0 else 0)

                    out_x += p_term + i_term + d_term
                    # 保底：out非零时至少0.0095才能克服静摩擦
                    if out_x > 0 and out_x < 0.0095: out_x = 0.0095
                    if out_x < 0 and out_x > -0.0095: out_x = -0.0095

                    # 钳制
                    MAX_SPEED = 0.02
                    if out_x > MAX_SPEED:  out_x = MAX_SPEED
                    if out_x < -MAX_SPEED: out_x = -MAX_SPEED

                # ---- 机械臂 y ----
                if delta_y is None:
                    out_y = 0.0
                else:
                    LEFTY = 0.003
                    if self.arm.side == "LEFT":
                        out_y = -LEFTY if (dy - delta_y) < 0 else (LEFTY if (dy - delta_y) > 0 else 0)
                    out_y += kp_y * (dy - delta_y)

                    # MAX_SPEED = 0.025
                    if out_y > 0.02:  out_y = 0.02
                    if out_y < -0.02: out_y = -0.02

                # ---- 日志标记 ----
                _dir_cn = "↑前进" if out_x > 0 else "↓后退"
                stage_str = "OK" if err_abs < 0.04 else ("SLOW" if err_abs < 0.1 else "FAST")
                lock_tag = " [LOCKED]" if x_locked else ""
                if frame_cnt == 1:
                    step += 1
                    print(f"[STEP {step}] 首次检测 | err_x={error_x:+.4f} dx={dx:+.3f}")

                # ---- 对齐退出（统一跳过y对齐，只对齐x） ----
                aligned_now = abs(dx - delta_x) < ALIGN_X_THR and (skip_y or delta_y is None or abs(dy - delta_y) < ALIGN_Y_THR)
                if aligned_now:
                    out_x = 0.0; out_y = 0.0
                    integral_x = 0.0; last_err_x = 0.0
                    self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                    remember_selected(det)
                    for _ in range(success_beeps):
                        self.beep()
                        time.sleep(0.1)
                    step += 1
                    print(f"[STEP {step}] 对齐成功 | 退出")
                    print(f"--- 视觉对准完成 ---")
                    if save_images:
                        self.save_align_debug_img(delta_x, dx, det[2], dy,
                            save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                            final=True, frame=frame_cnt, stage=stage_str, direction="F" if out_x>0 else "B", out_x=out_x, out_y=out_y, delta_y=delta_y, ball_no=ball_no)
                    finish_perf("aligned")
                    return det[0], det[2], dy

                # ---- 每帧日志+截图 ----
                _ey = (dy - delta_y) if delta_y is not None else 0.0
                print(f"  [视觉对准] #{frame_cnt} {stage_str}{lock_tag} | err_x={error_x:+.4f} err_y={_ey:+.4f} dx={dx:+.3f} {_dir_cn} out={out_x:+.4f} out_y={out_y:+.4f} | P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f}")
                if save_images:
                    self.save_align_debug_img(delta_x, dx, det[2], dy,
                        save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                        frame=frame_cnt, stage=stage_str+lock_tag, direction="F" if out_x>0 else "B", out_x=out_x, delta_y=delta_y, ball_no=ball_no)

                # --- 手动确认（需要时取消注释） ---
                # if out_x != 0 or out_y != 0:
                #     _dir = "前进" if out_x > 0 else "后退"
                #     print(f"  [MANUAL] ---- 待执行移动 ----")
                #     print(f"    err_x={dx-delta_x:+.4f}  err_y={dy-delta_y:+.4f}  {_dir}")
                #     print(f"    P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f}  out={out_x:+.4f}")
                #     input(f"    按回车执行...")
                #     self.beep()
            else:
                pass

            self.set_velocity(out_x, 0, 0)
            self.arm.x_speed(out_y)
            # time.sleep(DT)
            time.sleep(0.025)

            if time.time() > time_stop:
                self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                step += 1
                print(f"[STEP {step}] 超时 | 总帧={frame_cnt}")
                try:
                    remember_selected(det)
                    if save_images:
                        self.save_align_debug_img(delta_x, dx, det[2], dy, save_dir="det_target_debug")
                    finish_perf("timeout_with_detection")
                    return det[0], det[2], dy
                except:
                    finish_perf("timeout_no_detection")
                    return (None, None, None)
                
    def move_to_detection_target(
        self,
        delta_x=0.0,
        delta_y: Union[float, None] = 0.0,
        label=None,
        time_out=2.0,
        sort_pos=(0, 0),
        num=0,
    ):
        time_stop = time.time() + time_out
        x_count = CountRecord(2)
        y_count = CountRecord(2)

        # pid_x.output_limits((-0.7, 0.7))

        out_x = 0
        out_y = 0
        # print(f"手柄方向：{self.arm.side}")
        if self.arm.side == "RIGHT":
            kp_y = -0.08
            kp_x = -0.05
            ki_x = -0.0
        else:
            kp_y = 0.0
            kp_x = 0.05
            ki_x = 0.0

        pid_x = PID(kp_x, ki_x)
        pid_x.setpoint = delta_x
        while True:
            if self._stop_flag:
                self.set_velocity(0, 0, 0)
                self.arm.x_speed(0)
                return -1, "None", None

            dets = self.get_detection_results(sort_pos=sort_pos)

            if label is not None:
                dets = [item for item in dets if item[2] == label]

            if len(dets) > num:
                det = dets[num]
                dx, dy = det[4:6]
                # print(f"dx:{dx} dy:{dy}")
                out_x = -pid_x(dx)  # type: ignore
                if delta_y is None:
                    out_y = 0
                
                else:
                    out_y = kp_y * (dy - delta_y)

                flag_x = x_count(abs(dx) < 0.04)
                flag_y = y_count(abs(dy) < 0.04)
                if delta_y is None:
                    flag_y = True

                if flag_x:
                    out_x = 0
                if flag_y:
                    out_y = 0
                if flag_x and flag_y:
                    # logger.info(f"location{self.get_odometry()} ok, arm_pose{self.arm.x_pose_now}")
                    self.set_velocity(0, 0, 0)
                    self.arm.x_speed(0)
                    # return det[0],det[2]
            else:
                x_count(False)
                y_count(False)
            self.set_velocity(out_x, 0, 0)
            self.arm.x_speed(out_y)
            time.sleep(0.05)

            if time.time() > time_stop:
                self.set_velocity(0, 0, 0)
                self.arm.x_speed(0)
                logger.error("对齐目标超时")
                # logger.info(f"location{self.get_odometry()} ok, arm_pose{self.arm.x_pose_now}")

                try:
                    return det[0], det[2], dy
                except:
                    return (None, None, None)

    def move_to_hannuo(
        self,
        delta_x=-0.0,
        # delta_x=0.02,
        delta_y: Union[float, None] = -0.05,
        label=None,
        time_out=2.0,
        sort_pos=(0, 0),
        num=0,
        ball_no="",
        x_thr=0.06,        # x对齐阈值
        y_thr=0.06,        # y对齐阈值
        lock_thr=0.04,     # x锁阈值（误差<此值停车）
        skip_y=False,      # 跳过y对齐(水塔用)
        validate_frames=True,  # 帧验证：先确认dx稳定再开始控制
        use_hsv=False,     # True=HSV色域提取方块, False=AI模型检测
        hsv_color="blue",  # HSV模式目标颜色: "blue" / "red"
    ):
        """手动PID版：不依赖外部PID库，所有参数在函数内部直接可调。
        hsv_color: 当 use_hsv=True 时指定目标颜色，可选 "blue" 或 "red" """
        # ====== 1. 初始化 ======
        time_stop = time.time() + time_out
        out_x, out_y = 0.0, 0.0
        frame_cnt, step = 0, 0
        x_locked = False
        _tag = f" [{ball_no}]" if ball_no else ""
        print(f"[STEP] 视觉对准开始{_tag} | delta_x={delta_x:.2f} delta_y={delta_y} label={label} arm={self.arm.side} hsv_color={hsv_color}")

        # ====== 2. 手写PID参数（直接在这里改！） ======
        if self.arm.side == "RIGHT":
            kp_y, kp_x, ki_x, kd_x = -0.0, 0.018, 0.0, 0.0
        else:
            kp_y, kp_x, ki_x, kd_x =  0.02, -0.05, 0.0, 0.0
            # kp_y, kp_x, ki_x, kd_x =  0.08, -0.08, -0.4, 0.0

        integral_x  = 0.0      # Σ(ki * error * dt)
        last_err_x  = 0.0      # 上帧误差
        OUT_LIMIT   = 0.02     # out_x 限幅
        I_LIMIT     = 0.01     # |integral| 硬上限
        I_STEP_MAX  = 0.008    # |integral| 单帧增量上限
        DT          = 0.05     # 帧间隔(秒)
        CLEAR_I_ERR = 0.15     # |error|>此值,本帧积分清零
        LOCK_THRESH = lock_thr   # x锁阈值
        ALIGN_X_THR = x_thr    # x对齐判定阈值
        ALIGN_Y_THR = y_thr    # y对齐判定阈值
        hsv_align_count = 0    # HSV模式：连续对准帧计数
        HSV_ALIGN_NEED = 5     # HSV模式需要连续5帧对准
        # ===================================================

        # ---- 帧验证+主循环中 get_blue_center → _get_color_center ----
        _get_center = lambda sd: self._get_color_center(hsv_color, save_debug=sd)
        # ===================================================

        # ====== 帧验证：连续两帧 dx 差≤0.01 才确认稳定，期间车子不动 ======
        if validate_frames and not use_hsv:
            prev_dx = None
            validate_ok = 0          # 连续稳定帧计数
            print(f"[STEP] 帧验证：等待检测稳定（连续2帧dx差≤0.01）再开始控制...")
            while validate_ok < 2:
                if self._stop_flag:
                    self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                    return -1, "None", None
                if time.time() > time_stop:
                    self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                    print(f"[STEP] 帧验证超时")
                    return -1, "None", None

                if use_hsv:
                    dx, _, found, _ = _get_center(True)
                    if not found:
                        prev_dx = None; validate_ok = 0
                        self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                        time.sleep(0.05)
                        continue
                else:
                    dets = self.get_detection_results(sort_pos=sort_pos)
                    if label is not None:
                        dets = [d for d in dets if d[2] == label]
                    if len(dets) > num:
                        dx = dets[num][4]
                    else:
                        prev_dx = None; validate_ok = 0
                        self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                        time.sleep(0.05)
                        continue
                    if prev_dx is not None:
                        diff = abs(dx - prev_dx)
                        if diff <= 0.01:
                            validate_ok += 1
                            print(f"  帧验证 {validate_ok}/2 | dx={dx:.3f} prev={prev_dx:.3f} diff={diff:.3f} ✓")
                        else:
                            validate_ok = 0
                            print(f"  帧抖动 | dx={dx:.3f} prev={prev_dx:.3f} diff={diff:.3f} ✗ 重新计数")
                    prev_dx = dx
                self.set_velocity(0, 0, 0)
                self.arm.x_speed(0)
                time.sleep(0.05)
            print(f"[STEP] 帧验证通过 | dx={prev_dx:.3f} 稳定，开始PID控制")
        # =================================================================

        while True:
            if self._stop_flag:
                self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                print(f"[STEP] 中断 | 总帧={frame_cnt}")
                return -1, "None", None

            if use_hsv:
                dx, dy, found, _ = _get_center(True)
                if found:
                    det = [0, 0, "blue" if hsv_color == "blue" else "red", 0, dx, dy, 0, 0]
                else:
                    det = None
            else:
                dets = self.get_detection_results(sort_pos=sort_pos)
                if label is not None:
                    dets = [d for d in dets if d[2] == label]
                det = dets[num] if len(dets) > num else None

            if det is not None:
                frame_cnt += 1
                out_x = 0.0          # 每帧清零，新算
                dx, dy = det[4:6]
                # self.beep()


                err_abs = abs(dx - delta_x)
                error_x = delta_x - dx           # PID误差 (setpoint - input)
                self._last_err_x = error_x       # 供外部读取最终err_x

                # ---- x锁：OK后锁死，err漂出阈值就解锁 ----
                if x_locked and err_abs > LOCK_THRESH:
                    x_locked = False

                if x_locked:
                    p_term = i_term = d_term = 0.0
                    out_x = 0.0
                elif err_abs < LOCK_THRESH:
                    x_locked = True
                    integral_x = 0.0
                    p_term = i_term = d_term = 0.0
                    out_x = 0.0
                else:
                    # ==== 手动 P ====
                    p_term = kp_x * error_x

                    # ==== 手动 D ====
                    d_term = kd_x * (error_x - last_err_x) / DT
                    last_err_x = error_x

                    # ==== 手动 I（增量限幅+硬限幅） ====
                    # 防积分饱和：P+D 还推不动车时暂停积分，避免车动后过冲
                    _pd_sum = abs(p_term + d_term)
                    if _pd_sum < 0.095:
                        integral_x = 0.0
                    else:
                        i_delta = ki_x * error_x * DT
                        if i_delta > I_STEP_MAX:  i_delta = I_STEP_MAX
                        if i_delta < -I_STEP_MAX: i_delta = -I_STEP_MAX
                        integral_x += i_delta
                    if err_abs > CLEAR_I_ERR: integral_x = 0.0
                    if integral_x > I_LIMIT:  integral_x = I_LIMIT
                    if integral_x < -I_LIMIT: integral_x = -I_LIMIT
                    i_term = integral_x

                    # ==== 输出 = 基速(克服静摩擦) + PID ====
                    BASE_SPEED = 0.0095
                    BASE_SPEED_right = 0.0085
                    if self.arm.side == "RIGHT":
                        out_x = BASE_SPEED_right if error_x > 0 else (-BASE_SPEED_right if error_x < 0 else 0)
                    else:
                        out_x = BASE_SPEED if error_x < 0 else (-BASE_SPEED if error_x > 0 else 0)
                    
                    out_x += p_term + i_term + d_term
                    # 保底：out非零时至少0.0095才能克服静摩擦
                    if out_x > 0 and out_x < 0.0095: out_x = 0.0095
                    if out_x < 0 and out_x > -0.0095: out_x = -0.0095

                    # 钳制
                    MAX_SPEED = 0.02
                    if out_x > MAX_SPEED:  out_x = MAX_SPEED
                    if out_x < -MAX_SPEED: out_x = -MAX_SPEED

                # ---- 机械臂 y ----
                if delta_y is None:
                    out_y = 0.0
                else:
                    LEFTY = 0.003
                    if self.arm.side == "LEFT":
                        out_y = -LEFTY if (dy - delta_y) < 0 else (LEFTY if (dy - delta_y) > 0 else 0)
                    out_y += kp_y * (dy - delta_y)
                    
                    # MAX_SPEED = 0.025
                    if out_y > 0.02:  out_y = 0.02
                    if out_y < -0.02: out_y = -0.02

                # ---- 日志标记 ----
                _dir_cn = "↑前进" if out_x > 0 else "↓后退"
                stage_str = "OK" if err_abs < 0.04 else ("SLOW" if err_abs < 0.1 else "FAST")
                lock_tag = " [LOCKED]" if x_locked else ""
                if frame_cnt == 1:
                    step += 1
                    print(f"[STEP {step}] 首次检测 | err_x={error_x:+.4f} dx={dx:+.3f}")

                # ---- 对齐退出（RIGHT臂只对齐x，LEFT臂对齐x+y） ----
                _skip_y = (self.arm.side == "RIGHT") or skip_y
                aligned_now = abs(dx - delta_x) < ALIGN_X_THR and (_skip_y or delta_y is None or abs(dy - delta_y) < ALIGN_Y_THR)
                if aligned_now:
                    if use_hsv:
                        hsv_align_count += 1
                        if hsv_align_count < HSV_ALIGN_NEED:
                            out_x = 0.0; out_y = 0.0
                            print(f"  [HSV对准] {hsv_align_count}/{HSV_ALIGN_NEED} 连续对准中...")
                        else:
                            out_x = 0.0; out_y = 0.0
                            integral_x = 0.0; last_err_x = 0.0
                            self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                            for _ in range(3):
                                self.beep()
                                time.sleep(0.1)
                            step += 1
                            print(f"[STEP {step}] 对齐成功 | 退出 (HSV连续{HSV_ALIGN_NEED}帧)")
                            print(f"--- 视觉对准完成 ---")
                            self.save_align_debug_img(delta_x, dx, det[2], dy,
                                save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                                final=True, frame=frame_cnt, stage=stage_str, direction="F" if out_x>0 else "B", out_x=out_x, out_y=out_y, delta_y=delta_y, ball_no=ball_no)
                            return det[0], det[2], dy
                    else:
                        out_x = 0.0; out_y = 0.0
                        integral_x = 0.0; last_err_x = 0.0
                        self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                        for _ in range(3):
                            self.beep()
                            time.sleep(0.1)
                        step += 1
                        print(f"[STEP {step}] 对齐成功 | 退出")
                        print(f"--- 视觉对准完成 ---")
                        self.save_align_debug_img(delta_x, dx, det[2], dy,
                            save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                            final=True, frame=frame_cnt, stage=stage_str, direction="F" if out_x>0 else "B", out_x=out_x, out_y=out_y, delta_y=delta_y, ball_no=ball_no)
                        return det[0], det[2], dy
                else:
                    if use_hsv:
                        hsv_align_count = 0   # 不对准则归零

                # ---- 每帧日志+截图 ----
                _ey = (dy - delta_y) if delta_y is not None else 0.0
                print(f"  [视觉对准] #{frame_cnt} {stage_str}{lock_tag} | err_x={error_x:+.4f} err_y={_ey:+.4f} dx={dx:+.3f} {_dir_cn} out={out_x:+.4f} out_y={out_y:+.4f} | P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f}")
                self.save_align_debug_img(delta_x, dx, det[2], dy,
                    save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                    frame=frame_cnt, stage=stage_str+lock_tag, direction="F" if out_x>0 else "B", out_x=out_x, delta_y=delta_y, ball_no=ball_no)

                # --- 手动确认（需要时取消注释） ---
                # if out_x != 0 or out_y != 0:
                #     _dir = "前进" if out_x > 0 else "后退"
                #     print(f"  [MANUAL] ---- 待执行移动 ----")
                #     print(f"    err_x={dx-delta_x:+.4f}  err_y={dy-delta_y:+.4f}  {_dir}")
                #     print(f"    P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f}  out={out_x:+.4f}")
                #     input(f"    按回车执行...")
                #     self.beep()
            else:
                pass

            self.beep()
            self.set_velocity(out_x, 0, 0)
            self.arm.x_speed(out_y)
            # time.sleep(DT)
            time.sleep(0.025)

            if time.time() > time_stop:
                self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                step += 1
                print(f"[STEP {step}] 超时 | 总帧={frame_cnt}")
                try:
                    self.save_align_debug_img(delta_x, dx, det[2], dy, save_dir="det_target_debug")
                    return det[0], det[2], dy
                except:
                    return (None, None, None)
    def move_to_wt3(
        self,
        delta_x=-0.0,
        # delta_x=0.02,
        delta_y: Union[float, None] = -0.05,
        label=None,
        time_out=2.0,
        sort_pos=(0, 0),
        num=0,
        ball_no="",
        x_thr=0.06,        # x对齐阈值
        y_thr=0.06,        # y对齐阈值
        lock_thr=0.02,     # x锁阈值（误差<此值停车）
        skip_y=False,      # 跳过y对齐(水塔用)
        validate_frames=True,  # 帧验证：先确认dx稳定再开始控制
        use_hsv=False,     # True=HSV色域提取方块, False=AI模型检测
        color_mode=1,      # 1=浅蓝, 2=深蓝+黄色
        save_images=0,     # 0=不保存图片, 1=保存调试图片
    ):
        """手动PID版：不依赖外部PID库，所有参数在函数内部直接可调。
        color_mode: 当 use_hsv=True 时选择颜色模式:
            1 = 浅蓝色 (默认, 水块)
            2 = 深蓝 + 黄色 (汉诺塔)
        save_images: 0=不保存调试图(默认), 1=保存HSV过程图和对齐图"""
        save_images = bool(save_images)

        # ---- 颜色模式映射 ----
        _color_map = {1: "light_blue", 2: "dark_blue+yellow", 3: "gray"}
        _hsv_color = _color_map.get(color_mode, "light_blue")

        if color_mode == 2 and use_hsv:
            print(f"[STEP] 模式2：深蓝+黄色双色同检（免AI预判）")

        _get_center = lambda sd: self._get_color_center(_hsv_color, save_debug=sd)
        # ====== 1. 初始化 ======
        time_stop = time.time() + time_out
        out_x, out_y = 0.0, 0.0
        frame_cnt, step = 0, 0
        _tag = f" [{ball_no}]" if ball_no else ""
        self._debug_tag = ball_no if ball_no else ""
        print(f"[STEP] 视觉对准开始{_tag} | delta_x={delta_x:.2f} delta_y={delta_y} label={label} arm={self.arm.side} hsv={_hsv_color} save_img={save_images}")

        # ---- 每次调用必须重置状态，避免沿用上次成功结果 ----
        self.last_wt3_result = {
            "success": False, "timed_out": False, "reason": "running",
            "dx": None, "dy": None, "error_x": None,
            "frame_seq": 0, "capture_time": 0.0,
        }
        last_valid = None   # 本轮最后一次有效检测

        # ====== 2. 手写PID参数 ======
        if self.arm.side == "RIGHT":
            kp_y, kp_x = -0.0, 0.018
        else:
            kp_y, kp_x = 0.01, -0.03
        ki_x, kd_x = 0.0, 0.0

        # 精调脉冲参数（需实车标定）
        PULSE_SPEED = 0.009
        PULSE_DURATION = 0.07
        PULSE_WAIT = 0.15
        PULSE_THR = 0.04

        integral_x  = 0.0
        last_err_x  = 0.0
        OUT_LIMIT   = 0.015
        ALIGN_X_THR = x_thr
        ALIGN_Y_THR = y_thr
        # 带回差稳定判定：lock_thr 真正生效
        STABLE_IN   = min(lock_thr, ALIGN_X_THR)
        STABLE_OUT  = ALIGN_X_THR
        stable_candidate = False
        hsv_align_count = 0
        HSV_ALIGN_NEED = 3
        last_frame_seq = 0
        loss_start_time = None
        LOSS_TIMEOUT = 3.0
        last_log_time = 0.0
        LOG_INTERVAL = 0.2
        # ===================================================

        _HSV_LABEL_MAP = {"dark_blue": "ball_blue", "yellow": "ball_yellow",
                          "light_blue": "ball_blue", "red": "ball_red"}

        # ====== 内部收尾函数 ======
        def _finish_stopped():
            """人工/系统中断：立即停车，不继续抓取"""
            self.set_velocity(0, 0, 0)
            self.arm.x_speed(0)
            self.last_wt3_result = {
                "success": False, "timed_out": False, "reason": "stopped",
                "dx": None, "dy": None, "error_x": None,
                "frame_seq": 0, "capture_time": 0.0,
            }
            return -1, "None", None

        def _finish_visual_timeout(reason):
            """普通视觉超时：停车，保留last_valid，按任务规则继续抓取"""
            self.set_velocity(0, 0, 0)
            self.arm.x_speed(0)

            if last_valid is not None:
                self.last_wt3_result = {
                    "success": False, "timed_out": True,
                    "reason": f"{reason}_use_last_valid",
                    "dx": last_valid["dx"], "dy": last_valid["dy"],
                    "error_x": last_valid["error_x"],
                    "frame_seq": last_valid["frame_seq"],
                    "capture_time": last_valid["capture_time"],
                }
                self._last_err_x = last_valid["error_x"]
                return last_valid["cls_id"], last_valid["label"], last_valid["dy"]

            self.last_wt3_result = {
                "success": False, "timed_out": True,
                "reason": f"{reason}_no_valid_detection",
                "dx": None, "dy": None, "error_x": 0.0,
                "frame_seq": 0, "capture_time": 0.0,
            }
            self._last_err_x = 0.0
            return 0, "None", None
        # ===================================================

        # ====== 帧验证（AI模式） ======
        if validate_frames and not use_hsv:
            prev_dx = None
            validate_ok = 0
            print(f"[STEP] 帧验证：等待检测稳定（连续2帧dx差≤0.01）再开始控制...")
            while validate_ok < 2:
                if self._stop_flag:
                    return _finish_stopped()
                if time.time() > time_stop:
                    return _finish_visual_timeout("frame_validation_timeout")

                dets = self.get_detection_results(sort_pos=sort_pos)
                if label is not None:
                    dets = [d for d in dets if d[2] == label]
                if len(dets) > num:
                    dx = dets[num][4]
                    # 更新 last_valid（帧验证阶段也记录）
                    last_valid = {
                        "cls_id": dets[num][0], "label": dets[num][2],
                        "dx": dx, "dy": dets[num][5],
                        "error_x": delta_x - dx,
                        "frame_seq": getattr(self, '_last_det_seq', 0),
                        "capture_time": getattr(self, '_last_det_time', 0.0),
                    }
                else:
                    prev_dx = None; validate_ok = 0
                    self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                    time.sleep(0.05)
                    continue
                if prev_dx is not None:
                    diff = abs(dx - prev_dx)
                    if diff <= 0.01:
                        validate_ok += 1
                        print(f"  帧验证 {validate_ok}/2 | dx={dx:.3f} prev={prev_dx:.3f} diff={diff:.3f} ✓")
                    else:
                        validate_ok = 0
                        print(f"  帧抖动 | dx={dx:.3f} prev={prev_dx:.3f} diff={diff:.3f} ✗ 重新计数")
                prev_dx = dx
                self.set_velocity(0, 0, 0)
                self.arm.x_speed(0)
                time.sleep(0.05)
            print(f"[STEP] 帧验证通过 | dx={prev_dx:.3f} 稳定，开始PID控制")
        # =================================================================

        # 主控循环
        while True:
            if self._stop_flag:
                return _finish_stopped()

            # ---- 获取检测结果 ----
            if use_hsv:
                hsv_frame, cur_seq, cur_cap_time = self.cap_front.read_with_meta()

                if hsv_frame is None:
                    # 摄像头未就绪，作为无检测处理
                    det = None
                else:
                    # 元数据无条件更新（不受 save_images 控制）
                    self._last_align_seq = cur_seq
                    self._last_align_time = cur_cap_time
                    if save_images:
                        self._last_align_img = hsv_frame.copy()

                    dx, dy, found, color_tag = self._get_color_center_direct(_hsv_color, image=hsv_frame, save_debug=save_images)
                    if found:
                        det_label = _HSV_LABEL_MAP.get(color_tag, _hsv_color)
                        det = [0, 0, det_label, 0, dx, dy, 0, 0]
                    else:
                        det = None
            else:
                dets = self.get_detection_results(sort_pos=sort_pos)
                if label is not None:
                    dets = [d for d in dets if d[2] == label]
                det = dets[num] if len(dets) > num else None
                cur_seq = getattr(self, '_last_det_seq', 0)
                cur_cap_time = getattr(self, '_last_det_time', 0.0)

            # ---- 丢失目标：立即停车，重置状态 ----
            if det is None:
                self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                out_x = 0.0; out_y = 0.0
                integral_x = 0.0; last_err_x = 0.0
                hsv_align_count = 0
                stable_candidate = False
                if loss_start_time is None:
                    loss_start_time = time.time()
                elif time.time() - loss_start_time > LOSS_TIMEOUT:
                    step += 1
                    print(f"[STEP {step}] 目标丢失超时({LOSS_TIMEOUT}s) | 总帧={frame_cnt}")
                    return _finish_visual_timeout("target_loss_timeout")
                time.sleep(0.05)
                continue
            loss_start_time = None

            # ---- 有目标 ----
            frame_cnt += 1
            out_x = 0.0
            dx, dy = det[4:6]

            err_abs = abs(dx - delta_x)
            error_x = delta_x - dx
            self._last_err_x = error_x

            # 更新 last_valid（每轮有效检测都记录）
            last_valid = {
                "cls_id": det[0], "label": det[2],
                "dx": dx, "dy": dy,
                "error_x": error_x,
                "frame_seq": cur_seq, "capture_time": cur_cap_time,
            }

            # ---- 带回差稳定判定（帧序号变化才计入） ----
            is_new_frame = (cur_seq != last_frame_seq)
            last_frame_seq = cur_seq
            x_aligned_now = err_abs < ALIGN_X_THR

            if stable_candidate and is_new_frame and not x_aligned_now:
                stable_candidate = False
            if is_new_frame and err_abs <= STABLE_IN:
                stable_candidate = True

            # ---- 控制输出 ----
            p_term = i_term = d_term = 0.0

            if stable_candidate:
                out_x = 0.0
            elif err_abs <= PULSE_THR:
                # 精调区：短脉冲
                if self.arm.side == "RIGHT":
                    _dir = 1 if error_x > 0 else (-1 if error_x < 0 else 0)
                else:
                    _dir = 1 if error_x < 0 else (-1 if error_x > 0 else 0)
                out_x = PULSE_SPEED * _dir
                self.set_velocity(out_x, 0, 0)
                self.arm.x_speed(0)
                time.sleep(PULSE_DURATION)
                self.set_velocity(0, 0, 0)
                self.arm.x_speed(0)
                out_x = 0.0
                time.sleep(PULSE_WAIT)
                if time.time() > time_stop:
                    return _finish_visual_timeout("pulse_timeout")
                continue
            else:
                # 大偏差区：连续PID
                p_term = kp_x * error_x
                BASE_SPEED = 0.0088
                BASE_SPEED_right = 0.0085
                if self.arm.side == "RIGHT":
                    out_x = BASE_SPEED_right if error_x > 0 else (-BASE_SPEED_right if error_x < 0 else 0)
                else:
                    out_x = BASE_SPEED if error_x < 0 else (-BASE_SPEED if error_x > 0 else 0)
                out_x += p_term
                if out_x > 0 and out_x < 0.006: out_x = 0.006
                if out_x < 0 and out_x > -0.006: out_x = -0.006
                if out_x > OUT_LIMIT:  out_x = OUT_LIMIT
                if out_x < -OUT_LIMIT: out_x = -OUT_LIMIT
                if error_x < -0.4:
                    out_x = 0.03

            # ---- 机械臂 y ----
            if delta_y is None:
                out_y = 0.0
            else:
                LEFTY = 0.003
                if self.arm.side == "LEFT":
                    out_y = -LEFTY if (dy - delta_y) < 0 else (LEFTY if (dy - delta_y) > 0 else 0)
                out_y += kp_y * (dy - delta_y)
                if out_y > 0.02:  out_y = 0.02
                if out_y < -0.02: out_y = -0.02

            # ---- 日志限频 ----
            _dir_cn = "↑前进" if out_x > 0 else ("↓后退" if out_x < 0 else "·停车")
            stage_str = "STABLE" if stable_candidate else ("PULSE" if err_abs <= PULSE_THR else ("OK" if err_abs < 0.06 else ("SLOW" if err_abs < 0.1 else "FAST")))
            if frame_cnt == 1:
                step += 1
                print(f"[STEP {step}] 首次检测 | err_x={error_x:+.4f} dx={dx:+.3f} seq={cur_seq}")

            _now = time.time()
            if _now - last_log_time >= LOG_INTERVAL:
                _ey = (dy - delta_y) if delta_y is not None else 0.0
                print(f"  [视觉对准] #{frame_cnt} seq={cur_seq} {stage_str} | err_x={error_x:+.4f} err_y={_ey:+.4f} dx={dx:+.3f} {_dir_cn} out={out_x:+.4f} out_y={out_y:+.4f} | P={p_term:+.4f}")
                last_log_time = _now

            # ---- 对齐判定 ----
            _skip_y = (self.arm.side == "RIGHT") or skip_y
            aligned_now = abs(dx - delta_x) < ALIGN_X_THR and (_skip_y or delta_y is None or abs(dy - delta_y) < ALIGN_Y_THR)

            if aligned_now and stable_candidate:
                if use_hsv and is_new_frame:
                    hsv_align_count += 1
                    if hsv_align_count < HSV_ALIGN_NEED:
                        out_x = 0.0; out_y = 0.0
                        print(f"  [HSV对准] {hsv_align_count}/{HSV_ALIGN_NEED} 连续对准中... seq={cur_seq}")
                    else:
                        # ====== 最终复核：直接用 read_with_meta() 取带元数据新帧 ======
                        out_x = 0.0; out_y = 0.0
                        self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                        time.sleep(0.2)

                        stop_seq = cur_seq
                        final_samples = []
                        final_seen = set()
                        t_final = time.monotonic()

                        while len(final_samples) < 5 and time.monotonic() - t_final < 2.0:
                            f_img, f_seq, f_cap_time = self.cap_front.read_with_meta()
                            if f_img is None:
                                time.sleep(0.02)
                                continue
                            if f_seq <= stop_seq or f_seq in final_seen:
                                time.sleep(0.02)
                                continue

                            f_dx, f_dy, f_ok, f_color = self._get_color_center_direct(
                                _hsv_color, image=f_img, save_debug=False)
                            final_seen.add(f_seq)

                            if not f_ok:
                                continue

                            final_samples.append({
                                "dx": f_dx, "dy": f_dy,
                                "seq": f_seq, "capture_time": f_cap_time,
                                "image": f_img.copy() if save_images else None,
                            })
                            print(f"    [复核] {len(final_samples)}/5 dx={f_dx:+.4f} seq={f_seq}")

                        if len(final_samples) >= 3:
                            samples_by_dx = sorted(final_samples, key=lambda item: item["dx"])
                            median_sample = samples_by_dx[len(samples_by_dx) // 2]
                            final_dx = median_sample["dx"]
                            final_dy = median_sample["dy"]
                            final_seq = median_sample["seq"]
                            final_cap_time = median_sample["capture_time"]
                            final_err = abs(final_dx - delta_x)
                            print(f"  [复核结果] dx中位数={final_dx:+.4f} error={final_err:+.4f} seq={final_seq} (样本={len(final_samples)})")

                            if final_err <= ALIGN_X_THR:
                                for _ in range(3):
                                    self.beep()
                                    time.sleep(0.1)
                                step += 1
                                print(f"[STEP {step}] 对齐成功(复核通过) | 退出")
                                print(f"--- 视觉对准完成 ---")
                                self.last_wt3_result = {
                                    "success": True, "timed_out": False,
                                    "reason": "settled_review_passed",
                                    "dx": final_dx, "dy": final_dy,
                                    "error_x": delta_x - final_dx,
                                    "frame_seq": final_seq,
                                    "capture_time": final_cap_time,
                                }
                                self._last_err_x = delta_x - final_dx
                                if save_images:
                                    self._last_align_img = median_sample["image"]
                                    self._last_align_seq = final_seq
                                    self._last_align_time = final_cap_time
                                    self.save_align_debug_img_hsv(delta_x, final_dx, det[2], final_dy,
                                        final=True, frame=frame_cnt, stage="FINAL", out_x=0.0, out_y=0.0,
                                        delta_y=delta_y, ball_no=ball_no)
                                return det[0], det[2], final_dy
                            else:
                                print(f"  [复核失败] 中位误差{final_err:+.4f}>{ALIGN_X_THR}，继续修正...")
                                stable_candidate = False
                                hsv_align_count = 0
                        else:
                            print(f"  [复核失败] 有效帧不足({len(final_samples)}<3)，继续修正...")
                            stable_candidate = False
                            hsv_align_count = 0
                elif not use_hsv:
                    # AI模式：直接复核
                    out_x = 0.0; out_y = 0.0
                    self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                    time.sleep(0.2)

                    final_samples = []; final_seen = set()
                    t_final = time.monotonic()
                    while len(final_samples) < 5 and time.monotonic() - t_final < 2.0:
                        _dets = self.get_detection_results(sort_pos=sort_pos)
                        if label is not None:
                            _dets = [d for d in _dets if d[2] == label]
                        if len(_dets) > num:
                            _f_dx, _f_dy = _dets[num][4], _dets[num][5]
                            _f_seq = getattr(self, '_last_det_seq', 0)
                            _f_cap_time = getattr(self, '_last_det_time', 0.0)
                            if _f_seq not in final_seen:
                                final_seen.add(_f_seq)
                                final_samples.append({
                                    "dx": _f_dx, "dy": _f_dy,
                                    "seq": _f_seq, "capture_time": _f_cap_time,
                                })
                                print(f"    [复核] {len(final_samples)}/5 dx={_f_dx:+.4f} seq={_f_seq}")
                        time.sleep(0.05)

                    if len(final_samples) >= 3:
                        samples_by_dx = sorted(final_samples, key=lambda item: item["dx"])
                        median_sample = samples_by_dx[len(samples_by_dx) // 2]
                        final_dx = median_sample["dx"]
                        final_dy = median_sample["dy"]
                        final_seq = median_sample["seq"]
                        final_cap_time = median_sample["capture_time"]
                        final_err = abs(final_dx - delta_x)
                        if final_err <= ALIGN_X_THR:
                            for _ in range(3):
                                self.beep()
                                time.sleep(0.1)
                            step += 1
                            print(f"[STEP {step}] 对齐成功(复核通过) | 退出")
                            print(f"--- 视觉对准完成 ---")
                            self.last_wt3_result = {
                                "success": True, "timed_out": False,
                                "reason": "settled_review_passed",
                                "dx": final_dx, "dy": final_dy,
                                "error_x": delta_x - final_dx,
                                "frame_seq": final_seq,
                                "capture_time": final_cap_time,
                            }
                            self._last_err_x = delta_x - final_dx
                            if save_images:
                                self.save_align_debug_img(delta_x, final_dx, det[2], final_dy,
                                    save_dir="det_target_debug", final=True, frame=frame_cnt, stage="FINAL",
                                    out_x=0.0, out_y=0.0, delta_y=delta_y, ball_no=ball_no)
                            return det[0], det[2], final_dy
                        else:
                            print(f"  [复核失败] 中位误差{final_err:+.4f}>{ALIGN_X_THR}，继续修正...")
                            stable_candidate = False
                    else:
                        print(f"  [复核失败] 有效帧不足({len(final_samples)}<3)，继续修正...")
                        stable_candidate = False
            else:
                if use_hsv and is_new_frame and not aligned_now:
                    hsv_align_count = 0

            # ---- 周期保存（HSV/AI 分支） ----
            if save_images and frame_cnt % 5 == 0:
                if use_hsv:
                    self.save_align_debug_img_hsv(delta_x, dx, det[2], dy,
                        final=False, frame=frame_cnt, stage=stage_str, out_x=out_x, out_y=out_y,
                        delta_y=delta_y, ball_no=ball_no)
                else:
                    self.save_align_debug_img(delta_x, dx, det[2], dy,
                        save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                        frame=frame_cnt, stage=stage_str, direction="F" if out_x>0 else "B",
                        out_x=out_x, out_y=out_y, delta_y=delta_y, ball_no=ball_no)

            # ---- 下发速度 ----
            self.set_velocity(out_x, 0, 0)
            self.arm.x_speed(out_y)
            time.sleep(0.025)

            if time.time() > time_stop:
                step += 1
                print(f"[STEP {step}] 超时 | 总帧={frame_cnt}")
                return _finish_visual_timeout("align_timeout")

    def move_to_detection_target_wt3(
        self,
        delta_x=-0.0,
        # delta_x=0.02,
        delta_y: Union[float, None] = -0.05,
        label=None,
        time_out=2.0,
        sort_pos=(0, 0),
        num=0,
        ball_no="",
        x_thr=0.06,        # x对齐阈值
        y_thr=0.06,        # y对齐阈值
        lock_thr=0.04,     # x锁阈值（误差<此值停车）
        skip_y=False,      # 跳过y对齐(水塔用)
        validate_frames=True,  # 帧验证：先确认dx稳定再开始控制
        use_hsv=False,     # True=HSV色域提取方块, False=AI模型检测
        color_mode=1,      # 1=浅蓝, 2=深蓝+黄色
    ):
        """手动PID版：不依赖外部PID库，所有参数在函数内部直接可调。
        color_mode: 当 use_hsv=True 时选择颜色模式:
            1 = 浅蓝色 (默认, 水块)
            2 = 深蓝 + 黄色 (汉诺塔)"""
        # ---- 颜色模式映射 ----
        _color_map = {1: "light_blue", 2: "dark_blue+yellow", 3: "gray"}
        _hsv_color = _color_map.get(color_mode, "light_blue")
        _get_center = lambda sd: self._get_color_center(_hsv_color, save_debug=sd)
        # ====== 1. 初始化 ======
        time_stop = time.time() + time_out
        out_x, out_y = 0.0, 0.0
        frame_cnt, step = 0, 0
        x_locked = False
        _tag = f" [{ball_no}]" if ball_no else ""
        print(f"[STEP] 视觉对准开始{_tag} | delta_x={delta_x:.2f} delta_y={delta_y} label={label} arm={self.arm.side}")

        # ====== 2. 手写PID参数（直接在这里改！） ======
        if self.arm.side == "RIGHT":
            kp_y, kp_x, ki_x, kd_x = -0.0, 0.018, 0.0, 0.0
        else:
            kp_y, kp_x, ki_x, kd_x =  0.02, -0.05, 0.0, 0.0
            # kp_y, kp_x, ki_x, kd_x =  0.08, -0.08, -0.4, 0.0

        integral_x  = 0.0      # Σ(ki * error * dt)
        last_err_x  = 0.0      # 上帧误差
        OUT_LIMIT   = 0.02     # out_x 限幅
        I_LIMIT     = 0.01     # |integral| 硬上限
        I_STEP_MAX  = 0.008    # |integral| 单帧增量上限
        DT          = 0.05     # 帧间隔(秒)
        CLEAR_I_ERR = 0.15     # |error|>此值,本帧积分清零
        LOCK_THRESH = lock_thr   # x锁阈值
        ALIGN_X_THR = x_thr    # x对齐判定阈值
        ALIGN_Y_THR = y_thr    # y对齐判定阈值
        hsv_align_count = 0    # HSV模式：连续对准帧计数
        HSV_ALIGN_NEED = 5     # HSV模式需要连续5帧对准
        # ===================================================

        # ====== 帧验证：连续两帧 dx 差≤0.01 才确认稳定，期间车子不动 ======
        if validate_frames and not use_hsv:
            prev_dx = None
            validate_ok = 0          # 连续稳定帧计数
            print(f"[STEP] 帧验证：等待检测稳定（连续2帧dx差≤0.01）再开始控制...")
            while validate_ok < 2:
                if self._stop_flag:
                    self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                    return -1, "None", None
                if time.time() > time_stop:
                    self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                    print(f"[STEP] 帧验证超时")
                    return -1, "None", None

                if use_hsv:
                    dx, _, found, _ = _get_center(True)
                    if not found:
                        prev_dx = None; validate_ok = 0
                        self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                        time.sleep(0.05)
                        continue
                else:
                    dets = self.get_detection_results(sort_pos=sort_pos)
                    if label is not None:
                        dets = [d for d in dets if d[2] == label]
                    if len(dets) > num:
                        dx = dets[num][4]
                    else:
                        prev_dx = None; validate_ok = 0
                        self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                        time.sleep(0.05)
                        continue
                    if prev_dx is not None:
                        diff = abs(dx - prev_dx)
                        if diff <= 0.01:
                            validate_ok += 1
                            print(f"  帧验证 {validate_ok}/2 | dx={dx:.3f} prev={prev_dx:.3f} diff={diff:.3f} ✓")
                        else:
                            validate_ok = 0
                            print(f"  帧抖动 | dx={dx:.3f} prev={prev_dx:.3f} diff={diff:.3f} ✗ 重新计数")
                    prev_dx = dx
                self.set_velocity(0, 0, 0)
                self.arm.x_speed(0)
                time.sleep(0.05)
            print(f"[STEP] 帧验证通过 | dx={prev_dx:.3f} 稳定，开始PID控制")
        # =================================================================

        while True:
            if self._stop_flag:
                self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                print(f"[STEP] 中断 | 总帧={frame_cnt}")
                return -1, "None", None

            if use_hsv:
                dx, dy, found, _ = _get_center(True)
                if found:
                    det = [0, 0, _hsv_color, 0, dx, dy, 0, 0]
                else:
                    det = None
            else:
                dets = self.get_detection_results(sort_pos=sort_pos)
                if label is not None:
                    dets = [d for d in dets if d[2] == label]
                det = dets[num] if len(dets) > num else None

            if det is not None:
                frame_cnt += 1
                out_x = 0.0          # 每帧清零，新算
                dx, dy = det[4:6]
                # self.beep()


                err_abs = abs(dx - delta_x)
                error_x = delta_x - dx           # PID误差 (setpoint - input)
                self._last_err_x = error_x       # 供外部读取最终err_x

                # ---- x锁：OK后锁死，err漂出阈值就解锁 ----
                if x_locked and err_abs > LOCK_THRESH:
                    x_locked = False

                if x_locked:
                    p_term = i_term = d_term = 0.0
                    out_x = 0.0
                elif err_abs < LOCK_THRESH:
                    x_locked = True
                    integral_x = 0.0
                    p_term = i_term = d_term = 0.0
                    out_x = 0.0
                else:
                    # ==== 手动 P ====
                    p_term = kp_x * error_x

                    # ==== 手动 D ====
                    d_term = kd_x * (error_x - last_err_x) / DT
                    last_err_x = error_x

                    # ==== 手动 I（增量限幅+硬限幅） ====
                    # 防积分饱和：P+D 还推不动车时暂停积分，避免车动后过冲
                    _pd_sum = abs(p_term + d_term)
                    if _pd_sum < 0.095:
                        integral_x = 0.0
                    else:
                        i_delta = ki_x * error_x * DT
                        if i_delta > I_STEP_MAX:  i_delta = I_STEP_MAX
                        if i_delta < -I_STEP_MAX: i_delta = -I_STEP_MAX
                        integral_x += i_delta
                    if err_abs > CLEAR_I_ERR: integral_x = 0.0
                    if integral_x > I_LIMIT:  integral_x = I_LIMIT
                    if integral_x < -I_LIMIT: integral_x = -I_LIMIT
                    i_term = integral_x

                    # ==== 输出 = 基速(克服静摩擦) + PID ====
                    BASE_SPEED = 0.0095
                    BASE_SPEED_right = 0.0085
                    if self.arm.side == "RIGHT":
                        out_x = BASE_SPEED_right if error_x > 0 else (-BASE_SPEED_right if error_x < 0 else 0)
                    else:
                        out_x = BASE_SPEED if error_x < 0 else (-BASE_SPEED if error_x > 0 else 0)
                    
                    out_x += p_term + i_term + d_term
                    # 保底：out非零时至少0.0095才能克服静摩擦
                    if out_x > 0 and out_x < 0.0095: out_x = 0.0095
                    if out_x < 0 and out_x > -0.0095: out_x = -0.0095

                    # 钳制
                    MAX_SPEED = 0.02
                    if out_x > MAX_SPEED:  out_x = MAX_SPEED
                    if out_x < -MAX_SPEED: out_x = -MAX_SPEED

                # ---- 机械臂 y ----
                if delta_y is None:
                    out_y = 0.0
                else:
                    LEFTY = 0.003
                    if self.arm.side == "LEFT":
                        out_y = -LEFTY if (dy - delta_y) < 0 else (LEFTY if (dy - delta_y) > 0 else 0)
                    out_y += kp_y * (dy - delta_y)
                    
                    # MAX_SPEED = 0.025
                    if out_y > 0.02:  out_y = 0.02
                    if out_y < -0.02: out_y = -0.02

                # ---- 日志标记 ----
                _dir_cn = "↑前进" if out_x > 0 else "↓后退"
                stage_str = "OK" if err_abs < 0.04 else ("SLOW" if err_abs < 0.1 else "FAST")
                lock_tag = " [LOCKED]" if x_locked else ""
                if frame_cnt == 1:
                    step += 1
                    print(f"[STEP {step}] 首次检测 | err_x={error_x:+.4f} dx={dx:+.3f}")

                # ---- 对齐退出（RIGHT臂只对齐x，LEFT臂对齐x+y） ----
                _skip_y = (self.arm.side == "RIGHT") or skip_y
                aligned_now = abs(dx - delta_x) < ALIGN_X_THR and (_skip_y or delta_y is None or abs(dy - delta_y) < ALIGN_Y_THR)
                if aligned_now:
                    if use_hsv:
                        hsv_align_count += 1
                        if hsv_align_count < HSV_ALIGN_NEED:
                            out_x = 0.0; out_y = 0.0
                            print(f"  [HSV对准] {hsv_align_count}/{HSV_ALIGN_NEED} 连续对准中...")
                        else:
                            out_x = 0.0; out_y = 0.0
                            integral_x = 0.0; last_err_x = 0.0
                            self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                            for _ in range(3):
                                self.beep()
                                time.sleep(0.1)
                            step += 1
                            print(f"[STEP {step}] 对齐成功 | 退出 (HSV连续{HSV_ALIGN_NEED}帧)")
                            print(f"--- 视觉对准完成 ---")
                            self.save_align_debug_img(delta_x, dx, det[2], dy,
                                save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                                final=True, frame=frame_cnt, stage=stage_str, direction="F" if out_x>0 else "B", out_x=out_x, out_y=out_y, delta_y=delta_y, ball_no=ball_no)
                            return det[0], det[2], dy
                    else:
                        out_x = 0.0; out_y = 0.0
                        integral_x = 0.0; last_err_x = 0.0
                        self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                        for _ in range(3):
                            self.beep()
                            time.sleep(0.1)
                        step += 1
                        print(f"[STEP {step}] 对齐成功 | 退出")
                        print(f"--- 视觉对准完成 ---")
                        self.save_align_debug_img(delta_x, dx, det[2], dy,
                            save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                            final=True, frame=frame_cnt, stage=stage_str, direction="F" if out_x>0 else "B", out_x=out_x, out_y=out_y, delta_y=delta_y, ball_no=ball_no)
                        return det[0], det[2], dy
                else:
                    if use_hsv:
                        hsv_align_count = 0   # 不对准则归零

                # ---- 每帧日志+截图 ----
                _ey = (dy - delta_y) if delta_y is not None else 0.0
                print(f"  [视觉对准] #{frame_cnt} {stage_str}{lock_tag} | err_x={error_x:+.4f} err_y={_ey:+.4f} dx={dx:+.3f} {_dir_cn} out={out_x:+.4f} out_y={out_y:+.4f} | P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f}")
                self.save_align_debug_img(delta_x, dx, det[2], dy,
                    save_dir="det_target_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                    frame=frame_cnt, stage=stage_str+lock_tag, direction="F" if out_x>0 else "B", out_x=out_x, delta_y=delta_y, ball_no=ball_no)

                # --- 手动确认（需要时取消注释） ---
                # if out_x != 0 or out_y != 0:
                #     _dir = "前进" if out_x > 0 else "后退"
                #     print(f"  [MANUAL] ---- 待执行移动 ----")
                #     print(f"    err_x={dx-delta_x:+.4f}  err_y={dy-delta_y:+.4f}  {_dir}")
                #     print(f"    P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f}  out={out_x:+.4f}")
                #     input(f"    按回车执行...")
                #     self.beep()
            else:
                pass

            self.beep()
            self.set_velocity(out_x, 0, 0)
            self.arm.x_speed(out_y)
            # time.sleep(DT)
            time.sleep(0.025)

            if time.time() > time_stop:
                self.set_velocity(0, 0, 0); self.arm.x_speed(0)
                step += 1
                print(f"[STEP {step}] 超时 | 总帧={frame_cnt}")
                try:
                    self.save_align_debug_img(delta_x, dx, det[2], dy, save_dir="det_target_debug")
                    return det[0], det[2], dy
                except:
                    return (None, None, None)
       
    def move_to_detection_target_simple(
        self,
        delta_x=0.0,
        delta_y: Union[float, None] = 0.0,
        label=None,
        time_out=2.0,
        sort_pos=(0, 0),
        num=0,
    ):
        """
        前往目标位置

        参数:
            cls_id : 指定检测目标的 cls_id，默认None为距离中心最近的目标
            time_out: 设置超时时间
            包含目标检测信息的列表，格式为 [cls_id, obj_id,label, score, x_c, y_c, w, h]
        """
        time_stop = time.time() + time_out
        x_count = CountRecord(3)
        y_count = CountRecord(3)

        # pid_x.output_limits((-0.7, 0.7))

        out_x = 0
        out_y = 0
        # print(f"手柄方向：{self.arm.side}")
        if self.arm.side == "RIGHT":
            kp_y = -0.2
            kp_x = -0.25
            ki_x = -0.05
        else:
            kp_y = 0.2
            kp_x = 0.25
            ki_x = 0.05

        pid_x = PID(kp_x, ki_x)
        pid_x.setpoint = delta_x
        while True:
            if self._stop_flag:
                self.set_velocity(0, 0, 0)
                self.arm.x_speed(0)
                return -1, "None"

            dets = self.get_detection_results(sort_pos=sort_pos)

            if label is not None:
                dets = [item for item in dets if item[2] == label]

            if len(dets) > num:
                det = dets[num]
                dx, dy = det[4:6]
                # print(f"dx:{dx} dy:{dy}")
                out_x = -pid_x(dx)  # type: ignore
                if delta_y is None:
                    out_y = 0
                
                else:
                    out_y = kp_y * (dy - delta_y)

                flag_x = x_count(abs(dx) < 0.04)
                flag_y = y_count(abs(dy) < 0.02)
                if delta_y is None:
                    flag_y = True

                if flag_x:
                    out_x = 0
                if flag_y:
                    out_y = 0
                if flag_x and flag_y:
                    # logger.info(f"location{self.get_odometry()} ok, arm_pose{self.arm.x_pose_now}")
                    self.set_velocity(0, 0, 0)
                    self.arm.x_speed(0)
                    # return det[0],det[2]
            else:
                x_count(False)
                y_count(False)
            self.set_velocity(out_x, 0, 0)
            self.arm.x_speed(out_y)
            # time.sleep(0.05)

            if time.time() > time_stop:
                self.set_velocity(0, 0, 0)
                self.arm.x_speed(0)
                logger.error("对齐目标超时")
                # logger.info(f"location{self.get_odometry()} ok, arm_pose{self.arm.x_pose_now}")

                try:
                    return det[0], det[2]
                except:
                    return (None, None)




    def move_to_detection_target_v2(
        self,
        delta_x=0.0,
        delta_y=None,   # 保留参数，但不控制y方向（无机械臂）
        label=None,
        time_out=2.0,
        sort_pos=(0, 0),
        num=0,
):
        """
        仅用车体前后移动，将侧面摄像头中目标在x方向对齐到指定偏移量。
        不控制机械臂，稳定对齐后停止并蜂鸣，返回 (cls_id, label)。

        参数说明:
            delta_x : 目标在图像x轴（左右）的期望位置，默认0（居中）
            delta_y : 弃用（仅保留接口兼容）
            label   : 指定跟踪的物体标签，None表示任意
            time_out: 超时时间（秒）
            sort_pos: 传给get_detection_results的排序参考点
            num     : 选择第几个匹配目标（0为最近的那个）
        返回:
            成功: (cls_id, label)
            超时/中断: (None, None) 或 (-1, "None") 当被stop_flag中断时
        """
        time_stop = time.time() + time_out
        x_count = CountRecord(3)      # 连续3帧满足条件视为稳定
        last_det = None
        last_dy = None

        # 根据机械臂安装侧确定小车运动方向与图像坐标关系的符号
        #（摄像头可能随安装侧不同，导致移动方向对图像x的影响相反）
        if self.arm.side == "RIGHT":
            kp_x = -0.25
            ki_x = -0.05
        else:
            kp_x = 0.25
            ki_x = 0.05

        pid_x = PID(kp_x, ki_x)
        pid_x.setpoint = delta_x
        pid_x.output_limits = (-0.06, 0.06)   # 限制最大速度

        out_x = 0.0
        track_label = label  # 追踪锁定：若已指定label则直接锁定，否则首次检测后锁定

        print(f"  [视觉对准] 开始, 目标delta_x={delta_x:.2f}, "
              f"Kp={kp_x:.2f}, Ki={ki_x:.2f}, 超时={time_out:.1f}s")

        self._last_delta_x = delta_x
        self._last_dx = None

        frame_cnt = 0
        while True:
            frame_cnt += 1
            # 外部中断（如按键3）
            if self._stop_flag:
                self.set_velocity(0, 0, 0)
                print(f"  [视觉对准] ⛔ 外部中断")
                return -1, "None", None

            # 获取并过滤目标
            dets = self.get_detection_results(sort_pos=sort_pos)
            if label is not None:
                dets = [d for d in dets if d[2] == label]

            # --- 锁定追踪目标 ---
            det = None
            if len(dets) > 0:
                if track_label is not None:
                    # 优先查找已锁定的追踪目标
                    for d in dets:
                        if d[2] == track_label:
                            det = d
                            break
                    if det is None:
                        print(f"  [视觉对准] ⚠ 追踪目标 {track_label} 丢失，重新选择")
                if det is None and len(dets) > num:
                    # 首次进入 / 锁定目标丢失 → 取目标
                    det = dets[num]
                    if label is None:
                        track_label = det[2]
                        print(f"  [视觉对准] 🎯 锁定追踪目标: {track_label}")

            if det is not None:
                last_det = (det[0], det[2])   # cls_id, label
                last_dy = det[5]              # 归一化y坐标（目标远近）
                dx = det[4]                   # 归一化x坐标
                self._last_dx = dx            # 存储最新检测的dx（调试用）
                score = det[3]                # 置信度

                error_abs = abs(dx - delta_x)
                flag_x = x_count(error_abs < 0.02)

                # 每5帧打印一次状态
                if frame_cnt % 5 == 1 or flag_x:
                    print(f"  [视觉对准] 帧#{frame_cnt}: dx={dx:+.4f}, "
                          f"|error|={error_abs:.4f}, 速度out_x={out_x:+.3f}, "
                          f"score={score:.2f}, label={det[2]}, "
                          f"{'✅ 收敛!' if flag_x else '→ 调整中...'}")

                if flag_x:
                    # 对齐成功：停车、蜂鸣、返回
                    self.set_velocity(0, 0, 0)
                    self.ring.rings()         # 蜂鸣一下
                    print(f"  [视觉对准] ✅ 对齐成功! cls_id={last_det[0]}, label={last_det[1]}, dy={last_dy:+.4f}")
                    return last_det[0], last_det[1], last_dy

                # --- 移动控制（多级调速） ---
                if error_abs < 0.5:
                    if error_abs > 0.4:
                        # 接近但未收敛：极低速微调
                        pid_x.output_limits = (-0.001, 0.001)
                        out_x = -pid_x(dx)
                        pid_x.output_limits = (-0.03, 0.03)
                        self.set_velocity(out_x, 0, 0)
                        # time.sleep(0.5)
                    else:
                        # error <= 0.05，停车等待 CountRecord 确认收敛
                        pid_x.reset()
                        self.set_velocity(0, 0, 0)
                        out_x = 0.0
                elif error_abs < 0.8:
                    # 中等距离：降低速度
                    pid_x.output_limits = (-0.02, 0.02)
                    out_x = -pid_x(dx)
                    pid_x.output_limits = (-0.04, 0.04)
                    self.set_velocity(out_x, 0, 0)
                    # time.sleep(1.0)
                else:
                    # 较远距离：正常速度
                    out_x = -pid_x(dx)
                    self.set_velocity(out_x, 0, 0)
                    # time.sleep(1.0)
            else:
                # 无合适目标，重置计数器并停车
                x_count(False)
                self.set_velocity(0, 0, 0)
                if frame_cnt % 10 == 1:
                    print(f"  [视觉对准] 帧#{frame_cnt}: 未检测到目标, 等待中...")

            # 超时处理
            if time.time() > time_stop:
                self.set_velocity(0, 0, 0)
                # 超时返回最后一次检测到的目标，若无则None
                if last_det:
                    print(f"  [视觉对准] ⏰ 超时({time_out:.1f}s), 返回最后检测: cls_id={last_det[0]}, label={last_det[1]}, dy={last_dy:+.4f}")
                else:
                    print(f"  [视觉对准] ⏰ 超时({time_out:.1f}s), 未检测到任何目标")
                return (last_det[0], last_det[1], last_dy) if last_det else (None, None, None)
    def move_to_detection_target_ball(
        self,
        delta_x=0.0,
        delta_y=0.0,
        label=None,
        time_out=10.0,
        sort_pos=(0, 0),
        num=0,
    ):
        """
        视觉伺服定位：检测目标，通过 PID 控制车体移动，将目标对齐到期望位置。

        参数:
            delta_x:  目标在画面中的水平期望位置（归一化坐标），默认 0 = 居中
            delta_y:  目标在画面中的垂直期望位置（归一化坐标），默认 0 = 居中
            label:    指定检测目标的标签，默认 None = 选距离中心最近的目标
            time_out: 超时时间（秒）
            sort_pos: 排序参考点，默认 (0,0) = 画面中心
            num:      选排序后第几个目标，默认 0 = 最近

        返回:
            (cls_id, label, dy): 识别到的目标类别ID、标签名、目标的归一化y坐标
            超时或未检测到返回 (None, None, None)
        """
        time_stop = time.time() + time_out
        x_count = CountRecord(3)          # 连续3帧稳定才认为对齐
        y_count = CountRecord(3)          # y方向同理
        last_det = None                   # 记录最后一次检测，用于超时时保存调试截图
        last_dx = None
        last_dy = None
        track_id = None                   # 锁定追踪：首次检测到目标后锁定其 det_id，不再切换目标

        if self.arm.side == "RIGHT":
            kp_x = -0.1
            ki_x = -0.03
            kd_x = -0.0
        else:
            kp_x = 0.10
            ki_x = 0.03
            kd_x = 0.0

        pid_x = PID(kp_x, ki_x, kd_x)
        pid_x.output_limits = (-0.1, 0.1)   # 全局限幅
        pid_x.setpoint = delta_x

        out_x = 0
        frame_cnt = 0
        step = 0
        last_stage = ""
        print(f"[STEP] 视觉对准开始 | delta_x={delta_x:.2f} delta_y={delta_y:.2f} label={label} arm={self.arm.side}")

        while True:
            if self._stop_flag:
                self.set_velocity(0, 0, 0)
                print(f"[STEP] 视觉对准中断 | 总帧数={frame_cnt}")
                return -1, "None"

            dets = self.get_detection_results(sort_pos=sort_pos)

            # --- 目标锁定：一旦识别到目标就用 det_id 锁定，不再切换 ---
            det = None
            if track_id is not None:
                # 已锁定：优先查找同一 det_id 的目标
                for d in dets:
                    if d[1] == track_id:
                        det = d
                        break
                if det is None:
                    # 锁定目标丢失，尝试按 label 找回
                    if label is not None:
                        for d in dets:
                            if d[2] == label:
                                det = d
                                track_id = d[1]
                                print(f"  [视觉对准] * 锁定目标丢失，按 label 找回: {label}, 新 track_id={track_id}")
                                break
                    if det is None:
                        print(f"  [视觉对准] ! 锁定目标 track_id={track_id} 丢失，等待重新出现...")
            else:
                # 首次检测：按 label 过滤后选择目标并锁定
                if label is not None:
                    dets_label = [d for d in dets if d[2] == label]
                else:
                    dets_label = dets
                if len(dets_label) > num:
                    det = dets_label[num]
                    track_id = det[1]
                    step += 1
                    print(f"[STEP {step}] 首次锁定目标: label={det[2]} track_id={track_id}")

            if det is not None:
                frame_cnt += 1
                dx, dy = det[4:6]
                last_det = (det[0], det[2])   # 记录 cls_id, label
                last_dx = dx                   # 记录最新 dx
                last_dy = dy                   # 记录最新 dy（动态调整机械臂伸出距离）

                error_x_abs = abs(dx - delta_x)
                error_y_abs = abs(dy - delta_y)

                # 提取 PID 各分量
                p_term = getattr(pid_x, '_proportional', 0.0)
                i_term = getattr(pid_x, '_integral', 0.0)
                d_term = getattr(pid_x, '_derivative', 0.0)

                # ---- 三段式调速 ----
                _dir_cn = "←左移" if dx > delta_x else "右移→"
                _dir    = "L" if dx > delta_x else "R"
                # 阶段切换时打 step 日志
                new_stage = "OK" if error_x_abs < 0.01 else ("SLOW" if error_x_abs < 0.1 else "FAST")
                if new_stage != last_stage:
                    step += 1
                    last_stage = new_stage
                    print(f"[STEP {step}] 进入{new_stage}区 | err_x={error_x_abs:.4f} dx={dx:+.3f} {_dir_cn}")
                if error_x_abs < 0.01:
                    stage    = "OK"                                # 图片用
                    stage_cn = "🟢到位"                            # 终端用
                    print(f"  [视觉对准] #{frame_cnt} {stage_cn} | err_x={error_x_abs:.4f} | P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f} | 停车")
                    pid_x.reset()
                    self.set_velocity(0, 0, 0)
                    out_x = 0.0
                    time.sleep(0.05)
                elif error_x_abs < 0.1:
                    stage    = "SLOW"
                    stage_cn = "🟡微调"
                    pid_x.output_limits = (-0.06, 0.06)
                    out_x = -pid_x(dx)
                    pid_x.output_limits = (-0.20, 0.20)
                    print(f"  [视觉对准] #{frame_cnt} {stage_cn} | err_x={error_x_abs:.4f} dx={dx:+.3f} {_dir_cn} out={out_x:+.4f} | P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f}")
                    self.set_velocity(out_x, 0, 0)
                    time.sleep(0.05)
                else:
                    stage    = "FAST"
                    stage_cn = "🔴远距"
                    out_x = -pid_x(dx)
                    print(f"  [视觉对准] #{frame_cnt} {stage_cn} | err_x={error_x_abs:.4f} dx={dx:+.3f} {_dir_cn} out={out_x:+.4f} | P={p_term:+.4f} I={i_term:+.4f} D={d_term:+.4f}")
                    self.set_velocity(out_x, 0, 0)
                    time.sleep(0.05)

                # 每次调速都保存调试图（图片用英文字母，避免 OpenCV 中文乱码）
                self.save_align_debug_img(delta_x, dx, det[2], dy,
                    save_dir="ball_align_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                    frame=frame_cnt, stage=stage, direction=_dir, out_x=out_x)

                flag_x = x_count(error_x_abs < 0.01)
                # flag_y = y_count(error_y_abs < 0.02)

                # if flag_x and flag_y:
                if flag_x :
                    pid_x.reset()              # 到位后清除积分，避免残留影响下次调用
                    out_x = 0
                    self.set_velocity(0, 0, 0)
                    time.sleep(0.3)
                    step += 1
                    print(f"[STEP {step}] 对齐成功！| cls_id={det[0]} label={det[2]} frame={frame_cnt}")
                    self.save_align_debug_img(delta_x, dx, det[2], dy,
                        save_dir="ball_align_debug", p_term=p_term, i_term=i_term, d_term=d_term,
                        final=True, frame=frame_cnt, stage=stage, direction=_dir, out_x=out_x)
                    pid_x.reset()
                    return det[0], det[2], dy
            else:
                x_count(False)
                y_count(False)
                time.sleep(0.05)

            if time.time() > time_stop:
                self.set_velocity(0, 0, 0)
                logger.error("对齐目标超时")
                if last_det and last_dx is not None:
                    time.sleep(0.3)
                    self.save_align_debug_img(delta_x, last_dx, last_det[1], last_dy, save_dir="ball_align_debug")
                print(f"[STEP] 超时! 总帧={frame_cnt} last_err={last_dx-delta_x if last_dx else 'N/A'}")
                try:
                    return det[0], det[2], last_dy
                except:
                    return (None, None, None)

    def move_to_detection_target_tilt(
        self,
        delta_x=0.0,
        delta_y=None,   # 保留参数，但不控制y方向（无机械臂）
        label=None,
        time_out=2.0,
        sort_pos=(0, 0),
        num=0,
    ):
        """
        摄像头斜向下 ~45° 时的视觉对准函数。
        目标在画面中心，通过车体前后移动将目标在 x 方向对齐到指定偏移量。
        不控制机械臂，稳定对齐后停止并蜂鸣，返回 (cls_id, label, dy)。

        适用场景：机械臂 hand="DOWN"，摄像头斜向下对着桌面/地面的目标。
        与 move_to_detection_target_v2 的区别：
            - 放宽收敛阈值（0.20 vs 0.15）：斜向下透视 -> 目标在图像上的
              x 坐标抖动比水平视角更大，需要放宽判定条件
            - 降低 PID 增益（Kp: 0.20 vs 0.25）：倾斜视角下图像像素与真实
              距离的比值变化，降低增益避免过冲
            - CountRecord 从 3 帧降为 2 帧，加快收敛判定
            - 不做 y 轴区域过滤（目标即在画面中心）

        参数说明:
            delta_x  : 目标在图像x轴（左右）的期望位置，默认0（居中）
            delta_y  : 弃用（仅保留接口兼容）
            label    : 指定跟踪的物体标签，None表示任意
            time_out : 超时时间（秒）
            sort_pos : 传给get_detection_results的排序参考点
            num      : 选择第几个匹配目标（0为最近的那个）
        返回:
            成功: (cls_id, label, dy)
            超时/中断: (None, None, None) 或 (-1, "None", None) 当被stop_flag中断时
        """
        time_stop = time.time() + time_out
        x_count = CountRecord(2)      # 放宽：连续2帧满足条件即可（原v2为3帧）
        last_det = None
        last_dy = None

        # 斜向下45°透视：图像x轴 → 真实横向距离的映射比水平视角"压缩"，
        # 同样的误差像素对应更大的实际偏差 -> 降低增益避免震荡过冲
        if self.arm.side == "RIGHT":
            kp_x = -0.20
            ki_x = -0.03
        else:
            kp_x = 0.20
            ki_x = 0.03

        pid_x = PID(kp_x, ki_x)
        pid_x.setpoint = delta_x
        pid_x.output_limits = (-0.05, 0.05)   # 比v2略低的限幅

        out_x = 0.0
        track_label = label  # 追踪锁定：若已指定label则直接锁定，否则首次检测后锁定

        print(f"  [视觉对准-斜下] 开始, 目标delta_x={delta_x:.2f}, "
              f"Kp={kp_x:.2f}, Ki={ki_x:.2f}, 超时={time_out:.1f}s")

        frame_cnt = 0
        while True:
            frame_cnt += 1
            # 外部中断（如按键3）
            if self._stop_flag:
                self.set_velocity(0, 0, 0)
                print(f"  [视觉对准-斜下] ⛔ 外部中断")
                return -1, "None", None

            # 获取检测结果（目标在画面中心，不做y轴过滤）
            dets = self.get_detection_results(sort_pos=sort_pos)

            if label is not None:
                dets = [d for d in dets if d[2] == label]

            # --- 锁定追踪目标 ---
            det = None
            if len(dets) > 0:
                if track_label is not None:
                    # 优先查找已锁定的追踪目标
                    for d in dets:
                        if d[2] == track_label:
                            det = d
                            break
                    if det is None:
                        print(f"  [视觉对准-斜下] ⚠ 追踪目标 {track_label} 丢失，重新选择")
                if det is None and len(dets) > num:
                    # 首次进入 / 锁定目标丢失 → 取目标
                    det = dets[num]
                    if label is None:
                        track_label = det[2]
                        print(f"  [视觉对准-斜下] 🎯 锁定追踪目标: {track_label}")

            if det is not None:
                last_det = (det[0], det[2])   # cls_id, label
                last_dy = det[5]              # 归一化y坐标（目标远近）
                dx = det[4]                   # 归一化x坐标
                score = det[3]                # 置信度

                # 放宽收敛阈值到 0.20（原v2为0.15）：
                # 斜向下45°场景，图像x坐标受透视影响抖动更大，需要放宽
                error_abs = abs(dx - delta_x)
                flag_x = x_count(error_abs < 0.1)

                # 每5帧打印一次状态
                if frame_cnt % 5 == 1 or flag_x:
                    print(f"  [视觉对准-斜下] 帧#{frame_cnt}: dx={dx:+.4f}, "
                          f"|error|={error_abs:.4f}, 速度out_x={out_x:+.3f}, "
                          f"score={score:.2f}, label={det[2]}, dy={det[5]:+.4f}, "
                          f"{'✅ 收敛!' if flag_x else '→ 调整中...'}")

                if flag_x:
                    # 对齐成功：停车、蜂鸣、返回
                    self.set_velocity(0, 0, 0)
                    self.ring.rings()         # 蜂鸣一下
                    print(f"  [视觉对准-斜下] ✅ 对齐成功! cls_id={last_det[0]}, label={last_det[1]}, dy={last_dy:+.4f}")
                    return last_det[0], last_det[1], last_dy

                # --- 移动控制（多级调速，阈值比v2放宽） ---
                if error_abs < 0.20:
                    if error_abs > 0.1:
                        # 接近但未收敛：极低速微调，避免停死导致永远对不准
                        pid_x.output_limits = (-0.012, 0.012)
                        out_x = -pid_x(dx)
                        pid_x.output_limits = (-0.05, 0.05)
                        self.set_velocity(out_x, 0, 0)
                        time.sleep(0.5)
                    else:
                        # 已收敛，停车等待 CountRecord 确认
                        pid_x.reset()
                        self.set_velocity(0, 0, 0)
                        out_x = 0.0
                elif error_abs < 0.55:
                    # 中等距离：降低速度到一半，避免冲过头
                    pid_x.output_limits = (-0.025, 0.025)
                    out_x = -pid_x(dx)
                    pid_x.output_limits = (-0.05, 0.05)
                    self.set_velocity(out_x, 0, 0)
                    time.sleep(1.0)
                else:
                    # 较远距离：正常速度
                    out_x = -pid_x(dx)
                    self.set_velocity(out_x, 0, 0)
                    time.sleep(1.0)
            else:
                # 无合适目标，重置计数器并停车
                x_count(False)
                self.set_velocity(0, 0, 0)
                if frame_cnt % 10 == 1:
                    print(f"  [视觉对准-斜下] 帧#{frame_cnt}: 未检测到目标, 等待中...")

            # 超时处理
            if time.time() > time_stop:
                self.set_velocity(0, 0, 0)
                # 超时返回最后一次检测到的目标，若无则None
                if last_det:
                    print(f"  [视觉对准-斜下] ⏰ 超时({time_out:.1f}s), 返回最后检测: cls_id={last_det[0]}, label={last_det[1]}, dy={last_dy:+.4f}")
                else:
                    print(f"  [视觉对准-斜下] ⏰ 超时({time_out:.1f}s), 未检测到任何目标")
                return (last_det[0], last_det[1], last_dy) if last_det else (None, None, None)

    def move_to_detection_target_lateral(
        self,
        delta_y=0.0,
        label=None,
        time_out=2.0,
        sort_pos=(0, 0),
        num=0,
    ):
        """
        底盘 y 轴（左右）视觉微调对准函数。
        通过检测目标的图像 y 坐标，PID 控制底盘左右移动，
        将目标在画面 y 方向对齐到指定偏移量。

        适用场景：x 轴（前后）已大致对齐，仅需左右微调。
                  摄像头需处于斜向下 ~45° 姿态（hand=-60）。

        参数:
            delta_y  : 目标在图像 y 轴的期望位置，默认 0（居中）
            label    : 指定跟踪的物体标签，None 表示任意
            time_out : 超时时间（秒）
            sort_pos : 传给 get_detection_results 的排序参考点
            num      : 选择第几个匹配目标（0 为最近）

        返回:
            成功: (cls_id, label)
            超时/中断: (None, None)
        """
        time_stop = time.time() + time_out
        y_count = CountRecord(2)
        last_det = None

        kp_y = 0.12
        ki_y = 0.02
        pid_y = PID(kp_y, ki_y)
        pid_y.setpoint = delta_y
        pid_y.output_limits = (-0.04, 0.04)

        out_y = 0.0
        track_label = label

        print(f"  [视觉对准-左右] 开始, 目标delta_y={delta_y:.2f}, "
              f"Kp={kp_y:.2f}, Ki={ki_y:.2f}, 超时={time_out:.1f}s")

        frame_cnt = 0
        while True:
            frame_cnt += 1
            if self._stop_flag:
                self.set_velocity(0, 0, 0)
                print(f"  [视觉对准-左右] ⛔ 外部中断")
                return None, None

            dets = self.get_detection_results(sort_pos=sort_pos)

            if label is not None:
                dets = [d for d in dets if d[2] == label]

            # --- 锁定追踪目标 ---
            det = None
            if len(dets) > 0:
                if track_label is not None:
                    for d in dets:
                        if d[2] == track_label:
                            det = d
                            break
                    if det is None:
                        print(f"  [视觉对准-左右] ⚠ 追踪目标 {track_label} 丢失，重新选择")
                if det is None and len(dets) > num:
                    det = dets[num]
                    if label is None:
                        track_label = det[2]
                        print(f"  [视觉对准-左右] 🎯 锁定追踪目标: {track_label}")

            if det is not None:
                last_det = (det[0], det[2])
                dy = det[5]                    # 归一化 y 坐标 → 对应底盘左右
                error_abs = abs(dy - delta_y)

                # 每5帧打印一次状态
                if frame_cnt % 5 == 1:
                    print(f"  [视觉对准-左右] 帧#{frame_cnt}: dy={dy:+.4f}, "
                          f"|error|={error_abs:.4f}, 速度out_y={out_y:+.3f}, "
                          f"label={det[2]}")

                # --- 多级调速 ---
                if error_abs < 0.15:
                    pid_y.reset()
                    self.set_velocity(0, 0, 0)
                    out_y = 0.0
                elif error_abs < 0.40:
                    pid_y.output_limits = (-0.02, 0.02)
                    out_y = -pid_y(dy)
                    pid_y.output_limits = (-0.04, 0.04)
                    self.set_velocity(0, out_y, 0)
                    time.sleep(0.8)
                else:
                    out_y = -pid_y(dy)
                    self.set_velocity(0, out_y, 0)
                    time.sleep(0.8)

                flag_y = y_count(error_abs < 0.15)

                if flag_y:
                    self.set_velocity(0, 0, 0)
                    self.ring.rings()
                    print(f"  [视觉对准-左右] ✅ 左右对齐成功! cls_id={last_det[0]}, label={last_det[1]}")
                    return last_det
            else:
                y_count(False)
                self.set_velocity(0, 0, 0)
                if frame_cnt % 10 == 1:
                    print(f"  [视觉对准-左右] 帧#{frame_cnt}: 未检测到目标, 等待中...")

            if time.time() > time_stop:
                self.set_velocity(0, 0, 0)
                if last_det:
                    print(f"  [视觉对准-左右] ⏰ 超时({time_out:.1f}s), 返回最后检测")
                else:
                    print(f"  [视觉对准-左右] ⏰ 超时({time_out:.1f}s), 未检测到任何目标")
                return last_det if last_det else (None, None)

    def visual_alignment_debug(self, arm_x=0.30, arm_y=0.12, arm_side="RIGHT",
                                delta_x=0.0, fps=5, cycle_sec=10.0):
        """
        视觉对准调试模式（双模式交替）。

        模式1 - 自动校准：PID 控制车体前后移动，对齐目标到画面中心。
        模式2 - 手动观察：车体不动，用户手动推车观察 dx 变化。
        每个模式持续 cycle_sec 秒，交替循环。按 Ctrl+C 退出。

        Args:
            arm_x:     机械臂水平位置
            arm_y:     机械臂竖直高度
            arm_side:  机械臂方向 LEFT / RIGHT
            delta_x:   期望的目标 x 偏移（默认 0 = 居中）
            fps:       刷新频率
            cycle_sec: 每个模式的持续时间（秒）
        """
        MODE_AUTO   = "自动校准"
        MODE_MANUAL = "手动观察"

        # --- 机械臂进入预备检测姿态 ---
        print(f"\n{'='*60}")
        print(f"  视觉对准调试模式（{cycle_sec:.0f}秒交替）")
        print(f"  模式1: {MODE_AUTO}  |  模式2: {MODE_MANUAL}")
        print(f"  机械臂: {arm_side}侧, x={arm_x:.2f}, y={arm_y:.2f}")
        print(f"  按 Ctrl+C 退出")
        print(f"{'='*60}")
        print(f"  期望目标位置 delta_x = {delta_x:.2f} (0=画面中心)")
        print(f"  对齐阈值 |dx - delta_x| < 0.2 连续 3 帧")
        print(f"{'='*60}\n")

        self.arm.move_y_position(arm_y)
        self.arm.move_x_position(arm_x)
        self.arm.set_arm_pose(arm=arm_side)
        self.arm.set_hand_angle("DOWN")

        # --- PID 控制器 ---
        if arm_side == "RIGHT":
            kp_x, ki_x = -0.25, -0.05
        else:
            kp_x, ki_x = 0.25, 0.05
        pid_x = PID(kp_x, ki_x)
        pid_x.setpoint = delta_x
        pid_x.output_limits = (-0.12, 0.12)

        # --- 状态变量 ---
        x_count = CountRecord(3)
        frame_cnt = 0
        sleep_time = 1.0 / fps
        bar_width = 30

        # --- 模式管理 ---
        current_mode = MODE_AUTO
        mode_start_time = time.time()
        aligned_locked = False  # 一旦对准就锁住，不再移动
        aligned_label = None    # 对准的目标标签
        track_target_label = None  # 自动模式下锁定的追踪目标标签，避免在多个目标间跳来跳去
        self.ring.rings()       # 响一声 = 自动校准模式开始
        print(f"\n  🔊 嘀!  [{current_mode}] 模式开始 ({cycle_sec:.0f}秒)\n")

        def make_bar(dx_val, target=0.0):
            half = bar_width // 2
            center_pos = half
            pos = int(center_pos + dx_val * half)
            pos = max(0, min(bar_width - 1, pos))
            bar = ['-'] * bar_width
            bar[center_pos] = '|'
            if pos != center_pos:
                bar[pos] = '●'
            else:
                bar[center_pos] = '★'
            percent = abs(dx_val - target) * 100
            return ''.join(bar), percent

        def switch_mode(current):
            """切换模式，蜂鸣提示"""
            nonlocal track_target_label
            self.set_velocity(0, 0, 0)  # 先停车
            if not aligned_locked:
                pid_x.reset()           # 未对准时重置 PID 积分
            track_target_label = None   # 切换模式时清空追踪目标，新模式下重新锁定
            if current == MODE_AUTO:
                new_mode = MODE_MANUAL
                self.ring.rings()
                time.sleep(0.15)
                self.ring.rings()       # 响两声 = 手动观察
                if aligned_locked:
                    print(f"\n  🔊 嘀嘀! [{new_mode}] 模式开始 ({cycle_sec:.0f}秒) - 已对准 {aligned_label}，请勿移动\n")
                else:
                    print(f"\n  🔊 嘀嘀! [{new_mode}] 模式开始 ({cycle_sec:.0f}秒) - 车体不动, 请手动推车\n")
            else:
                new_mode = MODE_AUTO
                self.ring.rings()       # 响一声 = 自动校准
                if aligned_locked:
                    print(f"\n  🔊 嘀!  [{new_mode}] 模式开始 ({cycle_sec:.0f}秒) - 已对准 {aligned_label}，保持静止 ✅\n")
                else:
                    print(f"\n  🔊 嘀!  [{new_mode}] 模式开始 ({cycle_sec:.0f}秒) - PID 自动对齐中\n")
            return new_mode, time.time()

        try:
            while True:
                frame_cnt += 1
                loop_start = time.time()

                if self._stop_flag:
                    print("\n[调试模式] 检测到停止标志，退出")
                    break

                # --- 检查模式切换 ---
                elapsed = time.time() - mode_start_time
                if elapsed >= cycle_sec:
                    current_mode, mode_start_time = switch_mode(current_mode)
                    elapsed = 0

                remain = cycle_sec - elapsed

                # --- 检测目标 ---
                dets = self.get_detection_results(sort_pos=(delta_x, 0))

                # --- 获取目标的 dx（用于控制） ---
                # 自动模式下锁定追踪同一目标，避免在多个目标间来回跳变
                nearest_dx = None
                nearest_label = None
                if len(dets) > 0:
                    if current_mode == MODE_AUTO and track_target_label is not None:
                        # 优先查找已锁定的追踪目标
                        for det in dets:
                            if det[2] == track_target_label:
                                nearest_dx = det[4]
                                nearest_label = det[2]
                                break
                        if nearest_dx is None:
                            # 锁定的目标消失了，打印提示
                            print(f"  [{frame_cnt:04d}] ⚠ 追踪目标 {track_target_label} 丢失，重新选择")
                    if nearest_dx is None:
                        # 手动模式 / 首次进入自动模式 / 锁定目标丢失 → 取最近目标
                        nearest_dx = dets[0][4]
                        nearest_label = dets[0][2]
                        if current_mode == MODE_AUTO:
                            track_target_label = nearest_label
                            print(f"  [{frame_cnt:04d}] 🎯 锁定追踪目标: {track_target_label}")

                # --- 模式控制逻辑 ---
                out_x = 0.0
                if aligned_locked:
                    # 已对准 → 永久停止，不再移动
                    self.set_velocity(0, 0, 0)
                elif current_mode == MODE_AUTO and nearest_dx is not None:
                    error = abs(nearest_dx - delta_x)
                    if error < 0.2:
                        # 已在阈值内，清零积分停车，避免 PID 惯性推出阈值
                        pid_x.reset()
                        self.set_velocity(0, 0, 0)
                    else:
                        out_x = -pid_x(nearest_dx)
                        self.set_velocity(out_x, 0, 0)
                        time.sleep(1.5)
                else:
                    self.set_velocity(0, 0, 0)

                # --- 标题行（每 10 帧打印一次） ---
                if frame_cnt % 10 == 1 or frame_cnt == 1:
                    mode_marker = "🟢" if current_mode == MODE_AUTO else "🔵"
                    print(f"\n{'─'*60}")
                    print(f"  帧#{frame_cnt}  {mode_marker} [{current_mode}]"
                          f"  剩余 {remain:.0f}s  |  "
                          f"机械臂: {self.arm.side}侧"
                          f"  x={self.arm.x_get_position():.3f}  y={self.arm.y_get_position():.3f}")
                    print(f"{'─'*60}")

                if len(dets) == 0:
                    print(f"  [{frame_cnt:04d}] ⚠ 未检测到任何目标")
                else:
                    # --- 检查对准状态 ---
                    is_aligned_now = nearest_dx is not None and abs(nearest_dx - delta_x) < 0.2
                    x_count(is_aligned_now)
                    just_locked = x_count(is_aligned_now) and not aligned_locked

                    if just_locked and nearest_label is not None:
                        aligned_locked = True
                        aligned_label = nearest_label
                        self.set_velocity(0, 0, 0)
                        self.ring.rings()
                        time.sleep(0.1)
                        self.ring.rings()
                        time.sleep(0.1)
                        self.ring.rings()  # 三声蜂鸣 = 对准锁定
                        print(f"\n  🎯🎯🎯 已对准 {aligned_label}！锁定位置，不再移动 🎯🎯🎯\n")

                    for idx, det in enumerate(dets[:3]):
                        cls_id, det_id, label, score = det[0], det[1], det[2], det[3]
                        dx = det[4]
                        dy = det[5]
                        error_abs = abs(dx - delta_x)
                        bar_str, pct = make_bar(dx, delta_x)

                        marker = "  ← 对准目标" if idx == 0 else ""
                        if aligned_locked and idx == 0:
                            aligned_tag = " 🎯已锁定!"
                        elif idx == 0 and is_aligned_now:
                            aligned_tag = " ✅"
                        else:
                            aligned_tag = ""

                        print(f"  [{frame_cnt:04d}] 目标#{idx+1}: {label:<14s}"
                              f"  score={score:.2f}{marker}{aligned_tag}")
                        if aligned_locked:
                            print(f"          dx={dx:+.4f}  dy={dy:+.4f}"
                                  f"  |error|={error_abs:.4f}"
                                  f"  偏离{pct:.1f}%  🛑车已锁止")
                        elif current_mode == MODE_AUTO:
                            print(f"          dx={dx:+.4f}  dy={dy:+.4f}"
                                  f"  |error|={error_abs:.4f}"
                                  f"  🚗车速度={out_x:+.3f}"
                                  f"  偏离{pct:.1f}%")
                        else:
                            print(f"          dx={dx:+.4f}  dy={dy:+.4f}"
                                  f"  |error|={error_abs:.4f}"
                                  f"  偏离{pct:.1f}%")
                        print(f"          [{bar_str}]")

                        if idx == 0 and aligned_locked:
                            print(f"          >>> 🔒 对准 {aligned_label} 已锁定，车体永久停止")

                # 控制帧率
                loop_elapsed = time.time() - loop_start
                if loop_elapsed < sleep_time:
                    time.sleep(sleep_time - loop_elapsed)

        except KeyboardInterrupt:
            print(f"\n\n[调试模式] 用户退出，共运行 {frame_cnt} 帧")
        finally:
            self.set_velocity(0, 0, 0)
            print("[调试模式] 结束\n")

    def adjust_arm_position(self, dis=0.05):
        # print(f"arm side:{self.arm.side}")
        x_position = self.arm.x_get_position()
        if self.arm.side == "LEFT":
            self.arm.move_x_position(x_position + dis)
        elif self.arm.side == "RIGHT":
            self.arm.move_x_position(x_position - dis)

    def debug(self, inference=False):
        """
        调试方法,显示摄像头图像和检测结果，用于调试和测试。

        inference: 是否进行推理，默认为False
        """
        inference_flag = False
        grasp_flag = False
        while True:
            if self._stop_flag:
                return

            keys_val = self.blue_pad.read()

            # ==================== 1. 蓝牙手柄连接检测 ====================
            if keys_val == [-1, -1, -1, -1, 0]:
                self.car_state = [0.0, 0.0, 0.0]
                logger.error("未检测到蓝牙手柄")
                self.display.show("can't find bluetooth pad\n")
                self.beep()
                time.sleep(1)
                continue

            if inference_flag:  # 按键1: 显示车道检测结果
                self.get_lane_results()
                self.get_detection_results()
            else:
                self.streamer.update_frame(self.cap_front.read(), "cam1")
                self.streamer.update_frame(self.cap_side.read(), "cam2")

            # 执行车辆控制
            self.set_velocity(keys_val[1], -keys_val[0], -keys_val[2])

            # 射击 按下【4】
            if keys_val[4] == (1 << 11):
                self.shooting()

            if keys_val[4] == (1 << 14):  # 按键[1]: 切换推理显示
                inference_flag = not inference_flag
                self.beep()
                time.sleep(0.5)

            # 执行机械臂控制
            if keys_val[4] == (1 << 4):  # 按键△ : 向上移动机械臂
                self.arm.motor_y.set_velocity(0.5)
            elif keys_val[4] == (1 << 6):  # 按键▽: 向下移动机械臂
                self.arm.motor_y.set_velocity(-0.5)
            else:
                self.arm.motor_y.set_velocity(0.0)

            if keys_val[4] == (1 << 7):  # 按键◁ : 向左移动机械臂
                self.arm.motor_x.set_angular(50)
            elif keys_val[4] == (1 << 5):  # 按键▷: 向右移动机械臂
                self.arm.motor_x.set_angular(-50)
            else:
                self.arm.motor_x.set_angular(0.0)

            if keys_val[4] == (1 << 0):  # 按键^ : 控制手臂向上<>^v
                self.arm.set_hand_angle("UP")
            elif keys_val[4] == (1 << 2):  # 按键V: 控制手臂向下<>^v
                self.arm.set_hand_angle("DOWN")

            if keys_val[4] == (1 << 1):
                self.arm.set_arm_angle("LEFT")
            elif keys_val[4] == (1 << 3):
                self.arm.set_arm_angle("RIGHT")
            elif keys_val[4] == (1 << 10):
                self.arm.set_arm_angle(-110)
                self.arm.set_hand_angle(30)

            if keys_val[4] == (1 << 9):
                grasp_flag = not grasp_flag
                self.arm.grasp(grasp_flag)
                time.sleep(0.3)
            if keys_val[4] == (1 << 8):
                self.servo_1_flag = (self.servo_1_flag + 1) % 2
                angle = self.servo_1_angle_list[self.servo_1_flag]
                print(angle)
                self.servo_1.set_angle(angle)
                time.sleep(0.3)
            time.sleep(0.05)

    def walk_lane_test(self):
        """
        车道行走测试

        测试车道保持功能，以固定速度行驶。
        """

        def end_function():
            return True

        self.lane_base(0.3, end_function, stop=self.STOP_PARAM)

    def close(self):
        """
        关闭方法

        关闭所有线程和资源，包括按键线程、摄像头和流处理器。
        """
        self._stop_flag = False
        self._end_flag = True
        self.thread_key.join()
        self.cap_front.close()
        self.cap_side.close()
        self.streamer.stop()
        # self.grap_cam.close()

    def manage(self, programs_list: list, order_index=0):
        """
        程序管理方法

        管理和执行程序列表，通过按键选择要执行的程序。

        参数:
            programs_list: 程序列表，包含要执行的函数
            order_index: 初始选中的程序索引，默认为0
        """

        def all_task():
            time.sleep(4)
            for func in programs_list:
                func()

        def lane_test():
            self.lane_dis_offset(0.3, 30)

        programs_suffix = [all_task, lane_test, self.debug]
        programs = programs_list.copy()
        programs.extend(programs_suffix)
        # print(programs)
        # 选中的python脚本序号
        # 当前选中的序号
        win_num = 5
        win_order = 0
        # 把programs的函数名转字符串
        logger.info(order_index)
        programs_str = [str(i.__name__) for i in programs]
        logger.info(programs_str)
        dis_str = sellect_program(programs_str, order_index, win_order)
        self.display.show(dis_str)

        self.stop()
        run_flag = False
        stop_flag = False
        stop_count = 0
        while True:
            # self.button_all.event()
            btn = self.key.get_key()
            # 短按1=1,2=2,3=3,4=4
            # 长按1=5,2=6,3=7,4=8
            # logger.info(btn)
            # button_num = car.button_all.clicked()

            if btn != 0:
                # logger.info(btn)
                # 长按1按键，退出
                if btn == 5:
                    # run_flag = True
                    self._stop_flag = True
                    self._end_flag = True
                    break
                else:
                    if btn == 4:
                        # 序号减1
                        self.beep()
                        if order_index == 0:
                            order_index = len(programs) - 1
                            win_order = win_num - 1
                        else:
                            order_index -= 1
                            if win_order > 0:
                                win_order -= 1
                        # res = sllect_program(programs, num)
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.display.show(dis_str)

                    elif btn == 2:
                        self.beep()
                        # 序号加1
                        if order_index == len(programs) - 1:
                            order_index = 0
                            win_order = 0
                        else:
                            order_index += 1
                            if len(programs) < win_num:
                                win_num = len(programs)
                            if win_order != win_num - 1:
                                win_order += 1
                        # res = sllect_program(programs, num)
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.display.show(dis_str)

                    elif btn == 3:
                        # 确定执行
                        # 调用别的程序
                        dis_str = "\n{} running......\n".format(
                            str(programs_str[order_index])
                        )
                        self.display.show(dis_str)
                        self.beep()
                        self._stop_flag = False
                        programs[order_index]()
                        self._stop_flag = True
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.stop()
                        self.beep()

                        # 自动跳转下一条
                        # if order_index == len(programs)-1:
                        #     order_index = 0
                        #     win_order = 0
                        # else:
                        #     order_index += 1
                        #     if len(programs) < win_num:
                        #         win_num = len(programs)
                        #     if win_order != win_num-1:
                        #         win_order += 1
                        # res = sllect_program(programs, num)
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.display.show(dis_str)
                    logger.info(programs_str[order_index])
            else:
                self.delay(0.02)

            time.sleep(0.02)

        for i in range(2):
            self.beep()
            time.sleep(0.4)
        time.sleep(0.1)
        self.close()


def test_for_animal():
    try:
        while True:
            res = my_car.animal_image_analysis()
            print("\n\n")
            time.sleep(10)
    except KeyboardInterrupt:
        my_car.close()


if __name__ == "__main__":
    # kill_other_python()
    my_car = MyCar()
    # arm = ArmController()
    # time.sleep(1)

    # my_car.arm.reset_position()
    #my_car.arm.set_arm_pose(0,0,"LEFT","DOWN")
    # my_car.debug(False)

    #def ocr_test():
    #    print(my_car.get_ocr())
    # my_car.arm.set_hand_angle(-50)
    #my_car.arm.grasp(True)
    # ocr_test()
    # my_car.manage([ocr_test])

    # my_car.lane_time(0.3, 5)
    #my_car.shooting()
    # my_car.lane_dis_offset(0.3, 1.2)
    # my_car.lane_sensor(0.3, 0.5)
    # my_car.debug()

    # text = "犯人没有带着眼镜，穿着短袖"
    # criminal_attr = my_car.hum_analysis.get_res_json(text)
    # print(criminal_attr)
    # my_car.task.reset()
    # pt_tar = my_car.task.punish_crimall(arm_set=True)
    # hum_attr = my_car.get_hum_attr(pt_tar)
    # print(hum_attr)
    # res_bool = my_car.compare_humattr(criminal_attr, hum_attr)
    # print(res_bool)
    # pt_tar = [0, 1, 'pedestrian',  0, 0.02, 0.4, 0.22, 0.82]
    # for i in range(4):
    #     my_car.move_for([0.07, 0, 0])
    #     my_car.lane_det_location(0.1, pt_tar, det="mot", side=-1)
    # my_car.close()
    # text = my_car.get_ocr()
    # print(text)
    # pt_tar = my_car.task.pick_up_ball(arm_set=True)
    # my_car.lane_det_location(0.1, pt_tar)

    # my_car.debug()
    # while True:
    #     text = my_car.get_ocr()
    #     print(text)
    #my_car.arm.set_arm_angle("LEFT")
    # my_car.task.reset()
    # my_car.lane_advance(0.3, dis_offset=0.01, value_h=500, sides=-1)
    # my_car.lane_task_location(0.3, 2)
    # my_car.lane_time(0.3, 5)
    # my_car.debug()

    # my_car.debug()

    # my_car.task.pick_up_block()
    # my_car.task.put_down_self_block()
    # my_car.lane_time(0.2, 2)
    # my_car.lane_advance(0.3, dis_offset=0.01, value_h=500, sides=-1)
    # my_car.lane_task_location(0.3, 2)
    # my_car.task.pick_up_block()
    # my_car.close()
    # logger.info(time.time())
    # my_car.lane_task_location(0.3, 2)

    # my_car.debug()
    # programs = [func1, func2, func3, func4, func5, func6]
    # my_car.manage(programs)
    # import sys
    # test_ord = 0
    # if len(sys.argv) >= 2:
    #     test_ord = int(sys.argv[1])
    # logger.info("test:", test_ord)
    # car_test(test_ord)
