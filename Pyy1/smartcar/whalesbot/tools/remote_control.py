#!/usr/bin/python3
# -*- coding: utf-8 -*-
# remote_control.py - 遥控车数据收集系统（蓝牙手柄控制版）
"""
功能说明：
- 单摄像头数据采集（640x480）
- 蓝牙手柄控制遥控车运动
- 按键 3 按下时记录数据，松开停止
- 按键 1+2 同时按下退出程序
- 按键 1+V 删除最近 30 张图片
- 按键 1+O 清空所有数据
"""

import cv2
import threading
import time
import json
import subprocess
import os, sys

# ==================== sys.path 配置 ====================
_file_dir     = os.path.dirname(os.path.abspath(__file__))          # .../whalesbot/tools/
_smartcar_dir = os.path.dirname(os.path.dirname(_file_dir))         # .../smartcar/
_pyy1_dir     = os.path.dirname(_smartcar_dir)                      # .../Pyy1/
for _d in (_smartcar_dir, _pyy1_dir):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from whalesbot.vehicle import MecanumDriver, BluetoothPad, ScreenShow, Beep
from whalesbot.tools.camera import Camera
from whalesbot.tools.log_wrap import logger


class RemoteControlCar:
    def __init__(self, cap: Camera = None) -> None:
        # ---- 数据保存目录 ----
        path_dir = os.getcwd()
        self.dir = os.path.join(path_dir, "image_set2")
        os.makedirs(self.dir, exist_ok=True)

        self.index = 0

        # ---- 摄像头 ----
        if cap is None:
            self.cap = Camera(0, 640, 480)
        else:
            self.cap = cap

        # ---- 车辆 & 手柄 & 外设 ----
        self.car = MecanumDriver()
        self.rings = Beep()
        self.display = ScreenShow()
        self.blue_pad = BluetoothPad()

        # ---- 车辆控制参数 ----
        self.state_base = [0.15, 0.15, 0.3]
        self.state_start = [0.3, 0.3, 0.5]
        self.car_state = [0.0, 0.0, 0.0]

        # ---- 控制标志 ----
        self.run_flag = False
        self.exit_flag = False

        # ---- JSON 数据 ----
        self.json_data = []
        self.json_path = os.path.join(self.dir, "data.json")

        # ---- 启动 ----
        self.beep()
        logger.info("remote control start!!")
        self.display.show("press btn control\n 3 start\n 4+2 stop\n 4+v del 30pic\n 4+o del all\n")

        # 图片收集线程
        self.img_thread = threading.Thread(target=self.image_process, daemon=True)
        self.img_thread.start()

        # 进入手柄主循环
        self.car_process()

    def beep(self):
        self.rings.rings()

    def car_process(self):
        """蓝牙手柄主循环"""
        pad_exit_flag = False
        while not self.exit_flag:
            keys_val = self.blue_pad.read()

            # ---- 手柄连接检测 ----
            if keys_val == [-1, -1, -1, -1, 0]:
                pad_exit_flag = False
                self.car_state = [0.0, 0.0, 0.0]
                logger.error("no bluepad")
                self.display.show("no bluepad\n")
                self.beep()
                time.sleep(1)
                continue
            else:
                if not pad_exit_flag:
                    self.display.show("press btn control\n 3 pressing record\n 4+2 stop\n 4+v del 30pic\n 4+o del all\n")
                pad_exit_flag = True

            # ---- 按键处理 ----
            # 按键 3: 开始/停止记录
            if (keys_val[4] & 1024) != 0:
                self.run_flag = True
            else:
                self.run_flag = False

            # 按键 1+2 同时按: 退出
            if keys_val[4] == 34816:
                self.close()
                break

            # 按键 1+V: 删除最近 30 张
            elif keys_val[4] == 2052:
                self.del_last3s()

            # 按键 1+O: 清空所有
            elif keys_val[4] == 2304:
                self.restart()

            # ---- 车辆运动控制 ----
            if self.run_flag:
                self.car_state[0] = self.state_base[0]
                self.car_state[1] = -1 * self.state_base[1] * keys_val[0]
                self.car_state[2] = -3.14 * self.state_base[2] * keys_val[2]
            else:
                self.car_state[0] = self.state_start[0] * keys_val[1]
                self.car_state[1] = -1 * self.state_start[1] * keys_val[0]
                self.car_state[2] = -3.14 * self.state_start[2] * keys_val[2]

            self.car.set_velocity(*self.car_state)
            time.sleep(0.05)

    def image_process(self):
        """图片收集线程"""
        name_length = 4

        while not self.exit_flag:
            if self.run_flag:
                data_dict = dict()
                image = self.cap.read()

                img_name = str(self.index).zfill(name_length) + ".jpg"
                data_dict["img_path"] = img_name
                image_path = os.path.join(self.dir, img_name)
                cv2.imwrite(image_path, image)
                data_dict["state"] = self.car_state.copy()
                self.json_data.append(data_dict)

                logger.info("image:{}".format(img_name))
                self.index += 1

                if self.index % 10 == 0:
                    self.save_json()
                if self.index % 20 == 0:
                    self.display.show("image:{}\n".format(self.index))

            time.sleep(0.05)

    def del_last3s(self):
        """删除最近约 3 秒的图片（~30 张）"""
        self.beep()
        for i in range(30):
            try:
                data = self.json_data.pop()
                path = os.path.join(self.dir, data["img_path"])
                os.remove(path)
                self.index -= 1
            except IndexError:
                logger.info("image data zero now")
                return
        self.save_json()
        self.display.show("image:{}\n".format(self.index))

    def restart(self):
        """清空所有图片和数据"""
        self.beep()
        time.sleep(0.4)
        self.beep()
        subprocess.run(["find", self.dir, "-name", "*.jpg", "-delete"])
        self.json_data = []
        self.index = 0
        self.display.show("image:{}\n".format(self.index))

    def save_json(self):
        os.makedirs(self.dir, exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump(self.json_data, f)

    def close(self):
        self.save_json()
        self.display.show("control end!\nimage:{}\n".format(self.index))
        self.exit_flag = True
        self.img_thread.join(timeout=2.0)
        self.cap.close()
        for i in range(3):
            self.beep()
            time.sleep(0.4)
        logger.info("系统已关闭, 图片总数: {}".format(self.index))


if __name__ == "__main__":
    RemoteControlCar()
