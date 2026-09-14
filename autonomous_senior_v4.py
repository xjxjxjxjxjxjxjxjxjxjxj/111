"""Hardware motion helpers adapted from the senior project.

The old executable course sequence has been removed because it approached the
sign and cups by fixed seconds.  This module now exposes only hardware helpers
for the closed-loop runner.  Ball recognition, range alignment and post-pickup
checking are camera driven; cup navigation is a separate stage and may not be
silently executed by the legacy timing arguments.
"""

import atexit
from pathlib import Path

try:
    from xgolib import XGO
except ImportError:
    from xgo_lib import XGO
from xgo_edu import XGOEDU

from ball_closed_loop import BallClosedLoopController
from ball_vision import BallVision, load_ball_config
from camera_geometry import (
    CalibrationRequiredError,
    PinholeRangeModel,
    load_geometry_config,
)
import time

SCRIPT_DIR = Path(__file__).resolve().parent
BALL_CONFIG = load_ball_config(SCRIPT_DIR / "ball_config.json")
CAMERA_GEOMETRY = load_geometry_config(SCRIPT_DIR / "camera_geometry.json")
RANGE_MODEL = PinholeRangeModel.from_config(CAMERA_GEOMETRY)
MOTION_SETTLE_S = float(BALL_CONFIG["motion"]["settle_s"])

# 球的 HSV、识别形状过滤及旧版对准参数统一放在 ball_config.json。
# 重要：照片只用于识别，不是抓取点；calibrated_for_grasp=false 时禁止抓取动作。
angle_position=0
target_yaw=-4.85

class RobotDog:
    def __init__(self, model, vision, ball_vision, ball_controller):
        self.dog = XGO(model)
        self.vision = vision
        self.ball_vision = ball_vision
        self.ball_controller = ball_controller
        self.ballcnt = 0
        self.held_ball_color = None
        
    def rest_dog(self): 
        self.dog.reset()
        time.sleep(MOTION_SETTLE_S)

    def move_dog(self, direction, step,runtime): #正值左移，负值右移
        self.dog.pace('slow')
        if direction == 'x':
          self.dog.move(direction,step)
          if step > 0:
            self.dog.move('y',-1.5)
          else:
            self.dog.move('y',-2.5)
        else:
          self.dog.move(direction,step)
        time.sleep(runtime)
        self.dog.stop()
        # 用户要求四足动作切换之间必须等待，避免未站稳就执行下一动作。
        time.sleep(MOTION_SETTLE_S)
        
    def adjust_attitude(self, roll, pitch, yaw):
        # 机身姿态调整
        self.dog.attitude(['r', 'p', 'y'], [roll, pitch, yaw])
        # 等待姿态调整完成
        time.sleep(0.5)

    def periodic_motion(self, direction, period):
        # 机身周期运动
        if direction == 'x':
            self.dog.periodic_tran('x', period)
        elif direction == 'y':
            self.dog.periodic_tran('y', period)
        # 持续一定时间后停止
        time.sleep(period * 2)
        self.dog.periodic_tran(direction, 0)
        self.dog.stop()
        time.sleep(MOTION_SETTLE_S)

    def move_arm(self, arm_x, arm_z):
        """
        控制机械臂移动到指定位置
        :param arm_x: 水平面移动距离(mm)
        :param arm_z: 垂直面移动距离(mm)
        """
        # 检查输入值是否在机械臂工作范围内
        if -80 <= arm_x <= 155 and -50 <= arm_z <= 155:
            self.dog.arm(arm_x, arm_z)
        else:
            print("Error: Arm position out of range.")

    def control_claw(self, pos):
        """
        控制机械臂夹爪的开合
        :param pos: 夹爪开合位置(0-255)
        """
        self.dog.claw(pos)

    def _pick_up_aligned_ball(self, color_name, grab_angel=0):
        """只执行原地夹取和抬起；绝不按固定时间盲走到杯子。"""
        self.stop()
        time.sleep(MOTION_SETTLE_S)
        time.sleep(1)
        self.dog_translation('z', 90)
        time.sleep(1)
        self.dog.arm(100, 100)  #移动到某位置
        time.sleep(1)
        self.adjust_attitude(0,20,0) #调整倾角
        time.sleep(1)
        self.dog.claw(grab_angel)  #夹爪张开
        time.sleep(0.5)
        self.dog.arm(150, -10)  #移动到抓取位置
        time.sleep(1)
        self.dog.claw(255)  #抓取小球
        time.sleep(2)
        self.dog.arm(100, 100)  #移动回某位置

        self.adjust_attitude(0,0,0)

        time.sleep(1)
        self.held_ball_color = color_name
    def stop(self):
        self.dog.stop()

    def turn(self,step,runtime):
        self.dog.turn(step)
        time.sleep(runtime)
        self.dog.stop()
        time.sleep(MOTION_SETTLE_S)

    def adjust_direction(self):
        now_angle=self.dog.read_yaw()
        self.dog.turn_to(target_yaw-self.dog.init_yaw,10,1)
        self.dog.stop()
        time.sleep(MOTION_SETTLE_S)
        
    def dog_translation(self, direction, data):
        self.dog.translation(direction, data)

    def _prepare_ball_camera_pose(self):
        """恢复学长版观察姿态，再进行识别；这不是照片标定的抓取位。"""
        self.stop()
        time.sleep(MOTION_SETTLE_S)
        self.dog_translation('z', 90)
        time.sleep(MOTION_SETTLE_S)
        self.adjust_attitude(0,20,0)

    def _prepare_for_alignment_move(self):
        """先恢复稳定步态姿势，禁止在俯身状态下直接横移或前后走。"""
        self.adjust_attitude(0,0,0)
        self.dog_translation('z', 60)
        time.sleep(MOTION_SETTLE_S)

    def _apply_ball_correction(self, decision):
        motion = self.ball_controller.motion_for_decision(decision)
        if motion is None:
            return
        axis, command, duration = motion
        self._prepare_for_alignment_move()
        print("球位短步修正:", axis, round(command, 2), round(duration, 2), "s")
        self.move_dog(axis, command, duration)
        self._prepare_ball_camera_pose()

    def _align_ball_closed_loop(self, color_name):
        """逐批重拍并修正，返回连续对准的最后一批球观测。"""
        align_cfg = BALL_CONFIG["alignment"]
        aligned_batches = 0
        missed_batches = 0
        last_detection = None
        self._prepare_ball_camera_pose()

        for attempt in range(1, int(align_cfg["max_adjustments"]) + 1):
            detection = self.ball_vision.sample_from_camera(self.vision, color_name)
            if detection is None:
                missed_batches += 1
                aligned_batches = 0
                print(color_name, "球未获得稳定识别:", missed_batches)
                if missed_batches >= int(align_cfg["max_missed_batches"]):
                    print(color_name, "球识别连续失败，放弃该球")
                    self._prepare_for_alignment_move()
                    return None
                time.sleep(0.25)
                continue

            missed_batches = 0
            decision = self.ball_controller.decide(detection)
            print(
                color_name,
                "球第", attempt, "次对准:",
                "x=", round(detection.x, 1),
                "距离=", round(decision.distance_cm, 1), "cm",
                "动作=", decision.action,
            )

            if decision.action == "aligned":
                aligned_batches += 1
                last_detection = detection
                if aligned_batches >= int(align_cfg["required_aligned_batches"]):
                    return last_detection
                time.sleep(0.25)
                continue

            aligned_batches = 0
            self._apply_ball_correction(decision)
        print(color_name, "球超过最大对准次数，放弃该球")
        self._prepare_for_alignment_move()
        return None

    def grab_ball(self, color_name, grab_angel=0):
        """闭环对准、夹取并复查；成功后球保持在夹爪中。"""
        try:
            self.ball_controller.require_ready()
        except CalibrationRequiredError as exc:
            # 未标定时在任何四足或机械臂动作之前退出。
            print(color_name, "球闭环未启动:", exc)
            return False

        verify_cfg = BALL_CONFIG["grasp_verification"]
        for grasp_attempt in range(1, int(verify_cfg["max_grasp_attempts"]) + 1):
            source_detection = self._align_ball_closed_loop(color_name)
            if source_detection is None:
                return False

            # 只有连续两批距离和横向位置都合格才允许机械臂下降。
            self.stop()
            time.sleep(MOTION_SETTLE_S)
            self._pick_up_aligned_ball(color_name, grab_angel=grab_angel)

            # 回到观察姿态后检查球是否仍留在抓取前的图像区域。
            self._prepare_ball_camera_pose()
            check = self.ball_vision.confirm_source_removed_from_camera(
                self.vision, color_name, source_detection
            )
            print(
                color_name,
                "抓取复查", grasp_attempt,
                "有效帧=", check.valid_frames,
                "原位置命中=", check.source_hits,
                "结果=", "成功" if check.removed else "需要重试",
            )
            self._prepare_for_alignment_move()
            if check.removed:
                self.ballcnt += 1
                print(color_name, "球已闭环抓起；等待杯口闭环投放")
                return True

        print(color_name, "达到最大抓取重试次数，安全退出")
        return False

    def get_ball(self, color_name, cup_distance=None, grab_angel=0):
        """拒绝学长版兼容入口，防止固定时长投杯被误当成闭环。"""
        raise RuntimeError(
            "get_ball旧入口已禁用：请使用grab_ball后进入杯口视觉闭环阶段"
        )
            

    def read_yaw(self):
        return self.dog.read_yaw()

        

if __name__ == "__main__":
    raise SystemExit(
        "此文件是闭环硬件辅助库，已删除学长版固定时长赛道主程序。"
        "路牌测试请运行 sign_line_closed_loop.py；完整三球投杯流程需先完成"
        "投放后视觉复查和出发区返程标定。"
    )
