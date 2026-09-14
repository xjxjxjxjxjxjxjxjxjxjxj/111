#!/usr/bin/env python3
"""Topological memory for the rule-map route.

The rule drawing is a top-down schematic, while the XGO camera sees a low,
forward perspective.  Template-matching those two views would produce false
localization confidence.  This state machine instead remembers which landmark
has been completed and requires fresh visual evidence at every transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Set


class CoursePhase(Enum):
    START_ZONE = auto()
    OUTBOUND_LINE = auto()
    SIGN_APPROACH = auto()
    SIGN_ACTION = auto()
    LINE_AFTER_SIGN = auto()
    BALL_ZONE = auto()
    CUP_ALIGNMENT = auto()
    RETURN_REACQUIRE = auto()
    RETURN_LINE = auto()
    HOME_CONFIRM = auto()
    COMPLETE = auto()


@dataclass
class CourseMemory:
    """Small auditable memory of autonomous-stage progress."""

    phase: CoursePhase = CoursePhase.START_ZONE
    information_sign_color: Optional[str] = None
    information_sign_done: bool = False
    locked_ball_color: Optional[str] = None
    placed_balls: Set[str] = field(default_factory=set)
    event_log: List[str] = field(default_factory=list)

    def start(self, line_visible: bool) -> None:
        if self.phase is not CoursePhase.START_ZONE or not line_visible:
            raise RuntimeError("出发前必须在出发区看到引导线")
        self.phase = CoursePhase.OUTBOUND_LINE
        self.event_log.append("left_start_zone_on_visible_line")

    def begin_sign(self, color: str, stable_distance: bool) -> None:
        if self.information_sign_done:
            raise RuntimeError("规则只有一个随机信息立牌，禁止触发第二次")
        if self.phase is not CoursePhase.OUTBOUND_LINE or not stable_distance:
            raise RuntimeError("只有稳定的20cm视觉测距才能进入立牌动作")
        if color not in ("yellow", "black"):
            raise ValueError("信息立牌只能是yellow或black")
        self.information_sign_color = color
        self.phase = CoursePhase.SIGN_ACTION
        self.event_log.append("%s_sign_alarm_at_20cm" % color)

    def finish_sign(self, line_reacquired: bool) -> None:
        if self.phase is not CoursePhase.SIGN_ACTION or not line_reacquired:
            raise RuntimeError("立牌动作后必须重新获得黑线视觉证据")
        self.information_sign_done = True
        self.phase = CoursePhase.LINE_AFTER_SIGN
        self.event_log.append("single_information_sign_complete")

    def enter_ball_zone(
        self, line_gone_below_frame: bool, stable_ball_count: int
    ) -> None:
        if self.phase is not CoursePhase.LINE_AFTER_SIGN:
            raise RuntimeError("尚未完成唯一信息立牌任务")
        if not line_gone_below_frame or stable_ball_count < 1:
            raise RuntimeError("进入球区需要黑线下方消失且至少一个球稳定可见")
        self.phase = CoursePhase.BALL_ZONE
        self.event_log.append("entered_ball_zone_from_visual_transition")

    def lock_ball(self, color: str) -> None:
        if self.phase is not CoursePhase.BALL_ZONE:
            raise RuntimeError("只能在球区锁定球")
        if color in self.placed_balls:
            raise RuntimeError("该颜色球已经完成")
        self.locked_ball_color = color
        self.event_log.append("locked_%s_ball" % color)

    def begin_cup_alignment(self, fresh_cup_detection: bool) -> None:
        if self.locked_ball_color is None or not fresh_cup_detection:
            raise RuntimeError("持球后必须重新识别杯口才能移动到释放姿态")
        self.phase = CoursePhase.CUP_ALIGNMENT

    def confirm_ball_placed(self, color: str, visual_confirmation: bool) -> None:
        if self.phase is not CoursePhase.CUP_ALIGNMENT:
            raise RuntimeError("当前不在投杯阶段")
        if color != self.locked_ball_color or not visual_confirmation:
            raise RuntimeError("投杯完成必须与锁定颜色一致并有视觉复查")
        self.placed_balls.add(color)
        self.locked_ball_color = None
        self.event_log.append("placed_%s_ball" % color)
        if self.placed_balls == {"red", "green", "blue"}:
            self.phase = CoursePhase.RETURN_REACQUIRE
        else:
            self.phase = CoursePhase.BALL_ZONE

    def begin_return(self, line_reacquired: bool, return_heading_ready: bool) -> None:
        if self.phase is not CoursePhase.RETURN_REACQUIRE:
            raise RuntimeError("三球未全部完成，不能进入返程")
        if not line_reacquired or not return_heading_ready:
            raise RuntimeError("返程必须先重新找到黑线并转向原路方向")
        self.phase = CoursePhase.RETURN_LINE
        self.event_log.append("return_line_reacquired")

    def reach_start(
        self,
        boundary_pair_seen: bool,
        both_transverse_bands_gone: bool,
        full_body_inside: bool,
    ) -> None:
        if self.phase is not CoursePhase.RETURN_LINE:
            raise RuntimeError("当前不在返程黑线上")
        if not boundary_pair_seen:
            raise RuntimeError("必须先看到出发区长方形的两条横向黑边")
        if not both_transverse_bands_gone:
            raise RuntimeError("左右任一处仍有横向黑边时禁止判定返程完成")
        if not full_body_inside:
            raise RuntimeError("必须确认整机垂直投影完全回到出发区")
        self.phase = CoursePhase.COMPLETE
        self.event_log.append("autonomous_stage_complete_inside_start_zone")
