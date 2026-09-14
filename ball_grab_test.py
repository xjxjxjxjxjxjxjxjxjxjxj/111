#!/usr/bin/env python3
"""Single-ball closed-loop pickup harness; it never travels to a cup."""

import argparse
import atexit

from xgo_edu import XGOEDU

from autonomous_senior_v4 import BALL_CONFIG, RANGE_MODEL, RobotDog
from ball_closed_loop import BallClosedLoopController
from ball_vision import BallVision


GRAB_ANGLE = {"green": 30, "red": 0, "blue": 0}


def main():
    parser = argparse.ArgumentParser(description="Align and pick up one ball")
    parser.add_argument("color", choices=tuple(GRAB_ANGLE))
    args = parser.parse_args()

    vision = XGOEDU()
    ball_vision = BallVision(BALL_CONFIG)
    robot = RobotDog(
        "xgolite",
        vision,
        ball_vision,
        BallClosedLoopController(BALL_CONFIG, RANGE_MODEL),
    )
    atexit.register(robot.stop)
    success = robot.grab_ball(args.color, GRAB_ANGLE[args.color])
    print("单球抓起测试:", "已抓起并保持" if success else "失败")


if __name__ == "__main__":
    main()
