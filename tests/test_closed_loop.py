import copy
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from ball_closed_loop import BallClosedLoopController
from ball_vision import (
    BallDetection,
    BallSceneMemory,
    BallVision,
    load_ball_config,
)
from camera_geometry import (
    CalibrationRequiredError,
    PinholeRangeModel,
    fit_range_model,
    focal_length_from_reference,
    load_geometry_config,
)
from cup_vision import CupClosedLoopController, CupVision
from image_io import read_image
from course_memory import CourseMemory, CoursePhase
from sign_line_closed_loop import (
    BypassPlan,
    Detection,
    SignDistanceTracker,
    TargetLossSafetyLock,
    VisionProcessor,
    XGORobot,
    build_parser,
    gate_initial_reacquire_error,
    limit_reacquire_command,
    load_config,
    reacquire_search_turn,
)


ROOT = Path(__file__).resolve().parents[1]


def calibrated_model(focal_px=300.0):
    return PinholeRangeModel(True, focal_px, 160.0, 5.0, 2.8)


class CameraGeometryTests(unittest.TestCase):
    def test_rules_based_sign_and_ball_distance(self):
        model = calibrated_model(300.0)
        self.assertAlmostEqual(model.sign_distance_cm(75.0), 20.0)
        self.assertAlmostEqual(model.ball_distance_cm(42.0), 20.0)

    def test_reference_calibration_uses_median(self):
        focal = focal_length_from_reference([74.0, 75.0, 76.0], 20.0, 5.0)
        self.assertAlmostEqual(focal, 300.0)

    def test_uncalibrated_model_refuses_distance(self):
        model = PinholeRangeModel(False, None, 160.0, 5.0, 2.8)
        self.assertFalse(model.calibrated)
        with self.assertRaises(CalibrationRequiredError):
            model.sign_distance_cm(75.0)

    def test_two_distance_fit_recovers_camera_offset(self):
        # Synthetic camera: f=300 px, camera setback=4 cm.
        samples = [(20.0, 300.0 * 5.0 / 24.0), (30.0, 300.0 * 5.0 / 34.0)]
        focal, offset, residuals = fit_range_model(samples, 5.0)
        self.assertAlmostEqual(focal, 300.0)
        self.assertAlmostEqual(offset, 4.0)
        self.assertTrue(all(abs(value) < 1e-8 for value in residuals))


class LineFollowingRegressionTests(unittest.TestCase):
    """Regression coverage derived from the dog14 wrong-right-turn video."""

    @classmethod
    def setUpClass(cls):
        cls.vision = VisionProcessor(load_config(ROOT / "line_sign_config.json"))

    def setUp(self):
        # Each synthetic scene starts a new temporal track. 每个测试独立建轨。
        self.vision.reset_line_tracking()

    def test_centered_start_line_wins_over_broad_floor_shadow(self):
        frame = np.full((240, 320, 3), 220, dtype=np.uint8)
        # The aligned line appears at x~=173 with this camera mounting.  The old
        # gray_max=105/area-first detector selected the much larger value-95
        # region on the right and commanded the wrong-right turn instead.
        cv2.rectangle(frame, (164, 134), (182, 239), (15, 15, 15), -1)
        cv2.rectangle(frame, (210, 134), (319, 239), (95, 95, 95), -1)

        analysis = self.vision.analyze(frame)

        self.assertIsNotNone(analysis.line_error)
        self.assertAlmostEqual(analysis.line_error, 0.0, delta=0.06)

    def test_broad_dark_object_is_line_loss_not_a_steering_target(self):
        frame = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (200, 155), (319, 239), (15, 15, 15), -1)

        analysis = self.vision.analyze(frame)

        self.assertIsNone(analysis.line_error)
        self.assertEqual(analysis.line_coverage, 0.0)

    def test_merged_dark_region_higher_up_does_not_hide_the_guide_line(self):
        # dog15 evidence: a wide dark region merged with the guide line in
        # whole-ROI contours and made the detector report line loss on ~90% of
        # frames.  The bottom band must still isolate the narrow guide line.
        frame = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (164, 134), (182, 239), (15, 15, 15), -1)  # guide line
        cv2.rectangle(frame, (0, 134), (166, 200), (60, 60, 60), -1)  # merged dark area

        analysis = self.vision.analyze(frame)

        self.assertIsNotNone(analysis.line_error)
        self.assertAlmostEqual(analysis.line_error, 0.0, delta=0.06)

    def test_wide_near_field_floor_does_not_hide_the_guide_line(self):
        # dog16 evidence: the camera is close to the floor, so the guide line is
        # very wide in the near field and merges with the floor at the bottom
        # edge; a bottom-edge scan then reported line loss.  The mid-lower scan
        # band must still isolate the narrow guide line.
        frame = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (164, 134), (182, 239), (15, 15, 15), -1)  # guide line
        cv2.rectangle(frame, (0, 222), (319, 239), (15, 15, 15), -1)  # near-field floor

        analysis = self.vision.analyze(frame)

        self.assertIsNotNone(analysis.line_error)
        self.assertAlmostEqual(analysis.line_error, 0.0, delta=0.06)

    def test_single_band_dark_fragment_is_not_a_guide_line(self):
        frame = np.full((240, 320, 3), 220, dtype=np.uint8)
        # A short piece of the circular marking crosses only the primary band;
        # two-band consensus must reject it. 单带短黑段不能触发转向。
        cv2.rectangle(frame, (164, 198), (182, 208), (15, 15, 15), -1)

        analysis = self.vision.analyze(frame)

        self.assertIsNone(analysis.line_error)

    def test_previous_track_wins_over_new_nearer_center_distractor(self):
        first = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.rectangle(first, (205, 134), (215, 239), (15, 15, 15), -1)
        self.assertIsNotNone(self.vision.analyze(first).line_error)

        second = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.rectangle(second, (210, 134), (220, 239), (15, 15, 15), -1)
        cv2.rectangle(second, (169, 134), (179, 239), (15, 15, 15), -1)

        analysis = self.vision.analyze(second)

        self.assertIsNotNone(analysis.line_error)
        self.assertGreater(analysis.line_error, 0.15)

    def test_circle_arc_does_not_replace_continuous_straight_track(self):
        first = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.rectangle(first, (190, 134), (200, 239), (15, 15, 15), -1)
        self.assertIsNotNone(self.vision.analyze(first).line_error)

        second = first.copy()
        cv2.circle(second, (160, 210), 90, (15, 15, 15), 6)
        analysis = self.vision.analyze(second)

        self.assertIsNotNone(analysis.line_error)
        expected = (195.0 - 160.0) / 160.0 - 0.09
        self.assertAlmostEqual(analysis.line_error, expected, delta=0.05)

    def test_large_one_frame_track_jump_reports_loss_for_safe_stop(self):
        first = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.rectangle(first, (60, 134), (70, 239), (15, 15, 15), -1)
        self.assertIsNotNone(self.vision.analyze(first).line_error)

        jumped = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.rectangle(jumped, (260, 134), (270, 239), (15, 15, 15), -1)

        self.assertIsNone(self.vision.analyze(jumped).line_error)

    def test_gradual_guide_motion_does_not_latch_onto_five_band_edge(self):
        # dog16 contains a persistent dark edge that can win more scan-band
        # votes whenever the near-field guide widens. Continuity must keep the
        # gradually moving guide without reporting a long safety-stop loss.
        # 连续移动的真线优先于始终存在的边缘暗段。
        errors = []
        for center in range(170, 251, 20):
            frame = np.full((240, 320, 3), 220, dtype=np.uint8)
            cv2.rectangle(
                frame, (center - 5, 134), (center + 5, 239), (15, 15, 15), -1
            )
            cv2.rectangle(frame, (285, 134), (301, 239), (15, 15, 15), -1)
            errors.append(self.vision.analyze(frame).line_error)

        self.assertTrue(all(error is not None for error in errors))
        self.assertGreater(errors[-1], 0.30)
        self.assertLess(errors[-1], 0.70)


class SignDistanceLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "line_sign_config.json")

    def tracker(self):
        return SignDistanceTracker(
            self.config["sign"], self.config["distance_control"], calibrated_model()
        )

    def loss_lock(self):
        return TargetLossSafetyLock(
            self.config["sign"], self.config["distance_control"], calibrated_model()
        )

    def test_course_config_is_one_random_yellow_or_black_sign(self):
        course = self.config["course"]
        self.assertEqual(course["information_sign_count"], 1)
        self.assertEqual(
            course["information_sign_mode"], "single_random_yellow_or_black"
        )
        self.assertTrue(course["ignore_signs_after_first_action"])

    @staticmethod
    def sign(label="yellow", radius=37.5, x=160, confidence=0.9):
        # focal=300 px and plate=5 cm: diameter=75 px means distance=20 cm.
        return Detection(label, (x, 100), radius, 3000.0, confidence)

    def test_far_target_moves_forward_and_near_target_moves_backward(self):
        tracker = self.tracker()
        self.assertGreater(tracker.forward_command(40.0), 0)
        self.assertLess(tracker.forward_command(15.0), 0)
        self.assertEqual(tracker.forward_command(20.0), 0)

    def test_both_colours_trigger_only_after_stable_20cm_frames(self):
        for color in ("yellow", "black"):
            with self.subTest(color=color):
                tracker = self.tracker()
                outputs = [
                    tracker.update(self.sign(color, radius=37.5 + delta))
                    for delta in (0.0, 0.2, -0.2, 0.1, 0.0)
                ]
                self.assertTrue(all(item is None for item in outputs[:-1]))
                self.assertEqual(outputs[-1].label, color)
                self.assertAlmostEqual(tracker.last_distance_cm, 20.0, delta=0.2)

    def test_same_size_at_40cm_does_not_trigger(self):
        tracker = self.tracker()
        # diameter=37.5 px -> 40 cm for the configured model.
        for _ in range(6):
            result = tracker.update(self.sign(radius=18.75))
        self.assertIsNone(result)
        self.assertAlmostEqual(tracker.last_distance_cm, 40.0)

    def test_reported_sparse_candidates_do_not_arm_lost_target_stop(self):
        lock = self.loss_lock()
        # The failed run saw isolated 28-40 cm candidates, then stopped after
        # 0.30 s of loss. Such unconfirmed/far detections must not own the lock.
        lock.update(self.sign("black", radius=300.0 * 5.0 / 28.0 / 2.0), 0.0)
        lock.update(None, 0.1)
        # Include the run's isolated 21.2 cm false-near estimate: even a single
        # in-tolerance sample must not arm a permanent loss stop.
        lock.update(self.sign("black", radius=300.0 * 5.0 / 21.2 / 2.0), 0.2)
        lock.update(None, 0.3)
        self.assertFalse(lock.engaged)
        self.assertFalse(lock.should_stop(1.0))

    def test_consistent_near_target_arms_and_preserves_loss_stop(self):
        lock = self.loss_lock()
        lock.update(self.sign("black", radius=37.5), 0.0)
        lock.update(self.sign("black", radius=37.7, x=161), 0.1)
        self.assertTrue(lock.engaged)
        self.assertFalse(lock.should_stop(0.39))
        self.assertTrue(lock.should_stop(0.40))

    def test_synthetic_black_and_yellow_round_signs_are_detected(self):
        vision = VisionProcessor(self.config)
        for color_name, bgr in (("yellow", (0, 220, 245)), ("black", (8, 8, 8))):
            with self.subTest(color=color_name):
                frame = np.full((240, 320, 3), 225, dtype=np.uint8)
                # The measured 20 cm plate is about 38 px across at 320x240.
                cv2.line(frame, (160, 115), (160, 185), (15, 15, 15), 6)
                cv2.circle(frame, (160, 96), 19, bgr, -1)
                detection = vision.analyze(frame).sign
                self.assertIsNotNone(detection)
                self.assertEqual(detection.label, color_name)


class BallDistanceLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_config = load_ball_config(ROOT / "ball_config.json")

    def configured_controller(self):
        config = copy.deepcopy(self.raw_config)
        config["alignment"].update(
            {
                "calibrated_for_grasp": True,
            }
        )
        config["alignment"]["profiles"]["green"].update(
            {
                "target_x_px": 170.0,
                "target_distance_cm": 20.0,
                "x_tolerance_px": 8.0,
                "distance_tolerance_cm": 1.5,
            }
        )
        return BallClosedLoopController(config, calibrated_model())

    @staticmethod
    def ball_at(distance_cm, x=170.0):
        radius = (300.0 * 2.8 / distance_cm) / 2.0
        return BallDetection("green", x, 185.0, radius, 600.0, 0.88, 0.90)

    def test_default_config_refuses_unmeasured_grasp(self):
        geometry = PinholeRangeModel(False, None, 160.0, 5.0, 2.8)
        controller = BallClosedLoopController(self.raw_config, geometry)
        self.assertFalse(controller.ready)
        with self.assertRaises(CalibrationRequiredError):
            controller.decide(self.ball_at(20.0))

    def test_ball_uses_x_then_physical_distance(self):
        controller = self.configured_controller()
        self.assertEqual(controller.decide(self.ball_at(20.0, x=145)).action, "move_left")
        self.assertEqual(controller.decide(self.ball_at(20.0, x=195)).action, "move_right")
        self.assertEqual(controller.decide(self.ball_at(30.0)).action, "move_forward")
        self.assertEqual(controller.decide(self.ball_at(15.0)).action, "move_backward")
        self.assertEqual(controller.decide(self.ball_at(20.0)).action, "aligned")

    def test_each_correction_is_bounded(self):
        controller = self.configured_controller()
        decision = controller.decide(self.ball_at(45.0))
        axis, command, duration = controller.motion_for_decision(decision)
        self.assertEqual(axis, "x")
        self.assertGreater(command, 0)
        self.assertLessEqual(duration, 0.50)

    def test_nearest_ball_is_locked_and_other_positions_are_remembered(self):
        memory = BallSceneMemory(self.raw_config, calibrated_model())
        observations = {
            "blue": self.ball_at(28.0),
            "green": self.ball_at(18.0),
            "red": self.ball_at(24.0),
        }
        # Replace synthetic labels while keeping distinct apparent distances.
        observations = {
            color: BallDetection(
                color, item.x, item.y, item.radius, item.area,
                item.circularity, item.confidence,
            )
            for color, item in observations.items()
        }
        self.assertEqual(memory.update(observations, now=10.0), "green")
        self.assertEqual(set(memory.hints), {"blue", "green", "red"})
        # A later radius change cannot switch the currently locked target.
        observations["blue"] = BallDetection(
            "blue", 170.0, 185.0, 40.0, 600.0, 0.9, 0.9
        )
        self.assertEqual(memory.update(observations, now=11.0), "green")


class RealCalibrationPhotoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sign_vision = VisionProcessor(load_config(ROOT / "line_sign_config.json"))
        cls.range_model = PinholeRangeModel.from_config(
            load_geometry_config(ROOT / "camera_geometry.json")
        )
        cls.ball_config = load_ball_config(ROOT / "ball_config.json")

    def test_all_measured_sign_photos_classify_and_separate_20_from_30cm(self):
        groups = (
            ("sign_black_20cm", "black", True),
            ("sign_black_30cm", "black", False),
            ("sign_yellow_20cm", "yellow", True),
            ("sign_yellow_30cm", "yellow", False),
        )
        for folder, color, is_target in groups:
            for path in sorted((ROOT / "calibration_images" / folder).glob("*.jpg")):
                detection = self.sign_vision.analyze(read_image(path)).sign
                with self.subTest(file=path.name):
                    self.assertIsNotNone(detection)
                    self.assertEqual(detection.label, color)
                    distance = self.range_model.sign_distance_cm(2.0 * detection.radius)
                    if is_target:
                        self.assertLessEqual(abs(distance - 20.0), 2.5)
                    else:
                        self.assertGreater(distance, 22.5)

    def test_all_four_cup_release_poses_are_inside_calibrated_window(self):
        vision = CupVision(self.ball_config)
        controller = CupClosedLoopController(self.ball_config)
        paths = sorted((ROOT / "calibration_images" / "cup_drop_pose").glob("*.jpg"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            detection = vision.detect(read_image(path))
            with self.subTest(file=path.name):
                self.assertIsNotNone(detection)
                self.assertEqual(controller.decide(detection).action, "aligned")

    def test_each_ball_color_uses_only_its_own_successful_grasp_photos(self):
        vision = BallVision(self.ball_config)
        controller = BallClosedLoopController(self.ball_config, self.range_model)
        for folder, color in (
            ("ball_blue_grasp", "blue"),
            ("ball_green_grasp", "green"),
            ("ball_red_grasp", "red"),
        ):
            paths = sorted((ROOT / "calibration_images" / folder).glob("*.jpg"))
            self.assertGreaterEqual(len(paths), 2)
            for path in paths:
                detection = vision.detect(read_image(path), color)
                with self.subTest(file=path.name, color=color):
                    self.assertIsNotNone(detection)
                    self.assertEqual(controller.decide(detection).action, "aligned")

    def test_grasp_photos_do_not_look_like_centered_drop_cup(self):
        vision = CupVision(self.ball_config)
        for folder in ("ball_blue_grasp", "ball_green_grasp", "ball_red_grasp"):
            for path in (ROOT / "calibration_images" / folder).glob("*.jpg"):
                with self.subTest(file=path.name):
                    self.assertIsNone(vision.detect(read_image(path)))


class CourseMemoryTests(unittest.TestCase):
    def test_one_sign_three_balls_then_return(self):
        memory = CourseMemory()
        memory.start(line_visible=True)
        memory.begin_sign("yellow", stable_distance=True)
        memory.finish_sign(line_reacquired=True)
        memory.enter_ball_zone(line_gone_below_frame=True, stable_ball_count=3)
        for index, color in enumerate(("green", "red", "blue")):
            memory.lock_ball(color)
            memory.begin_cup_alignment(fresh_cup_detection=True)
            memory.confirm_ball_placed(color, visual_confirmation=True)
            if index < 2:
                self.assertEqual(memory.phase, CoursePhase.BALL_ZONE)
        self.assertEqual(memory.phase, CoursePhase.RETURN_REACQUIRE)
        memory.begin_return(line_reacquired=True, return_heading_ready=True)
        memory.reach_start(
            boundary_pair_seen=True,
            both_transverse_bands_gone=True,
            full_body_inside=True,
        )
        self.assertEqual(memory.phase, CoursePhase.COMPLETE)

    def test_second_information_sign_is_rejected(self):
        memory = CourseMemory()
        memory.start(line_visible=True)
        memory.begin_sign("black", stable_distance=True)
        memory.finish_sign(line_reacquired=True)
        memory.phase = CoursePhase.OUTBOUND_LINE
        with self.assertRaises(RuntimeError):
            memory.begin_sign("yellow", stable_distance=True)


class SupervisorSafetyArgumentTests(unittest.TestCase):
    """The per-frame stop/heartbeat options must stay wired into the CLI."""

    def test_supervisor_safety_arguments_have_safe_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.stop_request, "")
        self.assertEqual(args.heartbeat, "")
        self.assertEqual(args.pid_file, "")
        self.assertEqual(args.heartbeat_timeout_s, 2.0)

    def test_supervisor_safety_arguments_are_accepted(self):
        args = build_parser().parse_args(
            [
                "--stop-request", "/tmp/run.stop.request",
                "--heartbeat", "/tmp/supervisor.heartbeat",
                "--heartbeat-timeout-s", "1.5",
                "--pid-file", "/tmp/run.pid",
            ]
        )
        self.assertEqual(args.stop_request, "/tmp/run.stop.request")
        self.assertEqual(args.heartbeat, "/tmp/supervisor.heartbeat")
        self.assertEqual(args.heartbeat_timeout_s, 1.5)
        self.assertEqual(args.pid_file, "/tmp/run.pid")


class YellowBypassTests(unittest.TestCase):
    """The yellow detour must be re-issued per frame, not sent once.

    Field report 2026-09-14: the dog strafed a little and stopped, so the sign
    was never cleared.  Root cause was a single ``move`` packet followed by a
    sleep; the firmware advances only a few steps per packet.
    """

    def action_cfg(self, direction="right"):
        return {
            "bypass_direction": direction,
            "bypass_lateral_speed": 10,
            "bypass_lateral_s": 3.0,
            "bypass_forward_speed": 9,
            "bypass_forward_s": 4.5,
            "bypass_return_s": 3.0,
        }

    def test_plan_replays_three_phases_in_order(self):
        plan = BypassPlan(self.action_cfg())
        self.assertEqual(plan.command(0.0), ("y", -10.0))
        self.assertEqual(plan.command(2.9), ("y", -10.0))
        self.assertEqual(plan.command(3.1), ("x", 9.0))
        self.assertEqual(plan.command(7.5), ("y", 10.0))
        self.assertEqual(plan.command(10.7), None)
        self.assertTrue(plan.finished)

    def test_left_direction_mirrors_lateral_sign(self):
        plan = BypassPlan(self.action_cfg("left"))
        self.assertEqual(plan.command(0.0), ("y", 10.0))

    def test_update_bypass_reissues_move_packet(self):
        robot = object.__new__(XGORobot)
        robot.action_cfg = self.action_cfg()
        robot.min_write_interval = 0.08
        robot.last_command = None
        robot.last_write = -1.0

        calls = []

        class FakeDog:
            def move(self, axis, speed):
                calls.append((axis, speed))

            def stop(self):
                calls.append(("stop",))

        robot.dog = FakeDog()
        robot.start_bypass()
        now = 0.0
        while not robot.update_bypass(now):
            now += 0.1
            self.assertLess(now, 15.0, "bypass did not finish")
        lateral_calls = [c for c in calls if c == ("y", -10.0)]
        self.assertGreaterEqual(len(lateral_calls), 20)
        self.assertEqual(calls[-1], ("stop",))


class YellowReacquireTests(unittest.TestCase):
    """A far-off guide line must align in place, not spin with a saturated turn."""

    def action_cfg(self):
        return {
            "bypass_direction": "right",
            "yellow_reacquire_speed": 6,
            "yellow_reacquire_max_turn": 20,
            "yellow_align_error": 0.35,
            "yellow_align_kp": 24,
            "yellow_search_turn": 10,
            "yellow_search_error_deadband": 0.05,
        }

    def test_misaligned_line_holds_position_and_caps_turn(self):
        forward, turn = limit_reacquire_command(-1.0, 6.0, 48.0, self.action_cfg())
        self.assertEqual(forward, 0.0)
        self.assertEqual(turn, 20.0)

    def test_aligned_line_keeps_bounded_forward_follow(self):
        forward, turn = limit_reacquire_command(0.05, 6.0, 4.0, self.action_cfg())
        self.assertEqual(forward, 6.0)
        self.assertEqual(turn, 4.0)

    def test_forward_is_capped_by_reacquire_speed(self):
        forward, _ = limit_reacquire_command(0.05, 12.0, 4.0, self.action_cfg())
        self.assertEqual(forward, 6.0)

    def test_aligned_pid_spike_is_still_capped(self):
        forward, turn = limit_reacquire_command(
            0.05, 6.0, -48.0, self.action_cfg()
        )
        self.assertEqual(forward, 6.0)
        self.assertEqual(turn, -20.0)

    def test_lost_line_search_uses_last_side_without_forward_motion(self):
        cfg = self.action_cfg()
        self.assertEqual(reacquire_search_turn(-0.8, cfg), 10.0)
        self.assertEqual(reacquire_search_turn(0.8, cfg), -10.0)

    def test_centered_pre_bypass_error_falls_back_to_bypass_side(self):
        cfg = self.action_cfg()
        self.assertEqual(reacquire_search_turn(0.0, cfg), 10.0)
        cfg["bypass_direction"] = "left"
        self.assertEqual(reacquire_search_turn(None, cfg), -10.0)

    def test_far_circle_arc_is_rejected_before_first_line_lock(self):
        cfg = dict(self.action_cfg(), yellow_acquire_error_max=0.65)
        error, acquired = gate_initial_reacquire_error(-0.98, False, cfg)
        self.assertIsNone(error)
        self.assertFalse(acquired)

    def test_plausible_line_opens_initial_reacquire_gate(self):
        cfg = dict(self.action_cfg(), yellow_acquire_error_max=0.65)
        error, acquired = gate_initial_reacquire_error(-0.40, False, cfg)
        self.assertEqual(error, -0.40)
        self.assertTrue(acquired)

    def test_temporally_tracked_line_remains_usable_after_lock(self):
        cfg = dict(self.action_cfg(), yellow_acquire_error_max=0.65)
        error, acquired = gate_initial_reacquire_error(-0.80, True, cfg)
        self.assertEqual(error, -0.80)
        self.assertTrue(acquired)


if __name__ == "__main__":
    unittest.main()
