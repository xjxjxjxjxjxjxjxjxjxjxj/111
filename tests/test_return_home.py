import unittest
from pathlib import Path

import cv2
import numpy as np

from image_io import read_image
from return_home_vision import (
    HomeBoundaryObservation,
    ReturnGuideTracker,
    ReturnGuideVision,
    StartZoneTracker,
    StartZoneVision,
    load_return_config,
)


ROOT = Path(__file__).resolve().parents[1]


class RealReturnPhotoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_return_config(ROOT / "return_home_config.json")
        cls.guide = ReturnGuideVision(cls.config)
        cls.home = StartZoneVision(cls.config)

    def test_return_guide_ignores_right_boundary_at_three_lateral_offsets(self):
        expected = {
            "回程途中_1_完美.jpg": "center",
            "回程途中_2_偏左侧.jpg": "right",
            "回程途中_3_偏右侧.jpg": "left",
        }
        for path in sorted((ROOT / "calibration_images" / "return_path").glob("*.jpg")):
            detection = self.guide.detect(read_image(path))
            with self.subTest(file=path.name):
                self.assertIsNotNone(detection)
                self.assertTrue(detection.arc_reference_visible)
                self.assertFalse(detection.touches_right_edge)
                position = expected[path.name]
                if position == "center":
                    self.assertLess(abs(detection.normalized_error), 0.20)
                elif position == "right":
                    self.assertGreater(detection.normalized_error, 0.20)
                else:
                    self.assertLess(detection.normalized_error, -0.20)

    def test_start_zone_photos_require_both_transverse_bands_to_disappear(self):
        paths = sorted((ROOT / "calibration_images" / "start_zone_entry").glob("*.jpg"))
        self.assertEqual(len(paths), 3)
        observations = [self.home.detect(read_image(path)) for path in paths]
        self.assertGreaterEqual(observations[0].band_count, 2)
        self.assertGreaterEqual(observations[1].band_count, 2)
        self.assertEqual(observations[2].band_count, 0)


class ReturnSafetyStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_return_config(ROOT / "return_home_config.json")

    def test_tracker_cannot_finish_without_first_seeing_two_boundaries(self):
        tracker = StartZoneTracker(self.config)
        clear = HomeBoundaryObservation((), 320, 240)
        for _ in range(20):
            self.assertFalse(tracker.update_observation(clear))
        self.assertFalse(tracker.boundary_pair_armed)

    def test_one_remaining_side_fragment_resets_completion(self):
        tracker = StartZoneTracker(self.config)
        pair = HomeBoundaryObservation((180.0, 225.0), 320, 240)
        one_side_fragment = HomeBoundaryObservation((226.0,), 320, 240)
        clear = HomeBoundaryObservation((), 320, 240)

        for _ in range(tracker.seen_confirm_frames):
            self.assertFalse(tracker.update_observation(pair))
        self.assertTrue(tracker.boundary_pair_armed)
        for _ in range(tracker.clear_confirm_frames - 1):
            self.assertFalse(tracker.update_observation(clear))
        self.assertFalse(tracker.update_observation(one_side_fragment))
        for index in range(tracker.clear_confirm_frames):
            completed = tracker.update_observation(clear)
            self.assertEqual(completed, index == tracker.clear_confirm_frames - 1)

    def test_slanted_partial_line_at_left_edge_is_not_treated_as_clear(self):
        frame = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.line(frame, (0, 220), (115, 185), (10, 10, 10), 10)
        observation = StartZoneVision(self.config).detect(frame)
        self.assertGreaterEqual(observation.band_count, 1)

    def test_locked_guide_does_not_jump_to_right_edge_boundary(self):
        tracker = ReturnGuideTracker(ReturnGuideVision(self.config))
        guide_frame = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.rectangle(guide_frame, (151, 118), (169, 239), (8, 8, 8), -1)
        # Separate shallow component represents the known circular-arc cue.
        cv2.line(guide_frame, (25, 185), (105, 173), (20, 20, 20), 4)
        for _ in range(tracker.confirm_frames):
            detection = tracker.update(guide_frame)
        self.assertIsNotNone(detection)

        boundary_only = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.rectangle(boundary_only, (286, 100), (319, 239), (8, 8, 8), -1)
        self.assertIsNone(tracker.update(boundary_only))


if __name__ == "__main__":
    unittest.main()
