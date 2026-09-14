import json
import unittest
from pathlib import Path

import cv2

from ball_vision import BallVision, load_ball_config
from sign_vision import SignVision, load_sign_config


ROOT = Path(__file__).resolve().parents[1]


class RecognitionLibraryTests(unittest.TestCase):
    """Unknown-distance photos remain recognition-only regression samples."""

    @classmethod
    def setUpClass(cls):
        cls.ball_vision = BallVision(load_ball_config(ROOT / "ball_config.json"))
        cls.sign_vision = SignVision(load_sign_config(ROOT / "sign_config.json"))
        cls.library = json.loads(
            (ROOT / "recognition_library.json").read_text(encoding="utf-8")
        )

    def test_all_15_images_match_ball_labels(self):
        self.assertEqual(len(self.library["samples"]), 15)
        for sample in self.library["samples"]:
            frame = cv2.imread(str(ROOT / "test_images" / sample["file"]))
            self.assertIsNotNone(frame)
            actual = sorted(
                name
                for name, result in self.ball_vision.detect_all(frame).items()
                if result is not None
            )
            with self.subTest(filename=sample["file"]):
                self.assertEqual(actual, sorted(sample["expected_valid_balls"]))

    def test_yellow_evidence_labels(self):
        for sample in self.library["samples"]:
            frame = cv2.imread(str(ROOT / "test_images" / sample["file"]))
            actual = self.sign_vision.detect_yellow(frame) is not None
            with self.subTest(filename=sample["file"]):
                self.assertEqual(actual, sample["expected_yellow_evidence"])

    def test_library_forbids_physical_calibration(self):
        forbidden = set(self.library["forbidden_uses"])
        self.assertIn("ball grasp point calibration", forbidden)
        self.assertIn("sign 20 cm warning-point calibration", forbidden)


if __name__ == "__main__":
    unittest.main()
