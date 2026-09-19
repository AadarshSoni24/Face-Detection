"""
tests/test_liveness.py - Unit tests for Anti-Spoofing and Liveness Detection Engine.
"""

import unittest
import numpy as np
from recognition.face_detector import DetectedFace
from recognition.liveness_detector import LivenessDetector, LivenessAssessment


class TestLivenessDetector(unittest.TestCase):
    """Test suite verifying temporal micro-movement tracking and static photo rejection."""

    def setUp(self):
        self.detector = LivenessDetector(min_frames=5, buffer_size=10, threshold=0.60)
        self.dummy_frame = np.random.randint(60, 200, (480, 640, 3), dtype=np.uint8)

    def _create_mock_face(self, eye_x_offset=0.0, eye_y_offset=0.0):
        """Create a mock DetectedFace with 5 facial landmarks."""
        landmarks = [
            (200 + int(eye_x_offset), 180 + int(eye_y_offset)), # right eye
            (280 + int(eye_x_offset), 180 + int(eye_y_offset)), # left eye
            (240, 220),                                         # nose
            (210, 260),                                         # right mouth
            (270, 260)                                          # left mouth
        ]
        return DetectedFace(
            x=160,
            y=120,
            w=160,
            h=180,
            confidence=0.95,
            landmarks=landmarks,
            raw_face=np.zeros(15, dtype=np.float32)
        )

    def test_01_insufficient_frames_buffering(self):
        """Test provisional output when buffer is still gathering frames."""
        face = self._create_mock_face()
        res: LivenessAssessment = self.detector.evaluate(self.dummy_frame, face)
        self.assertFalse(res.is_challenge_complete)
        self.assertIn("Buffering", res.message)

    def test_02_static_image_spoof_detection(self):
        """Test that identical landmarks over 10 consecutive frames is flagged as spoof."""
        self.detector.reset()
        face = self._create_mock_face(eye_x_offset=0.0, eye_y_offset=0.0)

        # Push identical frames into buffer
        for _ in range(12):
            res = self.detector.evaluate(self.dummy_frame, face)

        self.assertTrue(res.is_challenge_complete)
        # Static zero-motion landmarks should yield low motion score
        self.assertLess(res.motion_score, 0.40)
        self.assertFalse(res.is_live)
        self.assertIn("Spoof", res.message)

    def test_03_live_human_dynamic_micro_motion(self):
        """Test that natural human physiological micro-motion passes liveness test."""
        self.detector.reset()

        # Simulate natural physiological breathing/head tremor shifts (1 to 3 pixels variation)
        offsets = [(0, 0), (1, 0), (2, 1), (1, 1), (0, 1), (-1, 0), (0, -1), (1, 0), (2, 1), (1, 0), (0, 0)]
        for ox, oy in offsets:
            face = self._create_mock_face(eye_x_offset=ox, eye_y_offset=oy)
            res = self.detector.evaluate(self.dummy_frame, face)

        self.assertTrue(res.is_challenge_complete)
        self.assertTrue(res.is_live)
        self.assertGreaterEqual(res.liveness_score, 0.60)
        self.assertIn("Live", res.message)


if __name__ == "__main__":
    unittest.main()
