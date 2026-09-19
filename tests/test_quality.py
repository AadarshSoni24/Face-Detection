"""
tests/test_quality.py - Unit tests for Face Quality Analyzer.
"""

import unittest
import numpy as np
import cv2
from recognition.quality_analyzer import QualityAnalyzer, QualityAssessment


class TestQualityAnalyzer(unittest.TestCase):
    """Test suite verifying illumination, focus sharpness, and face coverage checks."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_01_good_quality_frame(self):
        """Test that a well-lit, sharp frame with good face size passes quality checks."""
        # Create 640x480 gray image with good contrast and texture
        frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        # Add high-frequency edge gradients to ensure sharpness
        frame[100:300:10, 150:350:10] = 255

        face_bbox = (150, 100, 200, 200) # Area 40000 / 307200 = ~13%
        res: QualityAssessment = self.analyzer.evaluate_frame(frame, face_bbox=face_bbox)

        self.assertTrue(res.is_acceptable)
        self.assertEqual(res.status_code, "GOOD")
        self.assertGreater(res.brightness, 45.0)
        self.assertGreater(res.sharpness, 40.0)

    def test_02_too_dark_frame(self):
        """Test rejection when illumination is too dark."""
        frame = np.full((480, 640, 3), 20, dtype=np.uint8) # mean 20 < 45
        res = self.analyzer.evaluate_frame(frame)

        self.assertFalse(res.is_acceptable)
        self.assertEqual(res.status_code, "TOO_DARK")
        self.assertIn("Low Lighting", res.message)

    def test_03_too_bright_frame(self):
        """Test rejection when illumination is overexposed."""
        frame = np.full((480, 640, 3), 245, dtype=np.uint8) # mean 245 > 220
        res = self.analyzer.evaluate_frame(frame)

        self.assertFalse(res.is_acceptable)
        self.assertEqual(res.status_code, "TOO_BRIGHT")
        self.assertIn("Overexposed", res.message)

    def test_04_blurry_frame(self):
        """Test rejection when face is blurry (low Laplacian variance)."""
        # Smooth flat gradient with good brightness
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Apply heavy blur
        blurred = cv2.GaussianBlur(frame, (25, 25), 0)
        face_bbox = (150, 100, 200, 200)

        res = self.analyzer.evaluate_frame(blurred, face_bbox=face_bbox)
        self.assertFalse(res.is_acceptable)
        self.assertEqual(res.status_code, "BLURRY")

    def test_05_too_far_face(self):
        """Test warning when face bounding box is too small in the frame."""
        frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        # Face box 20x20 = 400 pixels / 307200 = 0.13% < 3%
        face_bbox = (200, 200, 20, 20)

        res = self.analyzer.evaluate_frame(frame, face_bbox=face_bbox)
        self.assertFalse(res.is_acceptable)
        self.assertEqual(res.status_code, "TOO_FAR")


if __name__ == "__main__":
    unittest.main()
