"""
tests/test_pass_generator.py - Unit tests for Verified Digital Exam Pass Generator.
"""

import unittest
import os
import shutil
import numpy as np
from services.pass_generator import PassGenerator, ExamPass
import config


class TestPassGenerator(unittest.TestCase):
    """Test suite for pass token generation, PIL image creation, and HTML receipt rendering."""

    def setUp(self):
        self.output_dir = os.path.join(config.DATA_DIR, "test_passes_dir")
        self.generator = PassGenerator(output_dir=self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_01_pass_token_deterministic(self):
        """Test pass code format and deterministic generation."""
        token = self.generator.generate_pass_code("CS101", "Algorithms", "2026-09-19 10:00:00")
        self.assertTrue(token.startswith("EXAM-CS101-"))
        self.assertGreater(len(token), 15)

    def test_02_create_exam_pass_image(self):
        """Test generating graphical pass image with face thumbnail."""
        face_crop = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)

        exam_pass: ExamPass = self.generator.create_exam_pass(
            student_name="Rahul Kumar",
            roll_number="101",
            course="Computer Science",
            exam_title="Data Structures",
            hall_number="Hall 302",
            similarity_score=0.88,
            liveness_status="LIVE VERIFIED",
            student_face_crop=face_crop
        )

        self.assertIsNotNone(exam_pass.pil_image)
        self.assertEqual(exam_pass.student_name, "Rahul Kumar")
        self.assertEqual(exam_pass.roll_number, "101")
        self.assertTrue(os.path.exists(exam_pass.image_path))
        self.assertEqual(exam_pass.pil_image.size, (700, 440))

    def test_03_generate_html_receipt(self):
        """Test generating printable HTML receipt file."""
        exam_pass: ExamPass = self.generator.create_exam_pass(
            student_name="Priya Sharma",
            roll_number="102",
            course="Physics",
            exam_title="Quantum Mechanics",
            hall_number="Hall A",
            similarity_score=0.92,
            liveness_status="LIVE VERIFIED"
        )

        html_path = self.generator.generate_html_receipt(exam_pass)
        self.assertTrue(os.path.exists(html_path))

        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Priya Sharma", content)
            self.assertIn("102", content)
            self.assertIn("Quantum Mechanics", content)
            self.assertIn("BIOMETRIC EXAMINATION ENTRY PASS", content)


if __name__ == "__main__":
    unittest.main()
