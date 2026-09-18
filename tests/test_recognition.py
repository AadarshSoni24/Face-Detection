"""
tests/test_recognition.py - Unit tests for Face Detection (YuNet), Face Encoding (SFace), and Matcher.
"""

import unittest
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from recognition.face_detector import FaceDetector, DetectedFace
from recognition.face_encoder import FaceEncoder
from recognition.face_matcher import FaceMatcher


class TestFaceRecognitionPipeline(unittest.TestCase):
    """Test suite verifying deep neural network computer vision algorithms."""

    @classmethod
    def setUpClass(cls):
        cls.detector = FaceDetector()
        cls.encoder = FaceEncoder()
        cls.matcher = FaceMatcher()

        cls.assets_dir = BASE_DIR / "data" / "sample_import"
        cls.face1_img = cv2.imread(str(cls.assets_dir / "rahul_face.jpg"))
        cls.face2_img = cv2.imread(str(cls.assets_dir / "priya_face.jpg"))
        cls.noface_img = cv2.imread(str(cls.assets_dir / "no_face.jpg"))
        cls.multiface_img = cv2.imread(str(cls.assets_dir / "multi_face.jpg"))

    def test_01_detect_single_face(self):
        """Verify YuNet detects exactly 1 face on a standard single-face portrait."""
        faces = self.detector.detect(self.face1_img)
        self.assertEqual(len(faces), 1)

        face = faces[0]
        self.assertGreater(face.w, 30)
        self.assertGreater(face.h, 30)
        self.assertGreater(face.confidence, 0.70)
        self.assertEqual(len(face.landmarks), 5)  # 5 facial landmarks

    def test_02_detect_no_face(self):
        """Verify YuNet detects 0 faces on a non-face image."""
        faces = self.detector.detect(self.noface_img)
        self.assertEqual(len(faces), 0)

    def test_03_detect_multiple_faces(self):
        """Verify YuNet detects >1 faces on a multi-person image."""
        faces = self.detector.detect(self.multiface_img)
        self.assertEqual(len(faces), 2)

    def test_04_face_encoding_dimension_and_norm(self):
        """Verify SFace extracts a 128-dimensional unit-norm (L2=1.0) feature vector."""
        faces = self.detector.detect(self.face1_img)
        embedding = self.encoder.encode_face(self.face1_img, faces[0])

        self.assertEqual(embedding.shape, (1, 128))
        self.assertEqual(embedding.dtype, np.float32)

        # L2 norm should be approximately 1.0
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_05_serialization_roundtrip(self):
        """Verify 128-d embedding preserves exact values through SQLite serialization."""
        faces = self.detector.detect(self.face1_img)
        orig_emb = self.encoder.encode_face(self.face1_img, faces[0])

        blob = self.encoder.serialize_embedding(orig_emb)
        self.assertEqual(len(blob), 128 * 4)  # 128 floats * 4 bytes = 512 bytes

        recovered = self.encoder.deserialize_embedding(blob)
        self.assertTrue(np.allclose(orig_emb, recovered))

    def test_06_matching_same_person(self):
        """Verify comparing identical face yields match with similarity ~ 1.0."""
        faces = self.detector.detect(self.face1_img)
        emb1 = self.encoder.encode_face(self.face1_img, faces[0])

        result = self.matcher.compare(emb1, emb1)
        self.assertTrue(result.is_match)
        self.assertGreaterEqual(result.similarity_score, 0.99)
        self.assertEqual(result.threshold, config.FACE_MATCH_THRESHOLD)

    def test_07_matching_different_persons(self):
        """Verify comparing Person A vs Person B yields score below threshold (rejected)."""
        faces1 = self.detector.detect(self.face1_img)
        emb1 = self.encoder.encode_face(self.face1_img, faces1[0])

        faces2 = self.detector.detect(self.face2_img)
        emb2 = self.encoder.encode_face(self.face2_img, faces2[0])

        result = self.matcher.compare(emb1, emb2)
        self.assertFalse(result.is_match)
        self.assertLess(result.similarity_score, config.FACE_MATCH_THRESHOLD)
        print(f"\n[Experimental Result] Different face similarity: {result.similarity_score:.4f} < {result.threshold:.4f}")

    def test_08_multi_reference_matching(self):
        """Verify multi-reference comparison picks the highest similarity score."""
        faces1 = self.detector.detect(self.face1_img)
        emb1 = self.encoder.encode_face(self.face1_img, faces1[0])

        faces2 = self.detector.detect(self.face2_img)
        emb2 = self.encoder.encode_face(self.face2_img, faces2[0])

        # Query against [different_face, same_face]
        result = self.matcher.compare_multi(emb1, [emb2, emb1])
        self.assertTrue(result.is_match)
        self.assertGreaterEqual(result.similarity_score, 0.99)


if __name__ == "__main__":
    unittest.main()
