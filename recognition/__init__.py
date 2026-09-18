"""
recognition/__init__.py - Face Detection, Feature Encoding, and Matching modules.
"""

from .face_detector import FaceDetector, DetectedFace
from .face_encoder import FaceEncoder
from .face_matcher import FaceMatcher

__all__ = ["FaceDetector", "DetectedFace", "FaceEncoder", "FaceMatcher"]
