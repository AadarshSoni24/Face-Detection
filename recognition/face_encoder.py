"""
recognition/face_encoder.py - Deep Learning Face Feature Extraction using SFace ONNX.

Face Recognition / Feature Encoding transforms an aligned facial image into a
compact 128-dimensional mathematical representation (embedding vector).
Faces of the same person yield embeddings close to each other in vector space.
"""

from typing import Optional
import cv2
import numpy as np
import logging
import config
from recognition.face_detector import DetectedFace
from recognition.model_utils import ensure_models_downloaded

logger = logging.getLogger("ExamAuth.FaceEncoder")


class FaceEncoder:
    """
    SFace Deep Neural Network Feature Encoder.
    Extracts 128-dimensional L2-normalized feature embeddings from aligned face crops.
    """

    def __init__(self, model_path: Optional[str] = None):
        ensure_models_downloaded()
        self.model_path = model_path or config.SFACE_MODEL_PATH
        logger.info(f"Initializing SFace FaceRecognizer from: {self.model_path}")
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config=""
        )

    def align_and_crop(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """
        Geometrically align the face using 5 facial landmarks (eyes, nose, mouth corners)
        and crop to standard 112x112 input tensor.
        """
        aligned_face = self.recognizer.alignCrop(image_bgr, face.raw_face)
        return aligned_face

    def extract_feature(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Compute the 128-dimensional L2-normalized embedding vector.
        Returns a (1, 128) float32 numpy array.
        """
        feature = self.recognizer.feature(aligned_face)
        # Ensure L2 normalization
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature.astype(np.float32)

    def encode_face(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Convenience method: Align crop and extract 128-d feature vector."""
        aligned = self.align_and_crop(image_bgr, face)
        return self.extract_feature(aligned)

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """Convert float32 embedding numpy array to raw bytes for SQLite BLOB storage."""
        if embedding is None:
            raise ValueError("Cannot serialize None embedding.")
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def deserialize_embedding(blob: bytes) -> np.ndarray:
        """Reconstruct (1, 128) float32 numpy array from SQLite BLOB bytes."""
        if not blob:
            raise ValueError("Empty or invalid embedding BLOB.")
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != config.EMBEDDING_DIM:
            raise ValueError(f"Invalid embedding dimension {arr.size}; expected {config.EMBEDDING_DIM}")
        return arr.reshape(1, config.EMBEDDING_DIM)
