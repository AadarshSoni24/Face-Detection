"""
recognition/face_matcher.py - Face Feature Comparison & Threshold Verification.

Provides similarity metrics and verification logic to compare a live facial embedding
against stored reference embedding(s).

========================================================================================
TECHNICAL DOCUMENTATION ON SIMILARITY METRIC AND THRESHOLD SELECTION:
========================================================================================
1. Metric: Cosine Similarity
   - For two L2-normalized 128-dimensional embedding vectors u and v:
     Cosine Similarity = dot(u, v) / (|u| * |v|) = dot(u, v)
   - Value Range: [-1.0, 1.0]
     * 1.0 represents mathematically identical facial feature representations.
     * Higher positive values represent strong facial similarity.
     * Lower or negative values represent distinct facial geometries.

2. Chosen Threshold: 0.363
   - Recommended by the official OpenCV Zoo benchmark for SFace on the LFW / MegaFace
     datasets.
   - At threshold = 0.363, the False Acceptance Rate (FAR) is approximately 0.1% (1e-3).
   - This provides a balanced operating point for academic examination authentication:
     minimizing unauthorized impostor entry while tolerating modest natural variations
     in lighting, head angle, and expressions.

3. Adjusting During Testing:
   - To make authentication stricter (fewer false accepts): Increase threshold (e.g., 0.45).
   - To make authentication more lenient (fewer false rejects): Decrease threshold (e.g., 0.30).
   - Configurable in config.py via FACE_MATCH_THRESHOLD.
========================================================================================
"""

from dataclasses import dataclass
from typing import List, Union, Tuple
import numpy as np
import logging
import config

logger = logging.getLogger("ExamAuth.FaceMatcher")


@dataclass
class MatchResult:
    """Encapsulates the outcome of a facial comparison."""
    is_match: bool
    similarity_score: float
    threshold: float
    metric: str
    details: str


class FaceMatcher:
    """
    Compares facial embedding vectors using Cosine Similarity against
    the configured verification threshold.
    """

    def __init__(self, threshold: float = config.FACE_MATCH_THRESHOLD, metric: str = config.FACE_MATCH_METRIC):
        self.threshold = threshold
        self.metric = metric.upper()

    @staticmethod
    def compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two 1D or (1, D) embedding vectors:
        cos_sim = dot(u, v) / (norm(u) * norm(v))
        """
        u = feat1.flatten()
        v = feat2.flatten()

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u == 0 or norm_v == 0:
            return 0.0

        sim = float(np.dot(u, v) / (norm_u * norm_v))
        # Clip to valid mathematical range [-1.0, 1.0] to guard against floating-point jitter
        return max(-1.0, min(1.0, sim))

    def compare(
        self,
        query_embedding: np.ndarray,
        reference_embedding: np.ndarray
    ) -> MatchResult:
        """
        Compare query embedding against a single reference embedding.
        Returns MatchResult with is_match, similarity score, and diagnostic details.
        """
        similarity = self.compute_cosine_similarity(query_embedding, reference_embedding)
        is_match = similarity >= self.threshold

        if is_match:
            details = f"Face match SUCCESS: similarity {similarity:.3f} >= threshold {self.threshold:.3f}"
        else:
            details = f"Face match FAILED: similarity {similarity:.3f} < threshold {self.threshold:.3f}"

        return MatchResult(
            is_match=is_match,
            similarity_score=similarity,
            threshold=self.threshold,
            metric=self.metric,
            details=details
        )

    def compare_multi(
        self,
        query_embedding: np.ndarray,
        reference_embeddings: List[np.ndarray]
    ) -> MatchResult:
        """
        Compare query embedding against multiple approved reference embeddings
        (e.g., student registered with multiple photos).
        Takes the best (maximum) similarity score among all approved references.
        """
        if not reference_embeddings:
            return MatchResult(
                is_match=False,
                similarity_score=0.0,
                threshold=self.threshold,
                metric=self.metric,
                details="No reference embeddings available for student."
            )

        best_score = -1.0
        for ref in reference_embeddings:
            score = self.compute_cosine_similarity(query_embedding, ref)
            if score > best_score:
                best_score = score

        is_match = best_score >= self.threshold
        if is_match:
            details = (
                f"Face match SUCCESS (best of {len(reference_embeddings)} references): "
                f"similarity {best_score:.3f} >= threshold {self.threshold:.3f}"
            )
        else:
            details = (
                f"Face match FAILED (best of {len(reference_embeddings)} references): "
                f"similarity {best_score:.3f} < threshold {self.threshold:.3f}"
            )

        return MatchResult(
            is_match=is_match,
            similarity_score=best_score,
            threshold=self.threshold,
            metric=self.metric,
            details=details
        )
