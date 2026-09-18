"""
recognition/model_utils.py - Model file verification and automatic download utility.
"""

import os
import urllib.request
import logging
import config

logger = logging.getLogger("ExamAuth.Models")


def ensure_models_downloaded() -> None:
    """Check that YuNet and SFace ONNX models are present; download if missing."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # 1. YuNet Face Detector
    if not os.path.exists(config.YUNET_MODEL_PATH):
        logger.info(f"Downloading YuNet face detector model to: {config.YUNET_MODEL_PATH}")
        try:
            urllib.request.urlretrieve(config.YUNET_MODEL_URL, config.YUNET_MODEL_PATH)
            logger.info("YuNet model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download YuNet model: {e}")
            raise RuntimeError(f"Could not download YuNet model from {config.YUNET_MODEL_URL}: {e}")

    # 2. SFace Face Recognizer
    if not os.path.exists(config.SFACE_MODEL_PATH):
        logger.info(f"Downloading SFace face recognizer model to: {config.SFACE_MODEL_PATH}")
        try:
            urllib.request.urlretrieve(config.SFACE_MODEL_URL, config.SFACE_MODEL_PATH)
            logger.info("SFace model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download SFace model: {e}")
            raise RuntimeError(f"Could not download SFace model from {config.SFACE_MODEL_URL}: {e}")
