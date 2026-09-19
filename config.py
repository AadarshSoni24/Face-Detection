"""
config.py - Centralized configuration for Biometric-Based Exam Authentication System.

All application paths, face recognition thresholds, camera parameters, and logging
settings are managed here.
"""

import os
import logging
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
TESTS_DIR = BASE_DIR / "tests"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Database Path
DB_PATH = str(DATA_DIR / "exam_auth.db")

# Deep Learning Model Configuration
# YuNet Face Detector (ONNX) - Detects bounding box and 5 facial landmarks
YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = str(MODELS_DIR / YUNET_MODEL_NAME)
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# SFace Face Recognizer (ONNX) - Extracts 128-dimensional L2-normalized feature embeddings
SFACE_MODEL_NAME = "face_recognition_sface_2021dec.onnx"
SFACE_MODEL_PATH = str(MODELS_DIR / SFACE_MODEL_NAME)
SFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Face Detection Parameters
FACE_DETECTION_SCORE_THRESHOLD = 0.80   # Min confidence score for YuNet detection [0.0 - 1.0]
FACE_DETECTION_NMS_THRESHOLD = 0.30     # Non-Maximum Suppression threshold
FACE_DETECTION_DEFAULT_SIZE = (320, 320) # Dynamic input size adjusted per frame

# Face Recognition & Matching Parameters
# SFace outputs a 128-dimensional unit embedding vector.
# Cosine Similarity metric: cos(u, v) = dot(u, v) / (|u| * |v|).
# Range: [-1.0, 1.0], where 1.0 represents an identical match.
# Official OpenCV Zoo evaluation benchmark:
# - Cosine Threshold: 0.363 (at 0.1% False Acceptance Rate / FAR 1e-3)
# - Scores >= 0.363 indicate the faces belong to the same identity.
# - Higher threshold (e.g. 0.45) is stricter (fewer false accepts, more false rejects).
# - Lower threshold (e.g. 0.30) is looser (more tolerant to illumination changes).
FACE_MATCH_METRIC = "COSINE"  # Supported: "COSINE", "NORM_L2"
FACE_MATCH_THRESHOLD = 0.363  # Cosine similarity threshold for verification
NORM_L2_THRESHOLD = 1.128     # L2 distance threshold (if using NORM_L2 metric)
EMBEDDING_DIM = 128           # SFace embedding dimension

# Camera Parameters
DEFAULT_CAMERA_INDEX = 0
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_FPS = 30

# UI Design Theme & Typography
APP_TITLE = "Biometric-Based Exam Authentication System"
APP_SUBTITLE = "Academic Prototype • Computer Vision & Biometrics"
APP_WINDOW_SIZE = "1180x760"
APP_MIN_SIZE = (1000, 680)

# Colors
COLOR_BG = "#F8FAFC"          # Slate 50
COLOR_CARD_BG = "#FFFFFF"     # Pure White
COLOR_HEADER_BG = "#0F172A"   # Slate 900
COLOR_SIDEBAR_BG = "#1E293B"  # Slate 800
COLOR_TEXT_PRIMARY = "#0F172A"# Slate 900
COLOR_TEXT_MUTED = "#64748B"  # Slate 500
COLOR_PRIMARY = "#2563EB"     # Royal Blue 600
COLOR_PRIMARY_HOVER = "#1D4ED8"
COLOR_SUCCESS = "#059669"     # Emerald 600
COLOR_SUCCESS_BG = "#ECFDF5"  # Emerald 50
COLOR_DANGER = "#DC2626"      # Red 600
COLOR_DANGER_BG = "#FEF2F2"   # Red 50
COLOR_WARNING = "#D97706"     # Amber 600
COLOR_WARNING_BG = "#FFFBEB"  # Amber 50
COLOR_BORDER = "#E2E8F0"      # Slate 200

# Anti-Spoofing & Liveness Detection Parameters
LIVENESS_CHECK_ENABLED_DEFAULT = True  # Whether liveness mode is toggled on by default
LIVENESS_THRESHOLD = 0.60              # Overall liveness confidence threshold [0.0 - 1.0]
LIVENESS_EAR_THRESHOLD = 0.21          # Eye Aspect Ratio threshold below which eye is considered closed/blinking
LIVENESS_BUFFER_SIZE = 20              # Sliding window of frame landmarks to evaluate temporal motion
LIVENESS_MIN_FRAMES_REQUIRED = 8       # Minimum frames needed to evaluate motion before verdict
LIVENESS_MICRO_MOTION_MIN = 0.003      # Minimum variance in normalized facial landmarks for live 3D face

# Image Quality Assessment Parameters
QUALITY_MIN_BRIGHTNESS = 45.0          # Minimum mean pixel intensity (avoid too dark)
QUALITY_MAX_BRIGHTNESS = 220.0         # Maximum mean pixel intensity (avoid washed out)
QUALITY_MIN_SHARPNESS = 40.0           # Minimum Laplacian variance for sharp focus
QUALITY_MIN_FACE_AREA_RATIO = 0.03     # Face bounding box must cover >= 3% of frame area
QUALITY_MAX_FACE_AREA_RATIO = 0.70     # Face bounding box must cover <= 70% of frame area

# Exam Session Defaults
DEFAULT_EXAM_HALL = "Hall 101 (Main Auditorium)"
DEFAULT_INSTITUTION_NAME = "Central University Examination Board"

# Logging Configuration
LOG_FILE = str(LOGS_DIR / "exam_auth.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ExamAuth")

# Academic & Security Notice
LIVENESS_DISCLAIMER = (
    "NOTICE: The system includes active landmark-based micro-motion & blink liveness analysis. "
    "For maximum security against 3D masks and IR spoofs, dedicated multi-spectral hardware cameras are recommended in production."
)

