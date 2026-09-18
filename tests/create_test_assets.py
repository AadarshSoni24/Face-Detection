"""
tests/create_test_assets.py - Prepares verified real-world test assets for automated testing and demo.

Generates:
1. Real facial image for Person A (Rahul Kumar) -> 1 face detected.
2. Real facial image for Person B (Priya Sharma) -> 1 face detected (different identity).
3. Synthetic composite image with multiple faces -> 2 faces detected.
4. Image with no face -> 0 faces detected.
5. Corrupted image file -> Corrupt format failure.
6. Sample CSV mapping file for testing bulk photo import.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import config
from recognition.face_detector import FaceDetector

TEST_ASSETS_DIR = config.DATA_DIR / "sample_import"


def setup_test_assets():
    """Download and construct verified test assets."""
    os.makedirs(TEST_ASSETS_DIR, exist_ok=True)

    face1_path = str(TEST_ASSETS_DIR / "rahul_face.jpg")
    face2_path = str(TEST_ASSETS_DIR / "priya_face.jpg")
    noface_path = str(TEST_ASSETS_DIR / "no_face.jpg")
    multiface_path = str(TEST_ASSETS_DIR / "multi_face.jpg")
    corrupt_path = str(TEST_ASSETS_DIR / "corrupt_image.jpg")
    csv_path = str(TEST_ASSETS_DIR / "students_mapping.csv")

    detector = FaceDetector()

    # 1. Person A: Rahul Kumar (Lena face)
    if not os.path.exists(face1_path):
        url_face1 = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        req1 = urllib.request.Request(url_face1, headers={"User-Agent": "Mozilla/5.0"})
        with open(face1_path, "wb") as f:
            f.write(urllib.request.urlopen(req1).read())

    img1 = cv2.imread(face1_path)
    f1_count = len(detector.detect(img1))
    print(f"Face 1 (Rahul Kumar) detected faces: {f1_count}")

    # 2. Person B: Priya Sharma (Distinct face from OpenCV samples: basketball1.png)
    # Ensure it is not an identical copy of Face 1
    needs_download = not os.path.exists(face2_path)
    if not needs_download:
        img_check = cv2.imread(face2_path)
        if img_check is not None and img1 is not None and np.array_equal(img_check, img1):
            needs_download = True

    if needs_download:
        url_face2 = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball1.png"
        req2 = urllib.request.Request(url_face2, headers={"User-Agent": "Mozilla/5.0"})
        with open(face2_path, "wb") as f:
            f.write(urllib.request.urlopen(req2).read())

    img2 = cv2.imread(face2_path)
    f2_count = len(detector.detect(img2))
    print(f"Face 2 (Priya Sharma) detected faces: {f2_count}")

    # 3. Multi-Face Image (Composite of Face 1 + Face 2)
    r1 = cv2.resize(img1, (400, 400))
    r2 = cv2.resize(img2, (400, 400))
    multi_img = np.hstack([r1, r2])
    cv2.imwrite(multiface_path, multi_img)
    f_multi_count = len(detector.detect(multi_img))
    print(f"Multi-face image detected faces: {f_multi_count}")

    # 4. No-Face Image (Color gradient)
    no_face_img = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(400):
        no_face_img[i, :, :] = (i % 255, (i * 2) % 255, (i * 3) % 255)
    cv2.imwrite(noface_path, no_face_img)
    f_none_count = len(detector.detect(no_face_img))
    print(f"No-face image detected faces: {f_none_count}")

    # 5. Corrupted Image File
    with open(corrupt_path, "wb") as f:
        f.write(b"CORRUPT_INVALID_JPEG_HEADER_NOT_AN_IMAGE_DATA_01010101")
    print("Corrupted image file created.")

    # 6. CSV Mapping File
    csv_content = (
        "roll_number,name,image_filename,course\n"
        "101,Rahul Kumar,rahul_face.jpg,Computer Science\n"
        "102,Priya Sharma,priya_face.jpg,Information Technology\n"
        "103,Amit Verma,no_face.jpg,Electronics\n"
        "104,Sneha Patel,multi_face.jpg,Mechanical\n"
        "105,Vikram Singh,corrupt_image.jpg,Civil\n"
    )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    print("CSV mapping file created.")

    # Remove any scratch temporary file
    temp_basket = config.DATA_DIR / "sample_import" / "test_basket.png"
    if temp_basket.exists():
        temp_basket.unlink()

    print(f"All test assets successfully initialized in: {TEST_ASSETS_DIR}")
    return {
        "face1": face1_path,
        "face2": face2_path,
        "multi_face": multiface_path,
        "no_face": noface_path,
        "corrupt": corrupt_path,
        "csv": csv_path
    }


if __name__ == "__main__":
    setup_test_assets()
