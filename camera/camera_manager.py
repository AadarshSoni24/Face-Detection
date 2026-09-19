"""
camera/camera_manager.py - Thread-safe OpenCV Webcam Capture & Lifecycle Management.

Handles camera initialization, background frame polling, safe release, error handling
for unavailable devices, and an optional test/file mode for environments without a webcam.
"""

import os
import threading
import time
from typing import Optional, Union, List
import cv2
import numpy as np
import logging
import config

logger = logging.getLogger("ExamAuth.Camera")


class CameraUnavailableError(Exception):
    """Raised when the specified webcam device cannot be accessed or opened."""
    pass


class CameraManager:
    """
    Thread-safe camera stream manager.
    Runs frame capture on a separate background thread to keep Tkinter GUI responsive.
    Guarantees proper hardware release on stop/cleanup.
    """

    def __init__(self, camera_index: int = config.DEFAULT_CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.current_frame: Optional[np.ndarray] = None
        self.worker_thread: Optional[threading.Thread] = None

        # Virtual/Test mode parameters (for environments without physical camera)
        self.is_virtual_mode = False
        self.virtual_image: Optional[np.ndarray] = None

    @staticmethod
    def get_available_cameras(max_tested: int = 3) -> List[int]:
        """Discover available webcam device indices on the system."""
        available = []
        for idx in range(max_tested):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(idx)
                cap.release()
            except Exception:
                pass
        return available if available else [0]

    def switch_camera(self, new_index: int) -> None:
        """Switch to a different camera device dynamically."""
        if self.is_running:
            self.stop()
        self.camera_index = new_index
        self.start(camera_index=new_index)

    def start(self, camera_index: Optional[int] = None) -> None:
        """
        Open the hardware webcam and begin background frame capture.
        Raises CameraUnavailableError if webcam cannot be opened.
        """

        if self.is_running:
            logger.debug("Camera stream is already active.")
            return

        if camera_index is not None:
            self.camera_index = camera_index

        logger.info(f"Attempting to open camera device index: {self.camera_index}")

        # Try configured camera index first, then fallbacks 0, 1, 2
        indices_to_try = [self.camera_index]
        for alt_idx in [0, 1, 2]:
            if alt_idx not in indices_to_try:
                indices_to_try.append(alt_idx)

        cap = None
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

        for idx in indices_to_try:
            for backend in backends:
                try:
                    test_cap = cv2.VideoCapture(idx, backend)
                    if test_cap.isOpened():
                        # Verify we can actually read a frame
                        ret, test_frame = test_cap.read()
                        if ret and test_frame is not None:
                            cap = test_cap
                            self.camera_index = idx
                            logger.info(f"Successfully opened camera at index {idx} with backend {backend}")
                            break
                        else:
                            test_cap.release()
                    else:
                        test_cap.release()
                except Exception as e:
                    logger.debug(f"Index {idx}, backend {backend} failed: {e}")
            if cap is not None:
                break

        if cap is None or not cap.isOpened():
            logger.error(f"Failed to access camera index {self.camera_index}")
            raise CameraUnavailableError("Camera could not be accessed.")

        # Configure frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)

        self.cap = cap
        self.is_virtual_mode = False
        self.is_running = True

        self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True, name="WebcamThread")
        self.worker_thread.start()
        logger.info("Webcam background capture thread started.")

    def start_virtual_mode(self, image_source: Union[str, np.ndarray]) -> None:
        """
        Start virtual camera mode using an image file or numpy array.
        Useful for testing in headless or VM environments without a physical camera.
        """
        if self.is_running:
            self.stop()

        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None:
                raise ValueError(f"Could not load virtual image from path: {image_source}")
            self.virtual_image = img
        elif isinstance(image_source, np.ndarray):
            self.virtual_image = image_source.copy()
        else:
            raise ValueError("image_source must be a file path string or numpy ndarray.")

        self.is_virtual_mode = True
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._virtual_loop, daemon=True, name="VirtualCamThread")
        self.worker_thread.start()
        logger.info("Virtual camera mode active.")

    def _capture_loop(self) -> None:
        """Continuously reads frames from physical webcam device."""
        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.current_frame = frame
            else:
                time.sleep(0.01)
            time.sleep(1.0 / config.CAMERA_FPS)

    def _virtual_loop(self) -> None:
        """Simulates camera feed from virtual image at target FPS."""
        while self.is_running and self.is_virtual_mode:
            if self.virtual_image is not None:
                with self.lock:
                    self.current_frame = self.virtual_image.copy()
            time.sleep(1.0 / config.CAMERA_FPS)

    def read_frame(self) -> Optional[np.ndarray]:
        """Thread-safe access to the latest captured video frame (BGR format)."""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def stop(self) -> None:
        """Stop background capture and release hardware camera resources."""
        if not self.is_running:
            return

        logger.info("Stopping camera and releasing resources...")
        self.is_running = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing VideoCapture: {e}")
            self.cap = None

        with self.lock:
            self.current_frame = None

        self.is_virtual_mode = False
        self.virtual_image = None
        logger.info("Camera successfully released.")

    def release(self) -> None:
        """Alias for stop() to support standard OpenCV idiom."""
        self.stop()

    def __del__(self):
        """Finalizer to ensure hardware release even if explicitly forgotten."""
        self.release()
