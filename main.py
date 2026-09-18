"""
main.py - Main Entry Point for Biometric-Based Exam Authentication System.

Orchestrates application initialization, deep learning model verification, database setup,
Tkinter window lifecycle, navigation state management, and camera hardware resource cleanup.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Dict

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from database.db import Database
from camera.camera_manager import CameraManager
from recognition.model_utils import ensure_models_downloaded
from recognition.face_detector import FaceDetector
from recognition.face_encoder import FaceEncoder
from recognition.face_matcher import FaceMatcher

from ui.styles import apply_theme
from ui.dashboard import DashboardView
from ui.registration import RegistrationView
from ui.authentication import AuthenticationView
from ui.students import StudentsView
from ui.attendance import AttendanceView
from ui.import_photos import ImportPhotosView

logger = logging.getLogger("ExamAuth.Main")


class ExamAuthApplication:
    """Master controller managing Tkinter window and screen lifecycle."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{config.APP_TITLE} — Academic Prototype")
        self.root.geometry(config.APP_WINDOW_SIZE)
        self.root.minsize(*config.APP_MIN_SIZE)

        # Apply design theme
        self.style = apply_theme(self.root)

        # Clean shutdown protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize Core Singletons
        logger.info("Initializing system singletons and database...")
        ensure_models_downloaded()
        self.db = Database(config.DB_PATH)
        self.camera_manager = CameraManager(camera_index=config.DEFAULT_CAMERA_INDEX)
        self.detector = FaceDetector(model_path=config.YUNET_MODEL_PATH)
        self.encoder = FaceEncoder(model_path=config.SFACE_MODEL_PATH)
        self.matcher = FaceMatcher(threshold=config.FACE_MATCH_THRESHOLD, metric=config.FACE_MATCH_METRIC)

        # Main View Container
        self.container = tk.Frame(self.root, bg=config.COLOR_BG)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Instantiate View Screens
        self.views: Dict[str, ttk.Frame] = {}
        self.current_view_name = ""

        self._init_views()
        self.show_view("dashboard")

    def _init_views(self) -> None:
        """Create and stack all application view screens."""
        logger.info("Initializing UI views...")

        # 1. Dashboard View
        self.views["dashboard"] = DashboardView(
            parent=self.container,
            db=self.db,
            navigate_callback=self.show_view
        )
        self.views["dashboard"].grid(row=0, column=0, sticky="nsew")

        # 2. Registration View
        self.views["registration"] = RegistrationView(
            parent=self.container,
            db=self.db,
            camera_manager=self.camera_manager,
            detector=self.detector,
            encoder=self.encoder,
            navigate_callback=self.show_view
        )
        self.views["registration"].grid(row=0, column=0, sticky="nsew")

        # 3. Authentication View
        self.views["authentication"] = AuthenticationView(
            parent=self.container,
            db=self.db,
            camera_manager=self.camera_manager,
            detector=self.detector,
            encoder=self.encoder,
            matcher=self.matcher,
            navigate_callback=self.show_view
        )
        self.views["authentication"].grid(row=0, column=0, sticky="nsew")

        # 4. Student Records View
        self.views["students"] = StudentsView(
            parent=self.container,
            db=self.db,
            navigate_callback=self.show_view
        )
        self.views["students"].grid(row=0, column=0, sticky="nsew")

        # 5. Attendance Records View
        self.views["attendance"] = AttendanceView(
            parent=self.container,
            db=self.db,
            navigate_callback=self.show_view
        )
        self.views["attendance"].grid(row=0, column=0, sticky="nsew")

        # 6. Bulk Photo Import View
        self.views["import_photos"] = ImportPhotosView(
            parent=self.container,
            db=self.db,
            detector=self.detector,
            encoder=self.encoder,
            navigate_callback=self.show_view
        )
        self.views["import_photos"].grid(row=0, column=0, sticky="nsew")

    def show_view(self, view_name: str) -> None:
        """Switch active screen, managing on_hide and on_show lifecycles."""
        if view_name not in self.views:
            logger.error(f"Requested non-existent view: '{view_name}'")
            return

        # Deactivate current screen
        if self.current_view_name and self.current_view_name in self.views:
            current_screen = self.views[self.current_view_name]
            if hasattr(current_screen, "on_hide"):
                current_screen.on_hide()

        # Activate target screen
        target_screen = self.views[view_name]
        target_screen.tkraise()
        self.current_view_name = view_name

        if hasattr(target_screen, "on_show"):
            target_screen.on_show()

        logger.info(f"Navigated to view: '{view_name}'")

    def on_close(self) -> None:
        """Safely release webcam hardware and exit."""
        logger.info("Application shutdown requested. Releasing resources...")
        try:
            # Notify active screen
            if self.current_view_name and self.current_view_name in self.views:
                screen = self.views[self.current_view_name]
                if hasattr(screen, "on_hide"):
                    screen.on_hide()

            # Release camera hardware
            self.camera_manager.stop()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            self.root.destroy()


def main():
    """Application bootstrap."""
    logger.info("Starting Biometric-Based Exam Authentication System...")
    root = tk.Tk()
    app = ExamAuthApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
