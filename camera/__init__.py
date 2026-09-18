"""
camera/__init__.py - Camera management package.
"""

from .camera_manager import CameraManager, CameraUnavailableError

__all__ = ["CameraManager", "CameraUnavailableError"]
