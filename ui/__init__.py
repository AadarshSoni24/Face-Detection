"""
ui/__init__.py - User Interface package.
"""

from .dashboard import DashboardView
from .registration import RegistrationView
from .authentication import AuthenticationView
from .students import StudentsView
from .attendance import AttendanceView
from .import_photos import ImportPhotosView
from .sessions import SessionsView
from .audit_logs import AuditLogsView

__all__ = [
    "DashboardView",
    "RegistrationView",
    "AuthenticationView",
    "StudentsView",
    "AttendanceView",
    "ImportPhotosView",
    "SessionsView",
    "AuditLogsView"
]
