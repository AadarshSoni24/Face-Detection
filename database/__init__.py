"""
database/__init__.py - Database package initialization.
"""
from .db import Database
from .schema import init_db

__all__ = ["Database", "init_db"]
