"""
PhotoSight API Layer

Provides RESTful and WebSocket APIs for GUI integration.
"""

from .app import create_app
from .models import APIResponse, ErrorResponse, SessionInfo
from .auth import APIAuth, UserSession

__all__ = [
    'create_app',
    'APIResponse',
    'ErrorResponse', 
    'SessionInfo',
    'APIAuth',
    'UserSession'
]