"""
Authentication and Authorization for PhotoSight API

Provides token-based authentication and session management.
"""

import secrets
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Set
from dataclasses import dataclass, field
import jwt
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """User session information."""
    user_id: str
    session_token: str
    api_key: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    permissions: Set[str] = field(default_factory=set)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Check if session has expired."""
        expiry_time = self.last_activity + timedelta(hours=timeout_hours)
        return datetime.utcnow() > expiry_time
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


class APIAuth:
    """
    Handles API authentication and authorization.
    
    Supports:
    - API key authentication
    - JWT token generation and validation
    - Session management
    - Permission-based access control
    """
    
    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.sessions: Dict[str, UserSession] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id mapping
        
        # Default permissions
        self.permission_sets = {
            'viewer': {'sessions.create', 'images.read', 'recipes.read'},
            'editor': {'sessions.create', 'sessions.update', 'images.read', 
                      'images.process', 'recipes.read', 'recipes.write'},
            'admin': {'*'}  # All permissions
        }
        
        logger.info("APIAuth initialized")
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate a new API key for a user."""
        # Generate secure random key
        api_key = f"psk_{secrets.token_urlsafe(32)}"
        
        # Store mapping
        self.api_keys[api_key] = user_id
        
        logger.info(f"Generated API key for user {user_id}")
        return api_key
    
    def create_session(self, api_key: str, permissions: Optional[Set[str]] = None) -> Optional[UserSession]:
        """
        Create a new session for an API key.
        
        Args:
            api_key: API key to authenticate
            permissions: Optional custom permissions, defaults to 'editor'
            
        Returns:
            UserSession if successful, None if authentication failed
        """
        # Validate API key
        user_id = self.api_keys.get(api_key)
        if not user_id:
            logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
            return None
        
        # Generate session token
        session_token = self._generate_jwt_token(user_id)
        
        # Create session
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            api_key=api_key,
            permissions=permissions or self.permission_sets['editor']
        )
        
        # Store session
        self.sessions[session_token] = session
        
        logger.info(f"Created session for user {user_id}")
        return session
    
    def validate_token(self, token: str) -> Optional[UserSession]:
        """
        Validate a session token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            UserSession if valid, None otherwise
        """
        try:
            # Decode JWT
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if session exists
            session = self.sessions.get(token)
            if not session:
                return None
            
            # Check expiry
            if session.is_expired(self.token_expiry_hours):
                del self.sessions[token]
                return None
            
            # Update activity
            session.update_activity()
            
            return session
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token attempted")
            if token in self.sessions:
                del self.sessions[token]
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token attempted")
            return None
    
    def check_permission(self, session: UserSession, permission: str) -> bool:
        """
        Check if session has a specific permission.
        
        Args:
            session: User session
            permission: Permission to check (e.g., 'images.process')
            
        Returns:
            True if permission granted
        """
        # Admin has all permissions
        if '*' in session.permissions:
            return True
        
        # Check specific permission
        return permission in session.permissions
    
    def revoke_session(self, session_token: str) -> bool:
        """Revoke a session."""
        if session_token in self.sessions:
            del self.sessions[session_token]
            logger.info(f"Revoked session {session_token[:10]}...")
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        expired = []
        for token, session in self.sessions.items():
            if session.is_expired(self.token_expiry_hours):
                expired.append(token)
        
        for token in expired:
            del self.sessions[token]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def _generate_jwt_token(self, user_id: str) -> str:
        """Generate a JWT token."""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        return token
    
    def get_session_info(self, session_token: str) -> Optional[Dict[str, any]]:
        """Get session information (safe for client)."""
        session = self.sessions.get(session_token)
        if not session:
            return None
        
        return {
            'user_id': session.user_id,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'permissions': list(session.permissions),
            'metadata': session.metadata
        }


class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    For production, use Redis or similar.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = {}
    
    def check_rate_limit(self, identifier: str) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: User ID or API key
            
        Returns:
            True if within limit, False if exceeded
        """
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Get request timestamps
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests
        self.requests[identifier] = [
            ts for ts in self.requests[identifier] if ts > minute_ago
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get number of remaining requests in current window."""
        if identifier not in self.requests:
            return self.requests_per_minute
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        active_requests = [
            ts for ts in self.requests[identifier] if ts > minute_ago
        ]
        
        return max(0, self.requests_per_minute - len(active_requests))