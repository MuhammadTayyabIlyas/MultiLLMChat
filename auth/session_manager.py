"""
Cookie-based session management for persistent authentication
"""
from __future__ import annotations

import hashlib
import secrets
import time
from typing import Optional

import extra_streamlit_components as stx
import streamlit as st

from auth.users import user_manager, User


class SessionManager:
    """Manages user sessions using cookies for persistence across page refreshes"""
    
    def __init__(self):
        self.cookie_manager = stx.CookieManager()
        self.cookie_name = "ai_chat_studio_session"
        self.cookie_max_age = 30 * 24 * 60 * 60  # 30 days
    
    def get_current_user(self) -> Optional[User]:
        """Get the currently logged-in user from cookies or session state"""
        # First check session state (already logged in)
        if "user" in st.session_state:
            return st.session_state.user
        
        # Check cookie for session token
        session_token = self.cookie_manager.get(self.cookie_name)
        if not session_token:
            return None
        
        # Validate session token
        try:
            user = self._validate_session_token(session_token)
            if user:
                # Restore to session state
                st.session_state.user = user
                return user
        except Exception:
            # Invalid token, clear cookie
            self.clear_session()
        
        return None
    
    def create_session(self, user: User) -> str:
        """Create a new session and set cookie"""
        # Generate session token
        session_token = self._generate_session_token(user.id)
        
        # Store in session state
        st.session_state.user = user
        
        # Set cookie
        self.cookie_manager.set(
            self.cookie_name,
            session_token,
            expires_at=time.time() + self.cookie_max_age,
            path="/"
        )
        
        return session_token
    
    def clear_session(self):
        """Clear session and delete cookie"""
        # Clear from session state
        if "user" in st.session_state:
            del st.session_state.user
        
        # Delete cookie
        self.cookie_manager.delete(self.cookie_name, path="/")
    
    def regenerate_session_token(self, user: User) -> str:
        """Regenerate session token for security (e.g., on privilege change)"""
        return self.create_session(user)
    
    def _generate_session_token(self, user_id: str) -> str:
        """Generate a secure session token"""
        # Create token with user_id, timestamp, and random salt
        timestamp = str(int(time.time()))
        salt = secrets.token_hex(16)
        
        # Format: user_id:timestamp:salt:hash
        data = f"{user_id}:{timestamp}:{salt}"
        token_hash = hashlib.sha256(data.encode()).hexdigest()
        
        return f"{user_id}:{timestamp}:{salt}:{token_hash}"
    
    def _validate_session_token(self, token: str) -> Optional[User]:
        """Validate session token and return user if valid"""
        try:
            # Parse token
            parts = token.split(":")
            if len(parts) != 4:
                return None
            
            user_id, timestamp, salt, token_hash = parts
            
            # Verify token integrity
            expected_hash = hashlib.sha256(f"{user_id}:{timestamp}:{salt}".encode()).hexdigest()
            if token_hash != expected_hash:
                return None
            
            # Check if not expired (30 days max)
            token_time = int(timestamp)
            if time.time() - token_time > self.cookie_max_age:
                return None
            
            # Get user from database
            return user_manager.get_user(user_id)
            
        except Exception:
            return None


# Global session manager instance
session_manager = SessionManager()


def require_auth():
    """Decorator to require authentication using session manager"""
    user = session_manager.get_current_user()
    if not user:
        st.error("🔒 Please log in to access this feature.")
        st.stop()
    return user


def get_current_user() -> Optional[User]:
    """Get currently logged in user"""
    return session_manager.get_current_user()