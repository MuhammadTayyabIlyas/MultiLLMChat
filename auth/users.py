"""
User authentication and subscription management
"""
from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import streamlit as st
from db import get_db_connection


@dataclass
class User:
    id: str
    email: str
    subscription_tier: str = "free"  # free, starter, pro, enterprise
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    created_at: int = 0
    last_login: int = 0
    api_key: Optional[str] = None  # For API access
    password_hash: Optional[str] = None  # For user authentication
    is_admin: bool = False  # For admin privileges
    reset_token: Optional[str] = None  # For password reset
    reset_token_expiry: int = 0  # Token expiration timestamp
    created_by: Optional[str] = None  # For team management


class UserManager:
    def __init__(self):
        self.init_table()
    
    def init_table(self):
        """Create users table if not exists"""
        conn = get_db_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                subscription_tier TEXT DEFAULT 'free',
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                created_at INTEGER,
                last_login INTEGER,
                api_key TEXT,
                password_hash TEXT,
                reset_token TEXT,
                reset_token_expiry INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def create_user(self, email: str, password: str) -> User:
        """Create a new user with hashed password"""
        user_id = secrets.token_hex(16)
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        password_hash = self._hash_password(password)
        
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO users (id, email, subscription_tier, created_at, last_login, api_key, password_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, email, "free", int(time.time()), int(time.time()), api_key, password_hash))
        conn.commit()
        conn.close()
        
        return self.get_user(user_id)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        
        if row:
            return User(**{k: row[k] for k in row.keys()})
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        
        if row:
            return User(**{k: row[k] for k in row.keys()})
        return None
    
    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Verify password and return user"""
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        
        if not row:
            return None
        
        if self._verify_password(password, row["password_hash"]):
            # Update last login
            self.update_last_login(row["id"])
            return User(**{k: row[k] for k in row.keys()})
        
        return None
    
    def update_subscription(self, user_id: str, tier: str, 
                          stripe_customer_id: Optional[str] = None,
                          stripe_subscription_id: Optional[str] = None):
        """Update user subscription"""
        conn = get_db_connection()
        conn.execute("""
            UPDATE users 
            SET subscription_tier = ?, stripe_customer_id = ?, stripe_subscription_id = ?
            WHERE id = ?
        """, (tier, stripe_customer_id, stripe_subscription_id, user_id))
        conn.commit()
        conn.close()
    
    def get_usage_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user usage statistics"""
        conn = get_db_connection()
        
        # Count messages from last 24 hours
        day_ago = int(time.time()) - (24 * 60 * 60)
        row = conn.execute("""
            SELECT COUNT(*) as message_count 
            FROM messages 
            WHERE session_id = ? AND role = 'user' AND timestamp > ?
        """, (user_id, day_ago)).fetchone()
        
        # Get comparison mode usage
        comp_row = conn.execute("""
            SELECT COUNT(*) as comp_count 
            FROM messages 
            WHERE session_id = ? AND model = 'Comparison' AND timestamp > ?
        """, (user_id, day_ago)).fetchone()
        
        conn.close()
        
        return {
            "messages_today": row["message_count"],
            "comparisons_today": comp_row["comp_count"] if comp_row else 0
        }
    
    def get_tier_limits(self, tier: str) -> Dict[str, Any]:
        """Get limits for a subscription tier"""
        limits = {
            "free": {
                "daily_messages": 50,
                "daily_comparisons": 5,
                "models": ["openai-gpt-4o-mini", "gemini-2.0-flash-lite"],
                "api_calls": 0
            },
            "starter": {
                "daily_messages": 500,
                "daily_comparisons": 50,
                "models": "all",
                "api_calls": 0
            },
            "pro": {
                "daily_messages": float('inf'),
                "daily_comparisons": float('inf'),
                "models": "all",
                "api_calls": 10000
            },
            "enterprise": {
                "daily_messages": float('inf'),
                "daily_comparisons": float('inf'),
                "models": "all",
                "api_calls": 100000
            }
        }
        return limits.get(tier, limits["free"])
    
    def can_access_model(self, user: User, model_key: str) -> bool:
        """Check if user can access a specific model"""
        limits = self.get_tier_limits(user.subscription_tier)
        
        if limits["models"] == "all":
            return True
        
        return model_key in limits["models"]
    
    def is_rate_limited(self, user: User) -> tuple[bool, str]:
        """Check if user has exceeded their rate limits"""
        if user.subscription_tier in ["pro", "enterprise"]:
            return False, ""
        
        usage = self.get_usage_stats(user.id)
        limits = self.get_tier_limits(user.subscription_tier)
        
        if usage["messages_today"] >= limits["daily_messages"]:
            return True, f"Daily message limit reached ({limits['daily_messages']}/day)"
        
        return False, ""
    
    def _hash_password(self, password: str) -> str:
        """Securely hash password"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{pwd_hash.hex()}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt, hash_hex = stored_hash.split(':')
            pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return pwd_hash.hex() == hash_hex
        except Exception:
            return False
    
    def update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        conn = get_db_connection()
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                    (int(time.time()), user_id))
        conn.commit()
        conn.close()
    
    def generate_reset_token(self, email: str) -> Optional[str]:
        """Generate password reset token"""
        token = secrets.token_urlsafe(32)
        expiry = int(time.time()) + (60 * 60)  # 1 hour
        
        conn = get_db_connection()
        conn.execute("""
            UPDATE users 
            SET reset_token = ?, reset_token_expiry = ? 
            WHERE email = ?
        """, (token, expiry, email))
        conn.commit()
        conn.close()
        
        return token


# Global user manager instance
user_manager = UserManager()


def require_auth():
    """Decorator to require authentication"""
    if "user" not in st.session_state:
        st.error("Please log in to access this feature.")
        st.stop()
    return st.session_state.user


def check_subscription_requirement(required_tier: str = "free"):
    """Check if user's subscription meets requirement"""
    if "user" not in st.session_state:
        return False
    
    user = st.session_state.user
    
    tier_order = ["free", "starter", "pro", "enterprise"]
    user_tier_idx = tier_order.index(user.subscription_tier)
    required_tier_idx = tier_order.index(required_tier)
    
    return user_tier_idx >= required_tier_idx