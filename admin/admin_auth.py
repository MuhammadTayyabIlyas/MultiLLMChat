"""
Admin authentication and authorization
"""
from __future__ import annotations

import os
import streamlit as st
from auth.users import user_manager, User
from db import get_db_connection
import time

# Admin emails (configure these in your secrets)
ADMIN_EMAILS = st.secrets.get("ADMIN_EMAILS", ["admin@aichatstudio.com", "tayyab@example.com"])


def is_admin(user: User) -> bool:
    """Check if user has admin privileges"""
    if not user:
        return False
    
    # Check if user email is in admin list
    if user.email in ADMIN_EMAILS:
        return True
    
    # Check if user has admin role in database
    conn = get_db_connection()
    row = conn.execute("SELECT is_admin FROM users WHERE id = ?", (user.id,)).fetchone()
    conn.close()
    
    return bool(row and row[0])


def require_admin():
    """Decorator to require admin access"""
    if "user" not in st.session_state:
        st.error("🔒 Please log in to access the admin panel")
        st.stop()
    
    user = st.session_state.user
    
    if not is_admin(user):
        st.error("🔐 You do not have admin privileges")
        st.info("This area is restricted to administrators only.")
        
        # Show current user's info
        st.markdown("---")
        st.markdown("### Your Account")
        st.write(f"**Email:** {user.email}")
        st.write(f"**Tier:** {user.subscription_tier}")
        
        if st.button("← Back to Chat"):
            st.switch_page("app.py")
        
        st.stop()


def init_admin_table():
    """Initialize admin flags in database"""
    conn = get_db_connection()
    
    # Add is_admin column if not exists
    try:
        conn.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE")
        conn.commit()
    except:
        pass  # Column already exists
    
    # Set admin status for configured emails
    for email in ADMIN_EMAILS:
        conn.execute("UPDATE users SET is_admin = TRUE WHERE email = ?", (email,))
    
    conn.commit()
    conn.close()


def make_admin(user_id: str):
    """Grant admin privileges to a user"""
    conn = get_db_connection()
    conn.execute("UPDATE users SET is_admin = TRUE WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def revoke_admin(user_id: str):
    """Revoke admin privileges from a user"""
    conn = get_db_connection()
    conn.execute("UPDATE users SET is_admin = FALSE WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def get_admin_stats() -> dict:
    """Get overall platform statistics for admins"""
    conn = get_db_connection()
    
    # User statistics
    total_users = conn.execute("SELECT COUNT(*) as count FROM users").fetchone()["count"]
    active_users = conn.execute("""
        SELECT COUNT(DISTINCT session_id) as count 
        FROM messages 
        WHERE timestamp > ?
    """, (int(time.time()) - (24 * 60 * 60),)).fetchone()["count"]
    
    # Subscription breakdown
    subs = conn.execute("""
        SELECT subscription_tier, COUNT(*) as count 
        FROM users 
        GROUP BY subscription_tier
    """).fetchall()
    
    subscription_breakdown = {row["subscription_tier"]: row["count"] for row in subs}
    
    # Revenue calculation (approximate)
    pricing = {
        "starter": 9,
        "pro": 29,
        "enterprise": 99
    }
    
    monthly_revenue = sum(
        subscription_breakdown.get(tier, 0) * price 
        for tier, price in pricing.items()
    )
    
    # Message statistics
    total_messages = conn.execute("SELECT COUNT(*) as count FROM messages").fetchone()["count"]
    today_messages = conn.execute("""
        SELECT COUNT(*) as count 
        FROM messages 
        WHERE timestamp > ?
    """, (int(time.time()) - (24 * 60 * 60),)).fetchone()["count"]
    
    # Model usage
    model_usage = conn.execute("""
        SELECT model, COUNT(*) as count 
        FROM messages 
        WHERE role = 'assistant'
        GROUP BY model 
        ORDER BY count DESC 
        LIMIT 10
    """).fetchall()
    
    conn.close()
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "subscription_breakdown": subscription_breakdown,
        "monthly_revenue": monthly_revenue,
        "total_messages": total_messages,
        "today_messages": today_messages,
        "model_usage": {row["model"]: row["count"] for row in model_usage}
    }


def get_user_details(user_id: str) -> dict:
    """Get detailed information about a specific user"""
    conn = get_db_connection()
    
    # Basic user info
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    
    if not user:
        conn.close()
        return None
    
    # Usage statistics
    usage = conn.execute("""
        SELECT 
            COUNT(*) as total_messages,
            COUNT(DISTINCT date(timestamp)) as active_days
        FROM messages 
        WHERE session_id = ?
    """, (user_id,)).fetchone()
    
    # Recent activity
    recent_messages = conn.execute("""
        SELECT content, model, timestamp 
        FROM messages 
        WHERE session_id = ? AND role = 'user'
        ORDER BY timestamp DESC 
        LIMIT 5
    """, (user_id,)).fetchall()
    
    # API usage (if any)
    api_usage = conn.execute("""
        SELECT COUNT(*) as count, SUM(cost_cents) as total_cost
        FROM api_usage 
        WHERE user_id = ?
    """, (user_id,)).fetchone()
    
    conn.close()
    
    return {
        "user": dict(user),
        "total_messages": usage["total_messages"] or 0,
        "active_days": usage["active_days"] or 0,
        "recent_messages": [dict(row) for row in recent_messages],
        "api_calls": api_usage["count"] or 0,
        "api_cost_cents": api_usage["total_cost"] or 0
    }


# Initialize admin table on import
init_admin_table()