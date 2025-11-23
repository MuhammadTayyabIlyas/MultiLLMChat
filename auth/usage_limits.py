"""
Usage tracking and rate limiting for subscription tiers
"""
from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional
from functools import wraps

import streamlit as st

from auth.users import user_manager, User
from backend import RateLimitError

logger = logging.getLogger(__name__)


class UsageTracker:
    """Track and enforce usage limits based on subscription tiers"""
    
    def __init__(self):
        self.init_tables()
    
    def init_tables(self):
        """Create usage tracking tables"""
        conn = get_db_connection()
        
        # Daily usage tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_usage (
                user_id TEXT,
                date TEXT,
                message_count INTEGER DEFAULT 0,
                comparison_count INTEGER DEFAULT 0,
                api_call_count INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, date)
            )
        """)
        
        # API usage tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                endpoint TEXT,
                timestamp INTEGER,
                tokens_used INTEGER,
                cost_cents REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def check_limits(self, user: User, action: str) -> tuple[bool, str]:
        """
        Check if user can perform an action
        
        Args:
            user: The user attempting the action
            action: Type of action ('message', 'comparison', 'api_call')
        
        Returns:
            tuple of (allowed: bool, reason: str)
        """
        if not user:
            return False, "User not authenticated"
        
        limits = user_manager.get_tier_limits(user.subscription_tier)
        
        if action == 'message':
            if limits["daily_messages"] == float('inf'):
                return True, ""
            
            count = self.get_daily_count(user.id, 'message')
            if count >= limits["daily_messages"]:
                return False, f"Daily message limit reached ({limits['daily_messages']}/day)"
        
        elif action == 'comparison':
            if limits["daily_comparisons"] == float('inf'):
                return True, ""
            
            count = self.get_daily_count(user.id, 'comparison')
            if count >= limits["daily_comparisons"]:
                return False, f"Daily comparison limit reached ({limits['daily_comparisons']}/day)"
        
        elif action == 'api_call':
            # API limits are monthly, not daily
            if limits["api_calls"] == 0:
                return False, "API access not available on your tier"
            
            if limits["api_calls"] == float('inf'):
                return True, ""
            
            count = self.get_monthly_api_count(user.id)
            if count >= limits["api_calls"]:
                return False, f"Monthly API call limit reached ({limits['api_calls']}/month)"
        
        return True, ""
    
    def record_usage(self, user: User, action: str, tokens: int = 0, cost_cents: float = 0.0):
        """Record usage for a user"""
        if not user:
            return
        
        today = time.strftime("%Y-%m-%d")
        
        conn = get_db_connection()
        
        if action in ['message', 'comparison']:
            # Update daily usage counter
            field = 'message_count' if action == 'message' else 'comparison_count'
            
            conn.execute(f"""
                INSERT INTO daily_usage (user_id, date, {field})
                VALUES (?, ?, 1)
                ON CONFLICT(user_id, date) DO UPDATE SET
                {field} = {field} + 1
            """, (user.id, today))
        
        elif action == 'api_call':
            # Record API call with details
            conn.execute("""
                INSERT INTO api_usage (user_id, endpoint, timestamp, tokens_used, cost_cents)
                VALUES (?, ?, ?, ?, ?)
            """, (user.id, st.session_state.get('api_endpoint', 'unknown'), 
                  int(time.time()), tokens, cost_cents))
            
            # Also increment daily counter
            conn.execute("""
                INSERT INTO daily_usage (user_id, date, api_call_count)
                VALUES (?, ?, 1)
                ON CONFLICT(user_id, date) DO UPDATE SET
                api_call_count = api_call_count + 1
            """, (user.id, today))
        
        conn.commit()
        conn.close()
    
    def get_daily_count(self, user_id: str, action: str) -> int:
        """Get daily count for an action"""
        today = time.strftime("%Y-%m-%d")
        
        conn = get_db_connection()
        
        if action == 'message':
            row = conn.execute("""
                SELECT message_count FROM daily_usage 
                WHERE user_id = ? AND date = ?
            """, (user_id, today)).fetchone()
        elif action == 'comparison':
            row = conn.execute("""
                SELECT comparison_count FROM daily_usage 
                WHERE user_id = ? AND date = ?
            """, (user_id, today)).fetchone()
        elif action == 'api_call':
            row = conn.execute("""
                SELECT api_call_count FROM daily_usage 
                WHERE user_id = ? AND date = ?
            """, (user_id, today)).fetchone()
        
        conn.close()
        
        return row[0] if row and row[0] else 0
    
    def get_monthly_api_count(self, user_id: str) -> int:
        """Get API call count for current month"""
        import calendar
        now = datetime.now()
        month_start = datetime(now.year, now.month, 1)
        month_end = datetime(now.year, now.month, calendar.monthrange(now.year, now.month)[1], 23, 59, 59)
        
        start_timestamp = int(month_start.timestamp())
        end_timestamp = int(month_end.timestamp())
        
        conn = get_db_connection()
        row = conn.execute("""
            SELECT COUNT(*) as count FROM api_usage 
            WHERE user_id = ? AND timestamp BETWEEN ? AND ?
        """, (user_id, start_timestamp, end_timestamp)).fetchone()
        conn.close()
        
        return row['count'] if row else 0
    
    def get_usage_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage analytics for a user"""
        conn = get_db_connection()
        
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Get daily usage data
        rows = conn.execute("""
            SELECT date, message_count, comparison_count, api_call_count
            FROM daily_usage
            WHERE user_id = ? AND date >= ?
            ORDER BY date
        """, (user_id, start_date)).fetchall()
        
        analytics = {
            "daily_usage": [],
            "total_messages": 0,
            "total_comparisons": 0,
            "total_api_calls": 0
        }
        
        for row in rows:
            analytics["daily_usage"].append({
                "date": row["date"],
                "messages": row["message_count"] or 0,
                "comparisons": row["comparison_count"] or 0,
                "api_calls": row["api_call_count"] or 0
            })
            analytics["total_messages"] += row["message_count"] or 0
            analytics["total_comparisons"] += row["comparison_count"] or 0
            analytics["total_api_calls"] += row["api_call_count"] or 0
        
        conn.close()
        return analytics


# Global usage tracker
usage_tracker = UsageTracker()


def require_subscription(min_tier: str = "free"):
    """
    Decorator to require minimum subscription tier
    
    Usage:
        @require_subscription("pro")
        def premium_feature():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if "user" not in st.session_state:
                st.error("⚠️ Please log in to access this feature.")
                st.stop()
            
            user = st.session_state.user
            
            tier_order = ["free", "starter", "pro", "enterprise"]
            user_tier_idx = tier_order.index(user.subscription_tier)
            required_tier_idx = tier_order.index(min_tier)
            
            if user_tier_idx < required_tier_idx:
                st.error(f"⚠️ This feature requires a {min_tier.title()} subscription or higher.")
                st.info("💳 Upgrade your plan to access premium features!")
                if st.button("View Plans"):
                    st.switch_page("pages/billing.py")  # Redirect to billing page
                st.stop()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def check_usage_limit(action: str) -> bool:
    """Check if user has remaining quota"""
    if "user" not in st.session_state:
        # For demo/testing without auth
        return True
    
    user = st.session_state.user
    allowed, reason = usage_tracker.check_limits(user, action)
    
    if not allowed:
        st.error(f"⚠️ {reason}")
        
        # Show upgrade button
        if user.subscription_tier == "free":
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info("💡 Upgrade to Starter for $9/month to increase your limits!")
            with col2:
                if st.button("Upgrade Now"):
                    st.switch_page("pages/billing.py")
        
        st.stop()
    
    return True


def record_user_usage(action: str, tokens: int = 0):
    """Record usage for current user"""
    if "user" in st.session_state:
        usage_tracker.record_usage(st.session_state.user, action, tokens)


def show_usage_widget():
    """Display usage widget in sidebar"""
    if "user" not in st.session_state:
        return
    
    user = st.session_state.user
    usage = usage_tracker.get_usage_analytics(user.id, days=7)
    limits = user_manager.get_tier_limits(user.subscription_tier)
    
    with st.expander("📊 Usage Stats", expanded=False):
        # Show today's usage
        today = time.strftime("%Y-%m-%d")
        today_stats = next((d for d in usage["daily_usage"] if d["date"] == today), None)
        
        if today_stats:
            if limits["daily_messages"] != float('inf'):
                st.progress(
                    min(today_stats["messages"] / limits["daily_messages"], 1.0),
                    f"Messages: {today_stats['messages']} / {limits['daily_messages']}"
                )
            
            if limits["daily_comparisons"] != float('inf'):
                st.progress(
                    min(today_stats["comparisons"] / limits["daily_comparisons"], 1.0),
                    f"Comparisons: {today_stats['comparisons']} / {limits['daily_comparisons']}"
                )
        
        # Show 7-day total
        st.markdown(f"**7-Day Total:**")
        st.write(f"Messages: {usage['total_messages']}")
        st.write(f"Comparisons: {usage['total_comparisons']}")
        
        if user.subscription_tier in ["pro", "enterprise"]:
            st.write(f"API Calls: {usage['total_api_calls']}")