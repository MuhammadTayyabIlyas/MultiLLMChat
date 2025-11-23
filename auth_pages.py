"""
Authentication pages - Login, Signup, Password Reset
"""
from __future__ import annotations

import streamlit as st
from auth.users import user_manager, User
from auth.usage_limits import usage_tracker
import re


def render_login_page():
    """Render login page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🤖</h1>
        <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">AI Chat Studio</h1>
        <p style="color: #8E8EA0; font-size: 1.1rem; margin-bottom: 2rem;">Access 11+ AI models in one unified platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Switch between login and signup
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Mode selector
        login_tab, signup_tab = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="your@email.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                
                submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
                
                if submitted:
                    if not email or not password:
                        st.error("Please enter both email and password")
                    else:
                        user = user_manager.authenticate(email, password)
                        if user:
                            st.session_state.user = user
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid email or password")
            
            # Password reset
            st.markdown("---")
            if st.button("Forgot Password?", type="secondary"):
                st.session_state.auth_mode = "reset"
                st.rerun()
        
        with signup_tab:
            with st.form("signup_form"):
                name = st.text_input("Full Name", placeholder="John Doe")
                email = st.text_input("Email", placeholder="your@email.com")
                password = st.text_input("Password", type="password", placeholder="Min. 8 characters")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
                
                if submitted:
                    # Validate inputs
                    if not all([name, email, password, confirm_password]):
                        st.error("Please fill in all fields")
                    elif not self._is_valid_email(email):
                        st.error("Please enter a valid email address")
                    elif len(password) < 8:
                        st.error("Password must be at least 8 characters")
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        # Check if email already exists
                        existing = user_manager.get_user_by_email(email)
                        if existing:
                            st.error("Email already registered")
                        else:
                            try:
                                user = user_manager.create_user(email, password)
                                st.session_state.user = user
                                
                                # Send welcome email (TODO: implement email)
                                st.success("✅ Account created successfully! Welcome to AI Chat Studio!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to create account: {str(e)}")
        
        # Features preview
        st.markdown("---")
        st.markdown("#### Why Join AI Chat Studio?")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("✅ **Access 11+ AI Models**")
            st.markdown("✅ **Compare Responses**")
            st.markdown("✅ **Free Tier Available**")
        with col_f2:
            st.markdown("✅ **Chat History**")
            st.markdown("✅ **Export Conversations**")
            st.markdown("✅ **Secure & Private**")


def render_reset_password_page():
    """Render password reset page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">🔐 Reset Password</h1>
        <p style="color: #8E8EA0; font-size: 1rem;">Enter your email to receive a reset link</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("reset_form"):
            email = st.text_input("Email", placeholder="your@email.com")
            submitted = st.form_submit_button("Send Reset Link", type="primary", use_container_width=True)
            
            if submitted:
                if not email:
                    st.error("Please enter your email address")
                elif not user_manager.get_user_by_email(email):
                    st.error("Email not found. Please check and try again.")
                else:
                    # Generate and send reset token
                    token = user_manager.generate_reset_token(email)
                    if token:
                        # TODO: Send actual email
                        reset_link = f"{st.runtime.get_instance().browser.server_url}?reset_token={token}"
                        st.success("✅ Password reset link sent to your email!")
                        st.info(f"Reset link: {reset_link}")  # For demo purposes
                    else:
                        st.error("Failed to generate reset token")
        
        st.markdown("---")
        if st.button("← Back to Login", type="secondary", use_container_width=True):
            st.session_state.auth_mode = "login"
            st.rerun()


def render_new_password_page(token: str):
    """Render new password setting page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">🔐 Set New Password</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("new_password_form"):
            new_password = st.text_input("New Password", type="password", placeholder="Min. 8 characters")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            submitted = st.form_submit_button("Set New Password", type="primary", use_container_width=True)
            
            if submitted:
                if len(new_password) < 8:
                    st.error("Password must be at least 8 characters")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    # Validate token and update password
                    if self._validate_reset_token(token):
                        if self._update_password_with_token(token, new_password):
                            st.success("✅ Password updated successfully!")
                            st.info("You can now log in with your new password")
                            time.sleep(2)
                            st.session_state.auth_mode = "login"
                            st.rerun()
                        else:
                            st.error("Failed to update password")
                    else:
                        st.error("Invalid or expired reset token")


def render_logout_page():
    """Render logout confirmation"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">👋 Logout</h1>
        <p style="color: #8E8EA0; font-size: 1rem;">Are you sure you want to logout?</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        col_yes, col_no = st.columns(2)
        
        with col_yes:
            if st.button("Yes, Logout", type="primary", use_container_width=True):
                # Clear session
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ You have been logged out")
                time.sleep(1)
                st.rerun()
        
        with col_no:
            if st.button("Cancel", type="secondary", use_container_width=True):
                st.switch_page("app.py")


class AuthUI:
    """Authentication UI helpers"""
    
    @staticmethod
    def _is_valid_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def require_login():
        """Redirect to login if not authenticated"""
        if "user" not in st.session_state:
            st.warning("Please log in to continue")
            st.switch_page("pages/login.py")
    
    @staticmethod
    def login_widget():
        """Compact login widget for sidebar"""
        if "user" not in st.session_state:
            with st.expander("🔐 Login", expanded=False):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                
                if st.button("Login", use_container_width=True):
                    user = user_manager.authenticate(email, password)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        else:
            user = st.session_state.user
            st.success(f"👤 {user.email}")
            st.caption(f"Tier: {user.subscription_tier}")
            
            if st.button("Logout", type="secondary", use_container_width=True):
                st.switch_page("pages/logout.py")
    
    @staticmethod
    def show_current_user():
        """Show current user info in sidebar"""
        if "user" in st.session_state:
            user = st.session_state.user
            
            with st.container():
                st.markdown(f"**👤 {user.email}**")
                
                # Subscription badge
                tier_colors = {
                    "free": "🟢",
                    "starter": "🟡",
                    "pro": "🔵",
                    "enterprise": "🔴"
                }
                tier_color = tier_colors.get(user.subscription_tier, "⚪")
                st.caption(f"{tier_color} {user.subscription_tier.title()} Tier")
                
                # Usage progress (for free tier)
                if user.subscription_tier == "free":
                    usage = user_manager.get_usage_stats(user.id)
                    limits = user_manager.get_tier_limits("free")
                    
                    progress = usage["messages_today"] / limits["daily_messages"]
                    st.progress(progress, f"Usage: {usage['messages_today']}/{limits['daily_messages']}")


# Page router for different auth pages
def render_auth_page(page: str):
    """Render appropriate auth page based on route"""
    if page == "login":
        render_login_page()
    elif page == "signup":
        render_login_page()  # Login page has signup tab
    elif page == "reset":
        render_reset_password_page()
    elif page == "logout":
        render_logout_page()
    else:
        st.error("Unknown auth page")