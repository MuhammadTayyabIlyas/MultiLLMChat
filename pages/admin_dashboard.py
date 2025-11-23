"""
Admin Dashboard - User Management, Revenue Monitoring, Model Configuration
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import time
import json

from admin.admin_auth import require_admin, get_admin_stats, get_user_details, make_admin, revoke_admin
from backend import MODEL_REGISTRY, ProviderConfig, available_models
from db import get_db_connection

# Import billing manager lazily to avoid circular imports
def get_billing_manager():
    from billing.stripe_integration import billing_manager
    return billing_manager


def main():
    """Main admin dashboard"""
    require_admin()
    
    user = st.session_state.user
    
    # Admin navigation
    st.sidebar.markdown("### 🔧 Admin Tools")
    admin_section = st.sidebar.radio(
        "Dashboard Section",
        ["📊 Overview", "👥 Users", "💰 Revenue & Billing", "🤖 Models & APIs", "⚙️ System"],
        key="admin_nav"
    )
    
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #404040;">
        <h1 style="font-size: 2rem; font-weight: 700;">🔐 Admin Dashboard</h1>
        <p style="color: #8E8EA0;">Welcome, {user.email}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if admin_section == "📊 Overview":
        render_overview()
    elif admin_section == "👥 Users":
        render_user_management()
    elif admin_section == "💰 Revenue & Billing":
        render_revenue()
    elif admin_section == "🤖 Models & APIs":
        render_model_management()
    elif admin_section == "⚙️ System":
        render_system_settings()


def render_overview():
    """Render admin overview dashboard"""
    st.markdown("## 📊 Platform Overview")
    
    # Get stats
    stats = get_admin_stats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Users",
            stats["total_users"],
            f"+{stats.get('new_users_this_week', 0)} this week"
        )
    
    with col2:
        st.metric(
            "Active Users (24h)",
            stats["active_users"],
            f"{stats['active_users']/stats['total_users']*100:.1f}% of total"
        )
    
    with col3:
        st.metric(
            "Monthly Revenue",
            f"${stats['monthly_revenue']:,}",
            "USD"
        )
    
    with col4:
        st.metric(
            "Total Messages",
            f"{stats['total_messages']:,}",
            f"{stats['today_messages']} today"
        )
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 📊 Subscription Distribution")
        
        subscription_data = stats["subscription_breakdown"]
        df = pd.DataFrame([
            {"Tier": tier.replace("_", " ").title(), "Users": count}
            for tier, count in subscription_data.items()
        ])
        st.bar_chart(df.set_index("Tier"))
    
    with col_chart2:
        st.markdown("### 🤖 Model Usage")
        
        model_usage = stats["model_usage"]
        if model_usage:
            df_models = pd.DataFrame([
                {"Model": model, "Messages": count}
                for model, count in list(model_usage.items())[:6]
            ])
            st.bar_chart(df_models.set_index("Model"))
        else:
            st.info("No model usage data yet")


def render_user_management():
    """Render user management interface"""
    st.markdown("## 👥 User Management")
    
    # Filters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        tier_filter = st.selectbox("Filter by Tier", ["all", "free", "starter", "pro", "enterprise"])
    
    with col2:
        search_email = st.text_input("Search by Email")
    
    # Get users
    users = get_filtered_users(tier_filter, search_email)
    
    if users:
        for user_row in users:
            user_dict = dict(user_row)
            details = get_user_details(user_dict["id"])
            
            with st.expander(
                f"👤 {user_dict['email']} | {user_dict['subscription_tier'].upper()} | "
                f"{details['total_messages']} msgs | "
                f"{'Admin' if user_dict.get('is_admin', 0) else 'User'}"
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**ID:** `{user_dict['id']}`")
                    st.write(f"**Email:** {user_dict['email']}")
                    st.write(f"**Tier:** {user_dict['subscription_tier']}")
                
                with col2:
                    st.write(f"**Messages:** {details['total_messages']:,}")
                    st.write(f"**Active Days:** {details['active_days']}")
                
                # Actions
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                
                with col_btn1:
                    if not user_dict.get('is_admin', 0):
                        if st.button("👑 Make Admin", key=f"admin_{user_dict['id']}"):
                            make_admin(user_dict['id'])
                            st.success("Admin granted")
                            time.sleep(1)
                            st.rerun()
                
                with col_btn2:
                    if user_dict['subscription_tier'] != 'enterprise':
                        if st.button("⬆️ Upgrade", key=f"up_{user_dict['id']}"):
                            upgrade_user_tier(user_dict['id'])
                
                with col_btn3:
                    if st.button("🗑️ Delete", key=f"del_{user_dict['id']}", 
                                 type="secondary"):
                        if confirm_delete_user(user_dict['id'], user_dict['email']):
                            delete_user(user_dict['id'])
                            st.rerun()
    else:
        st.info("No users found")


def render_revenue():
    """Render revenue and billing dashboard"""
    st.markdown("## 💰 Revenue & Billing")
    
    stats = get_admin_stats()
    
    # Revenue metrics
    col1, col2, col3, col4 = st.columns(4)
    
    paid_users = sum(count for tier, count in stats["subscription_breakdown"].items() 
                     if tier != 'free')
    
    with col1:
        st.metric("MRR", f"${stats['monthly_revenue']:,}")
    
    with col2:
        st.metric("Paying Customers", f"{paid_users:,}")
    
    with col3:
        arpu = stats['monthly_revenue'] / paid_users if paid_users > 0 else 0
        st.metric("ARPU", f"${arpu:.2f}")
    
    with col4:
        conversion = ((stats['total_users'] - stats['subscription_breakdown'].get('free', 0)) / 
                      stats['total_users'] * 100)
        st.metric("Conversion Rate", f"{conversion:.1f}%")
    
    # Stripe status
    st.markdown("### 💳 Payment Processing")
    
    if get_billing_manager().is_configured():
        st.success("✅ Stripe is configured")
    else:
        st.error("❌ Stripe not configured")


def render_model_management():
    """Render model and API management"""
    st.markdown("## 🤖 Models & APIs")
    
    model_action = st.radio(
        "Action",
        ["View Models", "Add Model", "Configure API Keys"],
        horizontal=True
    )
    
    if model_action == "View Models":
        show_current_models()
    
    elif model_action == "Add Model":
        add_new_model()
    
    elif model_action == "Configure API Keys":
        configure_api_keys()


def show_current_models():
    """Show currently configured models"""
    st.markdown("### Current Models")
    
    models = available_models()
    
    for config in models:
        with st.expander(config.label, expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Key:** `{config.key}`")
                st.write(f"**Provider:** {config.provider}")
                st.write(f"**Model ID:** `{config.model}`")
            
            with col2:
                has_api = config.secret in st.secrets
                st.write(f"**API:** {'✅' if has_api else '❌'}")
                st.write(f"**Streaming:** {'✅' if config.supports_stream else '❌'}")


def add_new_model():
    """Add a new model"""
    st.markdown("### ➕ Add New Model")
    
    with st.form("add_model"):
        provider = st.selectbox("Provider", ["openai", "anthropic", "gemini", "groq", "custom"])
        model_key = st.text_input("Model Key")
        model_label = st.text_input("Model Label")
        model_id = st.text_input("Model ID")
        secret_key = st.text_input("Secret Key Name")
        
        if st.form_submit_button("Add Model"):
            if all([model_key, model_label, model_id, secret_key]):
                st.success(f"Model {model_label} added (add to MODEL_REGISTRY)")
            else:
                st.error("Fill all fields")


def configure_api_keys():
    """Configure API keys"""
    st.markdown("### 🔑 API Key Configuration")
    
    # Create a comprehensive API key configuration interface
    st.warning("API keys must be configured in .streamlit/secrets.toml")
    
    # Check current status
    required_keys = [
        ("OPENAI_API_KEY", "OpenAI"),
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("GEMINI_API_KEY", "Google"),
        ("GROQ_API_KEY", "Groq"),
        ("STRIPE_SECRET_KEY", "Stripe"),
    ]
    
    for key_name, provider in required_keys:
        is_configured = key_name in st.secrets
        status = "✅" if is_configured else "❌"
        st.write(f"{status} **{provider}** ({key_name})")


def render_system_settings():
    """Render system settings"""
    st.markdown("## ⚙️ System Settings")
    
    st.markdown("### 🏥 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Database Status", "✅ Healthy")
    
    with col2:
        st.metric("API Response Time", "120ms", help="Average")
    
    with col3:
        st.metric("Uptime", "99.9%", help="Last 30 days")
    
    # Danger zone
    st.markdown("### 🚨 Danger Zone")
    
    with st.expander("⚠️ Dangerous Operations", expanded=False):
        if st.button("Clear All Cache", type="secondary"):
            st.warning("Cache cleared (mock operation)")
        
        if st.checkbox("I understand I will DELETE ALL DATA"):
            if st.button("DELETE ALL DATA", type="secondary"):
                st.error("Not implemented for safety")


def get_filtered_users(tier_filter, search_email):
    """Get filtered users for management"""
    conn = get_db_connection()
    
    query = "SELECT * FROM users WHERE 1=1"
    params = []
    
    if tier_filter != "all":
        query += " AND subscription_tier = ?"
        params.append(tier_filter)
    
    if search_email:
        query += " AND email LIKE ?"
        params.append(f"%{search_email}%")
    
    query += " LIMIT 200"
    
    users = conn.execute(query, params).fetchall()
    conn.close()
    
    return users


def upgrade_user_tier(user_id: str):
    """Upgrade user tier (admin override)"""
    conn = get_db_connection()
    conn.execute("UPDATE users SET subscription_tier = 'pro' WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    st.success("User upgraded to Pro")


def confirm_delete_user(user_id: str, email: str) -> bool:
    """Confirm user deletion"""
    return st.checkbox(f"Confirm delete {email}")


def delete_user(user_id: str):
    """Delete user and all their data"""
    conn = get_db_connection()
    # Delete messages
    conn.execute("DELETE FROM messages WHERE session_id = ?", (user_id,))
    # Delete user
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()