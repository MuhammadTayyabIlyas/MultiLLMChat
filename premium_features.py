"""
Premium features only available to paying subscribers
"""
from __future__ import annotations

import streamlit as st
from auth.users import require_subscription, User
from auth.usage_limits import check_usage_limit
import openai
import requests


class PremiumFeatures:
    """Collection of premium features for paying users"""
    
    @staticmethod
    @require_subscription("pro")
    def advanced_prompt_library():
        """Professional prompt templates and library"""
        st.markdown("### 🎯 Advanced Prompt Library")
        
        categories = {
            "Business": [
                ("Market Analysis", "Analyze the current market trends for [industry] including opportunities, threats, and competitive landscape."),
                ("SWOT Analysis", "Create a comprehensive SWOT analysis for [company/product] considering internal and external factors."),
                ("Business Plan", "Write a detailed business plan for [business idea] including executive summary, market analysis, and financial projections."),
            ],
            "Development": [
                ("Code Review", "Review this code for best practices, potential bugs, and optimization opportunities:\n\n[CODE]"),
                ("Architecture Design", "Design a scalable system architecture for [application] with [requirements]. Consider microservices, databases, and caching."),
                ("Debugging", "Help me debug this error: [ERROR_MESSAGE]. Analyze the stack trace and suggest potential causes and solutions."),
            ],
            "Creative": [
                ("Content Creation", "Create engaging [content_type] about [topic] for [audience] with a [tone] tone. Include relevant examples and actionable insights."),
                ("Story Writing", "Write a [genre] story about [theme] with compelling characters and a unique plot twist."),
                ("Marketing Copy", "Write persuasive marketing copy for [product/service] that highlights unique benefits and includes a strong call-to-action."),
            ]
        }
        
        category = st.selectbox("Choose Category", list(categories.keys()))
        
        if category:
            prompts = categories[category]
            prompt_names = [p[0] for p in prompts]
            selected = st.selectbox("Select Template", prompt_names)
            
            selected_prompt = next(p[1] for p in prompts if p[0] == selected)
            
            st.text_area("Template", selected_prompt, height=150)
            
            # Customization
            st.markdown("#### Customize Template")
            custom_vars = {}
            
            # Auto-detect variables in brackets
            import re
            variables = re.findall(r'\[(.*?)\]', selected_prompt)
            
            for var in variables:
                custom_vars[var] = st.text_input(f"Enter {var.replace('_', ' ').title()}")
            
            if st.button("Use This Template"):
                customized = selected_prompt
                for var, value in custom_vars.items():
                    customized = customized.replace(f"[{var}]", value)
                
                st.session_state.message_input = customized
                st.success("Template copied to chat input!")
    
    @staticmethod
    @require_subscription("pro")
    def analytics_dashboard():
        """Usage analytics and insights dashboard"""
        st.markdown("### 📈 Analytics Dashboard")
        
        from auth.usage_limits import usage_tracker
        user = st.session_state.user
        
        # Time range selector
        time_range = st.selectbox("Time Range", ["7 days", "30 days", "90 days"])
        days = int(time_range.split()[0])
        
        analytics = usage_tracker.get_usage_analytics(user.id, days=days)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Messages", analytics["total_messages"])
        
        with col2:
            st.metric("Total Comparisons", analytics["total_comparisons"])
        
        with col3:
            if user.subscription_tier in ["pro", "enterprise"]:
                st.metric("API Calls", analytics["total_api_calls"])
        
        # Daily usage chart
        if analytics["daily_usage"]:
            import pandas as pd
            
            df = pd.DataFrame(analytics["daily_usage"])
            df["date"] = pd.to_datetime(df["date"])
            
            st.markdown("#### Daily Usage Trends")
            
            # Create line chart
            chart_data = df[["date", "messages", "comparisons"]].copy()
            st.line_chart(
                chart_data.set_index("date"),
                use_container_width=True
            )
        
        # Model preferences
        st.markdown("#### Your Most Used Models")
        
        # Get model usage from message history
        conn = get_db_connection()
        rows = conn.execute("""
            SELECT model, COUNT(*) as count 
            FROM messages 
            WHERE session_id = ? AND role = 'assistant'
            GROUP BY model 
            ORDER BY count DESC 
            LIMIT 5
        """, (user.id,)).fetchall()
        conn.close()
        
        if rows:
            import pandas as pd
            model_df = pd.DataFrame(rows)
            st.bar_chart(model_df.set_index("model"))
    
    @staticmethod
    @require_subscription("pro")
    def team_sharing():
        """Team collaboration and sharing features"""
        st.markdown("### 👥 Team Sharing")
        
        user = st.session_state.user
        
        # Get team members
        conn = get_db_connection()
        team_members = conn.execute("""
            SELECT id, email, subscription_tier FROM users 
            WHERE created_by = ? 
            ORDER BY created_at DESC
        """, (user.id,)).fetchall()
        conn.close()
        
        # Invite new member
        with st.expander("➕ Invite Team Member", expanded=False):
            email = st.text_input("Team member email")
            role = st.selectbox("Role", ["member", "admin"])
            
            if st.button("Send Invitation"):
                self._send_team_invite(user, email, role)
        
        # Show team members
        if team_members:
            st.markdown("#### Current Team Members")
            for member in team_members:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(member["email"])
                with col2:
                    st.caption(f"Tier: {member['subscription_tier']}")
                with col3:
                    if st.button("Remove", key=f"remove_{member['id']}"):
                        self._remove_team_member(user, member['id'])
        else:
            st.info("No team members yet. Invite your first member above!")
    
    @staticmethod
    def _send_team_invite(owner: User, email: str, role: str):
        """Send team invitation"""
        # Generate invite token
        import secrets
        token = secrets.token_urlsafe(32)
        
        # Store invitation
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO team_invites (token, owner_id, email, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (token, owner.id, email, role, int(time.time())))
        conn.commit()
        conn.close()
        
        # TODO: Send email with invite link
        # For now, show the link
        invite_link = f"{st.runtime.get_instance().browser.server_url}?invite={token}"
        st.success(f"Invitation link: {invite_link}")
    
    @staticmethod
    def _remove_team_member(owner: User, member_id: str):
        """Remove team member"""
        conn = get_db_connection()
        conn.execute("DELETE FROM users WHERE id = ? AND created_by = ?", (member_id, owner.id))
        conn.commit()
        conn.close()
        st.success("Team member removed")
        st.rerun()
    
    @staticmethod
    @require_subscription("pro")
    def custom_model_integration():
        """Add custom/private model endpoints"""
        st.markdown("### 🔧 Custom Model Integration")
        
        user = st.session_state.user
        
        with st.expander("➕ Add Custom Model", expanded=False):
            model_name = st.text_input("Model Display Name")
            api_endpoint = st.text_input("API Endpoint URL")
            api_key = st.text_input("API Key", type="password")
            model_type = st.selectbox("Model Type", ["OpenAI Compatible", "Anthropic Compatible", "Custom"])
            
            if st.button("Add Custom Model"):
                if not all([model_name, api_endpoint, api_key]):
                    st.error("Please fill all fields")
                else:
                    self._add_custom_model(user, {
                        "name": model_name,
                        "endpoint": api_endpoint,
                        "api_key": api_key,
                        "type": model_type
                    })
        
        # Show custom models
        st.markdown("#### Your Custom Models")
        custom_models = self._get_custom_models(user.id)
        
        if custom_models:
            for model in custom_models:
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(model["name"])
                    with col2:
                        st.caption(model["endpoint"])
                    with col3:
                        if st.button("Remove", key=f"del_model_{model['id']}"):
                            self._delete_custom_model(model['id'])
        else:
            st.info("No custom models configured yet")
    
    @staticmethod
    def _add_custom_model(user: User, model_config: dict):
        """Add custom model to user account"""
        conn = get_db_connection()
        import json
        conn.execute("""
            INSERT INTO custom_models (user_id, name, config, created_at)
            VALUES (?, ?, ?, ?)
        """, (user.id, model_config["name"], json.dumps(model_config), int(time.time())))
        conn.commit()
        conn.close()
        st.success(f"Custom model '{model_config['name']}' added successfully!")
    
    @staticmethod
    def _get_custom_models(user_id: str):
        """Get user's custom models"""
        conn = get_db_connection()
        models = conn.execute("""
            SELECT id, name, config FROM custom_models WHERE user_id = ?
        """, (user_id,)).fetchall()
        conn.close()
        return models
    
    @staticmethod
    def _delete_custom_model(model_id: str):
        """Delete custom model"""
        conn = get_db_connection()
        conn.execute("DELETE FROM custom_models WHERE id = ?", (model_id,))
        conn.commit()
        conn.close()
        st.success("Custom model removed")
        st.rerun()
    
    @staticmethod
    @require_subscription("pro")
    def priority_support():
        """Priority support widget for Pro+ users"""
        st.markdown("### 🚀 Priority Support")
        
        st.info("✅ You have access to priority support!")
        
        issue_type = st.selectbox("Issue Type", [
            "Technical Problem",
            "Feature Request",
            "Billing Question",
            "Model Integration Help",
            "Other"
        ])
        
        subject = st.text_input("Subject")
        description = st.text_area("Describe your issue", height=150)
        
        if st.button("Submit Support Ticket"):
            if not subject or not description:
                st.error("Please fill in subject and description")
            else:
                # Create support ticket
                ticket_id = self._create_support_ticket(
                    st.session_state.user,
                    issue_type,
                    subject,
                    description
                )
                st.success(f"Support ticket #{ticket_id} submitted! We'll respond within 2 hours.")
    
    @staticmethod
    def _create_support_ticket(user: User, issue_type: str, subject: str, description: str) -> str:
        """Create support ticket in database"""
        import secrets
        ticket_id = f"TKT-{secrets.token_hex(8).upper()}"
        
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO support_tickets (ticket_id, user_id, type, subject, description, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'open', ?)
        """, (ticket_id, user.id, issue_type, subject, description, int(time.time())))
        conn.commit()
        conn.close()
        
        # TODO: Send email notification to support team
        return ticket_id
    
    @staticmethod
    @require_subscription("pro")
    def batch_processing():
        """Process multiple prompts at once"""
        st.markdown("### 📄 Batch Processing")
        
        st.info("Process multiple prompts simultaneously")
        
        input_method = st.radio("Input Method", ["Text Area (one per line)", "Upload CSV", "Upload TXT"])
        
        prompts = []
        
        if input_method == "Text Area (one per line)":
            text = st.text_area("Enter prompts (one per line)", height=200)
            prompts = [p.strip() for p in text.split('\n') if p.strip()]
        elif input_method == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded:
                import pandas as pd
                df = pd.read_csv(uploaded)
                prompts = df.iloc[:, 0].tolist()
        elif input_method == "Upload TXT":
            uploaded = st.file_uploader("Upload TXT", type=['txt'])
            if uploaded:
                content = uploaded.read().decode()
                prompts = [p.strip() for p in content.split('\n') if p.strip()]
        
        if prompts:
            st.info(f"📊 {len(prompts)} prompts loaded")
            
            # Model selection for batch
            model = st.selectbox("Model for batch processing", [
                "OpenAI · GPT-4o",
                "Anthropic · Claude Sonnet",
                "Google · Gemini",
                "Mixed (Best of each)"
            ])
            
            if st.button("Start Batch Processing"):
                if len(prompts) > 50 and st.session_state.user.subscription_tier != "enterprise":
                    st.error("Batch processing limited to 50 prompts on your tier. Upgrade to Enterprise for unlimited.")
                else:
                    self._process_batch(prompts, model)
    
    @staticmethod
    def _process_batch(prompts: list, model: str):
        """Process batch of prompts"""
        import time
        
        st.markdown("---")
        st.markdown("### Batch Results")
        
        progress_bar = st.progress(0)
        results = []
        
        for i, prompt in enumerate(prompts):
            # Simulate processing with a delay
            with st.spinner(f"Processing {i+1}/{len(prompts)}..."):
                # Here you would call the actual model
                # result = call_model(model, prompt)
                time.sleep(1)  # Simulate API call
                result = f"Processed: {prompt[:100]}..."
                
                results.append({
                    "prompt": prompt,
                    "result": result
                })
                
                progress_bar.progress((i + 1) / len(prompts))
        
        # Show results
        for idx, result in enumerate(results):
            with st.expander(f"Result #{idx + 1}", expanded=False):
                st.markdown(f"**Prompt:** {result['prompt']}")
                st.markdown(f"**Result:** {result['result']}")
        
        # Download results
        if st.button("Download Results as CSV"):
            import pandas as pd
            import io
            
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False)
            
            st.download_button(
                "📥 Download CSV",
                csv,
                "batch_results.csv",
                "text/csv"
            )


# Render premium section in sidebar
def render_premium_sidebar():
    """Render premium features in sidebar"""
    if "user" not in st.session_state:
        return
    
    user = st.session_state.user
    
    if user.subscription_tier == "free":
        with st.expander("🚀 Unlock Premium Features"):
            st.markdown("### Upgrade to Pro!")
            st.write("✓ Unlimited messages")
            st.write("✓ Premium models")
            st.write("✓ Advanced analytics")
            st.write("✓ API access")
            if st.button("View Plans", use_container_width=True):
                st.switch_page("pages/billing.py")
    
    else:
        st.markdown("---")
        st.markdown(f"### ⭐ {user.subscription_tier.title()} Features")
        
        if check_subscription_requirement("pro"):
            if st.button("📊 Analytics", use_container_width=True):
                PremiumFeatures.analytics_dashboard()
            
            if st.button("🎯 Prompt Library", use_container_width=True):
                PremiumFeatures.advanced_prompt_library()
            
            if st.button("📄 Batch Processing", use_container_width=True):
                PremiumFeatures.batch_processing()
        
        if check_subscription_requirement("enterprise"):
            if st.button("👥 Team Settings", use_container_width=True):
                PremiumFeatures.team_sharing()
            
            if st.button("🔧 Custom Models", use_container_width=True):
                PremiumFeatures.custom_model_integration()


def check_subscription_requirement(required_tier: str) -> bool:
    """Check if current user meets subscription requirement"""
    if "user" not in st.session_state:
        return False
    
    user = st.session_state.user
    tier_order = ["free", "starter", "pro", "enterprise"]
    user_tier_idx = tier_order.index(user.subscription_tier)
    required_tier_idx = tier_order.index(required_tier)
    
    return user_tier_idx >= required_tier_idx