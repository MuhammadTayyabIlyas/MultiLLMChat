"""
Billing and subscription management page
"""
from __future__ import annotations

import streamlit as st
from billing.stripe_integration import billing_manager, render_billing_section
from auth.users import user_manager, require_auth, User


def main():
    # Check authentication
    require_auth()
    
    user = st.session_state.user
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">💳 Billing & Plans</h1>
        <p style="color: #8E8EA0; font-size: 1rem;">Manage your subscription and billing preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Handle URL parameters
    if st.query_params.get("upgrade_success"):
        st.success("🎉 Your upgrade was successful! Your new features are now active.")
        # Refresh user data
        st.session_state.user = user_manager.get_user(user.id)
        
    elif st.query_params.get("upgrade_cancelled"):
        st.warning("❌ Upgrade cancelled. You can try again anytime.")
    
    st.divider()
    
    # Main billing section
    render_billing_section(user)
    
    st.divider()
    
    # Plan comparison table
    st.markdown("### 📊 Plan Comparison")
    
    plan_data = {
        "Feature": [
            "Daily Messages",
            "Model Access", 
            "Daily Comparisons",
            "Conversation History",
            "Export Options",
            "API Access",
            "Team Sharing",
            "Custom Models",
            "Support",
            "Analytics"
        ],
        "Free": [
            "50/day",
            "Basic only",
            "5/day",
            "7 days",
            "TXT only",
            "❌",
            "❌",
            "❌",
            "Email",
            "❌"
        ],
        "Starter": [
            "500/day",
            "All models",
            "50/day",
            "Unlimited",
            "TXT + JSON",
            "❌",
            "❌",
            "❌",
            "Email",
            "Limited"
        ],
        "Pro": [
            "✅ Unlimited",
            "All + Premium",
            "✅ Unlimited", 
            "✅ Unlimited",
            "✅ All formats",
            "10k/month",
            "5 members",
            "✅ Yes",
            "Priority (2hr)",
            "✅ Full"
        ],
        "Enterprise": [
            "✅ Unlimited",
            "All + Premium",
            "✅ Unlimited",
            "✅ Unlimited",
            "✅ All formats",
            "100k/month",
            "50 members",
            "✅ Yes",
            "Dedicated",
            "✅ Full + Custom"
        ]
    }
    
    import pandas as pd
    df = pd.DataFrame(plan_data)
    st.dataframe(df, use_container_width=True)
    
    # FAQ Section
    st.divider()
    st.markdown("### ❓ Frequently Asked Questions")
    
    faqs = [
        {
            "q": "Can I change my plan anytime?",
            "a": "Yes! You can upgrade or downgrade your plan at any time. Upgrades take effect immediately, while downgrades take effect at the end of your current billing period."
        },
        {
            "q": "What happens if I exceed my limits?",
            "a": "If you exceed your daily limits on the Free or Starter plans, you'll be prompted to upgrade. Pro and Enterprise plans have unlimited usage for messages and comparisons."
        },
        {
            "q": "Can I cancel my subscription?",
            "a": "Absolutely. You can cancel your subscription anytime from the billing portal. You'll continue to have access until the end of your current billing period."
        },
        {
            "q": "Is my payment information secure?",
            "a": "Yes! We use Stripe, a PCI-DSS compliant payment processor, to handle all payment information. We never store your credit card details on our servers."
        },
        {
            "q": "Do you offer refunds?",
            "a": "We offer a 7-day money-back guarantee for all paid plans. If you're not satisfied, contact us within 7 days for a full refund."
        }
    ]
    
    for i, faq in enumerate(faqs):
        with st.expander(f"Q: {faq['q']}", expanded=False):
            st.markdown(f"**A:** {faq['a']}")
    
    # Contact support
    st.divider()
    st.markdown("### 💬 Need Help?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Billing Questions**")
        st.info("For billing inquiries, email: billing@aichatstudio.com")
    
    with col2:
        st.markdown("**Technical Support**")
        if user.subscription_tier in ["pro", "enterprise"]:
            st.success("✅ You have priority support: support@aichatstudio.com")
        else:
            st.info("General support: support@aichatstudio.com")
    
    # Legal links
    st.markdown("---")
    st.caption("By subscribing, you agree to our Terms of Service and Privacy Policy.")
    
    col_legal1, col_legal2, col_legal3 = st.columns(3)
    
    with col_legal1:
        st.link_button("Terms of Service", "#")
    
    with col_legal2:
        st.link_button("Privacy Policy", "#")
    
    with col_legal3:
        st.link_button("Refund Policy", "#")


if __name__ == "__main__":
    main()