"""
Stripe billing integration for subscription management
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

import stripe
import streamlit as st

from auth.users import user_manager, User

logger = logging.getLogger(__name__)


class BillingManager:
    def __init__(self):
        self.stripe_key = st.secrets.get("STRIPE_SECRET_KEY")
        self.stripe_pub_key = st.secrets.get("STRIPE_PUBLISHABLE_KEY")
        
        if self.stripe_key:
            stripe.api_key = self.stripe_key
        
        self.plans = {
            "starter": {
                "name": "Starter",
                "price_id": st.secrets.get("STRIPE_STARTER_PRICE_ID", "price_starter"),
                "amount": 9,
                "currency": "usd",
                "interval": "month"
            },
            "pro": {
                "name": "Pro",
                "price_id": st.secrets.get("STRIPE_PRO_PRICE_ID", "price_pro"),
                "amount": 29,
                "currency": "usd",
                "interval": "month"
            },
            "enterprise": {
                "name": "Enterprise",
                "price_id": st.secrets.get("STRIPE_ENTERPRISE_PRICE_ID", "price_enterprise"),
                "amount": 99,
                "currency": "usd",
                "interval": "month"
            }
        }
    
    def is_configured(self) -> bool:
        """Check if Stripe is properly configured"""
        return bool(self.stripe_key and self.stripe_pub_key)
    
    def create_customer(self, user: User) -> Optional[str]:
        """Create Stripe customer for a user"""
        if not self.is_configured():
            return None
        
        try:
            customer = stripe.Customer.create(
                email=user.email,
                metadata={"user_id": user.id},
                name=f"User {user.id[:8]}"
            )
            user_manager.update_subscription(
                user.id,
                user.subscription_tier,
                stripe_customer_id=customer.id
            )
            return customer.id
        except Exception as e:
            logger.error(f"Failed to create Stripe customer: {e}")
            return None
    
    def create_checkout_session(self, user: User, tier: str, success_url: str, cancel_url: str) -> Optional[str]:
        """Create Stripe checkout session for subscription"""
        if not self.is_configured():
            st.error("Billing is not configured. Please contact support.")
            return None
        
        if tier not in self.plans:
            st.error(f"Invalid tier: {tier}")
            return None
        
        plan = self.plans[tier]
        
        try:
            # Get or create customer
            if not user.stripe_customer_id:
                customer_id = self.create_customer(user)
                if not customer_id:
                    return None
            else:
                customer_id = user.stripe_customer_id
            
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=['card'],
                line_items=[{
                    'price': plan['price_id'],
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    'user_id': user.id,
                    'tier': tier
                }
            )
            
            return session.url
            
        except Exception as e:
            logger.error(f"Failed to create checkout session: {e}")
            st.error(f"Failed to create checkout session: {str(e)}")
            return None
    
    def create_portal_session(self, user: User, return_url: str) -> Optional[str]:
        """Create Stripe customer portal session for subscription management"""
        if not self.is_configured():
            st.error("Billing is not configured. Please contact support.")
            return None
        
        if not user.stripe_customer_id:
            st.error("No billing information found.")
            return None
        
        try:
            session = stripe.billing_portal.Session.create(
                customer=user.stripe_customer_id,
                return_url=return_url,
            )
            return session.url
        except Exception as e:
            logger.error(f"Failed to create portal session: {e}")
            st.error(f"Failed to create billing portal: {str(e)}")
            return None
    
    def handle_webhook(self, payload: Dict, sig_header: str, webhook_secret: str) -> bool:
        """Handle Stripe webhook events"""
        if not self.is_configured():
            return False
        
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, webhook_secret
            )
        except ValueError as e:
            logger.error(f"Invalid webhook payload: {e}")
            return False
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid webhook signature: {e}")
            return False
        
        # Handle subscription events
        if event['type'] == 'checkout.session.completed':
            return self._handle_subscription_created(event)
        elif event['type'] == 'customer.subscription.updated':
            return self._handle_subscription_updated(event)
        elif event['type'] == 'customer.subscription.deleted':
            return self._handle_subscription_cancelled(event)
        
        return True
    
    def _handle_subscription_created(self, event: Dict) -> bool:
        """Handle new subscription"""
        session = event['data']['object']
        
        user_id = session.get('metadata', {}).get('user_id')
        tier = session.get('metadata', {}).get('tier')
        
        if not user_id or not tier:
            logger.error("Missing metadata in subscription event")
            return False
        
        # Get subscription details
        try:
            subscription = stripe.Subscription.retrieve(session['subscription'])
            
            user_manager.update_subscription(
                user_id,
                tier,
                stripe_customer_id=session['customer'],
                stripe_subscription_id=subscription['id']
            )
            
            logger.info(f"Activated {tier} subscription for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle subscription: {e}")
            return False
    
    def _handle_subscription_updated(self, event: Dict) -> bool:
        """Handle subscription updates"""
        subscription = event['data']['object']
        customer_id = subscription['customer']
        
        # Find user by customer ID
        conn = get_db_connection()
        row = conn.execute(
            "SELECT id FROM users WHERE stripe_customer_id = ?",
            (customer_id,)
        ).fetchone()
        conn.close()
        
        if not row:
            logger.error(f"User not found for customer {customer_id}")
            return False
        
        try:
            # Get the price ID to determine tier
            price_id = subscription['items']['data'][0]['price']['id']
            
            # Map price ID to tier
            tier = None
            for tier_name, plan_data in self.plans.items():
                if plan_data['price_id'] == price_id:
                    tier = tier_name
                    break
            
            if not tier:
                logger.error(f"Unknown price ID: {price_id}")
                return False
            
            user_manager.update_subscription(
                row['id'],
                tier,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription['id']
            )
            
            logger.info(f"Updated subscription to {tier} for user {row['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update subscription: {e}")
            return False
    
    def _handle_subscription_cancelled(self, event: Dict) -> bool:
        """Handle subscription cancellation"""
        subscription = event['data']['object']
        customer_id = subscription['customer']
        
        # Find user by customer ID
        conn = get_db_connection()
        row = conn.execute(
            "SELECT id FROM users WHERE stripe_customer_id = ?",
            (customer_id,)
        ).fetchone()
        conn.close()
        
        if not row:
            logger.error(f"User not found for customer {customer_id}")
            return False
        
        # Downgrade to free tier
        user_manager.update_subscription(
            row['id'],
            "free",
            stripe_customer_id=customer_id,
            stripe_subscription_id=None
        )
        
        logger.info(f"Cancelled subscription for user {row['id']}, downgraded to free")
        return True
    
    def get_upcoming_invoice(self, user: User) -> Optional[Dict[str, Any]]:
        """Get upcoming invoice for a user"""
        if not user.stripe_customer_id or not user.stripe_subscription_id:
            return None
        
        try:
            invoice = stripe.Invoice.upcoming(
                customer=user.stripe_customer_id,
                subscription=user.stripe_subscription_id
            )
            return {
                "amount_due": invoice.amount_due / 100,  # Convert from cents
                "currency": invoice.currency,
                "date": datetime.fromtimestamp(invoice.date),
                "period_start": datetime.fromtimestamp(invoice.period_start),
                "period_end": datetime.fromtimestamp(invoice.period_end),
            }
        except Exception as e:
            logger.error(f"Failed to get upcoming invoice: {e}")
            return None
    
    def format_price(self, amount: int, currency: str) -> str:
        """Format price for display"""
        if currency.lower() == 'usd':
            return f"${amount / 100:.2f}"
        return f"{amount / 100:.2f} {currency.upper()}"


# Global billing manager instance
billing_manager = BillingManager()


def render_billing_section(user: User):
    """Render billing UI in settings"""
    st.markdown("### 💳 Billing")
    
    current_tier = user.subscription_tier
    usage = user_manager.get_usage_stats(user.id)
    limits = user_manager.get_tier_limits(current_tier)
    
    # Show current tier
    tier_colors = {
        "free": "🟢",
        "starter": "🟡",
        "pro": "🔵",
        "enterprise": "🔴"
    }
    
    st.info(f"Current Tier: **{tier_colors.get(current_tier, '⚪')} {current_tier.title()}**")
    
    # Show usage
    if limits["daily_messages"] != float('inf'):
        progress = usage["messages_today"] / limits["daily_messages"]
        st.progress(progress, f"Messages today: {usage['messages_today']} / {limits['daily_messages']}")
    else:
        st.success(f"✅ Unlimited messages used today: {usage['messages_today']}")
    
    if limits["daily_comparisons"] != float('inf'):
        comp_progress = usage["comparisons_today"] / limits["daily_comparisons"]
        st.progress(comp_progress, f"Comparisons today: {usage['comparisons_today']} / {limits['daily_comparisons']}")
    else:
        st.success(f"✅ Unlimited comparisons used today: {usage['comparisons_today']}")
    
    st.divider()
    
    # Show upgrade options
    if current_tier == "free":
        st.markdown("#### 🚀 Upgrade Your Experience")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown("##### Starter\n**$9/month**")
                st.write("✓ 500 messages/day")
                st.write("✓ All AI models")
                st.write("✓ 50 comparisons/day")
                st.write("✓ Unlimited history")
                if st.button("Upgrade to Starter", type="primary", use_container_width=True):
                    handle_tier_upgrade(user, "starter")
        
        with col2:
            with st.container():
                st.markdown("##### Pro ⭐\n**$29/month**")
                st.write("✓ **Unlimited messages**")
                st.write("✓ Premium models")
                st.write("✓ Unlimited comparisons")
                st.write("✓ API access (10k/mo)")
                st.markdown("<span style='color: #10A37F;'>⭐ Most Popular</span>", unsafe_allow_html=True)
                if st.button("Upgrade to Pro", type="primary", use_container_width=True):
                    handle_tier_upgrade(user, "pro")
        
        with col3:
            with st.container():
                st.markdown("##### Enterprise\n**$99/month**")
                st.write("✓ Everything in Pro")
                st.write("✓ 100k API calls/mo")
                st.write("✓ Team sharing (50 users)")
                st.write("✓ SSO & White-label")
                if st.button("Contact Sales", type="secondary", use_container_width=True):
                    st.info("Contact: sales@aichatstudio.com")
    
    elif current_tier in ["starter", "pro", "enterprise"]:
        # Show billing portal for existing subscribers
        if st.button("Manage Billing", type="secondary"):
            portal_url = billing_manager.create_portal_session(
                user,
                st.runtime.get_instance().browser.server_url
            )
            if portal_url:
                st.link_button("Go to Billing Portal 🡕", portal_url, type="primary")
        
        if current_tier != "enterprise":
            st.markdown("---")
            if st.button("Change Plan"):
                # Show plan changer
                handle_plan_change(user)


def handle_tier_upgrade(user: User, tier: str):
    """Handle tier upgrade"""
    if not billing_manager.is_configured():
        st.error("Billing is not configured. Please contact support.")
        return
    
    # Create checkout session
    success_url = f"{st.runtime.get_instance().browser.server_url}?upgrade_success=true"
    cancel_url = f"{st.runtime.get_instance().browser.server_url}?upgrade_cancelled=true"
    
    checkout_url = billing_manager.create_checkout_session(
        user, tier, success_url, cancel_url
    )
    
    if checkout_url:
        st.link_button("Complete Payment 🡕", checkout_url, type="primary")
        st.info("You will be redirected to a secure checkout page.")


def handle_plan_change(user: User):
    """Show plan change options"""
    current_tier = user.subscription_tier
    
    st.markdown("### Change Your Plan")
    
    options = ["starter", "pro", "enterprise"]
    if current_tier in options:
        options.remove(current_tier)
    
    new_tier = st.selectbox("Select new plan:", options)
    
    if st.button(f"Change to {new_tier.title()}"):
        # For tier changes, we need to go through Stripe
        handle_tier_upgrade(user, new_tier)