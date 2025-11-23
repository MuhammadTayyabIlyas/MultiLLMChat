# Admin Dashboard Guide

## 🚀 Quick Start

### 1. Upgrade Your Database

Run the database schema upgrade script to add admin tables:

```bash
cd /home/tayyabcheema777/ali/Work/Translation/App_Streamlit
python db_schema_upgrade.py
```

### 2. Configure Admin Access

Add your email to the admin list in `.streamlit/secrets.toml`:

```toml
# Admin Configuration
ADMIN_EMAILS = ["your-email@example.com", "admin@example.com"]
```

### 3. Access the Admin Panel

After logging in as an admin user, visit:
`http://localhost:8501/admin_dashboard`

## 📊 Admin Features

### Overview Dashboard
- Total users and active users
- Monthly recurring revenue (MRR)
- Subscription distribution
- Model usage analytics
- Daily message statistics

### User Management
- View all users with filters (by tier, search by email)
- See user details: messages, active days, last activity
- Grant/revoke admin privileges
- Upgrade/downgrade user tiers
- Delete users (with confirmation)
- Export user data to CSV

### Revenue Monitoring
- Monthly recurring revenue tracking
- Subscription breakdown by tier
- Average revenue per user (ARPU)
- Conversion rate from free to paid
- Stripe integration status
- Failed payments and disputes

### Model & API Management
- View all configured models
- Check API key status for each provider
- Add new models to the registry
- Configure API keys
- Monitor model usage and costs
- Generate secrets.toml templates

### System Settings
- System health monitoring
- Database statistics
- API response times
- Uptime tracking
- Cache management
- Data export tools

## 🔐 Security Notes

- Admin access is restricted to emails in `ADMIN_EMAILS`
- Admin status is stored in the database (`is_admin` column)
- All admin actions are logged (add custom logging as needed)
- Use HTTPS in production for Stripe webhooks
- Regularly rotate API keys
- Monitor admin access logs

## 💰 Monetization Features

The admin dashboard monitors:
- Usage-based limits (messages/day, comparisons/day)
- Subscription tiers: Free, Starter ($9), Pro ($29), Enterprise ($99)
- API call metering for each tier
- Automatic upgrade prompts when limits reached
- Real-time revenue tracking
- User conversion analytics

## 📈 Key Metrics Tracked

### User Metrics
- Total registrations
- Daily active users (DAU)
- Retention rate
- Churn rate
- Average messages per user

### Revenue Metrics
- Monthly Recurring Revenue (MRR)
- Average Revenue Per User (ARPU)
- Conversion rate
- Tier distribution
- Failed payment rate

### Usage Metrics
- Total messages processed
- Messages per model
- API response times
- Daily usage trends
- Peak usage hours

## 🔧 Adding New Models

1. Go to Models & APIs section
2. Click "Add Model"
3. Fill in:
   - Provider type (OpenAI, Anthropic, etc.)
   - Internal model key
   - Display label
   - Model ID from provider
   - Secret key name in secrets.toml
4. Add the API key to your secrets.toml
5. Deploy the changes

## ⚠️ Danger Zone

### Available Dangerous Operations
- Clear all cache (use with caution)
- Delete all data (requires confirmation)
- Bulk user operations (export/delete)

Always backup your database before bulk operations!

## 📊 Data Export

The admin panel allows exporting:
- User list with subscription data
- Usage analytics (JSON/CSV)
- Revenue reports
- Model usage statistics
- API call logs

## 🐛 Troubleshooting

### Admin Access Not Working
1. Check your email is in ADMIN_EMAILS
2. Run db_schema_upgrade.py
3. Log out and log back in
4. Check database has is_admin column

### Stripe Not Configured
1. Get API keys from Stripe dashboard
2. Add to secrets.toml
3. Create products and price IDs
4. Configure webhook endpoint

### Models Not Showing
1. Check API keys are in secrets.toml
2. Verify model configuration
3. Check logs for API errors
4. Test API connection

## 📞 Support

For admin panel issues, check:
- Error logs in terminal
- Browser console for JS errors
- Network tab for failed API calls
- Database integrity