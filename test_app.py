#!/usr/bin/env python3
"""
Test script to verify app startup and new login system
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("🧪 Testing AI Chat Studio App...")
print("=" * 50)

try:
    # Test core imports
    print("\n1. Testing imports...")
    from app import *
    print("   ✅ All app imports successful")
    
    # Test database
    print("\n2. Testing database...")
    from db import init_db, get_db_connection
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"   ✅ Database connected with {len(tables)} tables")
    
    # Test models
    print("\n3. Testing AI models...")
    from backend import available_models
    models = available_models()
    print(f"   ✅ {len(models)} AI models available")
    
    # Test auth system
    print("\n4. Testing authentication...")
    from auth.users import user_manager
    print("   ✅ User manager loaded")
    
    # Test premium features
    print("\n5. Testing premium features...")
    from premium_features import render_premium_sidebar
    print("   ✅ Premium features loaded")
    
    print("\n" + "=" * 50)
    print("🎉 All systems operational!")
    print("\n📱 To start the app:")
    print("   streamlit run app.py")
    print("\n🔐 Login Options:")
    print("   • Sign Up: Create new account")
    print("   • Login: Use existing account")
    print("   • Legacy: Use APP_PASSWORD from secrets (admin access)")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)