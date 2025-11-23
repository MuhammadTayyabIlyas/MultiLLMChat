"""
Database schema upgrade for admin features and user management
Run this once to upgrade your existing database
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
import time
import json

# Same DB_PATH as your main db.py
DB_DIR = Path("data")
DB_PATH = DB_DIR / "chat.db"

def upgrade_database():
    """Upgrade database schema to support admin features"""
    
    print("🔧 Upgrading database schema...")
    
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    
    try:
        with conn:
            # Create users table if not exists
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
                    reset_token_expiry INTEGER,
                    is_admin BOOLEAN DEFAULT FALSE,
                    created_by TEXT
                )
            """)
            
            # Create daily usage table
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
            
            # Create API usage tracking table
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
            
            # Create custom models table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS custom_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    created_at INTEGER
                )
            """)
            
            # Create team invites table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS team_invites (
                    token TEXT PRIMARY KEY,
                    owner_id TEXT,
                    email TEXT,
                    role TEXT,
                    created_at INTEGER
                )
            """)
            
            # Create support tickets table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS support_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    type TEXT,
                    subject TEXT,
                    description TEXT,
                    status TEXT DEFAULT 'open',
                    created_at INTEGER,
                    resolved_at INTEGER
                )
            """)
            
            # Create admin models table (for model registry persistence)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS admin_models (
                    key TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    created_by TEXT,
                    created_at INTEGER
                )
            """)
            
            print("✅ Tables created successfully!")
            
            # Import existing session IDs as free users (optional)
            migrate_existing_sessions(conn)
            
            print("\n🎉 Database upgrade complete!")
            print("\nNext steps:")
            print("1. Add .streamlit/secrets.toml with your API keys")
            print("2. Set ADMIN_EMAILS in secrets.toml")
            print("3. Run your app and the new features will be available")
            
    except Exception as e:
        print(f"❌ Error upgrading database: {e}")
        conn.rollback()
    finally:
        conn.close()

def migrate_existing_sessions(conn):
    """Migrate existing session IDs to user accounts"""
    
    print("\n📤 Migrating existing sessions to users...")
    
    # Get all unique session IDs from messages
    cursor = conn.execute("SELECT DISTINCT session_id FROM messages")
    sessions = cursor.fetchall()
    
    migrated = 0
    for row in sessions:
        session_id = row['session_id']
        email = f"{session_id[:8]}@legacy.user"
        
        # Check if this session is already a user
        existing = conn.execute(
            "SELECT id FROM users WHERE id = ?",
            (session_id,)
        ).fetchone()
        
        if not existing:
            try:
                conn.execute("""
                    INSERT INTO users (id, email, subscription_tier, created_at, last_login)
                    VALUES (?, ?, 'free', ?, ?)
                """, (session_id, email, int(time.time()), int(time.time())))
                migrated += 1
            except sqlite3.IntegrityError:
                pass  # Already exists
    
    conn.commit()
    print(f"✅ Migrated {migrated} existing sessions to user accounts")

def check_schema():
    """Check current database schema"""
    print("🔍 Current Database Schema:")
    print("=" * 50)
    
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' 
        ORDER BY name
    """)
    
    tables = cursor.fetchall()
    
    if not tables:
        print("No tables found - database is new")
    else:
        for table in tables:
            table_name = table['name']
            print(f"\n📋 Table: {table_name}")
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for col in columns:
                print(f"  - {col['name']} ({col['type']}) {'PRIMARY KEY' if col['pk'] else ''}")
    
    conn.close()

if __name__ == "__main__":
    print("🚀 AI Chat Studio Database Upgrade")
    print("=" * 50)
    
    # Check current schema first
    check_schema()
    
    print("\n" + "=" * 50)
    
    # Prompt for confirmation
    response = input("\nProceed with upgrade? (y/N): ")
    
    if response.lower() == 'y':
        upgrade_database()
    else:
        print("Upgrade cancelled.")