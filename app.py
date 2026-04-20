"""
SecurePay Guard - Production Edition
====================================
Features: SQLite Database, SHA-256 Password Hashing, 
Live Public Sign-Up, Persistent Data Storage, and Explainable AI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime, timedelta
import sqlite3
import hashlib
import extra_streamlit_components as stx
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="SecurePay Enterprise", page_icon="🏦", layout="wide")

# --- DATABASE SETUP (The "Real App" Backend) ---
# Connect to SQLite Database (It creates a file named 'bank_data.db' automatically)
conn = sqlite3.connect('bank_data.db', check_same_thread=False)
c = conn.cursor()

def init_db():
    # Create Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT, name TEXT)''')
    # Create Transactions Table
    c.execute('''CREATE TABLE IF NOT EXISTS transactions 
                 (txn_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, amount REAL, 
                  hour INTEGER, city TEXT, is_new_device BOOLEAN, is_intl BOOLEAN, 
                  score INTEGER, status TEXT, date TEXT)''')
    
    # Create default Admin if it doesn't exist
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        hashed_pw = hashlib.sha256("admin2026".encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?)", ('admin', hashed_pw, 'admin', 'System Administrator'))
    conn.commit()

init_db()

# --- COOKIE MANAGER (For Persistent Login) ---
cookie_manager = stx.CookieManager(key="cookie_manager")

# Auto-Login check on page refresh
cached_user = cookie_manager.get(cookie="securepay_session")

if cached_user and not st.session_state['logged_in']:
    # If cookie exists, fetch user details and auto-login
    c.execute("SELECT * FROM users WHERE username=?", (cached_user,))
    user_data = c.fetchone()
    if user_data:
        st.session_state['logged_in'] = True
        st.session_state['current_user'] = user_data[0]
        st.session_state['role'] = user_data[2]
        st.session_state['name'] = user_data[3]

# --- SECURITY FUNCTIONS ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_login(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username.lower(), hash_password(password)))
    return c.fetchone()

# --- SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1E3A5F; font-weight: 800; text-align: center; border-bottom: 3px solid #FF6B35; padding-bottom: 10px; margin-bottom: 20px;}
    .login-box { max-width: 400px; margin: auto; padding: 30px; border: 1px solid #ddd; border-radius: 10px; background-color: #f8f9fa; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px;}
    .welcome-text { color: #28a745; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# --- HELPER: GAUGE CHART ---
def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Transaction Risk Level", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100]}, 'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [{'range': [0, 40], 'color': "#28a745"}, {'range': [40, 70], 'color': "#ffc107"}, {'range': [70, 100], 'color': "#dc3545"}],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': score}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- THE LOGIC ENGINE (Explainable AI) ---
def analyze_transaction(amount, hour, is_new_device, is_intl):
    score = 5 
    reasons = []
    
    if amount > 50000:
        score += 35
        reasons.append(f"High Value (₹{amount:,.0f})")
    if is_intl:
        score += 30
        reasons.append("International Gateway")
    if is_new_device:
        score += 20
        reasons.append("Unrecognized Device")
    if 1 <= hour <= 5:
        score += 25
        reasons.append("Suspicious Time (1 AM - 5 AM)")
        
    final_score = min(score, 100)
    status = "SAFE" if final_score < 40 else "WARNING" if final_score < 70 else "BLOCKED"
    return final_score, status, reasons

CITY_COORDS = {'Mumbai': [19.0760, 72.8777], 'Delhi': [28.7041, 77.1025], 'Bangalore': [12.9716, 77.5946], 'Chennai': [13.0827, 80.2707], 'Pune': [18.5204, 73.8567], 'Other': [20.5937, 78.9629]}

# --- MAIN APP ---
def main():
    st.markdown('<div class="main-header">🏦 SecurePay Global Network</div>', unsafe_allow_html=True)
    
    # =========================================================
    # AUTHENTICATION SYSTEM (Login / Sign Up)
    # =========================================================
    if not st.session_state['logged_in']:
        auth_mode = st.radio("Welcome! Please select an option:", ["Login", "Sign Up"], horizontal=True)
        
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        if auth_mode == "Login":
            st.subheader("🔒 Secure Login")
            username = st.text_input("Username").lower()
            password = st.text_input("Password", type="password")
            
            # --- Remember Me Checkbox ---
            remember_me = st.checkbox("Remember Me (Keep me logged in)")
            
            if st.button("Login", type="primary", use_container_width=True, key="main_login_button"):
                user_data = verify_login(username, password)
                if user_data:
                    st.session_state['logged_in'] = True
                    st.session_state['current_user'] = user_data[0]
                    st.session_state['role'] = user_data[2]
                    st.session_state['name'] = user_data[3]
                    
                    # --- Cookie Expiration Logic ---
                    if remember_me:
                        # Stays logged in for 30 days
                        expire_time = datetime.now() + timedelta(days=30)
                    else:
                        # Stays logged in for exactly 5 minutes
                        expire_time = datetime.now() + timedelta(minutes=5)

                    # Set the browser cookie!
                    cookie_manager.set("securepay_session", user_data[0], expires_at=expire_time)
                    
                    # --- THE FIX: Let the browser finish saving ---
                    time.sleep(0.5) 
                    
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")
            

        else:
            st.subheader("📝 Create a New Account")
            new_name = st.text_input("Full Name")
            new_user = st.text_input("Choose a Username").lower()
            new_pass = st.text_input("Choose a Password", type="password")
            
            if st.button("Sign Up & Register", type="primary", use_container_width=True):
                if new_name and new_user and new_pass:
                    c.execute("SELECT * FROM users WHERE username=?", (new_user,))
                    if c.fetchone():
                        st.error("⚠️ Username already exists! Choose another.")
                    else:
                        c.execute("INSERT INTO users VALUES (?, ?, ?, ?)", 
                                  (new_user, hash_password(new_pass), 'customer', new_name))
                        conn.commit()
                        st.success("✅ Account created successfully! Please Login.")
                else:
                    st.warning("Please fill all fields.")
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # SECURE PORTAL (If logged in)
    # =========================================================
    else:
        col1, col2 = st.columns([8, 1])
        with col1:
            st.markdown(f'<div class="welcome-text">Welcome back, {st.session_state["name"]} ({st.session_state["role"].capitalize()})</div>', unsafe_allow_html=True)
        with col2:
            if st.button("Logout", type="secondary"):
                # 1. Overwrite the cookie with nothing to instantly kill Auto-Login
                cookie_manager.set("securepay_session", "")
                
                # 2. Tell the browser to delete the file
                cookie_manager.delete("securepay_session")
                
                # 3. Clear the Python session RAM
                st.session_state['logged_in'] = False
                for key in ['current_user', 'role', 'name']:
                    st.session_state[key] = None
                    
                # 4. Give the browser 1 full second to process before restarting!
                time.sleep(1)
                
                st.rerun()

        # -----------------------------------------------------
        # VIEW 1: CUSTOMER VIEW
        # -----------------------------------------------------
        if st.session_state['role'] == 'customer':
            tab1, tab2 = st.tabs(["💳 My Transaction History", "💸 Send New Payment"])
            
            with tab1:
                st.write("### Database Record")
                # Fetch ONLY this user's data from the database
                user_df = pd.read_sql_query(f"SELECT date, amount, city, status FROM transactions WHERE username='{st.session_state['current_user']}' ORDER BY txn_id DESC", conn)
                
                if user_df.empty:
                    st.info("No transactions found in the database. Go to 'Send New Payment' to make your first transaction!")
                else:
                    st.dataframe(user_df.style.applymap(lambda x: 'background-color: #ffcccc' if x == 'BLOCKED' else '', subset=['status']), use_container_width=True)

            with tab2:
                st.write("### Initiate a Real-Time Payment")
                col_in, col_out = st.columns([1, 1.5])
                with col_in:
                    amount = st.number_input("Amount (₹)", min_value=100, value=5000, step=1000)
                    city = st.selectbox("Transaction City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Other"])
                    time_input = st.time_input("Transaction Time", datetime.now())
                    is_new_device = st.checkbox("Using a New Device / IP?")
                    is_intl = st.checkbox("International Transaction?")
                    
                    if st.button("Submit to Database 🚀", type="primary"):
                        score, status, reasons = analyze_transaction(amount, time_input.hour, is_new_device, is_intl)
                        
                        # SAVE TO DATABASE PERMANENTLY
                        c.execute('''INSERT INTO transactions 
                                     (username, amount, hour, city, is_new_device, is_intl, score, status, date) 
                                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                                  (st.session_state['current_user'], amount, time_input.hour, city, 
                                   is_new_device, is_intl, score, status, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        conn.commit()
                        
                        with col_out:
                            st.plotly_chart(create_gauge(score), use_container_width=True)
                            if status == "BLOCKED":
                                st.error(f"🛑 **TRANSACTION BLOCKED & RECORDED**")
                                for r in reasons: st.write(f"- {r}")
                            else:
                                st.success("✅ **TRANSACTION SUCCESSFUL & SAVED**")

                            st.caption("⚠️ *Disclaimer: This risk score is calculated using a heuristic rule engine based on historical banking typologies. It provides a high-confidence risk assessment, not an absolute guarantee of transaction safety.*")
       # -----------------------------------------------------
        # VIEW 2: ADMIN VIEW
        # -----------------------------------------------------
        elif st.session_state['role'] == 'admin':
            # --- We added a 3rd tab here for Bulk Upload ---
            tab1, tab2, tab3 = st.tabs(["🌐 Global Bank Analytics", "🧠 Explainable AI Logs", "📂 Bulk Data Ingestion"])
            
            with tab1:
                st.write("### 🌐 Global Bank Analytics (Live Database)")
                master_df = pd.read_sql_query("SELECT * FROM transactions", conn)
                
                if master_df.empty:
                    st.warning("Database is empty. Ask users to sign up and make transactions to populate the dashboard.")
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Network Volume", f"₹{master_df['amount'].sum():,.0f}")
                    m2.metric("Total Transactions", len(master_df))
                    m3.metric("Fraud Attempts Blocked", len(master_df[master_df['status']=='BLOCKED']))
                    
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        fig_pie = px.pie(master_df, names='status', title="Network Risk Distribution", color='status', color_discrete_map={'SAFE':'#28a745', 'WARNING':'#ffc107', 'BLOCKED':'#dc3545'})
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_chart2:
                        st.write("**📍 Global Fraud Hotspots**")
                        map_df = master_df[master_df['status'] == 'BLOCKED'].copy()
                        if not map_df.empty:
                            map_df['lat'] = map_df['city'].map(lambda x: CITY_COORDS.get(x, [20.5937, 78.9629])[0])
                            map_df['lon'] = map_df['city'].map(lambda x: CITY_COORDS.get(x, [20.5937, 78.9629])[1])
                            view_state = pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=4, pitch=45)
                            layer = pdk.Layer("ScatterplotLayer", data=map_df, get_position='[lon, lat]', get_color='[220, 53, 69, 180]', get_radius=60000)
                            st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=[layer]))
                        else:
                            st.info("No blocked transactions to map yet.")
                    
                    st.write("### Live Network Transaction Feed")
                    st.dataframe(master_df[['date', 'username', 'amount', 'city', 'score', 'status']].sort_values(by='date', ascending=False), use_container_width=True)

            with tab2:
                st.subheader("System Architecture & Explainability")
                st.write("This Admin panel oversees the **Explainable Rule-Based Engine (White Box Model)**. The system dynamically filters data based on the authenticated user's Role-Based Access Control (RBAC) token.")

            # --- THE NEW UPLOADER TAB ---
            with tab3:
                st.write("### 📂 Bulk CSV Ingestion Engine")
                st.info("Upload historical datasets (like the 50,000 transaction log) to run them through the AI engine.")
                
                uploaded_file = st.file_uploader("Upload Transaction Dataset (CSV format)", type="csv")
                
                if uploaded_file is not None:
                    # Read the massive file
                    bulk_df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Successfully loaded {len(bulk_df):,} rows into memory.")
                    
                    if st.button("Run AI Analytics Engine", type="primary"):
                        with st.spinner("Processing thousands of transactions through the Rule Engine..."):
                            # Apply our AI logic to every row
                            def get_score(row):
                                # .get() safely looks for the column. If it's missing, it uses a safe default!
                                amt = row.get('amount', row.get('Amount', 1000))
                                hr = row.get('hour', row.get('Hour', row.get('Time', 12)))
                                device = row.get('is_new_device', row.get('Is_New_Device', False))
                                intl = row.get('is_intl', row.get('Is_Intl', False))
                                
                                score, status, _ = analyze_transaction(amt, hr, device, intl)
                                return pd.Series([score, status])
                            
                            bulk_df[['Predicted_Score', 'Predicted_Status']] = bulk_df.apply(get_score, axis=1)
                            
                            # Show the first 1000 so the browser doesn't crash
                            st.write("### Analytics Complete (Showing top 1,0000 results)")
                            st.dataframe(bulk_df.head(100000).style.applymap(lambda x: 'background-color: #ffcccc' if x == 'BLOCKED' else ('background-color: #fff3cd' if x == 'WARNING' else ''), subset=['Predicted_Status']), use_container_width=True)

if __name__ == "__main__":
    main()
