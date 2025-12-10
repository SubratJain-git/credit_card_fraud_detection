"""
Credit Card Fraud Detection using Data Analytics (India)
=========================================================
B.Tech Computer Science Final Year Project

This is the main application file that runs the Streamlit web app.
The app provides fraud detection analysis for credit card transactions
using rule-based scoring system with Indian context.

How to Run:
-----------
1. Click the "Run" button in Replit
2. The app will be available in the webview
3. Navigate using the sidebar menu

Author: Subrat Jain
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Credit Card Fraud Detection - India",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #FF6B35;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .risk-high {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .risk-suspicious {
        background-color: #ffc107;
        color: #212529;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .risk-normal {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .section-header {
        color: #1E3A5F;
        border-left: 4px solid #FF6B35;
        padding-left: 1rem;
        margin: 1.5rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #ddd;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FRAUD DETECTION LOGIC - Rule-Based Scoring System
# ============================================================================

def calculate_risk_score(row, df):
    """
    Calculate risk score for a transaction based on rule-based scoring.
    
    Rules Applied:
    1. High amount (> ₹20,000) → +30 points
    2. International transaction → +25 points
    3. Online + late night (00:00-04:00) → +20 points
    4. Multiple transactions in short time → +15 points
    5. Uncommon merchant category → +10 points
    
    Parameters:
    -----------
    row : pandas Series
        Single transaction row
    df : pandas DataFrame
        Full dataset (for checking transaction patterns)
    
    Returns:
    --------
    tuple : (risk_score, triggered_rules)
    """
    risk_score = 0
    triggered_rules = []
    
    # Rule 1: High amount transaction (> ₹20,000)
    if row['amount_in_inr'] > 20000:
        risk_score += 30
        triggered_rules.append(f"High amount: ₹{row['amount_in_inr']:,.2f} (> ₹20,000)")
    
    # Rule 2: International transaction
    if row['is_international'] == True or str(row['is_international']).lower() == 'true':
        risk_score += 25
        triggered_rules.append("International transaction detected")
    
    # Rule 3: Online transaction during late night hours (00:00 - 04:00)
    try:
        if isinstance(row['transaction_datetime'], str):
            tx_time = datetime.strptime(row['transaction_datetime'], '%Y-%m-%d %H:%M:%S')
        else:
            tx_time = row['transaction_datetime']
        
        hour = tx_time.hour
        if row['channel'] == 'Online' and (0 <= hour < 4):
            risk_score += 20
            triggered_rules.append(f"Late night online transaction at {hour:02d}:00 hours")
    except:
        pass
    
    # Rule 4: Multiple transactions by same customer in short time
    try:
        customer_id = row['customer_id']
        customer_txns = df[df['customer_id'] == customer_id]
        if len(customer_txns) > 3:
            risk_score += 15
            triggered_rules.append(f"Customer has {len(customer_txns)} transactions (potential velocity attack)")
    except:
        pass
    
    # Rule 5: Uncommon merchant category or suspicious merchant
    suspicious_keywords = ['unknown', 'suspicious', 'midnight', 'foreign', 'night']
    merchant_name = str(row['merchant_name']).lower()
    if any(keyword in merchant_name for keyword in suspicious_keywords):
        risk_score += 10
        triggered_rules.append(f"Suspicious merchant name: {row['merchant_name']}")
    
    # Cap the score at 100
    risk_score = min(risk_score, 100)
    
    return risk_score, triggered_rules


def get_risk_label(risk_score):
    """
    Classify transaction based on risk score.
    
    Classification:
    - risk_score >= 70 → "High Risk"
    - 40 <= risk_score < 70 → "Suspicious"
    - risk_score < 40 → "Normal"
    """
    if risk_score >= 70:
        return "High Risk"
    elif risk_score >= 40:
        return "Suspicious"
    else:
        return "Normal"


def process_transactions(df):
    """
    Process all transactions and calculate risk scores.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Raw transaction data
    
    Returns:
    --------
    pandas DataFrame : Processed data with risk scores and labels
    """
    risk_data = df.apply(lambda row: calculate_risk_score(row, df), axis=1)
    df['risk_score'] = risk_data.apply(lambda x: x[0])
    df['triggered_rules'] = risk_data.apply(lambda x: x[1])
    df['risk_label'] = df['risk_score'].apply(get_risk_label)
    
    return df


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_sample_data():
    """
    Load the sample Indian transaction dataset from CSV file.
    This function is cached for performance.
    """
    try:
        df = pd.read_csv('data/transactions_sample_india.csv')
        df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
        df = process_transactions(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def show_home_page():
    """Display the Home Page with project overview."""
    
    st.markdown('<h1 class="main-header">Credit Card Fraud Detection using Data Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">India-Focused B.Tech Final Year Project By Subrat Jain</p>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Problem Statement</h2>', unsafe_allow_html=True)
    st.write("""
    With the rapid growth of digital payments in India, credit card fraud has become a significant concern. 
    The Reserve Bank of India (RBI) reports that digital payment transactions have increased exponentially, 
    making the financial ecosystem vulnerable to fraudulent activities. Credit card fraud can result in 
    substantial financial losses for both consumers and financial institutions.
    
    Common types of credit card fraud in India include:
    - **Card-not-present fraud**: Unauthorized online transactions
    - **Card cloning**: Duplicating card information at compromised POS terminals
    - **Identity theft**: Using stolen personal information to make transactions
    - **International fraud**: Suspicious transactions from foreign locations
    """)
    
    st.markdown('<h2 class="section-header">Objective</h2>', unsafe_allow_html=True)
    st.write("""
    This project aims to develop a web-based Credit Card Fraud Detection System that:
    
    1. **Analyzes transaction patterns** to identify suspicious activities
    2. **Implements rule-based scoring** to classify transactions by risk level
    3. **Provides visual analytics** through interactive dashboards
    4. **Offers transparency** by explaining why each transaction is flagged
    
    The system uses data analytics techniques to help identify potentially fraudulent 
    credit card transactions in the Indian context, focusing on patterns specific to 
    Indian merchants, cities, and transaction behaviors.
    """)
    
    st.markdown('<h2 class="section-header">Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Data Management**\n\n- Load sample Indian transaction dataset\n- Upload custom CSV files\n- View and filter transactions")
        
        st.success("**Fraud Scoring**\n\n- Rule-based risk assessment\n- Score range: 0-100\n- Three risk levels: Normal, Suspicious, High Risk")
    
    with col2:
        st.warning("**Analytics Dashboard**\n\n- Interactive visualizations\n- Fraud pattern analysis\n- State-wise and channel-wise breakdown")
        
        st.error("**Detailed Analysis**\n\n- Transaction detail view\n- Rule explanation for each flag\n- Comprehensive filtering")
    
    st.markdown('<h2 class="section-header">Important Disclaimer</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer-box">
        <h4>Academic Project Disclaimer</h4>
        <ul>
            <li>This is an <strong>academic demonstration project</strong> developed for educational purposes only</li>
            <li>Uses <strong>dummy/sample transaction data</strong> - no real credit card information</li>
            <li>Card numbers are shown as <strong>masked (last 4 digits only)</strong> for demonstration</li>
            <li>This system is <strong>NOT intended for real banking or financial use</strong></li>
            <li>The fraud detection rules are simplified for educational demonstration</li>
            <li>No actual financial decisions should be made based on this system</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Quick Statistics</h2>', unsafe_allow_html=True)
    
    df = load_sample_data()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(df))
        
        with col2:
            st.metric("Total Amount", f"₹{df['amount_in_inr'].sum():,.0f}")
        
        with col3:
            fraud_count = df[df['is_fraud'] == 1].shape[0]
            st.metric("Fraud Cases", fraud_count)
        
        with col4:
            fraud_pct = (fraud_count / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_pct:.1f}%")


def show_dataset_page():
    """Display the Dataset/Transactions Page."""
    
    st.markdown('<h1 class="main-header">Transaction Dataset</h1>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Load Data</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Sample Dataset", use_container_width=True):
            st.session_state['data_loaded'] = True
            st.rerun()
    
    with col2:
        uploaded_file = st.file_uploader("Or Upload Your CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
                df = process_transactions(df)
                st.session_state['custom_data'] = df
                st.session_state['data_loaded'] = True
                st.success("Custom data uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    if 'custom_data' in st.session_state:
        df = st.session_state['custom_data']
    else:
        df = load_sample_data()
    
    if df is None:
        st.warning("Please load the sample dataset or upload a CSV file.")
        return
    
    st.markdown('<h2 class="section-header">Summary Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    
    with col2:
        st.metric("Total Amount", f"₹{df['amount_in_inr'].sum():,.2f}")
    
    with col3:
        st.metric("Unique Customers", df['customer_id'].nunique())
    
    with col4:
        fraud_vs_genuine = f"{df[df['is_fraud']==1].shape[0]} / {df[df['is_fraud']==0].shape[0]}"
        st.metric("Fraud / Genuine", fraud_vs_genuine)
    
    st.markdown('<h2 class="section-header">Filters</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        states = ['All'] + sorted(df['state'].unique().tolist())
        selected_state = st.selectbox("State", states)
    
    with col2:
        cities = ['All'] + sorted(df['merchant_city'].unique().tolist())
        selected_city = st.selectbox("City", cities)
    
    with col3:
        channels = ['All'] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox("Channel", channels)
    
    with col4:
        risk_levels = ['All', 'High Risk', 'Suspicious', 'Normal']
        selected_risk = st.selectbox("Risk Level", risk_levels)
    
    search_query = st.text_input("Search by Transaction ID or Merchant Name")
    
    filtered_df = df.copy()
    
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['state'] == selected_state]
    
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['merchant_city'] == selected_city]
    
    if selected_channel != 'All':
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]
    
    if selected_risk != 'All':
        filtered_df = filtered_df[filtered_df['risk_label'] == selected_risk]
    
    if search_query:
        filtered_df = filtered_df[
            (filtered_df['transaction_id'].str.contains(search_query, case=False)) |
            (filtered_df['merchant_name'].str.contains(search_query, case=False))
        ]
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} transactions")
    
    st.markdown('<h2 class="section-header">Transaction Details</h2>', unsafe_allow_html=True)
    
    display_df = filtered_df[['transaction_id', 'card_last4', 'transaction_datetime', 
                              'amount_in_inr', 'merchant_name', 'merchant_city', 'state',
                              'channel', 'risk_score', 'risk_label', 'is_fraud']].copy()
    
    display_df['amount_in_inr'] = display_df['amount_in_inr'].apply(lambda x: f"₹{x:,.2f}")
    display_df['card_last4'] = display_df['card_last4'].apply(lambda x: f"****{x}")
    
    def highlight_risk(val):
        if val == 'High Risk':
            return 'background-color: #ffcccc'
        elif val == 'Suspicious':
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #d4edda'
    
    page_size = 20
    total_pages = max(1, len(display_df) // page_size + (1 if len(display_df) % page_size > 0 else 0))
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    st.dataframe(
        display_df.iloc[start_idx:end_idx].style.applymap(highlight_risk, subset=['risk_label']),
        use_container_width=True,
        height=500
    )
    
    st.caption(f"Page {page} of {total_pages}")
    
    st.markdown('<h2 class="section-header">Transaction Detail View</h2>', unsafe_allow_html=True)
    
    selected_txn = st.selectbox(
        "Select a transaction to view details:",
        filtered_df['transaction_id'].tolist()
    )
    
    if selected_txn:
        txn = df[df['transaction_id'] == selected_txn].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Information")
            st.write(f"**Transaction ID:** {txn['transaction_id']}")
            st.write(f"**Card (Last 4):** ****{txn['card_last4']}")
            st.write(f"**Date/Time:** {txn['transaction_datetime']}")
            st.write(f"**Amount:** ₹{txn['amount_in_inr']:,.2f}")
            st.write(f"**Merchant:** {txn['merchant_name']}")
            st.write(f"**Category:** {txn['merchant_category']}")
            st.write(f"**Location:** {txn['merchant_city']}, {txn['state']}")
            st.write(f"**Channel:** {txn['channel']}")
            st.write(f"**International:** {'Yes' if txn['is_international'] else 'No'}")
            st.write(f"**Chip Used:** {'Yes' if txn['is_chip_used'] else 'No'}")
        
        with col2:
            st.subheader("Risk Assessment")
            
            risk_color = '#dc3545' if txn['risk_label'] == 'High Risk' else '#ffc107' if txn['risk_label'] == 'Suspicious' else '#28a745'
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background-color: {risk_color}; color: white; border-radius: 10px;">
                <h2 style="margin: 0;">Risk Score: {txn['risk_score']}</h2>
                <p style="margin: 0; font-size: 1.2rem;">{txn['risk_label']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.subheader("Triggered Rules")
            
            if txn['triggered_rules']:
                for rule in txn['triggered_rules']:
                    st.warning(f"{rule}")
            else:
                st.success("No suspicious patterns detected")
            
            if txn['is_fraud'] == 1:
                st.error("**Labelled as Fraud in Dataset** (For Academic Study Only)")
            else:
                st.info("**Labelled as Genuine in Dataset**")


def show_dashboard_page():
    """Display the Fraud Analytics Dashboard."""
    
    st.markdown('<h1 class="main-header">Fraud Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if 'custom_data' in st.session_state:
        df = st.session_state['custom_data']
    else:
        df = load_sample_data()
    
    if df is None:
        st.warning("Please load the dataset first from the Dataset page.")
        return
    
    st.markdown('<h2 class="section-header">Filters</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_date = df['transaction_datetime'].min().date()
        max_date = df['transaction_datetime'].max().date()
        date_range = st.date_input("Date Range", [min_date, max_date])
    
    with col2:
        states = ['All'] + sorted(df['state'].unique().tolist())
        selected_state = st.selectbox("State", states, key='dash_state')
    
    with col3:
        channels = ['All'] + sorted(df['channel'].unique().tolist())
        selected_channel = st.selectbox("Channel", channels, key='dash_channel')
    
    with col4:
        risk_levels = ['All', 'High Risk', 'Suspicious', 'Normal']
        selected_risk = st.selectbox("Risk Level", risk_levels, key='dash_risk')
    
    filtered_df = df.copy()
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['transaction_datetime'].dt.date >= date_range[0]) &
            (filtered_df['transaction_datetime'].dt.date <= date_range[1])
        ]
    
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['state'] == selected_state]
    
    if selected_channel != 'All':
        filtered_df = filtered_df[filtered_df['channel'] == selected_channel]
    
    if selected_risk != 'All':
        filtered_df = filtered_df[filtered_df['risk_label'] == selected_risk]
    
    st.markdown('<h2 class="section-header">Key Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_txns = len(filtered_df)
    fraud_txns = filtered_df[filtered_df['is_fraud'] == 1].shape[0]
    fraud_pct = (fraud_txns / total_txns * 100) if total_txns > 0 else 0
    max_amount = filtered_df['amount_in_inr'].max() if len(filtered_df) > 0 else 0
    
    with col1:
        st.metric("Total Transactions", f"{total_txns:,}")
    
    with col2:
        st.metric("Fraud Transactions", f"{fraud_txns:,}")
    
    with col3:
        st.metric("Fraud Percentage", f"{fraud_pct:.1f}%")
    
    with col4:
        st.metric("Highest Amount", f"₹{max_amount:,.0f}")
    
    st.markdown('<h2 class="section-header">Visualizations</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fraud_counts = filtered_df['is_fraud'].value_counts().reset_index()
        fraud_counts.columns = ['Type', 'Count']
        fraud_counts['Type'] = fraud_counts['Type'].map({0: 'Genuine', 1: 'Fraud'})
        
        fig1 = px.bar(
            fraud_counts, 
            x='Type', 
            y='Count',
            color='Type',
            color_discrete_map={'Genuine': '#28a745', 'Fraud': '#dc3545'},
            title='Fraud vs Genuine Transactions'
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        channel_fraud = filtered_df[filtered_df['is_fraud'] == 1].groupby('channel').size().reset_index(name='Count')
        
        if len(channel_fraud) > 0:
            fig2 = px.pie(
                channel_fraud,
                values='Count',
                names='channel',
                title='Fraud Distribution by Channel',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No fraud cases in selected filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        state_fraud = filtered_df[filtered_df['is_fraud'] == 1].groupby('state').size().reset_index(name='Count')
        state_fraud = state_fraud.sort_values('Count', ascending=True).tail(10)
        
        if len(state_fraud) > 0:
            fig3 = px.bar(
                state_fraud,
                y='state',
                x='Count',
                orientation='h',
                title='Top States by Fraud Cases',
                color='Count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No fraud cases in selected filters")
    
    with col2:
        filtered_df['date'] = filtered_df['transaction_datetime'].dt.date
        fraud_over_time = filtered_df[filtered_df['is_fraud'] == 1].groupby('date').size().reset_index(name='Count')
        
        if len(fraud_over_time) > 0:
            fig4 = px.line(
                fraud_over_time,
                x='date',
                y='Count',
                title='Fraud Transactions Over Time',
                markers=True
            )
            fig4.update_traces(line_color='#dc3545')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No fraud cases in selected filters")
    
    st.markdown('<h3>Risk Level Distribution</h3>', unsafe_allow_html=True)
    
    risk_dist = filtered_df['risk_label'].value_counts().reset_index()
    risk_dist.columns = ['Risk Level', 'Count']
    
    fig5 = px.bar(
        risk_dist,
        x='Risk Level',
        y='Count',
        color='Risk Level',
        color_discrete_map={'Normal': '#28a745', 'Suspicious': '#ffc107', 'High Risk': '#dc3545'},
        title='Transactions by Risk Level'
    )
    fig5.update_layout(showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig6 = px.box(
            filtered_df,
            x='is_fraud',
            y='amount_in_inr',
            color='is_fraud',
            labels={'is_fraud': 'Transaction Type', 'amount_in_inr': 'Amount (₹)'},
            title='Amount Distribution: Fraud vs Genuine',
            color_discrete_map={0: '#28a745', 1: '#dc3545'}
        )
        fig6.update_xaxes(ticktext=['Genuine', 'Fraud'], tickvals=[0, 1])
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        category_fraud = filtered_df.groupby('merchant_category').agg({
            'is_fraud': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        category_fraud.columns = ['Category', 'Fraud Count', 'Total']
        category_fraud = category_fraud.sort_values('Fraud Count', ascending=False).head(8)
        
        fig7 = px.bar(
            category_fraud,
            x='Category',
            y='Fraud Count',
            title='Fraud Cases by Merchant Category',
            color='Fraud Count',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig7, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main function to run the Streamlit application."""
    
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    pages = {
        "Home": "home",
        "Dataset": "dataset",
        "Dashboard": "dashboard"
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Project Info")
    st.sidebar.info("""
    **Credit Card Fraud Detection**
    
    B.Tech CSE Final Year Project | Developed by Subrat Jain
    
    India-Focused Implementation
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Developed for Academic Purposes Only")
    
    page = pages[selection]
    
    if page == "home":
        show_home_page()
    elif page == "dataset":
        show_dataset_page()
    elif page == "dashboard":
        show_dashboard_page()
    
    st.markdown("""
    <div class="footer">
        <p>Credit Card Fraud Detection System | B.Tech CS Final Year Project | India | Subrat Jain</p>
        <p><small>Disclaimer: This is an academic project using dummy data. Not for real financial use.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
