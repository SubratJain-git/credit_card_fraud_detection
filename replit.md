# Credit Card Fraud Detection App

## Overview
A machine learning-powered web application for detecting fraudulent credit card transactions. Built with Streamlit and scikit-learn, it provides real-time fraud prediction, batch analysis, customizable thresholds, and interactive visualizations.

## Project Structure
```
.
├── app.py              # Main Streamlit application
├── model.py            # ML model utilities and functions
├── fraud_model.pkl     # Saved trained model (auto-generated)
├── .streamlit/
│   └── config.toml     # Streamlit server configuration
└── pyproject.toml      # Python dependencies
```

## Features

### Core Features
- **Dashboard**: Overview of transaction statistics with interactive charts
- **Single Transaction Analysis**: Real-time fraud prediction with confidence scores and detailed fraud indicator explanations
- **Batch Analysis**: Upload CSV files to analyze multiple transactions with export options
- **Model Insights**: Feature importance and model performance metrics

### Advanced Features
- **Customizable Risk Thresholds**: Adjust high/medium risk sensitivity via sidebar sliders
- **Fraud Indicator Drill-down**: Detailed analysis showing which factors contribute to fraud risk
- **Data Export**: Download flagged transactions (CSV), all results (CSV), and analysis reports (TXT)
- **Custom Model Training**: Upload your own historical data to train personalized fraud detection models

## Technology Stack
- **Frontend**: Streamlit
- **ML Framework**: scikit-learn (Random Forest Classifier)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Excel Export**: openpyxl

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Model Details
- Algorithm: Random Forest Classifier with balanced class weights
- Features: 10 transaction attributes including amount, time, location, and payment method
- Training: Uses synthetic data mimicking real fraud patterns (or custom user data)

## Feature Columns
The model uses these transaction features:
- `transaction_amount`: Monetary value of the transaction
- `transaction_hour`: Hour of day (0-23)
- `transaction_day_of_week`: Day of week (0=Monday, 6=Sunday)
- `distance_from_home`: Distance in km from cardholder home
- `distance_from_last_transaction`: Distance in km from previous transaction
- `ratio_to_median_purchase_price`: Amount relative to median spending
- `repeat_retailer`: Whether customer has purchased from retailer before (0/1)
- `used_chip`: Whether chip (EMV) was used (0/1)
- `used_pin_number`: Whether PIN was entered (0/1)
- `online_order`: Whether this was an online transaction (0/1)

## Risk Thresholds
- **High Risk**: Default 70%+ fraud probability (customizable)
- **Medium Risk**: Default 40-70% fraud probability (customizable)
- **Low Risk**: Below medium threshold

## Recent Changes
- December 2, 2025: Initial implementation with all core features
- December 2, 2025: Added customizable risk thresholds, data export, fraud indicator drill-down, and custom model training
