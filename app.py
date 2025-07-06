# app.py
import os

import streamlit as st
import pandas as pd
import datetime
import numpy as np


from indicators import add_technical_indicators
from fetch_data import fetch_stock_data

from models.svm_model import train_svm_model, predict_next_7_days as svm_predict
from models.rf_model import train_rf_model, predict_next_7_days as rf_predict
from models.lstm_model import train_lstm_model, predict_lstm_next_7_days
from models.rnn_model import train_rnn_model, predict_rnn_next_7_days

from aggregator import combine_ensemble_predictions
os.environ["XDG_CONFIG_HOME"] = os.path.join(os.getcwd(), ".streamlit")

st.set_page_config(page_title="Stock Forecast App", layout="wide", initial_sidebar_state="expanded")

# 💡 Custom Theme
st.markdown("""
    <style>
        body { background-color: #f4f9ff; }
        .main { background-color: #f9fbff; }
        .stApp { color: #111 !important; }
        h1, h2, h3 { color: #0074D9 !important; }
        .css-1aumxhk { color: #28a745 !important; }
    </style>
""", unsafe_allow_html=True)

# ⏱ Sidebar
st.sidebar.title("⚙️ Settings")

# NSE Stock Dropdown
tickers = {
    "Reliance Industries": "RELIANCE.BSE",
    "Tata Consultancy Services (TCS)": "TCS.BSE",
    "Infosys": "INFY.BSE",
    "Wipro": "WIPRO.BSE",
    "HDFC Bank": "HDFCBANK.BSE",
    "ICICI Bank": "ICICIBANK.BSE",
    "State Bank of India (SBI)": "SBIN.BSE",
    "Adani Enterprises": "ADANIENT.BSE",
    "Hindustan Unilever": "HINDUNILVR.BSE",
    "Bharti Airtel": "BHARTIARTL.BSE",
    "Larsen & Toubro": "LT.BSE",
    "Asian Paints": "ASIANPAINT.BSE",
    "Maruti Suzuki": "MARUTI.BSE",
    "Bajaj Finance": "BAJFINANCE.BSE",
    "Axis Bank": "AXISBANK.BSE",
    "Tech Mahindra": "TECHM.BSE",
    "Coal India": "COALINDIA.BSE",
    "Titan Company": "TITAN.BSE",
    "JSW Steel": "JSWSTEEL.BSE",
    "Tata Steel": "TATASTEEL.BSE"
}
selected_company = st.sidebar.selectbox("📈 Select an NSE Stock", list(tickers.keys()))
stock_symbol = tickers[selected_company]

# User selects the start date of historical data, default to 2 years ago
start_date = st.sidebar.date_input("Historical Data Start Date", datetime.date.today() - datetime.timedelta(days=365*2))

# User selects how many days of data to use for training (from the start date)
num_historical_days = st.sidebar.number_input("Historical Days for Training", min_value=60, max_value=1825, value=365,
                                              help="Number of past days' data to use for training models, starting from the selected 'Historical Data Start Date' (min 60 days required).")

# Calculate the end date for fetching historical data
end_date_historical = start_date + datetime.timedelta(days=num_historical_days)

num_prediction_days = st.sidebar.number_input("🔮 Predict Next N Days", min_value=1, max_value=30, value=7)
predict_btn = st.sidebar.button("🔮 Generate Prediction")

# 📊 Main Header
st.title("📈 Indian Stock Market Prediction using ML + DL Ensemble")

if predict_btn:
    st.info("📅 Fetching data...")

    api_key = st.secrets.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        st.error("Alpha Vantage API key not found. Please set it in .streamlit/secrets.toml. Example: ALPHAVANTAGE_API_KEY = \"YOUR_KEY\"")
        st.stop()

    # Validate the number of historical days selected
    if num_historical_days < 60:
        st.error("'Historical Days for Training' must be at least 60 for meaningful predictions.")
        st.stop()

    df = fetch_stock_data(stock_symbol, str(start_date), str(end_date_historical), api_key) # Pass api_key

    if df is None or df.empty:
        st.warning(f"No data found for {stock_symbol} from Alpha Vantage. Trying fallback ticker 'RELIANCE.BSE'...") # Fallback for Alpha Vantage
        fallback_symbol = "RELIANCE.BSE" # This is a hardcoded fallback if the initial selected symbol fails.
        df = fetch_stock_data(fallback_symbol, str(start_date), str(end_date_historical), api_key) # Pass api_key to fallback
        if df is None or df.empty:
            st.error("Failed to fetch real-time data for both selected ticker and fallback ticker. Please check your internet connection, API key, and Alpha Vantage rate limits. The app cannot proceed without real-time data.")
            st.stop()
    
    st.write(f"Fetched {len(df)} rows from Alpha Vantage.")

    df = add_technical_indicators(df)
    st.write(f"Rows remaining after adding indicators: {len(df)}")
    if df is None or df.empty:
        st.error("Not enough data after adding technical indicators. Please increase 'Historical Days for Training' (min 60 days recommended) or try a different stock.")
        st.stop()

    if len(df) < num_prediction_days:
        st.error(f"Not enough historical data ({len(df)} rows) to make {num_prediction_days} predictions. Please increase 'Historical Days for Training' or select fewer prediction days.")
        st.stop()

    try:
        with st.spinner("Training SVM & RF models..."):
            svm_open, svm_close, svm_scaler = train_svm_model(df)
            rf_open, rf_close, rf_scaler = train_rf_model(df)

            svm_preds = svm_predict(df, svm_open, svm_close, svm_scaler, num_prediction_days)
            rf_preds = rf_predict(df, rf_open, rf_close, rf_scaler, num_prediction_days)
    except Exception as e:
        st.error(f"Error during SVM/RF model training or prediction: {e}. Please ensure you have sufficient data and a valid API connection.")
        st.stop()

    try:
        with st.spinner("Training LSTM model..."):
            lstm_open, lstm_close, lstm_scaler, lstm_df_scaled, feature_cols = train_lstm_model(df)
            lstm_preds = predict_lstm_next_7_days(lstm_df_scaled, lstm_open, lstm_close, lstm_scaler, feature_cols, num_prediction_days)
    except Exception as e:
        st.error(f"Error during LSTM model training or prediction: {e}. Please ensure you have sufficient data and a valid API connection.")
        st.stop()

    try:
        with st.spinner("Training RNN model..."):
            rnn_open, rnn_close, rnn_scaler, rnn_df_scaled, feature_cols = train_rnn_model(df)
            rnn_preds = predict_rnn_next_7_days(rnn_df_scaled, rnn_open, rnn_close, rnn_scaler, feature_cols, num_prediction_days)
    except Exception as e:
        st.error(f"Error during RNN model training or prediction: {e}. Please ensure you have sufficient data and a valid API connection.")
        st.stop()

    # Combine all predictions
    ensemble_df = combine_ensemble_predictions(svm_preds, rf_preds, lstm_preds, rnn_preds)
    ensemble_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=num_prediction_days)

    st.subheader("📉 Ensemble Prediction (Next " + str(num_prediction_days) + " Days)")
    st.dataframe(ensemble_df.round(2))

    st.line_chart(ensemble_df)

    # Save results
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    ensemble_df.to_csv(os.path.join(results_dir, "predictions.csv"))
    st.success("✅ Prediction complete. CSV saved as `results/predictions.csv`")
