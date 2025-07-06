# app.py
import os

import streamlit as st
import pandas as pd
import datetime

from fetch_data import fetch_stock_data
from indicators import add_technical_indicators
from svm_model import train_svm_model, predict_next_7_days as svm_predict
from rf_model import train_rf_model, predict_next_7_days as rf_predict
from lstm_model import train_lstm_model, predict_lstm_next_7_days
from rnn_model import train_rnn_model, predict_rnn_next_7_days
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
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Infosys": "INFY.NS",
    "Wipro": "WIPRO.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India (SBI)": "SBIN.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Axis Bank": "AXISBANK.NS",
    "Tech Mahindra": "TECHM.NS",
    "Coal India": "COALINDIA.NS",
    "Titan Company": "TITAN.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS"
}
selected_company = st.sidebar.selectbox("📈 Select an NSE Stock", list(tickers.keys()))
stock_symbol = tickers[selected_company]

start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2024, 12, 31))
predict_btn = st.sidebar.button("🔮 Predict Next 7 Days")

# 📊 Main Header
st.title("📈 Indian Stock Market Prediction using ML + DL Ensemble")

if predict_btn:
    st.info("📅 Fetching data...")
    df = fetch_stock_data(stock_symbol, str(start_date), str(end_date))
    df = add_technical_indicators(df)
    st.success("✅ Data loaded and indicators added")

    with st.spinner("Training SVM & RF models..."):
        svm_open, svm_close, svm_scaler = train_svm_model(df)
        rf_open, rf_close, rf_scaler = train_rf_model(df)

        svm_preds = svm_predict(df, svm_open, svm_close, svm_scaler)
        rf_preds = rf_predict(df, rf_open, rf_close, rf_scaler)

    with st.spinner("Training LSTM model..."):
        lstm_open, lstm_close, lstm_scaler, lstm_df_scaled, feature_cols = train_lstm_model(df)
        lstm_preds = predict_lstm_next_7_days(lstm_df_scaled, lstm_open, lstm_close, lstm_scaler, feature_cols)

    with st.spinner("Training RNN model..."):
        rnn_open, rnn_close, rnn_scaler, rnn_df_scaled, feature_cols = train_rnn_model(df)
        rnn_preds = predict_rnn_next_7_days(rnn_df_scaled, rnn_open, rnn_close, rnn_scaler, feature_cols)

    # Combine all predictions
    ensemble_df = combine_ensemble_predictions(svm_preds, rf_preds, lstm_preds, rnn_preds)
    ensemble_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)

    st.subheader("📉 Ensemble Prediction (Next 7 Days)")
    st.dataframe(ensemble_df.round(2))

    st.line_chart(ensemble_df)

    # Save results
    ensemble_df.to_csv("results/predictions.csv")
    st.success("✅ Prediction complete. CSV saved as `results/predictions.csv`")
