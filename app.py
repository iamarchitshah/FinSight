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

start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
num_prediction_days = st.sidebar.number_input("🔮 Predict Next N Days", min_value=1, max_value=30, value=7)
predict_btn = st.sidebar.button("🔮 Generate Prediction")

# 📊 Main Header
st.title("📈 Indian Stock Market Prediction using ML + DL Ensemble")

if predict_btn:
    st.info("📅 Fetching data...")

    api_key = st.secrets.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        st.error("Alpha Vantage API key not found. Please set it in .streamlit/secrets.toml")
        st.stop()

    # Check minimum date range for historical data to train models
    min_historical_days = 60 # At least 60 days needed for MA50, and LSTM/RNN time_steps
    
    # Calculate days between start_date and today to check if enough historical data is requested
    if (datetime.date.today() - start_date).days < min_historical_days:
        st.error(f"Please select a 'Start Date' that provides at least {min_historical_days} days of historical data for meaningful predictions. (e.g., set Start Date to {datetime.date.today() - datetime.timedelta(days=min_historical_days + 30)} or earlier)") # Add buffer
        st.stop()

    df = fetch_stock_data(stock_symbol, str(start_date), str(datetime.date.today()), api_key) # Pass api_key

    if df is None or df.empty:
        st.warning(f"No data found for {stock_symbol}. Trying fallback ticker 'RELIANCE.BSE'...") # Updated fallback ticker
        fallback_symbol = "RELIANCE.BSE" # Ensured consistency here for Alpha Vantage
        df = fetch_stock_data(fallback_symbol, str(start_date), str(datetime.date.today()), api_key) # Pass api_key to fallback
        if df is None or df.empty:
            st.warning("Fallback ticker also failed. Loading sample data...")
            sample_size = 120  # Increased to ensure enough rows after indicators
            dates = pd.date_range(start=start_date, periods=sample_size)
            df = pd.DataFrame({
                'Open': np.random.uniform(1000, 2000, size=sample_size),
                'High': np.random.uniform(1000, 2000, size=sample_size),
                'Low': np.random.uniform(1000, 2000, size=sample_size),
                'Close': np.random.uniform(1000, 2000, size=sample_size),
                'Volume': np.random.randint(100000, 500000, size=sample_size)
            }, index=dates)
            st.info("Generated random sample data.")
    st.write(f"Fetched {len(df)} rows from Alpha Vantage or fallback.") # Updated message

    df = add_technical_indicators(df)
    st.write(f"Rows remaining after adding indicators: {len(df)}")
    if df is None or df.empty:
        st.error("Not enough data after adding technical indicators. Please select a longer date range (at least 60 days). ")
        st.stop()

    st.success("✅ Data loaded and indicators added")

    # Check for minimum rows for models
    if len(df) < 60:
        st.error(f"Not enough data for model training. At least 60 rows are required after indicators, but only {len(df)} rows are available. Please select a longer date range.")
        st.stop()
    if len(df) < num_prediction_days:
        st.error(f"Not enough historical data to make {num_prediction_days} predictions. Only {len(df)} rows available. Please select a longer date range or fewer prediction days.")
        st.stop()

    try:
        with st.spinner("Training SVM & RF models..."):
            svm_open, svm_close, svm_scaler = train_svm_model(df)
            rf_open, rf_close, rf_scaler = train_rf_model(df)

            svm_preds = svm_predict(df, svm_open, svm_close, svm_scaler, num_prediction_days)
            rf_preds = rf_predict(df, rf_open, rf_close, rf_scaler, num_prediction_days)
    except Exception as e:
        st.error(f"Error during SVM/RF model training or prediction: {e}")
        st.stop()

    try:
        with st.spinner("Training LSTM model..."):
            lstm_open, lstm_close, lstm_scaler, lstm_df_scaled, feature_cols = train_lstm_model(df)
            lstm_preds = predict_lstm_next_7_days(lstm_df_scaled, lstm_open, lstm_close, lstm_scaler, feature_cols, num_prediction_days)
    except Exception as e:
        st.error(f"Error during LSTM model training or prediction: {e}")
        st.stop()

    try:
        with st.spinner("Training RNN model..."):
            rnn_open, rnn_close, rnn_scaler, rnn_df_scaled, feature_cols = train_rnn_model(df)
            rnn_preds = predict_rnn_next_7_days(rnn_df_scaled, rnn_open, rnn_close, rnn_scaler, feature_cols, num_prediction_days)
    except Exception as e:
        st.error(f"Error during RNN model training or prediction: {e}")
        st.stop()

    # Combine all predictions
    ensemble_df = combine_ensemble_predictions(svm_preds, rf_preds, lstm_preds, rnn_preds)
    ensemble_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=num_prediction_days)

    st.subheader("📉 Ensemble Prediction (Next " + str(num_prediction_days) + " Days)") # Updated subheader
    st.dataframe(ensemble_df.round(2))

    st.line_chart(ensemble_df)

    # Save results
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    ensemble_df.to_csv(os.path.join(results_dir, "predictions.csv"))
    st.success("✅ Prediction complete. CSV saved as `results/predictions.csv`")
