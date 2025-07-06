# 📊 FinSight: An Ensemble-Based Stock Market Prediction System using ML & DL

FinSight is a powerful and user-friendly application that predicts the future performance of Indian stock market equities. It combines traditional Machine Learning models (SVM, Random Forest) with advanced Deep Learning techniques (LSTM, RNN) and crucial technical indicators (RSI, Moving Averages, Bollinger Bands, MFI) to generate accurate 7-day forecasts for Open and Close prices.

---

## 🛠️ Features

* ✅ Predicts Open & Close prices for the next 7 trading days
* ✅ Supports all Indian NSE-listed stocks via Yahoo Finance (e.g. RELIANCE.NS)
* ✅ Uses SVM, RF, LSTM, and RNN models in an ensemble
* ✅ Enhances prediction with technical indicators: RSI, MA, BB, MFI
* ✅ Streamlit dashboard with charts, custom theme, and CSV export
* ✅ Light/Dark mode ready

---

## 🔄 Architecture Overview

```
Historical Stock Data (Yahoo Finance)
       ⬇️
Add Technical Indicators (RSI, MA, BB, MFI)
       ⬇️
Train ML/DL Models (SVM, RF, LSTM, RNN)
       ⬇️
Predict Next 7 Days (Open, Close)
       ⬇️
Ensemble Averaging
       ⬇️
Output Predictions (Graph + CSV)
```

---

## 📆 Dataset

* Source: Yahoo Finance (`yfinance` Python API)
* Stocks Supported: All NSE stocks (e.g., `INFY.NS`, `TCS.NS`, `HDFCBANK.NS`, etc.)
* Timeframe: User-selectable (e.g., Jan 2022 to Dec 2024)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/iamarchitshah/stock_predict.git
cd stock_predict
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚨 Requirements

```
streamlit
pandas
numpy
scikit-learn
yfinance
tensorflow
matplotlib
seaborn
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push your project to GitHub
2. Go to: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New App" and connect your repo
4. Set `app.py` as the entry point
5. Done! Share the public URL with others

---

## 👩‍💼 Authors

**Shah Archit**
**Thakar Maitrey**
Charotar University of Science and Technology
Chandubhai S. Patel Institute of Technology
Department: Information Technology
Year: 3rd Year
Semester: 5
Batch: IT-2-D2

---

## 🏆 License

This project is licensed under the MIT License.
