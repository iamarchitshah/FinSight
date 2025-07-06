# rf_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def train_rf_model(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA20', 'MA50', 'BB_upper', 'BB_lower', 'MFI']
    target_cols = ['Open', 'Close']
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    X = df_scaled
    y_open = df['Open'].values
    y_close = df['Close'].values

    X_train, X_test, y_open_train, y_open_test = train_test_split(X, y_open, test_size=0.2, shuffle=False)
    _, _, y_close_train, y_close_test = train_test_split(X, y_close, test_size=0.2, shuffle=False)

    rf_open = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_open.fit(X_train, y_open_train)

    rf_close = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_close.fit(X_train, y_close_train)

    return rf_open, rf_close, scaler

def predict_next_7_days(df, model_open, model_close, scaler, num_prediction_days):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA20', 'MA50', 'BB_upper', 'BB_lower', 'MFI']
    last_days = df[-num_prediction_days:].copy() # Adjust based on num_prediction_days
    future_predictions = []

    for _ in range(num_prediction_days):
        input_scaled = scaler.transform(last_days[features])
        pred_open = model_open.predict(input_scaled[-1].reshape(1, -1))[0]
        pred_close = model_close.predict(input_scaled[-1].reshape(1, -1))[0]

        predicted_row = {
            'Open': pred_open,
            'High': last_days.iloc[-1]['High'],
            'Low': last_days.iloc[-1]['Low'],
            'Close': pred_close,
            'Volume': last_days.iloc[-1]['Volume'],
            'RSI': last_days.iloc[-1]['RSI'],
            'MA20': last_days.iloc[-1]['MA20'],
            'MA50': last_days.iloc[-1]['MA50'],
            'BB_upper': last_days.iloc[-1]['BB_upper'],
            'BB_lower': last_days.iloc[-1]['BB_lower'],
            'MFI': last_days.iloc[-1]['MFI']
        }

        future_predictions.append([pred_open, pred_close])
        predicted_df_row = pd.DataFrame([predicted_row], index=[last_days.index[-1] + pd.Timedelta(days=1)])
        last_days = pd.concat([last_days, predicted_df_row])

    return pd.DataFrame(future_predictions, columns=['Predicted_Open', 'Predicted_Close'])
