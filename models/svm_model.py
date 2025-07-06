# svm_model.py

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def train_svm_model(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA20', 'MA50', 'BB_upper', 'BB_lower', 'MFI']
    target_cols = ['Open', 'Close']
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    X = df_scaled
    y_open = df['Open'].values
    y_close = df['Close'].values

    X_train, X_test, y_open_train, y_open_test = train_test_split(X, y_open, test_size=0.2, shuffle=False)
    _, _, y_close_train, y_close_test = train_test_split(X, y_close, test_size=0.2, shuffle=False)

    svr_open = SVR(kernel='rbf')
    svr_open.fit(X_train, y_open_train)

    svr_close = SVR(kernel='rbf')
    svr_close.fit(X_train, y_close_train)

    return svr_open, svr_close, scaler

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
        # Append the new prediction to last_days to use for the next prediction
        # Need to re-index last_days to handle appending a dictionary properly or convert to DataFrame row
        # For simplicity, let's create a DataFrame row and concat. Or, more robustly, manage last_sequence as numpy array.
        # Given the existing structure, best to convert predicted_row to a DataFrame row.
        predicted_df_row = pd.DataFrame([predicted_row], index=[last_days.index[-1] + pd.Timedelta(days=1)])
        last_days = pd.concat([last_days, predicted_df_row])

    return pd.DataFrame(future_predictions, columns=['Predicted_Open', 'Predicted_Close'])
