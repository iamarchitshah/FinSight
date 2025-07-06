# lstm_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_lstm_data(df, feature_cols, target_col, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(df)):
        X.append(df[feature_cols].iloc[i - time_steps:i].values)
        y.append(df[target_col].iloc[i])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(df):
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA20', 'MA50', 'BB_upper', 'BB_lower', 'MFI']
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)

    X_open, y_open = prepare_lstm_data(df_scaled, feature_cols, target_col='Open')
    X_close, y_close = prepare_lstm_data(df_scaled, feature_cols, target_col='Close')

    model_open = build_lstm_model((X_open.shape[1], X_open.shape[2]))
    model_close = build_lstm_model((X_close.shape[1], X_close.shape[2]))

    model_open.fit(X_open, y_open, epochs=20, batch_size=16, verbose=0)
    model_close.fit(X_close, y_close, epochs=20, batch_size=16, verbose=0)

    return model_open, model_close, scaler, df_scaled, feature_cols



def predict_lstm_next_7_days(df_scaled, model_open, model_close, scaler, feature_cols, time_steps=60):
    future_predictions = []

    last_sequence = df_scaled[feature_cols].iloc[-time_steps:].values.copy()
    for _ in range(7):
        input_seq = last_sequence.reshape(1, time_steps, len(feature_cols))
        pred_open = model_open.predict(input_seq, verbose=0)[0][0]
        pred_close = model_close.predict(input_seq, verbose=0)[0][0]

        predicted_row = np.array([pred_open]*len(feature_cols))  # placeholder — ideally should use real ratios
        last_sequence = np.vstack([last_sequence[1:], predicted_row])

        inv_open = scaler.inverse_transform([[pred_open] * len(feature_cols)])[0][0]
        inv_close = scaler.inverse_transform([[pred_close] * len(feature_cols)])[0][3]
        future_predictions.append([inv_open, inv_close])

    return pd.DataFrame(future_predictions, columns=['Predicted_Open', 'Predicted_Close'])

