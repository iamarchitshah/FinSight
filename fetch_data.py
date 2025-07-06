# fetch_data.py

import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df is None or df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    data = fetch_stock_data("RELIANCE.NS", "2022-01-01", "2024-12-31")
    data.to_csv("data_sample.csv")
    print("Data saved to data_sample.csv")
