# fetch_data.py

from alpha_vantage.timeseries import TimeSeries
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date, api_key):
    if not api_key:
        print("Alpha Vantage API key not provided to fetch_stock_data.")
        return None

    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        # Using TIME_SERIES_DAILY as TIME_SERIES_DAILY_ADJUSTED is a premium endpoint.
        # This endpoint returns raw (as-traded) daily data.
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        
        if data.empty:
            print(f"No data found for {ticker} from Alpha Vantage (TIME_SERIES_DAILY).")
            return None

        # Alpha Vantage returns column names like '1. open', '2. high', etc.
        data.columns = [
            '1. open', '2. high', '3. low', '4. close', '5. volume' # Alpha Vantage column names
        ]
        df = data[['1. open', '2. high', '3. low', '4. close', '5. volume']].copy()
        
        # Rename to generic names for consistency in other modules
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Ensure index is datetime for .loc to work correctly
        df.index = pd.to_datetime(df.index)
        df = df.loc[start_date:end_date] # Filter by date range
        
        df.dropna(inplace=True)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Failed to fetch data for {ticker} from Alpha Vantage: {e}")
        return None

if __name__ == "__main__":
    # This block is for independent testing of fetch_data.py
    # It requires an API key to be set manually as an environment variable for testing outside Streamlit context
    import os
    dummy_api_key = os.getenv("ALPHAVANTAGE_API_KEY", "YOUR_DUMMY_API_KEY_HERE") # Replace with your actual key for testing
    if dummy_api_key == "YOUR_DUMMY_API_KEY_HERE":
        print("Please set ALPHAVANTAGE_API_KEY environment variable for standalone testing, or provide a dummy key.")

    data = fetch_stock_data("RELIANCE.BSE", "2022-01-01", "2024-12-31", dummy_api_key)
    if data is not None and not data.empty:
        print("Sample data fetched successfully:")
        print(data.head())
    else:
        print("Failed to fetch data for example in standalone test.")
