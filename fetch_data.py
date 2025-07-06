# fetch_data.py

import streamlit as st # Import streamlit for st.secrets
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    api_key = st.secrets.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("ALPHAVANTAGE_API_KEY not found in Streamlit secrets.")
        st.error("Alpha Vantage API key not found. Please set it in .streamlit/secrets.toml")
        return None

    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        # Alpha Vantage uses 'TIME_SERIES_DAILY_ADJUSTED' for daily data
        # It returns data in reverse chronological order
        data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        
        if data.empty:
            print(f"No data found for {ticker} from Alpha Vantage.")
            return None

        # Rename columns to match expected format: Open, High, Low, Close, Volume
        data.columns = [
            'Open', 'High', 'Low', 'Close', 'adjusted close', 'Volume', 'dividend amount', 'split coefficient'
        ]
        df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Filter by date range (Alpha Vantage returns all history by default)
        df = df.loc[start_date:end_date]
        
        df.dropna(inplace=True)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Failed to fetch data for {ticker} from Alpha Vantage: {e}")
        # Check for common Alpha Vantage errors
        if "limit" in str(e).lower():
            st.error("Alpha Vantage API daily/minute limit reached. Please wait or consider upgrading your plan.")
        elif "invalid api key" in str(e).lower():
            st.error("Invalid Alpha Vantage API key. Please check your .streamlit/secrets.toml file.")
        else:
            st.error(f"Failed to fetch data from Alpha Vantage. Error: {e}")
        return None

if __name__ == "__main__":
    # This part needs to be adapted as st.secrets is not available outside a Streamlit app context
    # For local testing of fetch_data.py, you would typically set an environment variable or manually assign the key
    print("Note: Direct execution of fetch_data.py using st.secrets will fail outside Streamlit app.")
    print("Please run the main app.py to test Alpha Vantage integration.")
    # Example usage for testing (requires setting ALPHAVANTAGE_API_KEY as env var manually for this script)
    # os.environ["ALPHAVANTAGE_API_KEY"] = os.getenv("ALPHAVANTAGE_API_KEY", "YOUR_DUMMY_API_KEY_HERE")
    # data = fetch_stock_data("RELIANCE.NS", "2022-01-01", "2024-12-31")
    # if data is not None:
    #    data.to_csv("data_sample_av.csv")
    #    print("Data saved to data_sample_av.csv")
    # else:
    #    print("Failed to fetch data for example.")
