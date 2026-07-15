import yfinance as yf
import pandas as pd

def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def download_initial(symbol, period="7d", interval="1m"):
    print(f"  Downloading {period} of {interval} data for {symbol}...")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    df = _flatten(df)
    df = df.reset_index()
    if "Datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["Datetime"])
    elif "Date" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"])
    df = df.sort_values("datetime").dropna(subset=["Close"])
    print(f"  Got {len(df)} rows")
    return df

def fetch_live(symbol, interval="1m"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1d", interval=interval)
    if df.empty:
        return None
    df = _flatten(df)
    df = df.reset_index()
    if "Datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["Datetime"])
    elif "Date" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"])
    df = df.sort_values("datetime").dropna(subset=["Close"])
    return df
