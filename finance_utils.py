import pandas as pd
import yfinance as yf


def get_stock_data(ticker, period="5y"):

    df = yf.download(ticker, period=period)

    df = df.reset_index()

    return df


def add_financial_features(df):

    df["Daily_Return"] = df["Close"].pct_change()

    df["MA_10"] = df["Close"].rolling(10).mean()

    df["MA_50"] = df["Close"].rolling(50).mean()

    df["Volatility"] = df["Daily_Return"].rolling(10).std()

    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod()

    return df


def create_data_summary(df):

    summary = f"""
Dataset Info

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Columns:
"""

    for col in df.columns:
        summary += f"- {col}\n"

    summary += "\nSample Data:\n"

    summary += df.head().to_string()

    return summary