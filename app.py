import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from alpaca_trade_api.rest import REST
from streamlit_autorefresh import st_autorefresh

# ======================================================
# CONFIG: Alpaca credentials (loaded from Streamlit secrets)
# ======================================================
API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]
BASE_URL = st.secrets["BASE_URL"]

alpaca = REST(API_KEY, API_SECRET, base_url=BASE_URL)

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def get_data(ticker, period="1mo", interval="1d"):
    """Fetch price data from Yahoo Finance"""
    try:
        df = yf.download(ticker, period=period, interval=interval)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Example simple strategies
def sma_strategy(df, i, short=10, long=30):
    if i < long:
        return None
    short_ma = df["Close"].iloc[i - short : i].mean()
    long_ma = df["Close"].iloc[i - long : i].mean()
    if short_ma > long_ma:
        return "buy"
    elif short_ma < long_ma:
        return "sell"
    return None

def rsi_strategy(df, i, period=14, overbought=70, oversold=30):
    if i < period:
        return None
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    value = rsi.iloc[i]
    if value < oversold:
        return "buy"
    elif value > overbought:
        return "sell"
    return None

# Backtest engine
def backtest(df, strategy_func, initial_balance=10000, trade_size=1):
    if df.empty or "Close" not in df.columns or strategy_func is None:
        return {
            "final_balance": initial_balance,
            "profit": 0,
            "trades": 0,
            "win_rate": 0,
            "equity_curve": [initial_balance],
        }

    balance = initial_balance
    position = 0
    equity_curve = []
    trades = 0
    wins = 0

    for i in range(len(df)):
        try:
            price = float(df["Close"].iloc[i])
        except Exception:
            continue

        if np.isnan(price):
            equity_curve.append(balance + position * 0)
            continue

        signal = strategy_func(df, i)

        if signal == "buy":
            position += trade_size
            balance -= trade_size * price
            trades += 1
        elif signal == "sell" and position > 0:
            position -= trade_size
            balance += trade_size * price
            trades += 1
            if balance > initial_balance:
                wins += 1

        equity_curve.append(balance + position * price)

    final_balance = balance + position * float(df["Close"].iloc[-1])
    profit = final_balance - initial_balance
    win_rate = (wins / trades * 100) if trades > 0 else 0

    return {
        "final_balance": final_balance,
        "profit": profit,
        "trades": trades,
        "win_rate": win_rate,
        "equity_curve": equity_curve,
    }

# ======================================================
# STREAMLIT DASHBOARD
# ======================================================
st.set_page_config(page_title="Trading Bot", layout="wide")

st.title("📈 Automated Trading Bot with Backtesting")

# Auto-refresh every 5s
st_autorefresh(interval=5000, limit=None, key="refresh")

# User Inputs
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Enter ticker symbol", "AAPL").upper()
with col2:
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=0)
with col3:
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

strategy_choice = st.selectbox("Strategy", ["SMA Crossover", "RSI"])

if ticker:
    df = get_data(ticker, period, interval)

    if df.empty:
        st.warning(f"No data found for {ticker}")
    else:
        # Select strategy
        if strategy_choice == "SMA Crossover":
            strategy_func = sma_strategy
        else:
            strategy_func = rsi_strategy

        # Backtest
        results = backtest(df, strategy_func)

        st.subheader("Backtest Results")
        st.write(f"💰 Final Balance: ${results['final_balance']:.2f}")
        st.write(f"📊 Profit: ${results['profit']:.2f}")
        st.write(f"📝 Trades: {results['trades']}")
        st.write(f"✅ Win Rate: {results['win_rate']:.2f}%")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Price"))
        fig.add_trace(go.Scatter(
            y=results["equity_curve"],
            x=df.index[: len(results["equity_curve"])],
            mode="lines",
            name="Equity Curve",
            yaxis="y2",
        ))

        fig.update_layout(
            title=f"{ticker} Price & Backtest Equity Curve",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Equity", overlaying="y", side="right"),
            legend=dict(x=0, y=1, traceorder="normal"),
        )

        st.plotly_chart(fig, use_container_width=True)


if df.empty or "Close" not in df.columns:
    st.warning(f"⚠️ No usable data for {ticker}. Please try another symbol/interval.")
else:
    results = backtest(df, strategy_func)
