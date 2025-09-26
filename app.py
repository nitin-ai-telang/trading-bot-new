import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from alpaca_trade_api import REST
import plotly.graph_objects as go
from datetime import datetime

# ========== CONFIG ==========
import streamlit as st

API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]
BASE_URL = st.secrets["BASE_URL"]


api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# --- SESSION STATE ---
if "trade_log" not in st.session_state:
    st.session_state.trade_log = []
if "last_prices" not in st.session_state:
    st.session_state.last_prices = {}

# ========== DATA LOADING ==========
def get_data(symbol, period="1y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    return df

def get_live_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period="1d", interval="1m")
    return todays_data["Close"].iloc[-1] if not todays_data.empty else None

# --- STRATEGIES ---
def sma_strategy(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    if df["SMA20"].iloc[-1] > df["SMA50"].iloc[-1]:
        return "BUY"
    elif df["SMA20"].iloc[-1] < df["SMA50"].iloc[-1]:
        return "SELL"
    return "HOLD"

def rsi_strategy(df, period=14, overbought=70, oversold=30):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    rsi = df["RSI"].iloc[-1]
    if rsi < oversold:
        return "BUY"
    elif rsi > overbought:
        return "SELL"
    return "HOLD"

def macd_strategy(df, fast=12, slow=26, signal=9):
    df["EMA12"] = df["Close"].ewm(span=fast).mean()
    df["EMA26"] = df["Close"].ewm(span=slow).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=signal).mean()
    if df["MACD"].iloc[-1] > df["Signal"].iloc[-1]:
        return "BUY"
    elif df["MACD"].iloc[-1] < df["Signal"].iloc[-1]:
        return "SELL"
    return "HOLD"

def bollinger_strategy(df, window=20, num_std=2):
    df["SMA"] = df["Close"].rolling(window).mean()
    df["STD"] = df["Close"].rolling(window).std()
    df["Upper"] = df["SMA"] + num_std * df["STD"]
    df["Lower"] = df["SMA"] - num_std * df["STD"]
    last_price = df["Close"].iloc[-1]
    if last_price < df["Lower"].iloc[-1]:
        return "BUY"
    elif last_price > df["Upper"].iloc[-1]:
        return "SELL"
    return "HOLD"

# --- BACKTEST ENGINE ---
def backtest(df, strategy_func, initial_balance=10000, trade_size=1):
    balance = initial_balance
    position = 0
    equity_curve = []
    trades = 0
    wins = 0

    for i in range(len(df)):
        sub_df = df.iloc[: i + 1]
        signal = strategy_func(sub_df)
        price = df["Close"].iloc[i]

        if signal == "BUY" and position == 0:
            position = trade_size
            entry_price = price
            trades += 1
        elif signal == "SELL" and position > 0:
            pnl = (price - entry_price) * position
            balance += pnl
            if pnl > 0:
                wins += 1
            position = 0

        equity_curve.append(balance + position * price)

    return {
        "final_balance": balance,
        "profit": balance - initial_balance,
        "trades": trades,
        "win_rate": (wins / trades * 100) if trades > 0 else 0,
        "equity_curve": equity_curve,
    }

# --- Trading ---
def execute_trade(signal, symbol, qty, stop_loss_pct, take_profit_pct):
    if signal in ["BUY", "SELL"]:
        price = get_live_price(symbol)
        api.submit_order(symbol=symbol, qty=qty, side=signal.lower(), type="market", time_in_force="gtc")
        
        # Log trade
        st.session_state.trade_log.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": signal,
            "qty": qty,
            "price": price,
            "stop_loss": price * (1 - stop_loss_pct/100) if signal == "BUY" else None,
            "take_profit": price * (1 + take_profit_pct/100) if signal == "BUY" else None,
        })
        return f"✅ Executed {signal} order for {symbol} at ${price:.2f}"
    return "No trade executed"

# ========== STREAMLIT DASHBOARD ==========
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")

# Auto-refresh every 5 seconds
#st_autorefresh = st.experimental_autorefresh(interval=5000, limit=None, key="refresh")
st_autorefresh(interval=5000, limit=None, key="refresh")

st.title("🤖 Multi-Symbol Automated Trading Bot")

# MAIN INPUT: Watchlist
st.subheader("🔎 Enter Tickers (comma separated)")
symbols = st.text_input("Example: AAPL, TSLA, MSFT", value="AAPL, TSLA").upper().replace(" ", "").split(",")

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
period = st.sidebar.selectbox("Period", ["5d", "30d", "90d", "1y"], index=3)
interval = st.sidebar.selectbox("Interval", ["5m", "30m", "1d"], index=2)
qty = st.sidebar.number_input("Trade Quantity per Symbol", min_value=1, value=1)
strategy_choice = st.sidebar.selectbox("Strategy", ["SMA", "RSI", "MACD", "Bollinger Bands"])
stop_loss_pct = st.sidebar.slider("Stop Loss %", 1, 20, 5)
take_profit_pct = st.sidebar.slider("Take Profit %", 1, 50, 10)
auto_trade = st.sidebar.checkbox("Enable Auto-Trade", value=False)

# Choose Strategy
strategy_map = {
    "SMA": sma_strategy,
    "RSI": rsi_strategy,
    "MACD": macd_strategy,
    "Bollinger Bands": bollinger_strategy
}
strategy_func = strategy_map[strategy_choice]

# ========== SYMBOL LOOP ==========
for symbol in symbols:
    if not symbol:
        continue
    
    st.markdown(f"### 📊 {symbol}")

    # Live price
    price = get_live_price(symbol)
    delta = None
    if symbol in st.session_state.last_prices:
        delta = round(price - st.session_state.last_prices[symbol], 2)
    st.metric(label=f"Live Price for {symbol}", value=f"${price:.2f}", delta=delta)
    st.session_state.last_prices[symbol] = price

    # Historical data
    df = get_data(symbol, period, interval)

    # Signal
    signal = strategy_func(df)
    st.write(f"📌 Strategy Signal: **{signal}**")

    # Auto-trade logic
    if auto_trade and signal in ["BUY", "SELL"]:
        result = execute_trade(signal, symbol, qty, stop_loss_pct, take_profit_pct)
        st.success(result)

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Candlestick"
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Backtest
    results = backtest(df, strategy_func)
    st.write(f"💵 Final Balance: ${results['final_balance']:.2f}")
    st.write(f"📊 Total Profit: ${results['profit']:.2f}")
    st.write(f"🔄 Trades Taken: {results['trades']}")
    st.write(f"✅ Win Rate: {results['win_rate']:.2f}%")

    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=df.index,
        y=results["equity_curve"],
        line=dict(color="blue"),
        name="Equity Curve"
    ))
    st.plotly_chart(fig_equity, use_container_width=True)

# Portfolio Info
account = api.get_account()
st.sidebar.write("💰 Account Balance:", account.cash)

# Trade Log
st.subheader("📜 Trade Log")
if st.session_state.trade_log:
    st.dataframe(pd.DataFrame(st.session_state.trade_log))
else:
    st.info("No trades executed yet.")

# ====== PORTFOLIO AGGREGATION ======
st.subheader("📊 Portfolio Performance")

portfolio_equity = []
portfolio_balance = 0
portfolio_trades = 0
portfolio_wins = 0

equity_curves = []

for symbol in symbols:
    if not symbol:
        continue
    df = get_data(symbol, period, interval)
    results = backtest(df, strategy_func)
    equity_curves.append(pd.Series(results["equity_curve"], index=df.index))
    portfolio_balance += results["final_balance"]
    portfolio_trades += results["trades"]
    portfolio_wins += (results["win_rate"]/100) * results["trades"]

# Align curves by date and sum to get portfolio equity
if equity_curves:
    combined_equity = pd.concat(equity_curves, axis=1).fillna(method="ffill").sum(axis=1)

    st.write(f"💰 **Final Portfolio Balance:** ${combined_equity.iloc[-1]:.2f}")
    st.write(f"📈 **Total Portfolio Profit:** ${combined_equity.iloc[-1] - len(symbols)*10000:.2f}")
    st.write(f"🔄 **Total Trades:** {portfolio_trades}")
    st.write(f"✅ **Portfolio Win Rate:** {(portfolio_wins/portfolio_trades*100 if portfolio_trades>0 else 0):.2f}%")

    # Chart
    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(go.Scatter(
        x=combined_equity.index,
        y=combined_equity,
        line=dict(color="green"),
        name="Portfolio Equity"
    ))
    st.plotly_chart(fig_portfolio, use_container_width=True)
else:
    st.info("No portfolio data available yet.")



