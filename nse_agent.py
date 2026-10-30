import time
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import pandas_ta as ta
import requests
import yfinance as yf
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NSE Quant Alert Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NSE Quant Alert Agent")
st.markdown("Automated technical analysis pipeline tracking momentum, volume breakouts, and RSI extremes.")

# ---------------- CONFIG ----------------
WATCHLIST = ["RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "TATAMOTORS", "WIPRO", "MARUTI", "KOTAKBANK", "BAJFINANCE", "BHARTIARTL", "TCS", "ITC", "LT", "POWERGRID", "UTKARSHBNK"]
INTERVAL = "5m"
ATR_SL_MULT = 1.5
ATR_TP_MULT = 4.3
VOL_SPIKE_MULT = 3.2

IST = ZoneInfo("Asia/Kolkata")

# ---------------- DATA + INDICATORS ----------------
@st.cache_data(ttl=300)
def fetch(symbol: str) -> pd.DataFrame | None:
    df = yf.download(f"{symbol}.NS", period="5d", interval=INTERVAL,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    df["EMA9"] = ta.ema(df["Close"], length=9)
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["RSI"] = ta.rsi(df["Close"], length=14)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["VWAP"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    df["VOL_AVG"] = df["Volume"].rolling(20).mean()
    return df.dropna()

def evaluate(df: pd.DataFrame) -> dict | None:
    if len(df) < 3:
        return None

    c, p = df.iloc[-2], df.iloc[-3]
    price, atr = float(c["Close"]), float(c["ATR"])
    rsi, rsi_prev = float(c["RSI"]), float(p["RSI"])

    vol_spike = c["Volume"] > VOL_SPIKE_MULT * c["VOL_AVG"]
    above = price > c["VWAP"] and price > c["EMA20"]
    below = price < c["VWAP"] and price < c["EMA20"]

    action, setup, score = None, "", 0

    if vol_spike and above and 55 <= rsi <= 75:
        action, setup = "BUY", "volume_breakout"
        score = 6 + (2 if c["EMA9"] > c["EMA20"] else 0) + (1 if rsi > rsi_prev else 0)
    elif rsi_prev < 30 <= rsi:
        action, setup = "BUY", "oversold_reversal"
        score = 5 + (2 if vol_spike else 0) + (1 if price > c["EMA9"] else 0)
    elif vol_spike and below and 25 <= rsi <= 45:
        action, setup = "SELL", "volume_breakdown"
        score = 6 + (2 if c["EMA9"] < c["EMA20"] else 0) + (1 if rsi < rsi_prev else 0)
    elif rsi_prev > 70 >= rsi:
        action, setup = "SELL", "overbought_reversal"
        score = 5 + (2 if vol_spike else 0) + (1 if price < c["EMA9"] else 0)
    elif rsi < 30 or rsi > 70:
        action, setup, score = "WATCH", "rsi_extreme", 3

    if not action:
        return None

    sign = 1 if action == "BUY" else -1
    sl = price - sign * ATR_SL_MULT * atr
    tp = price + sign * ATR_TP_MULT * atr

    if action == "WATCH":
        sl, tp = price - ATR_SL_MULT * atr, price + ATR_SL_MULT * atr

    return {
        "action": action, "setup": setup, "confidence_score": min(int(score), 10),
        "entry_price": round(price, 2), "stop_loss": round(sl, 2), "target": round(tp, 2),
        "facts": {
            "rsi": round(rsi, 1), "ema9": round(float(c["EMA9"]), 2),
            "ema20": round(float(c["EMA20"]), 2), "vwap": round(float(c["VWAP"]), 2),
            "atr": round(atr, 2),
            "volume_x_avg": round(float(c["Volume"] / c["VOL_AVG"]), 2),
        },
    }

# ---------------- DASHBOARD UI ----------------
if st.button("Run Market Scan"):
    with st.spinner("Fetching data and evaluating watchlist..."):
        now = datetime.now(IST)
        alerts = []
        for sym in WATCHLIST:
            df = fetch(sym)
            sig = evaluate(df) if df is not None else None
            if sig:
                alerts.append({
                    "Symbol": f"NSE:{sym}",
                    "Action": sig["action"],
                    "Confidence": f"{sig['confidence_score']}/10",
                    "Setup": sig["setup"],
                    "Entry": sig["entry_price"],
                    "Stop Loss": sig["stop_loss"],
                    "Target": sig["target"],
                    "RSI": sig["facts"]["rsi"]
                })
        
        if alerts:
            st.success(f"Scan complete. Found {len(alerts)} active setups.")
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
        else:
            st.info("No active signals triggered matching criteria at this moment.")
else:
    st.info("Click **Run Market Scan** above to pull live data from Yahoo Finance and evaluate indicators.")
    
