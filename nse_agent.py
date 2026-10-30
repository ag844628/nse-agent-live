"""NSE Quant Alert Agent
Pipeline: yfinance -> pandas/numpy (native TA) -> deterministic triggers -> Ollama (rationale only) -> Telegram
Setup:
    pip install yfinance pandas requests
    ollama pull llama3
    export TG_TOKEN="123456:ABC..."   # from @BotFather
    export TG_CHAT_ID="your_chat_id"
    python nse_agent.py
Alerts only. No order placement. Not financial advice.
"""

import json
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------- CONFIG ----------------
WATCHLIST = [
    "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "TATAMOTORS",
    "WIPRO", "MARUTI", "KOTAKBANK", "BAJFINANCE", "BHARTIARTL", "TCS",
    "ITC", "LT", "POWERGRID", "UTKARSHBNK"
]
INTERVAL = "5m"
POLL_SECONDS = 300
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

ATR_SL_MULT = 1.5          # stop loss = 1.5 x ATR
ATR_TP_MULT = 4.3          # target    = 4.3 x ATR  (1:2 risk-reward)
VOL_SPIKE_MULT = 3.2       # volume breakout = 3.2x 20-bar average
COOLDOWN_MIN = 40          # no repeat alert per symbol+action

IST = ZoneInfo("Asia/Kolkata")
_last_alert: dict = {}


# ---------------- NATIVE TA INDICATORS ----------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes EMA, RSI, ATR, and VWAP natively using pandas/numpy."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Exponential Moving Averages (EMA 9 & EMA 20)
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI 14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Average True Range (ATR 14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Volume Weighted Average Price (VWAP)
    typical_price = (high + low + close) / 3
    df["VWAP"] = (typical_price * volume).cumsum() / volume.cumsum()

    # Volume 20-bar Rolling Average
    df["VOL_AVG"] = volume.rolling(20).mean()

    return df


# ---------------- DATA FETCHING ----------------
def fetch(symbol: str) -> pd.DataFrame | None:
    df = yf.download(f"{symbol}.NS", period="5d", interval=INTERVAL,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    df = compute_indicators(df)
    return df.dropna()


# ---------------- DETERMINISTIC TRIGGER LOGIC ----------------
def evaluate(df: pd.DataFrame) -> dict | None:
    """Uses last COMPLETED candle (iloc[-2]); iloc[-1] may still be forming."""
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

    if action == "WATCH":  # no directional bias; show both-side levels from ATR anyway
        sl, tp = price - ATR_SL_MULT * atr, price + ATR_TP_MULT * atr

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


# ---------------- OLLAMA (RATIONALE ONLY) ----------------
def get_rationale(symbol: str, sig: dict) -> str:
    prompt = (
        "You are an NSE quant analyst. Using ONLY the verified facts below, write exactly "
        "2 concise sentences of technical justification. Do not invent any number not "
        'listed. Respond as JSON: {"rationale": "..."}\n\n'
        f"Symbol: {symbol}\nAction: {sig['action']}\nSetup: {sig['setup']}\n"
        f"Entry: {sig['entry_price']}\nFacts: {json.dumps(sig['facts'])}"
    )
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "format": "json",
            "stream": False, "options": {"temperature": 0.1},
        }, timeout=60)
        text = json.loads(r.json()["response"]).get("rationale", "").strip()
        if text:
            return text
    except Exception as e:
        print(f"[ollama] fallback: {e}")

    f = sig["facts"]  # deterministic fallback, no LLM
    return (f"{sig['setup'].replace('_', ' ').title()} with RSI {f['rsi']} and volume "
            f"{f['volume_x_avg']}x average. Price vs VWAP {f['vwap']} / EMA20 {f['ema20']} confirms bias.")


# ---------------- TELEGRAM ----------------
def push(alert: dict):
    icon = {"BUY": "🟢", "SELL": "🔴", "WATCH": "🟡"}[alert["action"]]
    msg = (
        f"{icon} *{alert['action']}* `{alert['symbol']}` (conf {alert['confidence_score']}/10)\n"
        f"Entry: {alert['entry_price']}\nSL: {alert['stop_loss']}\nTP: {alert['target']}\n"
        f"_{alert['rationale']}_\n{alert['timestamp']}"
    )
    print(json.dumps(alert, indent=2, ensure_ascii=False))

    if not (TG_TOKEN and TG_CHAT_ID):
        print("[telegram] token/chat id not set, skipping push")
        return

    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
                      timeout=15)
    except Exception as e:
        print(f"[telegram] failed: {e}")


# ---------------- MAIN LOOP ----------------
def market_open(now: datetime) -> bool:
    return now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) <= (15, 30)


def scan():
    now = datetime.now(IST)
    for sym in WATCHLIST:
        try:
            df = fetch(sym)
            sig = evaluate(df) if df is not None else None
            if not sig:
                continue

            key = (sym, sig["action"])
            if key in _last_alert and now - _last_alert[key] < timedelta(minutes=COOLDOWN_MIN):
                continue
            _last_alert[key] = now

            alert = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": f"NSE:{sym}",
                "action": sig["action"],
                "entry_price": sig["entry_price"],
                "stop_loss": sig["stop_loss"],
                "target": sig["target"],
                "confidence_score": sig["confidence_score"],
                "rationale": get_rationale(sym, sig),
            }
            push(alert)
        except Exception as e:
            print(f"[{sym}] error: {e}")


if __name__ == "__main__":
    print("NSE agent started.")
    while True:
        if market_open(datetime.now(IST)):
            scan()
        else:
            print("Market closed, idle.")
        time.sleep(POLL_SECONDS)
