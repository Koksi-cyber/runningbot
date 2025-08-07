import os
import joblib
import pandas as pd
import requests
from dataclasses import dataclass
from typing import Optional, List
from indicators import ema, rsi, candle_body_strength

BINANCE_URL = "https://fapi.binance.com"

# === Helper to fetch candles ===
def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = f"{BINANCE_URL}/fapi/v1/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    candles = [{
        "time": pd.to_datetime(k[0], unit="ms"),
        "open": float(k[1]),
        "high": float(k[2]),
        "low": float(k[3]),
        "close": float(k[4])
    } for k in data]
    return pd.DataFrame(candles)

# === Data container for signals ===
@dataclass
class TradeSignal:
    timestamp: pd.Timestamp
    direction: str
    probability: float
    entry_price: float
    stop_loss: float
    take_profit: float

# === Core TradeBot class ===
class TradeBot:
    def __init__(self, symbol="BTCUSDT", threshold=0.66):
        self.symbol = symbol
        self.threshold = threshold
        self.model = self.load_model()

        # Internal candles
        self.candles_1m = pd.DataFrame()
        self.candles_5m = pd.DataFrame()
        self.candles_1h = pd.DataFrame()
        self.candles_4h = pd.DataFrame()

    def load_model(self):
        model_path = os.path.join("models", "long_model_15R.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        return joblib.load(model_path)

    def refresh_candles(self):
        self.candles_1m = fetch_klines(self.symbol, "1m", 500)
        self.candles_5m = fetch_klines(self.symbol, "5m", 500)
        self.candles_1h = fetch_klines(self.symbol, "1h", 500)
        self.candles_4h = fetch_klines(self.symbol, "4h", 500)

        # Indicators
        self.candles_5m["ema20"] = ema(self.candles_5m["close"], 20)
        self.candles_5m["ema50"] = ema(self.candles_5m["close"], 50)
        self.candles_5m["rsi"] = rsi(self.candles_5m["close"], 14)
        self.candles_1h["ema50"] = ema(self.candles_1h["close"], 50)
        self.candles_4h["ema50"] = ema(self.candles_4h["close"], 50)

    def compute_features(self) -> Optional[List[float]]:
        if self.candles_1m.empty or self.candles_5m.empty or self.candles_1h.empty or self.candles_4h.empty:
            return None

        latest_1m = self.candles_1m.iloc[-1]
        t = latest_1m["time"]

        df5 = self.candles_5m[self.candles_5m["time"] <= t]
        df1h = self.candles_1h[self.candles_1h["time"] <= t]
        df4h = self.candles_4h[self.candles_4h["time"] <= t]
        if df5.empty or df1h.empty or df4h.empty:
            return None

        row5 = df5.iloc[-1]
        row1h = df1h.iloc[-1]
        row4h = df4h.iloc[-1]

        # Trend + momentum filters
        if row4h["close"] <= row4h["ema50"] or row1h["close"] <= row1h["ema50"]:
            return None
        if pd.isna(row5["ema20"]) or pd.isna(row5["ema50"]) or pd.isna(row5["rsi"]):
            return None
        if row5["ema20"] <= row5["ema50"] or row5["rsi"] >= 55:
            return None

        body_strength = candle_body_strength(
            pd.Series([latest_1m["open"]]),
            pd.Series([latest_1m["close"]]),
            pd.Series([latest_1m["high"]]),
            pd.Series([latest_1m["low"]])
        ).iloc[0]

        features = [
            row5["rsi"],
            row5["ema20"] - row5["ema50"],
            (row1h["close"] - row1h["ema50"]) / row1h["ema50"],
            (row4h["close"] - row4h["ema50"]) / row4h["ema50"],
            (latest_1m["close"] - row5["ema50"]) / row5["ema50"],
            body_strength
        ]
        return features

    def generate_signal(self) -> Optional[TradeSignal]:
        features = self.compute_features()
        if features is None:
            return None

        prob = float(self.model.predict_proba([features])[0][1])
        if prob < self.threshold:
            return None

        latest = self.candles_1m.iloc[-1]
        prev = self.candles_1m.iloc[-2] if len(self.candles_1m) > 1 else latest

        entry = latest["close"]
        stop = prev["low"]
        risk = entry - stop
        if risk <= 0:
            return None

        take = entry + 1.5 * risk

        return TradeSignal(
            timestamp=latest["time"],
            direction="LONG",
            probability=prob,
            entry_price=entry,
            stop_loss=stop,
            take_profit=take
        )
