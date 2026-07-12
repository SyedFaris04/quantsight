import pandas as pd
import pandas_ta as ta
import numpy as np
import os

# ── Settings ──────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "SPY"]

PROCESSED_FOLDER = "../data/processed/"
# ──────────────────────────────────────────────────────────

all_stocks = []

for ticker in TICKERS:
    print(f"Engineering features for {ticker}...")

    # Load cleaned stock data
    df = pd.read_csv(os.path.join(PROCESSED_FOLDER, f"{ticker}_clean.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ── TREND INDICATORS ──────────────────────────────────
    df["SMA_5"]  = ta.sma(df["Close"], length=5)
    df["SMA_10"] = ta.sma(df["Close"], length=10)
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["EMA_12"] = ta.ema(df["Close"], length=12)
    df["EMA_26"] = ta.ema(df["Close"], length=26)

    # MACD
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"]        = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACD_hist"]   = macd["MACDh_12_26_9"]

    # ── MOMENTUM INDICATORS ───────────────────────────────
    df["RSI"]       = ta.rsi(df["Close"], length=14)
    df["Return_5"]  = df["Close"].pct_change(5)
    df["Return_10"] = df["Close"].pct_change(10)

    # ── VOLATILITY INDICATORS ─────────────────────────────
    # Bollinger Bands with correct column names
    bbands = ta.bbands(df["Close"], length=20)
    df["BB_upper"] = bbands["BBU_20_2.0_2.0"]
    df["BB_lower"] = bbands["BBL_20_2.0_2.0"]
    df["BB_mid"]   = bbands["BBM_20_2.0_2.0"]
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]

    df["Volatility_20"] = df["Close"].rolling(20).std()
    df["ATR"]           = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # ── VOLUME INDICATORS ─────────────────────────────────
    df["Volume_change"] = df["Volume"].pct_change()
    df["Volume_MA_10"]  = df["Volume"].rolling(10).mean()

    # ── TARGET VARIABLE ───────────────────────────────────
    df["Forward_return"] = df["Close"].pct_change(5).shift(-5)
    df["Target"]         = (df["Forward_return"] > 0).astype(int)

    all_stocks.append(df)
    print(f"  ✅ {len(df)} rows, {len(df.columns)} features")

# ── Combine all stocks ────────────────────────────────────
print("\nCombining all stocks...")
combined = pd.concat(all_stocks, ignore_index=True)

before = len(combined)
combined = combined.dropna()
after = len(combined)
print(f"Removed {before - after} rows with NaN values (normal!)")

# Save
save_path = os.path.join(PROCESSED_FOLDER, "features_stock.csv")
combined.to_csv(save_path, index=False)

print(f"\n✅ Feature engineering complete!")
print(f"   Total rows     : {len(combined)}")
print(f"   Total features : {len(combined.columns)}")
print(f"\nFeature columns:")
print(combined.columns.tolist())