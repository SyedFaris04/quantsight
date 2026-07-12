import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os

# ── Settings ──────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "JNJ", "SPY",
    "AMD", "INTC", "CRM", "ORCL", "ADBE",
    "NFLX", "PYPL", "UBER", "SNAP", "TWTR",
    "BAC", "GS", "MS", "WFC", "C",
    "PFE", "MRNA", "ABBV", "UNH", "CVS",
    "WMT", "TGT", "COST", "NKE", "MCD",
    "XOM", "CVX", "COP", "SLB", "OXY",
    "QQQ", "DIA", "IWM", "GLD", "TLT"
]

START_DATE       = "2015-01-01"
END_DATE         = "2024-12-31"
RAW_FOLDER       = "../data/raw/"
PROCESSED_FOLDER = "../data/processed/"

os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
# ──────────────────────────────────────────────────────────

all_stocks = []
failed     = []

for ticker in TICKERS:
    print(f"Processing {ticker}...")

    try:
        # ── Download ──────────────────────────────────────
        df = yf.download(ticker, start=START_DATE,
                         end=END_DATE, auto_adjust=True,
                         progress=False)

        if len(df) < 100:
            print(f"  ⚠️ Skipping {ticker} — not enough data")
            failed.append(ticker)
            continue

        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        # ── Clean ─────────────────────────────────────────
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates(subset="Date")
        df = df.dropna()
        df = df[(df["Open"] > 0) & (df["Close"] > 0) &
                (df["High"] > 0) & (df["Low"]   > 0) &
                (df["Volume"] > 0)]
        df = df.sort_values("Date").reset_index(drop=True)
        df["Ticker"] = ticker

        # ── Feature engineering ───────────────────────────
        df["SMA_5"]  = ta.sma(df["Close"], length=5)
        df["SMA_10"] = ta.sma(df["Close"], length=10)
        df["SMA_20"] = ta.sma(df["Close"], length=20)
        df["EMA_12"] = ta.ema(df["Close"], length=12)
        df["EMA_26"] = ta.ema(df["Close"], length=26)

        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        df["MACD"]        = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        df["MACD_hist"]   = macd["MACDh_12_26_9"]

        df["RSI"]       = ta.rsi(df["Close"], length=14)
        df["Return_5"]  = df["Close"].pct_change(5)
        df["Return_10"] = df["Close"].pct_change(10)

        bbands = ta.bbands(df["Close"], length=20)
        df["BB_upper"] = bbands["BBU_20_2.0_2.0"]
        df["BB_lower"] = bbands["BBL_20_2.0_2.0"]
        df["BB_mid"]   = bbands["BBM_20_2.0_2.0"]
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]

        df["Volatility_20"] = df["Close"].rolling(20).std()
        df["ATR"]           = ta.atr(df["High"], df["Low"],
                                     df["Close"], length=14)
        df["Volume_change"] = df["Volume"].pct_change()
        df["Volume_MA_10"]  = df["Volume"].rolling(10).mean()

        # Target variable
        df["Forward_return"] = df["Close"].pct_change(5).shift(-5)
        df["Target"]         = (df["Forward_return"] > 0).astype(int)

        all_stocks.append(df)
        print(f"  ✅ {len(df)} rows, {len(df.columns)} features")

    except Exception as e:
        print(f"  ❌ Error with {ticker}: {e}")
        failed.append(ticker)

# ── Combine all stocks ────────────────────────────────────
print(f"\nCombining {len(all_stocks)} stocks...")
combined = pd.concat(all_stocks, ignore_index=True)

before = len(combined)
combined = combined.dropna()
after   = len(combined)
print(f"Removed {before - after:,} rows with NaN (normal!)")

# Save raw features without sentiment first
save_path = os.path.join(PROCESSED_FOLDER, "features_stock.csv")
combined.to_csv(save_path, index=False)

print(f"\n✅ Stock expansion complete!")
print(f"   Stocks processed : {len(all_stocks)}")
print(f"   Stocks failed    : {len(failed)} {failed}")
print(f"   Total rows       : {len(combined):,}")
print(f"   Total features   : {len(combined.columns)}")