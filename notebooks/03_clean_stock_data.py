import pandas as pd
import os

# ── Settings ──────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "SPY"]

RAW_FOLDER       = "../data/raw/"
PROCESSED_FOLDER = "../data/processed/"
# ──────────────────────────────────────────────────────────

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

for ticker in TICKERS:
    print(f"Cleaning {ticker}...")

    # Load raw data
    df = pd.read_csv(os.path.join(RAW_FOLDER, f"{ticker}.csv"))

    # ── Step 1: Fix the Date column ──────────────────────
    # Convert Date from text to a real date type
    df["Date"] = pd.to_datetime(df["Date"])

    # ── Step 2: Remove duplicate dates ──────────────────
    before = len(df)
    df = df.drop_duplicates(subset="Date")
    after = len(df)
    if before != after:
        print(f"  Removed {before - after} duplicate rows")

    # ── Step 3: Remove rows with missing values ──────────
    df = df.dropna()

    # ── Step 4: Remove rows where price is zero ──────────
    # A stock price of 0 is clearly wrong data
    df = df[(df["Open"] > 0) & (df["Close"] > 0) &
            (df["High"] > 0) & (df["Low"]  > 0) &
            (df["Volume"] > 0)]

    # ── Step 5: Sort by date (oldest first) ─────────────
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Step 6: Add a Ticker column ──────────────────────
    # Useful later when we combine all stocks into one file
    df["Ticker"] = ticker

    # Save cleaned file
    save_path = os.path.join(PROCESSED_FOLDER, f"{ticker}_clean.csv")
    df.to_csv(save_path, index=False)

    print(f"  ✅ {len(df)} rows saved → {save_path}")

print("\n🎉 All stock data cleaned!")