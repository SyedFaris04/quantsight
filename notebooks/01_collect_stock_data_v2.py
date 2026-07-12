import yfinance as yf
import pandas as pd
import os

# ── Settings ──────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "SPY"]

START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

SAVE_FOLDER = "../data/raw/"
# ──────────────────────────────────────────────────────────

os.makedirs(SAVE_FOLDER, exist_ok=True)

for ticker in TICKERS:
    print(f"Downloading {ticker}...")

    # Download data
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)

    # Flatten multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index so Date becomes a proper column
    df.reset_index(inplace=True)

    # Keep only the columns we need
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Save to CSV
    save_path = os.path.join(SAVE_FOLDER, f"{ticker}.csv")
    df.to_csv(save_path, index=False)

    print(f"  Saved {len(df)} rows → {save_path}")

print("\n✅ All stock data downloaded and cleaned!")