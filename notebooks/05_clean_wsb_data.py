import pandas as pd
import re
import os

# ── Settings ──────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "SPY"]

# Common ways each ticker is mentioned on Reddit
# e.g. people write $AAPL or just AAPL
TICKER_PATTERNS = [f"\\b{t}\\b|\\${t}\\b" for t in TICKERS]

RAW_FOLDER       = "../data/raw/"
PROCESSED_FOLDER = "../data/processed/"
# ──────────────────────────────────────────────────────────

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ── Load raw WSB data ─────────────────────────────────────
df = pd.read_csv(os.path.join(RAW_FOLDER, "reddit_wsb.csv"))
print(f"Raw WSB posts loaded: {len(df)} posts")

# ── Step 1: Remove deleted/removed posts ──────────────────
deleted_words = ["[deleted]", "[removed]", ""]
df = df[~df["title"].isin(deleted_words)]
df = df[df["title"].notna()]
print(f"After removing deleted posts: {len(df)} posts")

# ── Step 2: Remove posts with no title ───────────────────
df = df[df["title"].str.strip() != ""]
print(f"After removing empty titles: {len(df)} posts")

# ── Step 3: Remove low engagement posts ──────────────────
# Posts with very low score are likely spam or ignored
df = df[df["score"] >= 2]
print(f"After removing low score posts: {len(df)} posts")

# ── Step 4: Find which ticker each post mentions ──────────
def find_ticker(text):
    if not isinstance(text, str):
        return None
    text_upper = text.upper()
    for ticker in TICKERS:
        pattern = f"\\b{ticker}\\b|\\${ticker}\\b"
        if re.search(pattern, text_upper):
            return ticker
    return None

# Check both title and body for ticker mentions
df["ticker_found"] = df["title"].apply(find_ticker)

# If not found in title, check body
mask = df["ticker_found"].isna()
df.loc[mask, "ticker_found"] = df.loc[mask, "body"].apply(find_ticker)

# Keep only posts that mention at least one of our tickers
df = df[df["ticker_found"].notna()]
print(f"After filtering for ticker mentions: {len(df)} posts")

# ── Step 5: Clean the text ────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)         # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove special characters
    text = re.sub(r"\s+", " ", text)            # remove extra spaces
    text = text.strip()
    return text

df["title_clean"] = df["title"].apply(clean_text)
df["body_clean"]  = df["body"].apply(clean_text)

# ── Step 6: Fix timestamp ─────────────────────────────────
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["date"]      = df["timestamp"].dt.date

# ── Step 7: Keep only useful columns ─────────────────────
df = df[[
    "ticker_found", "date", "timestamp",
    "title", "title_clean",
    "body_clean", "score", "comms_num"
]].rename(columns={"ticker_found": "ticker"})

# ── Step 8: Sort by date ──────────────────────────────────
df = df.sort_values("timestamp").reset_index(drop=True)

# ── Save cleaned WSB data ─────────────────────────────────
save_path = os.path.join(PROCESSED_FOLDER, "wsb_clean.csv")
df.to_csv(save_path, index=False)

print(f"\n✅ Cleaned WSB data saved: {len(df)} posts → {save_path}")
print("\nSample posts:")
print(df[["ticker", "date", "title_clean", "score"]].head(8).to_string())