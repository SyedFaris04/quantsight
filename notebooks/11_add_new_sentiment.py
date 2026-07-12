import pandas as pd
import re
import os
from transformers import pipeline

# ── Settings ──────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "SPY"]

# Informal names people use on Reddit
TICKER_ALIASES = {
    "APPLE":  "AAPL",
    "TESLA":  "TSLA",
    "AMAZON": "AMZN",
    "NVIDIA": "NVDA",
    "GOOGLE": "GOOGL",
    "META":   "META",
}

RAW_FOLDER       = "../data/raw/"
PROCESSED_FOLDER = "../data/processed/"
# ──────────────────────────────────────────────────────────

# ── Load new WSB file ─────────────────────────────────────
print("Loading new WSB August 2021 data...")
df = pd.read_csv(os.path.join(RAW_FOLDER, "wsb_aug2021.csv"))
print(f"Raw posts loaded: {len(df)}")

# ── Step 1: Remove deleted/empty posts ───────────────────
df = df[df["title"].notna()]
df = df[~df["title"].isin(["[deleted]", "[removed]", ""])]
df = df[df["title"].str.strip() != ""]
print(f"After removing empty/deleted: {len(df)}")

# ── Step 2: Fix timestamp ─────────────────────────────────
df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s")
df["date"]      = df["timestamp"].dt.date

# ── Step 3: Find ticker mentions ──────────────────────────
def find_ticker(text):
    if not isinstance(text, str):
        return None
    text_upper = text.upper()

    # Check direct ticker symbols first
    for ticker in TICKERS:
        if re.search(rf'\b{ticker}\b|\${ticker}\b', text_upper):
            return ticker

    # Check informal names
    for alias, ticker in TICKER_ALIASES.items():
        if alias in text_upper:
            return ticker

    return None

df["ticker"] = df["title"].apply(find_ticker)

# Also check selftext if ticker not found in title
mask = df["ticker"].isna()
df.loc[mask, "ticker"] = df.loc[mask, "selftext"].apply(find_ticker)

# Keep only posts mentioning our tickers
df = df[df["ticker"].notna()]
print(f"After filtering for ticker mentions: {len(df)}")

# ── Step 4: Clean text ────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["title_clean"] = df["title"].apply(clean_text)

# ── Step 5: Remove low quality posts ─────────────────────
df = df[df["score"] >= 2]
print(f"After removing low score posts: {len(df)}")

# ── Step 6: Run FinBERT sentiment ─────────────────────────
print("\nLoading FinBERT...")
sentiment_pipeline = pipeline(
    task="text-classification",
    model="ProsusAI/finbert",
    top_k=None
)
print("✅ FinBERT loaded!")

print("Scoring sentiment...")
results = []
for text in df["title_clean"].tolist():
    if not isinstance(text, str) or text.strip() == "":
        results.append({"pos": 0.0, "neg": 0.0,
                        "neu": 1.0, "sentiment_score": 0.0})
        continue
    try:
        output  = sentiment_pipeline(text[:512])[0]
        scores  = {item["label"]: item["score"] for item in output}
        pos     = scores.get("positive", 0.0)
        neg     = scores.get("negative", 0.0)
        neu     = scores.get("neutral",  0.0)
        results.append({
            "pos": round(pos, 4),
            "neg": round(neg, 4),
            "neu": round(neu, 4),
            "sentiment_score": round(pos - neg, 4)
        })
    except:
        results.append({"pos": 0.0, "neg": 0.0,
                        "neu": 1.0, "sentiment_score": 0.0})

scores_df = pd.DataFrame(results)
df["pos"]             = scores_df["pos"].values
df["neg"]             = scores_df["neg"].values
df["neu"]             = scores_df["neu"].values
df["sentiment_score"] = scores_df["sentiment_score"].values

# ── Step 7: Keep only useful columns ─────────────────────
df_clean = df[[
    "ticker", "date", "timestamp",
    "title", "title_clean",
    "score", "sentiment_score", "pos", "neg", "neu"
]].reset_index(drop=True)

# ── Step 8: Merge with existing WSB sentiment ─────────────
print("\nMerging with existing WSB sentiment data...")
existing = pd.read_csv(os.path.join(PROCESSED_FOLDER, "wsb_sentiment.csv"))
print(f"Existing WSB posts: {len(existing)}")

# Align columns
df_clean = df_clean.rename(columns={"title": "title_original"})

# Make sure both have same columns before combining
common_cols = ["ticker", "date", "timestamp",
               "title_clean", "score", "sentiment_score"]
existing_aligned = existing[
    [c for c in common_cols if c in existing.columns]
]
new_aligned = df_clean[
    [c for c in common_cols if c in df_clean.columns]
]

combined = pd.concat([existing_aligned, new_aligned], ignore_index=True)

# Remove duplicates
combined = combined.drop_duplicates(subset=["ticker", "date", "title_clean"])
combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
combined = combined.sort_values("timestamp").reset_index(drop=True)

print(f"New combined WSB posts: {len(combined)}")

# Save
save_path = os.path.join(PROCESSED_FOLDER, "wsb_sentiment.csv")
combined.to_csv(save_path, index=False)

print(f"\n✅ Done! WSB sentiment updated → {save_path}")
print(f"   Before : {len(existing)} posts")
print(f"   After  : {len(combined)} posts")
print(f"   Added  : {len(combined) - len(existing)} new posts")