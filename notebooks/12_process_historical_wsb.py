import pandas as pd
import re
import os
from transformers import pipeline

# ── Settings ──────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "SPY"]

TICKER_ALIASES = {
    "APPLE":  "AAPL",
    "TESLA":  "TSLA",
    "AMAZON": "AMZN",
    "NVIDIA": "NVDA",
    "GOOGLE": "GOOGL",
    "FACEBOOK": "META",
    "FB":     "META",
}

RAW_FOLDER       = "../data/raw/"
PROCESSED_FOLDER = "../data/processed/"
# ──────────────────────────────────────────────────────────

# ── Load historical WSB data ──────────────────────────────
print("Loading historical WSB data (1M+ posts)...")
df = pd.read_csv(
    os.path.join(RAW_FOLDER, "wsb_historical.csv"),
    low_memory=False
)
print(f"Total posts loaded: {len(df):,}")

# ── Step 1: Fix timestamp ─────────────────────────────────
df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s")
df["date"]      = df["timestamp"].dt.date

# ── Step 2: Filter to our training period only ────────────
# We only need 2015–2021 to fill our training gap
print("\nFiltering to 2015–2021...")
df = df[(df["timestamp"] >= "2015-01-01") &
        (df["timestamp"] <= "2021-12-31")]
print(f"Posts in 2015–2021: {len(df):,}")

# ── Step 3: Remove deleted/empty posts ───────────────────
df = df[df["title"].notna()]
df = df[~df["title"].isin(["[deleted]", "[removed]", ""])]
df = df[df["title"].str.strip() != ""]
print(f"After removing deleted: {len(df):,}")

# ── Step 4: Remove low quality posts ─────────────────────
df = df[df["score"] >= 5]
print(f"After score filter (>=5): {len(df):,}")

# ── Step 5: Find ticker mentions ──────────────────────────
print("\nFinding ticker mentions...")

def find_ticker(text):
    if not isinstance(text, str):
        return None
    text_upper = text.upper()
    for ticker in TICKERS:
        if re.search(rf'\b{ticker}\b|\${ticker}\b', text_upper):
            return ticker
    for alias, ticker in TICKER_ALIASES.items():
        if alias in text_upper:
            return ticker
    return None

df["ticker"] = df["title"].apply(find_ticker)
df = df[df["ticker"].notna()]
print(f"Posts mentioning our tickers: {len(df):,}")

# ── Step 6: Clean text ────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["title_clean"] = df["title"].apply(clean_text)
df = df[df["title_clean"].str.strip() != ""]

# ── Step 7: Remove duplicates ─────────────────────────────
df = df.drop_duplicates(subset=["title_clean"])
print(f"After removing duplicates: {len(df):,}")

# Show date distribution
print("\nPosts per year:")
print(df["timestamp"].dt.year.value_counts().sort_index())

# ── Step 8: Run FinBERT ───────────────────────────────────
print(f"\nRunning FinBERT on {len(df):,} posts...")
print("This may take a few minutes...")

sentiment_pipe = pipeline(
    task="text-classification",
    model="ProsusAI/finbert",
    top_k=None,
    batch_size=32     # process 32 posts at a time = much faster!
)

texts = df["title_clean"].tolist()
results = []

# Process in batches and show progress
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    # Truncate each text to 512 chars
    batch = [t[:512] if isinstance(t, str) else "" for t in batch]

    try:
        outputs = sentiment_pipe(batch)
        for output in outputs:
            scores = {item["label"]: item["score"] for item in output}
            pos    = scores.get("positive", 0.0)
            neg    = scores.get("negative", 0.0)
            neu    = scores.get("neutral",  0.0)
            results.append({
                "pos": round(pos, 4),
                "neg": round(neg, 4),
                "neu": round(neu, 4),
                "sentiment_score": round(pos - neg, 4)
            })
    except Exception as e:
        for _ in batch:
            results.append({"pos": 0.0, "neg": 0.0,
                            "neu": 1.0, "sentiment_score": 0.0})

    # Show progress every 500 posts
    if (i // batch_size) % 5 == 0:
        print(f"  Processed {min(i+batch_size, len(texts)):,} / {len(texts):,} posts")

scores_df = pd.DataFrame(results)
df = df.reset_index(drop=True)
df["sentiment_score"] = scores_df["sentiment_score"].values
df["pos"]             = scores_df["pos"].values
df["neg"]             = scores_df["neg"].values
df["neu"]             = scores_df["neu"].values

# ── Step 9: Keep useful columns ───────────────────────────
df_final = df[[
    "ticker", "date", "timestamp",
    "title", "title_clean",
    "score", "sentiment_score", "pos", "neg", "neu"
]].reset_index(drop=True)

# ── Step 10: Merge with existing WSB sentiment ────────────
print("\nMerging with existing WSB sentiment...")
existing = pd.read_csv(os.path.join(PROCESSED_FOLDER, "wsb_sentiment.csv"))
print(f"Existing posts: {len(existing):,}")

# Combine
common_cols = ["ticker", "date", "timestamp",
               "title_clean", "score", "sentiment_score"]

existing_aligned = existing[[c for c in common_cols if c in existing.columns]]
new_aligned      = df_final[[c for c in common_cols if c in df_final.columns]]

combined = pd.concat([existing_aligned, new_aligned], ignore_index=True)
combined = combined.drop_duplicates(subset=["ticker", "date", "title_clean"])
combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
combined = combined.sort_values("timestamp").reset_index(drop=True)

print(f"Combined posts: {len(combined):,}")

# Save
save_path = os.path.join(PROCESSED_FOLDER, "wsb_sentiment.csv")
combined.to_csv(save_path, index=False)

print(f"\n✅ Historical WSB processing complete!")
print(f"   Before : {len(existing):,} posts")
print(f"   After  : {len(combined):,} posts")
print(f"   Added  : {len(combined) - len(existing):,} new posts")
print(f"\nPosts per year in final dataset:")
combined["year"] = pd.to_datetime(combined["timestamp"],
                                   errors="coerce").dt.year
print(combined["year"].value_counts().sort_index())