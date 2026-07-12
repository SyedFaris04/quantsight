import pandas as pd
import re
import os
from transformers import pipeline

# ── Settings ──────────────────────────────────────────────
# !! Replace with your actual filename !!
FILENAME = "combined_csv_F.csv"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "SPY"]

TICKER_ALIASES = {
    "APPLE":   "AAPL",
    "TESLA":   "TSLA",
    "AMAZON":  "AMZN",
    "NVIDIA":  "NVDA",
    "GOOGLE":  "GOOGL",
    "FACEBOOK":"META",
    "FB":      "META",
}

RAW_FOLDER       = "../data/raw/"
PROCESSED_FOLDER = "../data/processed/"
# ──────────────────────────────────────────────────────────

# ── Load data ─────────────────────────────────────────────
print("Loading 2021 WSB data...")
df = pd.read_csv(
    os.path.join(RAW_FOLDER, FILENAME),
    low_memory=False
)
print(f"Total posts loaded: {len(df):,}")

# ── Step 1: Fix column names ──────────────────────────────
df.columns = df.columns.str.strip()

# ── Step 2: Fix date ──────────────────────────────────────
df["timestamp"] = pd.to_datetime(df["Publish Date"], errors="coerce")
df["date"]      = df["timestamp"].dt.date
df = df.dropna(subset=["timestamp"])

# ── Step 3: Remove low quality posts ─────────────────────
df = df[df["Score"] >= 5]
print(f"After score filter: {len(df):,}")

# ── Step 4: Keep only high quality flairs ────────────────
# DD = Due Diligence (research posts), News, Discussion
# Skip Meme posts — they have no financial signal
quality_flairs = ["DD", "News", "Discussion", "Gain", "Loss",
                  "Technical Analysis", "Fundamentals", "Catalyst"]
if "Flair" in df.columns:
    df_quality = df[df["Flair"].isin(quality_flairs)]
    if len(df_quality) > 100:
        df = df_quality
        print(f"After flair filter: {len(df):,}")
    else:
        print("Flair filter too strict, keeping all flairs")

# ── Step 5: Remove deleted posts ─────────────────────────
df = df[df["Title"].notna()]
df = df[~df["Title"].isin(["[deleted]", "[removed]", ""])]
df = df[df["Title"].str.strip() != ""]
print(f"After removing deleted: {len(df):,}")

# ── Step 6: Find ticker mentions ──────────────────────────
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

df["ticker"] = df["Title"].apply(find_ticker)
df = df[df["ticker"].notna()]
print(f"Posts mentioning our tickers: {len(df):,}")

# ── Step 7: Clean text ────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["title_clean"] = df["Title"].apply(clean_text)
df = df[df["title_clean"].str.strip() != ""]
df = df.drop_duplicates(subset=["title_clean"])
print(f"After cleaning and dedup: {len(df):,}")

# Show distribution
print("\nPosts per month:")
print(df["timestamp"].dt.month.value_counts().sort_index())

# ── Step 8: Run FinBERT ───────────────────────────────────
print(f"\nRunning FinBERT on {len(df):,} posts...")

sentiment_pipe = pipeline(
    task="text-classification",
    model="ProsusAI/finbert",
    top_k=None,
    batch_size=32
)

texts   = df["title_clean"].tolist()
results = []

batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = [t[:512] if isinstance(t, str) else "" 
             for t in texts[i:i+batch_size]]
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
    except:
        for _ in batch:
            results.append({"pos": 0.0, "neg": 0.0,
                            "neu": 1.0, "sentiment_score": 0.0})

    if (i // batch_size) % 5 == 0:
        print(f"  Processed {min(i+batch_size, len(texts)):,} "
              f"/ {len(texts):,} posts")

scores_df             = pd.DataFrame(results)
df                    = df.reset_index(drop=True)
df["sentiment_score"] = scores_df["sentiment_score"].values
df["pos"]             = scores_df["pos"].values
df["neg"]             = scores_df["neg"].values
df["neu"]             = scores_df["neu"].values

# ── Step 9: Keep useful columns ───────────────────────────
df_final = df[[
    "ticker", "date", "timestamp",
    "Title", "title_clean",
    "Score", "sentiment_score", "pos", "neg", "neu"
]].rename(columns={"Title": "title", "Score": "score"})

# ── Step 10: Merge with existing WSB sentiment ────────────
print("\nMerging with existing WSB sentiment...")
existing = pd.read_csv(os.path.join(PROCESSED_FOLDER, "wsb_sentiment.csv"))
print(f"Existing posts: {len(existing):,}")

common_cols      = ["ticker", "date", "timestamp",
                    "title_clean", "score", "sentiment_score"]
existing_aligned = existing[[c for c in common_cols
                              if c in existing.columns]]
new_aligned      = df_final[[c for c in common_cols
                              if c in df_final.columns]]

combined = pd.concat([existing_aligned, new_aligned], ignore_index=True)
combined = combined.drop_duplicates(subset=["ticker", "date", "title_clean"])
combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
combined = combined.sort_values("timestamp").reset_index(drop=True)

save_path = os.path.join(PROCESSED_FOLDER, "wsb_sentiment.csv")
combined.to_csv(save_path, index=False)

print(f"\n✅ 2021 WSB processing complete!")
print(f"   Before : {len(existing):,} posts")
print(f"   After  : {len(combined):,} posts")
print(f"   Added  : {len(combined) - len(existing):,} new posts")
print(f"\nFinal posts per year:")
combined["year"] = pd.to_datetime(combined["timestamp"],
                                   errors="coerce").dt.year
print(combined["year"].value_counts().sort_index())