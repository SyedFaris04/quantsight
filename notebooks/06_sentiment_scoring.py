from transformers import pipeline
import pandas as pd
import os

# ── Settings ──────────────────────────────────────────────
PROCESSED_FOLDER = "../data/processed/"
# ──────────────────────────────────────────────────────────

# ── Step 1: Load FinBERT model ────────────────────────────
print("Loading FinBERT model (first time may take a few minutes)...")

sentiment_pipeline = pipeline(
    task="text-classification",
    model="ProsusAI/finbert",
    top_k=None      # return all 3 scores (positive, negative, neutral)
)

print("✅ FinBERT loaded successfully!")

# ── Helper function ───────────────────────────────────────
def score_texts(texts):
    """
    Takes a list of texts and returns a DataFrame with:
    - positive score
    - negative score  
    - neutral score
    - sentiment score (positive - negative)
    """
    results = []

    for text in texts:
        # Skip empty texts
        if not isinstance(text, str) or text.strip() == "":
            results.append({
                "pos": 0.0, "neg": 0.0,
                "neu": 1.0, "sentiment_score": 0.0
            })
            continue

        # Truncate long texts (FinBERT has a 512 token limit)
        text = text[:512]

        try:
            output = sentiment_pipeline(text)[0]

            # Convert list of dicts to a simple dict
            scores = {item["label"]: item["score"] for item in output}

            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral",  0.0)

            results.append({
                "pos": round(pos, 4),
                "neg": round(neg, 4),
                "neu": round(neu, 4),
                "sentiment_score": round(pos - neg, 4)
            })

        except Exception as e:
            print(f"  Error scoring text: {e}")
            results.append({
                "pos": 0.0, "neg": 0.0,
                "neu": 1.0, "sentiment_score": 0.0
            })

    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════
# PART A — Score Yahoo Finance News
# ════════════════════════════════════════════════════════════
print("\n── Scoring Yahoo Finance news ──────────────────────")

news_df = pd.read_csv(os.path.join(PROCESSED_FOLDER, "yahoo_news_clean.csv"))
print(f"Loaded {len(news_df)} news articles")

# Score the cleaned headlines
news_scores = score_texts(news_df["headline_clean"].tolist())

# Add scores back to the dataframe
news_df["pos"]             = news_scores["pos"].values
news_df["neg"]             = news_scores["neg"].values
news_df["neu"]             = news_scores["neu"].values
news_df["sentiment_score"] = news_scores["sentiment_score"].values

# Save
save_path = os.path.join(PROCESSED_FOLDER, "yahoo_news_sentiment.csv")
news_df.to_csv(save_path, index=False)
print(f"✅ News sentiment saved → {save_path}")
print(news_df[["ticker", "headline_clean", "sentiment_score"]].head(5).to_string())


# ════════════════════════════════════════════════════════════
# PART B — Score WSB Posts
# ════════════════════════════════════════════════════════════
print("\n── Scoring WSB Reddit posts ─────────────────────────")

wsb_df = pd.read_csv(os.path.join(PROCESSED_FOLDER, "wsb_clean.csv"))
print(f"Loaded {len(wsb_df)} WSB posts")

# Score the cleaned titles
wsb_scores = score_texts(wsb_df["title_clean"].tolist())

# Add scores back
wsb_df["pos"]             = wsb_scores["pos"].values
wsb_df["neg"]             = wsb_scores["neg"].values
wsb_df["neu"]             = wsb_scores["neu"].values
wsb_df["sentiment_score"] = wsb_scores["sentiment_score"].values

# Save
save_path = os.path.join(PROCESSED_FOLDER, "wsb_sentiment.csv")
wsb_df.to_csv(save_path, index=False)
print(f"✅ WSB sentiment saved → {save_path}")
print(wsb_df[["ticker", "title_clean", "sentiment_score"]].head(5).to_string())

print("\n🎉 All sentiment scoring complete!")