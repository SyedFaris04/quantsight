import pandas as pd
import os

# ── Settings ──────────────────────────────────────────────
PROCESSED_FOLDER = "../data/processed/"
# ──────────────────────────────────────────────────────────

# ── Load all our data ─────────────────────────────────────
print("Loading data...")
stock_df = pd.read_csv(os.path.join(PROCESSED_FOLDER, "features_stock.csv"))
news_df  = pd.read_csv(os.path.join(PROCESSED_FOLDER, "yahoo_news_sentiment.csv"))
wsb_df   = pd.read_csv(os.path.join(PROCESSED_FOLDER, "wsb_sentiment.csv"))

# Fix date columns
stock_df["Date"] = pd.to_datetime(stock_df["Date"]).dt.date
news_df["date"]  = pd.to_datetime(news_df["date"],  errors="coerce").dt.date
wsb_df["date"]   = pd.to_datetime(wsb_df["date"],   errors="coerce").dt.date

print(f"Stock features : {len(stock_df)} rows")
print(f"News articles  : {len(news_df)} rows")
print(f"WSB posts      : {len(wsb_df)} rows")

# ── Aggregate news sentiment per ticker per day ───────────
# Multiple articles on same day → take the average sentiment
print("\nAggregating news sentiment per ticker per day...")
news_daily = news_df.groupby(["ticker", "date"]).agg(
    news_sentiment  = ("sentiment_score", "mean"),
    news_count      = ("sentiment_score", "count")
).reset_index()
news_daily.columns = ["Ticker", "Date", "news_sentiment", "news_count"]

# ── Aggregate WSB sentiment per ticker per day ────────────
print("Aggregating WSB sentiment per ticker per day...")
wsb_daily = wsb_df.groupby(["ticker", "date"]).agg(
    wsb_sentiment     = ("sentiment_score", "mean"),
    wsb_count         = ("sentiment_score", "count"),
    wsb_avg_score     = ("score", "mean")
).reset_index()
wsb_daily.columns = ["Ticker", "Date",
                      "wsb_sentiment", "wsb_count", "wsb_avg_score"]

# ── Merge into stock features ─────────────────────────────
print("\nMerging sentiment into stock features...")

# Left join — keep all stock rows, add sentiment where available
df = stock_df.merge(news_daily, on=["Ticker", "Date"], how="left")
df = df.merge(wsb_daily,  on=["Ticker", "Date"], how="left")

# Fill missing sentiment with 0 (neutral)
# If no news/posts that day → assume neutral sentiment
df["news_sentiment"] = df["news_sentiment"].fillna(0.0)
df["news_count"]     = df["news_count"].fillna(0)
df["wsb_sentiment"]  = df["wsb_sentiment"].fillna(0.0)
df["wsb_count"]      = df["wsb_count"].fillna(0)
df["wsb_avg_score"]  = df["wsb_avg_score"].fillna(0.0)

# ── Create combined sentiment score ──────────────────────
# Simple average of news and WSB sentiment
df["combined_sentiment"] = (df["news_sentiment"] + df["wsb_sentiment"]) / 2

# Save final dataset
save_path = os.path.join(PROCESSED_FOLDER, "final_dataset.csv")
df.to_csv(save_path, index=False)

print(f"\n✅ Final dataset saved!")
print(f"   Total rows     : {len(df)}")
print(f"   Total features : {len(df.columns)}")
print(f"\nFinal columns:")
print(df.columns.tolist())
print(f"\nSample of sentiment columns:")
print(df[["Date", "Ticker", "Close", "news_sentiment",
          "wsb_sentiment", "combined_sentiment", "Target"]].head(10).to_string())