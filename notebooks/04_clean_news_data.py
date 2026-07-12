import pandas as pd
import re
import os

# ── Settings ──────────────────────────────────────────────
RAW_FOLDER       = "../data/raw/"
PROCESSED_FOLDER = "../data/processed/"
# ──────────────────────────────────────────────────────────

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ── Load the news data ────────────────────────────────────
df = pd.read_csv(os.path.join(RAW_FOLDER, "yahoo_news.csv"))
print(f"Raw news loaded: {len(df)} articles")

# ── Step 1: Drop rows with empty headlines ────────────────
df = df.dropna(subset=["headline"])
df = df[df["headline"].str.strip() != ""]
print(f"After removing empty headlines: {len(df)} articles")

# ── Step 2: Remove duplicate headlines ───────────────────
df = df.drop_duplicates(subset=["headline"])
print(f"After removing duplicates: {len(df)} articles")

# ── Step 3: Clean the headline text ──────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                        # lowercase
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove special characters
    text = re.sub(r"\s+", " ", text)           # remove extra spaces
    text = text.strip()
    return text

df["headline_clean"] = df["headline"].apply(clean_text)
df["summary_clean"]  = df["summary"].apply(clean_text)

# ── Step 4: Fix the datetime column ──────────────────────
df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

# Extract just the date (no time) for easier merging later
df["date"] = df["datetime"].dt.date

# ── Step 5: Drop rows where cleaning failed ───────────────
df = df.dropna(subset=["headline_clean"])
df = df[df["headline_clean"].str.strip() != ""]

# ── Step 6: Sort by date ──────────────────────────────────
df = df.sort_values("datetime").reset_index(drop=True)

# ── Save cleaned news ─────────────────────────────────────
save_path = os.path.join(PROCESSED_FOLDER, "yahoo_news_clean.csv")
df.to_csv(save_path, index=False)

print(f"\n✅ Cleaned news saved: {len(df)} articles → {save_path}")
print("\nSample cleaned headlines:")
print(df[["ticker", "date", "headline_clean"]].head(10).to_string())