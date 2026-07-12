import yfinance as yf
import pandas as pd
import os

# ── Settings ──────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "TSLA", "NVDA", "JPM", "JNJ", "SPY"]

SAVE_FOLDER = "../data/raw/"
# ──────────────────────────────────────────────────────────

os.makedirs(SAVE_FOLDER, exist_ok=True)

all_news = []

for ticker in TICKERS:
    print(f"Fetching news for {ticker}...")

    try:
        # Get the stock object
        stock = yf.Ticker(ticker)

        # Fetch news articles
        news = stock.news

        if news:
            for article in news:
                # Safely get content details
                content = article.get("content", {})
                
                # Get title
                title = content.get("title", "")
                
                # Get summary
                summary = content.get("summary", "")
                
                # Get publish date
                pub_date = content.get("pubDate", "")
                
                # Get provider/source
                provider = content.get("provider", {})
                source = provider.get("displayName", "") if provider else ""

                all_news.append({
                    "ticker":   ticker,
                    "datetime": pub_date,
                    "headline": title,
                    "summary":  summary,
                    "source":   source,
                })

            print(f"  Found {len(news)} articles")
        else:
            print(f"  No articles found")

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")

# Save to CSV
df_news = pd.DataFrame(all_news)
save_path = os.path.join(SAVE_FOLDER, "yahoo_news.csv")
df_news.to_csv(save_path, index=False)

print(f"\n✅ Done! Saved {len(df_news)} total articles to {save_path}")
print("\nFirst few rows:")
print(df_news.head())