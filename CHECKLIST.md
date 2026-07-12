# NuroQuant — Complete File Checklist & Run Order

Use this file to track your progress. Tick off each item as you complete it.

---

## PART A — Backend files (copy into nuroquant/backend/)

| # | File | Status | Notes |
|---|------|--------|-------|
| 1 | `fetch_news.py` | ✅ Done | Move from root — no code changes needed |
| 2 | `score_news_sentiment.py` | ☐ | New file — VADER scoring |
| 3 | `build_features.py` | ☐ | New file — merges all data sources |
| 4 | `train_xgboost.py` | ☐ | New file — trains XGBoost × 2 variants |
| 5 | `train_lstm.py` | ☐ | New file — trains LSTM × 2 variants |
| 6 | `copilot_engine.py` | ☐ | Replace your existing one |
| 7 | `main.py` | ☐ | New file — FastAPI server |
| 8 | `requirements.txt` | ☐ | New file — Python dependencies |

---

## PART B — Your existing data files (copy into nuroquant/backend/data/processed/)

| File | Status | Notes |
|------|--------|-------|
| `AAPL_clean.csv` (and all other tickers) | ☐ | Your existing OHLCV files |
| `wsb_sentiment.csv` | ☐ | Your existing WSB sentiment data |

---

## PART C — Run the data pipeline (in order)

```bash
cd nuroquant/backend
```

| # | Command | Status | What it creates |
|---|---------|--------|-----------------|
| 1 | `python fetch_news.py` | ☐ | `data/raw/news/*.parquet` |
| 2 | `python score_news_sentiment.py` | ☐ | `data/processed/gdelt_sentiment.csv` |
| 3 | `python build_features.py` | ☐ | `features_finance.csv` + `features_sentiment.csv` |
| 4 | `python train_xgboost.py` | ☐ | `xgb_finance.pkl` + `xgb_sentiment.pkl` + predictions |
| 5 | `python train_lstm.py` | ☐ | `lstm_finance.pt` + `lstm_sentiment.pt` + predictions |

---

## PART D — Frontend files (copy into nuroquant/frontend/)

| File | Location | Status |
|------|----------|--------|
| `package.json` | `frontend/` | ☐ |
| `vite.config.js` | `frontend/` | ☐ |
| `tailwind.config.js` | `frontend/` | ☐ |
| `postcss.config.js` | `frontend/` | ☐ |
| `index.html` | `frontend/` | ☐ |
| `.env` | `frontend/` | ☐ |
| `src/main.jsx` | `frontend/src/` | ☐ |
| `src/App.jsx` | `frontend/src/` | ☐ |
| `src/index.css` | `frontend/src/` | ☐ |
| `src/hooks/useApi.js` | `frontend/src/hooks/` | ☐ |
| `src/components/Navbar.jsx` | `frontend/src/components/` | ☐ |
| `src/pages/Overview.jsx` | `frontend/src/pages/` | ☐ |
| `src/pages/Compare.jsx` | `frontend/src/pages/` | ☐ |
| `src/pages/Detail.jsx` | `frontend/src/pages/` | ☐ |
| `src/pages/Game.jsx` | `frontend/src/pages/` | ☐ |

---

## PART E — Run locally

### Terminal 1 — Backend
```bash
cd nuroquant/backend
uvicorn main:app --reload --port 8000
```
Check: open http://localhost:8000 — should return JSON status

### Terminal 2 — Frontend
```bash
cd nuroquant/frontend
npm install
npm run dev
```
Check: open http://localhost:3000 — dashboard loads

---

## PART F — Deploy

### Backend → Render
- [ ] Push `backend/` to GitHub
- [ ] Create Web Service on render.com
- [ ] Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] Upload data files to Render persistent disk
- [ ] Note your Render URL: `https://_____________.onrender.com`

### Frontend → Vercel
- [ ] Push `frontend/` to GitHub
- [ ] Import project on vercel.com
- [ ] Set `VITE_API_URL` = your Render URL above
- [ ] Deploy
- [ ] Note your Vercel URL: `https://_____________.vercel.app`

---

## PART G — Presentation demo script

Suggested order to show your lecturer:

1. **Open Compare page** — show the accuracy metrics table first
   - Point out: LSTM+Sentiment has the highest accuracy
   - Point out: Adding sentiment improves every metric
   - Show the sentiment impact delta badges (+X.XX%)

2. **Open Overview page** — show the full ticker table
   - Filter to "Strong Agreement" — show tickers all 4 models agree on
   - Filter to "BUY" signals only

3. **Click a ticker** (e.g. AAPL or MSFT) → Detail page
   - Show the AI summary banner — plain English explanation
   - Show the signal contribution bars (Technical / Sentiment / Model)
   - Show the 4 model signal cards side by side
   - Show the technical indicator cards — explain RSI/MACD to lecturer
   - Show the GDELT news feed at the bottom

4. **Open Game page** — let lecturer try it
   - Set difficulty to Easy
   - Play 3–4 questions together

---

## Auto-generated files (do NOT manually create these)

These are created automatically by the pipeline scripts:

```
backend/data/raw/news/all_news.parquet
backend/data/raw/news/AAPL_news.parquet  (one per ticker)
backend/data/processed/gdelt_sentiment.csv
backend/data/processed/features_finance.csv
backend/data/processed/features_sentiment.csv
backend/data/models/xgb_finance.pkl
backend/data/models/xgb_sentiment.pkl
backend/data/models/lstm_finance.pt
backend/data/models/lstm_sentiment.pt
backend/data/predictions/xgb_finance_predictions.csv
backend/data/predictions/xgb_sentiment_predictions.csv
backend/data/predictions/lstm_finance_predictions.csv
backend/data/predictions/lstm_sentiment_predictions.csv
backend/data/predictions/model_metrics.json
```
