# QuantSight — How to Run

This project lives in `quantsightv2/`. Two servers must run **at the same time**:
a Python backend (FastAPI) and a React frontend (Vite). All the data, trained
models, and predictions are already built and committed to `backend/data/` —
so on a normal day you only need **Quick Start** below. The full pipeline
further down is only for when you need to regenerate data/models from scratch
(new data, retraining, etc.).

```
quantsightv2/
├── .github/workflows/
│   └── daily-predictions.yml    ← scheduled trigger for the live prediction tracker
├── backend/
│   ├── main.py              ← FastAPI app (the API server)
│   ├── copilot_engine.py    ← explanation/XAI engine used by /explain
│   ├── chatbot_engine.py    ← AI Assistant — Groq + tool-calling (used by /chat)
│   ├── prediction_tracker.py ← live prediction log + daily resolution job
│   ├── live_signals.py      ← live (Yahoo Finance) signal generation
│   ├── build_features.py    ← merges price + sentiment + emotion into training data
│   ├── train_xgboost.py     ← trains XGBoost (Finance / Finance+Sentiment)
│   ├── train_lstm.py        ← trains LSTM+Attention (Finance / Finance+Sentiment)
│   ├── extract_lstm_attention.py  ← saves LSTM's per-day attention weights
│   ├── score_wsb_emotion.py ← GoEmotions emotion scoring over WSB posts
│   ├── score_news_sentiment_finbert.py
│   ├── requirements.txt
│   ├── .env                  ← GROQ_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, ADMIN_JOB_SECRET (all optional, gitignored)
│   ├── .venv/                ← Python virtual environment (already created)
│   └── data/
│       ├── raw/               ← original stock/news/WSB CSVs
│       ├── processed/         ← features_finance.csv, features_sentiment.csv, ...
│       ├── models/            ← xgb_*.pkl, lstm_*.pt (already trained)
│       └── predictions/       ← *_predictions.csv, model_metrics.json, ...
├── frontend/
│   ├── package.json
│   ├── .env                  ← VITE_API_URL, VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY
│   └── src/
│       ├── context/AuthContext.jsx   ← Supabase auth state (user is null in guest mode)
│       ├── lib/supabaseClient.js     ← shared Supabase client
│       ├── lib/gameFx.js             ← Game page sound/confetti/prefs
│       └── components/ChatWidget.jsx ← floating AI Assistant, on every page
└── notebooks/                ← numbered one-off scripts (01–19), data pipeline history
```

---

## Quick Start — run it today

Data and models already exist on disk, so this is all you need. Two terminals.

### Terminal 1 — backend

```bash
cd backend
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\activate           # Windows cmd/PowerShell — use this line instead
# source .venv/bin/activate        # Mac/Linux

uvicorn main:app --reload --port 8000
```

Expected tail of the output:
```
INFO | Ready — 44 tickers available
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Check it's alive: open **http://127.0.0.1:8000/docs** — you should see the
Swagger UI listing every endpoint (`/explain/{ticker}`, `/overview`,
`/dashboard`, `/compare/{ticker}`, `/history/{ticker}`, `/stock/{ticker}`,
`/news/{ticker}`, `/metrics`, `/confidence-boost`, `/market-sentiment`,
`/tickers`...).

### Terminal 2 — frontend

```bash
cd frontend
npm install       # first time only, or after pulling new dependencies
npm run dev
```

Expected output:
```
VITE ready
➜  Local:   http://localhost:3000/
```

### Open it

**http://localhost:3000** — that's the whole app. Both terminals must stay
running while you use it. Navigation is a left sidebar (Dashboard / Market /
AI Compare / Leaderboard / Track Record / Portfolio / Game / Settings), plus
a floating AI Assistant button (bottom-right) on every page.

| Page | URL | What it shows |
|---|---|---|
| Dashboard | `/` | KPI summary, Top Opportunities, real-time Market Sentiment (VADER over current news), AI Insight summary, Market News |
| Market | `/market` | All 44 tickers as cards (with company names), all 4 model signals each, filters, Live Mode |
| Detail | `/detail/AAPL` (any ticker) | Real OHLC candlestick chart + volume, radial confidence gauge, AI Copilot explanation, tabs: **Overview / Technical / Sentiment & Emotion / History / Learn** — includes the LSTM attention chart and a per-ticker Sentiment Impact comparison |
| AI Compare | `/compare` | Accuracy table + charts across all 4 model variants, sentiment+emotion impact, Confidence Boost KPI |
| Leaderboard | `/leaderboard` | The 4 *model* variants ranked by accuracy — not the Game's player leaderboard, see below |
| Track Record | `/track-record` | Live, forward-looking prediction accuracy — needs the optional Supabase/GitHub Actions setup below to show real data, otherwise a clean empty state |
| Portfolio | `/portfolio` | Your tracked holdings (localStorage in guest mode, Supabase if signed in) + live prices + BUY suggestions |
| Game | `/game` | Prediction quiz — score/streak/level, sound/confetti/celebrations, a real cross-user leaderboard (needs sign-in to appear on it, viewable by anyone) |
| Login | `/login` | Optional — Google or email/password |
| Settings | `/settings` | Account, game preferences, accessibility, CSV export |

### Optional — sign-in, AI Assistant, live tracker

Everything above works with zero extra setup. Three features are additive
and need their own `.env` values, each degrading gracefully (clear message,
never a crash) if skipped:

| Feature | Needs | Where |
|---|---|---|
| Sign-in (Google/email) + synced Portfolio/Game | Supabase project | `frontend/.env`: `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` |
| AI Assistant (chat widget) | Free Groq key | `backend/.env`: `GROQ_API_KEY` |
| Live Prediction Track Record | Supabase service role key + admin secret + a scheduled trigger | `backend/.env`: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `ADMIN_JOB_SECRET` |

`backend/.env.example` documents every backend variable. See
[README.md → Deployment](README.md#deployment) for the full Supabase/Groq/
GitHub Actions setup, or **[CHECKLIST.md](CHECKLIST.md)** for the literal
click-by-click version.

If a page shows "Loading…" forever or an error banner, check Terminal 1 for a
Python traceback first — the frontend never fabricates data, so an empty/error
state almost always means the backend request failed.

---

## Full Pipeline — rebuilding data/models from scratch

Only needed if you're regenerating everything (new tickers, new date range,
retraining). Run in this exact order — each script's docstring says what it
needs and what runs after it.

**Data collection & cleaning** (one-time, already done — re-run only if you
need fresh raw data):
```
notebooks/01_collect_stock_data_v2.py
notebooks/02_collect_news_yahoo.py
notebooks/03_clean_stock_data.py
notebooks/04_clean_news_data.py
notebooks/05_clean_wsb_data.py
notebooks/11_add_new_sentiment.py
notebooks/12_process_historical_wsb.py
notebooks/13_process_wsb_2021.py
notebooks/14_expand_stocks.py
```

**Sentiment + emotion scoring:**
```bash
cd backend
python score_news_sentiment_finbert.py   # FinBERT sentiment over GDELT news
python score_wsb_emotion.py              # GoEmotions emotion over WSB posts (needs wsb_sentiment.csv)
```

**Build the training datasets:**
```bash
python build_features.py
# → data/processed/features_finance.csv
# → data/processed/features_sentiment.csv  (price + technicals + GDELT + WSB sentiment + emotion)
```

**Train the core 4 models** (used by the Compare page's architecture-vs-sentiment ablation):
```bash
python train_xgboost.py   # XGBoost: Finance-only + Finance+Sentiment  (~2-5 min)
python train_lstm.py      # LSTM+Attention: Finance-only + Finance+Sentiment  (~10-30 min CPU)
python extract_lstm_attention.py   # saves per-day attention weights from the trained LSTM checkpoints (no retraining, ~seconds)
```

**Train the extended roster** (optional — expands the Leaderboard page only,
`Compare.jsx` stays fixed to the core 4 above; cheapest/lowest-risk first):
```bash
python train_random_forest.py       # Random Forest: Finance-only + Finance+Sentiment  (~1 min)
python train_logistic_regression.py # Logistic Regression: Finance-only + Finance+Sentiment  (~10 sec)
python train_gru.py                 # GRU+Attention: Finance-only + Finance+Sentiment  (~2 min CPU)
python train_transformer.py         # TransformerEncoder: Finance-only + Finance+Sentiment  (~5-15 min CPU)
```
Every `train_*.py` script appends to the same `model_metrics.json` — running
any subset is fine, and the Leaderboard page (`/leaderboard`) picks up
whichever models exist automatically, no code changes needed.

**Analysis / validation scripts** (optional, feed the thesis write-up and the
Compare page, not required just to run the app):
```bash
cd ../notebooks
python 15_backtesting.py              # portfolio backtest vs SPY
python 16_confidence_calibration.py   # confidence-bucketed accuracy check
python 17_goemotions_validation.py    # validates the GoEmotions model against its official test set
python 18_shap_feature_selection.py   # SHAP feature importance (XGBoost variants)
python 19_ensemble_evaluation.py      # ensemble of all 4 core models — also powers the "Ensemble" row on the Leaderboard page
```

Realistic numbers to expect (this is a genuinely hard prediction problem —
don't expect much above 50-55% accuracy; see `notebooks/16_confidence_calibration.py`
for why that's still meaningful):

**Core 4** (used by `Compare.jsx`):

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| XGBoost — Finance only | ~51.7% | ~61.6% | ~50.2% |
| XGBoost — Finance+Sentiment | ~51.6% | ~61.6% | ~50.3% |
| LSTM+Attention — Finance only | ~51.6% | ~51.3% | ~53.1% |
| LSTM+Attention — Finance+Sentiment (best of the core 4) | ~53.0% | ~54.3% | ~54.0% |

**Extended roster** (shown only on the Leaderboard page — real numbers from
the first training run of this session):

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Ensemble — all 4 core variants (best overall) | 54.18% | 67.81% | 53.46% |
| GRU — Finance only | 53.87% | 61.26% | 54.71% |
| TransformerEncoder — Finance+Sentiment | 53.34% | 54.04% | 54.75% (best AUC) |
| TransformerEncoder — Finance only | 53.30% | 57.02% | 54.06% |
| GRU — Finance+Sentiment | 52.49% | 56.52% | 52.66% |
| Logistic Regression — Finance+Sentiment | 51.28% | 50.16% | 52.47% |
| Logistic Regression — Finance only | 51.22% | 50.47% | 52.61% |
| Random Forest — Finance only | 50.90% | 54.63% | 50.59% |
| Random Forest — Finance+Sentiment | 50.08% | 52.67% | 50.20% |

Notably: sentiment features helped LSTM and the Transformer, but *hurt*
Random Forest and GRU — a genuinely mixed, honest result rather than a
uniform "sentiment always helps" story, and itself worth a sentence in the
thesis discussion. All 13 results sit in the credible 50-55% band with no
sign of leakage.

Then start the servers as in **Quick Start** above.

---

## Common errors and fixes

**`ModuleNotFoundError: No module named '...'`**
```bash
cd backend && source .venv/Scripts/activate && pip install -r requirements.txt
```

**`port 8000 already in use` (backend)**
```bash
# find and stop whatever's holding it, then re-run uvicorn
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

**`port 3000 already in use` (frontend)** — same idea with `:3000`, or just
let Vite pick the next free port (it will tell you in its own output).

**Frontend shows "Could not load..." / red error banner**
Backend isn't running or crashed. Check Terminal 1's output for a traceback.
Confirm `frontend/.env` has `VITE_API_URL=http://127.0.0.1:8000` and that
`curl http://127.0.0.1:8000/` returns a JSON status.

**`No prediction file found for xgb_finance` / similar**
Means `train_xgboost.py` or `train_lstm.py` was never run, or the file was
deleted. Check `backend/data/predictions/` — you should see 4
`*_predictions.csv` files plus `model_metrics.json`.

**A page's Sentiment & Emotion tab shows "No WSB posts mentioning X..."**
This is expected, not a bug — WSB post data only covers up to 2021-08-31, but
the model predicts up to 2024-12-20. Any ticker's *latest* prediction date
almost always falls outside WSB's coverage window, so the honest empty state
shows instead of fabricated data. Historical dates within 2015–2021 do have
real coverage.

**Windows console crashes with a `UnicodeEncodeError` / `cp1252` error while
running a training script**
Some scripts print Unicode characters (▲, →) that the default Windows
terminal can't display. Either run inside VS Code's integrated terminal (uses
UTF-8) or ignore it — it only affects a `print()` line, not the actual
training/output files.

---

## Two terminals summary

| Terminal | Location | Command |
|---|---|---|
| 1 — backend | `quantsightv2/backend/` | `source .venv/Scripts/activate && uvicorn main:app --reload --port 8000` |
| 2 — frontend | `quantsightv2/frontend/` | `npm run dev` |

Both must stay running. Then open **http://localhost:3000**.
