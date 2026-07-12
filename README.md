# QuantSight — AI Decision Support for Stock Signals

> Final Year Project (FYP) — a decision-support platform that predicts next-day
> BUY/SELL direction for 44 US stocks/ETFs, and explains *why*, by combining
> technical indicators, news + social sentiment, and multi-label emotion
> detection across two model families.

For step-by-step run instructions, see **[HOW_TO_RUN.md](HOW_TO_RUN.md)**.
This file is the project overview: what it does, how it's built, and why.

---

## What This System Does

QuantSight predicts next-day stock direction and — more importantly for an FYP
about *decision support* rather than a black-box signal — explains the
prediction in plain language, shows how confident the model really is, and is
honest when a data source doesn't cover a given day rather than fabricating a
number.

It trains **4 model variants** so the value of adding sentiment/emotion data
is directly measurable, not assumed:

| # | Model | Data | Output file |
|---|-------|------|-------------|
| A | XGBoost | Finance only (price + technical indicators) | `xgb_finance_predictions.csv` |
| B | XGBoost | Finance + Sentiment + Emotion | `xgb_sentiment_predictions.csv` |
| C | LSTM + Attention | Finance only | `lstm_finance_predictions.csv` |
| D | LSTM + Attention | Finance + Sentiment + Emotion | `lstm_sentiment_predictions.csv` |

**Real, honestly-reported results** (chronological train/test split, 2015–2023
train / 2023–2024 test, purged walk-forward CV, no leakage):

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| XGBoost — Finance only | 51.66% | 61.57% | 50.24% |
| XGBoost — Finance+Sentiment | 51.63% | 61.55% | 50.31% |
| LSTM+Attention — Finance only | 51.56% | 51.25% | 53.09% |
| LSTM+Attention — Finance+Sentiment (**best**) | **53.02%** | **54.27%** | **54.01%** |

Daily stock direction is a genuinely hard prediction problem — accuracy in the
low-50s is consistent with published literature (see [Research Notes](#research-notes-and-related-work)
below), not a bug. The more interesting, defensible result is that adding
sentiment+emotion features gives the LSTM a real backtest lift: **29.27% → 48.49%
portfolio return**, Sharpe **0.735 → 1.279**, on a top-5 equal-weight
weekly-rebalanced portfolio vs. SPY.

This table stays focused on the core research question (architecture ×
sentiment). See the in-app **Leaderboard** (`/leaderboard`) for the full
ranked comparison, which also includes Random Forest, Logistic Regression,
GRU, a Transformer encoder, and an ensemble-of-all-4-core-models baseline —
all real, trained variants, none of them literature numbers.

### Beyond the base prediction

- **Multi-label emotion detection** — GoEmotions taxonomy (27 emotions +
  neutral) via a pretrained RoBERTa (`SamLowe/roberta-base-go_emotions`) run
  as a frozen feature extractor over WallStreetBets posts. 6 finance-relevant
  emotions kept: fear, optimism, anger, excitement, confusion, disappointment.
  Validated against the official GoEmotions test split (macro-F1 0.499)
  before use.
- **Explainability (XAI)**, two methods, matched to what each model actually
  supports:
  - SHAP `TreeExplainer` — global feature importance for both XGBoost variants.
  - The LSTM's own internal attention weights, exposed per-prediction (not a
    post-hoc approximation). Honest finding: weights come out **near-uniform**
    across the 10-day window rather than sharply peaked — a real result, not
    a bug, consistent with published attention-interpretability critiques.
- **Confidence calibration** — out-of-fold isotonic/Platt scaling (chosen by
  Brier score); the BUY/SELL decision is always driven by the raw probability,
  calibration only refines the *displayed* confidence number.
- **Purged walk-forward cross-validation** with embargo periods, to avoid
  label leakage across time.
- **Backtesting** vs. SPY — Sharpe, Sortino, max drawdown, transaction costs.
- **Risk assessment** — combines volatility, ATR, and model agreement into a
  Low/Medium/High rating per ticker.

Every number shown anywhere in the UI traces to one of these real, computed
sources — see [No-Mock-Data Policy](#no-mock-data-policy) below.

---

## Project Structure

```
quantsightv2/
│
├── backend/                              ← Python FastAPI server
│   ├── main.py                           FastAPI app — all endpoints
│   ├── copilot_engine.py                 XAI explanation engine (/explain)
│   ├── build_features.py                 Merges price + technicals + sentiment + emotion
│   ├── score_news_sentiment_finbert.py   FinBERT sentiment over GDELT news
│   ├── score_wsb_emotion.py              GoEmotions emotion over WSB posts
│   ├── train_xgboost.py                  Trains XGBoost × 2 variants
│   ├── train_lstm.py                     Trains LSTM+Attention × 2 variants
│   ├── extract_lstm_attention.py         Saves per-day attention weights (no retraining)
│   ├── requirements.txt
│   └── data/
│       ├── raw/                          Original stock/news/WSB data
│       ├── processed/                    features_finance.csv, features_sentiment.csv, ...
│       ├── models/                       xgb_*.pkl, lstm_*.pt (trained)
│       └── predictions/                  *_predictions.csv, model_metrics.json, ...
│
├── notebooks/                            Numbered pipeline scripts (01–19) — data
│                                          collection through backtesting/SHAP/calibration
│
└── frontend/                             ← React 18 + Vite + Tailwind
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── .env                              VITE_API_URL=http://127.0.0.1:8000
    └── src/
        ├── App.jsx                       Router — sidebar layout, 7 pages
        ├── components/
        │   ├── Sidebar.jsx               Left nav
        │   ├── CandlestickChart.jsx      Real OHLCV candlestick (custom Recharts shape)
        │   └── RadialProgress.jsx        SVG confidence ring
        ├── data/
        │   └── companyNames.js           Ticker → company name reference data
        └── pages/
            ├── Dashboard.jsx             KPIs, Top Opportunities, Market Sentiment, AI Insight
            ├── Overview.jsx              Market — all 44 tickers, 4-model signals, filters
            ├── Detail.jsx                Candlestick chart, AI Copilot, 5 tabs
            ├── Compare.jsx                4-model accuracy comparison, Confidence Boost
            ├── Leaderboard.jsx           4 models ranked by accuracy, sortable, vs. random baseline
            ├── Portfolio.jsx             Holdings tracker (localStorage) + live prices
            └── Game.jsx                  Prediction game — persistent score/streak/level
```

---

## Setup

See **[HOW_TO_RUN.md](HOW_TO_RUN.md)** for the full guide — quick start (data
and models already trained, just start two servers) plus the full pipeline
for rebuilding everything from scratch.

Short version:

```bash
# Terminal 1 — backend
cd backend
source .venv/Scripts/activate   # or .venv\Scripts\activate on Windows cmd/PowerShell
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Then open **http://localhost:3000**.

---

## Dashboard Pages

| Page | URL | What it shows |
|---|---|---|
| Dashboard | `/` | KPI summary, Top Opportunities, real-time Market Sentiment (VADER over current news), AI Insight summary, Market News |
| Market | `/market` | All 44 tickers as cards, all 4 model signals each, filters, Live Mode |
| Detail | `/detail/:ticker` | Real OHLC candlestick chart, AI Copilot explanation, tabs: Overview / Technical (incl. LSTM attention) / Sentiment & Emotion / History / Learn |
| AI Compare | `/compare` | Accuracy table + charts across all 4 variants, sentiment+emotion impact, Confidence Boost |
| Leaderboard | `/leaderboard` | The 4 model variants ranked by accuracy/F1/precision/recall/AUC-ROC, sortable columns, vs. a 50% random-guess baseline |
| Portfolio | `/portfolio` | Your tracked holdings, live prices, BUY/HOLD/SELL recommendations |
| Game | `/game` | Guess what the model predicted on a real historical day — score/streak/level persist across visits |

---

## No-Mock-Data Policy

Every number in this project traces to a real computed source. Where a
wireframe or feature idea implied data this project doesn't have, it was
either built for real or explicitly left out — never faked. Concretely:

- Portfolio has no "Win Rate" stat — there's no historical-outcome ground
  truth to back it.
- The Game page has no multi-user leaderboard — there's no user-account
  backend, so it shows your own real result history instead of invented
  players.
- The Sentiment & Emotion tab shows an honest empty state ("No WSB posts
  mentioning X on the latest prediction date") rather than a fake score, on
  any date outside Reddit's real coverage window (WSB data covers up to
  2021-08-31; the model predicts through 2024-12-20).
- The LSTM attention visualization reports its actual near-uniform result
  rather than being tuned to look more dramatic than it is.

---

## Research Notes and Related Work

A few citations that directly informed methodology choices, kept here so
they don't get lost:

- **López de Prado, M. (2018)**, *Advances in Financial Machine Learning* —
  the purge/embargo cross-validation scheme used throughout training.
- **Jain, S. & Wallace, B.C. (2019)**, *Attention is not Explanation*,
  NAACL-HLT, and **Wiegreffe, S. & Pinter, Y. (2019)**, *Attention is not not
  Explanation*, EMNLP — the founding debate on attention-weight
  interpretability; this project's near-uniform LSTM attention finding is an
  independent replication of that result on financial time series, not an
  anomaly.
- **Demszky, D. et al. (2020)**, *GoEmotions: A Dataset of Fine-Grained
  Emotions* — the taxonomy and dataset the emotion classifier is validated
  against.
- **Long, S.C. et al. (2023)**, *"I just like the stock": The role of Reddit
  sentiment in the GameStop share rally*, The Financial Review — informed the
  decision to treat WSB as a genuinely predictive (if noisy) signal source
  rather than dismiss retail sentiment.
- **Khan, F.S. et al. (2025)**, *Model-agnostic explainable artificial
  intelligence methods in finance: a systematic review*, Artificial
  Intelligence Review — a 150-paper systematic review whose MA-XAI taxonomy
  (model-agnostic vs. model-specific, local vs. global, post-hoc vs.
  intrinsic) is the framework this project's SHAP (post-hoc, tree-specific)
  vs. LSTM attention (intrinsic) split already follows; also confirms
  SHAP+LIME as the two dominant model-agnostic XAI techniques in finance,
  grounding the SHAP choice in `18_shap_feature_selection.py`.
- **Goswami, A. & Uddin, M. (2026)**, *Significance of predictors: revisiting
  stock return predictions using explainable AI* — validates SHAP
  `TreeExplainer` as the standard feature-attribution method for
  gradient-boosted trees in return prediction, the same technique already
  applied to the XGBoost variants here.
- **Arsenault, C., Wang, C. & Patenaude, G. (2025)**, *A Survey of XAI in
  Financial Time Series Forecasting*, ACM Computing Surveys — a fuller
  taxonomy of interpretable vs. explainable financial models than the
  founding attention-interpretability papers above; independently
  corroborates near-uniform attention weights as a known, real phenomenon in
  the field rather than an anomaly specific to this project.
- **Ibrahim, M.M., Khan, A.U.I. & Kaplan, M. (2025)**, *From headlines to
  stock trends: NLP and XAI approach to predicting Türkiye's financial
  pulse*, Borsa Istanbul Review — a FinBERT-sentiment-driven ensemble
  (methodologically close to this project's own FinBERT news-sentiment
  pipeline) reporting 80–90% direction-classification accuracy. Cited here as
  a contrast rather than a benchmark to match: their reported setup appears
  to pair same-day news sentiment with same-day returns, which risks leaking
  price-contemporaneous information into the "prediction" — a useful
  illustration of why this project restricts to next-day-only labels under
  purged walk-forward CV instead.

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/tickers` | List of all available tickers |
| GET | `/overview` | All tickers with 4-model signals |
| GET | `/market-news` | Recent GDELT headlines across all tickers |
| GET | `/dashboard` | KPIs + Top Opportunities + news (Dashboard page) |
| GET | `/stock/{ticker}?days=90` | OHLCV + indicators (up to ~10y of history) |
| GET | `/explain/{ticker}` | Full AI Copilot explanation (all 4 models) |
| GET | `/history/{ticker}?days=7` | Recent day-by-day signal history |
| GET | `/compare/{ticker}` | 4-model side-by-side for one ticker |
| GET | `/metrics` | All 4 model accuracy metrics |
| GET | `/confidence-boost` | Real avg confidence delta from adding sentiment |
| GET | `/market-sentiment?days=14` | Real-time VADER sentiment over current news |
| GET | `/news/{ticker}?limit=10` | GDELT news headlines for one ticker |
| GET | `/live/{ticker}` | Live signal for one ticker (Yahoo Finance + XGBoost) |
| GET | `/live-overview` | Live signals for all tickers |
| GET | `/realtime/{ticker}` | Real-time price for one ticker |
| GET | `/game/question` | Random game question |
| POST | `/game/answer` | Submit game answer |

Full interactive docs: `http://localhost:8000/docs`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, TailwindCSS, Recharts |
| Backend | FastAPI, Uvicorn |
| ML Models | XGBoost, PyTorch (LSTM + Attention) |
| Sentiment / Emotion | VADER, FinBERT, HuggingFace `transformers` (GoEmotions RoBERTa) |
| Explainability | SHAP, native LSTM attention weights |
| Data | pandas, numpy, scikit-learn, `datasets` (GoEmotions validation) |
| News source | GDELT Project |

---

## Deployment

### Backend → Render (free tier)

1. Push `backend/` to GitHub
2. [render.com](https://render.com) → New Web Service → connect the repo
3. Root directory: `backend` · Build: `pip install -r requirements.txt` ·
   Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Upload trained model files and data CSVs via Render's persistent disk, or
   commit them to a private repo — free tier spins down after 15 minutes
   idle, so first request may take 30–60s to wake up.

### Frontend → Vercel (free tier)

1. Push `frontend/` to GitHub
2. [vercel.com](https://vercel.com) → New Project → import the repo
3. Root directory: `frontend` · Framework: Vite
4. Environment variable: `VITE_API_URL` = your Render backend URL

---

## Contact

- GitHub: [SyedFaris04](https://github.com/SyedFaris04)
- WhatsApp: +60175401989
- Email: syedamirfaris@gmail.com
- Instagram/TikTok: @syedweb
