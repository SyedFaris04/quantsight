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
├── .github/workflows/
│   └── daily-predictions.yml             Scheduled trigger for the live prediction tracker
│
├── backend/                              ← Python FastAPI server
│   ├── main.py                           FastAPI app — all endpoints
│   ├── copilot_engine.py                 XAI explanation engine (/explain)
│   ├── chatbot_engine.py                 AI Assistant — Groq + tool-calling
│   ├── prediction_tracker.py             Live prediction log + daily resolution job
│   ├── live_signals.py                   Live (Yahoo Finance) signal generation
│   ├── build_features.py                 Merges price + technicals + sentiment + emotion
│   ├── score_news_sentiment_finbert.py   FinBERT sentiment over GDELT news
│   ├── score_wsb_emotion.py              GoEmotions emotion over WSB posts
│   ├── train_xgboost.py                  Trains XGBoost × 2 variants
│   ├── train_lstm.py                     Trains LSTM+Attention × 2 variants
│   ├── extract_lstm_attention.py         Saves per-day attention weights (no retraining)
│   ├── requirements.txt
│   ├── .env                              GROQ_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, ADMIN_JOB_SECRET (gitignored)
│   ├── .env.example                      Documents the required variables, safe to commit
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
    ├── .env                              VITE_API_URL, VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY
    └── src/
        ├── App.jsx                       Router — sidebar layout, 11 pages + global chat widget
        ├── components/
        │   ├── Sidebar.jsx               Left nav + account section (sign in / avatar)
        │   ├── ChatWidget.jsx            Floating AI Assistant, present on every page
        │   ├── CandlestickChart.jsx      Real OHLCV candlestick (custom Recharts shape)
        │   └── RadialProgress.jsx        SVG confidence ring
        ├── context/
        │   └── AuthContext.jsx           Supabase auth state — user is null in guest mode
        ├── lib/
        │   ├── supabaseClient.js         Shared Supabase client (auth + Postgres)
        │   └── gameFx.js                 Game page "juice" — sound, confetti, count-up, prefs
        ├── data/
        │   └── companyNames.js           Ticker → company name reference data
        └── pages/
            ├── Dashboard.jsx             KPIs, Top Opportunities, Market Sentiment, AI Insight
            ├── Overview.jsx              Market — all 44 tickers, 4-model signals, filters
            ├── Detail.jsx                Candlestick chart, AI Copilot, 5 tabs
            ├── Compare.jsx               4-model accuracy comparison, Confidence Boost
            ├── Leaderboard.jsx           4 models ranked by accuracy, sortable, vs. random baseline
            ├── TrackRecord.jsx           Live, forward-looking prediction accuracy (not backtested)
            ├── Portfolio.jsx             Holdings tracker (Supabase if signed in, else localStorage)
            ├── Game.jsx                  Prediction game — score/streak/level, player leaderboard
            ├── Login.jsx                 Google OAuth + email/password, optional
            └── Settings.jsx              Account, game prefs, accessibility, data export
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

Then open **http://localhost:3000**. Every core feature (predictions,
explanations, Portfolio, Game) works immediately with zero configuration —
sign-in, the AI Assistant, and the live prediction tracker are additive and
degrade gracefully without their `.env` values (see
[Deployment](#deployment) below for what each one unlocks).

---

## Pages

| Page | URL | What it shows |
|---|---|---|
| Dashboard | `/` | KPI summary, Top Opportunities, real-time Market Sentiment (VADER over current news), AI Insight summary, Market News |
| Market | `/market` | All 44 tickers as cards, all 4 model signals each, filters, Live Mode |
| Detail | `/detail/:ticker` | Real OHLC candlestick chart, AI Copilot explanation, tabs: Overview / Technical (incl. LSTM attention) / Sentiment & Emotion / History / Learn |
| AI Compare | `/compare` | Accuracy table + charts across all 4 variants, sentiment+emotion impact, Confidence Boost |
| Leaderboard | `/leaderboard` | The 4 *model* variants ranked by accuracy/F1/precision/recall/AUC-ROC, sortable columns, vs. a 50% random-guess baseline |
| Track Record | `/track-record` | QuantSight's own live, forward-looking accuracy — predictions logged daily, checked the next trading day (see [Live Prediction Track Record](#live-prediction-track-record)) |
| Portfolio | `/portfolio` | Your tracked holdings, live prices, BUY/HOLD/SELL recommendations — synced via Supabase if signed in, localStorage otherwise |
| Game | `/game` | Guess what the model predicted on a real historical day — score/streak/level, sound/confetti/celebrations, a real cross-*user* leaderboard, keyboard shortcuts (B/S) |
| Login | `/login` | Google OAuth or email/password — entirely optional, see [Accounts & Sign-In](#accounts--sign-in) |
| Settings | `/settings` | Account (password, display name, sign out), game preferences, accessibility (reduce motion), CSV export, clear local data |

An AI Assistant (floating chat button, bottom-right) is available on every
page — see [AI Assistant](#ai-assistant) below.

---

## Accounts & Sign-In

Signing in is entirely optional — every feature works fully in **guest
mode** (data in `localStorage`, on this device/browser only). Signing in
(Google or email/password, via Supabase Auth) upgrades Portfolio holdings
and Game progress to sync across devices through Postgres instead, with Row
Level Security scoping every row to its owner. The first time an existing
guest-mode user signs in, if their account has no data yet, they're offered
a one-time, non-destructive import of what's already in this browser.

`backend/main.py` has zero concept of accounts — auth and per-account data
are handled entirely by the frontend talking directly to Supabase
(`@supabase/supabase-js`), which is the standard, simplest pattern for this
scale and avoids adding JWT-verification middleware to the API.

---

## AI Assistant

A floating chat widget, present on every page, backed by
[Groq](https://console.groq.com)'s free tier (Llama 3.3 70B). It isn't a
generic chatbot bolted on for show — it's wired via tool-calling to
QuantSight's own real endpoints (`backend/chatbot_engine.py`), so a question
like "why is AAPL a BUY" or "can I trust this?" gets answered from an actual
tool call (current signal, full explanation, live price-based signal,
backtested per-ticker accuracy, live track record, market-wide sentiment),
never an invented number. Same no-mock-data policy as the rest of the app,
just applied to an LLM instead of a chart.

---

## Live Prediction Track Record

The forward-looking counterpart to the historical backtests: every trading
day, a scheduled [GitHub Actions workflow](.github/workflows/daily-predictions.yml)
(Render's free tier has no cron/background process of its own) triggers
`backend/prediction_tracker.py`, which:

1. Logs the live model's signal for every ticker (today's real prediction,
   locked in before the outcome is known)
2. Resolves any prediction logged on an earlier day by checking today's
   price against it — the same next-day-direction labeling the models were
   trained on

Writes only ever happen through Supabase's **service role** key, which
bypasses Row Level Security entirely — the `live_predictions` table's only
policy is public `SELECT`. No client, including this app's own frontend,
can insert, edit, or fake a row, which is what makes the resulting accuracy
number mean something. See `/track-record` in the app, or ask the AI
Assistant "do your predictions actually come true?"

---

## No-Mock-Data Policy

Every number in this project traces to a real computed source. Where a
wireframe or feature idea implied data this project doesn't have, it was
either built for real or explicitly left out — never faked. Concretely:

- Portfolio has no "Win Rate" stat — there's no historical-outcome ground
  truth to back it.
- The Game page's leaderboard is a real cross-user table read from Supabase
  (RLS opened to public `SELECT`, writes locked to each user's own row) —
  not invented players, and not fakeable by any client either.
- The Live Prediction Track Record (above) exists specifically because a
  claimed "our model works" is worthless without a tamper-proof, ongoing
  record proving it — predictions are locked in before outcomes are known,
  writes are server-only.
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
| GET | `/accuracy-history/{ticker}` | Real backtested per-ticker accuracy track record |
| GET | `/game/question` | Random game question |
| POST | `/game/answer` | Submit game answer |
| POST | `/chat` | AI Assistant — streamed reply, grounded via tool-calling |
| GET | `/live-track-record` | Live, forward-looking prediction accuracy (public) |
| POST | `/admin/run-daily-predictions` | Daily job — logs + resolves live predictions (GitHub Actions only, `X-Admin-Key` required) |

Full interactive docs: `http://localhost:8000/docs`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, TailwindCSS, Recharts |
| Backend | FastAPI, Uvicorn |
| Auth + Database | Supabase (Postgres + Auth, Row Level Security) |
| AI Assistant | Groq (Llama 3.3 70B, free tier), tool-calling |
| Scheduling | GitHub Actions (daily live-prediction job trigger) |
| ML Models | XGBoost, PyTorch (LSTM + Attention) |
| Sentiment / Emotion | VADER, FinBERT, HuggingFace `transformers` (GoEmotions RoBERTa) |
| Explainability | SHAP, native LSTM attention weights |
| Data | pandas, numpy, scikit-learn, `datasets` (GoEmotions validation) |
| News source | GDELT Project |
| Live prices | yfinance (Yahoo Finance) |

---

## Deployment

Everything below runs on free tiers. See **[CHECKLIST.md](CHECKLIST.md)** for
a literal step-by-step checklist of this same setup.

### Backend → Render (free tier)

1. Push `backend/` to GitHub
2. [render.com](https://render.com) → New Web Service → connect the repo
3. Root directory: `backend` · Build: `pip install -r requirements.txt` ·
   Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Upload trained model files and data CSVs via Render's persistent disk, or
   commit them to a private repo — free tier spins down after 15 minutes
   idle, so first request may take 30–60s to wake up.
5. Environment variables (Service → Environment):

   | Key | Value | Required for |
   |---|---|---|
   | `GROQ_API_KEY` | free key from [console.groq.com](https://console.groq.com) | AI Assistant |
   | `SUPABASE_URL` | your Supabase project URL | AI Assistant's track-record tool, live prediction job |
   | `SUPABASE_SERVICE_ROLE_KEY` | Supabase → Settings → API → service role secret (**never** the anon key) | live prediction job (bypasses RLS to write) |
   | `ADMIN_JOB_SECRET` | any random string, e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"` | authenticates the daily GitHub Actions job |

   Every one of these degrades gracefully if unset (clear error message, not
   a crash) — the app runs fine without any of them, just without that
   specific feature.

### Frontend → Vercel (free tier)

1. Push `frontend/` to GitHub
2. [vercel.com](https://vercel.com) → New Project → import the repo
3. Root directory: `frontend` · Framework: Vite
4. Environment variables (Project Settings → Environment Variables):

   | Key | Value |
   |---|---|
   | `VITE_API_URL` | your Render backend URL |
   | `VITE_SUPABASE_URL` | your Supabase project URL |
   | `VITE_SUPABASE_ANON_KEY` | Supabase → Settings → API → anon/publishable key (safe client-side — RLS is what actually protects data) |

### Supabase (free tier)

1. [supabase.com](https://supabase.com) → New Project
2. SQL Editor → run the schema for `portfolio_holdings`, `game_progress`
   (+ `display_name` column and public-read policy for the leaderboard), and
   `live_predictions` — see this repo's setup history / ask the AI Assistant's
   author for the exact statements, they're straightforward `create table` +
   `enable row level security` + `create policy` blocks
3. Authentication → Sign In / Providers → enable **Google** (needs a Google
   Cloud OAuth Client ID/Secret — Authorized redirect URI is
   `https://<project-ref>.supabase.co/auth/v1/callback`) and **Email**
   (on by default)
4. Authentication → URL Configuration → add your local (`http://localhost:3000`)
   and production (Vercel) URLs as allowed redirects

### GitHub Actions (free — the live prediction scheduler)

Repo → Settings → Secrets and variables → Actions → add:

| Secret | Value |
|---|---|
| `RENDER_BACKEND_URL` | your Render backend URL |
| `ADMIN_JOB_SECRET` | same value as the Render env var above |

The workflow (`.github/workflows/daily-predictions.yml`) runs automatically
on weekdays after US market close, and can be triggered manually anytime
from the Actions tab → "Daily live prediction tracker" → **Run workflow**.

---

## Contact

- GitHub: [SyedFaris04](https://github.com/SyedFaris04)
- WhatsApp: +60175401989
- Email: syedamirfaris@gmail.com
- Instagram/TikTok: @syedweb
