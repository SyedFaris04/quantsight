# QuantSight — Setup Checklist

A click-by-click checklist for getting the full app running, including the
optional pieces (sign-in, AI Assistant, live prediction tracker). For a
narrative explanation of what each piece does, see [README.md](README.md);
for local dev commands, see [HOW_TO_RUN.md](HOW_TO_RUN.md). This file is
just the ordered checklist.

---

## Part A — Run it locally (no extra accounts needed)

- [ ] `cd backend && source .venv/Scripts/activate && uvicorn main:app --reload --port 8000`
- [ ] `cd frontend && npm install && npm run dev`
- [ ] Open http://localhost:3000 — Dashboard, Market, Detail, AI Compare,
      Leaderboard, Portfolio (guest mode), Game (guest mode) all work fully
      right now, zero configuration

Everything below is additive. Skip any section you don't need — the app
degrades gracefully (clear message, never a crash) without it.

---

## Part B — Sign-in (Google + email), synced Portfolio/Game

- [ ] Create a free project at [supabase.com](https://supabase.com)
- [ ] Settings → API → copy the **Project URL** and the **anon/publishable key**
- [ ] SQL Editor → run the schema that creates `portfolio_holdings` and
      `game_progress` (holdings/scores tables, RLS scoped to `auth.uid()`)
- [ ] SQL Editor → run the leaderboard migration: adds a `display_name`
      column to `game_progress` and opens its RLS to public `SELECT` (writes
      stay locked to each user's own row) so the Game page's cross-user
      leaderboard can read everyone's high score
- [ ] `frontend/.env`: set `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`
- [ ] Authentication → URL Configuration → add `http://localhost:3000` and
      your production (Vercel) URL as allowed redirects
- [ ] **Google sign-in:**
  - [ ] [Google Cloud Console](https://console.cloud.google.com) → new/existing
        project → APIs & Services → OAuth consent screen → configure
        (External, app name, support email)
  - [ ] APIs & Services → Credentials → Create Credentials → OAuth Client ID
        → Web application
  - [ ] Authorized redirect URI: `https://<project-ref>.supabase.co/auth/v1/callback`
  - [ ] Copy the Client ID + Secret → Supabase → Authentication → Sign In /
        Providers → Google → paste both, toggle on
- [ ] Email/password needs no setup — on by default. Optional: Authentication
      → Sign In / Providers → Email → toggle **"Confirm email"** off for
      easier local testing (turn back on before real users show up)
- [ ] Test: sign up locally, confirm a row appears in Supabase → Authentication
      → Users

---

## Part C — AI Assistant (chat widget)

- [ ] Free key at [console.groq.com](https://console.groq.com) → API Keys →
      Create API Key (just an email, no card)
- [ ] `backend/.env`: set `GROQ_API_KEY`
- [ ] Restart the backend, open the chat widget (bottom-right, any page), ask
      "what's trending today?" — should answer with real numbers, not a
      generic reply

---

## Part D — Live Prediction Track Record

Needs Part B done first (same Supabase project).

- [ ] SQL Editor → run the schema that creates `live_predictions` (public
      `SELECT` policy only — no insert/update/delete for anon/authenticated,
      so only the backend's service role key can ever write a row)
- [ ] Supabase → Settings → API → **Secret keys** → copy the `service_role`
      key (labelled `sb_secret_...` on newer projects) — **not** the anon key,
      this one bypasses Row Level Security and must never reach the frontend
- [ ] `backend/.env`: set `SUPABASE_URL` (same project URL as Part B) and
      `SUPABASE_SERVICE_ROLE_KEY`
- [ ] `backend/.env`: set `ADMIN_JOB_SECRET` to any random string, e.g.
      `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- [ ] Restart the backend, test locally:
      `curl -X POST http://127.0.0.1:8000/admin/run-daily-predictions -H "X-Admin-Key: <your secret>"`
      → should return `{"logged": 44, "resolved": 0, ...}` on first run
- [ ] Open `/track-record` — should show 44 pending predictions

---

## Part E — Deploy

### Backend → Render (free tier)

- [ ] Push `backend/` to GitHub
- [ ] [render.com](https://render.com) → New Web Service → connect the repo
- [ ] Root directory: `backend` · Build: `pip install -r requirements.txt` ·
      Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] Environment → add every variable from Parts B–D that you set up:
      `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`,
      `ADMIN_JOB_SECRET`
- [ ] Note your Render URL: `https://_____________.onrender.com`

### Frontend → Vercel (free tier)

- [ ] Push `frontend/` to GitHub
- [ ] [vercel.com](https://vercel.com) → New Project → import the repo
- [ ] Root directory: `frontend` · Framework: Vite
- [ ] Environment variables: `VITE_API_URL` = your Render URL,
      `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`
- [ ] Note your Vercel URL: `https://_____________.vercel.app`
- [ ] Back in Supabase → Authentication → URL Configuration → add this
      Vercel URL as an allowed redirect (needed for Google sign-in to work
      in production)

### GitHub Actions — the live tracker's daily scheduler

Only needed if you did Part D.

- [ ] Repo → Settings → Secrets and variables → Actions → New repository secret:
      `RENDER_BACKEND_URL` = your Render URL
- [ ] Same page → New repository secret: `ADMIN_JOB_SECRET` = same value as
      the Render env var
- [ ] Actions tab → "Daily live prediction tracker" → **Run workflow** (manual
      trigger) → confirm it finishes with a green checkmark
- [ ] From here it runs automatically every weekday after US market close —
      no further action needed, ever

---

## Part F — Presentation demo script

Suggested order to show your lecturer:

1. **Open Compare page** — show the accuracy metrics table first
   - Point out: LSTM+Sentiment has the highest accuracy
   - Point out: Adding sentiment improves every metric
   - Show the sentiment impact delta badges (+X.XX%)

2. **Open Overview (Market) page** — show the full ticker table
   - Filter to "Strong Agreement" — show tickers all 4 models agree on
   - Filter to "BUY" signals only

3. **Click a ticker** (e.g. AAPL or MSFT) → Detail page
   - Show the AI summary banner — plain English explanation
   - Show the signal contribution bars (Technical / Sentiment / Model)
   - Show the 4 model signal cards side by side
   - Show the technical indicator cards — explain RSI/MACD to lecturer
   - Show the GDELT news feed at the bottom
   - History tab → Track Record panel — real backtested per-ticker accuracy

4. **Open Track Record page** (`/track-record`) — the live, forward-looking,
   tamper-proof accuracy log; explain the "predict today, verify tomorrow"
   mechanism and why writes are server-only

5. **Open the AI Assistant** (chat button, bottom-right) — ask "why is AAPL
   a BUY" or "can I trust this signal" and show it calling real tools instead
   of inventing an answer

6. **Open Game page** — let the lecturer try it
   - Set difficulty to Easy
   - Play 3–4 questions together
   - Point out the leaderboard, sound/confetti on a correct answer

---

## Auto-generated files (do NOT manually create these)

Created automatically by the ML pipeline scripts (`build_features.py`,
`train_*.py`, `extract_lstm_attention.py`):

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

Created automatically by the live prediction tracker (Part D), once running —
not something you create, just Supabase rows:

```
live_predictions table — one row per (ticker, predicted_date)
```
