# QuantSight: supervisor progress and demonstration

- **Meeting:** Monday morning, 28 September 2026.
- **Prepared:** 27 September 2026, Malaysia time.
- **Current status:** deployed working prototype; backtesting and research comparison are ready to demonstrate. Final predictive performance is still being validated.
- **Companion:** [Simple project cheat sheet](PROJECT_CHEAT_SHEET.md).

## 1. What to say first

- "QuantSight predicts whether a US stock or ETF will rise over the next five trading sessions."
- "I have added an auditable backtest with trading costs, benchmark comparisons and risk metrics."
- "I also audited the data and tested whether financial-news sentiment improves prediction. The current study does not establish an improvement, so I have retained that finding instead of claiming higher accuracy."
- "The application is deployed, and I have a local backup for this demonstration. My next priority is stronger validation on an unused evaluation period and genuine forward predictions."

## 2. Completed work

- **Working application:** Dashboard, Market, instrument details, AI Compare, Backtesting, model Leaderboard, Track Record, Portfolio, learning Game and account/settings pages.
- **Backtesting:** previous-close signals, next-open trading, share/cash accounting, weekly rebalancing, trading costs and final liquidation.
- **Comparisons:** 12 model variants, one four-model ensemble, SPY and an equal-weight portfolio: 15 portfolios altogether.
- **Evaluation:** net return, Sharpe, Sortino, maximum drawdown, volatility, costs, uncertainty and cost sensitivity.
- **Audit evidence:** downloadable metrics, daily portfolio values, trade records and complete report.
- **Interface:** redesigned Dashboard, visible data dates, clearer research results, metric explanations, mobile layouts and retry/error states.
- **Prediction consistency:** five-session labels and live feature calculations have been aligned; historical results and forward forecasts are separated.
- **Cloud setup:** GitHub, Vercel frontend, Render backend and Supabase are connected. The separate v2 live-prediction table has been created; the old table is preserved.
- **Research:** audited price/news timing, tested VADER and FinBERT, and prepared new price data with separate development and candidate final-evaluation files.

## 3. Backtesting results to show

- **Dates:** 22 February 2023 to 20 December 2024; 462 trading sessions.
- **Starting capital:** USD 100,000; long-only; five portfolio slots; weekly rebalance; no borrowing/leverage.
- **Costs:** 0.10% commission + 0.05% slippage per side; cash/risk-free rate assumed to be 0%.
- **Four-model ensemble:** net return **61.98%**; Sharpe **1.444**; maximum drawdown **-17.80%**.
- **SPY benchmark:** net return **51.69%**; Sharpe **1.888**; maximum drawdown **-9.97%**.
- **Equal-weight benchmark:** net return **42.89%**; Sharpe **1.524**; maximum drawdown **-11.48%**.
- **Interpretation:** the ensemble earned more total return than SPY in this simulation, but had poorer risk-adjusted performance and a larger fall from its peak.
- **Cost sensitivity:** at 0.30% total cost per side, the ensemble return falls to about **41.03%**, below SPY under the same cost assumption.
- **Accounting check:** independently replaying the exported trades reproduced all 15 portfolio values within USD 0.000001.
- **Important limit:** these are exploratory historical simulations. The period was reused during development; correct accounting does not prove an unbiased strategy or future profitability.

## 4. Research findings and accuracy

- **Data problem found:** the original price panel ends in December 2024; WSB posts end in August 2021; the saved Dashboard news archive ends in April 2026.
- **Why it matters:** text inputs are zero throughout the shared 2023–2024 historical evaluation sample. Different results from the old "finance + sentiment" variants do not establish a sentiment benefit.
- **New historical news archive:** 70,974 source rows; news coverage for 26 of 44 instruments; future-price fields excluded from features.
- **Controlled study:** train on 2017–2021; calibrate on 2022; evaluate on the same 2023 observations; remove five-session labels crossing the boundaries.
- **Constant training-prior baseline:** 55.23% accuracy; Brier loss 0.247623.
- **Logistic regression, finance only:** 47.43% accuracy; Brier loss 0.255580.
- **Logistic regression with VADER:** 47.49% accuracy; Brier loss 0.255890.
- **Logistic regression with FinBERT:** 47.33% accuracy; Brier loss 0.256194.
- **Conclusion:** lower Brier loss is better; these variants did not beat the simple baseline. FinBERT did not justify replacing the live model. These figures belong to this particular study, not every model or the entire app.
- **New data preparation:** 135,545 price rows across 44 instruments plus six sector ETFs; all 250 NYSE sessions in 2025 are present for all 50 instruments.
- **Prepared, not evaluated:** three purged development folds are ready; 2025 is only a candidate final evaluation period until its prior-use status is confirmed. No new 2025 model score is claimed.

## 5. What has been verified

- Latest completed software checks: **72 Python tests**, **11 database migration checks**, and frontend production build passed.
- Previous browser checks covered Dashboard, AI Compare and Backtesting, including mobile layouts and error recovery.
- Live checks on 27 September: website, Dashboard API, backtest API, research API and live-tracker read all returned HTTP 200.
- Live tracker currently has **0 logged and 0 resolved v2 forecasts**. There is no live accuracy result to present yet.
- Core local API checks passed; Dashboard, Backtesting and AI Compare were also checked in a browser using the local backend. See the demo verification note linked below.
- This validates the checked paths; it is not a claim that every account, game or external-service flow has completed user acceptance testing.

## 6. Live demonstration

- **Yes, the current progress can be shown live:** [quantsight.vercel.app](https://quantsight.vercel.app).
- Open the site and its main pages **10–15 minutes before the meeting** to allow backend startup and loading.
- Keep these tabs ready: [Dashboard](https://quantsight.vercel.app/), [Backtesting](https://quantsight.vercel.app/backtesting), [AI Compare](https://quantsight.vercel.app/compare).
- In the 27 September check, some hosted API requests took about **13–24 seconds**. This is one observation, not a guaranteed response time.
- If a page times out, allow the backend to become ready and use the retry button. Switch to localhost if waiting would interrupt the presentation.
- GitHub pushes trigger the configured Render/Vercel deployments. Supabase SQL migrations are a separate step; a push does not automatically run them.
- No additional SQL migration is required for this documentation/demo update.

## 7. Local backup: exact PowerShell commands

- Use **two PowerShell terminals**. The required environment, frontend dependencies and serving files already exist on this laptop.
- No retraining or package reinstall is required for this demonstration.
- In **Terminal 1**, start the backend:

```powershell
Set-Location 'C:\Users\syeda\OneDrive\Desktop\quantsightv2\backend'
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

- Wait for `Application startup complete` and the Uvicorn URL. Startup can take a little time.
- To check readiness, open [the local backend](http://127.0.0.1:8001/) and look for a JSON response.
- In **Terminal 2**, start the frontend:

```powershell
Set-Location 'C:\Users\syeda\OneDrive\Desktop\quantsightv2\frontend'
$env:VITE_API_URL = 'http://127.0.0.1:8001'
.\node_modules\.bin\vite.cmd --host 127.0.0.1 --port 3000 --strictPort
```

- Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)**. This frontend now calls your laptop's backend directly.
- The API override applies to this terminal session; it does not change the live website or its settings.
- Keep both terminals open. Use `Ctrl+C` in each when you want to stop the servers.
- If a port is already in use, first check the existing local URLs: the servers may already be running. Stop only a server you recognise before starting another.
- Use guest mode for the core demonstration. Localhost has separate browser storage from the deployed site.
- **Local saved-data demo:** Dashboard, historical Market/detail views, saved Backtesting and AI Compare use prepared local artifacts. The three main demo pages have been browser-checked.
- **Still internet-dependent:** live prices/Live Mode, cloud sign-in and account sync, forward tracking in Supabase, and the Groq AI chat.
- Local operation removes Render from the core demo path; it does not make every feature fully offline.

## 8. Suggested five-minute presentation

1. **30 seconds — Objective:** five-trading-session direction prediction and educational decision support.
2. **45 seconds — Dashboard:** show the dated historical snapshot and explain why data dates are visible.
3. **2 minutes — Backtesting:** compare ensemble with SPY; explain Sharpe, drawdown and costs; download the trade ledger or full report.
4. **1 minute — AI Compare:** show the baseline and FinBERT/VADER results; explain the data mismatch and why a negative finding is useful.
5. **45 seconds — Next steps:** untouched evaluation, walk-forward validation and genuinely recorded live outcomes.

- Before leaving for the meeting: start both local servers, open the three local pages, keep the laptop plugged in and avoid sleep during the demo.
- Download the backtest report/metrics beforehand; keep this Markdown file available as evidence if either demonstration has a problem.
- If asked about live accuracy: "The forward tracker is configured, but it has no resolved v2 predictions yet. Each new forecast needs five subsequent trading sessions before it can be evaluated."

## 9. Remaining work, in priority order

1. Verify genuine scheduled forecast writes and later their fifth-session outcomes.
2. Confirm an unused final evaluation period; freeze it before choosing further models.
3. Run purged walk-forward development experiments, fitting preprocessing and feature selection only on training data.
4. Evaluate against simple baselines using accuracy, balanced accuracy, MCC, ROC-AUC, Brier loss and calibration; retain all experiment results.
5. Backtest eligible new predictions after costs; assess Sharpe, drawdown, uncertainty and robustness across periods.
6. Finish remaining interface/account/game checks, measure responsiveness and conduct user testing.

- Supervisor discussion: agree on whether the main contribution is prediction quality, decision support, or their comparison; agree on the final evaluation protocol and deadline.
- Supporting detail: [technical progress brief](docs/SUPERVISOR_PROGRESS_2026-09-28.md), [research sources](docs/research/RESEARCH_RECOMMENDATIONS.md), [remaining roadmap](docs/IMPROVEMENT_ROADMAP.md), [demo verification](docs/DEMO_VERIFICATION_2026-09-27.md).
