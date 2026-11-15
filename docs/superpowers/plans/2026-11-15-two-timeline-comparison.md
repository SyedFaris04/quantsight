# Two-Timeline Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the four served models twice, trained from 2015 and from 2005, from one frozen price download. Add a crisis-year walk-forward study. Show both in a `/timeline` page and through a Dashboard switch.

**Architecture:**
- A new frozen price snapshot (v2, 2004–2025) feeds three things: a timeline feature build that reuses `build_features.py` steps, the existing training scripts (given CLI options and a fixed split date), and a walk-forward study that reuses `research/development_models.py` helpers.
- One comparison script turns every output into `timeline_comparison.json`. The API and the frontend only read that file and the timeline prediction CSVs.

**Tech Stack:** Python 3.12 (`backend/.venv`), pandas, xgboost 3.2, torch 2.10 (CPU), yfinance 1.2, FastAPI, unittest. React 18, Vite, Tailwind, Recharts, node --test.

**Spec:** `docs/superpowers/specs/2026-11-15-longer-price-history-design.md`

## Global Constraints

- Python commands run from `backend/` with `.venv/Scripts/python.exe`. The suite command is `.venv/Scripts/python.exe -m unittest discover -s tests`. The baseline is 206 tests passing.
- Download window: `download_start` 2004-01-01, `download_end_exclusive` 2026-01-01, `holdout_start` 2025-01-01. Never read `candidate_holdout/`.
- Timeline starts: "2015" = first training date 2015-05-01. "2005" = first training date 2005-01-03.
- Fixed test window: 2023-02-06 to 2024-12-20. Bars used for features are strictly before 2024-12-31, matching the old notebook 14 end date.
- Current production files in `data/models/*.pkl|*.pt` and `data/predictions/*.csv` are never overwritten.
- Timeline outputs:
  - `data/models/timelines/{2015,2005}/`
  - `data/predictions/timelines/{2015,2005}/` (seed 42 files, committed)
  - extra LSTM seeds in `data/research/timelines/seeds/` (gitignored)
- Every protocol JSON is written and hashed before any fit. Fit budgets are asserted.
- The live tracker, `/overview`, Market and Detail keep using production predictions.
- Work on branch `feature/timeline-comparison`. Never stage the user's unrelated changes: the deleted pptx files, `QuantSight_Project_Documentation.pdf` and `docs/presentations/`.

## Review Focus

1. **Old Yahoo data with gaps.** A pre-2015 internal session gap must make the v2 audit fail, then stop, with the gaps reported to the user. It must not be silently filled (Task 1 test).
2. **Late-listing instruments in the 2005 timeline.** META, TSLA, ABBV, PYPL, SNAP, MRNA and UBER must have no rows before listing plus indicator warm-up. Cross-sectional features use only the instruments present that day (Task 2 test).
3. **A shared URL with a bad `?timeline=` value.** The frontend falls back to 2015 instead of showing an error (Task 8 test). The API returns 422 for it (Task 7 test).
4. **Timeline files missing on the server.** `/dashboard` returns 503 with a clear message. Market, `/overview` and `/health/ready` are unaffected (Task 7 test).
5. **The test window drifts.** Adding older rows must not change the test rows (Task 3 test).

---

### Task 1: Price snapshot v2 (2004–2025)

**Files:**
- Create: `backend/research/protocols/fresh_price_snapshot_v2.json`
- Modify: `backend/research/freeze_prices.py` (`PROTOCOL` constant, `freeze()`, `__main__`)
- Test: `backend/tests/test_fresh_validation.py` (append)

**Interfaces:**
- Produces: `freeze(protocol_path: Path = PROTOCOL) -> dict` (the manifest). The snapshot directory is `data/research/fresh_prices/<attempt_id>/`. `load_development(directory, ticker)` is unchanged.

- [ ] **Step 1: Write the protocol.** Copy `research/fresh_data_protocol.json` and change:
  - `protocol_id` → `fresh_price_snapshot_v2`
  - `download_start` → `2004-01-01`
  - `purpose` → mention the two-timeline study
  - drop `development_folds` (the walk-forward study defines its own)
  - add the limitation "Pre-2015 history extends survivorship of a fixed 2026 universe."

  Keep `universe`, `context_tickers`, `download_options`, `holdout_start`, `download_end_exclusive` and `development_end_exclusive`.
- [ ] **Step 2: Write the failing test.** Add `test_internal_gap_before_2015_fails_audit`. Build a frame with 2004–2006 NYSE sessions minus one session in 2005. `audit_prices(frame, expected, '2025-01-01')` must give `valid` False, and the missing date must appear in `missing_internal_sessions`. Also add `test_freeze_reads_given_protocol`: patch `yfinance.download` to return an empty frame, call `freeze(tmp_protocol)`, and assert that `request.json` stores that protocol's `protocol_id`.
- [ ] **Step 3: Run it.** Run `.venv/Scripts/python.exe -m unittest tests.test_fresh_validation -v`. Expected: the new freeze test FAILS (`freeze` takes no argument).
- [ ] **Step 4: Implement.** `freeze(protocol_path=PROTOCOL)` reads the given path and hashes that path. `__main__` adds `--protocol` (Path, default `PROTOCOL`). Also patch `ROOT / 'data/research/fresh_prices'` in the test using a temporary directory.
- [ ] **Step 5: Run the suite.** Run `.venv/Scripts/python.exe -m unittest discover -s tests`. Expected: OK, 208 tests.
- [ ] **Step 6: Download.** Run `.venv/Scripts/python.exe -m research.freeze_prices --protocol research/protocols/fresh_price_snapshot_v2.json`.
  - Expected: the last line has `"data_quality_passed": true`.
  - **If false:** stop, list the failing tickers and their `missing_internal_sessions`, and ask the user. Do not fill gaps.
  - Record the `attempt_id` as `SNAPSHOT_V2` for later tasks.
- [ ] **Step 7: Check drift against v1.** Run a one-off check from the shell: load `development/<T>.csv` from v1 `20260927T141409402344Z` and v2 for every ticker, and compare `Adj Close` on shared dates. Record the max absolute relative difference per ticker in `docs/research/fresh_price_audit_<SNAPSHOT_V2>.json` under `v1_drift` (edit after it is written). Expected: the dates are identical for 2015–2024.
- [ ] **Step 8: Commit.** Stage `backend/research/protocols/fresh_price_snapshot_v2.json`, `backend/research/freeze_prices.py`, `backend/tests/test_fresh_validation.py` and `docs/research/fresh_price_audit_*.json`. Message: `research: freeze 2004-2025 price snapshot v2`.

### Task 2: Timeline feature build

**Files:**
- Modify: `backend/build_features.py`. Extract `finance_frame(data, sector_file=SECTOR_FILE)` from `main()` steps 1a2–3, and `sentiment_frame(finance_df)` from steps 4–6. `add_sector_relative_features(df, sector_file=SECTOR_FILE)`. `main()` keeps identical behaviour by calling them.
- Create: `backend/research/timeline_features.py`
- Test: `backend/tests/test_timeline_features.py`

**Interfaces:**
- Produces:
  - `snapshot_ohlcv(snapshot: Path, tickers: list[str], end_exclusive: str) -> DataFrame` with columns `Date, Open, High, Low, Close, Volume, Ticker`. O/H/L/C are scaled by `Adj Close / Close`, and rows with any nonpositive O/H/L/C/Volume are dropped (same as notebook 14).
  - `sector_returns(snapshot: Path, end_exclusive: str) -> DataFrame` with columns `date, sector_etf, sret_5, sret_20`.
  - `build(snapshot: Path, out_dir: Path) -> dict` writes `out_dir/features_finance.csv`, `features_sentiment.csv` and `sector_returns.csv`. It returns `{'rows', 'tickers', 'first_date_by_ticker', 'columns'}`.
  - CLI: `python -m research.timeline_features --snapshot <dir> --out data/research/timelines`.
- Column rule: the finance columns equal the current `data/processed/features_finance.csv` header, in that order. Extra `add_indicators` columns (`daily_return`, `high_low_range`, `close_open_gap`, `volume_ratio`) are dropped. Rows with NaN or inf in any kept column are dropped.

- [ ] **Step 1: Write the failing tests.**
  - `test_adjusts_ohlc_by_adj_close_factor`: one ticker with Close 100, Adj Close 50 and Open 98. Expect Open 49.
  - `test_drops_bars_on_or_after_end`: no Date ≥ 2024-12-31.
  - `test_late_listing_has_no_warmup_rows`: a ticker listed at session 300 of a synthetic 2-ticker, 600-session panel built via `build()` on a temporary fake snapshot (manifest and request with hashes, as in `test_fresh_validation`). Its first `features_finance` date must be ≥ session 300 + 49, and the other ticker's `rsi_rel_market` on dates before listing must equal the value for a one-ticker panel (median of itself → 0).
  - `test_finance_columns_match_production_header`: `build()` output columns equal the list read from the production CSV header.
  - `test_main_unchanged`: `finance_frame` on a small synthetic frame equals the old inline result. Snapshot the expected columns and a row as literals.
- [ ] **Step 2: Run them.** Run `.venv/Scripts/python.exe -m unittest tests.test_timeline_features -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement the `build_features.py` extraction, then `research/timeline_features.py`.**
  - `build()` calls `snapshot_ohlcv` → per-ticker `add_indicators` → `dropna(subset=['rsi','macd','bb_mid','signal'])` → `finance_frame(data, out_dir/'sector_returns.csv')` → column/NaN rule → `sentiment_frame`.
  - Universe and tickers come from the snapshot `request.json` protocol.
- [ ] **Step 4: Run the suite.** Expected: OK.
- [ ] **Step 5: Build.** Run `.venv/Scripts/python.exe -m research.timeline_features --snapshot data/research/fresh_prices/<SNAPSHOT_V2> --out data/research/timelines`.
  - Expected: the minimum date is in early 2004-Q2 (after warm-up), the maximum is 2024-12-20, and there are 44 tickers.
  - Check that 2015-05-01+ rows ≈ 104k (production has 104,076).
- [ ] **Step 6: Commit.** Stage `backend/build_features.py`, `backend/research/timeline_features.py` and `backend/tests/test_timeline_features.py`. Message: `feat: build timeline features from frozen snapshot`.

### Task 3: Fixed split and training CLI options

**Files:**
- Modify: `backend/cv_utils.py`, `backend/train_xgboost.py` (`load_and_split`, `train_variant`, `main`), `backend/train_lstm.py` (`train_variant`, `main`, `SEED` use)
- Test: `backend/tests/test_timeline_split.py`

**Interfaces:**
- Produces:
  - `cv_utils.fixed_split(df, split_date, start=None) -> (trainval_df, test_df)`. Rows with `date < start` are removed first. `trainval` = `date <= purge_train_test_boundary(df.date, split_date)` and `test` = `date > split_date`. The purge uses all dates in `df`.
  - `make_train_calib_split(dates, calib_frac=0.15, split_date=None)`: when `split_date` is given, it replaces the quantile.
  - Both training scripts accept these CLI options:
    - `--features-dir` (default `data/processed`)
    - `--start` (default None)
    - `--split-date` (default None = old 80% quantile, so a plain run is unchanged)
    - `--out-models`, `--out-predictions` (defaults: production dirs)
    - LSTM only: `--seed` (default 42) and `--calib-split` (default None)
  - `update_metrics_file` writes to `<out-predictions>/model_metrics.json`.
- Constants: `TIMELINE_SPLIT_DATE` is the exact value `df["date"].quantile(0.80)` gives on the production `features_finance.csv`. Compute and print it in Step 3, then pin it in `backend/research/protocols/timelines.json` together with:
  - `starts`: `{"2015": "2015-05-01", "2005": "2005-01-03"}`
  - `lstm_seeds`: `[42, 43, 44]`
  - `lstm_calib_split`: the value `make_train_calib_split` yields on the 2015 timeline's trainval dates, pinned so both timelines early-stop on the same window

- [ ] **Step 1: Write the failing tests.**
  - `test_older_history_does_not_change_test_rows`: a synthetic two-ticker frame covering 2010–2024. `fixed_split(df, '2023-02-03', start='2015-05-01')[1]` must equal `fixed_split(df, '2023-02-03', start='2010-01-04')[1]`.
  - `test_purge_gap`: the max trainval date is 5 trading days before the split date.
  - `test_calib_split_override`: `make_train_calib_split(dates, split_date=X)` puts no calibration row before X minus the embargo.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Print `TIMELINE_SPLIT_DATE` and confirm that production's first test date is 2023-02-06. Write `research/protocols/timelines.json`.
- [ ] **Step 4: Run the suite.** Expected: OK. Also confirm that a plain `python train_xgboost.py --help` lists the new options and that the defaults point at the production paths.
- [ ] **Step 5: Commit.** Message: `feat: fixed test split and timeline options for training scripts`.

### Task 4: Train both timelines

**Files:** none modified. This task produces artifacts only.

- [ ] **Step 1: XGBoost.** For each `T` in {2015, 2005}, run `.venv/Scripts/python.exe train_xgboost.py --features-dir data/research/timelines --start <start_T> --split-date <TIMELINE_SPLIT_DATE> --out-models data/models/timelines/T --out-predictions data/predictions/timelines/T`. Expected: test rows are identical across T (log line "Test :"), and the first date is 2023-02-06.
- [ ] **Step 2: LSTM, in the background.**
  - Seed 42 for each T writes to the timeline dirs.
  - Seeds 43 and 44 write to `data/research/timelines/seeds/T/seed<N>/`.
  - Always pass `--calib-split <lstm_calib_split>`.
  - Run all six jobs serially in one background command, because CPU threads are shared.
- [ ] **Step 3: Sanity check.** Rebuilt-2015 `xgb_finance` vs production `data/predictions/xgb_finance_predictions.csv` on common ticker/date rows. Report the row count, the direction agreement and the accuracy delta to the user. If agreement is below 80%, stop and investigate before Task 6.
- [ ] **Step 4: Commit.** Stage `data/models/timelines/` and `data/predictions/timelines/`. Message: `models: train served models from 2015 and 2005 on snapshot v2`.

### Task 5: Crisis walk-forward study

**Files:**
- Create: `backend/research/crisis_walkforward_protocol.json`, `backend/research/crisis_walkforward.py`
- Test: `backend/tests/test_crisis_walkforward.py`

**Interfaces:**
- Consumes: `research.development_models.{prepare_inputs, model_factory, raw_logit, reliability}`, `research.news_pilot.metrics` and `research.validation.purged_segments`.
- Produces:
  - `walkforward_folds(arms: dict) -> list[dict]`. Each fold has `name` `"<arm>_<year>"`, `arm`, `year`, `fit_start`, `fit_end_exclusive` = Jan 1 of year−1, `calibration_end_exclusive` = Jan 1 of year, and `evaluation_end_exclusive` = Jan 1 of year+1.
  - `episode_metrics(predictions, episodes, candidate) -> list[dict]`
  - `paired_difference(frame, a, b, metric, block=20, reps=1000, seed=42) -> {'difference', 'interval_95'}` (a minus b; `metric` is `'brier'` or `'accuracy'`; date-block circular bootstrap)
  - `run(snapshot, output_root) -> dict` writes `data/research_reports/crisis_walkforward.json` with: `results` (per fold × candidate metrics), `episodes` (per arm × candidate), `arm_comparison` (2017–2024, per candidate, `paired_difference` for Brier and accuracy), `rows_by_year`, `tickers_by_year` and `fits`.
- Protocol values:
  - `snapshot_id`: SNAPSHOT_V2
  - copy `features`, `models`, `calibration`, `horizon_sessions` and `holdout_start` from `development_model_protocol.json`
  - `arms`: `{"2005": {"fit_start": "2005-01-01", "years": [2008..2024]}, "2015": {"fit_start": "2015-01-01", "years": [2017..2024]}}`
  - `fit_budget`: `{"predictors": 75, "calibrators": 75}`
  - `episodes`:
    - `gfc` 2008-09-01 to 2009-03-31
    - `debt_ceiling_2011` 2011-07-22 to 2011-10-03
    - `china_oil_2015` 2015-08-10 to 2016-02-11
    - `q4_2018` 2018-10-01 to 2018-12-24
    - `covid_2020` 2020-02-19 to 2020-04-30
    - `rate_shock_2022` 2022-01-03 to 2022-10-12
  - Baselines: `fit_prior` and `always_up`.

- [ ] **Step 1: Write the failing tests.**
  - `test_folds`: 25 folds. The 2005 arm's 2008 fold has fit [2005-01-01, 2007-01-01), calibration [2007, 2008) and validation [2008, 2009).
  - `test_overlapping_years_have_identical_validation_rows`: on a synthetic panel, the 2005 and 2015 arms' 2018 validation (ticker, date) sets are equal.
  - `test_paired_difference_identical_is_zero`: the interval is [0, 0].
  - `test_paired_difference_sign`: a is better in Brier, so the difference is negative.
  - `test_episode_spans_two_folds`: the `gfc` rows include 2008 and 2009 validation dates.
  - `test_budget_asserted`: patch the protocol budget to 74 and expect a ValueError.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Follow `development_models.run`: registration written before features load, then per fold `purged_segments`, fit, sigmoid calibration, and outputs `fit_prior`, `always_up`, `<model>_raw` and `<model>_sigmoid`. Save the fold predictions CSV under the run dir, plus artifact hashes. Assert the same validation index across arms per overlapping year.
- [ ] **Step 4: Run the suite.** Expected: OK.
- [ ] **Step 5: Run the study.** Run `.venv/Scripts/python.exe -m research.crisis_walkforward --snapshot data/research/fresh_prices/<SNAPSHOT_V2>` in the background. Expected: `fits` = 75/75.
- [ ] **Step 6: Commit.** Stage the protocol, the module, the test and `data/research_reports/crisis_walkforward.json`. Message: `research: crisis-year walk-forward for 2005 and 2015 training starts`.

### Task 6: Timeline comparison report

**Files:**
- Create: `backend/research/timeline_comparison.py`
- Test: `backend/tests/test_timeline_comparison.py`

**Interfaces:**
- Consumes: Task 4 outputs, `data/research/timelines/features_*.csv`, `research/protocols/timelines.json`, `crisis_walkforward.json`, `paired_difference` (Task 5), `reliability` (development_models), and `backtesting.engine.{BacktestConfig, simulate, performance_metrics}`.
- Produces:
  - `agreement(a: DataFrame, b: DataFrame) -> {'rows', 'agree_rate', 'disagree_rows', 'a_correct_when_disagree', 'b_correct_when_disagree'}`
  - `ensemble(frames: dict[str, DataFrame]) -> DataFrame`: row-wise mean `confidence` over the four models on common rows; `predicted_signal = confidence >= 50`.
  - `top_shap(model_path, features_csv, split_date, start) -> list[{'feature', 'mean_abs'}]` (top 10, `pred_contribs=True` on scaled test rows)
  - `build() -> dict` writes `data/research_reports/timeline_comparison.json` with this schema:

```
schema_version 1, status "exploratory", created_at, input_hashes{path: sha256}
test_window{start, end, rows}
timelines{"2015"|"2005": {training{start, end, rows, tickers_by_year{year: n}, episodes_included[]},
          models{key: {accuracy, balanced_accuracy, auc, brier, buy_fraction, reliability[], lstm_seeds?{accuracy:{mean,min,max}, brier:{...}}}},
          backtest{ensemble{total_return, sharpe, max_drawdown, turnover}, spy{...}, evaluation{start,end,sessions}},
          shap{xgb_finance:[...], xgb_sentiment:[...]}}}
baselines{always_up_accuracy, train_prior_brier{"2015","2005"}}
differences{key: {accuracy: paired_difference, brier: paired_difference}}   # 2005 minus 2015
agreement{key: agreement(...)}
shap_rank_changes{key: [{feature, rank_2015, rank_2005}]}
production_check{key: {rows, agree_rate, accuracy_delta}}   # rebuilt 2015 vs current production
crisis (copied summary of crisis_walkforward.json: per-year xgboost_sigmoid + baselines per arm, episodes, arm_comparison)
limitations[]  (spec "Constraints and known limits", one sentence each)
```

- [ ] **Step 1: Write the failing tests.**
  - `test_agreement_counts`: a 6-row toy case with a hand-computed disagree split.
  - `test_ensemble_common_rows_only`: one model missing a row means that row is excluded.
  - `test_schema_keys`: `build()` run against a tiny fixture tree (patch the module paths) produces every top-level key above, and `json.dumps(allow_nan=False)` succeeds.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** For the backtest, prices come from the timeline `features_finance.csv` (`date`, `ticker`, `Open`, `Close`). Follow the start/end selection in `backtesting/report.generate`, and use `BacktestConfig()` defaults.
- [ ] **Step 4: Run the suite.** Expected: OK.
- [ ] **Step 5: Generate.** Run `.venv/Scripts/python.exe -m research.timeline_comparison`. Expected: the file is written, and both timelines have 4 models plus a backtest.
- [ ] **Step 6: Commit.** Stage the module, the test and the report JSON. Message: `research: two-timeline comparison report`.

### Task 7: API

**Files:**
- Modify: `backend/main.py` (`load_predictions`, startup cache, `get_overview`, `get_dashboard`), `backend/research/routes.py`
- Test: `backend/tests/test_dashboard_timeline.py` (follow `test_dashboard_evidence.py` for patching `main._cache`)

**Interfaces:**
- Produces:
  - `load_predictions(directory: Path = PREDICTIONS_DIR)`. Legacy fallbacks apply only for the default directory.
  - The startup cache gets `_cache["timeline_latest"] = {"2015": latest_rows, "2005": latest_rows}`. Build it the same way as `latest_prediction_rows`. A load failure logs a warning and leaves that key absent. Readiness is unaffected.
  - `get_latest_signals_for_ticker(ticker, latest=None)`. `None` means production.
  - `get_overview(latest=None)`. The cache key is `"overview_rows"` for production and `f"overview_rows_{T}"` for timelines.
  - `GET /dashboard?timeline=2015|2005` (`Literal`, default `"2015"`) adds `timeline` and `backtest` (that timeline's `backtest` block from `timeline_comparison.json`, or `null`). If the timeline is not loaded, it returns 503 with `"Timeline predictions are not available on this server."`.
  - `GET /research/timeline-comparison` returns `{'available': bool, 'report'?}`. If the report is invalid, it returns 503. Validation: `schema_version == 1` and both timelines present.

- [ ] **Step 1: Write the failing tests.**
  - `test_default_equals_2015`
  - `test_2005_reads_2005_rows`: fixture rows differ in direction, so the KPIs differ.
  - `test_unknown_timeline_422`
  - `test_missing_timeline_503_and_overview_still_ok`
  - `test_timeline_comparison_route_available_and_invalid`
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the suite.** Expected: OK. Then start uvicorn locally and run `curl "localhost:8001/dashboard?timeline=2005"`. Expected: 200 with `"timeline":"2005"`.
- [ ] **Step 5: Commit.** Message: `feat: timeline option for dashboard and comparison route`.

### Task 8: Frontend

**Files:**
- Create: `frontend/src/lib/timeline.js`, `frontend/src/pages/TimelineComparison.jsx`, `frontend/tests/timeline.test.js`
- Modify: `frontend/src/pages/Dashboard.jsx`, `frontend/src/App.jsx` (lazy route `/timeline`), `frontend/src/components/Sidebar.jsx` (link "Timelines", reusing an existing `Icon` name)

**Interfaces:**
- Produces:
  - `TIMELINES = ["2015", "2005"]`
  - `parseTimeline(search: string) -> "2015"|"2005"` (anything else → `"2015"`)
- Dashboard:
  - The header has a segmented control: `role="radiogroup"` with two `role="radio"` buttons, "Trained from 2015" and "Trained from 2005", using `aria-checked`. Arrow keys move the selection.
  - It sets `?timeline=` via `useSearchParams` (`replace: true`). `useApi(\`/dashboard?timeline=${t}\`)`.
  - `BacktestPreview` takes `backtest` from the dashboard data instead of `/backtest/summary`. It is labelled "Four served models averaged".
  - The sentiment and news panels get the caption "Not a model output; the same for both timelines."
  - A link reads "Compare timelines →" `/timeline`.
- `/timeline` page: one section per comparison-table row in the spec. Order:
  1. training data
  2. prediction quality table with baselines and difference intervals
  3. calibration (Recharts line chart of reliability bins, both versions)
  4. agreement
  5. trading simulation table
  6. top-10 features side by side with rank changes
  7. crisis years: a Recharts line chart of per-year Brier for both arms, with episode years marked by `ReferenceArea`, plus an episode table
  8. limitations

  Loading, error-with-retry and unavailable states follow `Compare.jsx` patterns.

- [ ] **Step 1: Write the failing test.** In `tests/timeline.test.js`:
  - `parseTimeline("?timeline=2005") === "2005"`
  - `parseTimeline("?timeline=1999") === "2015"`
  - `parseTimeline("") === "2015"`
- [ ] **Step 2: Run** `npm test` in `frontend/`. Expected: FAIL.
- [ ] **Step 3: Implement** `lib/timeline.js`, then the Dashboard, route, sidebar and page.
- [ ] **Step 4: Run** `npm test && npm run build`. Expected: pass, and the build succeeds.
- [ ] **Step 5: Browser check.** With the backend and `npm run dev` running, use Playwright (as in `backend/research/verify_dashboard_ui.py`):
  - the dashboard switch changes the KPIs
  - reload keeps `?timeline=2005`
  - arrow keys switch
  - `/timeline` renders all 8 sections
  - no console errors at 1440px and 390px

  Inspect the screenshots, then save them to `backend/data/research/timeline_ui/`.
- [ ] **Step 6: Commit.** Message: `feat: dashboard timeline switch and timeline comparison page`.

### Task 9: Docs

**Files:**
- Create: `docs/research/TIMELINE_COMPARISON.md` (method, the numbers from the report, limitations, and how to reproduce: the commands from Tasks 1–6)
- Modify: `docs/IMPROVEMENT_ROADMAP.md` (Stage 3 entry), `README.md` (one paragraph and a link)

- [ ] **Step 1: Write.** Every number is copied from `timeline_comparison.json` and `crisis_walkforward.json`, not rounded differently.
- [ ] **Step 2: Final verification.** Run the full Python suite and `npm test && npm run build`. Expected: all pass.
- [ ] **Step 3: Commit.** Message: `docs: two-timeline comparison results`.
