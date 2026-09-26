# Stage 2: data and forward evaluation

Implemented locally on 2026-09-26. Existing models, prediction CSVs and historical
feature datasets were not regenerated during this stage. The published Stage 1
backtest remains the same experiment. Cloud rollout and fresh sentiment data are
still outstanding; this stage does not claim an accuracy improvement.

## Findings from the saved data

The reproducible audit is `backend/data_audit.py`; its output is
`docs/data_audit.json`. It records source hashes, coverage, calendar gaps and
feature comparison errors.

| Item | Finding |
| --- | --- |
| Finance dataset | 104,076 rows, 44 tickers, 2015-05-01 through 2024-12-20 |
| Duplicate ticker/date keys | 0 |
| Missing NYSE sessions inside each ticker's own observed range | 0 |
| GDELT source | 2,127 daily rows, 2026-01-22 through 2026-04-23; no training overlap |
| WSB sentiment/emotions | 11,213 posts each; end 2021-08-31 |
| WSB-covered training rows | 3,736 of 104,076 (about 3.59%) |
| Text coverage during 2023-2024 | All existing GDELT, WSB and emotion features are zero |
| Model feature comparison | All 30 inputs pass absolute tolerance 0.00001 on 47,608 paired rows |

The feature comparison covers 2020-09-03 through 2024-12-20, after **every**
constituent has 300 prior observations. Warming up only the stock being compared
is insufficient: a newly listed constituent changes the cross-sectional RSI
median for the other stocks. The maximum observed absolute difference is
2.4023e-8. The audit also compares 13 individual indicators across all tickers.
It replays saved OHLCV and saved sector returns; it does not independently prove
the original provider's corporate-action adjustments or sector-return construction.

Historical adjustment metadata is absent. The fixed surviving universe also
does not establish point-in-time membership. These remain limitations of the
existing backtest, even though its accounting and timing have been corrected.

## Live inference contract

- Shared RSI, MACD, Bollinger and ATR implementations serve fallback historical
  feature construction and live inference. Existing saved features remain intact.
- Download two years of adjusted daily OHLCV. Require 300 recent complete NYSE
  sessions and the complete trained stock universe for cross-sectional statistics.
- Exclude unfinished candles. A daily bar becomes eligible 20 minutes after the
  scheduled close. Holidays, early closes and DST use `pandas-market-calendars`
  5.4.0 and its [NYSE schedule API](https://pandas-market-calendars.readthedocs.io/en/latest/usage.html).
- Missing required inputs, sector prices, stale sessions and expired failed
  refreshes produce an unavailable signal. They are not replaced with zeros.
  A failed fetch is cached for 60 seconds to avoid repeating an entire universe
  download for each ticker. Successful panels have a 15-minute cache.
- Apply the saved isotonic/Platt calibrator. Keep the existing raw-score >= 0.5
  direction rule so this stage does not silently redefine the model experiment.
- `probability_up` is calibrated P(up) in [0,1]. The compatibility `confidence`
  field is P(up) in percent. `direction_confidence` is P(up) for BUY or 1-P(up)
  for SELL, in percent. It can be below 50% when raw and calibrated scores disagree.
- Return the model artifact SHA-256, feature-code version, actual feature vector,
  data cutoff, fetch time, generation time, horizon and target date. The legacy
  artifact lacks an embedded training cutoff; its exact training provenance still
  needs the versioned model registry planned for Stage 3.

These remain educational model outputs. The currently served model uses finance
features only; showing news on the dashboard does not make its predictions
sentiment-aware.

## Forward evaluation contract

`backend/prediction_tracker.py` uses the new `live_predictions_v2` table with
protocol `nyse-close-5-v2`. The old `live_predictions` table is retained unchanged
and excluded from v2 accuracy because its horizons cannot be trusted.

1. Record a prediction only after a completed close plus 20 minutes and before
   the next NYSE open. Missed forecasts are not backfilled after that open.
2. Freeze the fifth subsequent NYSE session and its scheduled closing timestamp.
   Store the model/code versions, probabilities, input vector and source times.
3. Resolve due rows independently of whether today's model inference succeeds.
   Fetch the original and exact target-session adjusted closes together, on the
   same current Yahoo adjustment basis. A later close is never substituted.
4. Preserve the original recorded price. Save the re-fetched entry close as
   `resolution_entry_close`, plus target close and return, so corporate-action
   rebasing is visible. A missing target quote leaves the record pending and
   flags the scheduled job. Equal closes count as SELL, matching the label rule.
5. Use atomic insert-if-absent for `(ticker, predicted_date, protocol_version)`.
   Count only returned inserts; duplicates are counted separately. A deployment
   cannot replace a day's first prediction with a different model's forecast.
6. Return HTTP 503 on partial job failures so GitHub Actions does not report
   success while every ticker failed. Holidays/out-of-window runs can skip safely.

The migration restricts public users to SELECT, gives server-generated recording
timestamps, and freezes forecast fields and already-resolved outcomes through a
trigger. Database owners and service administrators remain trusted: this is an
auditable application record, not cryptographic proof against its operators.

The summary paginates the complete requested prediction-date window (default
30 calendar days). It reports accuracy, pending/resolved counts, same-sample
always-UP accuracy and probability Brier score. Five-day outcomes overlap; do not
treat these rows as independent trials or interpret direction accuracy as Sharpe.
Use the Backtesting page for strategy returns, costs and risk-adjusted metrics.

## Interface changes

Market live mode shows only the single live XGBoost signal, its data session and
target date. Historical per-model votes, agreement and risk scores are hidden in
live mode. A failed live fetch is shown as unavailable instead of a 2024 signal.
The Track Record page explains the five-session horizon, displayed time window,
legacy exclusions, direction confidence and classification baseline. It displays
the target session and actual recording timestamp for each record.

## News handling and the next experiment

The collector now preserves GDELT discovery timestamps and actual collector
retrieval timestamps. A `feature_session` is the first session closing at or after
retrieval. GDELT discovery time is not relabeled as publication time. Refreshes
retain previous observations and preserve the earliest timestamped copy of a URL;
failed refreshes keep existing data. Legacy date-only rows have unknown availability.

Future feature builds retain observation flags before numerical imputation and
explicitly mark the legacy date-based text join as timing-unverified. These flags
have **not** been added to existing trained models. The existing sentiment scorer
and builder still aggregate legacy daily dates: wiring timestamped article scores
through a separate experiment dataset is required before a causal news-effect
claim or a new sentiment model. Do not shift 2026 news backwards into 2024.

Next, freeze a new chronological evaluation window before tuning. Finance-only
research can proceed with a versioned updated price snapshot. A sentiment
comparison requires either timestamped historical text overlapping that window,
or a prospective collection period. WSB-only experiments before September 2021
are a separate historical ablation, not evidence about the 2023-2024 test period.

## Validation and rollout

Local checks: 39 automated tests passed; frontend production build passed.
Tests cover calendar boundaries, intraday candles, incomplete universes, stale
caches, calibration, missing features, exact target outcomes, missed jobs,
duplicate inserts, database failure and summaries beyond 500 rows. Existing
backtesting tests continue to pass. API smoke checks verified the partial-job
503 response and explicit unavailable track-record response.

A read-only live Yahoo check returned AAPL with data session 2026-09-25, target
2026-10-02, calibrated P(up) 0.5519455733 and direction confidence 55.2%. No
forecast was inserted by this check. This is a connectivity/inference check,
not a measured out-of-sample outcome.

The initial Supabase read-only check failed with `ConnectError`. A follow-up on
27 September reached Supabase and confirmed that the v2 table is missing from
the API schema (`PGRST205`). The migration now passes 11 isolated PostgreSQL WASM
tests, including permissions, RLS and immutable outcomes; it has **not** been
applied or integration-tested on hosted Supabase. No database DDL connection or
management token is configured locally. Render still serves the older backend.
No changes have been pushed or deployed. Browser visual QA remains outstanding.
The frontend still reports its existing large-bundle warning.
See [current rollout evidence and instructions](LIVE_ROLLOUT.md).

Deployment order:

1. Apply `backend/supabase/002_live_predictions_v2.sql` in Supabase SQL Editor.
   This additive migration retains the legacy table and does not alter user data.
2. Deploy backend dependencies/code and frontend together through the existing
   GitHub/Render/Vercel connections.
3. Verify public SELECT and rejected client writes; verify insert timestamps,
   duplicate handling, immutable fields and exact-date resolution against the
   actual database before treating the record as validated production evidence.
4. Observe the next after-close scheduled run, then its fifth-session resolution.
   Keep failed/missing coverage visible when reporting results.

Reproduce locally from the project root (PowerShell):

```powershell
.\backend\.venv\Scripts\python.exe backend/data_audit.py
Push-Location backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Pop-Location
Push-Location frontend
npm run build
Pop-Location
```
