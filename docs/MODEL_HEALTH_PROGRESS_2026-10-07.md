# Model-health progress — 7 October 2026

## What was added

- **Evaluation → Model health**: a new page for the live finance XGBoost model.
- A read-only `/model-health?days=30` API; window choices are 7, 30 and 90 days.
- Latest-session saved coverage, the latest recorded date and the recording-window state.
- Separate snapshots for each forecast date, model artifact, feature implementation and calibration method.
- Record checks for missing/non-finite model inputs, invalid probabilities, direction mismatches and timestamp problems.
- Per-ticker historical input-range checks for all 30 required features.
- Raw and calibrated probability ranges, fixed histograms, distinct values and largest tied groups.
- An aggregate JSON export of the displayed monitoring window. Individual feature vectors and account data are not exported.
- Clear empty, unavailable-reference, unsupported-model and retry states.
- A lazy-loaded page with collapsed technical tables and mobile/keyboard support.

## Where the reference comes from

- Source: existing `features_finance.csv`, dated **1 May 2015–30 December 2022** after filtering/warm-up.
- **82,252 rows, 44 tickers, 30 inputs**.
- Model feature names come from the trusted saved `xgb_finance.pkl` bundle.
- Quantiles are calculated separately for each ticker and feature using NumPy's linear method.
- Required percentiles: 1st, 25th, 50th, 75th and 99th. The monitoring band uses the 1st and 99th.
- Reference artifacts are checked against the feature source, model and protocol hashes. Text hashes tolerate Git's LF/CRLF conversion.
- CI rebuilds the reference and compares it with the committed artifact.
- The saved model has no training-date manifest. This is labelled a **historical development reference**, not a proven reconstruction of its training distribution.
- The separate 2025 candidate holdout was not accessed.

## Fixed review rules

- Rules were committed locally as `0dc7cc7` before building the reference or calculating the new monitoring results. This is a local registration, not an external preregistration.
- An input-range review needs at least **20 finite observations** and **20% outside** the same ticker's historical band.
- Values exactly on the band boundary are inside.
- A probability-spread flag needs at least **20 eligible forecasts** and a span of at most **0.005**, or 0.5 percentage points.
- Unsupported model artifacts are not compared with this input reference.
- Small samples do not receive range/spread flags.
- Missing inputs are not replaced with zeros.
- Duplicate keys or non-session dates make the report unavailable rather than silently duplicating results.
- Exact distinct values and largest tied counts are additional descriptive summaries. They were added after the first span results; the registered flag thresholds were not changed.
- Range flags and probability-spread flags are practical review rules, not significance tests, automatic retraining triggers or accuracy claims.

## Observed results

Read-only monitoring of **88 forecasts across two dates** on 7 October:

| Recorded date | Records | Input range flags | Raw distinct P(UP) | Calibrated distinct P(UP) | Largest calibrated tied group |
|---|---:|---:|---:|---:|---:|
| 5 October | 44 | 3 | 44 | 4 | 38/44 |
| 28 September | 44 | 3 | 44 | 4 | 40/44 |

- No detected missing required inputs, invalid probabilities, raw-decision mismatches or timestamp issues in these records.
- Flagged inputs on both dates: **ATR**, **20-day price volatility**, and **10-day average trading volume**.
- On 5 October, their outside-band rates were **36.4%, 29.5% and 25.0%** respectively.
- These include absolute dollar/share quantities. Price growth, liquidity changes and market conditions can move them beyond older historical ranges. The flags do not prove bad data or lost model skill.
- On 5 October, raw P(UP) ranged **32.8%–68.0%**, while calibrated P(UP) ranged **55.1%–56.7%**.
- Neither recorded date triggered the fixed 0.5-percentage-point whole-range rule. The distinct/tied counts still show that calibration provides limited ranking detail.
- Isotonic calibration can create ties. The recorded BUY/SELL rule remains the raw 50% threshold; a calibrated probability crossing 50% is not itself a stored-decision error.
- The latest completed session was 6 October, with **0/44 saved forecasts**. Its recording window had closed. The page reports the gap without creating retrospective forecasts.
- The new recovery scheduler still needs a genuine eligible scheduled run. Monitoring does not create or replace forecasts.

## Verification and setup

- **206 backend tests** pass in the serving and clean CI environments.
- **11 frontend tests**, the production build and the runtime verification pass.
- The reference checks pass in both Python environments.
- Browser checks cover actual API/table values, both date/version snapshots, refresh selection, window exports, keyboard scrolling and 390px/320px layouts.
- Isolated browser fixtures cover failed requests/retry, unavailable reference, empty windows, unsupported models, compression, small samples, record issues and export failure.
- Screenshots were inspected; the tables scroll inside their regions without widening the page.
- Uses the existing Supabase forecast table. No SQL migration, new key or extra package is needed.
- Trained models, calibrators, live feature formulas and the feature-code hash are unchanged.
- Local backup: `http://127.0.0.1:3000/model-health`.

The [registered method](MODEL_HEALTH_METHOD.md) links the primary [NumPy quantile reference](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html) and [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).
