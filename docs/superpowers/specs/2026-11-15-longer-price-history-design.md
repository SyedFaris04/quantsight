# Longer price history: two timelines compared — design

Date: 2026-11-15. Status: awaiting approval.

## Intent

The lecturer suggested training and analysing on more years, so the models see
market crises. The current panel starts on 2015-05-01, so it already includes
COVID (2020) and the 2022 bear market. It does not include the 2008 financial
crisis or the 2011 debt-ceiling selloff. This work builds **two versions of the
models, one trained from 2015 and one trained from 2005**, compares them on
identical test rows, and shows on the website what changes and why.

More history is a hypothesis, not a guaranteed improvement. Recent work argues
that long windows can hurt when markets change regime (Capponi et al., 2025,
arXiv:2512.23596). Both outcomes are reported.

## Goals

1. A frozen, hashed daily price snapshot from 2004-01-01 (one warm-up year) to
   2025-12-31 for the 44 instruments and the six sector ETFs. The 2025 candidate
   holdout stays separate and unopened.
2. **Two served-model timelines.** Train the four served models (XGBoost and
   LSTM, finance and sentiment) twice from the same new download, code and
   seeds: once from 2015, once from 2005. Use the same fixed test window,
   2023-02-06 to 2024-12-20.
3. **Crisis walk-forward study.** Fit on all prior years, predict each year,
   with two arms:
   - "from 2005": validation years 2008–2024
   - "from 2015": validation years 2017–2024

   The years 2017–2024 give a paired, year-by-year comparison of the two
   timelines. The 2005 arm alone covers 2008–2016. Results are reported per
   year and for named episodes: 2008 GFC, 2011 debt ceiling, 2015–16
   China/oil, 2018 Q4, 2020 COVID, 2022 rate shock.
4. **A Timeline Comparison page** on the website that shows the two versions
   side by side (contents below).

## What the comparison shows

For each served model, the two versions are compared on the same test rows.

| Area | Measures |
|---|---|
| Training data | Date range, training rows, instruments present each year, which crises are included |
| Prediction quality | Accuracy, balanced accuracy, AUC, Brier loss, next to the always-up and training-prior baselines |
| Certainty of the difference | Paired 20-session block-bootstrap 95% interval on the accuracy and Brier differences |
| Calibration | Reliability bins: when a version says 60%, how often it was right |
| Agreement | Share of test predictions where the versions agree; when they disagree, how often each was right |
| Trading simulation | Net return, Sharpe, max drawdown and turnover from the existing backtesting engine, vs SPY |
| What the model relies on | Top-10 XGBoost features by mean absolute TreeSHAP for each version, and the biggest rank changes |
| Crisis years | Year-by-year Brier and accuracy for both arms from the walk-forward study, crisis years highlighted, baselines shown |

Every panel shows the limits below where they apply.

## Non-goals

- Historical news backfill (Benzinga/FNSPID). That is the next project.
- Opening or scoring 2025.
- New features, architectures or hyperparameter searches.
- Starting earlier than 2005. That would lose GLD, GOOGL, CRM and TLT in the
  early years and add more survivorship. The start year stays a protocol
  field, so this can be revisited.
- Switching the live forecast tracker or the other pages between versions. The
  tracker records immutable forecasts from the current production models, so
  it stays on them. Market and Detail are unchanged. Only the Dashboard gets a
  timeline switch (below).

## Anti-goals

- Calling a difference an improvement when its interval includes zero.
- Silently moving the test window. `train_*.py` currently split at the 80% date
  quantile. With more history, that quantile moves into 2021–22.
- Confounding the comparison. Both timelines come from the same download, so
  vendor adjustment drift cannot masquerade as a training-history effect.

## Constraints and known limits

- **Unbalanced panel.** META, TSLA, ABBV, PYPL, SNAP, MRNA and UBER enter after
  their listing date plus warm-up. About 37 instruments exist in 2008.
- **Survivorship.** All 44 are today's survivors, so the pre-2015 years flatter
  the universe.
- **Sentiment models from 2005.** There is no text data before 2015, so the
  sentiment features are zero-filled with "not observed" flags for 2005–2014.
  The 2005 sentiment versions mostly learn more price history. This is shown on
  the page.
- **Vendor adjustment.** Yahoo/yfinance adjustments are retrieval-time, not
  point-in-time.
- **Examined test window.** 2023–24 has been looked at many times. Production
  comparisons are labelled exploratory; the walk-forward study is the cleaner
  evidence.
- **LSTM randomness.** Run-to-run noise is reported by training each LSTM
  version with 3 seeds.
- **House style.** Protocol JSON is saved and hashed before any fit, a fit
  budget is declared, and artifacts are hashed.

## Approaches considered

1. **Prices from 2005, both timelines built and compared, crisis walk-forward
   (chosen).**
2. **Prices from 2000.** Rejected for now (see non-goals).
3. **Prices plus historical news now.** Deferred to the next project.

The user chose to let the builder decide, then asked for both timelines to be
shown and compared.

## Design

### Data: `fresh_price_snapshot_v2`

- New protocol `backend/research/protocols/fresh_price_snapshot_v2.json`,
  copying v1 with `download_start` 2004-01-01.
- `freeze_prices.py` takes the protocol path. Same audit as v1, plus each
  instrument's first available session.
- v1 is untouched. The 2015–2024 price drift between v1 and v2 is reported.

### Served-model timelines

- `notebooks/14_expand_stocks.py` and `fetch_sector_data.py` take a start date.
  `features_stock.csv` and `features_*.csv` are built once from 2004 and then
  filtered by training start, so both timelines share identical feature values
  on overlapping dates.
- `train_xgboost.py` and `train_lstm.py` take `--start` and `--tag`. The test
  start is fixed at 2023-02-06 with the existing purge. Outputs go to
  `data/models/timelines/{2015,2005}/` and
  `data/predictions/timelines/{2015,2005}/`.
- The rebuilt 2015 version is checked against the current production models
  (agreement rate and metric drift) as a sanity check.
- The current production files are untouched.

### Crisis walk-forward: `crisis_walkforward_v1`

- Reuses `development_models.py`: the same 13 features, three model families,
  previous-year sigmoid calibration, baselines, purging and `paired_interval`.
- The fit budget is 25 folds × 3 families: 75 predictor fits and 75
  calibrators.
- Validation rows must be identical between the two arms in overlapping years.
  This is asserted.

### Comparison report

- `backend/research/timeline_comparison.py` writes
  `data/research_reports/timeline_comparison.json`. It holds every measure in
  the table above, input hashes, and the limits text.
- The backtest uses the existing `backend/backtesting` engine on both versions'
  predictions.
- It is served through the existing research routes.

### Website

- **Dashboard timeline switch.** A two-option segmented control, "Trained from
  2015" and "Trained from 2005", sits in the dashboard header. The default is
  2015.
  - The choice is kept in the URL (`/?timeline=2005`), so it survives a reload
    and can be shared.
  - Backend: `/dashboard?timeline=2015|2005` (default 2015) builds the snapshot,
    KPIs and BUY candidates from that timeline's prediction CSVs. Any other
    value returns 422.
  - The backtest card reads that timeline's numbers from
    `timeline_comparison.json`.
  - News and sentiment panels are not model outputs and don't change. They are
    labelled that way.
  - Both options use the rebuilt timeline predictions, so the switch shows
    exactly the pair compared on `/timeline`. Each option links to that page.
  - The current production prediction files stay on disk for the live tracker
    and the other pages.
- A new `/timeline` page, linked from the sidebar, with one component per row
  of the comparison table. It uses Recharts and the existing card styles, and
  will be restyled in the frontend project.

### Verification

- Unit tests:
  - protocol loading with a start date
  - identical test rows across timelines
  - no row on or after 2025-01-01 loaded
  - late-listing instruments have no rows before first session plus warm-up
  - the fixed split date leaves the test window unchanged
- API tests:
  - `/dashboard` with no timeline and with `timeline=2015` are identical
  - `timeline=2005` reads the 2005 files
  - an unknown timeline returns 422
- Full Python suite and frontend build pass.
- Browser checks at desktop width and 390px:
  - `/timeline`
  - the dashboard switch: it changes the numbers, survives a reload, and works
    with the keyboard

## Left to the builder

Exact episode date ranges, chart layouts, and report field names (following
`development_models.json`).
