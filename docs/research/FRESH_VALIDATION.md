# Fresh-data evaluation preparation - 27 September 2026

Snapshot `20260927T141409402344Z` is acquired and structurally validated. This step
prepares a stronger evaluation process; it does not report improved accuracy or
replace a serving model. No models were fitted and no 2025 outcomes were scored.

## What is frozen

- Yahoo Finance daily prices from 2 January 2015 through 31 December 2025 for
  the existing 44 instruments and six sector ETFs: **135,545 instrument/session rows**.
- Every requested instrument has all **250 NYSE sessions in 2025**. No internal
  calendar gaps, duplicate dates, invalid OHLC ranges or nonfinite required prices
  were found. Initial unavailable history for MRNA, PYPL, SNAP and UBER is recorded;
  this check alone does not establish their historical listing dates.
- 150 CSV artifacts: the vendor response for each ticker, development rows before
  2025, and a separate candidate 2025 holdout. SHA-256 hashes, software versions,
  acquisition settings, source code hash and the request protocol are recorded.
- OHLC, adjusted close, volume and available corporate-action columns are retained.
  Automatic adjustment, back-adjustment and repair were explicitly disabled. No
  filling or interpolation is used. The old application data were not overwritten.

The [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
defines the inclusive start, exclusive end and adjustment settings used here.
The [project's source-use notice](https://github.com/ranaroussi/yfinance) describes
its research/personal-use context. Raw downloaded data remain local; this release
publishes code and aggregate audit metadata, not the price CSVs.

## Why 2025 is still a candidate

An audit of 17 existing feature/prediction CSVs, including legacy exports, found
their latest date is 20 December 2024. This is evidence about the inspected files,
not proof that nobody used 2025 elsewhere. Live inference may have downloaded 2025
bars as trailing feature history without evaluating predictions on that period.

The user has not yet confirmed whether 2025 was previously used for training,
tuning or evaluation outside this workspace. Its status remains **unconfirmed**.
If previously used, it cannot be advertised as an untouched final test. Preserve
it as exploratory and choose an appropriately unused or prospective period instead.
The acquisition protocol was saved before download; it was not publicly registered
with an external preregistration service, and the final model experiment is not yet registered.

Only coverage and validity diagnostics were inspected for 2025. No label balance,
returns, performance charts, predictions or model scores were calculated. The split
is a procedural research safeguard, not encryption or a security access boundary.

## Development validation

The new development loader verifies each file hash and rejects any row dated in
2025 or later. The label builder uses the actual NYSE calendar; a missing session
cannot silently change a five-session target into a longer horizon. Unknown future
tail outcomes stay missing. Labels in this split demonstration use frozen adjusted
close; target and execution price conventions must be fixed in the model protocol.

The same whole-date segments apply across all instruments. Each row's outcome date
must fall strictly before the next segment, including the end of validation.

| Validation year | Fit rows | Calibration rows | Validation rows |
|---|---:|---:|---:|
| 2022 | 63,509 | 10,868 (2021) | 10,824 |
| 2023 | 74,597 | 10,824 (2022) | 10,780 |
| 2024 | 85,641 | 10,780 (2023) | 10,868 |

Counts are before feature warm-up/filtering. Development contains 107,949 rows
for the 44 instruments, including 220 unknown tail labels (five per instrument).
`prepare_development` reads only development CSVs and does not open holdout CSVs.
These already-studied years are for model selection, not an unbiased final claim.

[Scikit-learn's time-series validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
supports chronological validation for dependent observations. QuantSight adds
explicit five-session outcome purging and separate calibration periods.

## First development experiment completed

The [fixed finance study](FINANCE_DEVELOPMENT_STUDY.md) was registered locally and
run on 28 September, using the prepared development folds and 13 fixed inputs.
Nine predictor fits and nine calibrator fits compared three model families with
three baselines. None of the six model outputs beat the training-prior baseline
on equal-year mean Brier loss. No serving model was replaced and no 2025 holdout
file was opened. Prior external use of 2025 remains unconfirmed.

## Evaluation requirements

1. Register a compact feature set, candidate models, calibration method, selection
   criterion, trial budget and all price/target conventions before fitting.
2. Fit transformations and feature selection only within the appropriate training
   segment. Compare all candidates with a training-prior probability baseline and
   always-UP direction on identical development observations.
3. Select using development Brier loss, with balanced accuracy, MCC, ROC-AUC,
   calibration and regime stability as diagnostics. Do not choose on 2025 results.
4. Freeze chosen artifacts and a cost-aware trading rule before a single final
   holdout evaluation, once prior-use status is established. Preserve negative results.
5. Require fresh evidence and trading-cost/risk checks before considering a live
   model replacement. No automatic promotion is implemented here.

The fixed universe still has survivorship/selection bias. Vendor history can be
revised, and retrieval-time adjusted prices are not point-in-time vintage data.
Structural coverage does not establish financial correctness. No new aligned
2025 news dataset was acquired, so this preparation does not extend the sentiment claim.

## Reproduction and validation

From `backend`, install `requirements-price-research.txt`, then:

```powershell
.\.venv\Scripts\python.exe -m research.freeze_prices
.\.venv\Scripts\python.exe -m research.prepare_development data/research/fresh_prices/20260927T141409402344Z
```

The first command creates a new timestamped acquisition, never overwrites an old
snapshot, and can return a different vendor revision. Use its printed snapshot ID
for the second command. Completed split reports refuse overwrite; preserve prior
reports when registering another experiment. File hashes use exact bytes, so moving
raw artifacts through newline-converting tools changes their identity.

Validation: **72 Python tests passed**, including nine new tests for holidays,
missing bars, ties, per-instrument independence, segment overlap, holdout rejection,
price validity, date gaps and artifact tampering. The actual snapshot passes the
quality audit and all three development folds were constructed successfully.

Artifacts: [protocol](../../backend/research/fresh_data_protocol.json),
[acquisition request](fresh_price_request_20260927T141409402344Z.json),
[coverage and hashes](fresh_price_audit_20260927T141409402344Z.json),
[development folds](fresh_development_splits.json),
[existing-file audit](existing_evaluation_dates.json).
