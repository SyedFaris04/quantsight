# Controlled finance-model development study

Experiment run `99c94ee337b258ff`, fitted on 28 September 2026; interface added on 29 September.
This is model-selection evidence, not an unused final test or a trading-performance claim.

## Main finding

Follow-up: the [confidence/coverage audit](CONFIDENCE_COVERAGE_AUDIT.md) checks seven
fixed cutoffs using the same exported predictions, without new fits or holdout access.
It shows why stronger confidence and a smaller retained sample do not establish a gain.

- No tested predictor beat the training-prior baseline on the predeclared primary measure: **equal-year mean Brier loss**.
- Raw XGBoost had slightly higher average accuracy (52.54% versus 52.38%), but worse probability loss and balanced accuracy close to 50%.
- The rule therefore selected the **training-prior baseline**, not a new machine-learning model. Selection here is a research result, not deployment.
- No serving models, old predictions, historical backtests or database tables were changed.
- The separate 2025 candidate holdout was not opened or scored. Prior external use remains unconfirmed.

| Registered output | Mean accuracy | Mean balanced accuracy | Mean Brier loss ↓ | Predicts UP |
|---|---:|---:|---:|---:|
| Training-prior baseline | 52.38% | 50.00% | 0.250877 | 100.00% |
| Recent-prior baseline | 48.90% | 50.00% | 0.253134 | 66.67% |
| Always UP, probability 100% | 52.38% | 50.00% | 0.476156 | 100.00% |
| Logistic, raw | 52.53% | 50.19% | 0.251299 | 98.83% |
| Logistic, sigmoid calibrated | 49.06% | 50.11% | 0.253661 | 67.05% |
| Histogram boosting, raw | 52.50% | 50.28% | 0.252198 | 94.83% |
| Histogram boosting, sigmoid calibrated | 48.84% | 49.91% | 0.253701 | 66.98% |
| XGBoost, raw | 52.54% | 50.27% | 0.251999 | 95.79% |
| XGBoost, sigmoid calibrated | 48.90% | 49.89% | 0.254025 | 67.81% |

Means weight each validation year equally. The constant training-prior probability is
estimated from each fold's fit labels; it is not fitted to the validation labels.
The recent-prior control uses the calibration-year UP frequency. Always UP is an
explicit directional reference with P(UP)=1; it is not a sensible certainty claim.

### Year stability

| Year | Baseline accuracy | Raw XGBoost accuracy | Baseline Brier | Raw XGBoost Brier |
|---|---:|---:|---:|---:|
| 2022 | 47.32% | 48.15% | 0.257324 | 0.258879 |
| 2023 | 55.23% | 54.95% | 0.247391 | 0.249379 |
| 2024 | 54.60% | 54.50% | 0.247915 | 0.247738 |

XGBoost improves Brier loss only in 2024 here. Choosing that year alone would hide
the weaker average result. Calibration also worsened average loss for each tested
model; calibration is not automatically beneficial when relationships change across time.

Raw logistic's average Brier improvement versus the baseline is **−0.000422**, with
an exploratory 95% interval of **[−0.001329, 0.000489]**. It does not establish a gain.
Intervals use 1,000 paired circular 20-session date-block resamples separately within
each year, retaining all instruments on each sampled date and averaging year gains.
They are conditional on the fitted models and not adjusted for candidate selection
or multiple comparisons; they are not final-test confidence guarantees.

## Fixed design

- Protocol and implementation were committed locally as `2ce3ac6` before fitting.
  The run wrote its timestamped registration and hashes before loading features.
  This was not an externally registered or publicly timestamped preregistration.
- Source: frozen snapshot `20260927T141409402344Z`, 44 fixed instruments.
- Inputs: 13 fixed finance features: 1/5/10/20-session returns, 20-session return
  volatility, drawdown, 20/60-session moving-average distances, intraday range,
  relative volume, rolling RSI, SPY five-session return and relative return to SPY.
- RSI is the explicitly coded simple 14-session rolling gain/loss ratio, scaled
  to 0–1. No feature selection, ticker identifiers or news inputs were used.
- Returns/trends/labels use the frozen adjusted close. Intraday range uses
  same-session vendor high, low and close; volume uses a trailing ratio.
- Features use the completed signal session and earlier sessions only. Reindexing
  uses the NYSE calendar; missing sessions are not filled. Warm-up/missing-feature
  rows are excluded identically for all candidates.
- Target: adjusted close strictly higher after five NYSE sessions; ties are DOWN.
  Labels require every session in the window. Unknown tail outcomes remain missing.
- Whole-date boundaries and label-end purging separate fit, calibration and validation.

| Validation year | Fit rows | Calibration rows | Validation rows |
|---|---:|---:|---:|
| 2022 | 60,913 | 10,868 (2021) | 10,824 |
| 2023 | 72,001 | 10,824 (2022) | 10,780 |
| 2024 | 83,045 | 10,780 (2023) | 10,868 |

- Total: **32,472 validation rows**, nine predictor fits and nine sigmoid calibrator fits.
- Logistic uses a training-fitted StandardScaler. Tree models use the unscaled ratios.
- Parameters, seeds, features, calibration and selection rule were fixed before fitting;
  no additional trials were run in response to these results.
- Calibration fits logistic regression on clipped raw logits using only the separate
  calibration year. Raw and calibrated probabilities are separate candidate outputs;
  each uses its own 50% directional threshold. This does not alter legacy serving labels.
- Selection: lowest equal-year mean Brier among all registered outputs and baselines.
  Exact ties prefer baselines, then simpler families, with raw before calibrated.
- All fold predictions, fitted bundles, source hashes and software versions are saved
  locally. The published JSON contains aggregates, reliability bins and artifact hashes.

## Interface and reproduction

- **AI Compare → Do the models beat a simple baseline?**
- Switch among the equal-year mean and individual years. The expandable section
  shows reliability bins and uncertainty for any candidate.
- Download the same validated aggregate report as JSON.
- API: `/research/development-models`; export: `/research/development-models/export`.
- The serving API only reads the small report; it never runs training or opens raw prices.
- Missing or inconsistent reports are withheld. The existing historical model-metrics
  endpoint can fail independently without hiding the new study.
- No additional secrets, environment settings or Supabase migration are needed.

From `backend`, with the frozen snapshot present:

```powershell
.\.venv\Scripts\python.exe -m research.development_models
```

The command refuses to overwrite an existing run or published report. Preserve the
completed experiment; a separately documented protocol/run is needed for new work.
Unrecorded retries or parameter edits after seeing results would be additional trials.
Exact-byte source hashes can differ after Git newline conversion across operating systems;
software versions and the copied registration identify the actual local run.

## Checks and interpretation limits

- Thirteen new tests cover feature causality, unit scaling, missing-session handling,
  RSI/volume edge cases, probability transforms, reliability bins, bootstrap identity,
  training-only scaling, baseline selection, report reconciliation and API/export failures.
- Full backend suite: **103 tests pass**. Frontend production build passes with the
  existing large-bundle warning.
- Independent recomputation from exported predictions reproduces all 27 candidate/year
  metric sets within **3 × 10⁻¹⁰**; all 12 model/prediction artifact hashes match.
- Local Chromium checks pass for all four period options, nine reliability candidates,
  the real JSON download, error/retry, and independent legacy-metrics failure. Inspected
  desktop and 390/320-pixel views have no page-level horizontal overflow or uncaught
  page errors. A small overflow in the older comparison rows was corrected too.
- This study uses previously examined development years, a fixed survivor universe
  and retrieval-time adjusted vendor prices. It does not remove research-selection bias.
- These scores are not directly comparable to older serving-model headline metrics:
  the snapshot, features, fitting windows and evaluated periods differ.
- There is no new trading return, Sharpe or cost claim. Existing backtest results remain
  separate. A cost-aware strategy must be fixed before final evaluation/promotion.
- Next: retain the baseline as a benchmark; investigate whether the available inputs
  contain useful predictive information in a separately registered experiment. Do not
  search repeatedly until an attractive historical score appears. Confirm an unused
  final window and freeze the full prediction/trading procedure before scoring it.

References: [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
explains disjoint calibration data and why Brier loss also reflects discrimination;
[chronological validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
explains why ordinary random folds are unsuitable for dependent time-series observations.
