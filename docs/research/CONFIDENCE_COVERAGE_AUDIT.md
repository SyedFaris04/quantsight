# Does stronger model confidence improve prediction quality?

## What was added

- AI Compare → finance development study → **Does stronger confidence improve accuracy?**
- Nine candidates: six model outputs and three simple baselines.
- Seven fixed confidence cutoffs: 50%, 55%, 60%, 65%, 70%, 80% and 90%.
- Individual 2022/2023/2024 views and a pooled view of all **32,472 ticker-days**.
- Coverage, retained rows/dates, average confidence, direction accuracy, Brier loss,
  and the tendency to predict UP.
- The training-prior baseline is scored on the exact same retained rows.
- Small samples are flagged. Empty selections display undefined scores.
- Downloadable audit JSON and the original three development prediction exports.
- The report loads only when the section is first expanded. Changing controls or
  reopening it uses the already-loaded result.
- No additional user setup, database migration or model training is required.

## Main result

For **raw XGBoost from the development study**, pooling all three validation years:

| Minimum confidence | Retained rows | Coverage | Accuracy | Matching baseline accuracy | Model Brier | Baseline Brier |
|---|---:|---:|---:|---:|---:|---:|
| 50% | 32,472 | 100.00% | 52.53% | 52.38% | 0.251997 | 0.250878 |
| 55% | 20,449 | 62.97% | 52.20% | 52.11% | 0.253765 | 0.251261 |
| 60% | 4,193 | 12.91% | 51.51% | 51.49% | 0.260603 | 0.252275 |
| 65% | 375 | 1.15% | 56.80% | 56.80% | 0.254950 | 0.245682 |
| 70% | 32 | 0.10% | 62.50% | 62.50% | 0.248715 | 0.238212 |
| 80% / 90% | 0 | 0.00% | Undefined | Undefined | Undefined | Undefined |

- The 60% cutoff removes most predictions while accuracy falls.
- The attractive 62.5% score at 70% confidence comes from only **32 rows across
  24 signal dates**. Every retained prediction is UP, so the baseline achieves
  the same direction accuracy. Model Brier loss is worse than the baseline.
- The model's average confidence at that cutoff is 73.64%; observed accuracy is
  62.5%. Model confidence is not an observed success rate.
- The intentionally overconfident always-UP reference assigns P(UP)=100% yet
  achieves only 52.38% accuracy. Certainty in an output does not establish correctness.
- No threshold was selected or applied to the live system. These findings do not
  demonstrate an improvement worth promoting.
- These are **separately fitted development-study models**. The table is not a
  measurement of current live model accuracy or historical portfolio returns.

## Technique in simple terms

- Each prediction supplies P(UP). P(DOWN) is `1 - P(UP)`.
- Predicted direction is UP when P(UP) ≥ 50%; otherwise DOWN.
- Confidence score is `max(P(UP), P(DOWN))`.
- Keep rows whose confidence is at least the stated cutoff.
- Coverage = retained rows / all eligible rows.
- Accuracy = correct retained directions / retained rows.
- Conditional error rate = `1 - accuracy`; it is undefined for zero retained rows.
- Brier loss = average `(P(UP) - actual_direction)^2`, where actual direction is
  0 or 1. Lower is better. It measures overall probability quality, not calibration alone.
- Recompute baseline scores on the **same mask**, rather than comparing a filtered
  model to the baseline's full dataset.
- Pool all years by retained observation count. This differs from the equal-year
  averages used for the original model-selection comparison.
- Flag fewer than 200 retained ticker-days **or** fewer than 20 signal dates.
  This is a disclosure heuristic, not proof that larger samples are statistically valid.

The risk/coverage trade-off is motivated by [Geifman and El-Yaniv's selective
classification paper](https://papers.nips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf).
Their experiments and risk-control procedure concern other tasks; this diagnostic
does not reproduce their method or transfer its guarantees to financial predictions.
[Scikit-learn's calibration guidance](https://scikit-learn.org/stable/modules/calibration.html)
explains why confidence should be checked against observed outcomes and why Brier
loss also reflects discrimination. Our audit does not fit another calibrator.

## Reproducibility and integrity

- Parent study: `99c94ee337b258ff`; see [finance development study](FINANCE_DEVELOPMENT_STUDY.md).
- [Fixed audit protocol](../../backend/research/confidence_audit_protocol.json) and
  calculation/validation code were committed locally as `cda8bca` before the first
  audit calculation on 4 October 2026. This is local registration, not external preregistration.
- This audit was planned after seeing the parent study, so it is explicitly exploratory.
- **Zero new fits**, no price download, no holdout access and no live model changes.
- Only the original three validation prediction exports are read; their content
  must match the parent study's recorded SHA-256 hashes.
- Published copies live in
  `backend/data/research_predictions/development_99c94ee337b258ff/` (~6 MiB total).
  The local parent experiment's files remain unchanged.
- Aggregate: `backend/data/research_reports/development_confidence.json` (~133 KiB).
- Source/implementation hashes and an internal content checksum bind the report.
  Identical LF/CRLF text is accepted across Windows/Git/Linux.
- Validation reconciles all 50% rows with the original study's accuracy/Brier scores,
  checks decreasing retained counts, null empty scores, and weighted pooled values.
- API `/research/development-confidence` serves only checked evidence. Export:
  `/research/development-confidence/export`. Invalid/missing evidence has its own
  recovery state and does not hide the main development report.
- The serving loader uses only standard-library code. It never imports the offline
  builder or model-training libraries.

From `backend`:

```powershell
.venv/Scripts/python.exe -m research.confidence_audit --check
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

CI recalculates all nine candidates, seven thresholds and four period views from
the published prediction exports and compares every stored field. The builder
refuses to overwrite the published audit. New studies need a new documented protocol.

Validation on 4 October 2026:

- All **144 backend tests** passed, including eight new confidence-audit tests.
- The frontend production build and **seven frontend tests** passed.
- The serving-runtime smoke check passed, including both prepared TreeSHAP endpoints
  and the new evidence endpoint; no training libraries loaded on the serving path.
- Browser checks passed for **36 candidate/period combinations and 252 threshold
  rows**, exact JSON download, one initial expansion fetch, recovery states,
  keyboard scrolling and 390/320 px layouts, with no uncaught errors.
- QA screenshots/downloads stay in ignored `backend/data/research/confidence_ui/`.

## Interpretation and next step

- Every cutoff is shown; there is no automatic "best threshold" recommendation.
- Conditional samples differ between models and cutoffs. Their scores are not
  directly comparable across different retained populations.
- Overlapping five-session labels and cross-stock dependence mean ticker-days are
  not independent trials. No ordinary binomial confidence interval is implied.
- This descriptive audit has no significance claim, selection correction, fresh-data
  conclusion, trading-cost test or Sharpe claim.
- Keep the 2025 candidate holdout unopened until its prior-use status is confirmed
  and the final prediction/trading procedure is fixed. A code split alone cannot
  establish that the period was unused outside this folder.
- Use this result to motivate controlled feature/data experiments, rather than
  increasing the confidence cutoff to manufacture a better headline accuracy.
