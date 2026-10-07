# Forward evaluation improvement — 7 October 2026

## What changed

- The Forward results page compares the model with always predicting UP on the same resolved stocks and dates.
- It shows **resolved forecast dates**, alongside resolved stock forecasts. Many stocks from one date are not independent evidence.
- Added a 30/90/180/365-day forecast-window selector, refresh/retry controls and an evaluation JSON export.
- Resolved and pending records now have separate views. Resolved records appear first, even when the newest forecasts are pending.
- Record rows show UP/DOWN and calibrated P(UP). DOWN outcomes include ties. P(UP) is not the model's observed accuracy.
- Secondary information sits inside expandable sections: forecast-date results, probability groups, uncertainty/method and model versions.
- Daily cohorts show how many forecasts were logged, resolved and pending, with paired baseline results. Partial cohorts are marked explicitly.
- Different model/feature/calibration versions receive separate descriptive summaries. Pooled totals display a mixed-version warning.
- Pending forecasts whose target close plus the 20-minute data delay has passed are flagged. They remain excluded from scores.
- Collection gaps are visible: expected NYSE sessions since the first logged date in the chosen window, after 23:00 UTC. This allows one hour beyond the existing 22:00 UTC job schedule.
- The gap indicator identifies missing data, not its cause. It does not count absent forecasts as incorrect predictions or backdate new records.
- The summary query reads only the fields required for reporting, avoiding the large saved feature JSON for every database row. The original snapshots remain in Supabase.
- The public forward-results route is isolated in `forward_routes.py`, so its API contract tests do not require the full app's serving-only dependencies or dataset startup.

## What the current data says

Read-only observation on 7 October 2026; these values change as collection continues.

| Measure | Observed value |
|---|---:|
| Logged forecasts | 88 |
| Resolved stock forecasts | 44 |
| Pending stock forecasts | 44 |
| Distinct resolved forecast dates | **1** |
| Recorded model direction accuracy | 52.3% |
| Always-UP accuracy on the same rows | 59.1% |
| Accuracy difference | −6.8 percentage points |
| Balanced direction accuracy | 47.6% |
| Calibrated-probability Brier loss | 0.2444 |
| Fixed P(UP)=0.5 Brier loss | 0.2500 |

- Resolved cohort: **28 September 2026**; pending cohort: **5 October 2026**.
- The model is below the always-UP classification baseline in this small sample.
- Brier loss is slightly below the fixed 50% probability baseline. This does not establish calibration quality, generalization or investment returns.
- Two of seven expected collection sessions have records, through 6 October.
- Missing sessions: **29 September, 30 September, 1 October, 2 October and 6 October**.
- The [7 October scheduled workflow](https://github.com/SyedFaris04/quantsight/actions/runs/37556701339) failed in its backend-call step; the public annotation reports curl exit code 22. That identifies an HTTP error, not the underlying cause. Several earlier scheduled runs also failed; the 6 October run succeeded.
- The collection failure is **not fixed by this reporting change**. Diagnosing the backend responses and stabilizing collection is the next priority. Historical forecasts must never be created retrospectively to fill the gaps.

## How uncertainty is handled

- Reporting rules were committed in `5838e97` before the new reporting measurements. See [protocol](../backend/forward_protocol.json).
- Do not show an interval for 44 stocks from one forecast date.
- Require at least **60 distinct resolved forecast dates**, one known model/feature/calibration signature and no partly resolved date cohorts.
- Wholly pending dates do not enter the metrics or intervals.
- When eligible, use a circular moving-block bootstrap: **20 NYSE sessions**, **1,000 repetitions**, fixed PCG64 seed **42**.
- Keep all stocks from a date together. Keep missing sessions on the calendar grid with zero weight; do not compress gaps into adjacent observed dates.
- Recompute row-weighted direction accuracy, paired model-minus-always-UP accuracy and Brier loss from each draw. Show exploratory 95% percentile intervals.
- Require at least 950 nonempty bootstrap draws. Otherwise explain why uncertainty is unavailable.
- The 60-date threshold is a display safeguard, not a sample-size power calculation or proof that 60 dates are sufficient for a strong conclusion.
- Market regimes can change. The intervals do not correct every prior experiment or repeated monitoring decision. They are not a significance test or a model-promotion rule.
- This change improves evaluation and presentation; it does not retrain the models or claim an accuracy gain.

## Probability and metric definitions

- Direction accuracy: proportion of recorded raw-model UP/DOWN decisions matching the resolved outcome.
- Always-UP accuracy: actual UP proportion on those exact same rows. This is a classification baseline, not a probability forecast of 0.5.
- Balanced accuracy: mean of UP recall and DOWN recall. Show unknown when only one outcome class is observed.
- Brier loss: mean squared error between recorded calibrated P(UP) and the actual binary UP outcome; lower is better.
- Fixed probability baseline: P(UP)=0.5 for every row, giving Brier loss 0.25.
- The recorded direction can disagree with whether calibrated P(UP) exceeds 0.5. The existing serving rule is retained, and the page explains which output each metric scores.
- Probability groups use fixed ranges [0%,20%), [20%,40%), [40%,60%), [60%,80%), [80%,100%]. Empty groups show unknown rates, not zero success.

## Supervisor demonstration

1. Open **Forward results**; explain 44 resolved stock forecasts versus one resolved forecast date.
2. Compare 52.3% model accuracy with 59.1% always-UP accuracy. Explain the negative result plainly.
3. Open **Results by forecast date**; show the resolved and pending cohorts.
4. Show **Forecast collection has gaps**, with the missing dates.
5. Open **Uncertainty and evaluation method**; explain why intervals are currently withheld.
6. Switch **Resolved / Pending**. Explain P(UP), target date and the recorded direction.
7. Click **Export evaluation JSON**. The file preserves the selected window, data check time, reporting protocol, per-date results, metrics, model versions and interval method from the displayed snapshot.

The export does not include raw feature snapshots, account information or individual recent-record arrays. It is an aggregate reporting snapshot, not the full forecast ledger or an immutable signed research artifact. Original forecasts remain in Supabase.

## Technical validation

- **168 backend tests passed**, including 11 new forward-evidence regression tests.
- The same 168 tests also passed in a new isolated environment containing only `requirements-evaluation.txt`, matching the CI dependency scope. Serving runtime smoke checks passed after the route extraction.
- **11 frontend tests passed**; the production build passed.
- Tests cover hand-computable scores, calibration/raw-direction separation, all-UP identity, date-block replay, input ordering, missing calendar sessions, mixed/unknown versions, partial cohorts, duplicate keys, invalid probabilities and wrong stored correctness.
- Tests cover date-window query bounds, early-close/holiday collection deadlines and exact overdue cutoffs.
- Browser checks used the real read-only local API and isolated fixtures for error/retry, unavailability, empty samples, older backend responses, eligible intervals, mixed versions and export failure.
- Desktop, 390 px and 320 px layouts passed without page overflow. Tables can scroll with the keyboard; no uncaught page errors were observed. Screenshots were visually inspected.
- The downloaded real-data export matched the API evidence. Window changes were reflected in the export.
- Older backends that return a different window now show an explicit notice; exports identify the returned window.
- No Supabase migration, new cloud secrets, model fitting or reserved historical holdout access was needed. QA did not trigger the administrator collection job or write database records.
- Local QA files are ignored under `backend/data/research/forward_evidence_20261007`.

## Reproduce

From `backend`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m research.verify_forward_ui --browser "C:/path/to/chrome.exe"
```

The browser check needs Playwright/Chromium, a running local frontend/backend and readable live tracking data. Its extra scenarios use browser-only fixtures.

## Primary method references

- [Time-series block bootstrap documentation](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html): preserving dependence through blocks. QuantSight implements its registered calculation with NumPy; it does not add the arch package to serving dependencies.
- [scikit-learn metric definitions](https://scikit-learn.org/stable/modules/model_evaluation.html): balanced accuracy and Brier loss; Brier is not a pure calibration measure.
- [NumPy Generator documentation](https://numpy.org/doc/stable/reference/random/generator.html): seeded random draws; library versions still matter for reproducibility.
- [GitHub scheduled-event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule): scheduled jobs can be delayed. A missing record alone does not identify a scheduler or provider fault.
