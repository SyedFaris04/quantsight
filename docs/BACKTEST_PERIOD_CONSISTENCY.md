# Backtest performance across years and months

## What was added

- Backtesting now shows the selected strategy's return by year and month.
- Each row compares net return with SPY and the equal-weight universe on the same dates.
- The summary shows months beating SPY, tied months, and the best/worst observed month.
- The strategy selector updates this section for all 15 portfolios.
- A downloadable **Period report JSON** contains every strategy's period results and source hashes.
- No new Supabase migration, API key, package or user setup is needed.

## How the calculation works

- Use the published run's daily return CSVs, which already include execution costs,
  overnight holdings and the final liquidation.
- Period net return = `product(1 + daily_return) - 1`.
- Compare strategies and benchmarks on identical observed trading sessions.
- Preserve the existing continuous portfolio. Do not reset capital, liquidate or
  rebalance simply because the month/year changes.
- Show actual first/last observed dates and session counts. February 2023 and
  December 2024 are partial study months; both year rows are partial study windows.
- These period returns are not annualized. The original whole-period CAGR and
  Sharpe calculations remain unchanged.
- Return minus SPY is a **percentage-point difference**, not a relative return ratio.
- Monthly win counts are descriptive consistency measures, not classification
  accuracy, a significance test, or a promise of future profits.
- Returns compound across periods; adding monthly returns gives the wrong total.

Geometric linking and explicit partial-period dates follow the presentation
principles described in the [GIPS handbook](https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/)
and its [partial-period guidance](https://www.gipsstandards.org/qadatabase/5204/).
This project does not claim GIPS compliance.

## Current ensemble evidence

Saved backtest: 462 sessions, 22 February 2023 to 20 December 2024, after the
existing 15 bps per-side cost assumption.

| Observed window | Ensemble | SPY | Difference |
|---|---:|---:|---:|
| 22 February–29 December 2023 | 29.40% | 20.60% | +8.80 pp |
| 2 January–20 December 2024 | 25.18% | 25.78% | −0.60 pp |

- Ensemble beat SPY in **9 of 23 observed months**, with zero ties.
- **14 of 23** observed months had positive ensemble returns.
- Best observed month: May 2023, **+12.23%**.
- Worst observed month: April 2024, **−8.93%**.
- A positive overall advantage can coexist with many underperforming months.
  This is useful supervisor evidence about consistency and concentration.
- The models were developed using this history. These remain exploratory results;
  this breakdown is not independent fresh-data validation.

## Reproduce and verify

From the repository's `backend` directory:

```powershell
# Derive the small period report from the currently published saved backtest.
.venv/Scripts/python.exe -m backtesting.periods

# Recalculate independently from the daily CSVs and compare every stored field.
.venv/Scripts/python.exe -m backtesting.periods --check

.venv/Scripts/python.exe -m unittest discover -s tests -p test_backtest*.py -v
```

- Output: `backend/data/backtests/period_returns.json` (approximately 129 KiB).
- Run the builder again after publishing any future backtest run. CI requires it
  to reconcile before a release passes.
- The builder checks every daily equity against the saved curve, each daily
  return against consecutive equities, and compounded returns against the saved
  total. It rejects duplicate/missing/misaligned sessions and nonfinite values.
- The original report, trades, daily exports and headline metrics are unchanged.
- API `GET /backtest` includes `report.period_analysis`. Web requests load and
  validate the prepared file; they never fit models or run portfolio simulations.
- Source hashes bind the sidecar to the original report, all 15 daily files and
  the builder code. An internal content checksum detects accidental corruption.
- Identical LF/CRLF text is accepted across Windows/Git/Linux deployments.
- Missing, stale or invalid period analysis is hidden with a readable message;
  the existing whole-period report stays usable. Its own source freshness warning
  continues to apply. A mismatching sidecar cannot be downloaded.
- Export: `GET /backtest/export?kind=periods` returns the full period JSON.
- Nine new backend tests cover hand-calculated compounding, year boundaries,
  benchmark ties, invalid accounting/dates, cross-platform line endings,
  stale/corrupt artifacts, exports and path containment.
- The optional browser verifier checks all 15 selectors, annual and monthly
  values, downloads, unavailable states and keyboard table scrolling at desktop,
  390 px and 320 px widths. Measurements/screenshots stay under ignored
  `backend/data/research/backtest_periods_ui`.
- Verification on 4 October 2026: all **136 backend tests**, **7 frontend tests**
  and the production build passed. Browser checks passed for all 15 strategies,
  23 monthly rows, five downloads and both mobile widths, with no uncaught errors.

## Next research priority

- Confirm an unused final window before evaluating it; the candidate 2025
  holdout has not been opened by this change.
- Keep forward prediction tracking separate from historical simulation.
- Use future chronological experiments to test whether any model improvement
  beats a simple baseline with uncertainty. No model was promoted in this release.
