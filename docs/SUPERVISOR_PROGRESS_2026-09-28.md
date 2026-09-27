# Supervisor progress brief - Monday, 28 September 2026

For the simple bullet-point presentation version and tested local backup commands,
use [Supervisor progress and demonstration](../SUPERVISOR_PROGRESS.md).
For terminology, models and architecture, use [the project cheat sheet](../PROJECT_CHEAT_SHEET.md).

## Opening statement

"QuantSight now has a deployed, auditable backtesting system and a
consistent five-trading-session prediction target. I found that the original text
data did not overlap the recent evaluation period, so I acquired and audited another
news dataset. Controlled VADER and FinBERT comparisons did not establish a reason
to replace the live model. My next step is an unused evaluation window and stronger
validation, alongside forward tracking and interface refinement."

## Implemented and verified

- Backtesting with prior-close decisions, next-open execution, transaction costs,
  slippage, share/cash accounting and final liquidation.
- Comparisons with SPY and an equal-weight universe; Sharpe, Sortino, maximum
  drawdown, return, cost sensitivity and uncertainty measures.
- Backtesting interface and downloadable metrics, daily values and trade records.
- Shared five-session labels and live feature calculations; separate live forecast
  metadata and historical results. Supabase migration applied; hosted read checks pass.
- Timestamp-aware research data ingestion, source hashes and a fixed six-fit news study.
- 72 Python tests and 11 database migration tests passed. Frontend build/API checks
  passed; AI Compare was checked against the live deployment in Chromium on desktop/mobile
  with error recovery. Backtesting browser checks cover all 15 strategies, four downloads,
  keyboard scrolling, small screens and unavailable/stale-report handling.
- The redesigned Dashboard distinguishes the saved 20 December 2024 predictions
  from the news archive ending 23 April 2026. Data age is displayed explicitly;
  unavailable requests no longer appear as zero signals or neutral sentiment.
- New price-data preparation: 135,545 rows across 44 instruments and six sector
  ETFs; all 250 sessions in 2025 are covered. Development and candidate-holdout files
  are separate and hashed, with three purged development folds ready. The prior-use
  status of 2025 remains unconfirmed; no new model scores or holdout results are claimed.

The first release is on GitHub and deployed on Render/Vercel. Hosted backtest and
five-session database reads pass; actual scheduled writes and matured outcomes
still require verification. The FinBERT comparison adds two exploratory classifier fits.

## Historical backtest to demonstrate

Period: 22 February 2023 to 20 December 2024; 462 trading sessions.
Initial capital USD 100,000; weekly rebalance; five slots; no leverage.
Commission 10 basis points plus slippage 5 basis points per side; cash/risk-free rate 0%.

| Strategy | Net total return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|
| SPY buy-and-hold | 51.69% | 1.888 | -9.97% |
| Equal-weight universe | 42.89% | 1.524 | -11.48% |
| Four-model ensemble | 61.98% | 1.444 | -17.80% |
| GRU finance | 113.51% | 2.453 | -8.96% |

Interpretation: the ensemble has higher total return than SPY here, but lower Sharpe
and a deeper drawdown. At 30 basis points per side its total return drops to about
41.03%, below its equally costed SPY comparator. The GRU result is descriptive;
selecting it after viewing this table requires another untouched evaluation.

Independent replay of exported trades reproduced all 15 portfolios' equity within
USD 0.000001. This verifies accounting, **not** the absence of research bias.
The historical period was reused during development; the fixed universe and price
adjustment provenance need further validation. Present these as exploratory simulated
results, not live returns or proof of future profitability.

## New research finding

The original finance panel ends in December 2024, while the collected news is from
2026 and the WSB data ends in August 2021. All text features in the shared 2023-2024
historical evaluation sample are zero. Model-name differences therefore do not prove
that contemporary sentiment improved forecasts in that sample.

The new archive has 70,974 source rows and ticker links for 26 of the 44 instruments.
Future-price fields were excluded, and articles were aligned conservatively by
availability time. Coverage remains incomplete.

The fixed pilot trains on 2017-2021, calibrates on 2022 and evaluates on 2023, with
five-session boundary purging. It compares finance, coverage and sentiment inputs.

| 2023 comparison | Accuracy | Brier loss |
|---|---:|---:|
| Constant training-prior probability | 55.23% | 0.247623 |
| Logistic regression, finance only | 47.43% | 0.255580 |
| Logistic regression, finance + coverage + VADER | 47.49% | 0.255890 |
| Logistic regression, finance + coverage + FinBERT | 47.33% | 0.256194 |

Lower Brier loss is better. Sentiment's small improvement over the coverage-only
control has an uncertainty interval containing zero. The pilot did not beat the
simple baselines, and no live model was replaced. The FinBERT follow-up also did
not improve accuracy; boosting had a small Brier gain that still failed to beat the
constant baseline. See [the complete comparison](research/FINBERT_STUDY.md). This is not a fresh final test,
because 2023 and the existing finance feature list were already examined.

## Five-minute demonstration

1. **0:00-0:45:** State the five-session prediction objective and distinguish
   historical simulation from future live tracking.
2. **0:45-2:15:** Open Backtesting. Show ensemble versus SPY, drawdown and costs;
   export the trade ledger to demonstrate auditability.
3. **2:15-3:00:** Explain the original news/price date mismatch and the new coverage audit.
4. **3:00-4:00:** Show the small pilot table. Explain why a negative result is retained.
5. **4:00-5:00:** Present the fresh-window validation plan and remaining forward-tracking/UI work.

Before the meeting, open the [live Backtesting page](https://quantsight.vercel.app/backtesting)
and [AI Compare](https://quantsight.vercel.app/compare) early to allow Render to wake up.
If a cold start causes a request timeout, retry once the backend responds. Local
startup instructions are in [HOW_TO_RUN](../HOW_TO_RUN.md). Keep this brief and the saved report as
offline evidence if hosted services are unavailable. Do not rely on a live accuracy
claim: newly recorded five-session outcomes cannot mature before Monday.

## Next milestones and supervisor discussion

1. Observe genuine scheduled forecast writes and their fifth-session outcomes;
   the migration and hosted read checks are already complete.
2. Freeze an unused final evaluation period and use purged walk-forward validation
   within development data; record every model/feature trial.
3. Use the completed FinBERT/VADER study to define the next training-only experiment;
   investigate calibration/regime stability before selecting another architecture.
4. Backtest eligible new predictions after costs, then improve the interface and
   complete user testing and production integration.

Suggested supervisor questions: Is the primary FYP contribution prediction quality,
decision support, or their comparison? Is the proposed historical evaluation plus
prospective tracking sufficient for the final deadline? Should sentiment be retained
as an evaluated hypothesis even if it does not improve returns?

Measure prediction quality with baseline-relative accuracy, balanced accuracy, MCC,
ROC-AUC and probability scores; economic value with net return, Sharpe, Sortino,
drawdown and costs; system quality with reliability and user task completion.

Supporting material: [research recommendations and primary sources](research/RESEARCH_RECOMMENDATIONS.md),
[first backtest](BACKTEST_FIRST_RUN.md), [data/live evaluation](DATA_AND_LIVE_EVALUATION.md),
[remaining roadmap](IMPROVEMENT_ROADMAP.md).
