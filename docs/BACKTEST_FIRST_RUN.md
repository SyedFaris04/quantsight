# First corrected historical evaluation

Run: `20260926T102301904516Z`.
Trading period: 2023-02-22 through 2024-12-20 (462 sessions).
Prediction comparison: 20,372 shared ticker-days; 44 instruments.
Capital: USD 100,000; commission 10 bps + slippage 5 bps per side;
cash/risk-free assumption 0%; weekly rebalance; five slots; no leverage.

Selected descriptive results (all strategies remain available in the app/report):

| Strategy | Net total return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|
| SPY buy-and-hold | 51.69% | 1.888 | -9.97% |
| Equal-weight universe | 42.89% | 1.524 | -11.48% |
| Four-model probability ensemble | 61.98% | 1.444 | -17.80% |
| GRU finance | 113.51% | 2.453 | -8.96% |

The ensemble earned a higher total return than SPY in this sample, but had a
lower Sharpe and deeper drawdown. Its 95% block-bootstrap Sharpe interval is
approximately -0.05 to 2.87. A high point estimate alone is not enough evidence.
GRU finance is shown as a descriptive model result, not a newly selected production
winner. Selection after observing the table would require another untouched test.

The prespecified ensemble scored at the 97.5th Sharpe percentile among 1,000
random portfolios matched on its allocation schedule. This narrow null comparison
does not correct for the full history of model/feature experiments. At 30 bps
per side the ensemble's total return drops to about 41.03%, below its equally
costed SPY benchmark. Costs materially affect the conclusion.

Limitations remain substantive: this historical period was reused in model
development; no fresh final test was introduced; all text features in the shared
sample are zero; stock-universe selection and price-adjustment provenance need
further validation. These are exploratory simulations, not verified live returns.

Verification completed locally:

- 18 automated accounting/label/API tests passed.
- Frontend production build passed (existing large-bundle warning remains).
- API report and all four export types returned HTTP 200; source hashes matched.
- Existing health, overview and metrics endpoints returned HTTP 200.
- Every exported execution was independently replayed for all 15 portfolios.
  Maximum difference from published daily equity/cash was below USD 0.000001.
- 103,856 stored labels matched independently calculated five-session outcomes;
  220 trailing labels require prices beyond the stored feature snapshot.

No trained models, existing prediction files, cloud data, or live deployments were
changed by this stage. Browser visual inspection remains outstanding.
