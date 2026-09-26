# Historical backtesting protocol

## Purpose and status

This module measures how a specified trading policy would have performed using
the saved model predictions. It is separate from classification accuracy and from
the live forward prediction tracker. Current results are **exploratory**, since the
existing historical period has already influenced model development. An honest
fresh-window evaluation is a later roadmap stage.

The implementation replaces notebook 15's within-week look-ahead, missing weekend
accounting, flat weekly cost charge and incorrect Sharpe/Sortino formulas. Old
CSV summaries in `backend/data/predictions` are retained for provenance but are
not used by the new API or page.

## Run locally (PowerShell, from repository root)

```powershell
backend/.venv/Scripts/python.exe notebooks/15_backtesting.py --random-runs 1000 --bootstrap-samples 1000
```

The CLI works from any directory. A shorter development run can use
`--random-runs 100`; the exact count/seed appears in the report. Options include
`--commission-bps 10 --slippage-bps 5 --risk-free-rate 0.0 --top-n 5
--min-probability 0.5`. Risk-free rate is an annual effective decimal, not percent.
Changing settings produces another exploratory trial; do not choose the most
favourable settings on the final test window.

Open `/backtesting` in the frontend after starting the existing API/frontend.
No database migration, cloud credentials, downloads or retraining are needed.
The local frontend currently targets port 8001; match the backend port to
`VITE_API_URL` when starting it.

## Data and target

- Source prices: `backend/data/processed/features_finance.csv`, Open and Close.
- Twelve saved individual model variants and the four-model probability ensemble.
- The saved `confidence` field is treated as **P(up) in percent**, not confidence
  in whichever direction was selected. This reflects the training export contract.
- Shared ticker/date pairs establish classification comparisons. Strategy evaluation
  starts at the first price session after the first common signal date and ends at
  the common final date. The full intervening SPY session calendar is retained.
- Duplicate rows, nonpositive/nonfinite prices and incomplete price panels fail
  explicitly; no future/nearest-date alignment or price forward-fill is allowed.
- Training label: Close after five observed sessions is higher than today's Close.
  Equal prices count as SELL. Unknown future tails remain missing in new labels.
- Existing labels are recomputed wherever five later price rows exist. Mismatches
  abort publication. The final five rows per ticker retain supplied labels and are
  reported as not independently verifiable from this truncated price snapshot.
- Weekly portfolio holding time is a strategy choice, not a redefinition of the
  classifier's five-session target. Return and classification metrics differ.

## Trading policy

1. Enter at the first evaluation session's open. Thereafter rebalance at the first
   observed trading session of each new week (including weeks with holidays).
2. Read predictions only from the immediately preceding session's close. Do not
   use any prediction from the execution day or a later day.
3. Exclude SPY from model stock selection. Keep BUY predictions whose calibrated
   P(up) is at least the specified threshold. Rank descending by P(up), breaking
   ties alphabetically by ticker. The ensemble here averages probabilities; it is
   not the separate UI's majority-vote consensus engine.
4. Select up to five instruments. Each receives 1/5 of post-cost equity, capped at
   25%. Missing eligible slots remain cash rather than increasing other weights.
5. Carry fractional shares and cash continuously across sessions. Opening gaps and
   weekend returns are included because positions persist until explicitly sold.
6. Charge 10 bps commission plus 5 bps adverse slippage per side, by default, on
   actual changed notional. Slippage is a cash deduction equivalent to an adverse
   execution adjustment; the audit export retains reference price and cost separately.
   Solve target equity including costs, so rebalancing never borrows money.
7. Mark equity at every close. Liquidate remaining positions at the last close,
   charging exit costs. Benchmarks use the same evaluation window and cost rules.
8. Cash held after each opening trade earns a constant session-equivalent risk-free
   return. Default 0% is explicit; no claim is made to using historical Treasury rates.

The 25% target cap applies to model portfolios; SPY buy-and-hold is intentionally
100% SPY. Weights can drift between rebalances. No leverage, shorting, market impact,
taxes, partial fills or explicit dividend cashflows are simulated. Price adjustment
provenance still needs auditing; using adjusted prices and adding dividends again
would double-count them.

## Benchmarks and robustness

- SPY buy-and-hold, entered at the same open and liquidated at the same final close.
- Equal-weight non-SPY universe, rebalanced weekly using the same accounting.
- Random portfolios matched to the prespecified **ensemble** on rebalance dates,
  position counts, target exposure, costs and holding schedule. Portfolio tickers
  are sampled without replacement. The reference is not selected after seeing results.
- The null report includes Sharpe/return percentiles and a one-sided Monte Carlo
  tail probability with a +1 finite-simulation correction. It only assesses the
  ensemble's selection against this null, not all model or research choices.
- Ensemble cost sensitivity at 0, 5, 15, 30 and 50 bps per side, rerunning accounting
  and SPY under each setting. These are research scenarios, not estimated spreads.
- 95% circular block-bootstrap intervals: 1,000 resamples by default, block length
  10 sessions, fixed seed. Serial dependence is retained within blocks. Block-length
  sensitivity and paired inferential tests remain future work.

## Metrics and units

Machine-readable values are decimals, not percentages (0.10 means 10%). Undefined
ratios are JSON null, not invented zero or infinity. With daily net return r,
daily effective risk-free return rf and 252 sessions per year:

- Total return = final equity / initial capital - 1.
- CAGR = (final / initial)^(252 / number of observed sessions) - 1. This is a
  trading-session annualization convention, not a calendar-year calculation.
- Volatility = sample standard deviation(r) * sqrt(252).
- Sharpe = mean(r - rf) / sample standard deviation(r - rf) * sqrt(252).
- Sortino = mean(r - rf) / sqrt(mean(min(r - rf, 0)^2)) * sqrt(252).
  The minimum acceptable return is the configured risk-free rate; all days are in
  the downside RMS denominator, not only negative days.
- Drawdown = equity / prior running peak - 1, including initial capital as a peak.
  Duration counts consecutive underwater sessions; ongoing drawdowns count through
  the evaluation end. Calmar = CAGR / absolute maximum drawdown.
- VaR95 = negative fifth-percentile daily return; CVaR95 = negative mean return at
  or below that percentile. These describe empirical daily tail losses.
- One-way annual turnover = mean(daily traded notional / closing equity) * 252.
  Both buys and sells count; this is not half-turnover.
- Exposure = marked invested value / equity before final liquidation.
- Daily win rate is positive-return days, **not winning trades**. Trade profit
  factor/expectancy require round-trip lot accounting and are not claimed here.
- Beta and annual alpha are descriptive regression quantities, without significance
  claims. Information ratio uses mean(strategy - SPY) / sample standard deviation
  of that difference, annualized by sqrt(252).
- Classification includes accuracy, always-up/down references, balanced accuracy,
  macro-F1, MCC, ROC-AUC of calibrated P(up), average precision, Brier score, log
  loss, fixed-bin calibration error and confusion matrix. AUC can differ from the
  older model JSON that evaluated raw probabilities, especially with isotonic ties.

## Audit and reproducibility

`backend/data/backtests/report.json` points to a uniquely named run folder containing:

- `report.json`: assumptions, source SHA-256 hashes, data diagnostics, metrics and curves.
- `metrics.csv`: all strategy metrics.
- `<strategy>_daily.csv`: equity, return, cash, exposure, costs and traded notional.
- `<strategy>_trades.csv`: decision and execution date, instrument, side, quantity,
  reference price, notional, commission, slippage and reason.
- `<strategy>_rebalances.json`: selected instruments, target exposure and coverage.

Run exports are written before atomically publishing the report pointer. API
`GET /backtest` validates source hashes and flags changed/missing inputs. It does
accept LF/CRLF representations of identical text, so Git's Windows-to-Linux
newline conversion does not falsely mark a Render deployment stale. It does
not silently recompute, retrain or mix run versions. `GET /backtest/export` allows
only listed metrics, daily values, trades and report exports.

For deployment, include the current report and its referenced run folder with the
same source/prediction files. Incomplete prior run folders are not published results.
Keep larger long-term artifact history outside the deployment repository when needed.

## Verification

```powershell
Set-Location backend
.venv/Scripts/python.exe -m unittest discover -s tests -v
Set-Location ../frontend
npm run build
```

Tests cover hand-computable label/return paths, no same-day entry signal, overnight
gaps, cost accounting, no borrowing, initial drawdown, unavailable signals, holiday
session ordering, missing/duplicate data, null reproducibility and API export/freshness
behavior. CI runs these plus the frontend build without accessing cloud credentials.

## FYP interpretation

Present this as a reproducible, cost-aware historical experiment. Show strategy and
benchmark together, state date range/sample size/costs, open the trade CSV to explain
execution lag, and discuss both favourable and unfavourable outcomes. Do not claim
profitable deployment, a sentiment benefit without coverage, or selection-adjusted
significance. Survivorship bias, reused test data, calibration/feature validity and
price provenance still constrain the research conclusion.

Background: [probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
and [selection bias / deflated Sharpe](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
