# Training history: 2015 vs 2005

Prepared 16 November 2026. Lecturer suggestion: train on more years so the models
see major market crises. The previous panel started on 2015-05-01, so it already
contained COVID (2020) and the 2022 rate shock, but not the 2008 financial crisis
or the 2011 debt-ceiling selloff. This study adds 2005–2014 and measures what
changes. It does not open or score the 2025 candidate holdout.

Live forecasts, Market and Detail still use the current production models. The
Dashboard has a **Trained from 2015 / Trained from 2005** switch. The new
**Training history** page (`/timeline`) shows every comparison below.

## Data

- Frozen Yahoo Finance snapshot `20261115T154430187216Z`, from 2004-01-02 to 2025-12-31.
- It covers the 44 instruments and six sector ETFs, and every one passes the
  NYSE calendar and validity audit
  ([audit](fresh_price_audit_20261115T154430187216Z.json)).
- Its 2015–2024 dates match the September snapshot exactly. The largest
  adjusted-close difference is 1.44%, because the vendor re-adjusted prices for
  later dividends.
- Instruments with data: 37 in 2008, 40 in 2014, and all 44 from 2019. META,
  TSLA, ABBV, PYPL, SNAP, MRNA and UBER enter after they listed.
- Features are built by the unchanged production `build_features.py` path
  (`research/timeline_features.py`). The 2015-onward slice has 104,133 rows,
  against production's 104,076.

## Method

**Two served-model timelines.** The four served models (XGBoost and LSTM, each
finance-only and finance+sentiment) are trained twice from the same download,
code and settings:

| | Trained from 2015 | Trained from 2005 |
|---|---|---|
| Training rows | 83,057 (2015-05-01 to 2023-01-26) | 181,522 (2005-01-03 to 2023-01-26) |
| Crises in training | 2015–16, late 2018, COVID, 2022 | as 2015, plus 2008 and 2011 |

- **Test rows are identical:** 20,812 rows from 2023-02-06 to 2024-12-20. That
  is the same window as production, because the split is pinned at 2023-02-03.
- Each LSTM is trained with seeds 42, 43 and 44. The seed-42 versions are the
  ones the Dashboard serves.

**Crisis walk-forward.** This uses a compact 13-feature finance model.
- Each year from 2008 to 2024 is predicted by models fitted only on earlier years
  and calibrated on the year before.
- Two arms: one fitted from 2005, the other from 2015 (whose first predicted
  year is 2017).
- The protocol was saved before fitting, and the fit budget was asserted:
  75 predictor fits plus 75 calibrator fits.

Differences come with a paired 20-session block bootstrap 95% interval. None is
corrected for multiple comparisons.

## Results

### Served models, same test rows (2005 minus 2015)

| Model | Acc 2015 | Acc 2005 | Accuracy difference, pts [95%] | Brier 2015 | Brier 2005 | Brier difference [95%] |
|---|---:|---:|---|---:|---:|---|
| XGBoost finance | 52.97% | 52.90% | −0.07 [−0.86, +0.72] | 0.2483 | 0.2483 | +0.0000 [−0.0003, +0.0003] |
| XGBoost finance+sentiment | 52.77% | 52.83% | +0.06 [−0.96, +0.96] | 0.2483 | 0.2483 | −0.0001 [−0.0005, +0.0003] |
| LSTM finance | 53.32% | 51.91% | −1.41 [−2.90, +0.07] | 0.2506 | 0.2504 | −0.0002 [−0.0011, +0.0006] |
| LSTM finance+sentiment | 52.69% | 51.78% | −0.91 [−2.95, +0.83] | 0.2528 | 0.2509 | −0.0019 [−0.0041, −0.0002] |

On these rows, always predicting UP scores 54.19% (XGBoost rows) or 54.71% (LSTM
rows).

LSTM accuracy across the three seeds:
- Trained from 2015: finance 53.3–54.4%, finance+sentiment 52.7–54.4%.
- Trained from 2005: finance 51.9–53.2%, finance+sentiment 51.8–53.8%.

The 2005 LSTMs were slightly less accurate but slightly better calibrated. Only
the sentiment LSTM's Brier improvement has an interval that excludes zero.

### Trading simulation

The four models' probabilities are averaged. The strategy takes the top 5 BUY
calls, rebalances weekly at the next open, and pays 10 + 5 bps per side. It runs
from 2023-02-22 to 2024-12-20 (462 sessions).

| Strategy | Net return | Yearly | Sharpe | Worst drop |
|---|---:|---:|---:|---:|
| Trained from 2015 | 105.45% | 48.11% | 2.02 | −11.33% |
| Trained from 2005 | 37.81% | 19.11% | 1.25 | −11.45% |
| SPY buy and hold | 51.69% | 25.52% | 1.89 | −9.97% |

This is one exploratory path on a window that has been examined many times. The
classification differences above are not distinguishable from zero, so this gap
in returns should not be read as evidence that either history is better for
trading.

### What the models rely on

`rsi_rel_market` and `market_volatility` stay at the top for both versions. With
2005 history, plain `rsi` rises from rank 12 to rank 3 in XGBoost finance and to
rank 2 in XGBoost finance+sentiment. The page lists every rank change.

### Crisis walk-forward (paired 2017–2024, 85,494 predictions, 2005 minus 2015)

| Candidate | Brier difference [95%] | Accuracy difference, pts [95%] |
|---|---|---|
| XGBoost calibrated | −0.0005 [−0.0010, −0.0001] | +0.06 [−0.11, +0.23] |
| XGBoost raw | −0.0045 [−0.0066, −0.0026] | +1.78 [+0.60, +3.01] |
| Histogram boosting raw | −0.0049 [−0.0071, −0.0028] | +1.93 [+0.78, +3.18] |
| Logistic raw | −0.0021 [−0.0032, −0.0011] | +1.37 [+0.60, +2.34] |

The longer history clearly helps the **uncalibrated** compact models. After
year-ahead calibration, the gain shrinks to a small Brier improvement with no
accuracy change. In most years, neither arm beats the constant training-prior
probability on Brier loss. Among the named crisis episodes, only 2015–16 is a
near-tie: 0.2501 against a baseline of 0.2502. In every other episode the model
is worse than the baseline. For example, in the 2008 financial crisis the
calibrated XGBoost Brier was 0.2619 against a baseline of 0.2580.

## Conclusion

Ten more years of history, including 2008, does **not** make the served models
measurably more accurate on the 2023–2024 test window. XGBoost results are
unchanged. The LSTMs lose about 1 point of accuracy, but the 95% interval
includes zero. Calibration is slightly better.

The cleaner walk-forward evidence shows that the longer history helps raw model
scores. Most of that disappears once probabilities are calibrated, and no
version reliably beats a constant baseline. This fits the project's earlier
finding: at this scale, the signal is limited by the information available, not
by the amount of price history.

Production serving is unchanged. The Dashboard switch lets you inspect both
versions.

## Limitations

- Survivor universe: pre-2015 years flatter what a 2005 investor could have chosen.
- There is no text data before 2015, so the 2005 sentiment models mostly learn more price history.
- Vendor adjustments are retrieval-time, not point-in-time.
- The 2005 LSTM's own test set began at its 80% date quantile (2021-04-27), so its
  file keeps 440 rows the 2015 file drops at sequence warm-up. Comparisons use the
  20,372 common rows.
- The retrained 2015 models differ from production because of the new download and
  indicator code: XGBoost finance agrees on 87.7% of test predictions and LSTM
  finance+sentiment on 74.1%.

## Reproduce

From `backend/` with `.venv`:

```
python -m research.freeze_prices --protocol research/protocols/fresh_price_snapshot_v2.json
python -m research.timeline_features --snapshot data/research/fresh_prices/20261115T154430187216Z
python train_xgboost.py --features-dir data/research/timelines --start <2015-05-01|2005-01-03> --split-date 2023-02-03 --out-models data/models/timelines/<T> --out-predictions data/predictions/timelines/<T>
python -m research.train_timeline_lstm --timeline <T> --seed <42|43|44> --out-models <dir> --out-predictions <dir>
python -m research.crisis_walkforward --snapshot data/research/fresh_prices/20261115T154430187216Z
python -m research.timeline_comparison
```

`build_features.py`, `cv_utils.py` and `train_lstm.py` were deliberately left
unedited. Their hashes guard live forecast metadata, the served TreeSHAP
snapshots and the published XAI comparison.
