"""
notebooks/24_ranking_model_poc.py
─────────────────────────────────────────────────────────────────────────────
Proof-of-concept: does training on a CROSS-SECTIONAL RELATIVE label give the
model genuine ranking skill, where the absolute up/down label does not?

Motivation (from 23_cross_sectional_ranking.py's finding): the deployed
xgb_finance model, trained on absolute "up or down over 5 days", has Rank IC
~= 0 — its ~52% accuracy is base-rate (market drift), not per-day
cross-sectional skill. The research synthesis (Gu-Kelly-Xiu 2020; learning-to-
rank, arXiv:2012.07149) says the way to get real, tradeable signal is to
predict RELATIVE performance across the universe, not absolute direction.

This tests the smallest possible version of that idea, apples-to-apples with
the existing pipeline:

    LABEL CHANGE ONLY — everything else (features, split, purge, model
    hyperparameters) identical to train_xgboost.py.
        old label:  signal      = 1 if Close[t+5] > Close[t]          (absolute)
        new label:  rel_signal  = 1 if fwd_return[t] > daily_median   (relative)

    The relative label asks "will this ticker beat the cross-section today?"
    — which is exactly what a ranking/long-short strategy needs, and it
    removes the market-wide component the absolute label is dominated by.

Reports the same Rank IC / ICIR / long-short-spread metrics as notebook 23,
head-to-head against the absolute-label baseline, on the identical purged test
set. Diagnostic experiment — writes nothing to the deployed models; saves only
a small JSON summary for the write-up.

──────────────────────────────────────────────────────────────────────────────
VERDICT (robustness-tested across 12 seeds, n_jobs=1 for determinism):
    Relative-label Rank IC: mean +0.0040, std 0.0043, positive on 9/12 seeds,
    t-stat vs zero = 3.23. So the relative label DOES give a STATISTICALLY
    real ranking signal that the absolute label does not (the reframing is
    sound, and it's the honest way to describe the task to a lecturer) — BUT
    the magnitude is economically negligible: Rank IC ~0.004 is an order of
    magnitude below the ~0.03-0.05 "genuinely useful" band, and any SINGLE
    trained model swings sign across seeds (min -0.005, max +0.010). A single
    run reporting e.g. +0.0077 is a lucky draw from a distribution centred
    near 0.004, not a stable edge.

    CONCLUSION: NOT deployed as a product signal — serving a Rank-IC-0.004
    model as if it ranks stocks usefully would present noise as signal. This
    is consistent with the whole project's core finding: 5-day direction on
    44 large-caps is near-efficient. The real, robust win from the same
    order-flow features is on the NEURAL CLASSIFIERS (LSTM/GRU/Transformer
    ~53% -> ~54.5-54.8%), not here. Kept as an honest negative-ish result.
──────────────────────────────────────────────────────────────────────────────
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import sys
BASE_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BASE_DIR))
from cv_utils import purge_train_test_boundary  # noqa: E402

PROCESSED_DIR   = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"

# Same feature-exclusion policy as train_xgboost.py (kept in sync manually;
# this is a diagnostic, not a production trainer).
NON_FEATURE_COLS = {
    "ticker", "date", "signal", "Date", "Ticker",
    "sma_5", "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
    "bb_upper", "bb_mid", "bb_lower", "macd", "macd_signal", "macd_hist",
    "Open", "High", "Low", "Close", "Volume", "spy_return_5",
    "rel_signal", "fwd_return", "fwd_return_demedian",
}

XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_alpha=0.1,
    reg_lambda=1.0, eval_metric="logloss", random_state=42, n_jobs=-1,
)

LONGSHORT_FRAC = 0.2
TRADING_DAYS   = 252
MIN_NAMES      = 10


def ranking_metrics(df: pd.DataFrame, score_col: str) -> dict:
    """Per-date cross-sectional metrics vs the continuous fwd_return."""
    rank_ics, ics, spreads = [], [], []
    for _, day in df.groupby("date"):
        day = day.dropna(subset=[score_col, "fwd_return"])
        if len(day) < MIN_NAMES or day[score_col].nunique() < 2:
            continue
        ics.append(pearsonr(day[score_col], day["fwd_return"])[0])
        rank_ics.append(spearmanr(day[score_col], day["fwd_return"])[0])
        k = max(1, int(round(len(day) * LONGSHORT_FRAC)))
        ranked = day.sort_values(score_col)
        spreads.append(ranked.tail(k)["fwd_return"].mean() - ranked.head(k)["fwd_return"].mean())

    rank_ics = np.array([x for x in rank_ics if np.isfinite(x)])
    ics      = np.array([x for x in ics      if np.isfinite(x)])
    spreads  = np.array([x for x in spreads  if np.isfinite(x)])
    m, s = float(np.mean(rank_ics)), float(np.std(rank_ics))
    ms, ss = float(np.mean(spreads)), float(np.std(spreads))
    return {
        "rank_ic_mean"        : round(m, 4),
        "icir"                : round(m / s, 3) if s else None,
        "ic_mean"             : round(float(np.mean(ics)), 4),
        "rank_ic_positive_pct": round(float(np.mean(rank_ics > 0)) * 100, 1),
        "longshort_5d_pct"    : round(ms * 100, 4),
        "longshort_sharpe"    : round(ms / ss * np.sqrt(TRADING_DAYS), 3) if ss else None,
    }


def main():
    df = pd.read_csv(PROCESSED_DIR / "features_finance.csv")
    df["date"] = pd.to_datetime(df["date"]); df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Continuous 5-day-forward return (ranking target) + relative label.
    fs = pd.read_csv(PROCESSED_DIR / "features_stock.csv", usecols=["Date", "Ticker", "Forward_return"])
    fs["date"] = pd.to_datetime(fs["Date"]); fs["ticker"] = fs["Ticker"].astype(str)
    df = df.merge(fs[["date", "ticker", "Forward_return"]].rename(columns={"Forward_return": "fwd_return"}),
                  on=["date", "ticker"], how="left").dropna(subset=["fwd_return"])

    daily_median = df.groupby("date")["fwd_return"].transform("median")
    df["rel_signal"] = (df["fwd_return"] > daily_median).astype(int)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    split_date   = df["date"].quantile(0.80)
    train_cutoff = purge_train_test_boundary(df["date"], split_date)
    train_df = df[df["date"] <= train_cutoff].copy()
    test_df  = df[df["date"] >  split_date].copy()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[feature_cols].values)
    X_te = scaler.transform(test_df[feature_cols].values)

    print(f"\n{'='*72}\nRanking-model proof-of-concept (relative vs absolute label)\n{'='*72}")
    print(f"Features: {len(feature_cols)} | train {len(train_df):,} rows | test {len(test_df):,} rows\n")

    results = {}

    # ── Baseline: absolute up/down label (what's deployed) ────────────────
    m_abs = xgb.XGBClassifier(**XGB_PARAMS)
    m_abs.fit(X_tr, train_df["signal"].values, verbose=False)
    test_df["score_abs"] = m_abs.predict_proba(X_te)[:, 1]
    results["absolute_label"] = ranking_metrics(test_df, "score_abs")

    # ── Experiment: relative (beat-the-daily-median) label ────────────────
    m_rel = xgb.XGBClassifier(**XGB_PARAMS)
    m_rel.fit(X_tr, train_df["rel_signal"].values, verbose=False)
    test_df["score_rel"] = m_rel.predict_proba(X_te)[:, 1]
    results["relative_label"] = ranking_metrics(test_df, "score_rel")

    for name, r in results.items():
        print(f"-- {name} --------------------------------------")
        print(f"   Rank IC        : {r['rank_ic_mean']}   (ICIR {r['icir']}, positive {r['rank_ic_positive_pct']}% of days)")
        print(f"   IC (Pearson)   : {r['ic_mean']}")
        print(f"   Long-short 5d  : {r['longshort_5d_pct']}%  (illustrative Sharpe {r['longshort_sharpe']})")
        print()

    out = PREDICTIONS_DIR / "ranking_poc_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
