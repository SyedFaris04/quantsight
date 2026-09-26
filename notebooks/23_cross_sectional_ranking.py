"""
notebooks/23_cross_sectional_ranking.py
─────────────────────────────────────────────────────────────────────────────
Cross-sectional ranking evaluation — the single most defensible way to show
this project's signal carries real, replicable value, per the research
synthesis (Gu-Kelly-Xiu 2020; Microsoft Qlib model zoo; "Building
Cross-Sectional Systematic Strategies By Learning to Rank", arXiv:2012.07149).

WHY THIS EXISTS
    Per-stock binary direction accuracy on liquid large-caps at a 5-day
    horizon is ~50-53% in every leakage-checked study — the market is close
    to efficient at that horizon, so a high accuracy number is either luck,
    an easier target (index/ETF, single trending ticker), or leakage. The
    honest, standard-in-quant way to measure signal is NOT "how often is the
    up/down call right" but "on a given day, does the model rank the 44
    tickers in the right order by subsequent return?" — because that ranking
    is what a real strategy trades (long the top, short/avoid the bottom).

    A model can sit at 52% accuracy and STILL have a genuinely useful,
    tradeable ranking signal — the two measure different things.

METRICS (all computed per-date across the 44-ticker cross-section, then
aggregated over the test period — never pooled across dates, which would
leak market-wide up/down days into the "skill" number):
    IC       — daily Pearson corr(raw_prob, forward_return), mean over dates.
    Rank IC  — daily Spearman rank corr, mean over dates (robust to outliers;
               the headline number quant desks quote). Rank IC ~0.03-0.05 is
               a real, useful signal at this scale (cf. Qlib's best models).
    ICIR     — mean(Rank IC) / std(Rank IC): consistency, not just size. This
               is the closest single number to an "information ratio" for the
               signal and is what actually predicts whether it survives live.
    Decile spread — each day, long the top-decile tickers by raw_prob and
               short the bottom decile; the mean forward-return gap is what
               the signal is worth if traded. Reported as mean daily spread
               and an annualized Sharpe (×√252).

IMPORTANT — uses RAW model probability, NOT the saved calibrated confidence.
    Isotonic calibration bins probabilities into a few flat levels (great for
    an honest confidence *number*, useless for *ranking* — most tickers tie
    on a given day). Ranking needs the raw predict_proba resolution, so this
    reproduces the exact test split and recomputes raw probabilities from the
    saved model bundles.

Diagnostic only — does not retrain or alter any model. Reproduces the same
purged 80/20 test split the training scripts use (cv_utils.purge_train_test_boundary),
so these numbers describe the exact deployed models on genuinely held-out data.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

import sys
BASE_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BASE_DIR))
from cv_utils import purge_train_test_boundary  # noqa: E402

PROCESSED_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "data" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
METRICS_FILE    = PREDICTIONS_DIR / "model_metrics.json"

# Only the XGBoost variants are evaluated here: they're the reference/strongest
# production models and expose raw predict_proba directly from a single saved
# bundle, so ranking can be recomputed exactly with no sequence-window
# reconstruction. LSTM/GRU/Transformer ranking is a straightforward extension
# (rebuild their 10-day sequences, same forward-return join) — left as a
# follow-up rather than duplicating that fragile logic here.
MODELS = {
    "xgb_finance"   : (MODELS_DIR / "xgb_finance.pkl",   PROCESSED_DIR / "features_finance.csv"),
    "xgb_sentiment" : (MODELS_DIR / "xgb_sentiment.pkl", PROCESSED_DIR / "features_sentiment.csv"),
}

# Top/bottom fraction for the long-short spread. 0.2 (quintiles) rather than
# strict deciles because with only 44 names a decile is ~4 stocks — too few
# for a stable daily spread; quintiles (~9 per side) are the standard
# small-universe compromise.
LONGSHORT_FRAC = 0.2
TRADING_DAYS   = 252
MIN_NAMES_PER_DATE = 10  # skip dates with too few names to rank meaningfully


def load_forward_returns() -> pd.DataFrame:
    """Continuous 5-day-forward return (the ranking target) from the same
    source the binary label is derived from — features_stock.csv's
    Forward_return, which is Close.pct_change(5).shift(-5)."""
    fs = pd.read_csv(PROCESSED_DIR / "features_stock.csv", usecols=["Date", "Ticker", "Forward_return"])
    fs["date"]   = pd.to_datetime(fs["Date"])
    fs["ticker"] = fs["Ticker"].astype(str)
    return fs[["date", "ticker", "Forward_return"]]


def reproduce_test_predictions(model_path: Path, csv_path: Path) -> pd.DataFrame:
    """Reproduce the exact purged 80/20 test split and recompute RAW
    probabilities for it — matches what train_xgboost.py deployed."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model, scaler, feature_cols = bundle["model"], bundle["scaler"], bundle["features"]

    df = pd.read_csv(csv_path)
    df["date"]   = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    split_date = df["date"].quantile(0.80)
    test_df = df[df["date"] > split_date].copy()

    X = scaler.transform(test_df[feature_cols].values)
    test_df["raw_prob"] = model.predict_proba(X)[:, 1]
    return test_df[["date", "ticker", "raw_prob"]]


def compute_ranking_metrics(df: pd.DataFrame) -> dict:
    """df has columns: date, raw_prob, Forward_return. Computes per-date
    cross-sectional metrics, then aggregates."""
    ics, rank_ics, spreads = [], [], []

    for _, day in df.groupby("date"):
        day = day.dropna(subset=["raw_prob", "Forward_return"])
        if len(day) < MIN_NAMES_PER_DATE:
            continue
        # Constant predictions on a day carry no ranking info — skip (corr undefined).
        if day["raw_prob"].nunique() < 2:
            continue

        ics.append(pearsonr(day["raw_prob"], day["Forward_return"])[0])
        rank_ics.append(spearmanr(day["raw_prob"], day["Forward_return"])[0])

        k = max(1, int(round(len(day) * LONGSHORT_FRAC)))
        ranked = day.sort_values("raw_prob")
        short_leg = ranked.head(k)["Forward_return"].mean()
        long_leg  = ranked.tail(k)["Forward_return"].mean()
        spreads.append(long_leg - short_leg)

    ics      = np.array([x for x in ics      if np.isfinite(x)])
    rank_ics = np.array([x for x in rank_ics if np.isfinite(x)])
    spreads  = np.array([x for x in spreads  if np.isfinite(x)])

    mean_rank_ic = float(np.mean(rank_ics)) if len(rank_ics) else float("nan")
    std_rank_ic  = float(np.std(rank_ics))  if len(rank_ics) else float("nan")
    icir = (mean_rank_ic / std_rank_ic) if std_rank_ic else float("nan")

    # Spread is a 5-day-forward return; the label horizon overlaps day to day,
    # so this Sharpe is an optimistic, illustrative annualization (independent-
    # period assumption is violated by overlapping windows) — reported as a
    # relative comparison across models, not a live-tradeable Sharpe.
    mean_spread = float(np.mean(spreads)) if len(spreads) else float("nan")
    std_spread  = float(np.std(spreads))  if len(spreads) else float("nan")
    spread_sharpe = (mean_spread / std_spread * np.sqrt(TRADING_DAYS)) if std_spread else float("nan")

    return {
        "n_dates"              : int(len(rank_ics)),
        "ic_mean"              : round(float(np.mean(ics)), 4) if len(ics) else None,
        "rank_ic_mean"         : round(mean_rank_ic, 4),
        "rank_ic_std"          : round(std_rank_ic, 4),
        "icir"                 : round(icir, 3),
        "rank_ic_positive_pct" : round(float(np.mean(rank_ics > 0)) * 100, 1) if len(rank_ics) else None,
        "longshort_spread_mean_5d"      : round(mean_spread * 100, 4) if np.isfinite(mean_spread) else None,
        "longshort_spread_sharpe_annae" : round(spread_sharpe, 3) if np.isfinite(spread_sharpe) else None,
    }


def main():
    fwd = load_forward_returns()
    all_ranking = {}

    print(f"\n{'='*72}\nCross-Sectional Ranking Evaluation (raw probabilities, purged test set)\n{'='*72}")
    print("Interpreting Rank IC: ~0.02 weak | ~0.03-0.05 genuinely useful at this")
    print("scale (cf. Qlib's best models) | >0.10 on 44 large-caps = audit for leakage.\n")

    for key, (model_path, csv_path) in MODELS.items():
        if not model_path.exists():
            print(f"  {key}: model not found, skipping")
            continue

        preds = reproduce_test_predictions(model_path, csv_path)
        merged = preds.merge(fwd, on=["date", "ticker"], how="left")
        metrics = compute_ranking_metrics(merged)
        all_ranking[key] = metrics

        print(f"-- {key} ---------------------------------------------")
        print(f"   Dates evaluated      : {metrics['n_dates']}")
        print(f"   IC (Pearson)         : {metrics['ic_mean']}")
        print(f"   Rank IC (Spearman)   : {metrics['rank_ic_mean']}  (std {metrics['rank_ic_std']})")
        print(f"   ICIR                 : {metrics['icir']}")
        print(f"   Rank IC > 0 on       : {metrics['rank_ic_positive_pct']}% of dates")
        print(f"   Long-short 5d spread : {metrics['longshort_spread_mean_5d']}%  "
              f"(illustrative Sharpe {metrics['longshort_spread_sharpe_annae']})")
        print()

    # Write to a SEPARATE file, not model_metrics.json — the training scripts
    # iterate model_metrics.json expecting every entry to have the per-model
    # schema (model/variant/accuracy), so a ranking block inside it crashes
    # their summary print. Keeping ranking metrics standalone avoids that.
    out_path = PREDICTIONS_DIR / "ranking_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_ranking, f, indent=2)
    print(f"Saved ranking metrics -> {out_path}")


if __name__ == "__main__":
    main()
