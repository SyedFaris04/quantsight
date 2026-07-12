"""
notebooks/15_backtesting.py
─────────────────────────────────────────────────────────────────────────────
Backtests all FOUR model configurations required by the proposal (Objective 2
/ Section 6.7): XGBoost Finance, XGBoost + Sentiment, LSTM Finance, LSTM +
Sentiment — each evaluated as a top-5, equal-weight, weekly-rebalanced
portfolio with a 0.1% transaction cost, benchmarked against SPY buy-and-hold.

Previously this script only backtested a single combined XGBoost/LSTM pair
plus an ad hoc ensemble (not the 4-way Finance/Finance+Sentiment split the
proposal requires), and pointed at data/processed/ for prediction files that
actually live in data/predictions/ — both fixed here.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────
PREDICTIONS_FOLDER = "../backend/data/predictions/"
FEATURES_FILE       = "../backend/data/processed/features_finance.csv"
OUTPUTS_FOLDER      = "../backend/data/predictions/"
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

TRANSACTION_COST = 0.001   # 0.1% per trade
TOP_N            = 5       # pick top 5 stocks each week
INITIAL_CAPITAL  = 100000  # start with $100,000

MODEL_FILES = {
    "XGBoost Finance"        : "xgb_finance_predictions.csv",
    "XGBoost + Sentiment"    : "xgb_sentiment_predictions.csv",
    "LSTM Finance"           : "lstm_finance_predictions.csv",
    "LSTM + Sentiment"       : "lstm_sentiment_predictions.csv",
    "Ensemble (avg. of 4)"   : "ensemble_predictions.csv",
}
# ──────────────────────────────────────────────────────────


def load_prices() -> pd.DataFrame:
    """Close prices aren't in the prediction CSVs — merge them in from the feature file."""
    prices = pd.read_csv(FEATURES_FILE, usecols=["ticker", "date", "Close"])
    prices["date"] = pd.to_datetime(prices["date"])
    return prices


def load_predictions(filename: str, prices: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PREDICTIONS_FOLDER, filename))
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(prices, on=["ticker", "date"], how="left")
    # confidence (0-100, raw P(BUY)) doubles as the "buy probability" ranking signal
    df["prob"] = df["confidence"] / 100.0
    return df


# ── Helper: calculate performance metrics ─────────────────
# NOTE: returns passed in are WEEKLY (one observation per rebalance week),
# not daily — annualize with 52 (weeks/year), not 252 (trading days/year).
# Using 252 here previously inflated annual return/Sharpe/Sortino severely
# (e.g. it made SPY buy-and-hold look like a ~180%/year strategy).
PERIODS_PER_YEAR = 52


def calculate_metrics(returns, name="Strategy"):
    total_return  = (1 + returns).prod() - 1
    n_periods     = len(returns)
    annual_return = (1 + total_return) ** (PERIODS_PER_YEAR / n_periods) - 1 if n_periods > 0 else 0
    annual_vol    = returns.std() * np.sqrt(PERIODS_PER_YEAR)
    sharpe        = annual_return / annual_vol if annual_vol > 0 else 0

    downside      = returns[returns < 0].std() * np.sqrt(PERIODS_PER_YEAR)
    sortino       = annual_return / downside if downside > 0 else 0

    cumulative    = (1 + returns).cumprod()
    rolling_max   = cumulative.cummax()
    max_drawdown  = ((cumulative - rolling_max) / rolling_max).min()
    win_rate      = (returns > 0).mean()

    return {
        "Strategy"      : name,
        "Total Return"  : f"{total_return*100:.2f}%",
        "Annual Return" : f"{annual_return*100:.2f}%",
        "Annual Vol"    : f"{annual_vol*100:.2f}%",
        "Sharpe Ratio"  : f"{sharpe:.3f}",
        "Sortino Ratio" : f"{sortino:.3f}",
        "Max Drawdown"  : f"{max_drawdown*100:.2f}%",
        "Win Rate"      : f"{win_rate*100:.2f}%",
    }


# ── Core backtest function ────────────────────────────────
def run_backtest(df, prob_col, name):
    """
    Runs weekly rebalancing backtest on given predictions.
    df       : dataframe with date, ticker, Close-equivalent price, prob_col
    prob_col : column name for buy probability
    name     : strategy name for display
    """
    print(f"\n-- Running {name} " + "-" * (50 - len(name)))

    df         = df.copy()
    df["Week"] = df["date"].dt.to_period("W")
    weeks      = sorted(df["Week"].unique())

    portfolio_returns = []
    portfolio_dates   = []
    weekly_holdings   = []

    for week in weeks:
        week_data = df[df["Week"] == week].copy()
        if len(week_data) == 0:
            continue

        # Pick top N stocks by probability (excluding SPY — it's the benchmark, not tradeable universe)
        candidates = week_data[week_data["ticker"] != "SPY"]
        top_stocks = (candidates
                      .sort_values(prob_col, ascending=False)
                      .drop_duplicates("ticker")
                      .head(TOP_N))

        if len(top_stocks) == 0:
            continue

        actual_returns = []
        for _, row in top_stocks.iterrows():
            ticker     = row["ticker"]
            week_start = week_data["date"].min()
            week_end   = week_data["date"].max()

            ticker_data = df[
                (df["ticker"] == ticker) &
                (df["date"] >= week_start) &
                (df["date"] <= week_end)
            ].sort_values("date")

            if len(ticker_data) >= 2:
                start_price = ticker_data.iloc[0]["Close"]
                end_price   = ticker_data.iloc[-1]["Close"]
                ret         = (end_price - start_price) / start_price
                actual_returns.append(ret)

        if actual_returns:
            avg_return = np.mean(actual_returns) - TRANSACTION_COST
            portfolio_returns.append(avg_return)
            portfolio_dates.append(week_data["date"].min())
            weekly_holdings.append(top_stocks["ticker"].tolist())

    returns_series  = pd.Series(portfolio_returns, index=portfolio_dates)
    portfolio_value = INITIAL_CAPITAL * (1 + returns_series).cumprod()

    print(f"Weeks traded       : {len(portfolio_returns)}")
    print(f"Starting capital   : ${INITIAL_CAPITAL:,}")
    if len(portfolio_value) > 0:
        print(f"Final value        : ${portfolio_value.iloc[-1]:,.2f}")
    print(f"Top stocks example : {weekly_holdings[0] if weekly_holdings else []}")

    return returns_series, portfolio_value, weekly_holdings


def main():
    print("Loading prices and predictions for all 4 model configurations...")
    prices = load_prices()
    dfs = {name: load_predictions(fname, prices) for name, fname in MODEL_FILES.items()}
    for name, df in dfs.items():
        print(f"  {name:<22}: {len(df):,} rows, {df['ticker'].nunique()} tickers "
              f"({df['date'].min().date()} -> {df['date'].max().date()})")

    # SPY benchmark — sourced from any one of the prediction files (all share the same price history)
    any_df = next(iter(dfs.values()))
    spy_df = any_df[any_df["ticker"] == "SPY"][["date", "Close"]].sort_values("date")
    spy_weekly = (spy_df.set_index("date")["Close"]
                  .resample("W").last()
                  .pct_change()
                  .dropna())
    spy_value = INITIAL_CAPITAL * (1 + spy_weekly).cumprod()

    # ── Run backtest for each of the 4 model configurations ──
    results = {}
    for name, df in dfs.items():
        returns, value, holdings = run_backtest(df, "prob", name)
        results[name] = {"returns": returns, "value": value, "holdings": holdings}

    # ── Metrics table ──
    all_metrics = [calculate_metrics(r["returns"], name) for name, r in results.items()]
    all_metrics.append(calculate_metrics(spy_weekly, "SPY Buy&Hold"))

    print("\n")
    print("=" * 95)
    print("BACKTESTING RESULTS -- ALL 4 MODEL CONFIGURATIONS vs SPY")
    print("=" * 95)
    metrics_keys = ["Total Return", "Annual Return", "Annual Vol",
                    "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate"]
    header = f"\n{'Metric':<16}" + "".join(f"{m['Strategy']:>18}" for m in all_metrics)
    print(header)
    print("-" * len(header))
    for key in metrics_keys:
        row = f"{key:<16}" + "".join(f"{m[key]:>18}" for m in all_metrics)
        print(row)

    # ── Save results ──
    print("\nSaving results...")
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(os.path.join(OUTPUTS_FOLDER, "backtest_results.csv"), index=False)

    # Portfolio values over time — align all strategies to XGBoost Finance's dates
    ref_dates = results["XGBoost Finance"]["returns"].index
    portfolio_df = pd.DataFrame({"date": ref_dates})
    for name, r in results.items():
        col = name.replace(" ", "_").replace("+", "Plus")
        aligned = r["value"].reindex(ref_dates, method="nearest")
        portfolio_df[f"{col}_Value"] = aligned.values
    spy_aligned = spy_value.reindex(ref_dates, method="nearest")
    portfolio_df["SPY_Value"] = spy_aligned.values
    portfolio_df.to_csv(os.path.join(OUTPUTS_FOLDER, "portfolio_values.csv"), index=False)

    # Weekly holdings per model
    max_weeks = max(len(r["holdings"]) for r in results.values())
    holdings_data = {"Week": [str(i + 1) for i in range(max_weeks)]}
    for name, r in results.items():
        col = name.replace(" ", "_").replace("+", "Plus")
        h = r["holdings"] + [[]] * (max_weeks - len(r["holdings"]))
        holdings_data[f"{col}_Holdings"] = [", ".join(w) for w in h]
    holdings_df = pd.DataFrame(holdings_data)
    holdings_df.to_csv(os.path.join(OUTPUTS_FOLDER, "weekly_holdings.csv"), index=False)

    print(f"\nFull backtesting complete!")
    print(f"   Results saved    -> {OUTPUTS_FOLDER}backtest_results.csv")
    print(f"   Portfolio values -> {OUTPUTS_FOLDER}portfolio_values.csv")
    print(f"   Weekly holdings  -> {OUTPUTS_FOLDER}weekly_holdings.csv")


if __name__ == "__main__":
    main()
