"""Long-only daily accounting with prior-close decisions and next-open execution.

The engine does not train, tune, download data or access future predictions.
All prices must be on a consistent adjustment basis. Missing bars fail closed.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    top_n: int = 5
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    annual_risk_free_rate: float = 0.0
    min_probability: float = 0.5
    max_position_weight: float = 0.25
    periods_per_year: int = 252
    seed: int = 42

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all():
            raise ValueError("Configuration must be finite")
        if self.initial_capital <= 0 or self.top_n < 1 or int(self.top_n) != self.top_n:
            raise ValueError("Positive capital and integer top_n required")
        if not 0 <= self.commission_bps + self.slippage_bps < 1000:
            raise ValueError("Combined per-side costs must be below 1000 bps")
        if min(self.commission_bps, self.slippage_bps) < 0:
            raise ValueError("Costs cannot be negative")
        if not 0 <= self.min_probability <= 1 or not 0 < self.max_position_weight <= 1:
            raise ValueError("Invalid probability or weight")
        if self.annual_risk_free_rate <= -1 or self.periods_per_year < 1:
            raise ValueError("Invalid risk-free rate or annualization")

    @property
    def cost_rate(self):
        return (self.commission_bps + self.slippage_bps) / 10_000

    @property
    def daily_risk_free(self):
        return (1 + self.annual_risk_free_rate) ** (1 / self.periods_per_year) - 1


def normalise_rows(frame, required):
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None).dt.normalize()
    if frame[["date", "ticker"]].isna().any().any():
        raise ValueError("Missing date or ticker")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate ticker/session rows")
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def prepare_prices(prices):
    prices = normalise_rows(prices, ["date", "ticker", "Open", "Close"])
    if not np.isfinite(prices[["Open", "Close"]].to_numpy(float)).all():
        raise ValueError("Prices must be finite")
    if (prices[["Open", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    opens = prices.pivot(index="date", columns="ticker", values="Open")
    closes = prices.pivot(index="date", columns="ticker", values="Close")
    if "SPY" not in opens:
        raise ValueError("SPY is required to define the observed trading calendar")
    dates = opens.index[opens["SPY"].notna() & closes["SPY"].notna()]
    return opens.loc[dates], closes.loc[dates]


def prepare_predictions(predictions):
    result = normalise_rows(predictions, ["date", "ticker", "predicted_signal", "confidence"])
    if not result.predicted_signal.isin([0, 1]).all():
        raise ValueError("predicted_signal must be binary")
    if not np.isfinite(result.confidence.to_numpy(float)).all() or not result.confidence.between(0, 100).all():
        raise ValueError("confidence must contain P(up) in percent, within [0, 100]")
    result["p_up"] = result.confidence / 100
    return result


def _post_cost_equity(equity, current_values, weights, cost_rate):
    """Solve target values AND their costs together, preventing borrowed cash."""
    if cost_rate == 0:
        return equity
    # The map is a contraction: sum(weights) <= 1 and cost_rate < .1.
    # This converges to the same self-financing solution without dozens of
    # bisections for each portfolio in the random-null study.
    value = equity
    for _ in range(50):
        updated = equity - cost_rate * np.abs(weights * value - current_values).sum()
        if abs(updated - value) <= 1e-10:
            return updated
        value = updated
    raise ArithmeticError("Rebalance cost solver did not converge")


def simulate(prices, predictions, config, start, end, mode="model", random_seed=None, record_trades=True):
    """Modes: model, spy (buy/hold), universe (weekly), random (matched slots).

    Initial entry is at the first evaluation open, subsequent rebalances at the
    first observed session of each week. Each decision uses ONLY the immediately
    preceding session. Missing prediction slots stay cash, never backfilled.
    Random mode matches the model's number of eligible positions per rebalance.
    Final holdings are liquidated at the last close, with costs for every mode.
    """
    if mode not in {"model", "spy", "universe", "random"}:
        raise ValueError("Unknown strategy mode")
    opens, closes = prepare_prices(prices)
    dates = opens.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    indices = np.flatnonzero((dates >= start) & (dates <= end))
    if len(indices) < 2 or indices[0] == 0:
        raise ValueError("Need at least two evaluation sessions and a prior signal session")
    if dates[indices[0]] != start or dates[indices[-1]] != end:
        raise ValueError("Evaluation boundaries must be observed trading sessions")
    tickers = list(opens.columns)
    universe = [t for t in tickers if t != "SPY"]
    if not universe:
        raise ValueError("Empty investment universe")
    # Never silently forward-fill an untradeable or missing instrument.
    relevant = slice(indices[0] - 1, indices[-1] + 1)
    if opens.iloc[relevant].isna().any().any() or closes.iloc[relevant].isna().any().any():
        raise ValueError("Incomplete OHLCV panel in evaluation window; repair data before evaluation")
    predictions = prepare_predictions(predictions) if predictions is not None else None
    if mode in {"model", "random"} and predictions is None:
        raise ValueError("Predictions required")
    groups = {} if predictions is None else {d: g for d, g in predictions.groupby("date")}
    op, cl = opens.to_numpy(float), closes.to_numpy(float)
    shares = np.zeros(len(tickers))
    cash, previous_equity = config.initial_capital, config.initial_capital
    daily, trades, rebalances = [], [], []
    rng = np.random.default_rng(config.seed if random_seed is None else random_seed)

    def log_trades(day, signal_day, delta, price, reason):
        if not record_trades:
            return
        for j in np.flatnonzero(np.abs(delta * price) > 1e-7):
            notional = abs(delta[j] * price[j])
            trades.append({"date": str(day.date()), "signal_date": str(signal_day.date()),
                           "ticker": tickers[j], "side": "BUY" if delta[j] > 0 else "SELL",
                           "quantity": float(abs(delta[j])), "reference_price": float(price[j]),
                           "notional": float(notional),
                           "commission": float(notional * config.commission_bps / 10000),
                           "slippage": float(notional * config.slippage_bps / 10000), "reason": reason})

    for number, i in enumerate(indices):
        day, signal_day = dates[i], dates[i - 1]
        rebalance = number == 0 or (mode != "spy" and day.to_period("W") != signal_day.to_period("W"))
        daily_cost, traded, coverage = 0.0, 0.0, None
        if rebalance:
            weights = np.zeros(len(tickers))
            if mode == "spy":
                weights[tickers.index("SPY")] = 1.0
                selected = ["SPY"]
            elif mode == "universe":
                selected = universe
                for t in selected:
                    weights[tickers.index(t)] = 1 / len(universe)
            else:
                rows = groups.get(signal_day)
                eligible = []
                if rows is not None:
                    rows = rows[rows.ticker.isin(universe)]
                    coverage = float(len(rows) / len(universe))
                    rows = rows[(rows.predicted_signal == 1) & (rows.p_up >= config.min_probability)]
                    eligible = rows.sort_values(["p_up", "ticker"], ascending=[False, True]).ticker.tolist()
                else:
                    coverage = 0.0
                selected = eligible[:config.top_n]
                if mode == "random" and selected:
                    selected = rng.choice(universe, size=len(selected), replace=False).tolist()
                for t in selected:
                    weights[tickers.index(t)] = min(1 / config.top_n, config.max_position_weight)
            current_values = shares * op[i]
            opening_equity = cash + current_values.sum()
            after_cost = _post_cost_equity(opening_equity, current_values, weights, config.cost_rate)
            target_shares = weights * after_cost / op[i]
            delta = target_shares - shares
            traded = float(np.abs(delta * op[i]).sum())
            daily_cost = traded * config.cost_rate
            cash = float(opening_equity - (target_shares * op[i]).sum() - daily_cost)
            if cash < -1e-6:
                raise ArithmeticError("Rebalance borrowed cash")
            cash = max(0.0, cash)
            log_trades(day, signal_day, delta, op[i], "rebalance")
            shares = target_shares
            rebalances.append({"date": str(day.date()), "signal_date": str(signal_day.date()),
                               "holdings": selected, "target_exposure": float(weights.sum()),
                               "prediction_coverage": coverage})
        # Explicit assumption: cash held after the open earns one session's rate.
        cash *= 1 + config.daily_risk_free
        position_value = float((shares * cl[i]).sum())
        equity = cash + position_value
        exposure = position_value / equity
        if number == len(indices) - 1:
            liquidation_cost = position_value * config.cost_rate
            log_trades(day, signal_day, -shares, cl[i], "final liquidation")
            traded += position_value
            daily_cost += liquidation_cost
            equity -= liquidation_cost
            cash, shares = equity, np.zeros(len(tickers))
        if not np.isfinite(equity) or equity <= 0:
            raise ArithmeticError("Invalid portfolio equity")
        daily.append({"date": str(day.date()), "equity": float(equity),
                      "return": float(equity / previous_equity - 1), "cash": float(cash),
                      "exposure": float(exposure), "cost": float(daily_cost),
                      "traded_notional": float(traded)})
        previous_equity = equity
    return {"daily": pd.DataFrame(daily), "trades": trades, "rebalances": rebalances}


def performance_metrics(daily, config, benchmark_returns=None):
    r = daily["return"].to_numpy(float)
    if len(r) < 2 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Need at least two finite portfolio returns greater than -100%")
    n, periods = len(r), config.periods_per_year
    excess = r - config.daily_risk_free
    wealth = np.r_[1.0, np.cumprod(1 + r)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    cagr = wealth[-1] ** (periods / n) - 1
    vol = np.std(r, ddof=1) * np.sqrt(periods)
    excess_vol = np.std(excess, ddof=1) * np.sqrt(periods)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * np.sqrt(periods)
    max_dd = float(drawdown.min())
    duration, longest = 0, 0
    for value in drawdown[1:]:
        duration = duration + 1 if value < -1e-12 else 0
        longest = max(longest, duration)
    q = np.quantile(r, .05)
    result = {
        "total_return": float(wealth[-1] - 1), "cagr": float(cagr),
        "annual_volatility": float(vol),
        "sharpe": float(excess.mean() * periods / excess_vol) if excess_vol > 1e-12 else None,
        "sortino": float(excess.mean() * periods / downside) if downside > 1e-12 else None,
        "calmar": float(cagr / abs(max_dd)) if max_dd < -1e-12 else None,
        "max_drawdown": max_dd, "max_drawdown_duration_sessions": longest,
        "daily_win_rate": float((r > 0).mean()), "var_95": float(-q),
        "cvar_95": float(-r[r <= q].mean()), "sessions": n,
        "final_equity": float(daily.equity.iloc[-1]), "total_cost": float(daily.cost.sum()),
        "average_exposure": float(daily.exposure.mean()),
        "annual_one_way_turnover": float((daily.traded_notional / daily.equity).mean() * periods),
    }
    if benchmark_returns is not None:
        b = np.asarray(benchmark_returns, dtype=float)
        if len(b) != n:
            raise ValueError("Benchmark dates must exactly match strategy dates")
        active, market = r - b, b - config.daily_risk_free
        te = active.std(ddof=1) * np.sqrt(periods)
        beta = np.cov(excess, market, ddof=1)[0, 1] / np.var(market, ddof=1) if np.var(market) > 1e-15 else None
        result.update({"excess_total_return": float(wealth[-1] - np.prod(1 + b)),
                       "tracking_error": float(te),
                       "information_ratio": float(active.mean() * periods / te) if te > 1e-12 else None,
                       "beta": float(beta) if beta is not None else None,
                       "annual_alpha_descriptive": float((excess.mean() - beta * market.mean()) * periods) if beta is not None else None})
    return result


def block_bootstrap_intervals(returns, config, samples=1000, block_size=10):
    """Circular blocks retain local dependence; intervals are descriptive, not DSR."""
    r = np.asarray(returns, dtype=float)
    if samples < 1 or block_size < 1 or len(r) < 2:
        raise ValueError("Positive resamples/block size and at least two returns required")
    rng = np.random.default_rng(config.seed)
    starts = rng.integers(0, len(r), size=(samples, int(np.ceil(len(r) / block_size))))
    indices = ((starts[:, :, None] + np.arange(block_size)) % len(r)).reshape(samples, -1)[:, :len(r)]
    draws = r[indices]
    excess = draws - config.daily_risk_free
    std = excess.std(axis=1, ddof=1)
    sharpes = np.divide(excess.mean(axis=1) * np.sqrt(config.periods_per_year), std,
                        out=np.full(samples, np.nan), where=std > 1e-12)
    total = np.prod(1 + draws, axis=1) - 1
    finite = sharpes[np.isfinite(sharpes)]
    return {"method": "circular block bootstrap", "samples": samples, "block_sessions": block_size,
            "confidence_level": .95,
            "sharpe": [float(x) for x in np.quantile(finite, [.025, .975])] if len(finite) else None,
            "total_return": [float(x) for x in np.quantile(total, [.025, .975])]}
