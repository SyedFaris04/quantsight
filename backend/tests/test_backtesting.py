"""Small, hand-computable portfolios protect research validity and accounting."""
import unittest

import numpy as np
import pandas as pd

from prediction_contract import forward_direction
from backtesting.engine import (BacktestConfig, simulate, performance_metrics,
                               block_bootstrap_intervals, prepare_predictions)
from backtesting.report import classification_metrics


DATES = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10",
                        "2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17"])


def fixtures():
    prices = pd.DataFrame([{"date": d, "ticker": t, "Open": 10., "Close": 10.}
                           for d in DATES for t in ["A", "B", "SPY"]])
    predictions = pd.DataFrame([{"date": d, "ticker": t, "predicted_signal": 1,
                                 "confidence": 80. if t == "A" else 60.}
                                for d in DATES for t in ["A", "B", "SPY"]])
    return prices, predictions


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.prices, self.predictions = fixtures()
        self.config = BacktestConfig(initial_capital=1000, top_n=1, max_position_weight=1,
                                     commission_bps=0, slippage_bps=0)

    def run_model(self, **kwargs):
        return simulate(kwargs.pop("prices", self.prices), kwargs.pop("predictions", self.predictions),
                        kwargs.pop("config", self.config), DATES[1], DATES[-1], **kwargs)

    def test_five_session_label_and_unknown_tail(self):
        close = pd.Series([10, 20, 30, 40, 50, 11, 19, 30], dtype=float)
        ret, label = forward_direction(close)
        np.testing.assert_allclose(ret.iloc[:3], [.1, -.05, 0])
        self.assertEqual(label.iloc[:3].tolist(), [1, 0, 0])
        self.assertTrue(label.iloc[-5:].isna().all())

    def test_no_same_day_or_future_signal_used(self):
        original = self.run_model()
        changed = self.predictions.copy()
        changed.loc[(changed.date >= DATES[1]) & (changed.ticker == "B"), "confidence"] = 99
        updated = self.run_model(predictions=changed)
        self.assertEqual(original["rebalances"][0]["holdings"], ["A"])
        self.assertEqual(updated["rebalances"][0]["holdings"], ["A"])
        self.assertEqual(updated["rebalances"][1]["holdings"], ["B"])
        for rebalance in updated["rebalances"]:
            self.assertLess(rebalance["signal_date"], rebalance["date"])

    def test_entry_open_and_overnight_gap(self):
        p = self.prices.copy()
        p.loc[(p.ticker == "A") & (p.date == DATES[1]), "Close"] = 11
        p.loc[(p.ticker == "A") & (p.date >= DATES[2]), ["Open", "Close"]] = 20
        daily = self.run_model(prices=p)["daily"]
        self.assertAlmostEqual(daily.equity.iloc[0], 1100)
        self.assertAlmostEqual(daily.equity.iloc[1], 2000)
        self.assertAlmostEqual(daily.equity.iloc[-1], 2000)

    def test_per_side_costs_and_no_borrowing(self):
        cfg = BacktestConfig(initial_capital=1000, top_n=1, max_position_weight=1,
                             commission_bps=10, slippage_bps=5)
        run = self.run_model(config=cfg)
        expected = 1000 / 1.0015 * .9985
        self.assertAlmostEqual(run["daily"].equity.iloc[-1], expected, places=7)
        self.assertAlmostEqual(run["daily"].cost.sum(), 1000 - expected, places=7)
        self.assertTrue(run["daily"].cash.ge(-1e-8).all())
        self.assertEqual(len(run["trades"]), 2)

    def test_missing_predictions_keep_cash_not_stale_signals(self):
        pred = self.predictions[self.predictions.date != DATES[0]]
        run = self.run_model(predictions=pred)
        self.assertEqual(run["rebalances"][0]["holdings"], [])
        self.assertAlmostEqual(run["daily"].equity.iloc[0], 1000)

    def test_unused_slots_stay_cash_and_position_cap(self):
        pred = self.predictions.copy()
        pred.loc[pred.ticker == "B", "predicted_signal"] = 0
        cfg = BacktestConfig(initial_capital=1000, top_n=5, commission_bps=0, slippage_bps=0)
        run = self.run_model(predictions=pred, config=cfg)
        self.assertAlmostEqual(run["daily"].cash.iloc[0], 800)
        self.assertAlmostEqual(run["daily"].exposure.iloc[0], .2)

    def test_holiday_week_uses_last_observed_session(self):
        second = self.run_model()["rebalances"][1]
        self.assertEqual(second["date"], "2024-01-16")
        self.assertEqual(second["signal_date"], "2024-01-12")

    def test_prices_and_duplicate_rows_fail_closed(self):
        incomplete = self.prices.drop(self.prices[(self.prices.date == DATES[3]) & (self.prices.ticker == "A")].index)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.run_model(prices=incomplete)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_model(predictions=pd.concat([self.predictions, self.predictions.iloc[:1]]))

    def test_no_trades_means_cash_path(self):
        pred = self.predictions.assign(predicted_signal=0)
        run = self.run_model(predictions=pred)
        self.assertEqual(run["trades"], [])
        self.assertTrue(run["daily"].equity.eq(1000).all())
        metrics = performance_metrics(run["daily"], self.config)
        self.assertIsNone(metrics["sharpe"])
        self.assertIsNone(metrics["sortino"])

    def test_sharpe_sortino_initial_drawdown_hand_calculation(self):
        r = np.array([-.10, .05, -.02, .04])
        daily = pd.DataFrame({"return": r, "equity": 1000 * np.cumprod(1+r),
                              "cost": 0., "exposure": 1., "traded_notional": 0.})
        metrics = performance_metrics(daily, self.config)
        self.assertAlmostEqual(metrics["sharpe"], np.sqrt(252)*r.mean()/r.std(ddof=1))
        self.assertAlmostEqual(metrics["sortino"], np.sqrt(252)*r.mean()/np.sqrt(np.mean(np.minimum(r, 0)**2)))
        self.assertAlmostEqual(metrics["max_drawdown"], -.10)
        self.assertEqual(metrics["max_drawdown_duration_sessions"], 4)

    def test_random_null_matches_slots_and_is_reproducible(self):
        model = self.run_model()
        first = self.run_model(mode="random", random_seed=7)
        second = self.run_model(mode="random", random_seed=7)
        self.assertEqual(first["rebalances"], second["rebalances"])
        for a, b in zip(model["rebalances"], first["rebalances"]):
            self.assertEqual(a["target_exposure"], b["target_exposure"])
            self.assertEqual(len(a["holdings"]), len(b["holdings"]))
            self.assertNotIn("SPY", b["holdings"])

    def test_baseline_and_calibration_sample_counts(self):
        pred = prepare_predictions(self.predictions)
        labels = self.predictions[["ticker", "date"]].assign(signal=1)
        metrics = classification_metrics(pred, labels)
        self.assertEqual(metrics["always_up_accuracy"], 1)
        self.assertEqual(sum(x["count"] for x in metrics["calibration_bins"]), len(pred))

    def test_bootstrap_repeatable_and_finite(self):
        r = [.01, -.02, .03, .01, -.01, .005]
        a = block_bootstrap_intervals(r, self.config, samples=100, block_size=2)
        self.assertEqual(a, block_bootstrap_intervals(r, self.config, samples=100, block_size=2))
        self.assertLessEqual(a["sharpe"][0], a["sharpe"][1])


if __name__ == "__main__":
    unittest.main()
