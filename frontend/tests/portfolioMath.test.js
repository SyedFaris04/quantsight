import test from "node:test";
import assert from "node:assert/strict";
import { dailyQuoteProfit } from "../src/data/portfolioMath.js";
test("daily P&L uses previous close as denominator", () => {
  assert.ok(Math.abs(dailyQuoteProfit(1100, 10, true) - 100) < 1e-9);
  assert.ok(Math.abs(dailyQuoteProfit(900, -10, true) + 100) < 1e-9);
  assert.equal(dailyQuoteProfit(1000, 0, true), 0);
});
test("old quotes and unavailable changes have undefined daily P&L", () => {
  assert.equal(dailyQuoteProfit(1000, 10, false), null);
  assert.equal(dailyQuoteProfit(1000, undefined, true), null);
  assert.equal(dailyQuoteProfit(0, -100, true), null);
});
