/**
 * frontend/src/pages/Portfolio.jsx
 * ─────────────────────────────────────────────────────────────────
 * Portfolio Tracker — enter your holdings, get model-based
 * recommendations (HOLD / SELL / BUY MORE) and see what to buy next.
 *
 * All data stays in the browser (localStorage) — no account needed.
 * Prices come from your existing /stock/{ticker} endpoint.
 * Signals come from /explain/{ticker}.
 * ─────────────────────────────────────────────────────────────────
 */

import { useState, useEffect, useMemo, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import api from "../hooks/useApi";
import { companyName } from "../data/companyNames";

// ── Storage helpers ────────────────────────────────────────────────────────────
const STORAGE_KEY = "nuroquant_portfolio";

function loadPortfolio() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function savePortfolio(holdings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(holdings));
}

// ── Recommendation logic ───────────────────────────────────────────────────────
function getRecommendation(signal, confidence, plPct) {
  if (!signal) return { label: "LOADING", color: "text-gray-400", bg: "bg-gray-800 border-gray-700", icon: "⏳" };

  if (signal === "SELL") {
    return { label: "SELL", color: "text-red-400", bg: "bg-red-900/20 border-red-800", icon: "▼" };
  }

  if (signal === "HOLD") {
    // Model itself is undecided (2-2 split) — keep position, no strong signal either way
    return { label: "HOLD", color: "text-amber-400", bg: "bg-amber-900/20 border-amber-800", icon: "◆" };
  }

  // BUY signal
  if (confidence >= 65 && plPct >= 0) {
    return { label: "BUY MORE", color: "text-green-400", bg: "bg-green-900/20 border-green-800", icon: "▲▲" };
  }
  if (confidence >= 65 && plPct < -10) {
    return { label: "HOLD", color: "text-amber-400", bg: "bg-amber-900/20 border-amber-800", icon: "◆" };
  }
  return { label: "HOLD", color: "text-amber-400", bg: "bg-amber-900/20 border-amber-800", icon: "◆" };
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function SectionTitle({ children, sub }) {
  return (
    <div className="mb-4">
      <h2 className="text-base font-semibold text-white">{children}</h2>
      {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
    </div>
  );
}

// Single holding row — fetches its own real-time price + signal
function HoldingRow({ holding, onRemove, onStats }) {
  const { ticker, shares, buyPrice } = holding;
  const navigate = useNavigate();

  const { data: priceData }  = useApi(`/realtime/${ticker}`,  [ticker]);
  const { data: explanation } = useApi(`/explain/${ticker}`,  [ticker]);

  const currentPrice = priceData?.price ?? null;
  const isLive       = priceData?.source === "live";
  const signal       = explanation?.overall_signal    ?? null;
  const confidence   = explanation?.overall_confidence ?? 50;

  const plAmt  = currentPrice != null ? (currentPrice - buyPrice) * shares : null;
  const plPct  = currentPrice != null ? ((currentPrice - buyPrice) / buyPrice) * 100 : null;
  const value  = currentPrice != null ? currentPrice * shares : null;

  const rec = getRecommendation(signal, confidence, plPct);

  // Report this row's real numbers up to the parent so it can show
  // portfolio-wide totals (Total Value / Total Return / Today's P&L)
  useEffect(() => {
    if (currentPrice == null) return;
    const todaysPlAmt = priceData?.change_pct != null
      ? value * (priceData.change_pct / 100)
      : 0;
    onStats(ticker, { value, costBasis: buyPrice * shares, plAmt, todaysPlAmt });
  }, [ticker, currentPrice, value, plAmt, priceData?.change_pct, buyPrice, shares, onStats]);

  return (
    <tr className="hover:bg-gray-800/30 transition-colors">
      {/* Ticker */}
      <td className="py-3 px-4 border-b border-gray-800/50">
        <button
          onClick={() => navigate(`/detail/${ticker}`)}
          className="text-left hover:text-indigo-400 transition-colors"
        >
          <div className="font-bold text-white font-mono">{ticker}</div>
          {companyName(ticker) && (
            <div className="text-xs text-gray-500 font-normal">{companyName(ticker)}</div>
          )}
        </button>
      </td>

      {/* Shares */}
      <td className="py-3 px-4 border-b border-gray-800/50 text-gray-300">
        {shares}
      </td>

      {/* Buy price */}
      <td className="py-3 px-4 border-b border-gray-800/50 text-gray-400 font-mono">
        ${buyPrice.toFixed(2)}
      </td>

      {/* Current price */}
      <td className="py-3 px-4 border-b border-gray-800/50 font-mono">
        {currentPrice != null ? (
          <div className="flex flex-col gap-0.5">
            <div className="flex items-center gap-1.5">
              <span className="text-white">${currentPrice.toFixed(2)}</span>
              {isLive
                ? <span className="text-xs px-1.5 py-0.5 rounded-full bg-green-900/40 text-green-400 border border-green-800">● live</span>
                : <span className="text-xs px-1.5 py-0.5 rounded-full bg-gray-800 text-gray-500 border border-gray-700">hist</span>
              }
            </div>
            {priceData?.change_pct != null && (
              <span className={`text-xs ${priceData.change_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                {priceData.change_pct >= 0 ? "+" : ""}{priceData.change_pct}% today
              </span>
            )}
          </div>
        ) : (
          <span className="text-gray-600">Loading…</span>
        )}
      </td>

      {/* Current value */}
      <td className="py-3 px-4 border-b border-gray-800/50 font-mono">
        {value != null
          ? <span className="text-gray-300">${value.toFixed(2)}</span>
          : <span className="text-gray-600">—</span>
        }
      </td>

      {/* P&L */}
      <td className="py-3 px-4 border-b border-gray-800/50">
        {plAmt != null ? (
          <div>
            <span className={`font-mono font-semibold ${plAmt >= 0 ? "text-green-400" : "text-red-400"}`}>
              {plAmt >= 0 ? "+" : ""}${plAmt.toFixed(2)}
            </span>
            <span className={`ml-2 text-xs ${plPct >= 0 ? "text-green-500" : "text-red-500"}`}>
              ({plPct >= 0 ? "+" : ""}{plPct.toFixed(2)}%)
            </span>
          </div>
        ) : <span className="text-gray-600">—</span>}
      </td>

      {/* Model signal */}
      <td className="py-3 px-4 border-b border-gray-800/50">
        {signal ? (
          <span className={`text-xs font-bold ${
            signal === "BUY" ? "text-green-400" : signal === "SELL" ? "text-red-400" : "text-amber-400"
          }`}>
            {signal === "BUY" ? "▲" : signal === "SELL" ? "▼" : "◆"} {signal}
            <span className="text-gray-500 font-normal ml-1">({confidence}%)</span>
          </span>
        ) : <span className="text-gray-600 text-xs">Loading…</span>}
      </td>

      {/* Recommendation */}
      <td className="py-3 px-4 border-b border-gray-800/50">
        <span className={`text-xs font-bold px-2 py-1 rounded-lg border ${rec.bg} ${rec.color}`}>
          {rec.icon} {rec.label}
        </span>
      </td>

      {/* Remove */}
      <td className="py-3 px-4 border-b border-gray-800/50">
        <button
          onClick={() => onRemove(ticker)}
          className="text-gray-700 hover:text-red-500 transition-colors text-sm"
        >
          ✕
        </button>
      </td>
    </tr>
  );
}

// What to buy next — top BUY signals not in portfolio
function WhatToBuyNext({ portfolioTickers }) {
  const { data: overviewData, loading } = useApi("/overview");

  const suggestions = useMemo(() => {
    if (!overviewData?.data) return [];
    return overviewData.data
      .filter(row =>
        row.overall_signal === "BUY" &&
        !portfolioTickers.includes(row.ticker) &&
        row.agreement_level === "Strong"
      )
      .sort((a, b) => b.overall_confidence - a.overall_confidence)
      .slice(0, 5);
  }, [overviewData, portfolioTickers]);

  const navigate = useNavigate();

  return (
    <div className="card">
      <SectionTitle sub="Top BUY signals from tickers not in your portfolio — all 4 models agree">
        What to Consider Buying Next
      </SectionTitle>

      <div className="text-xs text-amber-500 bg-amber-900/20 border border-amber-800 rounded-lg px-3 py-2 mb-4">
        ⚠️ These are model signals only — not financial advice. Always do your own research before investing.
      </div>

      {loading ? (
        <div className="space-y-2">
          {[1,2,3].map(i => <div key={i} className="h-12 bg-gray-800 rounded-lg animate-pulse" />)}
        </div>
      ) : suggestions.length === 0 ? (
        <p className="text-gray-600 text-sm">
          No strong BUY signals found for tickers outside your portfolio right now.
        </p>
      ) : (
        <div className="space-y-2">
          {suggestions.map((row, i) => (
            <div
              key={row.ticker}
              className="flex items-center justify-between bg-gray-800/50 rounded-xl px-4 py-3
                         border border-gray-700 hover:border-green-800 transition-colors cursor-pointer"
              onClick={() => navigate(`/detail/${row.ticker}`)}
            >
              <div className="flex items-center gap-3">
                <span className="text-xs text-gray-500 w-4">#{i + 1}</span>
                <div>
                  <span className="font-bold text-white font-mono">{row.ticker}</span>
                  {companyName(row.ticker) && (
                    <div className="text-xs text-gray-500">{companyName(row.ticker)}</div>
                  )}
                </div>
                <span className="text-xs px-2 py-0.5 rounded-full bg-green-900/30 text-green-400 border border-green-800">
                  Strong Agreement
                </span>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-green-400 font-semibold text-sm">▲ BUY</div>
                  <div className="text-xs text-gray-500">{row.overall_confidence}% confidence</div>
                </div>
                <span className="text-gray-600 text-xs">View →</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Portfolio summary cards
function SummaryCard({ label, value, sub, color }) {
  return (
    <div className={`card border ${color}`}>
      <div className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-1">{label}</div>
      <div className="text-2xl font-bold text-white">{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function Portfolio() {
  const [holdings,    setHoldings]    = useState(loadPortfolio);
  const [showForm,    setShowForm]    = useState(false);
  const [formTicker,  setFormTicker]  = useState("");
  const [formShares,  setFormShares]  = useState("");
  const [formPrice,   setFormPrice]   = useState("");
  const [formError,   setFormError]   = useState("");
  const [holdingStats, setHoldingStats] = useState({}); // ticker -> {value, costBasis, plAmt, todaysPlAmt}

  const { data: tickerData } = useApi("/tickers");
  const availableTickers = tickerData?.tickers || [];

  const updateStats = useCallback((ticker, stats) => {
    setHoldingStats(prev => ({ ...prev, [ticker]: stats }));
  }, []);

  // Portfolio-wide KPIs — real aggregates from each row's live-fetched price.
  // No "Win Rate" card here: that would need historical trade outcomes we
  // don't track, so it's left out rather than invented.
  const portfolioSummary = useMemo(() => {
    const rows = holdings.map(h => holdingStats[h.ticker]).filter(Boolean);
    if (rows.length === 0) return null;
    const totalValue     = rows.reduce((s, r) => s + r.value, 0);
    const totalCostBasis = rows.reduce((s, r) => s + r.costBasis, 0);
    const totalPlAmt     = rows.reduce((s, r) => s + r.plAmt, 0);
    const todaysPlAmt    = rows.reduce((s, r) => s + r.todaysPlAmt, 0);
    const totalReturnPct = totalCostBasis > 0 ? (totalPlAmt / totalCostBasis) * 100 : 0;
    return { totalValue, totalPlAmt, totalReturnPct, todaysPlAmt, pricedCount: rows.length };
  }, [holdings, holdingStats]);

  // Persist to localStorage whenever holdings change
  useEffect(() => {
    savePortfolio(holdings);
  }, [holdings]);

  function addHolding() {
    setFormError("");
    const ticker   = formTicker.trim().toUpperCase();
    const shares   = parseFloat(formShares);
    const buyPrice = parseFloat(formPrice);

    if (!ticker)           return setFormError("Please select a ticker.");
    if (!availableTickers.includes(ticker))
                           return setFormError(`${ticker} is not available in NuroQuant.`);
    if (isNaN(shares) || shares <= 0)
                           return setFormError("Shares must be a positive number.");
    if (isNaN(buyPrice) || buyPrice <= 0)
                           return setFormError("Buy price must be a positive number.");
    if (holdings.find(h => h.ticker === ticker))
                           return setFormError(`${ticker} is already in your portfolio.`);

    setHoldings(prev => [...prev, { ticker, shares, buyPrice }]);
    setFormTicker("");
    setFormShares("");
    setFormPrice("");
    setShowForm(false);
  }

  function removeHolding(ticker) {
    setHoldings(prev => prev.filter(h => h.ticker !== ticker));
  }

  const portfolioTickers = holdings.map(h => h.ticker);

  return (
    <div className="space-y-6">

      {/* ── Header ── */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-semibold text-white">My Portfolio</h1>
          <p className="text-sm text-gray-500 mt-1">
            Track your holdings and get model-based recommendations
          </p>
        </div>
        <button
          onClick={() => setShowForm(s => !s)}
          className="btn-primary flex items-center gap-2"
        >
          {showForm ? "✕ Cancel" : "+ Add Stock"}
        </button>
      </div>

      {/* ── Portfolio KPI row — real aggregates, only shown once at least one holding has priced ── */}
      {holdings.length > 0 && (
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
          <SummaryCard
            label="Total Value"
            value={portfolioSummary ? `$${portfolioSummary.totalValue.toFixed(2)}` : "Loading…"}
            sub={portfolioSummary ? `${portfolioSummary.pricedCount} of ${holdings.length} priced` : null}
            color="border-indigo-600/40 bg-indigo-600/5"
          />
          <SummaryCard
            label="Total Return"
            value={portfolioSummary ? `${portfolioSummary.totalReturnPct >= 0 ? "+" : ""}${portfolioSummary.totalReturnPct.toFixed(2)}%` : "Loading…"}
            sub={portfolioSummary ? `${portfolioSummary.totalPlAmt >= 0 ? "+" : ""}$${portfolioSummary.totalPlAmt.toFixed(2)} all-time` : null}
            color="border-green-600/40 bg-green-600/5"
          />
          <SummaryCard
            label="Today's P&L"
            value={portfolioSummary ? `${portfolioSummary.todaysPlAmt >= 0 ? "+" : ""}$${portfolioSummary.todaysPlAmt.toFixed(2)}` : "Loading…"}
            sub="from live-priced holdings only"
            color="border-amber-600/40 bg-amber-600/5"
          />
        </div>
      )}

      {/* ── Add stock form ── */}
      {showForm && (
        <div className="card border border-indigo-800/50 bg-indigo-900/10">
          <SectionTitle>Add a stock to your portfolio</SectionTitle>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-3">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Ticker</label>
              <select
                className="input w-full"
                value={formTicker}
                onChange={e => setFormTicker(e.target.value)}
              >
                <option value="">— select ticker —</option>
                {availableTickers
                  .filter(t => !portfolioTickers.includes(t))
                  .map(t => <option key={t} value={t}>{t}</option>)
                }
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Number of shares</label>
              <input
                className="input w-full"
                type="number"
                min="0.01"
                step="0.01"
                placeholder="e.g. 10"
                value={formShares}
                onChange={e => setFormShares(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Your buy price (USD)</label>
              <input
                className="input w-full"
                type="number"
                min="0.01"
                step="0.01"
                placeholder="e.g. 150.00"
                value={formPrice}
                onChange={e => setFormPrice(e.target.value)}
              />
            </div>
          </div>
          {formError && (
            <p className="text-red-400 text-xs mb-3">{formError}</p>
          )}
          <button onClick={addHolding} className="btn-primary">
            Add to Portfolio
          </button>
        </div>
      )}

      {/* ── Empty state ── */}
      {holdings.length === 0 && !showForm && (
        <div className="card text-center py-16">
          <div className="text-4xl mb-3">📊</div>
          <p className="text-gray-400 text-sm mb-1">Your portfolio is empty</p>
          <p className="text-gray-600 text-xs mb-4">
            Add your stock holdings to get personalised BUY / HOLD / SELL recommendations
          </p>
          <button onClick={() => setShowForm(true)} className="btn-primary">
            + Add your first stock
          </button>
        </div>
      )}

      {/* ── Holdings table ── */}
      {holdings.length > 0 && (
        <div className="card p-0 overflow-hidden">
          <div className="px-5 pt-5 pb-3">
            <SectionTitle sub="Click any ticker to see full AI analysis">
              Your Holdings
            </SectionTitle>
            <p className="text-xs text-amber-500 bg-amber-900/20 border border-amber-800 rounded-lg px-3 py-2">
              ⚠️ Current prices are fetched live from Yahoo Finance where available, shown with a ● live badge.
              If live data is unavailable, the last dataset price (Dec 2024) is used instead, shown as hist.
              Model signals are based on historical training data — not financial advice.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead className="bg-gray-900">
                <tr>
                  <th>Ticker</th>
                  <th>Shares</th>
                  <th>Buy Price</th>
                  <th>Current Price</th>
                  <th>Value</th>
                  <th>P&amp;L</th>
                  <th>Model Signal</th>
                  <th>Recommendation</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {holdings.map(h => (
                  <HoldingRow key={h.ticker} holding={h} onRemove={removeHolding} onStats={updateStats} />
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-5 py-3 border-t border-gray-800 text-xs text-gray-600">
            <span className="text-amber-400 font-medium">◆ HOLD</span> — model says BUY, keep your position &nbsp;·&nbsp;
            <span className="text-green-400 font-medium">▲▲ BUY MORE</span> — strong BUY signal + you're profitable &nbsp;·&nbsp;
            <span className="text-red-400 font-medium">▼ SELL</span> — model says SELL, consider exiting
          </div>
        </div>
      )}

      {/* ── What to buy next ── */}
      <WhatToBuyNext portfolioTickers={portfolioTickers} />

    </div>
  );
}
