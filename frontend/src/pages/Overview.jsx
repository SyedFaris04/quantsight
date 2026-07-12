/**
 * frontend/src/pages/Overview.jsx
 * Market Overview — ticker table with 4-model signals + accuracy badges
 */

import { useState, useMemo, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { companyName } from "../data/companyNames";

// ── Model column definitions ───────────────────────────────────────────────────
const MODEL_COLS = [
  { key: "xgb_finance",    short: "XGB",  sub: "Finance",    color: "text-blue-400"   },
  { key: "xgb_sentiment",  short: "XGB",  sub: "+Sentiment", color: "text-indigo-400" },
  { key: "lstm_finance",   short: "LSTM", sub: "Finance",    color: "text-amber-400"  },
  { key: "lstm_sentiment", short: "LSTM", sub: "+Sentiment", color: "text-purple-400" },
];

// Maps model_key → metrics key in model_metrics.json
const METRICS_KEY_MAP = {
  xgb_finance    : "XGBoost_Finance Only",
  xgb_sentiment  : "XGBoost_Finance + Sentiment",
  lstm_finance   : "LSTM+Transformer_Finance Only",
  lstm_sentiment : "LSTM+Transformer_Finance + Sentiment",
};

// ── Sub-components ─────────────────────────────────────────────────────────────

function SignalBadge({ signal, confidence }) {
  if (!signal || signal === "N/A") return <span className="text-gray-600 text-xs">—</span>;

  if (signal === "HOLD") {
    return (
      <div className="flex flex-col items-center gap-0.5">
        <span className="badge-hold">◆ HOLD</span>
        {confidence != null && (
          <span className="text-xs text-gray-500">{confidence}%</span>
        )}
      </div>
    );
  }

  const isBuy = signal === "BUY";
  return (
    <div className="flex flex-col items-center gap-0.5">
      <span className={isBuy ? "badge-buy" : "badge-sell"}>
        {isBuy ? "▲" : "▼"} {signal}
      </span>
      {confidence != null && (
        <span className="text-xs text-gray-500">{confidence}%</span>
      )}
    </div>
  );
}

function AgreementPill({ level }) {
  const styles = {
    Strong   : "bg-green-900/30 text-green-400 border-green-800",
    Moderate : "bg-amber-900/30 text-amber-400 border-amber-800",
    Mixed    : "bg-gray-800 text-gray-400 border-gray-700",
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${styles[level] || styles.Mixed}`}>
      {level}
    </span>
  );
}

function RiskBadge({ level }) {
  if (!level) return <span className="text-gray-600 text-xs">—</span>;
  const styles = {
    Low    : "bg-green-900/30 text-green-400 border-green-800",
    Medium : "bg-amber-900/30 text-amber-400 border-amber-800",
    High   : "bg-red-900/30 text-red-400 border-red-800",
  };
  const icons = { Low: "●", Medium: "●●", High: "●●●" };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border font-medium inline-flex items-center gap-1 ${styles[level] || styles.Medium}`}>
      <span className="tracking-tighter">{icons[level] || "●●"}</span> {level}
    </span>
  );
}

function MetricCard({ label, value, sub, accent }) {
  const accentMap = {
    indigo : "border-indigo-600/40 bg-indigo-600/5",
    green  : "border-green-600/40 bg-green-600/5",
    amber  : "border-amber-600/40 bg-amber-600/5",
    blue   : "border-blue-600/40 bg-blue-600/5",
  };
  return (
    <div className={`card border ${accentMap[accent] || "border-gray-800"} flex flex-col gap-1`}>
      <span className="text-xs text-gray-500 font-medium uppercase tracking-wider">{label}</span>
      <span className="text-2xl font-semibold text-white">{value}</span>
      {sub && <span className="text-xs text-gray-500">{sub}</span>}
    </div>
  );
}

function LoadingCard() {
  return <div className="card h-40 animate-pulse" />;
}

// ── Ticker card — one stock, overall signal + all 4 model variants ────────────
function TickerCard({ row, onClick }) {
  return (
    <div
      onClick={onClick}
      className="card cursor-pointer hover:border-gray-600 transition-colors flex flex-col gap-3"
    >
      {/* Ticker + overall signal */}
      <div className="flex items-start justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="font-mono font-bold text-white text-base">{row.ticker}</span>
            {row.is_live && (
              <span className="text-xs px-1.5 py-0.5 rounded-full bg-green-900/40 text-green-400 border border-green-800 font-medium">
                ● live
              </span>
            )}
          </div>
          {companyName(row.ticker) && (
            <div className="text-xs text-gray-500 truncate">{companyName(row.ticker)}</div>
          )}
        </div>
        <SignalBadge signal={row.overall_signal} confidence={row.overall_confidence} />
      </div>

      {/* 4-model mini breakdown */}
      <div className="grid grid-cols-2 gap-2">
        {MODEL_COLS.map(m => {
          const s = row.signals?.[m.key];
          return (
            <div key={m.key} className="bg-gray-800/50 rounded-lg px-2 py-1.5 text-center">
              <div className={`text-xs font-semibold ${m.color}`}>
                {m.short} <span className="text-gray-600 font-normal">{m.sub}</span>
              </div>
              <SignalBadge signal={s?.signal_label} confidence={s?.confidence} />
            </div>
          );
        })}
      </div>

      {/* Agreement + Risk */}
      <div className="flex items-center justify-between pt-1 border-t border-gray-800">
        <AgreementPill level={row.agreement_level} />
        <RiskBadge level={row.risk_level} />
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function Overview() {
  const navigate = useNavigate();
  const { data, loading }         = useApi("/overview");
  const { data: metricsRaw }      = useApi("/metrics");

  const [search,       setSearch]       = useState("");
  const [filterSignal, setFilterSignal] = useState("ALL");
  const [filterAgree,  setFilterAgree]  = useState("ALL");
  const [sortCol,      setSortCol]      = useState("ticker");
  const [sortDir,      setSortDir]      = useState("asc");
  const [page,         setPage]         = useState(1);
  const PAGE_SIZE = 12;
  const [liveMode,     setLiveMode]     = useState(false);
  const [liveData,     setLiveData]     = useState(null);
  const [liveLoading,  setLiveLoading]  = useState(false);
  const [liveError,    setLiveError]    = useState(null);
  const [liveGenerated, setLiveGenerated] = useState(null);

  const rows = data?.data || [];

  // ── Extract accuracy per model from metrics ─────────────────────────
  const modelAccuracy = useMemo(() => {
    if (!metricsRaw) return {};
    const out = {};
    MODEL_COLS.forEach(m => {
      const key = METRICS_KEY_MAP[m.key];
      if (metricsRaw[key]) out[m.key] = metricsRaw[key].accuracy;
    });
    return out;
  }, [metricsRaw]);

  // ── Fetch live signals ────────────────────────────────────────────────────
  async function fetchLiveSignals() {
    setLiveLoading(true);
    setLiveError(null);
    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL || "/api"}/live-overview`
      );
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const json = await res.json();
      setLiveData(json.data);
      setLiveGenerated(json.generated);
    } catch (e) {
      setLiveError("Could not fetch live signals. Check your internet connection.");
    } finally {
      setLiveLoading(false);
    }
  }

  function handleLiveToggle() {
    const next = !liveMode;
    setLiveMode(next);
    if (next && !liveData) fetchLiveSignals();
  }

  // ── Merge live signals into rows when live mode is on ─────────────────────
  const displayRows = useMemo(() => {
    if (!liveMode || !liveData) return rows;
    return rows.map(row => {
      const live = liveData[row.ticker];
      if (!live || live.source === "fallback" || !live.signal_label) return row;
      return {
        ...row,
        overall_signal    : live.signal_label,
        overall_confidence: live.confidence ?? row.overall_confidence,
        is_live           : true,
        live_date         : live.date,
      };
    });
  }, [rows, liveMode, liveData]);

  // ── Derived metrics ──────────────────────────────────────────────────
  const metrics = useMemo(() => {
    if (!rows.length) return null;
    const src = liveMode ? displayRows : rows;
    const buyCount    = src.filter(r => r.overall_signal === "BUY").length;
    const sellCount   = src.filter(r => r.overall_signal === "SELL").length;
    const holdCount   = src.filter(r => r.overall_signal === "HOLD").length;
    const strongCount = src.filter(r => r.agreement_level === "Strong").length;
    const avgConf     = src.reduce((s, r) => s + r.overall_confidence, 0) / src.length;
    return { total: src.length, buyCount, sellCount, holdCount, strongCount, avgConf: avgConf.toFixed(1) };
  }, [rows, displayRows, liveMode]);

  // ── Filter + sort ────────────────────────────────────────────────────
  const filtered = useMemo(() => {
    let out = [...displayRows];
    if (search.trim())          out = out.filter(r => r.ticker.toLowerCase().includes(search.toLowerCase()));
    if (filterSignal !== "ALL") out = out.filter(r => r.overall_signal === filterSignal);
    if (filterAgree  !== "ALL") out = out.filter(r => r.agreement_level === filterAgree);
    out.sort((a, b) => {
      const av = sortCol === "confidence" ? a.overall_confidence : sortCol === "signal" ? a.overall_signal : a.ticker;
      const bv = sortCol === "confidence" ? b.overall_confidence : sortCol === "signal" ? b.overall_signal : b.ticker;
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1  : -1;
      return 0;
    });
    return out;
  }, [displayRows, search, filterSignal, filterAgree, sortCol, sortDir]);

  // Reset to page 1 whenever the filtered set changes shape (new search/filter)
  useEffect(() => { setPage(1); }, [search, filterSignal, filterAgree]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pagedRows   = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  function toggleSort(col) {
    if (sortCol === col) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortCol(col); setSortDir("asc"); }
  }

  function SortIcon({ col }) {
    if (sortCol !== col) return <span className="text-gray-700 ml-1">↕</span>;
    return <span className="text-indigo-400 ml-1">{sortDir === "asc" ? "↑" : "↓"}</span>;
  }

  return (
    <div className="space-y-6">

      {/* ── Page header ── */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-white">Market</h1>
          <p className="text-sm text-gray-500 mt-1">
            {liveMode
              ? `Live signals from today's market data · ${liveGenerated ? `Updated ${liveGenerated}` : "Loading..."}`
              : `Scan all ${loading ? "..." : rows.length} stocks and find opportunities · cached signals from Dec 2024 dataset`
            }
          </p>
        </div>

        {/* Live Mode toggle */}
        <div className="flex flex-col items-end gap-1">
          <button
            onClick={handleLiveToggle}
            disabled={liveLoading}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                        border transition-all duration-200 ${
              liveMode
                ? "bg-green-900/30 border-green-700 text-green-400 hover:bg-green-900/50"
                : "bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-500 hover:text-gray-200"
            } ${liveLoading ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
          >
            {liveLoading ? (
              <>
                <span className="w-3 h-3 border-2 border-green-600 border-t-transparent rounded-full animate-spin" />
                Fetching live data...
              </>
            ) : liveMode ? (
              <>
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                Live Mode ON
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-gray-600" />
                Switch to Live Mode
              </>
            )}
          </button>
          {liveMode && (
            <button
              onClick={fetchLiveSignals}
              disabled={liveLoading}
              className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
            >
              ↻ Refresh signals
            </button>
          )}
        </div>
      </div>

      {/* Live mode disclaimer */}
      {liveMode && !liveLoading && !liveError && (
        <div className="bg-green-900/10 border border-green-800/50 rounded-lg px-4 py-2.5 text-xs text-green-600">
          ● Live Mode — signals generated from today's Yahoo Finance data using the trained XGBoost model.
          The model was trained on 2015–2023 patterns. Treat as educational signals, not financial advice.
        </div>
      )}

      {/* Live mode error */}
      {liveError && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg px-4 py-2.5 text-xs text-red-400">
          ⚠️ {liveError} — showing cached signals instead.
        </div>
      )}

      {/* Live loading state */}
      {liveLoading && (
        <div className="bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-sm text-gray-400 flex items-center gap-3">
          <span className="w-4 h-4 border-2 border-green-600 border-t-transparent rounded-full animate-spin flex-shrink-0" />
          <div>
            <p className="font-medium text-gray-300">Fetching live signals for all tickers...</p>
            <p className="text-xs text-gray-600 mt-0.5">This may take 20–40 seconds. Yahoo Finance is being queried for each stock.</p>
          </div>
        </div>
      )}

      {/* ── Metric cards ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Total Tickers"     value={loading ? "—" : metrics?.total ?? 0}          sub="tickers analysed"             accent="indigo" />
        <MetricCard label="BUY Signals"       value={loading ? "—" : metrics?.buyCount ?? 0}       sub={`${metrics?.sellCount ?? 0} SELL · ${metrics?.holdCount ?? 0} HOLD`} accent="green" />
        <MetricCard label="Strong Agreement"  value={loading ? "—" : metrics?.strongCount ?? 0}    sub="all 4 models agree"           accent="amber" />
        <MetricCard label="Avg Confidence"    value={loading ? "—" : `${metrics?.avgConf ?? 0}%`}  sub="across all models"            accent="blue"  />
      </div>

      {/* ── Filters ── */}
      <div className="flex flex-wrap gap-3 items-center">
        <input
          className="input w-48"
          placeholder="Search ticker..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <div className="flex rounded-lg overflow-hidden border border-gray-700 text-sm">
          {["ALL", "BUY", "SELL", "HOLD"].map(v => (
            <button key={v} onClick={() => setFilterSignal(v)}
              className={`px-3 py-1.5 font-medium transition-colors ${
                filterSignal === v
                  ? v === "BUY" ? "bg-green-700 text-white" : v === "SELL" ? "bg-red-700 text-white" : v === "HOLD" ? "bg-amber-700 text-white" : "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              }`}
            >{v}</button>
          ))}
        </div>
        <div className="flex rounded-lg overflow-hidden border border-gray-700 text-sm">
          {["ALL", "Strong", "Moderate", "Mixed"].map(v => (
            <button key={v} onClick={() => setFilterAgree(v)}
              className={`px-3 py-1.5 font-medium transition-colors ${
                filterAgree === v ? "bg-indigo-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              }`}
            >{v}</button>
          ))}
        </div>
        <span className="ml-auto text-xs text-gray-600">
          {filtered.length} of {rows.length} tickers
        </span>
      </div>

      {/* ── Sort control (card grid replaces the old table — sort still applies) ── */}
      <div className="flex items-center gap-2 text-xs text-gray-500">
        <span>Sort:</span>
        {[
          { key: "ticker",     label: "Ticker" },
          { key: "signal",     label: "Signal" },
          { key: "confidence", label: "Confidence" },
        ].map(c => (
          <button
            key={c.key}
            onClick={() => toggleSort(c.key)}
            className={`px-2 py-1 rounded-md border transition-colors ${
              sortCol === c.key
                ? "border-indigo-700 bg-indigo-900/30 text-indigo-300"
                : "border-gray-800 hover:border-gray-700 text-gray-500"
            }`}
          >
            {c.label} <SortIcon col={c.key} />
          </button>
        ))}
      </div>

      {/* ── Card grid ── */}
      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => <LoadingCard key={i} />)}
        </div>
      ) : filtered.length === 0 ? (
        <div className="card py-16 text-center text-gray-600">
          No tickers match your filters
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {pagedRows.map(row => (
              <TickerCard
                key={row.ticker}
                row={row}
                onClick={() => navigate(`/detail/${row.ticker}`)}
              />
            ))}
          </div>

          {/* ── Pagination ── */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between text-xs text-gray-500 pt-2">
              <span>
                Showing {(page - 1) * PAGE_SIZE + 1}-{Math.min(page * PAGE_SIZE, filtered.length)} of {filtered.length} stocks
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="px-2.5 py-1 rounded-md border border-gray-800 hover:border-gray-600 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  ‹
                </button>
                {Array.from({ length: totalPages }).map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setPage(i + 1)}
                    className={`w-7 h-7 rounded-md border transition-colors ${
                      page === i + 1
                        ? "border-indigo-600 bg-indigo-600 text-white"
                        : "border-gray-800 hover:border-gray-600 text-gray-400"
                    }`}
                  >
                    {i + 1}
                  </button>
                ))}
                <button
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  className="px-2.5 py-1 rounded-md border border-gray-800 hover:border-gray-600 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  ›
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── Legend + accuracy key ── */}
      <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-gray-500">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-blue-400">XGB Finance</span>
          <span>— XGBoost, price/technical only</span>
          {modelAccuracy.xgb_finance && <span className="text-gray-600">({modelAccuracy.xgb_finance.toFixed(1)}% acc)</span>}
        </div>
        <div className="flex items-center gap-2">
          <span className="font-semibold text-indigo-400">XGB +Sentiment</span>
          <span>— XGBoost + WSB + GDELT</span>
          {modelAccuracy.xgb_sentiment && <span className="text-gray-600">({modelAccuracy.xgb_sentiment.toFixed(1)}% acc)</span>}
        </div>
        <div className="flex items-center gap-2">
          <span className="font-semibold text-amber-400">LSTM Finance</span>
          <span>— LSTM+Attention, price only</span>
          {modelAccuracy.lstm_finance && <span className="text-gray-600">({modelAccuracy.lstm_finance.toFixed(1)}% acc)</span>}
        </div>
        <div className="flex items-center gap-2">
          <span className="font-semibold text-purple-400">LSTM +Sentiment</span>
          <span>— LSTM+Attention + WSB + GDELT</span>
          {modelAccuracy.lstm_sentiment && <span className="text-gray-600">({modelAccuracy.lstm_sentiment.toFixed(1)}% acc)</span>}
        </div>
        <div className="flex items-center gap-2 w-full text-gray-600">
          Accuracy % shown in column headers — click any row to see full AI explanation
        </div>
      </div>

    </div>
  );
}
