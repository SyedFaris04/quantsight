/**
 * frontend/src/pages/Overview.jsx
 * Market Overview — ticker table with 4-model signals + accuracy badges
 */

import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
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

const SIGNAL_STYLE = { BUY: "text-green-400", SELL: "text-red-400", HOLD: "text-amber-300" };
const ARROW = { BUY: "▲", SELL: "▼", HOLD: "◆" };

// Text label always present; colour only reinforces it.
function Signal({ value }) {
  if (!value || value === "N/A") return <span className="text-gray-500">—</span>;
  return (
    <span className={`font-medium ${SIGNAL_STYLE[value] || ""}`}>
      <span aria-hidden="true" className="text-[10px] mr-1">{ARROW[value]}</span>{value}
    </span>
  );
}

const pct = (v) => (v == null ? "—" : `${Number(v).toFixed(1)}%`);

function SortHeader({ col, label, sortCol, sortDir, onSort, align = "left" }) {
  const active = sortCol === col;
  return (
    <th scope="col" aria-sort={active ? (sortDir === "asc" ? "ascending" : "descending") : "none"}
        className={align === "right" ? "!text-right" : ""}>
      <button onClick={() => onSort(col)} className={`uppercase hover:text-gray-200 ${active ? "text-gray-200" : ""}`}>
        {label}<span aria-hidden="true" className="ml-1">{active ? (sortDir === "asc" ? "↑" : "↓") : ""}</span>
      </button>
    </th>
  );
}

function Segmented({ label, options, value, onChange, disabled }) {
  return (
    <div role="group" aria-label={label} className="inline-flex rounded border border-gray-700 p-0.5 text-sm">
      {options.map(v => (
        <button key={v} disabled={disabled} aria-pressed={value === v} onClick={() => onChange(v)}
          className={`px-2.5 py-1 rounded-sm ${value === v ? "bg-gray-700 text-white" : "text-gray-400 hover:text-gray-200"}`}>
          {v === "ALL" ? "All" : v}
        </button>
      ))}
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function Overview() {
  const { data, loading, error }  = useApi("/overview");
  const { data: metricsRaw }      = useApi("/metrics");

  const [search,       setSearch]       = useState("");
  const [filterSignal, setFilterSignal] = useState("ALL");
  const [filterAgree,  setFilterAgree]  = useState("ALL");
  const [sortCol,      setSortCol]      = useState("ticker");
  const [sortDir,      setSortDir]      = useState("asc");
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
    setLiveData(null);
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
    setFilterAgree("ALL");
    if (next && !liveData) fetchLiveSignals();
  }

  // ── Merge live signals into rows when live mode is on ─────────────────────
  const displayRows = useMemo(() => {
    if (!liveMode) return rows;
    return rows.map(row => {
      const live = liveData?.[row.ticker];
      const available = live?.source === "live" && ["BUY", "SELL"].includes(live.signal_label);
      return {
        ticker: row.ticker,
        overall_signal: available ? live.signal_label : "N/A",
        overall_confidence: available ? live.direction_confidence : null,
        signals: {}, agreement_level: null, risk_level: null,
        live_mode: true, is_live: available,
        live_date: available ? live.date : null,
        target_date: available ? live.target_date : null,
        live_error: available ? null : live?.error || "Live signal unavailable",
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
    const validConfidence = src.filter(r => r.overall_confidence != null);
    const avgConf = validConfidence.length ? validConfidence.reduce((sum, r) => sum + r.overall_confidence, 0) / validConfidence.length : null;
    return { total: src.length, buyCount, sellCount, holdCount, strongCount, avgConf: avgConf == null ? "N/A" : avgConf.toFixed(1) };
  }, [rows, displayRows, liveMode]);

  // ── Filter + sort ────────────────────────────────────────────────────
  const filtered = useMemo(() => {
    let out = [...displayRows];
    if (search.trim())          out = out.filter(r => `${r.ticker} ${companyName(r.ticker)}`.toLowerCase().includes(search.trim().toLowerCase()));
    if (filterSignal !== "ALL") out = out.filter(r => r.overall_signal === filterSignal);
    if (!liveMode && filterAgree !== "ALL") out = out.filter(r => r.agreement_level === filterAgree);
    out.sort((a, b) => {
      const av = sortCol === "confidence" ? a.overall_confidence : sortCol === "signal" ? a.overall_signal : a.ticker;
      const bv = sortCol === "confidence" ? b.overall_confidence : sortCol === "signal" ? b.overall_signal : b.ticker;
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1  : -1;
      return 0;
    });
    return out;
  }, [displayRows, search, filterSignal, filterAgree, sortCol, sortDir, liveMode]);

  function toggleSort(col) {
    if (sortCol === col) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortCol(col); setSortDir("asc"); }
  }

  return (
    <div className="mx-auto max-w-7xl space-y-6">

      <header className="page-heading">
        <div>
          <h1>Market</h1>
          <p>
            {liveMode
              ? `One XGBoost finance model on the latest completed session${liveGenerated ? ` · updated ${liveGenerated}` : ""}.`
              : `All ${loading ? "" : rows.length} instruments with their saved December 2024 votes from four models. Open a ticker for the full analysis.`}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {liveMode && (
            <button onClick={fetchLiveSignals} disabled={liveLoading} className="text-sm text-indigo-300 hover:text-indigo-200">
              Refresh
            </button>
          )}
          <button onClick={handleLiveToggle} disabled={liveLoading} aria-pressed={liveMode} className="btn-secondary">
            {liveLoading ? "Fetching live data…" : liveMode ? "Show saved votes" : "Show live signals"}
          </button>
        </div>
      </header>

      {liveMode && !liveLoading && !liveError && (
        <p className="text-sm text-gray-400 border-l-2 border-gray-700 pl-3">
          Live mode shows one XGBoost finance model using the latest completed session, five trading sessions ahead.
          Direction probability is the calibrated probability of the shown direction. Agreement and risk do not apply.
        </p>
      )}
      {liveError && <p role="alert" className="text-sm text-red-300">{liveError} Live signals are unavailable.</p>}
      {liveLoading && (
        <p role="status" className="text-sm text-gray-400">
          Fetching live signals for every ticker. Yahoo Finance is queried per stock, so this can take 20–40 seconds.
        </p>
      )}

      <dl className="flex flex-wrap gap-x-10 gap-y-3 border-y border-gray-800 py-3 text-sm">
        {[
          ["Instruments", loading || error ? "—" : metrics?.total],
          [liveMode ? "UP forecasts" : "BUY / SELL / HOLD", loading || error || liveLoading || liveError ? "—"
            : liveMode ? metrics?.buyCount : `${metrics?.buyCount} / ${metrics?.sellCount} / ${metrics?.holdCount}`],
          ...(liveMode ? [] : [["All four models agree", loading || error ? "—" : metrics?.strongCount]]),
          [liveMode ? "Mean direction probability" : "Mean P(up)", !metrics || metrics.avgConf === "N/A" || liveLoading ? "—" : `${metrics.avgConf}%`],
        ].map(([k, v]) => (
          <div key={k}>
            <dt className="text-xs text-gray-400">{k}</dt>
            <dd className="mt-0.5 text-lg font-medium tabular-nums">{v}</dd>
          </div>
        ))}
        <p className="self-end text-xs text-gray-400 sm:ml-auto">Agreement and averages describe the models, not their accuracy.</p>
      </dl>

      <div className="flex flex-wrap gap-3 items-center">
        <label className="sr-only" htmlFor="market-search">Search ticker or company</label>
        <input id="market-search" type="search" className="input w-full sm:w-56" placeholder="Search ticker or company"
          value={search} onChange={e => setSearch(e.target.value)} />
        <Segmented label="Filter by signal" options={["ALL", "BUY", "SELL", "HOLD"]} value={filterSignal} onChange={setFilterSignal} />
        {!liveMode && <Segmented label="Filter by agreement" options={["ALL", "Strong", "Moderate", "Mixed"]} value={filterAgree} onChange={setFilterAgree} />}
        <span className="sm:ml-auto text-xs text-gray-400" aria-live="polite">{filtered.length} of {rows.length} instruments</span>
      </div>

      {loading ? (
        <p role="status" className="py-10 text-sm text-gray-400">Loading instruments…</p>
      ) : error ? (
        <p role="alert" className="py-10 text-sm text-red-300">Could not load instruments: {error}</p>
      ) : filtered.length === 0 ? (
        <p className="py-10 text-sm text-gray-400">No instruments match these filters.</p>
      ) : (
        <div className="overflow-x-auto border border-gray-800 rounded-md">
          <table className="data-table min-w-[56rem]">
            <caption className="sr-only">Saved model votes per instrument</caption>
            <thead className="bg-gray-900">
              <tr>
                <SortHeader col="ticker" label="Instrument" sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
                <SortHeader col="signal" label={liveMode ? "Direction" : "Consensus"} sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} />
                <SortHeader col="confidence" label={liveMode ? "Direction P" : "Mean P(up)"} sortCol={sortCol} sortDir={sortDir} onSort={toggleSort} align="right" />
                {liveMode ? (
                  <><th scope="col">Data close</th><th scope="col">Target close</th></>
                ) : (
                  <>
                    {MODEL_COLS.map(m => (
                      <th key={m.key} scope="col" className="!text-right">
                        {m.short} {m.sub === "Finance" ? "fin." : "+sent."}
                        {modelAccuracy[m.key] && <span className="block normal-case tracking-normal text-gray-500">{modelAccuracy[m.key].toFixed(1)}% acc.</span>}
                      </th>
                    ))}
                    <th scope="col">Agreement</th>
                    <th scope="col">Risk</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {filtered.map(row => (
                <tr key={row.ticker}>
                  <th scope="row" className="text-left">
                    <Link to={`/detail/${row.ticker}`} className="font-mono font-medium text-white hover:text-indigo-300">{row.ticker}</Link>
                    <span className="block text-xs font-normal text-gray-400">{companyName(row.ticker)}</span>
                  </th>
                  <td><Signal value={row.overall_signal} /></td>
                  <td className="text-right">{pct(row.overall_confidence)}</td>
                  {liveMode ? (
                    row.is_live
                      ? <><td>{row.live_date}</td><td>{row.target_date}</td></>
                      : <td colSpan={2} className="text-amber-300">{row.live_error}</td>
                  ) : (
                    <>
                      {MODEL_COLS.map(m => {
                        const s = row.signals?.[m.key];
                        return (
                          <td key={m.key} className="text-right whitespace-nowrap">
                            <Signal value={s?.signal_label} />
                            <span className="ml-2 text-gray-400">{pct(s?.confidence)}</span>
                          </td>
                        );
                      })}
                      <td>{row.agreement_level || "—"}</td>
                      <td className={row.risk_level === "High" ? "text-red-300" : row.risk_level === "Medium" ? "text-amber-300" : "text-gray-300"}>
                        {row.risk_level || "—"}
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!liveMode && (
        <p className="text-xs leading-relaxed text-gray-400 max-w-4xl">
          Columns show each model&apos;s vote and saved P(up): XGBoost finance (price and technical indicators),
          XGBoost + sentiment (adds WSB and GDELT text), and the two attention-based LSTM sequence models.
          Accuracy is each model&apos;s saved test-set accuracy. Agreement is not accuracy.
        </p>
      )}
    </div>
  );
}
