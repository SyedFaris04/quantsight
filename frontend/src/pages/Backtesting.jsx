import { useMemo, useState } from "react";
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine } from "recharts";
import api, { useApi } from "../hooks/useApi";

const pct = value => value == null ? "—" : `${(value * 100).toFixed(2)}%`;
const num = value => value == null ? "—" : value.toFixed(2);
const money = value => value == null ? "—" : new Intl.NumberFormat("en-US", {
  style: "currency", currency: "USD", maximumFractionDigits: 0,
}).format(value);
const chartTooltip = { backgroundColor: "#111827", border: "1px solid #374151", borderRadius: 12 };

function Metric({ label, value, note }) {
  return <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
    <p className="text-sm text-gray-400">{label}</p>
    <p className="mt-2 text-2xl font-semibold tabular-nums text-white">{value}</p>
    <p className="mt-2 text-xs leading-relaxed text-gray-400">{note}</p>
  </div>;
}

export default function Backtesting() {
  const { data, loading, error, refetch } = useApi("/backtest");
  const [selected, setSelected] = useState("ensemble");
  const [downloadError, setDownloadError] = useState("");
  const report = data?.report;
  const strategy = report?.strategies.find(s => s.key === selected);
  const spy = report?.strategies.find(s => s.key === "spy");
  const universe = report?.strategies.find(s => s.key === "universe");
  const chart = useMemo(() => strategy?.curve.map((row, i) => ({
    date: row.date, strategy: row.equity, spy: spy?.curve[i]?.equity,
    universe: universe?.curve[i]?.equity, drawdown: row.drawdown * 100,
  })) || [], [strategy, spy, universe]);

  async function download(kind) {
    setDownloadError("");
    try {
      const response = await api.get("/backtest/export", { params: { kind, strategy: selected }, responseType: "blob" });
      const url = URL.createObjectURL(response.data);
      const link = document.createElement("a");
      link.href = url;
      link.download = `quantsight-${kind}-${selected}.${kind === "report" ? "json" : "csv"}`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      setDownloadError("The export could not be downloaded. Please try again.");
    }
  }

  if (loading) return <div role="status" className="space-y-5"><p className="text-gray-300">Loading historical evaluation…</p>
    <div className="h-28 rounded-xl bg-gray-900 animate-pulse" /><div className="h-80 rounded-xl bg-gray-900 animate-pulse" /></div>;
  if (error) return <div role="alert" className="card"><h1 className="text-xl font-semibold">Backtesting</h1>
    <p className="my-4 text-gray-300">The evaluation could not be loaded.</p>
    <button className="px-4 py-2 rounded-lg bg-indigo-600 focus-visible:outline focus-visible:outline-2" onClick={refetch}>Try again</button></div>;
  if (!report || !strategy) return <div className="card"><h1 className="text-xl font-semibold">Backtesting</h1>
    <p className="mt-3 text-gray-300">A historical evaluation has not been published yet. Results will appear here when available.</p></div>;

  const m = strategy.metrics;
  const c = strategy.classification;
  const cfg = report.config;
  const ci = strategy.intervals.sharpe;
  const nullResult = report.random_null;
  const stale = report.freshness && !report.freshness.matches_current_files;
  return <div className="max-w-7xl mx-auto space-y-6">
    <div className="flex flex-col xl:flex-row xl:items-end justify-between gap-5">
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-indigo-400">Research & evaluation</p>
        <h1 className="text-3xl font-semibold tracking-tight text-white mt-2">Backtesting</h1>
        <p className="text-gray-400 text-sm mt-2 max-w-2xl">Historical portfolio performance after trading costs, compared on the same dates.
          This simulation is separate from the live prediction track record.</p>
      </div>
      <label className="text-sm text-gray-300">Strategy
        <select className="block mt-2 w-full xl:w-80 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2.5 text-white focus:ring-2 focus:ring-indigo-500"
          value={selected} onChange={e => setSelected(e.target.value)}>
          {report.strategies.map(s => <option key={s.key} value={s.key}>{s.name}</option>)}
        </select>
      </label>
    </div>

    {stale && <div role="alert" className="rounded-xl border border-red-800 bg-red-950/30 p-4 text-red-200 text-sm">
      <strong>Results need refreshing.</strong> Some data or evaluation files changed after this report was generated. These numbers belong to the saved run below.
    </div>}
    <div className="rounded-xl border border-amber-800/70 bg-amber-950/20 px-5 py-4 text-sm text-amber-100 leading-relaxed">
      <strong>Exploratory results.</strong> These models were developed using this historical period. Performance is not evidence of a verified live trading advantage.
      Sentiment coverage and other study limitations are listed below.
    </div>
    <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-gray-300 rounded-xl border border-gray-800 px-5 py-3">
      <span>{report.evaluation.start} → {report.evaluation.end}</span>
      <span>{report.evaluation.sessions} trading sessions</span><span>Starting capital {money(cfg.initial_capital)}</span>
      <span>Costs: {cfg.commission_bps + cfg.slippage_bps} bps per side</span>
      <span>Risk-free assumption: {pct(cfg.annual_risk_free_rate)}</span>
    </div>

    <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
      <Metric label="Net total return" value={pct(m.total_return)} note={`${money(m.final_equity)} ending value, after final liquidation`} />
      <Metric label="Sharpe ratio" value={num(m.sharpe)} note={ci ? `95% block-bootstrap interval: ${num(ci[0])} to ${num(ci[1])}` : "Undefined when excess-return volatility is zero"} />
      <Metric label="Maximum drawdown" value={pct(m.max_drawdown)} note={`Longest underwater period: ${m.max_drawdown_duration_sessions} sessions`} />
      <Metric label="Return minus SPY" value={pct(m.excess_total_return)} note="Difference in total return, in percentage points" />
    </div>

    <section className="card" aria-label="Equity curve">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-5"><h2 className="text-lg font-semibold">Growth of {money(cfg.initial_capital)}</h2>
        <span className="text-xs text-gray-400">Daily portfolio value · net of costs</span></div>
      <div className="h-80" role="img" aria-label="Daily portfolio value compared with SPY and the equal-weight universe. Exact figures are available in the daily CSV export.">
        <ResponsiveContainer width="100%" height="100%"><LineChart data={chart} margin={{ left: 10, right: 15 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="date" minTickGap={60} tickFormatter={v => v.slice(0, 7)} />
          <YAxis width={65} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} domain={["auto", "auto"]} />
          <Tooltip contentStyle={chartTooltip} formatter={value => money(value)} /><Legend />
          <Line name={strategy.name} dataKey="strategy" stroke="#818cf8" strokeWidth={2.5} dot={false} isAnimationActive={false} />
          {selected !== "spy" && <Line name="SPY buy and hold" dataKey="spy" stroke="#34d399" strokeDasharray="6 3" dot={false} isAnimationActive={false} />}
          {selected !== "universe" && <Line name="Equal-weight universe" dataKey="universe" stroke="#fbbf24" strokeDasharray="2 3" dot={false} isAnimationActive={false} />}
        </LineChart></ResponsiveContainer>
      </div>
    </section>
    <section className="card" aria-label="Drawdown chart"><h2 className="text-lg font-semibold mb-4">Decline from previous peak</h2>
      <div className="h-44" role="img" aria-label={`Maximum peak-to-trough decline: ${pct(m.max_drawdown)}`}>
        <ResponsiveContainer width="100%" height="100%"><AreaChart data={chart}>
          <XAxis dataKey="date" minTickGap={70} tickFormatter={v => v.slice(0, 7)} /><YAxis tickFormatter={v => `${v.toFixed(0)}%`} width={60} />
          <Tooltip contentStyle={chartTooltip} formatter={v => `${v.toFixed(2)}%`} /><ReferenceLine y={0} stroke="#6b7280" />
          <Area name="Drawdown" dataKey="drawdown" stroke="#fb7185" fill="#fb7185" fillOpacity={.12} isAnimationActive={false} />
        </AreaChart></ResponsiveContainer></div>
    </section>

    <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-4">
      <Metric label="Annualized return (CAGR)" value={pct(m.cagr)} note={`Annualized volatility: ${pct(m.annual_volatility)}`} />
      <Metric label="Sortino / Calmar" value={`${num(m.sortino)} / ${num(m.calmar)}`} note="Return relative to downside variation / worst drawdown" />
      <Metric label="Average market exposure" value={pct(m.average_exposure)} note={`Costs paid: ${money(m.total_cost)} · ${strategy.trade_count.toLocaleString()} executions`} />
      <Metric label="Positive-return days" value={pct(m.daily_win_rate)} note="Daily portfolio hit rate, not the percentage of winning trades" />
    </div>

    <section className="card !p-0 overflow-hidden"><h2 className="text-lg font-semibold px-5 pt-5 pb-3">All strategies · same evaluation period</h2>
      <div className="overflow-x-auto"><table className="data-table"><thead><tr>
        <th>Strategy</th><th>Net return</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>Max drawdown</th><th>Exposure</th>
      </tr></thead><tbody>{report.strategies.map(s => <tr key={s.key} className={s.key === selected ? "bg-indigo-950/30" : ""}>
        <td><button onClick={() => setSelected(s.key)} className="text-left text-gray-200 hover:text-indigo-300 focus-visible:underline">{s.name}</button></td>
        <td className="tabular-nums">{pct(s.metrics.total_return)}</td><td>{pct(s.metrics.cagr)}</td>
        <td>{num(s.metrics.sharpe)}</td><td>{num(s.metrics.sortino)}</td><td>{pct(s.metrics.max_drawdown)}</td><td>{pct(s.metrics.average_exposure)}</td>
      </tr>)}</tbody></table></div>
    </section>

    {c && <section className="card"><h2 className="text-lg font-semibold">Prediction quality is a separate measurement</h2>
      <p className="text-sm text-gray-400 mt-2">Five-session direction labels on {c.rows.toLocaleString()} matching ticker-days. These are classification outcomes, not trade returns.</p>
      <dl className="grid grid-cols-2 md:grid-cols-4 gap-5 mt-5">{[
        ["Accuracy", pct(c.accuracy)], ["Always predict up", pct(c.always_up_accuracy)],
        ["Balanced accuracy", pct(c.balanced_accuracy)], ["Macro F1", pct(c.macro_f1)],
        ["ROC-AUC · calibrated P(up)", num(c.roc_auc)], ["MCC", num(c.mcc)],
        ["Brier score · lower is better", num(c.brier)], ["BUY prediction rate", pct(c.buy_rate)],
      ].map(([label, value]) => <div key={label}><dt className="text-xs text-gray-400">{label}</dt><dd className="text-lg text-gray-100 mt-1 tabular-nums">{value}</dd></div>)}</dl>
    </section>}

    <div className="grid lg:grid-cols-2 gap-5">
      <section className="card"><h2 className="text-lg font-semibold">Could random selection do this?</h2>
        <p className="text-sm text-gray-300 leading-relaxed mt-3">{nullResult.runs.toLocaleString()} random portfolios match the <strong>four-model ensemble's</strong> position count, exposure and rebalance schedule. This comparison always refers to that ensemble, regardless of the selected chart.</p>
        <p className="mt-4 text-sm text-gray-300">Ensemble Sharpe percentile: <strong>{num(nullResult.reference_sharpe_percentile)} / 100</strong></p>
        <p className="mt-2 text-sm text-gray-300">Random Sharpe, 5th / median / 95th percentile: {nullResult.sharpe_p05_p50_p95?.map(num).join(" / ") || "—"}</p>
        <p className="mt-3 text-xs text-gray-400">A high percentile does not remove historical selection bias or replace a fresh test.</p>
      </section>
      <section className="card"><h2 className="text-lg font-semibold">Cost sensitivity · ensemble</h2>
        <table className="data-table mt-3"><thead><tr><th>Cost per side</th><th>Net return</th><th>Sharpe</th></tr></thead>
          <tbody>{report.cost_sensitivity.map(row => <tr key={row.one_way_cost_bps}><td>{row.one_way_cost_bps} bps</td><td>{pct(row.total_return)}</td><td>{num(row.sharpe)}</td></tr>)}</tbody></table>
      </section>
    </div>
    <details className="card"><summary className="cursor-pointer font-semibold text-gray-100">Methodology and study limitations</summary>
      <dl className="space-y-3 mt-4">{Object.entries(report.methodology).map(([key, value]) => <div key={key}>
        <dt className="text-sm font-medium capitalize text-gray-200">{key.replaceAll("_", " ")}</dt><dd className="text-sm text-gray-400 mt-1">{value}</dd>
      </div>)}</dl>
      <ul className="list-disc pl-5 mt-5 space-y-2 text-sm text-amber-100/90">{report.warnings.map(w => <li key={w}>{w}</li>)}</ul>
    </details>
    <section className="card"><h2 className="text-lg font-semibold">Download the evidence</h2>
      <p className="text-sm text-gray-400 mt-2">Daily values and executions allow the portfolio calculations to be checked independently.</p>
      <div className="flex flex-wrap gap-3 mt-4">{[["metrics", "All metrics CSV"], ["daily", "Selected daily values CSV"], ["trades", "Selected trades CSV"], ["report", "Full report JSON"]].map(([kind, label]) =>
        <button key={kind} onClick={() => download(kind)} className="px-4 py-2 text-sm rounded-lg border border-gray-700 bg-gray-800 hover:bg-gray-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-400">{label}</button>)}</div>
      {downloadError && <p role="alert" className="mt-3 text-sm text-red-300">{downloadError}</p>}
      <p className="text-xs text-gray-400 mt-5 break-all">Generated {report.generated_at.slice(0, 19).replace("T", " ")} UTC · Run {report.run_id}</p>
    </section>
  </div>;
}
