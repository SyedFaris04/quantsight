import { useState } from "react";
import { Link } from "react-router-dom";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceArea, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { useApi } from "../hooks/useApi";

// Validated against the dark card surface (dataviz validator: all checks pass).
const COLOR = { 2015: "#8b5cf6", 2005: "#0d9488", baseline: "#7f8a99" };
const MODELS = {
  xgb_finance: "XGBoost · finance",
  xgb_sentiment: "XGBoost · finance + sentiment",
  lstm_finance: "LSTM · finance",
  lstm_sentiment: "LSTM · finance + sentiment",
};
const pct = (v, d = 1) => (Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : "—");
const num = (v, d = 4) => (Number.isFinite(v) ? v.toFixed(d) : "—");
const signed = (v, d, scale = 1) =>
  Number.isFinite(v) ? `${v >= 0 ? "+" : "−"}${Math.abs(v * scale).toFixed(d)}` : "—";

function Interval({ value, d = 4, scale = 1, unit = "" }) {
  const [lo, hi] = value.interval_95;
  const clear = lo > 0 || hi < 0;
  return (
    <span className="tabular-nums">
      {signed(value.difference, d, scale)}
      {unit}
      <span className="block text-xs text-gray-400">
        [{signed(lo, d, scale)}, {signed(hi, d, scale)}] {clear ? "· clear" : "· could be zero"}
      </span>
    </span>
  );
}

function Section({ id, title, intro, children }) {
  return (
    <section className="card" aria-labelledby={id}>
      <h2 id={id} className="text-lg font-semibold">{title}</h2>
      {intro && <p className="mt-1 text-sm text-gray-400 max-w-3xl">{intro}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

function ModelPicker({ value, onChange, options = Object.keys(MODELS), id }) {
  return (
    <label className="text-sm text-gray-300 flex items-center gap-2" htmlFor={id}>
      Model
      <select id={id} className="input py-1" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((m) => <option key={m} value={m}>{MODELS[m]}</option>)}
      </select>
    </label>
  );
}

const tooltipStyle = { background: "#151b23", border: "1px solid #343d49", borderRadius: 6, fontSize: 12 };

function Training({ report, episodes }) {
  const years = Object.keys(report.timelines["2005"].training.tickers_by_year);
  const rows = years.map((year) => ({
    year,
    2005: report.timelines["2005"].training.tickers_by_year[year] ?? null,
    2015: report.timelines["2015"].training.tickers_by_year[year] ?? null,
  }));
  return (
    <Section id="training" title="What each version learned from"
      intro="Same models, settings and test days. The only difference is the first training date.">
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead><tr><th scope="col">Version</th><th scope="col">Training period</th><th scope="col">Rows</th>
            <th scope="col">Share up after 5 days</th><th scope="col">Crises seen in training</th></tr></thead>
          <tbody>
            {["2015", "2005"].map((t) => {
              const tr = report.timelines[t].training;
              return (
                <tr key={t}>
                  <th scope="row" className="px-4 py-3 text-left text-sm font-medium">Trained from {t}</th>
                  <td className="tabular-nums">{tr.start} to {tr.end}</td>
                  <td className="tabular-nums">{tr.rows.toLocaleString()}</td>
                  <td className="tabular-nums">{pct(tr.up_rate)}</td>
                  <td>{tr.episodes_included.map((k) => episodes[k]).join(", ") || "None of the listed crises"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <h3 className="mt-6 text-sm font-medium">Instruments available each year</h3>
      <div className="h-48 mt-2" role="img" aria-label="Number of instruments with training data each year, by version">
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: -16 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="year" tickLine={false} />
            <YAxis domain={[30, 45]} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend />
            <Line isAnimationActive={false} dataKey="2005" name="Trained from 2005" stroke={COLOR[2005]} strokeWidth={2} dot={false} />
            <Line isAnimationActive={false} dataKey="2015" name="Trained from 2015" stroke={COLOR[2015]} strokeWidth={2} dot={false} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Section>
  );
}

function Quality({ report }) {
  const { timelines, differences, baselines } = report;
  return (
    <Section id="quality" title="Prediction quality on the same test days"
      intro={`${report.test_window.start} to ${report.test_window.end}. Brier loss measures probability error (lower is better). The bracket is a 95% range for the difference; "could be zero" means no clear difference.`}>
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead><tr><th scope="col">Model</th><th scope="col">Accuracy 2015</th><th scope="col">Accuracy 2005</th>
            <th scope="col">Difference (pts)</th><th scope="col">Brier 2015</th><th scope="col">Brier 2005</th>
            <th scope="col">Difference</th><th scope="col">AUC 2015 / 2005</th></tr></thead>
          <tbody>
            {Object.keys(MODELS).map((m) => {
              const a = timelines["2015"].models[m], b = timelines["2005"].models[m];
              return (
                <tr key={m}>
                  <th scope="row" className="px-4 py-3 text-left text-sm font-medium">{MODELS[m]}</th>
                  <td className="tabular-nums">{pct(a.accuracy, 2)}</td>
                  <td className="tabular-nums">{pct(b.accuracy, 2)}</td>
                  <td><Interval value={differences[m].accuracy} d={2} scale={100} /></td>
                  <td className="tabular-nums">{num(a.brier)}</td>
                  <td className="tabular-nums">{num(b.brier)}</td>
                  <td><Interval value={differences[m].brier} /></td>
                  <td className="tabular-nums">{num(a.auc, 3)} / {num(b.auc, 3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-gray-400">
        Baselines: always predicting UP scores {pct(baselines.always_up_accuracy.xgb_finance, 2)} accuracy.
        A constant training-period probability scores Brier {num(baselines.train_prior_brier["2015"].xgb_finance)} (2015)
        and {num(baselines.train_prior_brier["2005"].xgb_finance)} (2005).
      </p>
      <p className="mt-2 text-xs text-gray-400">
        LSTM run-to-run spread over seeds {timelines["2015"].models.lstm_finance.lstm_seeds.seeds.join(", ")}:{" "}
        {["lstm_finance", "lstm_sentiment"].map((m) => ["2015", "2005"].map((t) => {
          const s = timelines[t].models[m].lstm_seeds.accuracy;
          return `${MODELS[m]} ${t}: ${pct(s.min, 1)}–${pct(s.max, 1)}`;
        }).join("; ")).join("; ")}.
      </p>
    </Section>
  );
}

function Calibration({ report }) {
  const [model, setModel] = useState("xgb_finance");
  const points = (t) => report.timelines[t].models[model].reliability.map((b) => ({ x: b.mean_probability, y: b.observed_up_rate }));
  const all = ["2015", "2005"].flatMap(points);
  const lo = Math.min(...all.map((p) => Math.min(p.x, p.y))), hi = Math.max(...all.map((p) => Math.max(p.x, p.y)));
  const ideal = [{ x: lo, y: lo }, { x: hi, y: hi }];
  return (
    <Section id="calibration" title="When a version says 60%, how often was it right?"
      intro="Predictions are sorted by stated probability and split into ten equal groups. Each point compares a group's average stated probability with how often it actually went up; the dashed line is perfect calibration.">
      <ModelPicker id="calibration-model" value={model} onChange={setModel} />
      <div className="h-64 mt-4" role="img" aria-label={`Calibration of ${MODELS[model]} for both versions`}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 16, bottom: 16, left: -8 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="x" type="number" domain={[lo, hi]} tickFormatter={(v) => pct(v, 0)} tickLine={false}
              name="Stated" label={{ value: "Stated chance of going up", position: "insideBottom", offset: -8, fill: "#a1aab8", fontSize: 11 }} />
            <YAxis dataKey="y" type="number" domain={[lo, hi]} tickFormatter={(v) => pct(v, 0)} tickLine={false} axisLine={false}
              name="Went up" />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => pct(v, 1)} />
            <Legend verticalAlign="top" />
            <Scatter isAnimationActive={false} data={ideal} name="Perfect calibration" line={{ stroke: COLOR.baseline, strokeDasharray: "4 4" }} shape={() => null} legendType="plainline" fill={COLOR.baseline} />
            <Scatter isAnimationActive={false} data={points("2015")} name="Trained from 2015" fill={COLOR[2015]} line={{ stroke: COLOR[2015], strokeWidth: 2 }} />
            <Scatter isAnimationActive={false} data={points("2005")} name="Trained from 2005" fill={COLOR[2005]} line={{ stroke: COLOR[2005], strokeWidth: 2 }} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Section>
  );
}

function Agreement({ report }) {
  return (
    <Section id="agreement" title="How often the two versions agree"
      intro="When they disagree, which one was right more often?">
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead><tr><th scope="col">Model</th><th scope="col">Same call</th><th scope="col">Disagreements</th>
            <th scope="col">2015 right</th><th scope="col">2005 right</th></tr></thead>
          <tbody>
            {Object.keys(MODELS).map((m) => {
              const a = report.agreement[m];
              return (
                <tr key={m}>
                  <th scope="row" className="px-4 py-3 text-left text-sm font-medium">{MODELS[m]}</th>
                  <td className="tabular-nums">{pct(a.agree_rate)}</td>
                  <td className="tabular-nums">{a.disagree_rows.toLocaleString()}</td>
                  <td className="tabular-nums">{pct(a.a_correct_when_disagree)}</td>
                  <td className="tabular-nums">{pct(a.b_correct_when_disagree)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function Trading({ report }) {
  const runs = [["Trained from 2015", report.timelines["2015"].backtest.ensemble],
    ["Trained from 2005", report.timelines["2005"].backtest.ensemble], ["SPY buy and hold", report.timelines["2015"].backtest.spy]];
  const ev = report.timelines["2015"].backtest.evaluation, cfg = report.timelines["2015"].backtest.config;
  return (
    <Section id="trading" title="Trading simulation"
      intro={`Four served models averaged, top ${cfg.top_n} BUY calls, weekly rebalance at next open, ${cfg.commission_bps} + ${cfg.slippage_bps} bps costs per side. ${ev.start} to ${ev.end}, ${ev.sessions} sessions.`}>
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead><tr><th scope="col">Strategy</th><th scope="col">Net return</th><th scope="col">Yearly</th>
            <th scope="col">Sharpe</th><th scope="col">Worst drop</th><th scope="col">Yearly turnover</th></tr></thead>
          <tbody>
            {runs.map(([name, m]) => (
              <tr key={name}>
                <th scope="row" className="px-4 py-3 text-left text-sm font-medium">{name}</th>
                <td className="tabular-nums">{pct(m.total_return)}</td>
                <td className="tabular-nums">{pct(m.cagr)}</td>
                <td className="tabular-nums">{num(m.sharpe, 2)}</td>
                <td className="tabular-nums">{pct(m.max_drawdown)}</td>
                <td className="tabular-nums">{num(m.annual_one_way_turnover, 1)}×</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-amber-200/90">Exploratory simulation on reused historical data; not a live result.</p>
    </Section>
  );
}

function Features({ report }) {
  const [model, setModel] = useState("xgb_finance");
  return (
    <Section id="features" title="What each version relies on"
      intro="Top indicators by average TreeSHAP contribution on the test days (XGBoost only). Rank 1 matters most.">
      <ModelPicker id="feature-model" value={model} onChange={setModel} options={["xgb_finance", "xgb_sentiment"]} />
      <div className="overflow-x-auto mt-4">
        <table className="data-table">
          <thead><tr><th scope="col">Indicator</th><th scope="col">Rank 2015</th><th scope="col">Rank 2005</th>
            <th scope="col">Change</th></tr></thead>
          <tbody>
            {report.shap_rank_changes[model].map((r) => (
              <tr key={r.feature}>
                <th scope="row" className="px-4 py-3 text-left text-sm font-mono font-normal normal-case tracking-normal">{r.feature}</th>
                <td className="tabular-nums">{r.rank_2015}</td>
                <td className="tabular-nums">{r.rank_2005}</td>
                <td className="tabular-nums">
                  {r.rank_2015 === r.rank_2005 ? "Same" : r.rank_2005 < r.rank_2015
                    ? `Up ${r.rank_2015 - r.rank_2005}` : `Down ${r.rank_2005 - r.rank_2015}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function Crisis({ report, episodes }) {
  const { crisis } = report;
  const pick = (arm, candidate, year) => crisis.years.find((r) => r.arm === arm && r.candidate === candidate && r.year === year);
  const years = [...new Set(crisis.years.map((r) => r.year))].sort();
  const rows = years.map((year) => ({
    year,
    2005: pick("2005", "xgboost_sigmoid", year)?.brier ?? null,
    2015: pick("2015", "xgboost_sigmoid", year)?.brier ?? null,
    baseline: pick("2005", "fit_prior", year)?.brier ?? null,
  }));
  const crisisYears = new Set(crisis.episodes_defined.flatMap((e) => {
    const out = [];
    for (let y = +e.start.slice(0, 4); y <= +e.end.slice(0, 4); y++) out.push(y);
    return out;
  }));
  const xgb = crisis.arm_comparison.find((c) => c.candidate === "xgboost_sigmoid");
  const episodeRows = crisis.episodes.filter((e) => e.candidate === "xgboost_sigmoid");
  const baseline = (e) => crisis.episodes.find((b) => b.candidate === "fit_prior" && b.arm === e.arm && b.episode === e.episode);
  return (
    <Section id="crisis" title="Crisis years, one year at a time"
      intro="Each year is predicted by a model trained only on earlier years (compact 13-indicator XGBoost, calibrated on the year before). Shaded years contain a listed crisis. Lower is better; the grey line is a constant-probability baseline.">
      <div className="h-72" role="img" aria-label="Yearly Brier loss 2008 to 2024 for both training starts and a baseline">
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} />
            {[...crisisYears].filter((y) => years.includes(y)).map((y) => (
              <ReferenceArea key={y} x1={y - 0.5} x2={y + 0.5} fill="#ffffff" fillOpacity={0.06} stroke="none" />
            ))}
            <XAxis dataKey="year" type="number" domain={[years[0] - 0.5, years[years.length - 1] + 0.5]}
              ticks={years} interval="preserveStartEnd" tickLine={false} />
            <YAxis domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(3)} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => num(v)} />
            <Legend />
            <Line isAnimationActive={false} dataKey="baseline" name="Constant-probability baseline" stroke={COLOR.baseline} strokeDasharray="4 4" dot={false} />
            <Line isAnimationActive={false} dataKey="2015" name="Trained from 2015" stroke={COLOR[2015]} strokeWidth={2} dot={{ r: 4 }} />
            <Line isAnimationActive={false} dataKey="2005" name="Trained from 2005" stroke={COLOR[2005]} strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {xgb && (
        <p className="mt-4 text-sm">
          Paired over {xgb.years[0]}–{xgb.years[xgb.years.length - 1]} ({xgb.rows.toLocaleString()} predictions),
          starting in 2005 changed Brier loss by <Interval value={xgb.brier} /> and accuracy by{" "}
          <Interval value={xgb.accuracy} d={2} scale={100} unit=" pts" />
        </p>
      )}
      <div className="overflow-x-auto mt-5">
        <table className="data-table">
          <caption className="text-left text-sm font-medium pb-2">Named episodes</caption>
          <thead><tr><th scope="col">Episode</th><th scope="col">Version</th><th scope="col">Predictions</th>
            <th scope="col">Accuracy</th><th scope="col">Always UP</th><th scope="col">Brier</th><th scope="col">Baseline Brier</th></tr></thead>
          <tbody>
            {episodeRows.map((e) => (
              <tr key={`${e.episode}-${e.arm}`}>
                <th scope="row" className="px-4 py-3 text-left text-sm font-medium">{episodes[e.episode]}</th>
                <td>From {e.arm}</td>
                <td className="tabular-nums">{e.n.toLocaleString()}</td>
                <td className="tabular-nums">{pct(e.metrics.accuracy)}</td>
                <td className="tabular-nums">{pct(e.always_up_accuracy)}</td>
                <td className="tabular-nums">{num(e.metrics.brier_score)}</td>
                <td className="tabular-nums">{num(baseline(e)?.metrics.brier_score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <details className="mt-4 text-sm">
        <summary>Yearly values as a table</summary>
        <div className="overflow-x-auto mt-3">
          <table className="data-table">
            <thead><tr><th scope="col">Year</th><th scope="col">Brier from 2015</th><th scope="col">Brier from 2005</th>
              <th scope="col">Baseline</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.year}>
                  <th scope="row" className="px-4 py-2 text-left text-sm font-normal">{r.year}</th>
                  <td className="tabular-nums">{num(r[2015])}</td>
                  <td className="tabular-nums">{num(r[2005])}</td>
                  <td className="tabular-nums">{num(r.baseline)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </Section>
  );
}

export default function TimelineComparison() {
  const { data, loading, error, refetch, isSlow } = useApi("/research/timeline-comparison");
  const report = data?.report;
  const episodes = Object.fromEntries((report?.crisis.episodes_defined || []).map((e) => [e.key, e.label]));
  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <header className="page-heading">
        <div>
          <p className="eyebrow">Evaluation</p>
          <h1>Training history: 2015 vs 2005</h1>
          <p>Does training on ten more years, including 2008, change the predictions?</p>
        </div>
        <Link to="/?timeline=2005" className="text-sm text-indigo-300">Dashboard with 2005 models →</Link>
      </header>
      {loading ? (
        <p role="status" className="text-sm text-gray-400">
          {isSlow ? "Still waiting. The server may be starting or busy." : "Loading comparison…"}
        </p>
      ) : error ? (
        <div role="alert" className="card text-sm">
          <p>{error}</p>
          <button onClick={refetch} className="btn-secondary mt-3">Try again</button>
        </div>
      ) : !data?.available ? (
        <p className="card text-sm text-gray-400">The timeline comparison has not been generated on this server.</p>
      ) : (
        <>
          <Training report={report} episodes={episodes} />
          <Quality report={report} />
          <Calibration report={report} />
          <Agreement report={report} />
          <Trading report={report} />
          <Features report={report} />
          <Crisis report={report} episodes={episodes} />
          <Section id="limits" title="Limits of this comparison">
            <ul className="list-disc pl-5 space-y-1.5 text-sm text-gray-300">
              {report.limitations.map((l) => <li key={l}>{l}</li>)}
              <li>
                The retrained 2015 XGBoost matches the current live model on{" "}
                {pct(report.production_check.xgb_finance.agree_rate)} of test predictions; live forecasts still use the
                current models.
              </li>
            </ul>
          </Section>
        </>
      )}
    </div>
  );
}
