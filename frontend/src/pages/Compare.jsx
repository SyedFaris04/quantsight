/**
 * frontend/src/pages/Compare.jsx
 * ─────────────────────────────────────────────────────────────────
 * Page 2 — Model Comparison
 *
 * This is the page your lecturer specifically asked for.
 * Shows ALL 4 model variants side by side so the impact of:
 *   (a) model architecture  (XGBoost vs LSTM+Transformer)
 *   (b) adding sentiment    (Finance Only vs Finance+Sentiment)
 * ...is clearly visible.
 *
 * Layout:
 *   ┌──────────────────────────────────────────────────┐
 *   │  Section 1 — Accuracy Metrics Table              │
 *   │  4 rows × 5 metrics, best score highlighted      │
 *   ├──────────────────────────────────────────────────┤
 *   │  Section 2 — Grouped Bar Chart                   │
 *   │  Finance Only vs Finance+Sentiment per model     │
 *   ├──────────────────────────────────────────────────┤
 *   │  Section 3 — Sentiment Impact Summary            │
 *   │  How much did adding sentiment improve each?     │
 *   ├──────────────────────────────────────────────────┤
 *   │  Section 4 — Per-Ticker Signal Comparison        │
 *   │  Pick a ticker → see all 4 model signals         │
 *   └──────────────────────────────────────────────────┘
 * ─────────────────────────────────────────────────────────────────
 */

import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import { useApi } from "../hooks/useApi";

// ── Constants ──────────────────────────────────────────────────────────────────

const MODEL_ORDER = [
  {
    key     : "XGBoost_Finance Only",
    label   : "XGBoost",
    variant : "Finance Only",
    color   : "#3b82f6",   // blue
    short   : "XGB-F",
  },
  {
    key     : "XGBoost_Finance + Sentiment",
    label   : "XGBoost",
    variant : "Finance + Sentiment",
    color   : "#6366f1",   // indigo
    short   : "XGB-S",
  },
  {
    key     : "LSTM+Transformer_Finance Only",
    label   : "LSTM+Transformer",
    variant : "Finance Only",
    color   : "#f59e0b",   // amber
    short   : "LSTM-F",
  },
  {
    key     : "LSTM+Transformer_Finance + Sentiment",
    label   : "LSTM+Transformer",
    variant : "Finance + Sentiment",
    color   : "#a855f7",   // purple
    short   : "LSTM-S",
  },
];

const METRICS = [
  { key: "accuracy",  label: "Accuracy",  unit: "%", desc: "Overall correct predictions" },
  { key: "f1",        label: "F1 Score",  unit: "%", desc: "Balance of precision & recall" },
  { key: "precision", label: "Precision", unit: "%", desc: "Of predicted BUYs, how many were right" },
  { key: "recall",    label: "Recall",    unit: "%", desc: "Of actual BUYs, how many were caught" },
  { key: "auc_roc",   label: "AUC-ROC",  unit: "%", desc: "Area under ROC curve — overall discrimination" },
];

// ── Sub-components ─────────────────────────────────────────────────────────────

function SectionTitle({ children, sub }) {
  return (
    <div className="mb-4">
      <h2 className="text-base font-semibold text-white">{children}</h2>
      {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
    </div>
  );
}

function DeltaBadge({ delta }) {
  if (delta === null || delta === undefined) return null;
  const pos = delta >= 0;
  return (
    <span
      className={`text-xs font-medium px-1.5 py-0.5 rounded ${
        pos
          ? "bg-green-900/40 text-green-400"
          : "bg-red-900/40 text-red-400"
      }`}
    >
      {pos ? "+" : ""}{delta.toFixed(2)}%
    </span>
  );
}

// Custom tooltip for bar chart
function CustomBarTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-gray-300 font-medium mb-2">{label}</p>
      {payload.map(p => (
        <div key={p.name} className="flex items-center gap-2 mb-1">
          <span
            className="w-2 h-2 rounded-full"
            style={{ background: p.fill }}
          />
          <span className="text-gray-400">{p.name}:</span>
          <span className="text-white font-medium">{p.value?.toFixed(2)}%</span>
        </div>
      ))}
    </div>
  );
}

// Ticker signal comparison panel
function TickerCompare({ tickers }) {
  const [selected, setSelected] = useState("");
  const { data, loading } = useApi(
    selected ? `/compare/${selected}` : null,
    [selected]
  );
  const navigate = useNavigate();

  const modelMap = {
    xgb_finance    : { color: "#3b82f6", short: "XGB Finance"     },
    xgb_sentiment  : { color: "#6366f1", short: "XGB +Sentiment"  },
    lstm_finance   : { color: "#f59e0b", short: "LSTM Finance"    },
    lstm_sentiment : { color: "#a855f7", short: "LSTM +Sentiment" },
  };

  return (
    <div className="card">
      <SectionTitle
        sub="Select a ticker to see how all 4 models signal it"
      >
        Per-Ticker Signal Comparison
      </SectionTitle>

      <div className="flex gap-3 mb-5">
        <select
          className="input flex-1 max-w-xs"
          value={selected}
          onChange={e => setSelected(e.target.value)}
        >
          <option value="">— select a ticker —</option>
          {tickers.map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        {selected && (
          <button
            className="btn-secondary text-xs"
            onClick={() => navigate(`/detail/${selected}`)}
          >
            Full explanation →
          </button>
        )}
      </div>

      {!selected && (
        <p className="text-gray-600 text-sm text-center py-8">
          Select a ticker above to compare model signals
        </p>
      )}

      {selected && loading && (
        <div className="flex justify-center py-8">
          <div className="spinner" />
        </div>
      )}

      {selected && !loading && data && (
        <div className="space-y-4">
          {/* Agreement banner */}
          <div className={`rounded-lg px-4 py-3 flex items-center justify-between
            ${data.agreement_level === "Strong"
              ? "bg-green-900/20 border border-green-800"
              : data.agreement_level === "Moderate"
              ? "bg-amber-900/20 border border-amber-800"
              : "bg-gray-800 border border-gray-700"
            }`}
          >
            <div>
              <span className="text-sm font-semibold text-white">
                {data.ticker}
              </span>
              <span className="text-gray-500 text-sm ml-2">
                Overall: {" "}
                <span
                  className={
                    data.overall_signal === "BUY"
                      ? "text-green-400 font-semibold"
                      : data.overall_signal === "SELL"
                        ? "text-red-400 font-semibold"
                        : "text-amber-400 font-semibold"
                  }
                >
                  {data.overall_signal}
                </span>
                {" "}({data.overall_confidence?.toFixed(1)}% avg confidence)
              </span>
            </div>
            <span
              className={`text-xs font-medium px-2 py-1 rounded-full border ${
                data.agreement_level === "Strong"
                  ? "bg-green-900/30 text-green-400 border-green-700"
                  : data.agreement_level === "Moderate"
                  ? "bg-amber-900/30 text-amber-400 border-amber-700"
                  : "bg-gray-700 text-gray-400 border-gray-600"
              }`}
            >
              {data.agreement_level} Agreement
            </span>
          </div>

          {/* 4 model signal cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {data.models?.map(m => {
              const meta  = modelMap[m.model_key] || {};
              const isBuy = m.signal_label === "BUY";
              return (
                <div
                  key={m.model_key}
                  className="bg-gray-800/50 rounded-xl p-4 flex flex-col gap-2
                             border border-gray-700"
                >
                  <span
                    className="text-xs font-medium"
                    style={{ color: meta.color }}
                  >
                    {meta.short}
                  </span>
                  <span
                    className={`text-xl font-bold ${
                      isBuy ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {isBuy ? "▲ BUY" : "▼ SELL"}
                  </span>
                  <div className="w-full bg-gray-700 rounded-full h-1.5 mt-1">
                    <div
                      className={`h-1.5 rounded-full ${
                        isBuy ? "bg-green-500" : "bg-red-500"
                      }`}
                      style={{ width: `${m.confidence}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-400">
                    {m.confidence?.toFixed(1)}% confidence
                  </span>
                  <span className="text-xs text-gray-600">
                    {m.uses_sentiment ? "📰 incl. sentiment" : "📈 finance only"}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function Compare() {
  const { data: metricsRaw, loading: metricsLoading, error: metricsError } =
    useApi("/metrics");
  const { data: tickerData } = useApi("/tickers");
  const { data: confBoost, loading: confBoostLoading } = useApi("/confidence-boost");

  const tickers = tickerData?.tickers || [];

  // ── Normalise metrics into ordered array ───────────────────────
  const models = useMemo(() => {
    if (!metricsRaw) return [];
    return MODEL_ORDER.map(m => ({
      ...m,
      ...(metricsRaw[m.key] || {}),
    })).filter(m => m.accuracy != null);
  }, [metricsRaw]);

  // ── Find best score per metric column ─────────────────────────
  const bestScores = useMemo(() => {
    const best = {};
    METRICS.forEach(({ key }) => {
      best[key] = Math.max(...models.map(m => m[key] ?? 0));
    });
    return best;
  }, [models]);

  // ── Bar chart data — grouped by metric ────────────────────────
  const barData = useMemo(() => {
    return METRICS.map(({ key, label }) => {
      const entry = { metric: label };
      MODEL_ORDER.forEach(m => {
        const found = models.find(x => x.key === m.key);
        if (found) entry[m.short] = found[key];
      });
      return entry;
    });
  }, [models]);

  // ── Sentiment impact delta (Finance Only → Finance+Sentiment) ──
  const sentimentImpact = useMemo(() => {
    const xgbF = models.find(m => m.key === "XGBoost_Finance Only");
    const xgbS = models.find(m => m.key === "XGBoost_Finance + Sentiment");
    const lstmF = models.find(m => m.key === "LSTM+Transformer_Finance Only");
    const lstmS = models.find(m => m.key === "LSTM+Transformer_Finance + Sentiment");
    return { xgbF, xgbS, lstmF, lstmS };
  }, [models]);

  // ── Top KPI summary — all derived directly from model_metrics.json ─
  const summaryKpis = useMemo(() => {
    if (!models.length) return null;
    const best = models.reduce((a, b) => (b.accuracy > a.accuracy ? b : a));

    const xgbF  = models.find(m => m.key === "XGBoost_Finance Only");
    const xgbS  = models.find(m => m.key === "XGBoost_Finance + Sentiment");
    const lstmF = models.find(m => m.key === "LSTM+Transformer_Finance Only");
    const lstmS = models.find(m => m.key === "LSTM+Transformer_Finance + Sentiment");

    const accDeltas = [];
    if (xgbF && xgbS)   accDeltas.push(xgbS.accuracy - xgbF.accuracy);
    if (lstmF && lstmS) accDeltas.push(lstmS.accuracy - lstmF.accuracy);
    const bestAccDelta = accDeltas.length ? Math.max(...accDeltas) : null;

    const f1Deltas = [];
    if (xgbF && xgbS)   f1Deltas.push(xgbS.f1 - xgbF.f1);
    if (lstmF && lstmS) f1Deltas.push(lstmS.f1 - lstmF.f1);
    const bestF1Delta = f1Deltas.length ? Math.max(...f1Deltas) : null;

    return {
      bestModelLabel : `${best.label} ${best.variant === "Finance + Sentiment" ? "+ Sentiment" : "(Finance)"}`,
      bestAccuracy   : best.accuracy,
      bestAccDelta,
      bestF1Delta,
    };
  }, [models]);

  // ── Radar chart data ───────────────────────────────────────────
  const radarData = useMemo(() => {
    return METRICS.map(({ key, label }) => {
      const entry = { metric: label };
      models.forEach(m => { entry[m.short] = m[key]; });
      return entry;
    });
  }, [models]);

  if (metricsError) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-4">
        <span className="text-4xl">⚠️</span>
        <p className="text-gray-400 text-sm text-center max-w-md">
          Could not load model metrics. Make sure{" "}
          <code className="text-indigo-400">train_xgboost.py</code> and{" "}
          <code className="text-indigo-400">train_lstm.py</code> have been run
          and <code className="text-indigo-400">model_metrics.json</code> exists.
        </p>
        <p className="text-xs text-gray-600">{metricsError}</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">

      {/* ── Page header ── */}
      <div>
        <h1 className="text-xl font-semibold text-white">Model Comparison</h1>
        <p className="text-sm text-gray-500 mt-1">
          XGBoost vs LSTM+Transformer — with and without sentiment data
        </p>
      </div>

      {/* ══ KPI summary row — all values derived from model_metrics.json ═══ */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        {metricsLoading ? (
          [1, 2, 3, 4, 5].map(i => <div key={i} className="h-20 bg-gray-900 rounded-xl animate-pulse" />)
        ) : summaryKpis ? (
          <>
            <div className="card border border-indigo-600/40 bg-indigo-600/5">
              <div className="text-xs text-gray-500 uppercase tracking-wider">Best Model</div>
              <div className="text-lg font-semibold text-white mt-1">{summaryKpis.bestModelLabel}</div>
            </div>
            <div className="card border border-green-600/40 bg-green-600/5">
              <div className="text-xs text-gray-500 uppercase tracking-wider">Highest Accuracy</div>
              <div className="text-2xl font-semibold text-white mt-1">{summaryKpis.bestAccuracy.toFixed(2)}%</div>
            </div>
            <div className="card border border-amber-600/40 bg-amber-600/5">
              <div className="text-xs text-gray-500 uppercase tracking-wider">Best Accuracy Lift</div>
              <div className="text-2xl font-semibold text-white mt-1">
                {summaryKpis.bestAccDelta != null
                  ? `${summaryKpis.bestAccDelta >= 0 ? "+" : ""}${summaryKpis.bestAccDelta.toFixed(2)}%`
                  : "—"}
              </div>
              <div className="text-xs text-gray-500 mt-0.5">accuracy vs finance-only</div>
            </div>
            <div className="card border border-blue-600/40 bg-blue-600/5">
              <div className="text-xs text-gray-500 uppercase tracking-wider">Best F1 Lift</div>
              <div className="text-2xl font-semibold text-white mt-1">
                {summaryKpis.bestF1Delta != null
                  ? `${summaryKpis.bestF1Delta >= 0 ? "+" : ""}${summaryKpis.bestF1Delta.toFixed(2)}%`
                  : "—"}
              </div>
              <div className="text-xs text-gray-500 mt-0.5">F1 vs finance-only</div>
            </div>
            <div className="card border border-purple-600/40 bg-purple-600/5">
              <div className="text-xs text-gray-500 uppercase tracking-wider">Confidence Boost</div>
              <div className="text-2xl font-semibold text-white mt-1">
                {confBoostLoading ? "—" : confBoost?.best && confBoost[confBoost.best]
                  ? `${confBoost[confBoost.best].avg_confidence_boost >= 0 ? "+" : ""}${confBoost[confBoost.best].avg_confidence_boost.toFixed(2)}pp`
                  : "—"}
              </div>
              <div className="text-xs text-gray-500 mt-0.5">
                {confBoost?.best ? `avg ${confBoost.best.toUpperCase()} confidence Δ` : "avg confidence delta"}
              </div>
            </div>
          </>
        ) : null}
      </div>

      {/* ══ Section 1 — Accuracy Metrics Table ══════════════════════════════ */}
      <div className="card">
        <SectionTitle
          sub="Each metric shows all 4 variants — best score in each column is highlighted"
        >
          Accuracy Metrics — All 4 Model Variants
        </SectionTitle>

        {metricsLoading ? (
          <div className="space-y-3">
            {[1,2,3,4].map(i => (
              <div key={i} className="h-10 bg-gray-800 rounded animate-pulse" />
            ))}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="w-64">Model</th>
                  <th className="w-24">Variant</th>
                  {METRICS.map(m => (
                    <th key={m.key} className="text-center">
                      <div>{m.label}</div>
                      <div className="text-gray-600 font-normal normal-case text-xs">
                        {m.desc}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.map((model, idx) => (
                  <tr key={model.key}>
                    {/* Model name */}
                    <td>
                      <div className="flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                          style={{ background: model.color }}
                        />
                        <span className="font-medium text-white text-sm">
                          {model.label}
                        </span>
                      </div>
                    </td>

                    {/* Variant */}
                    <td>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full border ${
                          model.variant === "Finance + Sentiment"
                            ? "bg-indigo-900/30 text-indigo-400 border-indigo-800"
                            : "bg-gray-800 text-gray-400 border-gray-700"
                        }`}
                      >
                        {model.variant === "Finance + Sentiment"
                          ? "+Sentiment"
                          : "Finance"}
                      </span>
                    </td>

                    {/* Metric values */}
                    {METRICS.map(({ key }) => {
                      const val     = model[key];
                      const isBest  = val === bestScores[key];
                      return (
                        <td key={key} className="text-center">
                          <span
                            className={`text-sm font-medium ${
                              isBest
                                ? "text-green-400"
                                : "text-gray-300"
                            }`}
                          >
                            {val?.toFixed(2) ?? "—"}%
                          </span>
                          {isBest && (
                            <span className="ml-1 text-green-500 text-xs">★</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>

            <p className="text-xs text-gray-600 mt-3 px-4">
              ★ Best score in each column &nbsp;|&nbsp;
              Higher is better for all metrics
            </p>
          </div>
        )}
      </div>

      {/* ══ Section 2 — Grouped Bar Chart ═══════════════════════════════════ */}
      <div className="card">
        <SectionTitle
          sub="Side-by-side comparison of all 4 variants across every metric"
        >
          Performance Bar Chart
        </SectionTitle>

        {metricsLoading ? (
          <div className="h-72 bg-gray-800 rounded animate-pulse" />
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={barData}
              margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
              barCategoryGap="25%"
              barGap={2}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="metric"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                axisLine={{ stroke: "#374151" }}
                tickLine={false}
              />
              <YAxis
                domain={[50, 100]}
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={v => `${v}%`}
              />
              <Tooltip content={<CustomBarTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: "12px", color: "#9ca3af", paddingTop: "12px" }}
              />
              {MODEL_ORDER.map(m => (
                <Bar key={m.short} dataKey={m.short} fill={m.color} radius={[3,3,0,0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ══ Section 3 — Sentiment & Emotion Impact ═══════════════════════════ */}
      <div className="card">
        <SectionTitle
          sub="How much did adding WSB sentiment + GoEmotions emotion features improve each model?"
        >
          Sentiment & Emotion Impact Analysis
        </SectionTitle>

        {metricsLoading ? (
          <div className="h-32 bg-gray-800 rounded animate-pulse" />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* XGBoost impact */}
            {sentimentImpact.xgbF && sentimentImpact.xgbS && (
              <div className="space-y-3">
                <div className="flex items-center gap-2 mb-3">
                  <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                  <span className="text-sm font-medium text-white">
                    XGBoost — sentiment + emotion impact
                  </span>
                </div>
                {METRICS.map(({ key, label }) => {
                  const delta = (sentimentImpact.xgbS[key] ?? 0) -
                                (sentimentImpact.xgbF[key] ?? 0);
                  return (
                    <div key={key} className="flex items-center justify-between text-sm">
                      <span className="text-gray-400 w-24">{label}</span>
                      <div className="flex items-center gap-3 flex-1 ml-4">
                        <span className="text-gray-500 text-xs w-14 text-right">
                          {sentimentImpact.xgbF[key]?.toFixed(1)}%
                        </span>
                        <div className="flex-1 h-1.5 bg-gray-800 rounded-full relative">
                          <div
                            className="h-1.5 rounded-full bg-blue-600"
                            style={{
                              width: `${Math.min(sentimentImpact.xgbF[key] ?? 0, 100)}%`,
                            }}
                          />
                        </div>
                        <span className="text-gray-300 text-xs w-14">
                          {sentimentImpact.xgbS[key]?.toFixed(1)}%
                        </span>
                        <DeltaBadge delta={delta} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* LSTM impact */}
            {sentimentImpact.lstmF && sentimentImpact.lstmS && (
              <div className="space-y-3">
                <div className="flex items-center gap-2 mb-3">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                  <span className="text-sm font-medium text-white">
                    LSTM+Transformer — sentiment + emotion impact
                  </span>
                </div>
                {METRICS.map(({ key, label }) => {
                  const delta = (sentimentImpact.lstmS[key] ?? 0) -
                                (sentimentImpact.lstmF[key] ?? 0);
                  return (
                    <div key={key} className="flex items-center justify-between text-sm">
                      <span className="text-gray-400 w-24">{label}</span>
                      <div className="flex items-center gap-3 flex-1 ml-4">
                        <span className="text-gray-500 text-xs w-14 text-right">
                          {sentimentImpact.lstmF[key]?.toFixed(1)}%
                        </span>
                        <div className="flex-1 h-1.5 bg-gray-800 rounded-full relative">
                          <div
                            className="h-1.5 rounded-full bg-amber-600"
                            style={{
                              width: `${Math.min(sentimentImpact.lstmF[key] ?? 0, 100)}%`,
                            }}
                          />
                        </div>
                        <span className="text-gray-300 text-xs w-14">
                          {sentimentImpact.lstmS[key]?.toFixed(1)}%
                        </span>
                        <DeltaBadge delta={delta} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ══ Section 4 — Radar Chart ══════════════════════════════════════════ */}
      <div className="card">
        <SectionTitle
          sub="Spider chart view — shows relative strengths of each model across all metrics"
        >
          Model Profile — Radar Chart
        </SectionTitle>

        {metricsLoading ? (
          <div className="h-72 bg-gray-800 rounded animate-pulse" />
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#1f2937" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[50, 100]}
                tick={{ fill: "#4b5563", fontSize: 9 }}
                tickCount={4}
              />
              {MODEL_ORDER.map(m => (
                <Radar
                  key={m.short}
                  name={m.short}
                  dataKey={m.short}
                  stroke={m.color}
                  fill={m.color}
                  fillOpacity={0.08}
                  strokeWidth={2}
                />
              ))}
              <Legend
                wrapperStyle={{ fontSize: "12px", color: "#9ca3af" }}
              />
              <Tooltip content={<CustomBarTooltip />} />
            </RadarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ══ Section 5 — Per-Ticker Comparison ═══════════════════════════════ */}
      <TickerCompare tickers={tickers} />

    </div>
  );
}
