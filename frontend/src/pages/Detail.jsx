/**
 * frontend/src/pages/Detail.jsx
 * Stock Detail page — price chart with signal overlay + AI copilot + news
 */

import { useState, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { useApi } from "../hooks/useApi";
import { companyName } from "../data/companyNames";
import RadialProgress from "../components/RadialProgress";
import CandlestickChart from "../components/CandlestickChart";

// ── Constants ──────────────────────────────────────────────────────────────────

const MODEL_META = {
  xgb_finance    : { label: "XGBoost Finance",    color: "#3b82f6", icon: "📈" },
  xgb_sentiment  : { label: "XGBoost +Sentiment", color: "#6366f1", icon: "📰" },
  lstm_finance   : { label: "LSTM Finance",        color: "#f59e0b", icon: "📈" },
  lstm_sentiment : { label: "LSTM +Sentiment",     color: "#a855f7", icon: "📰" },
};

// No "1D" option — the dataset is daily-granularity only (one row = one
// trading day), so an intraday period would have nothing to show.
const CHART_PERIODS = [
  { label: "1W",  days: 7    },
  { label: "1M",  days: 30   },
  { label: "3M",  days: 90   },
  { label: "1Y",  days: 365  },
  { label: "5Y",  days: 1825 },
  { label: "All", days: 3650 },
];

const OVERLAY_MODELS = [
  { key: "xgb_finance",    label: "XGB Finance"    },
  { key: "xgb_sentiment",  label: "XGB +Sentiment" },
  { key: "lstm_finance",   label: "LSTM Finance"   },
  { key: "lstm_sentiment", label: "LSTM +Sentiment"},
];

const TABS = [
  { key: "overview",   label: "Overview"   },
  { key: "technical",  label: "Technical"  },
  { key: "sentiment",  label: "Sentiment & Emotion"  },
  { key: "history",    label: "History"    },
  { key: "learn",      label: "Learn"      },
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

// Custom dot for BUY signal markers on chart
function BuyDot(props) {
  const { cx, cy } = props;
  if (!cx || !cy) return null;
  return (
    <g>
      <polygon
        points={`${cx},${cy - 8} ${cx - 5},${cy} ${cx + 5},${cy}`}
        fill="#22c55e"
        opacity={0.9}
      />
    </g>
  );
}

// Custom dot for SELL signal markers on chart
function SellDot(props) {
  const { cx, cy } = props;
  if (!cx || !cy) return null;
  return (
    <g>
      <polygon
        points={`${cx},${cy + 8} ${cx - 5},${cy} ${cx + 5},${cy}`}
        fill="#ef4444"
        opacity={0.9}
      />
    </g>
  );
}

// Custom tooltip for the price chart
function ChartTooltip({ active, payload, label, showSignal }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 text-xs shadow-xl min-w-32">
      <p className="text-gray-400 mb-2 font-medium">{label}</p>
      {d?.Close != null && (
        <p className="text-white font-semibold mb-1">
          Close: <span className="text-indigo-400">${d.Close.toFixed(2)}</span>
        </p>
      )}
      {d?.rsi != null && (
        <p className="text-white font-semibold mb-1">
          RSI: <span className="text-amber-400">{d.rsi.toFixed(1)}</span>
        </p>
      )}
      {d?.macd != null && (
        <p className="text-white font-semibold mb-1">
          MACD: <span className="text-green-400">{d.macd.toFixed(4)}</span>
        </p>
      )}
      {showSignal && d?.signal_label && (
        <p className={`font-bold mt-1 ${
          d.signal_label === "BUY" ? "text-green-400" : "text-red-400"
        }`}>
          {d.signal_label === "BUY" ? "▲ BUY signal" : "▼ SELL signal"}
        </p>
      )}
    </div>
  );
}

function ScoreBar({ label, score, color }) {
  const pct    = Math.min(Math.max(score ?? 50, 0), 100);
  const signal = pct >= 60 ? "Bullish" : pct <= 40 ? "Bearish" : "Neutral";
  const sc     = pct >= 60 ? "text-green-400" : pct <= 40 ? "text-red-400" : "text-gray-400";
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-400 font-medium">{label}</span>
        <div className="flex items-center gap-2">
          <span className={`font-medium ${sc}`}>{signal}</span>
          <span className="text-gray-500">{pct.toFixed(1)}%</span>
        </div>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-2">
        <div
          className="h-2 rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

function ModelSignalCard({ model }) {
  const meta  = MODEL_META[model.model_key] || {};
  const isBuy = model.signal_label === "BUY";
  const conf  = model.confidence ?? 50;
  return (
    <div className="bg-gray-800/60 rounded-xl p-3 border border-gray-700 flex flex-col gap-2">
      <div className="flex items-center gap-1.5">
        <span className="text-sm">{meta.icon}</span>
        <span className="text-xs font-medium" style={{ color: meta.color }}>
          {meta.label}
        </span>
      </div>
      <span className={`text-lg font-bold ${isBuy ? "text-green-400" : "text-red-400"}`}>
        {isBuy ? "▲ BUY" : "▼ SELL"}
      </span>
      <div className="w-full bg-gray-700 rounded-full h-1">
        <div
          className={`h-1 rounded-full ${isBuy ? "bg-green-500" : "bg-red-500"}`}
          style={{ width: `${conf}%` }}
        />
      </div>
      <span className="text-xs text-gray-500">{conf.toFixed(1)}% confidence</span>
    </div>
  );
}

function DecisionTimeline({ history, loading }) {
  if (loading) {
    return (
      <div className="flex gap-2 overflow-x-auto pb-2">
        {[1,2,3,4,5,6,7].map(i => (
          <div key={i} className="flex-shrink-0 w-32 h-32 bg-gray-800 rounded-xl animate-pulse" />
        ))}
      </div>
    );
  }

  if (!history || history.length === 0) {
    return <p className="text-gray-600 text-sm">No prediction history available for this ticker.</p>;
  }

  return (
    <div>
      {/* Horizontal scrollable day cards */}
      <div className="flex gap-2 overflow-x-auto pb-3 mb-1">
        {history.map((day, i) => {
          const isBuy = day.signal_label === "BUY";
          const isLast = i === history.length - 1;
          const dateObj = day.date ? new Date(day.date + "T00:00:00") : null;
          const weekday = dateObj ? dateObj.toLocaleDateString("en-US", { weekday: "short" }) : "—";
          const monthDay = dateObj ? dateObj.toLocaleDateString("en-US", { month: "short", day: "numeric" }) : day.date;

          return (
            <div key={day.date || i} className="flex items-center flex-shrink-0">
              <div
                className={`w-32 rounded-xl border p-3 flex flex-col gap-1.5 ${
                  isLast
                    ? isBuy
                      ? "bg-green-900/30 border-green-600 ring-1 ring-green-600/50"
                      : "bg-red-900/30 border-red-600 ring-1 ring-red-600/50"
                    : isBuy
                      ? "bg-green-900/10 border-green-800/60"
                      : "bg-red-900/10 border-red-800/60"
                }`}
              >
                <div className="text-xs text-gray-500">{weekday}</div>
                <div className="text-xs text-gray-400 -mt-1">{monthDay}</div>
                <div className={`text-base font-bold ${isBuy ? "text-green-400" : "text-red-400"}`}>
                  {isBuy ? "▲" : "▼"} {day.signal_label}
                </div>
                {day.confidence != null && (
                  <div className="text-xs text-gray-500">{day.confidence}% conf.</div>
                )}
                {isLast && (
                  <span className="text-xs font-medium text-indigo-400 mt-0.5">Latest</span>
                )}
              </div>

              {i < history.length - 1 && (
                <span className="text-gray-700 text-lg px-1 flex-shrink-0">→</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Reasons list, most recent first */}
      <div className="space-y-1.5 mt-3">
        {[...history].reverse().map((day, i) => (
          <div
            key={day.date || i}
            className="flex items-start gap-3 text-xs bg-gray-800/40 rounded-lg px-3 py-2"
          >
            <span className="text-gray-500 font-mono w-16 flex-shrink-0">{day.date?.slice(5) || "—"}</span>
            <span className={`font-semibold w-10 flex-shrink-0 ${day.signal_label === "BUY" ? "text-green-400" : "text-red-400"}`}>
              {day.signal_label}
            </span>
            <span className="text-gray-400 leading-relaxed">{day.reason}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrackRecordCard({ model }) {
  const meta = MODEL_META[model.model_key] || {};
  const recentAcc  = model.recent?.accuracy;
  const overallAcc = model.overall?.accuracy;
  const good = recentAcc != null && recentAcc >= 50;
  return (
    <div className="bg-gray-800/60 rounded-xl p-3 border border-gray-700 flex flex-col gap-2">
      <div className="flex items-center gap-1.5">
        <span className="text-sm">{meta.icon}</span>
        <span className="text-xs font-medium" style={{ color: meta.color }}>
          {meta.label}
        </span>
      </div>
      {recentAcc != null ? (
        <>
          <span className={`text-lg font-bold ${good ? "text-green-400" : "text-red-400"}`}>
            {recentAcc.toFixed(0)}%
          </span>
          <span className="text-xs text-gray-500">
            {model.recent.correct}/{model.recent.n} correct — last {model.recent.n}
          </span>
          <div className="w-full bg-gray-700 rounded-full h-1">
            <div
              className={`h-1 rounded-full ${good ? "bg-green-500" : "bg-red-500"}`}
              style={{ width: `${Math.min(recentAcc, 100)}%` }}
            />
          </div>
          <span className="text-xs text-gray-600">
            {overallAcc != null ? `${overallAcc.toFixed(1)}% over all ${model.overall.n} test-period predictions` : "—"}
          </span>
        </>
      ) : (
        <span className="text-xs text-gray-600">No test-period predictions for this ticker</span>
      )}
    </div>
  );
}

function TrackRecordPanel({ data, loading }) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[1,2,3,4].map(i => <div key={i} className="h-28 bg-gray-800 rounded-xl animate-pulse" />)}
      </div>
    );
  }
  if (!data?.models || data.models.length === 0) {
    return <p className="text-gray-600 text-sm">No track record available for this ticker yet.</p>;
  }
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {data.models.map(m => <TrackRecordCard key={m.model_key} model={m} />)}
    </div>
  );
}

function IndicatorCard({ signal }) {
  const c = {
    bullish: { bg: "bg-green-900/20", border: "border-green-800", dot: "bg-green-500", text: "text-green-400" },
    bearish: { bg: "bg-red-900/20",   border: "border-red-800",   dot: "bg-red-500",   text: "text-red-400"  },
    neutral: { bg: "bg-gray-800/40",  border: "border-gray-700",  dot: "bg-gray-500",  text: "text-gray-400" },
  }[signal.signal] || { bg: "bg-gray-800/40", border: "border-gray-700", dot: "bg-gray-500", text: "text-gray-400" };

  return (
    <div className={`rounded-xl p-4 border ${c.bg} ${c.border}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-white">{signal.name}</span>
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${c.dot}`} />
          <span className={`text-xs font-medium capitalize ${c.text}`}>{signal.signal}</span>
        </div>
      </div>
      {signal.value != null && (
        <p className="text-xs text-gray-500 mb-1.5">
          Value: <span className="text-gray-300 font-mono">
            {typeof signal.value === "number" ? signal.value.toFixed(3) : signal.value}
          </span>
        </p>
      )}
      <p className="text-xs text-gray-400 leading-relaxed">{signal.reason}</p>
    </div>
  );
}

function RiskMeterGauge({ explanation }) {
  if (!explanation) return null;
  const { risk_level, risk_score, risk_reason, risk_volatility_pct, risk_atr_pct } = explanation;
  if (!risk_level) return null;

  const config = {
    Low    : { color: "#22c55e", bg: "bg-green-900/20", border: "border-green-800", text: "text-green-400", pct: 20 },
    Medium : { color: "#f59e0b", bg: "bg-amber-900/20", border: "border-amber-800", text: "text-amber-400", pct: 55 },
    High   : { color: "#ef4444", bg: "bg-red-900/20",   border: "border-red-800",   text: "text-red-400",   pct: 88 },
  }[risk_level] || { color: "#6b7280", bg: "bg-gray-800/40", border: "border-gray-700", text: "text-gray-400", pct: 50 };

  const gaugePct = risk_score != null ? Math.min(Math.max(risk_score, 0), 100) : config.pct;

  return (
    <div className={`rounded-xl p-4 border ${config.bg} ${config.border}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-white">Investment Risk Level</span>
        <span className={`text-sm font-bold ${config.text}`}>{risk_level}</span>
      </div>

      {/* Gauge bar */}
      <div className="w-full bg-gray-800 rounded-full h-2.5 mb-1 relative overflow-hidden">
        <div
          className="h-2.5 rounded-full transition-all duration-500"
          style={{ width: `${gaugePct}%`, background: config.color }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-600 mb-3">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
      </div>

      {/* Contributing factors */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        {risk_volatility_pct != null && (
          <div className="bg-gray-800/60 rounded-lg px-3 py-2">
            <div className="text-xs text-gray-500">20-Day Volatility</div>
            <div className="text-sm font-mono text-gray-200">{risk_volatility_pct}% of price</div>
          </div>
        )}
        {risk_atr_pct != null && (
          <div className="bg-gray-800/60 rounded-lg px-3 py-2">
            <div className="text-xs text-gray-500">Avg. True Range</div>
            <div className="text-sm font-mono text-gray-200">{risk_atr_pct}% of price</div>
          </div>
        )}
      </div>

      <p className="text-xs text-gray-400 leading-relaxed">{risk_reason}</p>
    </div>
  );
}

function SentimentCard({ signal }) {
  const c = {
    positive: { bg: "bg-green-900/20", border: "border-green-800", text: "text-green-400", icon: "📈" },
    negative: { bg: "bg-red-900/20",   border: "border-red-800",   text: "text-red-400",   icon: "📉" },
    neutral:  { bg: "bg-gray-800/40",  border: "border-gray-700",  text: "text-gray-400",  icon: "➖" },
  }[signal.signal] || { bg: "bg-gray-800/40", border: "border-gray-700", text: "text-gray-400", icon: "➖" };

  return (
    <div className={`rounded-xl p-4 border ${c.bg} ${c.border}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span>{c.icon}</span>
          <span className="text-sm font-medium text-white">{signal.source}</span>
        </div>
        <span className={`text-xs font-semibold ${c.text} capitalize`}>{signal.signal}</span>
      </div>
      <div className="flex items-center gap-2 mb-2">
        <div className="flex-1 h-1.5 bg-gray-700 rounded-full">
          <div
            className={`h-1.5 rounded-full ${
              signal.signal === "positive" ? "bg-green-500" :
              signal.signal === "negative" ? "bg-red-500" : "bg-gray-500"
            }`}
            style={{ width: `${((signal.weight ?? 0.5) * 100).toFixed(0)}%` }}
          />
        </div>
        <span className="text-xs text-gray-500 font-mono">
          {signal.score >= 0 ? "+" : ""}{signal.score?.toFixed(3)}
        </span>
      </div>
      <p className="text-xs text-gray-400 leading-relaxed">{signal.reason}</p>
    </div>
  );
}

function AttentionChart({ attention }) {
  if (!attention || attention.length === 0) return null;
  const maxW = Math.max(...attention.map(d => d.weight));
  const focusDay = attention.reduce((a, b) => (b.weight > a.weight ? b : a), attention[0]);

  return (
    <div>
      <div className="space-y-2">
        {attention.map(d => {
          const isFocus = d.date === focusDay.date;
          const relPct = maxW > 0 ? (d.weight / maxW) * 100 : 0;
          return (
            <div key={d.date} className="flex items-center gap-3 text-xs">
              <span className={`font-mono w-20 flex-shrink-0 ${isFocus ? "text-purple-300 font-semibold" : "text-gray-500"}`}>
                {d.date}
              </span>
              <div className="flex-1 h-4 bg-gray-800 rounded overflow-hidden">
                <div
                  className={`h-4 rounded transition-all duration-500 ${isFocus ? "bg-purple-500" : "bg-purple-700/50"}`}
                  style={{ width: `${relPct}%` }}
                />
              </div>
              <span className="text-gray-400 font-mono w-16 text-right">{(d.weight * 100).toFixed(2)}%</span>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-gray-500 leading-relaxed mt-3">
        Bars are scaled relative to each other so small differences are visible — in absolute terms every day is
        close to the 10.00% uniform baseline (1 ÷ 10 days). This means the model spreads its attention fairly
        evenly across the window rather than fixating on one day; the most-weighted day here is{" "}
        <span className="text-purple-300 font-medium">{focusDay.date}</span> at {(focusDay.weight * 100).toFixed(2)}%.
      </p>
    </div>
  );
}

// Real per-ticker sentiment impact — computed entirely client-side from the
// 4 models' predictions already returned by /explain/{ticker} (no backend
// change needed). Compares the finance-only vs finance+sentiment prediction
// for each model family, in BUY-direction confidence terms so the delta is
// directly comparable regardless of which side each model landed on.
function SentimentImpactCard({ models, ticker }) {
  const byKey = Object.fromEntries((models || []).map(m => [m.model_key, m]));
  const pairs = [
    { label: "XGBoost",          fin: byKey.xgb_finance,  sent: byKey.xgb_sentiment  },
    { label: "LSTM+Transformer", fin: byKey.lstm_finance, sent: byKey.lstm_sentiment },
  ].filter(p => p.fin && p.sent);

  if (!pairs.length) return null;

  const buyConf = (m) => (m.signal === 1 ? m.confidence : 100 - m.confidence);

  return (
    <div className="card">
      <SectionTitle sub={`Real per-ticker comparison — how ${ticker}'s BUY-direction confidence changed when sentiment+emotion features were added`}>
        Sentiment Impact
      </SectionTitle>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {pairs.map(p => {
          const withoutC = buyConf(p.fin);
          const withC    = buyConf(p.sent);
          const delta    = withC - withoutC;
          return (
            <div key={p.label} className="bg-gray-800/40 rounded-xl p-4 border border-gray-700">
              <div className="text-sm font-medium text-white mb-3">{p.label}</div>
              <div className="flex items-center justify-between text-xs mb-2">
                <span className="text-gray-500">Without sentiment</span>
                <span className="font-mono text-gray-300">{withoutC.toFixed(1)}% BUY-lean</span>
              </div>
              <div className="flex items-center justify-between text-xs mb-3">
                <span className="text-gray-500">With sentiment</span>
                <span className="font-mono text-gray-200">{withC.toFixed(1)}% BUY-lean</span>
              </div>
              <div className={`text-center rounded-lg py-1.5 text-sm font-semibold ${
                delta > 0 ? "bg-green-900/30 text-green-400" : delta < 0 ? "bg-red-900/30 text-red-400" : "bg-gray-800 text-gray-400"
              }`}>
                {delta >= 0 ? "+" : ""}{delta.toFixed(1)}pp {delta >= 0 ? "improvement" : "reduction"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function NewsCard({ article }) {
  return (
    <a
      href={article.url} target="_blank" rel="noopener noreferrer"
      className="block bg-gray-800/40 hover:bg-gray-800 border border-gray-700
                 hover:border-gray-600 rounded-xl p-4 transition-all group"
    >
      <p className="text-sm text-gray-200 group-hover:text-white leading-snug mb-2 line-clamp-2">
        {article.title}
      </p>
      <div className="flex items-center gap-3 text-xs text-gray-600">
        <span>{article.date}</span>
        <span>·</span>
        <span className="text-gray-500">{article.source}</span>
        <span className="ml-auto text-indigo-500 group-hover:text-indigo-400">Read →</span>
      </div>
    </a>
  );
}

// Returns { color, arrow } for a BUY/SELL/HOLD signal label — keeps
// coloring consistent (green/red/amber) everywhere a signal is shown.
function signalStyle(signal) {
  if (signal === "BUY")  return { color: "text-green-400", arrow: "▲" };
  if (signal === "SELL") return { color: "text-red-400",   arrow: "▼" };
  return { color: "text-amber-400", arrow: "◆" }; // HOLD
}

function AgreementBanner({ explanation }) {
  if (!explanation) return null;
  const { agreement_level, overall_signal, overall_confidence, risk_level } = explanation;
  const styles = {
    Strong  : "bg-green-900/20 border-green-800",
    Moderate: "bg-amber-900/20 border-amber-800",
    Mixed   : "bg-gray-800 border-gray-700",
  };
  const riskStyles = {
    Low    : "text-green-400 border-green-800 bg-green-900/20",
    Medium : "text-amber-400 border-amber-800 bg-amber-900/20",
    High   : "text-red-400 border-red-800 bg-red-900/20",
  };
  const riskIcons = { Low: "●", Medium: "●●", High: "●●●" };
  return (
    <div className={`rounded-xl px-4 py-3 border flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 ${styles[agreement_level] || styles.Mixed}`}>
      <p className="text-sm text-gray-300 leading-relaxed">{explanation.summary}</p>
      <div className="sm:ml-4 flex-shrink-0 flex items-center gap-4">
        <div className="text-center">
          <div className={`text-lg font-bold ${signalStyle(overall_signal).color}`}>
            {signalStyle(overall_signal).arrow} {overall_signal}
          </div>
          <div className="text-xs text-gray-500">{overall_confidence}% avg</div>
          <div className="text-xs font-medium mt-0.5">{agreement_level} Agreement</div>
        </div>
        {risk_level && (
          <div className={`text-center border rounded-lg px-3 py-1.5 ${riskStyles[risk_level] || riskStyles.Medium}`}>
            <div className="text-xs font-bold tracking-tighter">{riskIcons[risk_level]}</div>
            <div className="text-xs font-semibold">{risk_level} Risk</div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function Detail() {
  const { ticker } = useParams();
  const navigate   = useNavigate();

  const [period,       setPeriod]       = useState(90);
  const [chartType,    setChartType]    = useState("candle");
  const [showSignals,  setShowSignals]  = useState(true);
  const [overlayModel, setOverlayModel] = useState("xgb_sentiment");
  const [activeTab,    setActiveTab]    = useState("overview");

  const { data: stockData,   loading: stockLoading  } = useApi(`/stock/${ticker}?days=${period}`, [ticker, period]);
  const { data: explanation, loading: explLoading   } = useApi(`/explain/${ticker}`, [ticker]);
  const { data: newsData,    loading: newsLoading   } = useApi(`/news/${ticker}?limit=8`, [ticker]);
  const { data: timelineData, loading: timelineLoading } = useApi(`/history/${ticker}?days=7`, [ticker]);
  const { data: trackRecordData, loading: trackRecordLoading } = useApi(`/accuracy-history/${ticker}`, [ticker]);

  const priceData = stockData?.data  || [];
  const articles  = newsData?.articles || [];

  const chartConfig = {
    candle: { key: "Close", label: "Price (Candlestick)", color: "#6366f1", format: v => `$${v?.toFixed(2)}` },
    rsi   : { key: "rsi",   label: "RSI (14)",        color: "#f59e0b", format: v => v?.toFixed(1)       },
    macd  : { key: "macd",  label: "MACD",            color: "#10b981", format: v => v?.toFixed(4)       },
  };
  const activeChart = chartConfig[chartType];

  // ── Merge signal overlay into price data ───────────────────────────────
  // Gets prediction data from the explanation object (already fetched)
  // and attaches signal_label to each price row by date
  const chartData = useMemo(() => {
    if (!priceData.length) return [];

    // Build a date → signal map from the models array in explanation
    const signalMap = {};
    if (explanation?.models) {
      // We use the overlayModel's signal on the latest date
      // For historical signals we'd need a separate endpoint — 
      // here we mark the most recent date with the current signal
      const modelPred = explanation.models.find(m => m.model_key === overlayModel);
      if (modelPred && priceData.length > 0) {
        const latestDate = priceData[priceData.length - 1]?.date;
        if (latestDate) signalMap[latestDate] = modelPred.signal_label;
      }
    }

    return priceData.map(row => ({
      ...row,
      signal_label : signalMap[row.date] || null,
      // For scatter overlay — only set value when there's a signal
      buy_marker   : signalMap[row.date] === "BUY"  ? row[activeChart.key] : null,
      sell_marker  : signalMap[row.date] === "SELL" ? row[activeChart.key] : null,
    }));
  }, [priceData, explanation, overlayModel, activeChart.key]);

  // ── Backtest-style: mark all prediction rows in the chart window ───────
  // Fetch predictions from the overview endpoint via explanation models
  const signalOverlayData = useMemo(() => {
    if (!showSignals || !priceData.length || !explanation?.models) return chartData;

    // Use the XGB sentiment predictions (most complete) to mark chart
    // The explanation gives us the latest signal — for historical we use
    // a simple rule: mark every 5th day alternating for visual demo
    // Real implementation would call /stock endpoint with signals merged
    return chartData;
  }, [chartData, showSignals, explanation, priceData]);

  const yDomain = useMemo(() => {
    if (!priceData.length) return ["auto", "auto"];
    // Candlestick mode needs the y-axis to cover the real High/Low range,
    // not just Close, otherwise wicks would clip at the chart edges.
    const keys = chartType === "candle" ? ["High", "Low"] : [activeChart.key];
    const vals = priceData.flatMap(d => keys.map(k => d[k])).filter(v => v != null);
    if (!vals.length) return ["auto", "auto"];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const pad = (max - min) * 0.05;
    return [min - pad, max + pad];
  }, [priceData, activeChart.key, chartType]);

  return (
    <div className="space-y-6">

      {/* ── Header ── */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <button
            onClick={() => navigate("/market")}
            className="text-xs text-gray-500 hover:text-gray-300 mb-2 flex items-center gap-1 transition-colors"
          >
            ← Back to Market
          </button>
          <h1 className="text-2xl font-bold text-white font-mono">{ticker}</h1>
          {companyName(ticker) && (
            <p className="text-sm text-gray-400">{companyName(ticker)}</p>
          )}
          <p className="text-sm text-gray-500 mt-0.5">AI Copilot Analysis — all 4 model variants</p>
        </div>
        {explanation && !explLoading && (
          <div className={`flex items-center gap-4 px-6 py-3 rounded-xl border ${
            explanation.overall_signal === "BUY"
              ? "bg-green-900/20 border-green-700"
              : explanation.overall_signal === "SELL"
                ? "bg-red-900/20 border-red-700"
                : "bg-amber-900/20 border-amber-700"
          }`}>
            <RadialProgress
              value={explanation.overall_confidence}
              color={explanation.overall_signal === "BUY" ? "#22c55e" : explanation.overall_signal === "SELL" ? "#ef4444" : "#f59e0b"}
              label="confidence"
            />
            <div className="text-center">
              <div className={`text-2xl font-bold ${signalStyle(explanation.overall_signal).color}`}>
                {signalStyle(explanation.overall_signal).arrow} {explanation.overall_signal}
              </div>
              <div className="text-xs text-gray-500 mt-0.5">{explanation.agreement_level} Agreement</div>
            </div>
          </div>
        )}
      </div>

      {/* ── Tab bar ── */}
      <div className="flex gap-1 border-b border-gray-800 overflow-x-auto">
        {TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === t.key
                ? "border-indigo-500 text-white"
                : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ══════════════════════════ OVERVIEW TAB ══════════════════════════ */}
      {activeTab === "overview" && <>

      {/* ── AI summary banner ── */}
      {explLoading
        ? <div className="h-20 bg-gray-900 rounded-xl border border-gray-800 animate-pulse" />
        : <AgreementBanner explanation={explanation} />
      }

      {/* ══ Row 1 — Chart + Signal contribution ═══════════════════════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ── Price Chart with Signal Overlay ── */}
        <div className="lg:col-span-2 card">
          <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
            <SectionTitle>{ticker} Price Chart</SectionTitle>
            <div className="flex gap-2 flex-wrap">
              {/* Chart type */}
              <div className="flex rounded-lg overflow-hidden border border-gray-700 text-xs">
                {Object.entries(chartConfig).map(([k]) => (
                  <button key={k} onClick={() => setChartType(k)}
                    className={`px-2.5 py-1.5 font-medium transition-colors ${
                      chartType === k ? "bg-indigo-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                    }`}
                  >{k.toUpperCase()}</button>
                ))}
              </div>
              {/* Period */}
              <div className="flex rounded-lg overflow-hidden border border-gray-700 text-xs">
                {CHART_PERIODS.map(p => (
                  <button key={p.label} onClick={() => setPeriod(p.days)}
                    className={`px-2.5 py-1.5 font-medium transition-colors ${
                      period === p.days ? "bg-indigo-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                    }`}
                  >{p.label}</button>
                ))}
              </div>
            </div>
          </div>

          {/* ── Signal overlay controls (RSI/MACD views only — the candlestick
               already shows real up/down days directly via candle color) ── */}
          {chartType !== "candle" && (
          <div className="flex items-center gap-3 mb-4 flex-wrap">
            <label className="flex items-center gap-2 cursor-pointer">
              <div
                onClick={() => setShowSignals(s => !s)}
                className={`w-8 h-4 rounded-full transition-colors relative cursor-pointer ${
                  showSignals ? "bg-indigo-600" : "bg-gray-700"
                }`}
              >
                <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${
                  showSignals ? "left-4" : "left-0.5"
                }`} />
              </div>
              <span className="text-xs text-gray-400">Show signals</span>
            </label>

            {showSignals && (
              <select
                className="input text-xs py-1"
                value={overlayModel}
                onChange={e => setOverlayModel(e.target.value)}
              >
                {OVERLAY_MODELS.map(m => (
                  <option key={m.key} value={m.key}>{m.label}</option>
                ))}
              </select>
            )}

            {showSignals && (
              <div className="flex items-center gap-3 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <span className="text-green-400 font-bold">▲</span> BUY signal
                </span>
                <span className="flex items-center gap-1">
                  <span className="text-red-400 font-bold">▼</span> SELL signal
                </span>
              </div>
            )}
          </div>
          )}

          {stockLoading ? (
            <div className="h-52 bg-gray-800 rounded-lg animate-pulse" />
          ) : priceData.length === 0 ? (
            <div className="h-52 flex items-center justify-center text-gray-600 text-sm">
              No price data available for {ticker}
            </div>
          ) : chartType === "candle" ? (
            <CandlestickChart data={priceData} yDomain={yDomain} />
          ) : (
            <ResponsiveContainer width="100%" height={240}>
              <ComposedChart data={signalOverlayData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "#6b7280", fontSize: 10 }}
                  tickLine={false}
                  axisLine={{ stroke: "#374151" }}
                  interval={Math.floor(signalOverlayData.length / 6)}
                  tickFormatter={d => d?.slice(5)}
                />
                <YAxis
                  domain={yDomain}
                  tick={{ fill: "#6b7280", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={activeChart.format}
                  width={65}
                />
                <Tooltip content={<ChartTooltip showSignal={showSignals} />} />

                {/* RSI reference lines */}
                {chartType === "rsi" && <>
                  <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="4 2" strokeWidth={1}
                    label={{ value: "70", fill: "#ef4444", fontSize: 9, position: "insideTopRight" }} />
                  <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="4 2" strokeWidth={1}
                    label={{ value: "30", fill: "#22c55e", fontSize: 9, position: "insideBottomRight" }} />
                </>}
                {chartType === "macd" && (
                  <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="4 2" strokeWidth={1} />
                )}

                {/* Main price/indicator line */}
                <Line
                  type="monotone"
                  dataKey={activeChart.key}
                  stroke={activeChart.color}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: activeChart.color }}
                />

                {/* BUY signal markers */}
                {showSignals && (
                  <Scatter dataKey="buy_marker" shape={<BuyDot />} legendType="none" />
                )}

                {/* SELL signal markers */}
                {showSignals && (
                  <Scatter dataKey="sell_marker" shape={<SellDot />} legendType="none" />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* ── Signal contribution ── */}
        <div className="card flex flex-col gap-5">
          <SectionTitle sub="How each layer contributes to the signal">
            Signal Contribution
          </SectionTitle>

          {explLoading ? (
            <div className="space-y-4">
              {[1,2,3].map(i => <div key={i} className="h-8 bg-gray-800 rounded animate-pulse" />)}
            </div>
          ) : explanation ? (
            <>
              <ScoreBar label="Technical Indicators" score={explanation.technical_score}  color="#6366f1" />
              <ScoreBar label="Sentiment (GDELT+WSB)" score={explanation.sentiment_score} color="#10b981" />
              <ScoreBar label="Emotion (WSB/GoEmotions)" score={explanation.emotion_score} color="#ec4899" />
              <ScoreBar label="Model Confidence"      score={explanation.model_score}     color="#f59e0b" />

              <div className="border-t border-gray-800 pt-4">
                <p className="section-title">4 Model Signals</p>
                <div className="grid grid-cols-2 gap-2">
                  {explanation.models?.map(m => (
                    <ModelSignalCard key={m.model_key} model={m} />
                  ))}
                </div>
              </div>
            </>
          ) : (
            <p className="text-gray-600 text-sm text-center py-8">No explanation data</p>
          )}
        </div>
      </div>

      {/* ── Risk Assessment ── */}
      <div className="card">
        <SectionTitle sub="How volatile and uncertain this prediction is">
          Risk Assessment
        </SectionTitle>
        {explLoading ? (
          <div className="h-40 bg-gray-800 rounded-xl animate-pulse" />
        ) : explanation?.risk_level ? (
          <RiskMeterGauge explanation={explanation} />
        ) : (
          <p className="text-gray-600 text-sm">No risk data available.</p>
        )}
      </div>

      </>}

      {/* ══════════════════════════ TECHNICAL TAB ══════════════════════════ */}
      {activeTab === "technical" && <>
      <div className="card">
        <SectionTitle sub="Rule-based analysis of each indicator">
          Technical Indicator Analysis
        </SectionTitle>
        {explLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[1,2,3,4].map(i => <div key={i} className="h-24 bg-gray-800 rounded-xl animate-pulse" />)}
          </div>
        ) : explanation?.indicator_signals?.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {explanation.indicator_signals.map(s => <IndicatorCard key={s.name} signal={s} />)}
          </div>
        ) : (
          <p className="text-gray-600 text-sm">No indicator data available.</p>
        )}
      </div>

      {explLoading ? (
        <div className="card"><div className="h-40 bg-gray-800 rounded-xl animate-pulse" /></div>
      ) : explanation?.lstm_attention?.length > 0 ? (
        <div className="card">
          <SectionTitle sub="LSTM+Transformer's own internal attention mechanism — which of the last 10 trading days it weighted most heavily for this prediction. This is the model's exact reasoning, not a post-hoc approximation like SHAP/LIME.">
            AI Model Attention — Last 10 Trading Days
          </SectionTitle>
          <AttentionChart attention={explanation.lstm_attention} />
        </div>
      ) : explanation && !explLoading ? (
        <div className="card">
          <SectionTitle sub="LSTM+Transformer's own internal attention mechanism — which of the last 10 trading days it weighted most heavily for this prediction.">
            AI Model Attention — Last 10 Trading Days
          </SectionTitle>
          <p className="text-gray-600 text-sm">
            No attention data available for {ticker}'s latest prediction date.
          </p>
        </div>
      ) : null}
      </>}

      {/* ══════════════════════════ SENTIMENT TAB ══════════════════════════ */}
      {activeTab === "sentiment" && <>

      {explanation?.models?.length > 0 && (
        <SentimentImpactCard models={explanation.models} ticker={ticker} />
      )}

      {explanation?.sentiment_signals?.length > 0 && (
        <div className="card">
          <SectionTitle sub="News and social media sentiment from GDELT and Reddit WSB">
            Sentiment Analysis
          </SectionTitle>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {explanation.sentiment_signals.map(s => <SentimentCard key={s.source} signal={s} />)}
          </div>
        </div>
      )}

      {explanation?.emotion_signals?.length > 0 ? (
        <div className="card">
          <SectionTitle sub="Multi-label emotion detected in Reddit/WSB posts (GoEmotions) — distinct from sentiment polarity above">
            Emotion Impact
          </SectionTitle>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {explanation.emotion_signals.map(s => <SentimentCard key={s.source} signal={s} />)}
          </div>
        </div>
      ) : explanation && !explLoading ? (
        <div className="card">
          <SectionTitle sub="Multi-label emotion detected in Reddit/WSB posts (GoEmotions) — distinct from sentiment polarity above">
            Emotion Impact
          </SectionTitle>
          <p className="text-gray-600 text-sm">
            No WSB posts mentioning {ticker} on the latest prediction date — emotion is only available on days with Reddit coverage.
          </p>
        </div>
      ) : null}

      <div className="card">
        <SectionTitle sub="Latest headlines from GDELT news database">
          Recent News — {ticker}
        </SectionTitle>
        {newsLoading ? (
          <div className="space-y-3">
            {[1,2,3,4].map(i => <div key={i} className="h-16 bg-gray-800 rounded-xl animate-pulse" />)}
          </div>
        ) : articles.length === 0 ? (
          <p className="text-gray-600 text-sm">
            No news articles found for {ticker}.
            Run <code className="text-indigo-400">fetch_news.py</code> to populate the news database.
          </p>
        ) : (
          <div className="space-y-2">
            {articles.map((a, i) => <NewsCard key={i} article={a} />)}
          </div>
        )}
      </div>

      </>}

      {/* ══════════════════════════ HISTORY TAB ════════════════════════════ */}
      {activeTab === "history" && (
      <div className="space-y-6">
        <div className="card">
          <SectionTitle sub={`Was each model actually right about ${ticker}? Real accuracy over its test-period predictions for this specific stock — not the dataset-wide average.`}>
            Track Record
          </SectionTitle>
          <TrackRecordPanel data={trackRecordData} loading={trackRecordLoading} />
        </div>

        <div className="card">
          <SectionTitle sub="How the signal for this stock has changed over the last 7 trading days">
            Decision Timeline
          </SectionTitle>
          <DecisionTimeline history={timelineData?.history} loading={timelineLoading} />
        </div>
      </div>
      )}

      {/* ══════════════════════════ LEARN TAB ══════════════════════════════ */}
      {activeTab === "learn" && <GlossarySection />}

    </div>
  );
}

// ── Glossary Component ────────────────────────────────────────────────────────

const GLOSSARY_TERMS = [
  {
    term : "Decision Timeline",
    short: "Shows how a stock's BUY/SELL signal has changed over the last 7 trading days.",
    detail: "Each day's signal is shown with the reason behind it — for example, an RSI oversold reading or a MACD crossover. Watching the timeline helps you see whether a signal is brand new or has been consistent for several days. A signal that just flipped is often less certain than one that has held steady all week.",
  },
  {
    term : "Investment Risk Level",
    short: "How volatile and uncertain a stock's prediction is — Low, Medium, or High.",
    detail: "Combines the stock's price volatility, Average True Range (ATR), how close the model's confidence is to 50% (a coin flip), and whether the 4 models disagree with each other. A BUY signal with High risk means the prediction could be right, but the stock price swings a lot — so the outcome is less certain either way. Low risk means the stock is stable and the model is confident.",
  },
  {
    term : "RSI (Relative Strength Index)",
    short: "Momentum indicator — measures how fast a stock is moving up or down.",
    detail: "RSI ranges from 0 to 100. Above 70 = overbought (price may drop soon). Below 30 = oversold (price may rebound). Between 30–70 = neutral. Our model uses RSI(14) — calculated over the last 14 trading days.",
  },
  {
    term : "MACD (Moving Average Convergence Divergence)",
    short: "Trend indicator — shows when a stock momentum is shifting.",
    detail: "MACD = EMA(12) minus EMA(26). When MACD crosses above the signal line → bullish momentum. When it crosses below → bearish. The histogram shows the gap between MACD and its signal line.",
  },
  {
    term : "Bollinger Bands",
    short: "Volatility indicator — shows whether a price is high or low relative to recent history.",
    detail: "Three lines: upper band, middle (20-day SMA), lower band. Price near the lower band = potentially oversold. Price near the upper band = potentially overbought. Wide bands = high volatility. Narrow bands = consolidation.",
  },
  {
    term : "SMA / EMA (Moving Averages)",
    short: "Trend-following indicators — smooth out price noise to show direction.",
    detail: "SMA (Simple Moving Average) weights all days equally. EMA (Exponential Moving Average) weights recent days more. When short-term average crosses above long-term → uptrend. Below → downtrend.",
  },
  {
    term : "BUY / SELL Signal",
    short: "The model prediction — will this stock price be higher or lower tomorrow?",
    detail: "BUY (1) = model predicts tomorrow close will be higher than today. SELL (0) = model predicts it will be lower. Confidence % shows how certain the model is. This is directional prediction only, not a price target.",
  },
  {
    term : "Model Confidence",
    short: "How certain the model is about its BUY or SELL prediction.",
    detail: "Shown as a percentage. Closer to 50% = model is uncertain. Above 65% = stronger signal. Never treat any confidence as a guarantee of profit.",
  },
  {
    term : "Sentiment Score (VADER / GDELT)",
    short: "How positive or negative news coverage is for a stock on a given day.",
    detail: "GDELT collects global news headlines. VADER scores each headline from -1 (very negative) to +1 (very positive). We average all headlines per ticker per day. Positive sentiment does not always mean the price goes up — it is one signal among many.",
  },
  {
    term : "WSB Sentiment (Reddit WallStreetBets)",
    short: "Social media mood from the WallStreetBets Reddit community.",
    detail: "WSB is a Reddit community known for retail trading. High positive WSB sentiment = retail investors are bullish. This captures hype and fear, which sometimes predicts short-term price moves.",
  },
  {
    term : "AUC-ROC",
    short: "A model quality metric — how well the model separates BUY from SELL days.",
    detail: "Ranges from 0.5 (random) to 1.0 (perfect). Our models score 50–52%. This is consistent with academic research — stock markets are genuinely hard to predict consistently.",
  },
  {
    term : "Strong / Moderate / Mixed Agreement",
    short: "How much the 4 model variants agree on the signal for a stock.",
    detail: "Strong = all 4 models agree (most reliable). Moderate = 3 of 4 agree. Mixed = 2 of 4 agree (treat with caution). Disagreement usually means the signal is genuinely ambiguous.",
  },
  {
    term : "LSTM Attention Weights",
    short: "Which of the last 10 trading days the LSTM model focused on most when making its prediction.",
    detail: "The LSTM+Transformer model has a built-in attention layer that scores each of the 10 days in its lookback window before combining them into a prediction. Unlike SHAP or LIME (which approximate a model's reasoning after the fact), these weights are the model's own exact internal calculation. In this project the weights come out close to uniform (~10% per day) rather than sharply peaked on one day — a real, honest finding: the model relies on the whole 10-day window fairly evenly rather than any single standout day. This mirrors a well-documented result in attention-mechanism research (attention weights don't always concentrate sharply, and don't always agree with other explanation methods).",
  },
];

function GlossarySection() {
  const [open,     setOpen]     = useState(false);
  const [expanded, setExpanded] = useState(null);

  return (
    <div className="card">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between text-left"
      >
        <div>
          <h2 className="text-base font-semibold text-white">
            📖 Learn — Indicator Glossary
          </h2>
          <p className="text-xs text-gray-500 mt-0.5">
            New to trading? Click to learn what RSI, MACD, sentiment and other terms mean
          </p>
        </div>
        <span className={`text-gray-500 transition-transform duration-200 ${open ? "rotate-180" : ""}`}>
          ▼
        </span>
      </button>

      {open && (
        <div className="mt-5 space-y-2">
          {GLOSSARY_TERMS.map((item, i) => (
            <div key={i} className="border border-gray-800 rounded-xl overflow-hidden">
              <button
                onClick={() => setExpanded(expanded === i ? null : i)}
                className="w-full flex items-center justify-between px-4 py-3
                           hover:bg-gray-800/40 transition-colors text-left"
              >
                <div>
                  <span className="text-sm font-medium text-white">{item.term}</span>
                  <p className="text-xs text-gray-500 mt-0.5">{item.short}</p>
                </div>
                <span className={`text-gray-600 ml-4 flex-shrink-0 transition-transform duration-150 ${
                  expanded === i ? "rotate-180" : ""
                }`}>▼</span>
              </button>
              {expanded === i && (
                <div className="px-4 pb-4 bg-gray-800/20">
                  <p className="text-sm text-gray-300 leading-relaxed">{item.detail}</p>
                </div>
              )}
            </div>
          ))}
          <p className="text-xs text-gray-600 pt-2 text-center">
            NuroQuant is a learning tool. All signals are model predictions — not financial advice.
          </p>
        </div>
      )}
    </div>
  );
}
