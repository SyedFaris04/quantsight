/**
 * frontend/src/pages/Dashboard.jsx
 * Home page — market summary at a glance.
 *
 * ALL DATA IS REAL:
 *   - KPI cards          → /dashboard kpis (computed from /overview)
 *   - Top Opportunities  → /dashboard top_opportunities
 *   - Market News        → /dashboard news (recent GDELT headlines)
 *   - Market Sentiment   → /market-sentiment (real-time VADER over the
 *                           CURRENT news feed — genuinely computed, not
 *                           precomputed into a file)
 *   - AI Insight         → one sentence assembled client-side from the
 *                           already-fetched real KPI numbers, no
 *                           text-generation backend involved
 *
 * No mockup data. No sector claims (no sector data exists in this project).
 */

import { useNavigate } from "react-router-dom";
import { AreaChart, Area, ResponsiveContainer, YAxis } from "recharts";
import { useApi } from "../hooks/useApi";

// ── Small helpers ──────────────────────────────────────────────────────────────
function signalColor(signal) {
  if (signal === "BUY")  return "text-green-400";
  if (signal === "SELL") return "text-red-400";
  return "text-amber-400"; // HOLD
}
function signalArrow(signal) {
  if (signal === "BUY")  return "▲";
  if (signal === "SELL") return "▼";
  return "◆";
}
function agreementColor(level) {
  if (level === "Strong")   return "text-green-400 bg-green-900/20 border-green-800";
  if (level === "Moderate") return "text-amber-400 bg-amber-900/20 border-amber-800";
  return "text-gray-400 bg-gray-800 border-gray-700";
}
function riskColor(level) {
  if (level === "Low")  return "text-green-400";
  if (level === "High") return "text-red-400";
  return "text-amber-400";
}
function riskDots(level) {
  return level === "Low" ? "●" : level === "High" ? "●●●" : "●●";
}

// ── KPI card ───────────────────────────────────────────────────────────────────
function KpiCard({ label, value, sub, accent }) {
  const accentBar = {
    indigo: "bg-indigo-500",
    green:  "bg-green-500",
    amber:  "bg-amber-500",
    blue:   "bg-blue-500",
  }[accent] || "bg-indigo-500";

  return (
    <div className="relative bg-gray-900 border border-gray-800 rounded-xl p-4 overflow-hidden">
      <div className={`absolute left-0 top-0 bottom-0 w-1 ${accentBar}`} />
      <div className="pl-2">
        <div className="text-3xl font-bold text-white">{value}</div>
        <div className="text-sm font-medium text-gray-300 mt-1">{label}</div>
        {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
      </div>
    </div>
  );
}

// ── Market Sentiment panel ───────────────────────────────────────────────────
function MarketSentimentPanel() {
  const { data, loading } = useApi("/market-sentiment");

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="mb-3">
        <h2 className="text-base font-semibold text-white">Market Sentiment</h2>
        <p className="text-xs text-gray-500">Real-time VADER score over current news headlines</p>
      </div>

      {loading ? (
        <div className="h-32 bg-gray-800 rounded-lg animate-pulse" />
      ) : !data || data.article_count === 0 ? (
        <p className="text-sm text-gray-600 py-6 text-center">No recent news to score.</p>
      ) : (
        <>
          {data.trend?.length > 1 && (
            <div className="h-16 -mx-1 mb-3">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.trend}>
                  <YAxis hide domain={["dataMin", "dataMax"]} />
                  <defs>
                    <linearGradient id="sentimentFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6366f1" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area
                    type="monotone"
                    dataKey="avg_compound"
                    stroke="#818cf8"
                    strokeWidth={1.5}
                    fill="url(#sentimentFill)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-green-400 font-bold text-lg">{data.positive_pct}%</div>
              <div className="text-xs text-gray-500">Positive</div>
            </div>
            <div>
              <div className="text-gray-400 font-bold text-lg">{data.neutral_pct}%</div>
              <div className="text-xs text-gray-500">Neutral</div>
            </div>
            <div>
              <div className="text-red-400 font-bold text-lg">{data.negative_pct}%</div>
              <div className="text-xs text-gray-500">Negative</div>
            </div>
          </div>
          <p className="text-xs text-gray-600 mt-3 text-center">
            {data.article_count.toLocaleString()} headlines · last {data.days} days
          </p>
        </>
      )}
    </div>
  );
}

// ── AI Insight card — real sentence from already-fetched KPIs ──────────────
function AiInsightCard({ kpis, loading }) {
  const insight = kpis
    ? `${kpis.buy_signals} of ${kpis.total_tickers} tickers show a BUY signal today, with ` +
      `${kpis.strong_agreement} in strong 4-model agreement. Average model confidence across ` +
      `all stocks is ${kpis.avg_confidence}%.`
    : null;

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-start gap-3">
        <span className="w-8 h-8 rounded-lg bg-indigo-500/20 border border-indigo-600/40
                         flex items-center justify-center text-base flex-shrink-0">
          🤖
        </span>
        <div className="flex-1 min-w-0">
          <h2 className="text-base font-semibold text-white mb-1">AI Insight</h2>
          {loading ? (
            <div className="h-10 bg-gray-800 rounded animate-pulse" />
          ) : (
            <p className="text-sm text-gray-400 leading-relaxed">{insight}</p>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main ────────────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const navigate = useNavigate();
  const { data, loading, error } = useApi("/dashboard");

  const kpis = data?.kpis;
  const opps = data?.top_opportunities || [];
  const news = data?.news || [];

  return (
    <div className="space-y-6">

      {/* ── Header ── */}
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">
            Good day, Investor <span className="text-indigo-400">👋</span>
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Your AI market overview — {loading ? "loading…" : `${kpis?.total_tickers ?? 0} stocks analysed`}
          </p>
        </div>
        <button
          onClick={() => navigate("/market")}
          className="text-sm font-medium text-indigo-400 hover:text-indigo-300 border border-indigo-800
                     bg-indigo-900/20 rounded-lg px-4 py-2 transition-colors"
        >
          View Full Market →
        </button>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg px-4 py-3 text-sm text-red-400">
          Could not load dashboard data. Is the backend running on port 8000?
        </div>
      )}

      {/* ── KPI row ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {loading ? (
          [1, 2, 3, 4].map(i => <div key={i} className="h-24 bg-gray-900 rounded-xl animate-pulse" />)
        ) : (
          <>
            <KpiCard label="Total Tickers"    value={kpis?.total_tickers ?? 0}    sub="US stocks analysed"         accent="blue" />
            <KpiCard label="BUY Signals"      value={kpis?.buy_signals ?? 0}      sub={`${kpis?.sell_signals ?? 0} SELL · ${kpis?.hold_signals ?? 0} HOLD`} accent="green" />
            <KpiCard label="Strong Agreement" value={kpis?.strong_agreement ?? 0} sub="all 4 models agree"          accent="indigo" />
            <KpiCard label="Avg Confidence"   value={`${kpis?.avg_confidence ?? 0}%`} sub="across all stocks"       accent="amber" />
          </>
        )}
      </div>

      {/* ── Row: Top Opportunities + Market Sentiment ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Top Opportunities */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-semibold text-white">Top Opportunities</h2>
              <p className="text-xs text-gray-500">Highest-confidence BUY signals right now</p>
            </div>
            <span className="text-xs text-indigo-400 bg-indigo-900/20 border border-indigo-800 rounded-full px-2 py-0.5">
              Live signals
            </span>
          </div>

          {loading ? (
            <div className="space-y-2">{[1,2,3,4,5].map(i => <div key={i} className="h-12 bg-gray-800 rounded-lg animate-pulse" />)}</div>
          ) : opps.length === 0 ? (
            <p className="text-sm text-gray-600 py-6 text-center">No strong BUY opportunities right now.</p>
          ) : (
            <div className="space-y-2">
              {opps.map((o, i) => (
                <button
                  key={o.ticker}
                  onClick={() => navigate(`/detail/${o.ticker}`)}
                  className="w-full flex items-center gap-3 bg-gray-800/50 hover:bg-gray-800
                             border border-gray-800 hover:border-gray-700 rounded-lg px-3 py-2.5
                             transition-colors text-left"
                >
                  <span className="text-indigo-400 font-bold text-sm w-5 flex-shrink-0">{i + 1}</span>
                  <div className="flex-1 min-w-0">
                    <div className="font-semibold text-white text-sm">{o.ticker}</div>
                    <div className="text-xs text-gray-500">
                      <span className={`inline-block px-1.5 py-0.5 rounded-full border text-[10px] ${agreementColor(o.agreement_level)}`}>
                        {o.agreement_level}
                      </span>
                      <span className={`ml-2 ${riskColor(o.risk_level)}`}>{riskDots(o.risk_level)} {o.risk_level} risk</span>
                    </div>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <div className={`font-bold text-sm ${signalColor(o.overall_signal)}`}>
                      {signalArrow(o.overall_signal)} {o.overall_signal}
                    </div>
                    <div className="text-xs text-gray-500">{o.confidence}% conf.</div>
                  </div>
                  <span className="text-gray-600 text-sm flex-shrink-0">→</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <MarketSentimentPanel />
      </div>

      {/* ── Row: Market News + AI Insight ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Market News */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-semibold text-white">Market News</h2>
              <p className="text-xs text-gray-500">Recent headlines across all stocks (GDELT)</p>
            </div>
          </div>

          {loading ? (
            <div className="space-y-3">{[1,2,3,4].map(i => <div key={i} className="h-14 bg-gray-800 rounded-lg animate-pulse" />)}</div>
          ) : news.length === 0 ? (
            <p className="text-sm text-gray-600 py-6 text-center">No recent news available.</p>
          ) : (
            <div className="space-y-3">
              {news.map((article, i) => (
                <a
                  key={i}
                  href={article.url || "#"}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block border-b border-gray-800 last:border-0 pb-3 last:pb-0
                             hover:bg-gray-800/30 -mx-2 px-2 rounded transition-colors"
                >
                  <div className="flex items-start gap-2">
                    {article.ticker && (
                      <span className="text-xs font-bold text-indigo-400 bg-indigo-900/20 border border-indigo-800
                                       rounded px-1.5 py-0.5 flex-shrink-0 mt-0.5">
                        {article.ticker}
                      </span>
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-gray-200 leading-snug line-clamp-2">{article.title}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-xs text-gray-600">{String(article.date).slice(0, 10)}</span>
                        {article.source && <span className="text-xs text-gray-600">· {article.source}</span>}
                      </div>
                    </div>
                  </div>
                </a>
              ))}
            </div>
          )}
        </div>

        <AiInsightCard kpis={kpis} loading={loading} />
      </div>

      {/* Disclaimer */}
      <p className="text-xs text-gray-600 text-center">
        Signals are model predictions for educational decision support — not financial advice.
      </p>
    </div>
  );
}
