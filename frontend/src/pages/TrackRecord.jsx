/**
 * frontend/src/pages/TrackRecord.jsx
 * ─────────────────────────────────────────────────────────────────
 * Live prediction track record — forward-looking, not backtested.
 * Every trading day, a scheduled job (see backend/prediction_tracker.py
 * + .github/workflows/daily-predictions.yml) logs the live model's
 * signal for every ticker, then checks it the next trading day against
 * what the price actually did.
 *
 * Writes only ever happen server-side via Supabase's service role key
 * (bypasses RLS) — this page, like everyone else, only ever reads. No
 * client, including this one, can edit a row, which is what makes the
 * number here mean something.
 *
 * Looking for one specific stock's historical (backtested) accuracy
 * instead? That's the Detail page → History tab → Track Record panel.
 * ─────────────────────────────────────────────────────────────────
 */

import { useApi } from "../hooks/useApi";
import { companyName } from "../data/companyNames";

function KpiCard({ label, value, sub, accent }) {
  const accentBar = {
    indigo: "bg-indigo-500",
    green:  "bg-green-500",
    amber:  "bg-amber-500",
    gray:   "bg-gray-600",
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

function StatusBadge({ row }) {
  if (!row.resolved) {
    return (
      <span className="text-xs px-2 py-0.5 rounded-full border border-gray-700 bg-gray-800 text-gray-500">
        Pending
      </span>
    );
  }
  return row.correct ? (
    <span className="text-xs px-2 py-0.5 rounded-full border border-green-800 bg-green-900/30 text-green-400">
      ✓ Correct
    </span>
  ) : (
    <span className="text-xs px-2 py-0.5 rounded-full border border-red-800 bg-red-900/30 text-red-400">
      ✗ Wrong
    </span>
  );
}

export default function TrackRecord() {
  const { data, loading, error } = useApi("/live-track-record");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white flex items-center gap-2">
          <span>🎯</span> Live Track Record
        </h1>
        <p className="text-sm text-gray-500 mt-1 max-w-2xl">
          Every trading day, QuantSight logs its live model's prediction for every
          ticker, then checks the next trading day whether it was actually right.
          Nothing here is backtested or cherry-picked — predictions are locked in
          before the outcome is known, and only the server can write a result.
        </p>
      </div>

      {error && (
        <div className="card text-sm text-gray-500">
          Couldn't load the track record right now. Try again shortly.
        </div>
      )}

      {!error && !data?.available && !loading && (
        <div className="card text-center py-12">
          <div className="text-4xl mb-3">📡</div>
          <p className="text-gray-400 text-sm">
            The live tracker hasn't logged anything yet — check back after the next
            trading day's market close.
          </p>
        </div>
      )}

      {(loading || data?.available) && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <KpiCard
              label="Live Accuracy"
              value={loading ? "…" : data.accuracy_pct != null ? `${data.accuracy_pct}%` : "—"}
              sub="of resolved predictions"
              accent="indigo"
            />
            <KpiCard
              label="Resolved"
              value={loading ? "…" : data.total_resolved}
              sub="checked against real outcomes"
              accent="green"
            />
            <KpiCard
              label="Pending"
              value={loading ? "…" : data.total_pending}
              sub="waiting on next trading day"
              accent="amber"
            />
            <KpiCard
              label="Total Logged"
              value={loading ? "…" : data.total_logged}
              sub="all-time"
              accent="gray"
            />
          </div>

          <div className="card !p-0 overflow-hidden">
            <div className="px-5 pt-5 pb-3">
              <p className="section-title mb-0">Recent Predictions</p>
            </div>
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Predicted</th>
                    <th>Signal</th>
                    <th>Confidence</th>
                    <th>Actual</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {loading ? (
                    [1, 2, 3, 4, 5].map(i => (
                      <tr key={i}>
                        <td colSpan={6}><div className="h-8 bg-gray-800 rounded animate-pulse my-1" /></td>
                      </tr>
                    ))
                  ) : data.recent.length === 0 ? (
                    <tr><td colSpan={6} className="text-gray-600 text-sm py-6 text-center">No predictions logged yet.</td></tr>
                  ) : (
                    data.recent.map((row, i) => (
                      <tr key={i}>
                        <td>
                          <span className="font-mono font-medium text-white">{row.ticker}</span>
                          {companyName(row.ticker) && (
                            <div className="text-xs text-gray-500">{companyName(row.ticker)}</div>
                          )}
                        </td>
                        <td className="text-gray-400 text-sm">{row.predicted_date}</td>
                        <td>
                          <span className={`text-xs font-bold ${row.predicted_signal === "BUY" ? "text-green-400" : "text-red-400"}`}>
                            {row.predicted_signal === "BUY" ? "▲" : "▼"} {row.predicted_signal}
                          </span>
                        </td>
                        <td className="text-gray-400 text-sm">{row.confidence?.toFixed(1)}%</td>
                        <td className="text-gray-400 text-sm">
                          {row.resolved
                            ? <span className={row.actual_signal === "BUY" ? "text-green-400" : "text-red-400"}>{row.actual_signal}</span>
                            : "—"}
                        </td>
                        <td><StatusBadge row={row} /></td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
