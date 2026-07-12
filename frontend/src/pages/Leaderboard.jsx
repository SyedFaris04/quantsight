/**
 * frontend/src/pages/Leaderboard.jsx
 * ─────────────────────────────────────────────────────────────────
 * Model Leaderboard — a SQuAD-explorer-style ranked table of
 * QuantSight's own trained model variants (no external papers mixed
 * in — different datasets/tasks aren't apples-to-apples, and several
 * report numbers that look leakage-inflated; see README).
 *
 * Rows are built dynamically from whatever /metrics actually returns
 * (not a hardcoded list) — any new model added to model_metrics.json
 * (by any train_*.py script) shows up here automatically.
 *
 * Same data source as the Compare page (/metrics) — Compare stays a
 * fixed 4-model architecture-vs-sentiment ablation (that's the page
 * the lecturer specifically asked for), this is the "everyone ranked
 * together" view, which is exactly what a leaderboard is for.
 * ─────────────────────────────────────────────────────────────────
 */

import { useState, useMemo } from "react";
import { useApi } from "../hooks/useApi";

// ── Constants ──────────────────────────────────────────────────────────────

// Color per model FAMILY (not per variant) — both variants of the same
// model read as clearly related. Anything not listed falls back to gray.
const MODEL_COLORS = {
  "XGBoost"             : "#3b82f6",  // blue
  "LSTM+Transformer"    : "#a855f7",  // purple
  "RandomForest"        : "#22c55e",  // green
  "LogisticRegression"  : "#ec4899",  // pink
  "GRU"                 : "#14b8a6",  // teal
  "TransformerEncoder"  : "#eab308",  // yellow
  "Ensemble"            : "#f97316",  // orange
};
const DEFAULT_COLOR = "#6b7280";

// Cosmetic-only spacing for camelCase-ish model keys — doesn't touch the
// underlying model_metrics.json values or any other page's lookups.
const MODEL_DISPLAY_NAME = {
  "RandomForest"       : "Random Forest",
  "LogisticRegression" : "Logistic Regression",
  "TransformerEncoder" : "Transformer Encoder",
};

const COLUMNS = [
  { key: "accuracy",  label: "Accuracy"  },
  { key: "f1",        label: "F1"        },
  { key: "precision", label: "Precision" },
  { key: "recall",    label: "Recall"    },
  { key: "auc_roc",   label: "AUC-ROC"   },
];

const RANDOM_BASELINE = { accuracy: 50.0, f1: 50.0, precision: 50.0, recall: 50.0, auc_roc: 50.0 };

// ── Sub-components ────────────────────────────────────────────────────────

function SortIcon({ active, dir }) {
  if (!active) return <span className="text-gray-700 ml-1">↕</span>;
  return <span className="text-indigo-400 ml-1">{dir === "desc" ? "↓" : "↑"}</span>;
}

// Finance-only / +Sentiment / Ensemble badge — the Ensemble's variant
// string ("All 4 Variants (avg. calibrated probability)") is neither, so
// it gets its own distinct pill instead of being force-matched.
function DataBadge({ variant }) {
  if (variant === "Finance + Sentiment") {
    return (
      <span className="text-xs px-2 py-0.5 rounded-full border bg-indigo-900/30 text-indigo-400 border-indigo-800">
        +Sentiment
      </span>
    );
  }
  if (variant === "Finance Only") {
    return (
      <span className="text-xs px-2 py-0.5 rounded-full border bg-gray-800 text-gray-400 border-gray-700">
        Finance
      </span>
    );
  }
  return (
    <span className="text-xs px-2 py-0.5 rounded-full border bg-orange-900/30 text-orange-400 border-orange-800">
      Ensemble
    </span>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export default function Leaderboard() {
  const { data: metricsRaw, loading, error } = useApi("/metrics");

  const [sortKey, setSortKey] = useState("accuracy");
  const [sortDir, setSortDir] = useState("desc");

  const models = useMemo(() => {
    if (!metricsRaw) return [];
    return Object.entries(metricsRaw)
      .map(([key, m]) => ({
        key,
        ...m,
        displayName: MODEL_DISPLAY_NAME[m.model] || m.model,
        color: MODEL_COLORS[m.model] || DEFAULT_COLOR,
      }))
      .filter(m => m.accuracy != null);
  }, [metricsRaw]);

  const ranked = useMemo(() => {
    const sorted = [...models].sort((a, b) => {
      const diff = (a[sortKey] ?? 0) - (b[sortKey] ?? 0);
      return sortDir === "desc" ? -diff : diff;
    });
    return sorted.map((m, i) => ({ ...m, rank: i + 1 }));
  }, [models, sortKey, sortDir]);

  function handleSort(key) {
    if (key === sortKey) {
      setSortDir(d => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-4">
        <span className="text-4xl">⚠️</span>
        <p className="text-gray-400 text-sm text-center max-w-md">
          Could not load model metrics. Make sure at least one{" "}
          <code className="text-indigo-400">train_*.py</code> script has been
          run and <code className="text-indigo-400">model_metrics.json</code> exists.
        </p>
        <p className="text-xs text-gray-600">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">

      {/* ── Page header ── */}
      <div>
        <h1 className="text-xl font-semibold text-white flex items-center gap-2">
          <span>🏆</span> Model Leaderboard
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          Every model variant QuantSight has trained, ranked by test-set performance.
          Click a column header to sort.
        </p>
      </div>

      {/* ── Ranked table ── */}
      <div className="card !p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th className="w-14 text-center">Rank</th>
                <th className="w-56">Model</th>
                <th className="w-32">Data</th>
                {COLUMNS.map(c => (
                  <th
                    key={c.key}
                    className="text-center cursor-pointer select-none hover:text-gray-300"
                    onClick={() => handleSort(c.key)}
                  >
                    {c.label}
                    <SortIcon active={sortKey === c.key} dir={sortDir} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                [1, 2, 3, 4, 5].map(i => (
                  <tr key={i}>
                    <td colSpan={8}>
                      <div className="h-8 bg-gray-800 rounded animate-pulse my-1" />
                    </td>
                  </tr>
                ))
              ) : (
                <>
                  {ranked.map(m => (
                    <tr
                      key={m.key}
                      className={m.rank === 1 ? "bg-indigo-600/5" : ""}
                    >
                      <td className="text-center">
                        <span
                          className={`text-sm font-semibold ${
                            m.rank === 1 ? "text-indigo-400" : "text-gray-500"
                          }`}
                        >
                          #{m.rank}
                        </span>
                      </td>
                      <td>
                        <div className="flex items-center gap-2">
                          <span
                            className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                            style={{ background: m.color }}
                          />
                          <span className="font-medium text-white text-sm">{m.displayName}</span>
                        </div>
                      </td>
                      <td>
                        <DataBadge variant={m.variant} />
                      </td>
                      {COLUMNS.map(c => (
                        <td key={c.key} className="text-center">
                          <span
                            className={`text-sm font-medium ${
                              sortKey === c.key ? "text-white" : "text-gray-300"
                            }`}
                          >
                            {m[c.key] != null ? `${m[c.key].toFixed(2)}%` : "—"}
                          </span>
                        </td>
                      ))}
                    </tr>
                  ))}

                  {/* Baseline reference row — not a ranked entry */}
                  <tr className="opacity-60">
                    <td className="text-center">
                      <span className="text-gray-700 text-xs">—</span>
                    </td>
                    <td>
                      <span className="text-gray-500 text-sm italic">Random Guess</span>
                    </td>
                    <td>
                      <span className="text-xs px-2 py-0.5 rounded-full border border-dashed border-gray-700 text-gray-600">
                        baseline
                      </span>
                    </td>
                    {COLUMNS.map(c => (
                      <td key={c.key} className="text-center">
                        <span className="text-sm text-gray-600">
                          {RANDOM_BASELINE[c.key].toFixed(2)}%
                        </span>
                      </td>
                    ))}
                  </tr>
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-xs text-gray-600 px-1">
        Chronological 2015–2023 train / 2023–2024 test split, purged walk-forward
        cross-validation, no leakage. "Random Guess" is a fixed 50% reference floor
        for binary classification, not a measured result — shown for scale, since
        daily stock direction is a genuinely hard prediction problem where accuracy
        in the low-50s is meaningful, not a bug.
      </p>
    </div>
  );
}
