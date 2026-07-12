/**
 * frontend/src/pages/Settings.jsx
 * ─────────────────────────────────────────────────────────────────
 * User preferences — everything here is real and persisted (Supabase
 * for account actions, localStorage for device-local preferences).
 * No placeholder toggles: if a setting doesn't do anything yet, it's
 * not on this page.
 * ─────────────────────────────────────────────────────────────────
 */

import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { supabase } from "../lib/supabaseClient";
import {
  isSoundOn, setSoundOn,
  isReduceMotionOverride, setReduceMotionOverride,
  getDefaultDifficulty, setDefaultDifficulty,
  getDefaultModel, setDefaultModel,
} from "../lib/gameFx";

const PORTFOLIO_KEY = "nuroquant_portfolio";
const GAME_KEY = "quantsight_game_progress";

const DIFFICULTIES = [
  { key: "easy",   label: "Easy"   },
  { key: "medium", label: "Medium" },
  { key: "hard",   label: "Hard"   },
];

const MODEL_OPTIONS = [
  { key: "xgb_finance",    label: "XGBoost Finance"    },
  { key: "xgb_sentiment",  label: "XGBoost +Sentiment" },
  { key: "lstm_finance",   label: "LSTM Finance"       },
  { key: "lstm_sentiment", label: "LSTM +Sentiment"    },
];

function SectionCard({ title, sub, children }) {
  return (
    <div className="card space-y-4">
      <div>
        <h2 className="text-sm font-semibold text-white">{title}</h2>
        {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
      </div>
      {children}
    </div>
  );
}

function ToggleRow({ label, sub, checked, onChange }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <div className="text-sm text-gray-200">{label}</div>
        {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
      </div>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
          checked ? "bg-indigo-600" : "bg-gray-700"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
            checked ? "translate-x-5" : ""
          }`}
        />
      </button>
    </div>
  );
}

function downloadCsv(filename, headers, rows) {
  const escape = (v) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = [headers.join(","), ...rows.map(r => r.map(escape).join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function Settings() {
  const { user, signOut } = useAuth();

  // ── Game preferences ──────────────────────────────────────────
  const [soundOn, setSoundOnState] = useState(isSoundOn);
  const [reduceMotion, setReduceMotionState] = useState(isReduceMotionOverride);
  const [difficulty, setDifficultyState] = useState(getDefaultDifficulty);
  const [modelKey, setModelKeyState] = useState(getDefaultModel);

  // ── Account ────────────────────────────────────────────────────
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [pwBusy, setPwBusy] = useState(false);
  const [pwMessage, setPwMessage] = useState(null); // { type: "error"|"success", text }

  // ── Data management ───────────────────────────────────────────
  const [exportBusy, setExportBusy] = useState(null); // "portfolio" | "game" | null
  const [confirmingClear, setConfirmingClear] = useState(false);
  const [clearMessage, setClearMessage] = useState("");

  useEffect(() => {
    if (!confirmingClear) return;
    const t = setTimeout(() => setConfirmingClear(false), 4000);
    return () => clearTimeout(t);
  }, [confirmingClear]);

  async function handlePasswordChange(e) {
    e.preventDefault();
    setPwMessage(null);
    if (newPassword.length < 6) {
      setPwMessage({ type: "error", text: "Password must be at least 6 characters." });
      return;
    }
    if (newPassword !== confirmPassword) {
      setPwMessage({ type: "error", text: "Passwords don't match." });
      return;
    }
    setPwBusy(true);
    const { error } = await supabase.auth.updateUser({ password: newPassword });
    setPwBusy(false);
    if (error) {
      setPwMessage({ type: "error", text: error.message });
      return;
    }
    setNewPassword("");
    setConfirmPassword("");
    setPwMessage({ type: "success", text: "Password updated." });
  }

  async function handleSignOutEverywhere() {
    await supabase.auth.signOut({ scope: "global" });
  }

  async function exportPortfolio() {
    setExportBusy("portfolio");
    try {
      let holdings = [];
      if (user) {
        const { data, error } = await supabase
          .from("portfolio_holdings")
          .select("ticker, shares, buy_price, created_at")
          .order("created_at");
        if (error) throw error;
        holdings = data.map(r => [r.ticker, r.shares, r.buy_price, r.created_at]);
      } else {
        const raw = localStorage.getItem(PORTFOLIO_KEY);
        const local = raw ? JSON.parse(raw) : [];
        holdings = local.map(h => [h.ticker, h.shares, h.buyPrice, ""]);
      }
      downloadCsv("quantsight-portfolio.csv", ["Ticker", "Shares", "Buy Price", "Added"], holdings);
    } catch (err) {
      console.error("Failed to export portfolio:", err);
    } finally {
      setExportBusy(null);
    }
  }

  async function exportGameHistory() {
    setExportBusy("game");
    try {
      let history = [];
      if (user) {
        const { data, error } = await supabase
          .from("game_progress")
          .select("history")
          .eq("user_id", user.id)
          .maybeSingle();
        if (error) throw error;
        history = data?.history || [];
      } else {
        const raw = localStorage.getItem(GAME_KEY);
        history = raw ? JSON.parse(raw)?.history || [] : [];
      }
      const rows = history.map(h => [h.ticker, h.date, h.userGuess, h.actual, h.correct ? "Yes" : "No", h.points]);
      downloadCsv("quantsight-game-history.csv", ["Ticker", "Date", "Your Guess", "Actual", "Correct", "Points"], rows);
    } catch (err) {
      console.error("Failed to export game history:", err);
    } finally {
      setExportBusy(null);
    }
  }

  function handleClearLocalData() {
    if (!confirmingClear) {
      setConfirmingClear(true);
      return;
    }
    localStorage.removeItem(PORTFOLIO_KEY);
    localStorage.removeItem(GAME_KEY);
    setConfirmingClear(false);
    setClearMessage("Local guest-mode data cleared.");
    setTimeout(() => setClearMessage(""), 3000);
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Settings</h1>
        <p className="text-sm text-gray-500 mt-1">
          Preferences are saved to this browser, or to your account if you're signed in.
        </p>
      </div>

      {/* ── Account ── */}
      <SectionCard
        title="Account"
        sub={user ? user.email : "You're browsing as a guest"}
      >
        {user ? (
          <>
            <form onSubmit={handlePasswordChange} className="space-y-2.5">
              <p className="text-xs text-gray-500">Change password</p>
              <input
                type="password"
                placeholder="New password"
                value={newPassword}
                onChange={e => setNewPassword(e.target.value)}
                className="input w-full"
                autoComplete="new-password"
              />
              <input
                type="password"
                placeholder="Confirm new password"
                value={confirmPassword}
                onChange={e => setConfirmPassword(e.target.value)}
                className="input w-full"
                autoComplete="new-password"
              />
              {pwMessage && (
                <p className={`text-xs ${pwMessage.type === "error" ? "text-red-400" : "text-green-400"}`}>
                  {pwMessage.text}
                </p>
              )}
              <button type="submit" disabled={pwBusy} className="btn-primary text-sm disabled:opacity-50">
                {pwBusy ? "Updating…" : "Update password"}
              </button>
            </form>

            <div className="flex items-center gap-3 pt-3 border-t border-gray-800">
              <button onClick={() => signOut()} className="btn-secondary text-xs">
                Sign out
              </button>
              <button onClick={handleSignOutEverywhere} className="text-xs text-gray-500 hover:text-gray-300">
                Sign out of all devices
              </button>
            </div>
          </>
        ) : (
          <Link to="/login" className="btn-primary inline-block text-sm">
            Sign in to sync across devices
          </Link>
        )}
      </SectionCard>

      {/* ── Game preferences ── */}
      <SectionCard title="Game Preferences" sub="Applied the next time you start a round">
        <ToggleRow
          label="Sound effects"
          sub="Chimes for correct/wrong answers, streaks, and level-ups"
          checked={soundOn}
          onChange={(v) => { setSoundOnState(v); setSoundOn(v); }}
        />

        <div>
          <p className="text-sm text-gray-200 mb-2">Default difficulty</p>
          <div className="grid grid-cols-3 gap-2">
            {DIFFICULTIES.map(d => (
              <button
                key={d.key}
                onClick={() => { setDifficultyState(d.key); setDefaultDifficulty(d.key); }}
                className={`px-3 py-2 rounded-lg text-xs font-medium border transition-colors ${
                  difficulty === d.key
                    ? "bg-indigo-600 border-indigo-600 text-white"
                    : "bg-gray-800/40 border-gray-700 text-gray-400 hover:border-gray-600"
                }`}
              >
                {d.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <p className="text-sm text-gray-200 mb-2">Default model</p>
          <div className="grid grid-cols-2 gap-2">
            {MODEL_OPTIONS.map(m => (
              <button
                key={m.key}
                onClick={() => { setModelKeyState(m.key); setDefaultModel(m.key); }}
                className={`px-3 py-2 rounded-lg text-xs font-medium border transition-colors text-left ${
                  modelKey === m.key
                    ? "bg-indigo-900/30 border-indigo-700 text-indigo-300"
                    : "bg-gray-800/40 border-gray-700 text-gray-400 hover:border-gray-600"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
      </SectionCard>

      {/* ── Accessibility ── */}
      <SectionCard title="Accessibility">
        <ToggleRow
          label="Reduce motion"
          sub="Turns off confetti and count-up animations on the Game page, even if your system doesn't already request it"
          checked={reduceMotion}
          onChange={(v) => { setReduceMotionState(v); setReduceMotionOverride(v); }}
        />
      </SectionCard>

      {/* ── Data ── */}
      <SectionCard title="Data" sub="Your Portfolio and Game history, on your terms">
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={exportPortfolio}
            disabled={exportBusy === "portfolio"}
            className="btn-secondary text-xs disabled:opacity-50"
          >
            {exportBusy === "portfolio" ? "Exporting…" : "Export Portfolio (CSV)"}
          </button>
          <button
            onClick={exportGameHistory}
            disabled={exportBusy === "game"}
            className="btn-secondary text-xs disabled:opacity-50"
          >
            {exportBusy === "game" ? "Exporting…" : "Export Game History (CSV)"}
          </button>
        </div>

        <div className="pt-3 border-t border-gray-800 space-y-2">
          <p className="text-xs text-gray-500">
            Clears Portfolio and Game data stored in this browser (guest mode). This does not touch your synced account data.
          </p>
          <button
            onClick={handleClearLocalData}
            className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-colors ${
              confirmingClear
                ? "bg-red-900/40 border border-red-700 text-red-300"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700"
            }`}
          >
            {confirmingClear ? "Click again to confirm" : "Clear local guest-mode data"}
          </button>
          {clearMessage && <p className="text-xs text-green-400">{clearMessage}</p>}
        </div>
      </SectionCard>
    </div>
  );
}
