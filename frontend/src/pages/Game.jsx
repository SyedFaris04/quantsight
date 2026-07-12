/**
 * frontend/src/pages/Game.jsx
 * ─────────────────────────────────────────────────────────────────
 * Stock Prediction Game — players guess what a real trained model
 * predicted (BUY/SELL) for a real ticker on a real, already-resolved
 * historical date. Score/streak/level/history persist in localStorage
 * across visits (see loadProgress/saveProgress above).
 *
 * The wireframe this page is styled after has a multi-player
 * Leaderboard slot — there is no user-account/multi-player backend in
 * this project, so that slot is intentionally filled with the
 * player's own real "Recent Results" history instead of invented
 * other players.
 *
 * Layout:
 *   ┌──────────────────────────────────────────────────┐
 *   │  Header — level badge, score, streak, high score │
 *   ├──────────────────────────────────────────────────┤
 *   │  Settings — difficulty + model selector,          │
 *   │  "Your Progress" summary if any history exists    │
 *   ├──────────────────────────────────────────────────┤
 *   │  Question card                                   │
 *   │    Ticker | Date | Close price                   │
 *   │    Hint: RSI value + MACD direction               │
 *   │    [ ▲ BUY ]  [ ▼ SELL ]                          │
 *   ├──────────────────────────────────────────────────┤
 *   │  Result card (after answer)                       │
 *   │    Correct/Wrong + explanation + points earned    │
 *   │    [ Next Question → ]                            │
 *   ├──────────────────────────────────────────────────┤
 *   │  Recent Results (real, persisted per-question log)│
 *   └──────────────────────────────────────────────────┘
 * ─────────────────────────────────────────────────────────────────
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useApi, postApi } from "../hooks/useApi";
import { companyName } from "../data/companyNames";
import { useAuth } from "../context/AuthContext";
import { supabase } from "../lib/supabaseClient";
import { useCountUp, playTone, burstConfetti, isSoundOn, setSoundOn } from "../lib/gameFx";

// ── Persistence — mirrors Portfolio.jsx's localStorage pattern, so
// "Your Progress" is genuinely cumulative across visits instead of
// resetting every time the page reloads. Explicit "Restart" is the only
// way to zero it out, same as removing a holding is the only way to
// clear portfolio data. ─────────────────────────────────────────────
const STORAGE_KEY = "quantsight_game_progress";

function loadProgress() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch { return null; }
}

function saveProgress(progress) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
}

// Real deterministic function of cumulative score — not fabricated,
// just a threshold ladder over a number the player actually earned.
const LEVEL_TIERS = [
  { min: 0,    title: "Rookie Analyst"  },
  { min: 250,  title: "Junior Analyst"  },
  { min: 750,  title: "Senior Analyst"  },
  { min: 1500, title: "Lead Analyst"    },
  { min: 3000, title: "Master Trader"   },
];

function getLevel(totalScore) {
  const level = Math.floor((totalScore ?? 0) / 250) + 1;
  const title = [...LEVEL_TIERS].reverse().find(t => (totalScore ?? 0) >= t.min)?.title ?? "Rookie Analyst";
  return { level, title };
}

// ── Constants ──────────────────────────────────────────────────────────────────

const DIFFICULTIES = [
  {
    key   : "easy",
    label : "Easy",
    desc  : "High-confidence signals — model is very sure",
    color : "text-green-400",
    bg    : "bg-green-900/20 border-green-800",
  },
  {
    key   : "medium",
    label : "Medium",
    desc  : "Mixed confidence — requires more intuition",
    color : "text-amber-400",
    bg    : "bg-amber-900/20 border-amber-800",
  },
  {
    key   : "hard",
    label : "Hard",
    desc  : "Low-confidence signals — even the model is unsure",
    color : "text-red-400",
    bg    : "bg-red-900/20 border-red-800",
  },
];

const MODEL_OPTIONS = [
  { key: "xgb_finance",    label: "XGBoost Finance"       },
  { key: "xgb_sentiment",  label: "XGBoost +Sentiment"    },
  { key: "lstm_finance",   label: "LSTM Finance"          },
  { key: "lstm_sentiment", label: "LSTM +Sentiment"       },
];

const STREAK_MILESTONES = [3, 5, 10, 15, 20];

// ── Sub-components ─────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, accent, animate = false }) {
  const accentMap = {
    indigo : "border-indigo-600/40",
    green  : "border-green-600/40",
    amber  : "border-amber-600/40",
    purple : "border-purple-600/40",
  };
  // Only numeric values can count up — strings like "87%" render as-is.
  const numeric = typeof value === "number";
  const displayed = useCountUp(numeric ? value : 0, 500);
  return (
    <div className={`card border ${accentMap[accent] || "border-gray-800"} text-center transition-transform`}>
      <div className="text-2xl font-bold text-white tabular-nums">
        {animate && numeric ? displayed : value}
      </div>
      <div className="text-xs text-gray-500 mt-0.5">{label}</div>
      {sub && <div className="text-xs text-gray-600 mt-0.5">{sub}</div>}
    </div>
  );
}

function StreakBadge({ streak }) {
  if (streak < 2) return null;
  const milestone = [...STREAK_MILESTONES].reverse().find(m => streak >= m);
  const emoji = streak >= 20 ? "🔥🔥🔥" : streak >= 10 ? "🔥🔥" : "🔥";
  return (
    <div className="flex items-center gap-2 bg-amber-900/30 border border-amber-800
                    rounded-full px-3 py-1 text-sm font-semibold text-amber-400 animate-pulse">
      {emoji} {streak} streak!
      {milestone && streak === milestone && (
        <span className="text-xs bg-amber-700 text-amber-100 px-1.5 py-0.5 rounded-full ml-1">
          Milestone!
        </span>
      )}
    </div>
  );
}

function HintRow({ icon, label, value, color }) {
  if (value == null) return null;
  return (
    <div className="flex items-center gap-3 bg-gray-800/60 rounded-lg px-4 py-2.5">
      <span className="text-lg">{icon}</span>
      <span className="text-sm text-gray-400 w-20">{label}</span>
      <span className={`text-sm font-semibold font-mono ${color}`}>{value}</span>
    </div>
  );
}

function RsiHint({ rsi }) {
  if (rsi == null) return null;
  const zone =
    rsi < 30 ? { label: "Oversold",    color: "text-green-400" } :
    rsi > 70 ? { label: "Overbought",  color: "text-red-400"   } :
               { label: "Neutral",     color: "text-gray-400"  };
  return (
    <div className="flex items-center gap-3 bg-gray-800/60 rounded-lg px-4 py-2.5">
      <span className="text-lg">📊</span>
      <span className="text-sm text-gray-400 w-20">RSI (14)</span>
      <span className={`text-sm font-semibold font-mono ${zone.color}`}>
        {rsi.toFixed(1)}{" "}
      </span>
      <span className={`text-xs px-2 py-0.5 rounded-full border ${
        rsi < 30 ? "border-green-700 bg-green-900/30 text-green-400" :
        rsi > 70 ? "border-red-700 bg-red-900/30 text-red-400" :
                   "border-gray-700 bg-gray-800 text-gray-400"
      }`}>
        {zone.label}
      </span>
    </div>
  );
}

function ResultCard({ result, onNext }) {
  const { correct, actual_signal, confidence, explanation, points_earned } = result;
  return (
    <div
      className={`qs-card-in rounded-2xl border p-6 space-y-4 ${
        correct
          ? "bg-green-900/20 border-green-700"
          : "bg-red-900/20 border-red-700"
      }`}
    >
      {/* Result header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-3xl qs-pop-in">{correct ? "✅" : "❌"}</span>
          <div>
            <div className={`text-lg font-bold ${correct ? "text-green-400" : "text-red-400"}`}>
              {correct ? "Correct!" : "Wrong!"}
            </div>
            <div className="text-sm text-gray-400">
              Actual signal:{" "}
              <span
                className={`font-semibold ${
                  actual_signal === "BUY" ? "text-green-400" : "text-red-400"
                }`}
              >
                {actual_signal === "BUY" ? "▲" : "▼"} {actual_signal}
              </span>
              <span className="text-gray-600 ml-2">
                ({confidence?.toFixed(1)}% model confidence)
              </span>
            </div>
          </div>
        </div>

        {/* Points */}
        {correct && (
          <div className="qs-pop-in text-center bg-amber-900/30 border border-amber-700
                          rounded-xl px-4 py-2">
            <div className="text-2xl font-bold text-amber-400">+{points_earned}</div>
            <div className="text-xs text-amber-600">points</div>
          </div>
        )}
      </div>

      {/* Explanation */}
      <p className="text-sm text-gray-300 leading-relaxed bg-gray-800/40
                    rounded-lg px-4 py-3">
        {explanation}
      </p>

      {/* Next button */}
      <button
        onClick={onNext}
        className="btn-primary w-full text-center py-3"
      >
        Next Question →
      </button>
    </div>
  );
}

function HistoryRow({ entry, index }) {
  return (
    <div className="flex items-center gap-3 py-2 border-b border-gray-800/50 last:border-0">
      <span className="text-gray-600 text-xs w-5">#{index + 1}</span>
      <span className="font-mono text-sm text-white w-14">{entry.ticker}</span>
      <span className="text-xs text-gray-500 w-24">{entry.date}</span>
      <span className={`text-xs font-medium ${
        entry.userGuess === "BUY" ? "text-green-400" : "text-red-400"
      }`}>
        You: {entry.userGuess}
      </span>
      <span className={`text-xs font-medium ${
        entry.actual === "BUY" ? "text-green-400" : "text-red-400"
      }`}>
        Actual: {entry.actual}
      </span>
      <span className="ml-auto">
        {entry.correct
          ? <span className="text-green-500 text-xs">✓ +{entry.points}</span>
          : <span className="text-red-500 text-xs">✗</span>
        }
      </span>
    </div>
  );
}

function CelebrationToast({ celebration }) {
  if (!celebration) return null;
  return (
    <div
      key={celebration.key}
      className="qs-toast fixed top-6 left-1/2 z-50 flex items-center gap-2
                 rounded-full border border-indigo-600/50 bg-gray-900/95 backdrop-blur-sm
                 px-5 py-2.5 shadow-2xl shadow-indigo-950/50"
    >
      <span className="text-xl">{celebration.emoji}</span>
      <span className="text-sm font-semibold text-white">{celebration.text}</span>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function Game() {
  const { user } = useAuth();
  const [soundOn, setSoundOnState] = useState(isSoundOn);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [celebration, setCelebration] = useState(null);
  const buyBtnRef = useRef(null);
  const sellBtnRef = useRef(null);
  const celebrationTimerRef = useRef(null);

  function toggleSound() {
    const next = !soundOn;
    setSoundOnState(next);
    setSoundOn(next);
  }

  function celebrate(emoji, text) {
    clearTimeout(celebrationTimerRef.current);
    setCelebration({ emoji, text, key: Date.now() });
    celebrationTimerRef.current = setTimeout(() => setCelebration(null), 2400);
  }

  function confettiFromButton(ref, colors) {
    const el = ref.current;
    if (!el) { burstConfetti({ colors }); return; }
    const rect = el.getBoundingClientRect();
    burstConfetti({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2, colors });
  }

  // ── Settings ────────────────────────────────────────────────────
  const [difficulty, setDifficulty]   = useState("medium");
  const [modelKey,   setModelKey]     = useState("xgb_sentiment");
  const [gameStarted, setGameStarted] = useState(false);

  // ── Game state ──────────────────────────────────────────────────
  const [question,  setQuestion]  = useState(null);
  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [answering, setAnswering] = useState(false);

  // ── Score tracking — initialized from localStorage so progress is
  // genuinely cumulative across visits, not wiped on every page load.
  // Overridden from Supabase below once signed in. ──────────────────
  const saved = loadProgress();
  const [score,     setScore]     = useState(saved?.score ?? 0);
  const [streak,    setStreak]    = useState(saved?.streak ?? 0);
  const [highScore, setHighScore] = useState(saved?.highScore ?? 0);
  const [total,     setTotal]     = useState(saved?.total ?? 0);
  const [correct,   setCorrect]   = useState(saved?.correct ?? 0);
  const [history,   setHistory]   = useState(saved?.history ?? []);
  const [importPrompt, setImportPrompt] = useState(false);

  const accuracy = total > 0 ? ((correct / total) * 100).toFixed(0) : "—";
  const { level, title: levelTitle } = getLevel(score);

  // On sign-in, load progress from Supabase (one row per user). If the
  // account has no row yet but this browser has real guest-mode progress,
  // offer a one-time, non-destructive import instead of silently losing it.
  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    (async () => {
      const { data, error } = await supabase
        .from("game_progress")
        .select("score, streak, high_score, total, correct, history")
        .eq("user_id", user.id)
        .maybeSingle();
      if (cancelled) return;
      if (error) {
        console.error("Failed to load game progress from Supabase:", error);
        return;
      }
      if (data) {
        setScore(data.score);
        setStreak(data.streak);
        setHighScore(data.high_score);
        setTotal(data.total);
        setCorrect(data.correct);
        setHistory(data.history || []);
      } else {
        const local = loadProgress();
        if (local && (local.total ?? 0) > 0) setImportPrompt(true);
      }
    })();
    return () => { cancelled = true; };
  }, [user]);

  async function importLocalProgress() {
    const local = loadProgress();
    setImportPrompt(false);
    if (!user || !local) return;
    const { error } = await supabase.from("game_progress").upsert({
      user_id    : user.id,
      score      : local.score ?? 0,
      streak     : local.streak ?? 0,
      high_score : local.highScore ?? 0,
      total      : local.total ?? 0,
      correct    : local.correct ?? 0,
      history    : local.history ?? [],
    });
    if (error) {
      console.error("Failed to import game progress:", error);
      return;
    }
    setScore(local.score ?? 0);
    setStreak(local.streak ?? 0);
    setHighScore(local.highScore ?? 0);
    setTotal(local.total ?? 0);
    setCorrect(local.correct ?? 0);
    setHistory(local.history ?? []);
  }

  // Persist on every change — Supabase (one upserted row) when signed in,
  // localStorage (guest mode, untouched by signing in) otherwise.
  useEffect(() => {
    if (user) {
      supabase.from("game_progress").upsert({
        user_id: user.id, score, streak, high_score: highScore, total, correct, history,
      }).then(({ error }) => {
        if (error) console.error("Failed to save game progress to Supabase:", error);
      });
    } else {
      saveProgress({ score, streak, highScore, total, correct, history });
    }
  }, [score, streak, highScore, total, correct, history, user]);

  // ── Fetch a new question ─────────────────────────────────────────
  const fetchQuestion = useCallback(async () => {
    setLoading(true);
    setResult(null);
    setQuestion(null);
    setSelectedAnswer(null);
    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL || "/api"}/game/question` +
        `?difficulty=${difficulty}&model_key=${modelKey}`
      );
      const data = await res.json();
      if (data.detail) throw new Error(data.detail);
      setQuestion(data);
    } catch (e) {
      console.error("Failed to fetch question:", e);
    } finally {
      setLoading(false);
    }
  }, [difficulty, modelKey]);

  // ── Submit answer ────────────────────────────────────────────────
  async function handleAnswer(userSignal) {
    if (!question || answering) return;
    setAnswering(true);
    setSelectedAnswer(userSignal);

    try {
      const res = await postApi("/game/answer", {
        ticker      : question.ticker,
        date        : question.date,
        user_signal : userSignal,
        model_key   : modelKey,
      });

      setResult(res);
      setTotal(t => t + 1);

      if (res.correct) {
        const newScore  = score + res.points_earned;
        const newStreak = streak + 1;
        const oldLevel  = getLevel(score).level;
        const newLevel  = getLevel(newScore).level;

        setScore(newScore);
        setCorrect(c => c + 1);
        setStreak(newStreak);
        if (newScore > highScore) setHighScore(newScore);

        const btnRef = userSignal === "BUY" ? buyBtnRef : sellBtnRef;
        if (newLevel > oldLevel) {
          playTone("levelup");
          confettiFromButton(buyBtnRef, ["#6366f1", "#818cf8", "#fbbf24", "#22c55e"]);
          celebrate("🚀", `Level up! You're now a ${getLevel(newScore).title}`);
        } else if (STREAK_MILESTONES.includes(newStreak)) {
          playTone("milestone");
          confettiFromButton(btnRef, ["#fbbf24", "#f59e0b", "#22c55e"]);
          celebrate("🔥", `${newStreak} in a row! You're on fire!`);
        } else {
          playTone("correct");
          confettiFromButton(btnRef, ["#22c55e", "#4ade80", "#6366f1"]);
        }
      } else {
        setStreak(0);
        playTone("wrong");
      }

      setHistory(h => [
        {
          ticker    : question.ticker,
          date      : question.date,
          userGuess : userSignal,
          actual    : res.actual_signal,
          correct   : res.correct,
          points    : res.points_earned,
        },
        ...h.slice(0, 19),  // keep last 20
      ]);
    } catch (e) {
      console.error("Failed to submit answer:", e);
    } finally {
      setAnswering(false);
    }
  }

  // ── Settings screen ──────────────────────────────────────────────
  if (!gameStarted) {
    return (
      <div className="max-w-2xl mx-auto space-y-6">
        <div>
          <h1 className="text-xl font-semibold text-white">Stock Prediction Game</h1>
          <p className="text-sm text-gray-500 mt-1">
            Test your market intuition — guess BUY or SELL based on technical hints
          </p>
        </div>

        {/* Import prompt — shown once, only if signing in finds an empty
            account but real guest-mode progress on this device */}
        {importPrompt && (
          <div className="bg-indigo-900/20 border border-indigo-800 rounded-lg px-4 py-3
                           flex items-center justify-between gap-4 text-sm">
            <span className="text-indigo-300">
              You have game progress saved on this device from before signing in. Import it into your account?
            </span>
            <div className="flex items-center gap-2 flex-shrink-0">
              <button onClick={importLocalProgress} className="btn-primary text-xs px-3 py-1.5">
                Import
              </button>
              <button
                onClick={() => setImportPrompt(false)}
                className="text-xs text-gray-400 hover:text-gray-200 px-2"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {/* Your Progress — real, persisted across visits */}
        {total > 0 && (
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <p className="section-title mb-0">Your Progress</p>
              <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-900/30 text-indigo-300 border border-indigo-700 font-medium">
                Level {level} · {levelTitle}
              </span>
            </div>
            <div className="grid grid-cols-4 gap-3">
              <StatCard label="Games Played" value={total} accent="indigo" />
              <StatCard label="Correct" value={correct} accent="green" />
              <StatCard label="Accuracy" value={`${accuracy}%`} accent="amber" />
              <StatCard label="Current Streak" value={streak} accent="purple" />
            </div>
          </div>
        )}

        {/* Difficulty */}
        <div className="card space-y-3">
          <p className="section-title">Difficulty</p>
          <div className="grid grid-cols-3 gap-3">
            {DIFFICULTIES.map(d => (
              <button
                key={d.key}
                onClick={() => setDifficulty(d.key)}
                className={`rounded-xl p-4 border text-left transition-all ${
                  difficulty === d.key
                    ? d.bg
                    : "bg-gray-800/40 border-gray-700 hover:border-gray-600"
                }`}
              >
                <div className={`font-semibold text-sm mb-1 ${
                  difficulty === d.key ? d.color : "text-gray-300"
                }`}>
                  {d.label}
                </div>
                <div className="text-xs text-gray-500">{d.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Model selector */}
        <div className="card space-y-3">
          <p className="section-title">Model to play against</p>
          <div className="grid grid-cols-2 gap-3">
            {MODEL_OPTIONS.map(m => (
              <button
                key={m.key}
                onClick={() => setModelKey(m.key)}
                className={`rounded-xl p-3 border text-left text-sm transition-all ${
                  modelKey === m.key
                    ? "bg-indigo-900/30 border-indigo-700 text-indigo-300"
                    : "bg-gray-800/40 border-gray-700 text-gray-400 hover:border-gray-600"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-600">
            The game will show you a stock question and reveal whether
            the selected model predicted BUY or SELL after you answer.
          </p>
        </div>

        {/* How to play */}
        <div className="card bg-gray-900/50 space-y-2">
          <p className="section-title">How to play</p>
          <ul className="text-sm text-gray-400 space-y-1.5">
            <li className="flex gap-2"><span className="text-indigo-400">1.</span> A random stock + date is shown with hint indicators</li>
            <li className="flex gap-2"><span className="text-indigo-400">2.</span> Guess whether the model predicted BUY or SELL</li>
            <li className="flex gap-2"><span className="text-indigo-400">3.</span> Earn 10 points for each correct answer</li>
            <li className="flex gap-2"><span className="text-indigo-400">4.</span> Bonus points for hard questions (low model confidence)</li>
            <li className="flex gap-2"><span className="text-indigo-400">5.</span> Build streaks of correct answers for milestones 🔥</li>
          </ul>
        </div>

        <button
          className="btn-primary w-full py-3 text-base"
          onClick={() => { setGameStarted(true); fetchQuestion(); }}
        >
          Start Game →
        </button>
      </div>
    );
  }

  // ── Active game ──────────────────────────────────────────────────
  return (
    <div className="max-w-2xl mx-auto space-y-6">

      <CelebrationToast celebration={celebration} />

      {/* ── Header stats ── */}
      <div className="flex items-center justify-between">
        <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-900/30 text-indigo-300 border border-indigo-700 font-medium">
          Level {level} · {levelTitle}
        </span>
        <button
          onClick={toggleSound}
          title={soundOn ? "Mute sound effects" : "Unmute sound effects"}
          className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-400
                     hover:bg-gray-800 hover:text-white transition-colors text-sm"
        >
          {soundOn ? "🔊" : "🔇"}
        </button>
      </div>
      <div className="grid grid-cols-4 gap-3">
        <StatCard label="Score"      value={score}         accent="indigo" animate />
        <StatCard label="High Score" value={highScore}     accent="amber"  animate />
        <StatCard label="Accuracy"   value={`${accuracy}%`} accent="green" />
        <StatCard label="Answered"   value={total}         accent="purple" animate />
      </div>

      {/* Streak badge */}
      {streak >= 2 && (
        <div className="flex justify-center">
          <StreakBadge streak={streak} />
        </div>
      )}

      {/* ── Settings bar ── */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex rounded-lg overflow-hidden border border-gray-700 text-xs">
          {DIFFICULTIES.map(d => (
            <button
              key={d.key}
              onClick={() => { setDifficulty(d.key); setResult(null); }}
              className={`px-3 py-1.5 font-medium transition-colors ${
                difficulty === d.key
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              }`}
            >
              {d.label}
            </button>
          ))}
        </div>

        <select
          className="input text-xs py-1.5"
          value={modelKey}
          onChange={e => { setModelKey(e.target.value); setResult(null); }}
        >
          {MODEL_OPTIONS.map(m => (
            <option key={m.key} value={m.key}>{m.label}</option>
          ))}
        </select>

        <button
          className="btn-secondary text-xs ml-auto"
          onClick={() => {
            // Only leaves the active round — your cumulative score/streak/
            // history is real, persisted progress and stays intact.
            setGameStarted(false);
            setQuestion(null); setResult(null);
          }}
        >
          ← Change Settings
        </button>
      </div>

      {/* ── Question card ── */}
      {loading ? (
        <div className="card space-y-4">
          <div className="h-6 bg-gray-800 rounded animate-pulse w-32" />
          <div className="h-4 bg-gray-800 rounded animate-pulse w-48" />
          <div className="h-12 bg-gray-800 rounded animate-pulse" />
          <div className="h-12 bg-gray-800 rounded animate-pulse" />
          <div className="grid grid-cols-2 gap-3">
            <div className="h-14 bg-gray-800 rounded animate-pulse" />
            <div className="h-14 bg-gray-800 rounded animate-pulse" />
          </div>
        </div>
      ) : question && !result ? (
        <div key={`${question.ticker}-${question.date}`} className="qs-card-in card space-y-5">
          {/* Ticker + date */}
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold text-white font-mono">
                  {question.ticker}
                </span>
                <span className={`text-xs px-2 py-0.5 rounded-full border ${
                  difficulty === "easy"   ? "border-green-700 bg-green-900/30 text-green-400" :
                  difficulty === "hard"   ? "border-red-700 bg-red-900/30 text-red-400" :
                                           "border-amber-700 bg-amber-900/30 text-amber-400"
                }`}>
                  {difficulty}
                </span>
              </div>
              {companyName(question.ticker) && (
                <div className="text-xs text-gray-500">{companyName(question.ticker)}</div>
              )}
              <div className="text-sm text-gray-500 mt-0.5">{question.date}</div>
            </div>
            {question.close_price && (
              <div className="text-right">
                <div className="text-lg font-semibold text-white">
                  ${question.close_price.toFixed(2)}
                </div>
                <div className="text-xs text-gray-600">Close price</div>
              </div>
            )}
          </div>

          {/* Hints */}
          <div>
            <p className="section-title mb-2">Hints</p>
            <div className="space-y-2">
              <RsiHint rsi={question.hint_rsi} />
              {question.hint_macd && (
                <HintRow
                  icon="📉"
                  label="MACD"
                  value={question.hint_macd.charAt(0).toUpperCase() + question.hint_macd.slice(1)}
                  color={question.hint_macd === "bullish" ? "text-green-400" : "text-red-400"}
                />
              )}
              <HintRow
                icon="🤖"
                label="Model"
                value={MODEL_OPTIONS.find(m => m.key === modelKey)?.label}
                color="text-indigo-400"
              />
            </div>
          </div>

          {/* Question */}
          <p className="text-sm text-gray-400 text-center py-1">
            What did the model predict for{" "}
            <span className="text-white font-semibold">{question.ticker}</span>{" "}
            on <span className="text-white">{question.date}</span>?
          </p>

          {/* BUY / SELL buttons */}
          <div className="grid grid-cols-2 gap-4">
            <button
              ref={buyBtnRef}
              onClick={() => handleAnswer("BUY")}
              disabled={answering}
              className={`py-5 rounded-2xl border-2 border-green-700 bg-green-900/20
                         hover:bg-green-900/40 text-green-400 font-bold text-xl
                         transition-all active:scale-95 disabled:cursor-not-allowed
                         ${selectedAnswer === "BUY" ? "qs-locked scale-95" : ""}
                         ${answering && selectedAnswer !== "BUY" ? "opacity-30" : ""}`}
            >
              ▲ BUY
            </button>
            <button
              ref={sellBtnRef}
              onClick={() => handleAnswer("SELL")}
              disabled={answering}
              className={`py-5 rounded-2xl border-2 border-red-700 bg-red-900/20
                         hover:bg-red-900/40 text-red-400 font-bold text-xl
                         transition-all active:scale-95 disabled:cursor-not-allowed
                         ${selectedAnswer === "SELL" ? "qs-locked scale-95" : ""}
                         ${answering && selectedAnswer !== "SELL" ? "opacity-30" : ""}`}
            >
              ▼ SELL
            </button>
          </div>
        </div>
      ) : null}

      {/* ── Result card ── */}
      {result && (
        <ResultCard result={result} onNext={fetchQuestion} />
      )}

      {/* ── Recent results — real per-question history, persisted across
           visits. This is what fills the wireframe's "Leaderboard" slot:
           there's no multi-user backend to source other players' scores
           from, so this shows your own real result history instead of
           inventing other players. ── */}
      {history.length > 0 && (
        <div className="card">
          <p className="section-title mb-3">Recent Results</p>
          <div className="space-y-0">
            {history.map((entry, i) => (
              <HistoryRow key={i} entry={entry} index={i} />
            ))}
          </div>
          <div className="mt-3 pt-3 border-t border-gray-800 flex justify-between text-xs text-gray-500">
            <span>{correct} correct out of {total} answered (all-time)</span>
            <span>Total score: {score} pts</span>
          </div>
        </div>
      )}

    </div>
  );
}
