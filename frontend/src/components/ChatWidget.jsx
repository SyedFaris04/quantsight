/**
 * frontend/src/components/ChatWidget.jsx
 * ─────────────────────────────────────────────────────────────────
 * Floating AI Assistant — mounted once in App.jsx outside <Routes>,
 * so it persists across every page. Talks to POST /chat, which is
 * grounded in QuantSight's real model data via tool-calling on the
 * backend (see backend/chatbot_engine.py) — not a generic chatbot.
 *
 * History persists to localStorage so it survives navigation/reload,
 * same pattern as guest-mode Portfolio/Game.
 * ─────────────────────────────────────────────────────────────────
 */

import { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import Icon from "./Icon";

const STORAGE_KEY = "quantsight_chat_history";
const MAX_STORED = 30;
const API_URL = (import.meta.env.VITE_API_URL || "/api").replace(/\/$/, "");

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter(m =>
      m && ["user", "assistant"].includes(m.role) && typeof m.content === "string" && m.content.trim()
    ).slice(-MAX_STORED) : [];
  } catch { return []; }
}

function saveHistory(msgs) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(msgs.slice(-MAX_STORED))); }
  catch { /* Storage restrictions must not prevent a conversation. */ }
}

const SUGGESTIONS = [
  "Explain AAPL's saved signal",
  "Explain RSI in simple terms",
  "Summarize the saved news sentiment",
  "Should I look at AAPL?",
];

// Minimal markdown — bold + bullet lists, enough for typical LLM replies
// without pulling in a markdown dependency for one chat widget.
function formatInline(text, keyPrefix) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**")
      ? <strong key={`${keyPrefix}-${i}`} className="text-white">{part.slice(2, -2)}</strong>
      : <span key={`${keyPrefix}-${i}`}>{part}</span>
  );
}

function MessageContent({ text }) {
  return (
    <div className="space-y-1.5">
      {text.split("\n").map((line, i) => {
        const trimmed = line.trim();
        if (!trimmed) return null;
        if (/^[-*]\s+/.test(trimmed)) {
          return (
            <div key={i} className="flex gap-2">
              <span className="text-indigo-400 flex-shrink-0">•</span>
              <span>{formatInline(trimmed.replace(/^[-*]\s+/, ""), i)}</span>
            </div>
          );
        }
        return <p key={i}>{formatInline(line, i)}</p>;
      })}
    </div>
  );
}

function TypingDots() {
  return (
    <span className="inline-flex gap-1 py-1">
      {[0, 1, 2].map(i => (
        <span
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce"
          style={{ animationDelay: `${i * 120}ms` }}
        />
      ))}
    </span>
  );
}

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState(loadHistory);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState(null);
  const requestRef = useRef(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const location = useLocation();

  useEffect(() => { saveHistory(messages); }, [messages]);
  useEffect(() => () => requestRef.current?.abort(), []);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, isOpen, error]);

  useEffect(() => {
    if (isOpen) inputRef.current?.focus();
  }, [isOpen]);

  async function sendMessage(text, retry = false) {
    const content = text.trim();
    if (!content || streaming || requestRef.current) return;

    const history = retry ? messages : [...messages, { role: "user", content }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setStreaming(true);
    setError(null);
    const controller = new AbortController();
    requestRef.current = controller;
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 90000);

    const detailMatch = location.pathname.match(/^\/detail\/([^/]+)/);
    const page_context = detailMatch
      ? { ticker: detailMatch[1].toUpperCase() }
      : { path: location.pathname };

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history.slice(-16), page_context }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const detail = body.detail;
        const message = typeof detail?.message === "string" ? detail.message
          : typeof detail === "string" ? detail
          : res.status === 429 ? "Too many requests. Please wait and retry."
          : "The AI Assistant could not complete this request. Please retry.";
        throw new Error(message);
      }
      if (!res.body) throw new Error("The AI service returned no answer. Please retry.");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        setMessages(prev => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: acc };
          return copy;
        });
      }
      acc += decoder.decode();
      if (!acc.trim()) throw new Error("The AI service returned no answer. Please retry.");
      setMessages([...history, { role: "assistant", content: acc }]);
    } catch (e) {
      setMessages(history);
      setError(e.name === "AbortError"
        ? timedOut ? "The request timed out. The server may still be starting. Please retry." : "Reply stopped. You can retry your message."
        : e instanceof TypeError ? "Could not connect to the AI Assistant. Check your connection and retry." : e.message);
    } finally {
      clearTimeout(timeout);
      requestRef.current = null;
      setStreaming(false);
    }
  }

  return (
    <>
      <button
        onClick={() => setIsOpen(o => !o)}
        aria-label={isOpen ? "Close AI Assistant" : "Open AI Assistant"}
        className="fixed bottom-6 right-6 z-50 w-12 h-12 rounded-lg
                   bg-indigo-700 text-white
                   shadow-lg shadow-indigo-950/50 border border-indigo-400/30
                   flex items-center justify-center text-2xl
                   hover:scale-105 active:scale-95 transition-transform duration-150"
      >
        <Icon name={isOpen ? "close" : "chat"} />
      </button>

      {isOpen && (
        <div
          className="qs-card-in fixed z-50 bottom-24 right-4 sm:right-6
                     w-[min(24rem,calc(100vw-2rem))] h-[min(34rem,calc(100vh-8rem))]
                     bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl
                     flex flex-col overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center gap-2.5 px-4 h-14 border-b border-gray-800 flex-shrink-0">
            <span className="w-7 h-7 rounded-full bg-indigo-500/20 border border-indigo-600/40
                             flex items-center justify-center text-sm flex-shrink-0">
              <Icon name="chat" className="w-4 h-4" />
            </span>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-semibold text-white">QuantSight Assistant</div>
              <div className="text-[10px] text-gray-500">Grounded in real model data</div>
            </div>
            {messages.length > 0 && (
              <button
                onClick={() => { setMessages([]); setError(null); }}
                disabled={streaming}
                className="text-xs text-gray-500 hover:text-gray-300 flex-shrink-0 disabled:opacity-40"
              >
                Clear
              </button>
            )}
          </div>

          {/* Messages */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {messages.length === 0 && (
              <div className="space-y-3">
                <p className="text-sm text-gray-500">
                  Ask me about any stock, model signal, market sentiment, or general investing question.
                </p>
                <div className="flex flex-wrap gap-2">
                  {SUGGESTIONS.map(s => (
                    <button
                      key={s}
                      onClick={() => sendMessage(s)}
                      className="text-xs px-2.5 py-1.5 rounded-full bg-gray-800 hover:bg-gray-700
                                 text-gray-300 border border-gray-700 transition-colors"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
                    m.role === "user"
                      ? "bg-indigo-600 text-white rounded-br-sm"
                      : "bg-gray-800 text-gray-200 rounded-bl-sm"
                  }`}
                >
                  {m.content ? <MessageContent text={m.content} /> : <TypingDots />}
                </div>
              </div>
            ))}
            {error && <div role="alert" className="rounded-xl border border-amber-700/60 bg-amber-950/30 p-3 text-sm text-amber-100">
              <p>{error}</p>
              <button type="button" onClick={() => sendMessage(messages[messages.length - 1]?.content || "", true)}
                className="mt-2 underline underline-offset-4">Try again</button>
            </div>}
          </div>

          {/* Input */}
          <form
            onSubmit={(e) => { e.preventDefault(); sendMessage(input); }}
            className="flex items-center gap-2 p-3 border-t border-gray-800 flex-shrink-0"
          >
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Ask about a stock…"
              aria-label="Message to AI Assistant"
              maxLength={2000}
              disabled={streaming}
              className="input flex-1 text-sm disabled:opacity-50"
            />
            {streaming ? <button type="button" onClick={() => requestRef.current?.abort()}
              className="btn-primary text-sm px-3.5">Stop</button> : <button
              type="submit"
              disabled={streaming || !input.trim()}
              aria-label="Send"
              className="btn-primary text-sm px-3.5 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              →
            </button>}
          </form>
        </div>
      )}
    </>
  );
}
