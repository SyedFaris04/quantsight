"""
backend/chatbot_engine.py
─────────────────────────────────────────────────────────────────────────────
The AI Assistant's brain — talks to Groq (Llama 3.3 70B, free tier), with
tool-calling wired to QuantSight's own trained models and real data. The
model never invents a signal or confidence number: for anything ticker- or
market-specific, it calls a tool and answers from the actual result.

This module has zero knowledge of FastAPI or the request/response cycle —
main.py owns the /chat endpoint and passes in TOOL_EXECUTORS (plain callables
bound to its already-loaded cache/model functions), which keeps this module
free of circular imports.

Design choice — simulated streaming, not token-level streaming:
Groq's raw completion speed (~500-800 tok/s) makes "generate the full answer,
then stream it to the client in small chunks" visually indistinguishable from
true incremental streaming, while being far simpler and more reliable than
accumulating partial tool-call JSON across a live stream. See run_chat_stream.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
import time
from groq import Groq

logger = logging.getLogger("nuroquant-api")

MODEL = "llama-3.3-70b-versatile"
MAX_TOOL_ROUNDS = 3
MAX_HISTORY_MESSAGES = 16
MAX_MESSAGE_CHARS = 2000
MAX_TOOL_RESULT_CHARS = 4000

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set — the AI Assistant needs a free key from "
                "console.groq.com (API Keys → Create API Key)."
            )
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are the QuantSight AI Assistant, built into the QuantSight stock \
decision-support platform. You help users understand stock signals, technical \
indicators, sentiment, and how QuantSight's own models arrived at a prediction.

QuantSight runs 4 trained models per ticker (XGBoost and LSTM+Transformer, each \
with a "Finance only" and "Finance + Sentiment" variant) over a historical \
dataset through December 2024. A separate live pipeline (Yahoo Finance + the \
XGBoost model) can score a ticker using today's real price data — use the \
get_live_price_signal tool specifically when the user asks about "right now" \
or "today".

Rules:
- Never invent a signal, confidence %, or price. If a question is about a \
specific ticker or the overall market, call a tool first and answer from its \
real result.
- If a ticker isn't recognized, call list_tickers and tell the user what's \
actually available rather than guessing.
- Keep answers tight and scannable: short paragraphs, bullet points and \
**bold** for key numbers/signals where it helps.
- QuantSight's signals are model outputs from historical patterns, not \
financial advice. Add a brief one-line reminder of that only when you're \
directly answering a "should I buy/sell" style question — don't repeat it \
on every message.
- General finance/investing education questions (e.g. "what is RSI") don't \
need a tool call — answer directly from what you know.
- If the user is currently viewing a specific ticker's page (given in the \
context below), you can assume questions like "what about this one" refer \
to it."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tickers",
            "description": "List every stock ticker available in QuantSight. Call this if a ticker the user mentions might not be covered, or they ask what's available.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_overview",
            "description": "Snapshot of the whole market: counts of BUY/SELL/HOLD tickers, average model confidence, and the top current BUY opportunities. Use for broad questions like 'what's trending' or 'what looks good today'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_signal",
            "description": "The current BUY/SELL/HOLD signal, confidence, model agreement, and risk level for one ticker, from all 4 trained models.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ai_explanation",
            "description": "Full explanation for one ticker: per-model signals, technical indicators (RSI, MACD, Bollinger, moving averages), news/social sentiment, risk level, and a plain-English summary. Use when the user asks 'why' or wants real detail on a specific stock.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_price_signal",
            "description": "Right-now signal for one ticker, computed fresh from today's live Yahoo Finance price data (not the historical dataset). Use this specifically for 'today' / 'right now' / 'currently' questions.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_sentiment",
            "description": "Real-time overall news sentiment (% positive/neutral/negative) across recent market headlines.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _sanitize_history(messages: list[dict]) -> list[dict]:
    trimmed = messages[-MAX_HISTORY_MESSAGES:]
    out = []
    for m in trimmed:
        role = m.get("role")
        content = str(m.get("content", ""))[:MAX_MESSAGE_CHARS]
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def _execute_tool(name: str, args: dict, tool_executors: dict) -> dict:
    fn = tool_executors.get(name)
    if not fn:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return fn(**args)
    except Exception as e:
        logger.warning(f"Chat tool '{name}' failed: {e}")
        return {"error": f"{name} failed: {e}"}


def run_chat_stream(messages: list[dict], tool_executors: dict, page_context: dict | None = None):
    """
    Generator — yields the assistant's reply as small text chunks.

    messages: full conversation history from the client, oldest first,
              each { role: "user"|"assistant", content: str }.
    tool_executors: { tool_name: callable(**kwargs) -> dict }, provided by
              main.py so this module never imports it directly.
    """
    client = get_client()

    system = SYSTEM_PROMPT
    if page_context and page_context.get("ticker"):
        system += f"\n\nContext: the user is currently viewing the page for {page_context['ticker']}."

    working_messages = [{"role": "system", "content": system}] + _sanitize_history(messages)

    final_content = None
    for _ in range(MAX_TOOL_ROUNDS):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=working_messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.4,
                max_tokens=900,
            )
        except Exception as e:
            logger.error(f"Groq call failed: {e}")
            yield "Sorry — I couldn't reach the AI service just now. Please try again in a moment."
            return

        choice = resp.choices[0]
        msg = choice.message

        if msg.tool_calls:
            working_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = _execute_tool(tc.function.name, args, tool_executors)
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _truncate(json.dumps(result, default=str), MAX_TOOL_RESULT_CHARS),
                })
            continue

        final_content = msg.content or "I'm not sure how to answer that — could you rephrase?"
        break

    if final_content is None:
        final_content = "That took more digging than expected — could you narrow the question down?"

    # Chunk word-by-word so the client sees a natural typing effect over the
    # real HTTP stream, without a second round-trip to the model.
    words = final_content.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.012)
