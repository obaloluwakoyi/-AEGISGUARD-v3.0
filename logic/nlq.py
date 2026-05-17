"""
aegisguard/logic/nlq.py
─────────────────────────────────────────────────────────────────────────────
Area 4 — Natural Language Query Interface   (Multi-Provider AI Engine)

Supports 12 AI back-ends:

  ── No-API / Local (free) ─────────────────────────────────────────────────
  1. Ollama          localhost:11434   (llama3, mistral, gemma, etc.)
  2. LM Studio       localhost:1234    (OpenAI-compatible server)
  3. GPT4All         localhost:4891    (OpenAI-compatible)
  4. Jan.ai          localhost:1337    (OpenAI-compatible)
  5. Kobold.cpp      localhost:5001    (KoboldAI API)
  6. llama.cpp       localhost:8080    (llama.cpp built-in server)

  ── API-Based (key required) ──────────────────────────────────────────────
  7.  Groq           api.groq.com           llama3-70b-8192
  8.  OpenAI         api.openai.com         gpt-4o-mini
  9.  Anthropic      api.anthropic.com      claude-sonnet-4-20250514
  10. Google Gemini  generativelanguage…    gemini-1.5-flash
  11. Mistral AI     api.mistral.ai         mistral-large-latest
  12. Together AI    api.together.xyz       meta-llama/Llama-3-70b-chat

Usage: selected via the sidebar dropdown in app.py.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import json, re, requests
from dataclasses import dataclass, field
from datetime    import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Provider registry
# ─────────────────────────────────────────────────────────────────────────────

# Each entry:  (display_label, provider_key, needs_api_key, default_model, hint)
PROVIDERS: list[tuple[str, str, bool, str, str]] = [
    # ── No-API / Local ────────────────────────────────────────────────────────
    ("🦙 Ollama (local)",        "ollama",    False, "llama3",              "localhost:11434 — run `ollama serve`"),
    ("🖥️ LM Studio (local)",    "lmstudio",  False, "local-model",         "localhost:1234  — start server in LM Studio"),
    ("🤗 GPT4All (local)",      "gpt4all",   False, "local-model",         "localhost:4891  — enable API in GPT4All"),
    ("❄️ Jan.ai (local)",       "jan",       False, "local-model",         "localhost:1337  — start Jan server"),
    ("🧟 Kobold.cpp (local)",   "kobold",    False, "",                    "localhost:5001  — run koboldcpp"),
    ("🦎 llama.cpp (local)",    "llamacpp",  False, "",                    "localhost:8080  — run llama-server"),
    # ── API-Based ─────────────────────────────────────────────────────────────
    ("⚡ Groq",                  "groq",      True,  "llama3-70b-8192",     "api.groq.com — free tier available"),
    ("🟢 OpenAI",               "openai",    True,  "gpt-4o-mini",         "api.openai.com"),
    ("🟣 Anthropic (Claude)",   "anthropic", True,  "claude-sonnet-4-20250514", "api.anthropic.com"),
    ("🔵 Google Gemini",        "gemini",    True,  "gemini-1.5-flash",    "aistudio.google.com/app/apikey"),
    ("🟠 Mistral AI",           "mistral",   True,  "mistral-large-latest","api.mistral.ai"),
    ("🌐 Together AI",          "together",  True,  "meta-llama/Llama-3-70b-chat-hf", "api.together.xyz"),
]

NO_API_KEYS  = [p[1] for p in PROVIDERS if not p[2]]   # local provider keys
API_KEYS     = [p[1] for p in PROVIDERS if     p[2]]   # remote provider keys
PROVIDER_MAP = {p[1]: p for p in PROVIDERS}            # key → tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NLQResponse:
    question:  str
    answer:    str
    sources:   list[str]
    provider:  str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M UTC"))


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are AegisGuard AI — an expert institutional portfolio risk analyst embedded in the AegisGuard platform.

You have access to the firm's live portfolio data including zone classifications, MVO weights, regime signals, risk scores, anomalies, news sentiment, and scenario stress-test results.

When answering questions:
- Be specific: cite actual tickers, numbers, and percentages from the context
- Be concise but complete — 3-6 sentences unless detail is explicitly requested
- If the data doesn't contain what's needed, say so clearly and suggest what action to take
- Use professional financial language appropriate for institutional portfolio managers
- Never fabricate data; only use what is provided in the context
- If recommending trades, note they require human approval per the SOP"""


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio context builder
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio_context(
    filter_result=None,
    opt_result=None,
    regime_report=None,
    risk_scores: list = None,
    anomalies:   list = None,
    sentiment_summaries: list = None,
    scenario_results: list = None,
) -> str:
    """Serialise the full portfolio state into a compact context string."""
    lines = ["=== AEGISGUARD PORTFOLIO CONTEXT ===\n"]

    if filter_result:
        lines.append(f"GREEN ZONE: {', '.join(filter_result.green_zone) or 'None'}")
        lines.append(f"NO-GO ZONE: {', '.join(filter_result.no_go_zone) or 'None'}")
        lines.append(f"BLACKLISTED: {', '.join(filter_result.blacklisted) or 'None'}")

    if opt_result and opt_result.status == "optimal":
        top = sorted(opt_result.weights.items(), key=lambda x: -x[1])[:8]
        lines.append(f"TOP WEIGHTS: {', '.join(f'{t}={w*100:.1f}%' for t,w in top)}")
        lines.append(f"PORTFOLIO SHARPE: {opt_result.sharpe_ratio:.3f}")
        lines.append(f"EXPECTED RETURN: {opt_result.expected_return*100:.2f}% p.a.")
        lines.append(f"VOLATILITY: {opt_result.volatility*100:.2f}% p.a.")

    if regime_report:
        lines.append(f"MARKET STANCE: {regime_report.overall_signal}")
        lines.append(f"SPX TREND: {regime_report.trend.signal} (dist {regime_report.trend.distance_pct:+.1f}%)")
        lines.append(f"VIX: {regime_report.vix.vix_level:.1f} ({regime_report.vix.signal})")
        lines.append(f"CASH RECOMMENDATION: {regime_report.vix.suggested_cash_pct:.0f}%")

    if risk_scores:
        top_risks = risk_scores[:5]
        lines.append("TOP RISK SCORES: " + "; ".join(
            f"{s.ticker}={s.score:.0f}/100({s.band})" for s in top_risks))

    if anomalies:
        crit = [a for a in anomalies if a.severity in ("CRITICAL","HIGH")][:3]
        if crit:
            lines.append("ACTIVE ANOMALIES: " + " | ".join(a.description for a in crit))

    if sentiment_summaries:
        lines.append("NEWS SENTIMENT: " + "; ".join(
            f"{s.ticker}={s.signal}({s.avg_sentiment:+.2f})" for s in sentiment_summaries[:6]))

    if scenario_results:
        lines.append("SCENARIO STRESS TESTS:")
        for r in scenario_results:
            lines.append(f"  {r.scenario_label}: portfolio {r.portfolio_return*100:.1f}%, "
                         f"${r.portfolio_value_1m:+,.0f} on $1M, survival={r.survival_rate*100:.0f}%")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sources_from_text(text: str) -> list[str]:
    tickers = list(set(re.findall(r'\b[A-Z]{2,5}\b', text)))
    return tickers[:6]


def _openai_compat_call(
    base_url: str,
    api_key:  str,
    model:    str,
    messages: list[dict],
    timeout:  int = 30,
) -> str:
    """Shared caller for all OpenAI-compatible endpoints."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  800,
        "temperature": 0.3,
    }
    r = requests.post(f"{base_url}/chat/completions",
                      headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Per-provider call functions
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama(model: str, messages: list[dict], **_) -> str:
    """Ollama native /api/chat endpoint."""
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model or "llama3", "messages": messages, "stream": False},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _call_lmstudio(model: str, messages: list[dict], **_) -> str:
    return _openai_compat_call("http://localhost:1234/v1", "", model, messages, timeout=60)


def _call_gpt4all(model: str, messages: list[dict], **_) -> str:
    return _openai_compat_call("http://localhost:4891/v1", "", model, messages, timeout=60)


def _call_jan(model: str, messages: list[dict], **_) -> str:
    return _openai_compat_call("http://localhost:1337/v1", "", model, messages, timeout=60)


def _call_kobold(messages: list[dict], **_) -> str:
    """KoboldAI /api/v1/generate endpoint (instruction template)."""
    prompt = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in messages
    ) + "\nAssistant:"
    r = requests.post(
        "http://localhost:5001/api/v1/generate",
        json={"prompt": prompt, "max_length": 800, "temperature": 0.3},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["results"][0]["text"].strip()


def _call_llamacpp(messages: list[dict], **_) -> str:
    """llama.cpp server /v1/chat/completions (OpenAI-compatible)."""
    return _openai_compat_call("http://localhost:8080/v1", "", "", messages, timeout=60)


def _call_groq(model: str, messages: list[dict], api_key: str, **_) -> str:
    return _openai_compat_call(
        "https://api.groq.com/openai/v1", api_key,
        model or "llama3-70b-8192", messages, timeout=25,
    )


def _call_openai(model: str, messages: list[dict], api_key: str, **_) -> str:
    return _openai_compat_call(
        "https://api.openai.com/v1", api_key,
        model or "gpt-4o-mini", messages, timeout=30,
    )


def _call_anthropic(model: str, messages: list[dict], api_key: str, **_) -> str:
    """Anthropic Messages API (non-OpenAI format)."""
    # Strip system turn from messages list; pass as top-level field
    non_system = [m for m in messages if m["role"] != "system"]
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type":            "application/json",
            "x-api-key":               api_key,
            "anthropic-version":       "2023-06-01",
        },
        json={
            "model":      model or "claude-sonnet-4-20250514",
            "max_tokens": 800,
            "system":     SYSTEM_PROMPT,
            "messages":   non_system,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()


def _call_gemini(model: str, messages: list[dict], api_key: str, **_) -> str:
    """Google Gemini generateContent API."""
    # Convert messages to Gemini parts format
    parts = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        parts.append({"role": role, "parts": [{"text": m["content"]}]})

    model_id = model or "gemini-1.5-flash"
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
        params={"key": api_key},
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": parts,
            "generationConfig": {"maxOutputTokens": 800, "temperature": 0.3},
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_mistral(model: str, messages: list[dict], api_key: str, **_) -> str:
    return _openai_compat_call(
        "https://api.mistral.ai/v1", api_key,
        model or "mistral-large-latest", messages, timeout=30,
    )


def _call_together(model: str, messages: list[dict], api_key: str, **_) -> str:
    return _openai_compat_call(
        "https://api.together.xyz/v1", api_key,
        model or "meta-llama/Llama-3-70b-chat-hf", messages, timeout=30,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table
# ─────────────────────────────────────────────────────────────────────────────

_DISPATCH = {
    "ollama":    _call_ollama,
    "lmstudio":  _call_lmstudio,
    "gpt4all":   _call_gpt4all,
    "jan":       _call_jan,
    "kobold":    _call_kobold,
    "llamacpp":  _call_llamacpp,
    "groq":      _call_groq,
    "openai":    _call_openai,
    "anthropic": _call_anthropic,
    "gemini":    _call_gemini,
    "mistral":   _call_mistral,
    "together":  _call_together,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ask_portfolio(
    question:  str,
    context:   str,
    history:   list[dict] | None = None,
    provider:  str  = "ollama",
    model:     str  = "",
    api_key:   str  = "",
) -> NLQResponse:
    """
    Send a natural-language question to the selected AI provider.

    Parameters
    ----------
    question  : user's plain-English question
    context   : serialised portfolio context from build_portfolio_context()
    history   : prior chat turns (list of {"role": …, "content": …})
    provider  : one of the provider keys in PROVIDERS
    model     : model name / ID (empty = use default for provider)
    api_key   : API key (empty for local/no-API providers)
    """
    history = history or []

    # Build message list
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = f"Portfolio Context (live data):\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})

    fn = _DISPATCH.get(provider)
    if fn is None:
        return NLQResponse(
            question=question,
            answer=f"Unknown AI provider '{provider}'. Please select a valid provider in the sidebar.",
            sources=[],
            provider=provider,
        )

    try:
        answer = fn(model=model, messages=messages, api_key=api_key)
        return NLQResponse(
            question=question,
            answer=answer,
            sources=_sources_from_text(answer),
            provider=provider,
        )

    except requests.exceptions.ConnectionError:
        label = PROVIDER_MAP.get(provider, ("",))[0]
        hint  = PROVIDER_MAP.get(provider, ("","","","",""))[4]
        return NLQResponse(
            question=question,
            answer=(
                f"❌ **Cannot connect to {label}.**\n\n"
                f"**Setup:** {hint}\n\n"
                f"Make sure the local server is running before sending a question."
            ),
            sources=[],
            provider=provider,
        )

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status == 401:
            msg = f"❌ **Authentication failed** ({provider}). Check your API key in `.streamlit/secrets.toml`."
        elif status == 429:
            msg = f"⚠️ **Rate limit hit** ({provider}). Wait a moment and retry."
        else:
            msg = f"❌ **HTTP {status}** from {provider}: {e}"
        return NLQResponse(question=question, answer=msg, sources=[], provider=provider)

    except Exception as exc:
        return NLQResponse(
            question=question,
            answer=f"⚠️ AI query failed ({provider}): {exc}",
            sources=[],
            provider=provider,
        )
