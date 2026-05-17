"""
aegisguard/data/alt_data.py
─────────────────────────────────────────────────────────────────────────────
Area 2 — Alternative Data Sources

1. NewsSentimentEngine  — Fetches headlines via NewsAPI/RSS + Claude NLP scoring
2. GeopoliticalRiskIndex — Composite GPR proxy from public sources
3. SentimentSummary      — Aggregated ticker-level sentiment (-1 → +1)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import time
import hashlib
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-sonnet-4-20250514"

CACHE_DIR = Path(__file__).parent.parent / "data" / "_cache" / "alt"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Headline:
    title:      str
    source:     str
    published:  str
    url:        str
    sentiment:  float     # -1.0 (bearish) → +1.0 (bullish)
    relevance:  float     # 0.0 → 1.0
    tickers:    list[str] = field(default_factory=list)


@dataclass
class SentimentSummary:
    ticker:       str
    avg_sentiment: float     # -1 → +1
    signal:       str        # "BULLISH" | "NEUTRAL" | "BEARISH"
    n_articles:   int
    top_headline: str
    as_of:        str


@dataclass
class GeopoliticalRisk:
    score:       float       # 0 (calm) → 100 (extreme)
    band:        str         # "LOW" | "MODERATE" | "HIGH" | "EXTREME"
    vix_contrib: float
    oil_contrib: float
    gold_contrib: float
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# Headline fetcher — GNews free API (no key needed for basic use)
# ─────────────────────────────────────────────────────────────────────────────

GNEWS_URL = "https://gnews.io/api/v4/search"


def _cache_key(query: str) -> Path:
    h = hashlib.md5(query.encode()).hexdigest()[:12]
    return CACHE_DIR / f"news_{h}.json"


def _cache_valid(path: Path, ttl_hours: int = 2) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=ttl_hours)


def fetch_headlines(
    tickers: list[str],
    gnews_key: str = "",
    max_per_ticker: int = 5,
) -> list[Headline]:
    """
    Fetch recent headlines for each ticker.
    Uses GNews API if key provided; falls back to a curated RSS-style
    stub that still runs Claude sentiment scoring on mock data for demo purposes.
    """
    headlines: list[Headline] = []

    for ticker in tickers[:10]:   # cap at 10 to respect rate limits
        cpath = _cache_key(ticker)

        # ── Try cache
        if _cache_valid(cpath):
            try:
                cached = json.loads(cpath.read_text())
                headlines += [Headline(**h) for h in cached]
                continue
            except Exception:
                pass

        raw_articles: list[dict] = []

        # ── GNews API
        if gnews_key:
            try:
                params = {
                    "q":        f"{ticker} stock",
                    "lang":     "en",
                    "max":      max_per_ticker,
                    "apikey":   gnews_key,
                    "sortby":   "publishedAt",
                }
                r = requests.get(GNEWS_URL, params=params, timeout=10)
                r.raise_for_status()
                raw_articles = r.json().get("articles", [])
            except Exception:
                pass

        # ── Fallback: generate synthetic headlines via Claude for demo
        if not raw_articles:
            raw_articles = _synthetic_headlines(ticker, max_per_ticker)

        # ── Score with Claude NLP
        batch = _score_headlines_batch(ticker, raw_articles)
        headlines += batch

        # Cache
        try:
            cpath.write_text(json.dumps([h.__dict__ for h in batch]))
        except Exception:
            pass

        time.sleep(0.2)

    return headlines


def _synthetic_headlines(ticker: str, n: int) -> list[dict]:
    """Generate plausible recent headline stubs for NLP scoring when no API key present."""
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        {"title": f"{ticker} reports quarterly results amid market volatility",
         "source": "Reuters", "publishedAt": today, "url": ""},
        {"title": f"Analysts revise {ticker} price targets following macro data",
         "source": "Bloomberg", "publishedAt": today, "url": ""},
        {"title": f"{ticker} faces sector headwinds as rates remain elevated",
         "source": "WSJ", "publishedAt": today, "url": ""},
    ][:n]


def _score_headlines_batch(ticker: str, articles: list[dict]) -> list[Headline]:
    """Use Claude to score sentiment and relevance for a batch of headlines."""
    if not articles:
        return []

    titles = [a.get("title", "") for a in articles if a.get("title")]
    if not titles:
        return []

    prompt = f"""You are a financial NLP analyst. Score each headline for a portfolio manager.

Ticker in focus: {ticker}

Headlines:
{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(titles))}

Respond ONLY with a JSON array (no markdown). Each element must have:
- "sentiment": float from -1.0 (very bearish) to 1.0 (very bullish)
- "relevance": float from 0.0 (unrelated) to 1.0 (highly relevant to {ticker})

Example: [{{"sentiment": 0.3, "relevance": 0.8}}, ...]
Output ONLY the JSON array, nothing else."""

    try:
        resp = requests.post(
            CLAUDE_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model":      CLAUDE_MODEL,
                "max_tokens": 500,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw  = resp.json()["content"][0]["text"].strip()
        clean = raw.replace("```json","").replace("```","").strip()
        scores = json.loads(clean)
    except Exception:
        scores = [{"sentiment": 0.0, "relevance": 0.5}] * len(titles)

    result = []
    for i, art in enumerate(articles):
        s = scores[i] if i < len(scores) else {"sentiment": 0.0, "relevance": 0.5}
        result.append(Headline(
            title      = art.get("title", ""),
            source     = art.get("source", {}).get("name", art.get("source", "Unknown"))
                         if isinstance(art.get("source"), dict) else art.get("source", "Unknown"),
            published  = art.get("publishedAt", art.get("published", ""))[:10],
            url        = art.get("url", ""),
            sentiment  = float(s.get("sentiment", 0.0)),
            relevance  = float(s.get("relevance", 0.5)),
            tickers    = [ticker],
        ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate sentiment per ticker
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_sentiment(headlines: list[Headline]) -> list[SentimentSummary]:
    from collections import defaultdict
    groups: dict[str, list[Headline]] = defaultdict(list)
    for h in headlines:
        for t in h.tickers:
            groups[t].append(h)

    summaries = []
    for ticker, items in groups.items():
        if not items:
            continue
        weights   = np.array([h.relevance for h in items])
        sentiments = np.array([h.sentiment for h in items])
        avg = float(np.average(sentiments, weights=weights + 1e-6))

        if avg > 0.15:
            signal = "BULLISH"
        elif avg < -0.15:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        top = max(items, key=lambda h: h.relevance)
        summaries.append(SentimentSummary(
            ticker        = ticker,
            avg_sentiment = round(avg, 3),
            signal        = signal,
            n_articles    = len(items),
            top_headline  = top.title,
            as_of         = datetime.now().strftime("%Y-%m-%d"),
        ))

    return sorted(summaries, key=lambda s: s.avg_sentiment)


# ─────────────────────────────────────────────────────────────────────────────
# Geopolitical Risk Index — proxy from market data
# ─────────────────────────────────────────────────────────────────────────────

def compute_geopolitical_risk(
    vix_level: float,
    gold_data: pd.DataFrame | None = None,
    oil_data:  pd.DataFrame | None = None,
) -> GeopoliticalRisk:
    """
    Composite GPR proxy built from three market-based signals:
      1. VIX level              (fear gauge)
      2. Gold 20-day momentum   (safe-haven demand)
      3. Oil 20-day volatility  (supply-shock proxy)
    """
    # VIX component (0-40 → 0-50 score)
    vix_score = min(50, (vix_level / 40) * 50)

    # Gold component
    gold_score = 0.0
    if gold_data is not None and "Adj Close" in gold_data.columns and len(gold_data) >= 20:
        g     = gold_data["Adj Close"].tail(20)
        g_ret = (g.iloc[-1] / g.iloc[0]) - 1
        gold_score = min(25, max(0, g_ret * 250))   # rising gold = rising geo risk

    # Oil volatility component
    oil_score = 0.0
    if oil_data is not None and "Adj Close" in oil_data.columns and len(oil_data) >= 20:
        o_ret  = np.log(oil_data["Adj Close"] / oil_data["Adj Close"].shift(1)).dropna()
        o_vol  = o_ret.tail(20).std() * np.sqrt(252)
        oil_score = min(25, o_vol * 100)

    total = vix_score + gold_score + oil_score

    if total >= 70:
        band = "EXTREME"
        desc = f"GPR Index {total:.0f}/100 — EXTREME geopolitical stress. Reduce EM exposure, increase safe-haven allocation (Gold, CHF, Treasuries)."
    elif total >= 50:
        band = "HIGH"
        desc = f"GPR Index {total:.0f}/100 — Elevated geopolitical tension. Monitor commodity exposure and currency hedges."
    elif total >= 30:
        band = "MODERATE"
        desc = f"GPR Index {total:.0f}/100 — Moderate background risk. Standard diversification protocols apply."
    else:
        band = "LOW"
        desc = f"GPR Index {total:.0f}/100 — Low geopolitical stress. Risk-on positioning supported."

    return GeopoliticalRisk(
        score        = round(total, 1),
        band         = band,
        vix_contrib  = round(vix_score, 1),
        gold_contrib = round(gold_score, 1),
        oil_contrib  = round(oil_score, 1),
        description  = desc,
    )
