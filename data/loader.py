"""
aegisguard/data/loader.py
─────────────────────────────────────────────────────────────────────────────
API-Agnostic Data Pipeline
Supports: yfinance (default) → AlphaVantage → Polygon.io

Each provider is wrapped in its own function. The public function
`load_price_data()` tries providers in order of preference and falls back
automatically if one fails, surfacing a clean error string to the caller
rather than crashing the application.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
import streamlit as st

# ── optional heavy import — yfinance
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── local cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_HOURS = 4          # re-fetch if cached file is older than this


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(ticker: str, source: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}_{source}.parquet"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=CACHE_TTL_HOURS)


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path)
    except Exception as exc:
        logger.warning("Cache write failed: %s", exc)


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a DatetimeIndex and an 'Adj Close' column."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Some providers return lowercase column names
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Provider: yfinance
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_yfinance(ticker: str, period_days: int = 400) -> pd.DataFrame:
    if not _YF_AVAILABLE:
        raise ImportError("yfinance is not installed.")

    end   = datetime.today()
    start = end - timedelta(days=period_days)

    t   = yf.Ticker(ticker)
    raw = t.history(start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}.")

    # yfinance with auto_adjust=True already gives adjusted prices in 'Close'
    raw = raw.rename(columns={"Close": "Adj Close"})
    return _normalise(raw[["Open", "High", "Low", "Adj Close", "Volume"]])


# ─────────────────────────────────────────────────────────────────────────────
# Provider: AlphaVantage
# ─────────────────────────────────────────────────────────────────────────────

AV_BASE = "https://www.alphavantage.co/query"


def _fetch_alphavantage(ticker: str, api_key: str) -> pd.DataFrame:
    if not api_key:
        raise ValueError("AlphaVantage API key is missing.")

    params = {
        "function":   "TIME_SERIES_DAILY_ADJUSTED",
        "symbol":     ticker,
        "outputsize": "full",
        "apikey":     api_key,
        "datatype":   "json",
    }
    resp = requests.get(AV_BASE, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if "Error Message" in payload:
        raise ValueError(f"AlphaVantage error: {payload['Error Message']}")
    if "Note" in payload:
        raise ValueError("AlphaVantage rate-limit hit. Wait 60 s or upgrade plan.")

    series = payload.get("Time Series (Daily)", {})
    if not series:
        raise ValueError(f"AlphaVantage returned empty series for {ticker}.")

    df = pd.DataFrame.from_dict(series, orient="index")
    df = df.rename(columns={
        "1. open":              "Open",
        "2. high":              "High",
        "3. low":               "Low",
        "4. close":             "Close",
        "5. adjusted close":    "Adj Close",
        "6. volume":            "Volume",
    })
    df = df[["Open", "High", "Low", "Adj Close", "Volume"]].astype(float)
    return _normalise(df)


# ─────────────────────────────────────────────────────────────────────────────
# Provider: Polygon.io
# ─────────────────────────────────────────────────────────────────────────────

POLY_BASE = "https://api.polygon.io/v2/aggs/ticker"


def _fetch_polygon(ticker: str, api_key: str, period_days: int = 400) -> pd.DataFrame:
    if not api_key:
        raise ValueError("Polygon.io API key is missing.")

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=period_days)).strftime("%Y-%m-%d")
    url   = f"{POLY_BASE}/{ticker}/range/1/day/{start}/{end}"

    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") == "ERROR":
        raise ValueError(f"Polygon error: {payload.get('error', 'unknown')}")

    results = payload.get("results", [])
    if not results:
        raise ValueError(f"Polygon returned no data for {ticker}.")

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("date")
    df = df.rename(columns={
        "o": "Open", "h": "High", "l": "Low",
        "c": "Adj Close", "v": "Volume",
    })
    return _normalise(df[["Open", "High", "Low", "Adj Close", "Volume"]])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_price_data(
    ticker: str,
    period_days: int = 400,
    preferred_source: str = "yfinance",
    av_key: str = "",
    poly_key: str = "",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame | None, str]:
    """
    Load OHLCV data for *ticker*.

    Returns
    -------
    (DataFrame | None, status_message)
        - DataFrame with DatetimeIndex and columns
          [Open, High, Low, Adj Close, Volume]
        - A human-readable status string (success or error description)
    """
    ticker = ticker.upper().strip()
    cache_key = f"{ticker}_{preferred_source}"
    cpath = _cache_path(ticker, preferred_source)

    # ── Try cache first
    if not force_refresh and _is_cache_valid(cpath):
        df = _load_cache(cpath)
        if df is not None:
            return df, f"✅ {ticker} loaded from cache ({preferred_source})"

    # ── Provider order (preferred first, then fallbacks)
    provider_order = [preferred_source]
    for p in ["yfinance", "alphavantage", "polygon"]:
        if p not in provider_order:
            provider_order.append(p)

    last_error = "Unknown error"
    for source in provider_order:
        try:
            if source == "yfinance":
                df = _fetch_yfinance(ticker, period_days)
            elif source == "alphavantage":
                df = _fetch_alphavantage(ticker, av_key)
            elif source == "polygon":
                df = _fetch_polygon(ticker, poly_key, period_days)
            else:
                continue

            _save_cache(df, _cache_path(ticker, source))
            return df, f"✅ {ticker} loaded via {source}"

        except Exception as exc:
            last_error = str(exc)
            logger.warning("Provider %s failed for %s: %s", source, ticker, exc)
            time.sleep(0.3)   # polite pause before next provider

    return None, f"❌ All providers failed for {ticker}. Last error: {last_error}"


def load_multiple(
    tickers: list[str],
    period_days: int = 400,
    **kwargs,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Load data for a list of tickers.

    Returns
    -------
    (price_dict, errors)
        price_dict — {ticker: DataFrame}
        errors     — list of human-readable error strings
    """
    price_dict: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for ticker in tickers:
        df, msg = load_price_data(ticker, period_days, **kwargs)
        if df is not None:
            price_dict[ticker] = df
        else:
            errors.append(msg)

    return price_dict, errors


def build_adj_close_matrix(price_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine a dict of per-ticker DataFrames into a single wide matrix
    of adjusted-close prices, aligned on the common date index.
    """
    frames = {
        ticker: df["Adj Close"].rename(ticker)
        for ticker, df in price_dict.items()
        if "Adj Close" in df.columns
    }
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    # Drop rows where ALL tickers are NaN, then forward-fill gaps ≤ 3 days
    combined = combined.dropna(how="all").ffill(limit=3)
    return combined
