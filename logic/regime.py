"""
aegisguard/logic/regime.py
─────────────────────────────────────────────────────────────────────────────
Step 5 — Market Regime Detection & Exit Plan

Three sub-modules:
  1. TrendRegime      — 200-day SMA on S&P 500 (^GSPC)
  2. VixRegime        — VIX > 30 → Defensive Mode
  3. TrailingStopCalc — ATR-based dynamic exit prices for Green Zone assets
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from data.loader import load_price_data


# ─────────────────────────────────────────────────────────────────────────────
# 1. Market Trend — 200-day SMA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrendRegime:
    as_of_date:       date
    spx_price:        float
    sma_200:          float
    is_bull:          bool    # price > SMA200
    distance_pct:     float   # (price - SMA200) / SMA200 * 100
    signal:           str     # "BULL" | "BEAR"
    description:      str


def detect_trend_regime(spx_data: Optional[pd.DataFrame] = None) -> TrendRegime:
    """
    Load S&P 500 data and compute the 200-day SMA regime signal.
    If spx_data is not provided, fetches it live.
    """
    if spx_data is None:
        spx_data, msg = load_price_data("^GSPC", period_days=450)
        if spx_data is None:
            # Return a neutral unknown regime
            return TrendRegime(
                as_of_date   = date.today(),
                spx_price    = 0,
                sma_200      = 0,
                is_bull      = True,
                distance_pct = 0,
                signal       = "UNKNOWN",
                description  = f"Could not fetch S&P 500 data: {msg}",
            )

    prices  = spx_data["Adj Close"].dropna()
    latest  = float(prices.iloc[-1])
    sma200  = float(prices.tail(200).mean())
    gap_pct = (latest - sma200) / sma200 * 100
    is_bull = latest > sma200

    if is_bull:
        sig  = "BULL"
        desc = (
            f"S&P 500 is {abs(gap_pct):.1f}% ABOVE its 200-day SMA. "
            "Market trend is constructive. Risk-on positioning is supported."
        )
    else:
        sig  = "BEAR"
        desc = (
            f"S&P 500 is {abs(gap_pct):.1f}% BELOW its 200-day SMA. "
            "Market is in a downtrend. Reduce equity exposure, increase cash/bonds."
        )

    return TrendRegime(
        as_of_date   = prices.index[-1].date() if hasattr(prices.index[-1], "date") else date.today(),
        spx_price    = round(latest, 2),
        sma_200      = round(sma200, 2),
        is_bull      = is_bull,
        distance_pct = round(gap_pct, 2),
        signal       = sig,
        description  = desc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. VIX Regime — Defensive Mode trigger
# ─────────────────────────────────────────────────────────────────────────────

DEFENSIVE_VIX_THRESHOLD = 30.0
ELEVATED_VIX_THRESHOLD  = 20.0

@dataclass
class VixRegime:
    as_of_date:        date
    vix_level:         float
    is_defensive:      bool    # VIX > 30
    is_elevated:       bool    # VIX 20–30
    suggested_cash_pct: float  # percentage of portfolio to hold in cash
    signal:            str     # "NORMAL" | "ELEVATED" | "DEFENSIVE"
    description:       str


def detect_vix_regime(vix_data: Optional[pd.DataFrame] = None) -> VixRegime:
    if vix_data is None:
        vix_data, msg = load_price_data("^VIX", period_days=30)
        if vix_data is None:
            return VixRegime(
                as_of_date         = date.today(),
                vix_level          = 0,
                is_defensive       = False,
                is_elevated        = False,
                suggested_cash_pct = 10.0,
                signal             = "UNKNOWN",
                description        = f"Could not fetch VIX data: {msg}",
            )

    latest_vix = float(vix_data["Adj Close"].dropna().iloc[-1])

    if latest_vix > DEFENSIVE_VIX_THRESHOLD:
        signal      = "DEFENSIVE"
        cash_pct    = min(50.0, 10 + (latest_vix - DEFENSIVE_VIX_THRESHOLD) * 2)
        description = (
            f"⚠️  VIX = {latest_vix:.1f} — DEFENSIVE MODE ACTIVATED. "
            f"Suggested cash allocation: {cash_pct:.0f}%. "
            "Reduce equity risk, tighten stop-losses, avoid new long entries."
        )
    elif latest_vix > ELEVATED_VIX_THRESHOLD:
        signal      = "ELEVATED"
        cash_pct    = 15.0
        description = (
            f"VIX = {latest_vix:.1f} — volatility is elevated but not extreme. "
            f"Maintain modest cash buffer ({cash_pct:.0f}%). Monitor closely."
        )
    else:
        signal      = "NORMAL"
        cash_pct    = 5.0
        description = (
            f"VIX = {latest_vix:.1f} — volatility is within normal range. "
            f"Standard equity allocation; keep a {cash_pct:.0f}% reserve."
        )

    return VixRegime(
        as_of_date         = vix_data.index[-1].date() if hasattr(vix_data.index[-1], "date") else date.today(),
        vix_level          = round(latest_vix, 2),
        is_defensive       = latest_vix > DEFENSIVE_VIX_THRESHOLD,
        is_elevated        = latest_vix > ELEVATED_VIX_THRESHOLD,
        suggested_cash_pct = round(cash_pct, 1),
        signal             = signal,
        description        = description,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trailing Stop Calculator — ATR-based exit prices
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitPlan:
    ticker:       str
    current_price: float
    atr_14:       float          # 14-day Average True Range
    stop_price:   float          # current_price − (atr_multiplier × ATR)
    stop_pct_below: float        # how far below current price
    atr_multiplier: float
    as_of_date:   date
    action:       str            # "HOLD" | "EXIT NOW"


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute the most recent N-period Average True Range."""
    h = df["High"].values
    l = df["Low"].values
    c = df["Adj Close"].values

    if len(c) < period + 1:
        return float("nan")

    # True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    tr = np.maximum.reduce([
        h[1:] - l[1:],
        np.abs(h[1:] - c[:-1]),
        np.abs(l[1:] - c[:-1]),
    ])

    # Wilder smoothing
    atr = np.zeros(len(tr))
    atr[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return float(atr[-1]) if atr[-1] > 0 else float("nan")


def calculate_exit_plans(
    price_dict: dict[str, pd.DataFrame],
    green_zone_tickers: list[str],
    atr_period: int = 14,
    atr_multiplier: float = 2.5,
) -> list[ExitPlan]:
    """
    For each Green Zone asset, compute the ATR-based trailing stop.

    Stop Price = Latest Close − (atr_multiplier × ATR14)

    atr_multiplier = 2.5 is the institutional default (Wilder's ATR stop).
    A tighter stop (1.5 × ATR) can be used in Defensive Mode.
    """
    plans: list[ExitPlan] = []

    for ticker in green_zone_tickers:
        if ticker not in price_dict:
            continue

        df = price_dict[ticker]
        required = {"High", "Low", "Adj Close"}
        if not required.issubset(df.columns):
            continue

        try:
            recent  = df.tail(atr_period * 3)
            atr_val = _calc_atr(recent, atr_period)

            if np.isnan(atr_val):
                continue

            latest    = float(df["Adj Close"].iloc[-1])
            stop_px   = latest - atr_multiplier * atr_val
            pct_below = (latest - stop_px) / latest * 100
            action    = "HOLD" if latest > stop_px else "EXIT NOW"

            plans.append(ExitPlan(
                ticker         = ticker,
                current_price  = round(latest, 4),
                atr_14         = round(atr_val, 4),
                stop_price     = round(stop_px, 4),
                stop_pct_below = round(pct_below, 2),
                atr_multiplier = atr_multiplier,
                as_of_date     = df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
                action         = action,
            ))

        except Exception:
            continue

    return plans


# ─────────────────────────────────────────────────────────────────────────────
# Composite Regime Report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeReport:
    trend:      TrendRegime
    vix:        VixRegime
    exit_plans: list[ExitPlan]

    @property
    def overall_signal(self) -> str:
        """Combine trend + VIX into a single market stance."""
        if self.vix.is_defensive or not self.trend.is_bull:
            return "RISK-OFF"
        if self.vix.is_elevated:
            return "CAUTION"
        return "RISK-ON"

    @property
    def cash_recommendation_pct(self) -> float:
        base = self.vix.suggested_cash_pct
        if not self.trend.is_bull:
            base = max(base, 25.0)
        return base

    def exit_plan_df(self) -> pd.DataFrame:
        if not self.exit_plans:
            return pd.DataFrame()
        rows = []
        for ep in self.exit_plans:
            rows.append({
                "Ticker":           ep.ticker,
                "Current Price":    f"${ep.current_price:,.2f}",
                "ATR-14":           f"${ep.atr_14:,.3f}",
                "Exit Price":       f"${ep.stop_price:,.2f}",
                "Buffer Below":     f"{ep.stop_pct_below:.2f}%",
                "Action":           ep.action,
                "As-of Date":       str(ep.as_of_date),
            })
        return pd.DataFrame(rows)
