"""
aegisguard/logic/filters.py
─────────────────────────────────────────────────────────────────────────────
Step 3 — The "No-Go" Logic: Three institutional risk filters

1. Liquidity Gate    — trade-size vs. ADV check (slippage protection)
2. Volatility Cap    — rolling 20-day σ vs. 1-year average (spike detection)
3. Correlation Matrix — pairwise ρ > 0.85 flags "fake diversification"

Each filter returns a FilterResult dataclass containing:
  - passed   : list of tickers that cleared this filter
  - rejected : list of tickers that failed
  - details  : per-ticker diagnostic dict (shown in the UI)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    filter_name: str
    passed:   list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    details:  dict      = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        total = len(self.passed) + len(self.rejected)
        return len(self.passed) / total if total else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Filter 1 — Liquidity Gate
# ─────────────────────────────────────────────────────────────────────────────

def liquidity_gate(
    price_dict: dict[str, pd.DataFrame],
    firm_trade_size_usd: float = 500_000,
    max_adv_pct: float = 0.01,          # 1 % of ADV
    adv_lookback_days: int = 20,
) -> FilterResult:
    """
    Reject any ticker where the firm's typical trade size would
    represent more than `max_adv_pct` (default 1 %) of the
    20-day Average Daily (dollar) Volume.

    Parameters
    ----------
    price_dict          : {ticker: OHLCV DataFrame}
    firm_trade_size_usd : the client's expected single-trade notional
    max_adv_pct         : maximum fraction of ADV allowed
    adv_lookback_days   : rolling window for ADV calculation
    """
    result = FilterResult(filter_name="Liquidity Gate")

    for ticker, df in price_dict.items():
        try:
            required_cols = {"Adj Close", "Volume"}
            if not required_cols.issubset(df.columns):
                result.rejected.append(ticker)
                result.details[ticker] = {"error": "Missing price/volume data"}
                continue

            recent = df.tail(adv_lookback_days).copy()
            dollar_vol = (recent["Adj Close"] * recent["Volume"]).mean()

            if dollar_vol == 0:
                result.rejected.append(ticker)
                result.details[ticker] = {"error": "Zero dollar volume — illiquid", "adv_usd": 0}
                continue

            pct_of_adv = firm_trade_size_usd / dollar_vol

            result.details[ticker] = {
                "adv_usd":        round(dollar_vol, 0),
                "trade_size_usd": firm_trade_size_usd,
                "pct_of_adv":     round(pct_of_adv * 100, 4),
                "threshold_pct":  round(max_adv_pct * 100, 2),
                "passes":         pct_of_adv <= max_adv_pct,
            }

            if pct_of_adv <= max_adv_pct:
                result.passed.append(ticker)
            else:
                result.rejected.append(ticker)

        except Exception as exc:
            result.rejected.append(ticker)
            result.details[ticker] = {"error": str(exc)}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Filter 2 — Volatility Cap
# ─────────────────────────────────────────────────────────────────────────────

def volatility_cap(
    price_dict: dict[str, pd.DataFrame],
    short_window: int = 20,
    long_window: int = 252,
    spike_multiplier: float = 1.5,
) -> FilterResult:
    """
    Compute the rolling 20-day annualised return standard-deviation.
    If the most recent 20-day σ exceeds `spike_multiplier` × (1-year σ),
    the asset is moved to the Blacklist.

    spike_multiplier = 1.5 means: flag if current vol > 150 % of annual avg.
    """
    result = FilterResult(filter_name="Volatility Cap")

    for ticker, df in price_dict.items():
        try:
            if "Adj Close" not in df.columns or len(df) < long_window:
                result.details[ticker] = {
                    "error": f"Insufficient history ({len(df)} rows, need {long_window})"
                }
                result.rejected.append(ticker)
                continue

            log_ret = np.log(df["Adj Close"] / df["Adj Close"].shift(1)).dropna()

            # Annualised σ
            sigma_short = log_ret.tail(short_window).std() * np.sqrt(252)
            sigma_long  = log_ret.tail(long_window).std()  * np.sqrt(252)

            is_spiking = sigma_short > spike_multiplier * sigma_long

            result.details[ticker] = {
                f"vol_{short_window}d_ann":  round(sigma_short * 100, 2),
                "vol_1yr_ann":               round(sigma_long  * 100, 2),
                "spike_threshold_pct":       round(spike_multiplier * sigma_long * 100, 2),
                "is_spiking":                bool(is_spiking),
                "passes":                    not is_spiking,
            }

            if not is_spiking:
                result.passed.append(ticker)
            else:
                result.rejected.append(ticker)

        except Exception as exc:
            result.rejected.append(ticker)
            result.details[ticker] = {"error": str(exc)}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Filter 3 — Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────

def correlation_matrix_filter(
    adj_close_matrix: pd.DataFrame,
    max_corr: float = 0.85,
    lookback_days: int = 252,
) -> FilterResult:
    """
    Compute pairwise Pearson correlation on log-returns.
    When two assets share ρ > max_corr, flag the one with *lower* average
    daily liquidity for removal (keeps the more tradeable asset).

    Returns a FilterResult where:
      passed   = tickers that are NOT over-correlated with any survivor
      rejected = tickers flagged as redundant
      details  = full correlation matrix + flagged pairs
    """
    result = FilterResult(filter_name="Correlation Matrix")

    if adj_close_matrix.empty or len(adj_close_matrix.columns) < 2:
        result.passed   = list(adj_close_matrix.columns)
        result.details  = {"note": "Not enough assets for correlation analysis"}
        return result

    prices   = adj_close_matrix.tail(lookback_days).dropna(how="all").ffill()
    log_rets = np.log(prices / prices.shift(1)).dropna(how="all")
    corr_mat = log_rets.corr()

    flagged_pairs: list[dict] = []
    blacklisted: set[str]     = set()

    tickers = list(corr_mat.columns)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b  = tickers[i], tickers[j]
            rho   = corr_mat.loc[a, b]
            if abs(rho) > max_corr:
                flagged_pairs.append({"asset_1": a, "asset_2": b, "rho": round(rho, 4)})
                # Keep the ticker that appears earlier in the list (higher up the
                # original watchlist); mark the other as blacklisted
                if b not in blacklisted:
                    blacklisted.add(b)

    result.passed   = [t for t in tickers if t not in blacklisted]
    result.rejected = list(blacklisted)
    result.details  = {
        "correlation_matrix": corr_mat.round(4).to_dict(),
        "flagged_pairs":      flagged_pairs,
        "max_corr_threshold": max_corr,
    }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Composite runner — runs all three filters and returns combined output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompositeFilterResult:
    liquidity:    FilterResult
    volatility:   FilterResult
    correlation:  FilterResult
    green_zone:   list[str]   = field(default_factory=list)   # passed all 3
    no_go_zone:   list[str]   = field(default_factory=list)   # failed ≥ 1
    blacklisted:  list[str]   = field(default_factory=list)   # vol spike

    def zone_of(self, ticker: str) -> str:
        if ticker in self.blacklisted:
            return "BLACKLIST"
        if ticker in self.no_go_zone:
            return "NO-GO"
        return "GREEN"


def run_all_filters(
    price_dict: dict[str, pd.DataFrame],
    adj_close_matrix: pd.DataFrame,
    firm_trade_size_usd: float = 500_000,
    max_adv_pct: float = 0.01,
    spike_multiplier: float = 1.5,
    max_corr: float = 0.85,
) -> CompositeFilterResult:
    liq  = liquidity_gate(price_dict, firm_trade_size_usd, max_adv_pct)
    vol  = volatility_cap(price_dict, spike_multiplier=spike_multiplier)
    corr = correlation_matrix_filter(adj_close_matrix, max_corr)

    blacklisted = set(vol.rejected)
    no_go       = set(liq.rejected) | blacklisted | set(corr.rejected)
    all_tickers = set(price_dict.keys())
    green       = all_tickers - no_go

    return CompositeFilterResult(
        liquidity   = liq,
        volatility  = vol,
        correlation = corr,
        green_zone  = sorted(green),
        no_go_zone  = sorted(no_go),
        blacklisted = sorted(blacklisted),
    )
