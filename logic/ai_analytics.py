"""
aegisguard/logic/ai_analytics.py
─────────────────────────────────────────────────────────────────────────────
Area 1 — AI-Driven Predictive Analytics

Three sub-systems:
  1. AnomalyDetector      — Z-score + IQR ensemble to flag unusual price/vol behaviour
  2. PredictiveRiskScorer — Rolling momentum + macro-regime scoring (0-100)
  3. PrescriptiveEngine   — Uses the shared multi-provider AI engine (nlq.py)
                            to generate hedging / rebalancing advice.
                            Honours the provider selected in the sidebar.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Anomaly Detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Anomaly:
    ticker:       str
    date:         str
    anomaly_type: str          # "price_spike" | "vol_burst" | "volume_surge"
    severity:     str          # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    z_score:      float
    description:  str


def detect_anomalies(
    price_dict: dict[str, pd.DataFrame],
    z_threshold_medium:  float = 2.0,
    z_threshold_high:    float = 3.0,
    z_threshold_critical: float = 4.0,
    lookback: int = 60,
) -> list[Anomaly]:
    """
    Ensemble anomaly detection:
      - Z-score on daily log returns  (price anomaly)
      - Z-score on rolling 5-day vol  (volatility burst)
      - Z-score on dollar volume      (unusual liquidity event)
    """
    anomalies: list[Anomaly] = []

    for ticker, df in price_dict.items():
        if "Adj Close" not in df.columns or len(df) < lookback:
            continue

        recent   = df.tail(lookback).copy()
        log_ret  = np.log(recent["Adj Close"] / recent["Adj Close"].shift(1)).dropna()
        mu_ret   = log_ret.mean()
        sd_ret   = log_ret.std() + 1e-10

        # ── Price return anomaly
        for date, ret in log_ret.tail(5).items():
            z = (ret - mu_ret) / sd_ret
            if abs(z) >= z_threshold_medium:
                sev = ("CRITICAL" if abs(z) >= z_threshold_critical else
                       "HIGH"     if abs(z) >= z_threshold_high     else "MEDIUM")
                anomalies.append(Anomaly(
                    ticker       = ticker,
                    date         = str(date)[:10],
                    anomaly_type = "price_spike",
                    severity     = sev,
                    z_score      = round(z, 2),
                    description  = (
                        f"{ticker} posted a {ret*100:+.2f}% return on {str(date)[:10]} "
                        f"({z:+.1f}σ from the {lookback}-day mean). "
                        f"{'Extreme downside move — review immediately.' if ret < 0 else 'Unusual upside move — check for news catalyst.'}"
                    ),
                ))

        # ── Volatility burst
        roll_vol = log_ret.rolling(5).std() * np.sqrt(252)
        mu_vol   = roll_vol.mean()
        sd_vol   = roll_vol.std() + 1e-10
        latest_vol = roll_vol.iloc[-1]
        z_vol = (latest_vol - mu_vol) / sd_vol
        if abs(z_vol) >= z_threshold_medium:
            sev = ("CRITICAL" if abs(z_vol) >= z_threshold_critical else
                   "HIGH"     if abs(z_vol) >= z_threshold_high     else "MEDIUM")
            anomalies.append(Anomaly(
                ticker       = ticker,
                date         = str(roll_vol.index[-1])[:10],
                anomaly_type = "vol_burst",
                severity     = sev,
                z_score      = round(z_vol, 2),
                description  = (
                    f"{ticker} 5-day vol = {latest_vol*100:.1f}% ann. "
                    f"({z_vol:+.1f}σ above {lookback}-day avg). "
                    "Regime instability signal."
                ),
            ))

        # ── Volume surge
        if "Volume" in df.columns:
            vol_ser  = recent["Volume"].replace(0, np.nan).dropna()
            mu_v     = vol_ser.mean()
            sd_v     = vol_ser.std() + 1e-10
            latest_v = vol_ser.iloc[-1]
            z_v      = (latest_v - mu_v) / sd_v
            if z_v >= z_threshold_high:
                anomalies.append(Anomaly(
                    ticker       = ticker,
                    date         = str(vol_ser.index[-1])[:10],
                    anomaly_type = "volume_surge",
                    severity     = "HIGH" if z_v < z_threshold_critical else "CRITICAL",
                    z_score      = round(z_v, 2),
                    description  = (
                        f"{ticker} volume was {z_v:.1f}σ above average. "
                        "Possible institutional block trade or news event."
                    ),
                ))

    # Sort by severity
    sev_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    return sorted(anomalies, key=lambda a: sev_rank.get(a.severity, 9))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Predictive Risk Scorer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskScore:
    ticker:        str
    score:         float        # 0 (safe) → 100 (extreme risk)
    band:          str          # "SAFE" | "WATCH" | "ELEVATED" | "DANGER"
    momentum_z:    float
    vol_trend_z:   float
    drawdown_pct:  float
    trend_score:   float
    description:   str


def score_predictive_risk(
    price_dict: dict[str, pd.DataFrame],
    lookback_short: int = 20,
    lookback_long:  int = 126,
) -> list[RiskScore]:
    """
    Composite forward-looking risk score (0–100) built from four signals:
      1. Return momentum Z-score   (negative momentum → higher risk)
      2. Volatility trend Z-score  (rising vol → higher risk)
      3. Maximum drawdown from peak
      4. Price vs 50-day SMA distance (trend proxy)
    """
    scores: list[RiskScore] = []

    for ticker, df in price_dict.items():
        if "Adj Close" not in df.columns or len(df) < lookback_long:
            continue

        prices  = df["Adj Close"].dropna()
        log_ret = np.log(prices / prices.shift(1)).dropna()

        # Component 1 — momentum
        mom_short = log_ret.tail(lookback_short).mean()
        mom_long  = log_ret.tail(lookback_long).mean()
        mom_sd    = log_ret.tail(lookback_long).std() + 1e-10
        mom_z     = (mom_short - mom_long) / mom_sd   # negative = worsening

        # Component 2 — vol trend
        vol_short = log_ret.tail(lookback_short).std() * np.sqrt(252)
        vol_long  = log_ret.tail(lookback_long).std()  * np.sqrt(252)
        vol_trend = (vol_short - vol_long) / (vol_long + 1e-10)   # positive = rising

        # Component 3 — drawdown from 52-week high
        peak      = prices.tail(252).max()
        latest    = prices.iloc[-1]
        drawdown  = (latest - peak) / peak    # negative

        # Component 4 — SMA-50 distance
        sma50    = prices.tail(50).mean()
        sma_dist = (latest - sma50) / sma50   # negative = below SMA

        # Normalise to 0-100 (higher = more risk)
        def _clamp(x, lo=-3, hi=3):
            return max(lo, min(hi, x))

        s_mom  = (_clamp(-mom_z)    + 3) / 6 * 100     # negative momentum = high risk
        s_vol  = (_clamp(vol_trend * 5) + 3) / 6 * 100
        s_dd   = (_clamp(-drawdown * 10) + 3) / 6 * 100
        s_sma  = (_clamp(-sma_dist * 10) + 3) / 6 * 100

        composite = 0.35 * s_mom + 0.30 * s_vol + 0.20 * s_dd + 0.15 * s_sma

        if composite >= 75:
            band = "DANGER"
        elif composite >= 55:
            band = "ELEVATED"
        elif composite >= 35:
            band = "WATCH"
        else:
            band = "SAFE"

        scores.append(RiskScore(
            ticker       = ticker,
            score        = round(composite, 1),
            band         = band,
            momentum_z   = round(mom_z, 3),
            vol_trend_z  = round(vol_trend, 3),
            drawdown_pct = round(drawdown * 100, 2),
            trend_score  = round(sma_dist * 100, 2),
            description  = (
                f"{ticker} composite risk: {composite:.0f}/100 ({band}). "
                f"Momentum: {mom_z:+.2f}σ | Vol trend: {vol_trend*100:+.1f}% | "
                f"Drawdown: {drawdown*100:.1f}% | SMA-50 dist: {sma_dist*100:+.1f}%"
            ),
        ))

    return sorted(scores, key=lambda s: -s.score)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prescriptive Engine — Claude API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Prescription:
    summary:        str
    hedging_ideas:  list[str]
    rebalance_ops:  list[str]
    alt_assets:     list[str]
    sharpe_outlook: str
    raw_response:   str


def generate_prescriptions(
    filter_result,
    opt_result,
    regime_report,
    risk_scores: list[RiskScore],
    anomalies:   list[Anomaly],
    provider:  str = "ollama",
    model:     str = "",
    api_key:   str = "",
) -> Prescription:
    """
    Generate prescriptive portfolio recommendations using the selected AI provider.

    Routes through the same multi-provider engine as the NLQ analyst tab,
    so whichever provider the user picks in the sidebar applies here too.

    Parameters
    ----------
    provider  : AI provider key (e.g. 'groq', 'ollama', 'openai')
    model     : model name override (empty = use provider default)
    api_key   : API key for remote providers (empty for local)
    """
    # Import here to avoid circular dependency at module load time
    from logic.nlq import _DISPATCH, PROVIDER_MAP

    # Build a compact portfolio snapshot for the prompt
    green  = getattr(filter_result, "green_zone",  [])
    no_go  = getattr(filter_result, "no_go_zone",  [])
    blist  = getattr(filter_result, "blacklisted", [])

    weights_str = ""
    if opt_result and opt_result.status == "optimal":
        top_w = sorted(opt_result.weights.items(), key=lambda x: -x[1])[:8]
        weights_str = ", ".join(f"{t}:{w*100:.1f}%" for t, w in top_w)

    regime_str = ""
    if regime_report:
        regime_str = (
            f"Market stance: {regime_report.overall_signal}. "
            f"S&P trend: {regime_report.trend.signal} "
            f"(SMA distance: {regime_report.trend.distance_pct:+.1f}%). "
            f"VIX: {regime_report.vix.vix_level:.1f} ({regime_report.vix.signal}). "
            f"Suggested cash: {regime_report.vix.suggested_cash_pct:.0f}%."
        )

    top_risks      = [f"{s.ticker}({s.score:.0f}/100 {s.band})" for s in risk_scores[:5]]
    crit_anomalies = [a.description for a in anomalies if a.severity in ("CRITICAL","HIGH")][:4]

    user_content = f"""You are a senior institutional portfolio risk manager at a top-tier asset management firm.
You have just completed an automated risk scan. Here is the current portfolio state:

ZONE CLASSIFICATION:
- Green Zone (tradeable): {', '.join(green) or 'None'}
- No-Go Zone (restricted): {', '.join(no_go) or 'None'}
- Blacklisted (immediate exit): {', '.join(blist) or 'None'}

CURRENT MVO WEIGHTS:
{weights_str or 'Not available'}

MARKET REGIME:
{regime_str or 'Not available'}

TOP PREDICTIVE RISK SCORES (0=safe, 100=critical):
{chr(10).join(top_risks) if top_risks else 'None computed'}

ACTIVE ANOMALIES:
{chr(10).join(crit_anomalies) if crit_anomalies else 'No critical anomalies'}

Based on this data, provide a structured prescriptive analysis. Respond ONLY with a valid JSON object
(no markdown fences, no preamble) with these exact keys:
{{
  "summary": "2-3 sentence executive summary of the portfolio's current risk posture",
  "hedging_ideas": ["specific hedge 1", "specific hedge 2", "specific hedge 3"],
  "rebalance_ops": ["specific rebalance action 1", "specific rebalance action 2", "specific rebalance action 3"],
  "alt_assets": ["alternative asset or ETF 1 with rationale", "alternative asset 2", "alternative asset 3"],
  "sharpe_outlook": "1-2 sentences on expected Sharpe trajectory given current conditions"
}}

Be specific: name actual instruments (e.g. 'Buy GLD to hedge equity vol',
'Rotate XOM to ENPH given energy transition'). Do not use placeholder text."""

    messages = [{"role": "user", "content": user_content}]

    fn = _DISPATCH.get(provider)
    if fn is None:
        return Prescription(
            summary=f"Unknown AI provider '{provider}'.",
            hedging_ideas=[], rebalance_ops=[], alt_assets=[],
            sharpe_outlook="", raw_response="",
        )

    try:
        raw_text = fn(model=model, messages=messages, api_key=api_key)

        # Strip any accidental markdown fences
        clean  = raw_text.replace("```json", "").replace("```", "").strip()
        # Some models wrap JSON in extra prose — try to extract the JSON block
        brace_start = clean.find("{")
        brace_end   = clean.rfind("}")
        if brace_start != -1 and brace_end != -1:
            clean = clean[brace_start : brace_end + 1]
        parsed = json.loads(clean)

        return Prescription(
            summary        = parsed.get("summary", ""),
            hedging_ideas  = parsed.get("hedging_ideas", []),
            rebalance_ops  = parsed.get("rebalance_ops", []),
            alt_assets     = parsed.get("alt_assets", []),
            sharpe_outlook = parsed.get("sharpe_outlook", ""),
            raw_response   = raw_text,
        )

    except json.JSONDecodeError:
        # Model returned prose instead of JSON — wrap it gracefully
        return Prescription(
            summary        = raw_text[:500] if raw_text else "AI returned non-JSON output.",
            hedging_ideas  = [],
            rebalance_ops  = [],
            alt_assets     = [],
            sharpe_outlook = "",
            raw_response   = raw_text,
        )

    except Exception as exc:
        prov_info = PROVIDER_MAP.get(provider, (provider,))
        label     = prov_info[0]
        return Prescription(
            summary        = f"AI prescription unavailable via {label}: {exc}",
            hedging_ideas  = [],
            rebalance_ops  = [],
            alt_assets     = [],
            sharpe_outlook = "",
            raw_response   = str(exc),
        )
