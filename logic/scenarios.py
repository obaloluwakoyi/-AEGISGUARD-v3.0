"""
aegisguard/logic/scenarios.py
─────────────────────────────────────────────────────────────────────────────
Area 5 — Proactive Incident Response & Scenario Modeling

1. ScenarioEngine   — What-if market crash / rate-spike simulations
2. EscalationEngine — Automated threshold-breach workflow & notification builder
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime    import datetime
from typing      import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pre-defined Scenario Library
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_LIBRARY = {
    "2008_GFC": {
        "label":       "2008 Global Financial Crisis",
        "equity_shock":  -0.55,
        "vol_mult":       4.5,
        "rate_delta":    -0.02,
        "credit_spread":  0.06,
        "description": "Peak-to-trough drawdown from Sep 2008 to Mar 2009. S&P 500 fell 55%, VIX peaked at 80.",
    },
    "2020_COVID": {
        "label":       "2020 COVID Crash",
        "equity_shock":  -0.34,
        "vol_mult":       3.8,
        "rate_delta":    -0.015,
        "credit_spread":  0.035,
        "description": "33-day fastest bear market in history. S&P 500 -34% Feb–Mar 2020.",
    },
    "2022_RATES": {
        "label":       "2022 Rate Shock",
        "equity_shock":  -0.25,
        "vol_mult":       1.8,
        "rate_delta":     0.04,
        "credit_spread":  0.02,
        "description": "Fed raised rates 425bps in 2022. Growth/tech equities -30% to -70%.",
    },
    "MILD_CORRECTION": {
        "label":       "Mild Correction (-15%)",
        "equity_shock":  -0.15,
        "vol_mult":       1.5,
        "rate_delta":     0.0,
        "credit_spread":  0.01,
        "description": "A routine 15% market correction — typical once every 1–2 years.",
    },
    "SEVERE_BEAR": {
        "label":       "Severe Bear Market (-40%)",
        "equity_shock":  -0.40,
        "vol_mult":       3.0,
        "rate_delta":     0.0,
        "credit_spread":  0.04,
        "description": "Protracted bear market similar to dot-com bust (2000–2002).",
    },
    "CUSTOM": {
        "label":       "Custom Scenario",
        "equity_shock":   0.0,
        "vol_mult":       1.0,
        "rate_delta":     0.0,
        "credit_spread":  0.0,
        "description": "User-defined shock parameters.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AssetScenarioResult:
    ticker:         str
    current_price:  float
    shocked_price:  float
    price_change:   float
    pct_change:     float
    beta_used:      float
    breaches_stop:  bool
    stop_price:     float


@dataclass
class PortfolioScenarioResult:
    scenario_name:      str
    scenario_label:     str
    equity_shock:       float
    vol_multiplier:     float
    rate_delta:         float
    portfolio_return:   float      # expected portfolio return under scenario
    portfolio_value_1m: float      # $ value change on $1M portfolio
    max_drawdown_est:   float
    assets:             list[AssetScenarioResult] = field(default_factory=list)
    survival_rate:      float = 0.0    # % of positions above their ATR stop
    recommendation:     str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Engine
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(
    scenario_key:    str,
    weights:         dict[str, float],
    price_dict:      dict,
    exit_plans:      list,
    adj_close:       pd.DataFrame,
    custom_shock:    float = 0.0,
    custom_vol_mult: float = 1.0,
    custom_rate:     float = 0.0,
    lookback:        int   = 252,
) -> PortfolioScenarioResult:
    """
    Apply a market scenario shock to the current portfolio and compute:
      - Per-asset shocked prices
      - Portfolio-level P&L
      - How many positions breach their ATR stop
    """
    params = SCENARIO_LIBRARY.get(scenario_key, SCENARIO_LIBRARY["CUSTOM"]).copy()

    if scenario_key == "CUSTOM":
        params["equity_shock"]  = custom_shock
        params["vol_mult"]      = custom_vol_mult
        params["rate_delta"]    = custom_rate

    eq_shock  = params["equity_shock"]
    vol_mult  = params["vol_mult"]

    # Compute market betas for each asset
    betas   = _compute_betas(adj_close, lookback)

    # Build stop-price lookup
    stop_map = {ep.ticker: ep.stop_price for ep in exit_plans}

    asset_results: list[AssetScenarioResult] = []
    portfolio_return = 0.0

    for ticker, w in weights.items():
        if ticker not in price_dict:
            continue
        df      = price_dict[ticker]
        if "Adj Close" not in df.columns:
            continue

        curr_px  = float(df["Adj Close"].iloc[-1])
        beta     = betas.get(ticker, 1.0)

        # Asset-level shock = beta × market shock (simplified single-factor)
        asset_shock  = beta * eq_shock
        shocked_px   = curr_px * (1 + asset_shock)
        stop_px      = stop_map.get(ticker, 0.0)

        asset_results.append(AssetScenarioResult(
            ticker         = ticker,
            current_price  = round(curr_px, 2),
            shocked_price  = round(shocked_px, 2),
            price_change   = round(shocked_px - curr_px, 2),
            pct_change     = round(asset_shock * 100, 2),
            beta_used      = round(beta, 3),
            breaches_stop  = stop_px > 0 and shocked_px < stop_px,
            stop_price     = round(stop_px, 2),
        ))

        portfolio_return += w * asset_shock

    survivors     = sum(1 for a in asset_results if not a.breaches_stop)
    survival_rate = survivors / len(asset_results) if asset_results else 1.0

    # Recommendation
    if portfolio_return < -0.25:
        reco = f"Portfolio would lose ~{abs(portfolio_return)*100:.0f}% under this scenario. Immediate hedging required: buy SPY puts or raise cash to ≥40%."
    elif portfolio_return < -0.10:
        reco = f"Portfolio would decline ~{abs(portfolio_return)*100:.0f}%. Consider buying VIX calls or TLT for downside protection."
    else:
        reco = f"Portfolio shows resilience (~{portfolio_return*100:.1f}%). Current diversification is adequate for this stress level."

    return PortfolioScenarioResult(
        scenario_name    = scenario_key,
        scenario_label   = params["label"],
        equity_shock     = eq_shock,
        vol_multiplier   = vol_mult,
        rate_delta       = params["rate_delta"],
        portfolio_return = round(portfolio_return, 4),
        portfolio_value_1m = round(portfolio_return * 1_000_000, 0),
        max_drawdown_est = round(portfolio_return * 1.3, 4),   # rough tail estimate
        assets           = sorted(asset_results, key=lambda a: a.pct_change),
        survival_rate    = round(survival_rate, 3),
        recommendation   = reco,
    )


def _compute_betas(adj_close: pd.DataFrame, lookback: int = 252) -> dict[str, float]:
    """Compute beta vs equal-weighted index proxy for each asset."""
    if adj_close.empty or len(adj_close) < 20:
        return {}

    prices  = adj_close.tail(lookback).dropna(how="all").ffill()
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")

    # Market proxy = equal-weighted average
    market_ret = log_ret.mean(axis=1)
    mkt_var    = market_ret.var()
    if mkt_var < 1e-12:
        return {c: 1.0 for c in adj_close.columns}

    betas = {}
    for col in log_ret.columns:
        cov  = log_ret[col].cov(market_ret)
        betas[col] = round(cov / mkt_var, 3)

    return betas


def run_all_scenarios(
    weights:    dict[str, float],
    price_dict: dict,
    exit_plans: list,
    adj_close:  pd.DataFrame,
) -> list[PortfolioScenarioResult]:
    """Run all pre-defined scenarios and return sorted by severity."""
    results = []
    for key in ["MILD_CORRECTION", "2022_RATES", "2020_COVID", "SEVERE_BEAR", "2008_GFC"]:
        try:
            r = run_scenario(key, weights, price_dict, exit_plans, adj_close)
            results.append(r)
        except Exception:
            continue
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Escalation Engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EscalationEvent:
    severity:    str          # "INFO" | "WARNING" | "CRITICAL"
    category:    str          # "ZONE_CHANGE" | "VIX_SPIKE" | "ATR_BREACH" | "ANOMALY" | "SANCTIONS"
    ticker:      str
    message:     str
    action:      str
    notify:      list[str]    # roles to notify
    deadline:    str
    timestamp:   str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M UTC"))
    email_draft: str = ""


def build_escalation_events(
    filter_result,
    regime_report,
    anomalies:        list,
    sar_flags:        list,
    sanctions:        list,
) -> list[EscalationEvent]:
    """
    Scan all risk outputs and create a prioritised escalation queue.
    Each event includes: who to notify, what action to take, and deadline.
    """
    events: list[EscalationEvent] = []

    # ── CRITICAL: Blacklisted assets
    for ticker in getattr(filter_result, "blacklisted", []):
        events.append(EscalationEvent(
            severity  = "CRITICAL",
            category  = "ZONE_CHANGE",
            ticker    = ticker,
            message   = f"{ticker} has been BLACKLISTED due to volatility spike.",
            action    = "Liquidate full position. No new entries.",
            notify    = ["Head Trader", "Head of Risk", "CIO"],
            deadline  = "Within 4 hours",
            email_draft = _draft_email(ticker, "BLACKLIST", "Liquidate full position within 4 hours."),
        ))

    # ── HIGH: No-Go zone (non-blacklisted)
    no_go_only = [t for t in getattr(filter_result, "no_go_zone", [])
                  if t not in getattr(filter_result, "blacklisted", [])]
    for ticker in no_go_only:
        events.append(EscalationEvent(
            severity  = "WARNING",
            category  = "ZONE_CHANGE",
            ticker    = ticker,
            message   = f"{ticker} moved to NO-GO ZONE. New positions prohibited.",
            action    = "Halt new orders. Review existing positions for exit.",
            notify    = ["Portfolio Manager", "Head of Risk"],
            deadline  = "Before next trade",
            email_draft = _draft_email(ticker, "NO-GO", "Halt new positions. Review existing exposure."),
        ))

    # ── CRITICAL: ATR stop breaches
    if regime_report:
        for ep in getattr(regime_report, "exit_plans", []):
            if getattr(ep, "action", "") == "EXIT NOW":
                events.append(EscalationEvent(
                    severity  = "CRITICAL",
                    category  = "ATR_BREACH",
                    ticker    = ep.ticker,
                    message   = f"{ep.ticker} breached ATR stop at ${ep.stop_price:.2f}. Current: ${ep.current_price:.2f}.",
                    action    = "Execute full exit order immediately.",
                    notify    = ["Head Trader", "Portfolio Manager", "Risk"],
                    deadline  = "Within 4 hours",
                    email_draft = _draft_email(ep.ticker, "ATR_BREACH",
                                               f"Exit price ${ep.stop_price:.2f} breached. Execute full exit."),
                ))

        # ── Defensive mode
        if getattr(regime_report.vix, "is_defensive", False):
            events.append(EscalationEvent(
                severity  = "CRITICAL",
                category  = "VIX_SPIKE",
                ticker    = "^VIX",
                message   = f"VIX = {regime_report.vix.vix_level:.1f}. DEFENSIVE MODE activated.",
                action    = f"Raise cash to {regime_report.vix.suggested_cash_pct:.0f}%. No new long entries.",
                notify    = ["CIO", "Head of Risk", "Portfolio Manager"],
                deadline  = "Same trading day",
                email_draft = _draft_email("^VIX", "DEFENSIVE_MODE",
                                           f"VIX={regime_report.vix.vix_level:.1f}. Raise cash to {regime_report.vix.suggested_cash_pct:.0f}%."),
            ))

    # ── HIGH: Critical anomalies
    for anomaly in anomalies:
        if getattr(anomaly, "severity", "") in ("CRITICAL", "HIGH"):
            events.append(EscalationEvent(
                severity  = "WARNING",
                category  = "ANOMALY",
                ticker    = anomaly.ticker,
                message   = anomaly.description,
                action    = "Review position size. Check for news catalyst. Consider reducing exposure.",
                notify    = ["Portfolio Manager", "Risk Analyst"],
                deadline  = "Within 2 hours",
            ))

    # ── CRITICAL: Sanctions hits
    for sr in sanctions:
        if getattr(sr, "is_flagged", False):
            events.append(EscalationEvent(
                severity  = "CRITICAL",
                category  = "SANCTIONS",
                ticker    = sr.ticker,
                message   = f"{sr.ticker} matched OFAC sanctions screening.",
                action    = "DO NOT TRADE. Escalate to Compliance immediately.",
                notify    = ["Chief Compliance Officer", "Legal", "CIO"],
                deadline  = "Immediate",
                email_draft = _draft_email(sr.ticker, "SANCTIONS",
                                           "OFAC match detected. Do not trade. Escalate to CCO."),
            ))

    # Sort: CRITICAL first
    rank = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    return sorted(events, key=lambda e: rank.get(e.severity, 9))


def _draft_email(ticker: str, event_type: str, action: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"Subject: [AEGISGUARD ALERT] {event_type} — {ticker}\n\n"
        f"To: Risk Management Team\n"
        f"Priority: HIGH\n\n"
        f"This is an automated alert generated by AegisGuard at {ts}.\n\n"
        f"Asset:  {ticker}\n"
        f"Event:  {event_type}\n"
        f"Action Required: {action}\n\n"
        f"Please log in to the AegisGuard dashboard for full context.\n\n"
        f"— AegisGuard Automated Risk Platform v2.0"
    )
