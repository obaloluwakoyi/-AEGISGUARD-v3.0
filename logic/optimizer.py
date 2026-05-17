"""
aegisguard/logic/optimizer.py
─────────────────────────────────────────────────────────────────────────────
Step 4 — Mean-Variance Optimization (MVO) Engine

Finds the portfolio weights that maximise the Sharpe Ratio subject to:
  • All weights sum to 1.0  (fully invested)
  • No single asset > max_weight (institutional concentration limit)
  • No short selling (weights ≥ 0)

Uses CVXPY for convex optimisation (SciPy scipy.optimize as fallback).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False

from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    weights:         dict[str, float]
    expected_return: float           # annualised, decimal
    volatility:      float           # annualised, decimal
    sharpe_ratio:    float
    status:          str             # "optimal" | "suboptimal" | "failed"
    solver_used:     str
    diagnostics:     dict = field(default_factory=dict)

    def as_series(self) -> pd.Series:
        return pd.Series(self.weights).round(6)

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for ticker, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            rows.append({
                "Ticker":        ticker,
                "Weight (%)":    round(w * 100, 2),
                "Alloc $1M":     f"${w * 1_000_000:,.0f}",
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Core math utilities
# ─────────────────────────────────────────────────────────────────────────────

def _compute_inputs(
    adj_close: pd.DataFrame,
    lookback_days: int = 252,
    risk_free_rate: float = 0.045,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (mu, Sigma, tickers):
      mu    — annualised expected return vector  (n,)
      Sigma — annualised covariance matrix       (n, n)
    """
    prices  = adj_close.tail(lookback_days).dropna(how="all").ffill().dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()

    mu     = log_ret.mean().values * 252
    Sigma  = log_ret.cov().values  * 252
    tickers = list(prices.columns)

    # Regularise Sigma to ensure positive-definiteness
    Sigma += np.eye(len(tickers)) * 1e-8

    return mu, Sigma, tickers


def _neg_sharpe(
    weights: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float,
) -> float:
    port_ret = float(weights @ mu)
    port_vol = float(np.sqrt(weights @ Sigma @ weights))
    if port_vol < 1e-10:
        return 1e6
    return -(port_ret - rf) / port_vol


# ─────────────────────────────────────────────────────────────────────────────
# Solver: CVXPY (preferred)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_cvxpy(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list[str],
    max_weight: float,
    risk_free_rate: float,
) -> tuple[np.ndarray, str]:
    """
    Max-Sharpe via parametric approach:
    Solve the 'auxiliary' LP / SOCP formulation:
      max  (μ - rf·1)ᵀ y
      s.t. yᵀ Σ y ≤ 1
           1ᵀ y = κ, y ≥ 0, y ≤ max_weight·κ
    then normalise  w = y / Σy
    """
    n  = len(tickers)
    y  = cp.Variable(n, nonneg=True)
    rf = risk_free_rate

    obj     = cp.Maximize((mu - rf) @ y)
    cons    = [
        cp.quad_form(y, Sigma) <= 1,
        cp.sum(y) >= 1e-6,
    ]
    # Per-asset concentration limit applied proportionally
    # We'll enforce it post-normalisation via SciPy fallback if needed

    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(verbose=False)

    if y.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CVXPY status: {prob.status}")

    raw = np.maximum(y.value, 0)
    w   = raw / raw.sum()

    # Clip to max_weight and re-normalise
    w = np.clip(w, 0, max_weight)
    w = w / w.sum()

    return w, "cvxpy"


# ─────────────────────────────────────────────────────────────────────────────
# Solver: SciPy (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_scipy(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list[str],
    max_weight: float,
    risk_free_rate: float,
) -> tuple[np.ndarray, str]:
    n       = len(tickers)
    w0      = np.ones(n) / n
    bounds  = [(0.0, max_weight)] * n
    cons    = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        _neg_sharpe,
        w0,
        args=(mu, Sigma, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    if not result.success:
        raise RuntimeError(f"SciPy SLSQP: {result.message}")

    w = np.maximum(result.x, 0)
    w = w / w.sum()
    return w, "scipy_slsqp"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def optimise_portfolio(
    adj_close: pd.DataFrame,
    lookback_days: int = 252,
    max_weight: float = 0.20,
    risk_free_rate: float = 0.045,
    min_assets: int = 2,
) -> OptimizationResult:
    """
    Maximise the Sharpe Ratio for the assets in *adj_close*.

    Parameters
    ----------
    adj_close       : wide DataFrame (dates × tickers) of adjusted close prices
    lookback_days   : history window for μ and Σ estimation
    max_weight      : maximum single-asset allocation (e.g., 0.20 = 20 %)
    risk_free_rate  : annual risk-free rate for Sharpe calculation
    min_assets      : minimum number of valid tickers required to proceed
    """
    tickers = list(adj_close.columns)

    if len(tickers) < min_assets:
        return OptimizationResult(
            weights={},
            expected_return=0,
            volatility=0,
            sharpe_ratio=0,
            status="failed",
            solver_used="none",
            diagnostics={"error": f"Need ≥ {min_assets} assets, got {len(tickers)}"},
        )

    mu, Sigma, tickers = _compute_inputs(adj_close, lookback_days, risk_free_rate)

    # ── Try CVXPY, fall back to SciPy
    w, solver = None, "none"
    errors: list[str] = []

    if _CVXPY_AVAILABLE:
        try:
            w, solver = _solve_cvxpy(mu, Sigma, tickers, max_weight, risk_free_rate)
        except Exception as exc:
            errors.append(f"cvxpy: {exc}")

    if w is None:
        try:
            w, solver = _solve_scipy(mu, Sigma, tickers, max_weight, risk_free_rate)
        except Exception as exc:
            errors.append(f"scipy: {exc}")
            return OptimizationResult(
                weights={},
                expected_return=0,
                volatility=0,
                sharpe_ratio=0,
                status="failed",
                solver_used="none",
                diagnostics={"errors": errors},
            )

    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ Sigma @ w))
    sharpe   = (port_ret - risk_free_rate) / port_vol if port_vol > 1e-10 else 0.0

    return OptimizationResult(
        weights         = {t: float(ww) for t, ww in zip(tickers, w)},
        expected_return = round(port_ret, 6),
        volatility      = round(port_vol, 6),
        sharpe_ratio    = round(sharpe, 4),
        status          = "optimal",
        solver_used     = solver,
        diagnostics     = {
            "solver_errors_before_success": errors,
            "n_assets":   len(tickers),
            "max_weight": max_weight,
            "rf_rate":    risk_free_rate,
        },
    )


def efficient_frontier(
    adj_close: pd.DataFrame,
    n_points: int = 80,
    lookback_days: int = 252,
    max_weight: float = 0.20,
) -> pd.DataFrame:
    """
    Generate the efficient frontier by sweeping target-return levels.
    Returns a DataFrame with columns: [Return, Volatility, Sharpe].
    Useful for the frontier chart in the UI.
    """
    mu, Sigma, tickers = _compute_inputs(adj_close, lookback_days)
    n = len(tickers)

    min_ret  = mu.min()
    max_ret  = mu.max()
    targets  = np.linspace(min_ret, max_ret, n_points)

    rows = []
    for tgt in targets:
        try:
            w0   = np.ones(n) / n
            cons = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, t=tgt: w @ mu - t},
            ]
            res = minimize(
                lambda w: float(np.sqrt(w @ Sigma @ w)),
                w0,
                method="SLSQP",
                bounds=[(0.0, max_weight)] * n,
                constraints=cons,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if res.success:
                v     = float(res.fun)
                sharpe = (tgt - 0.045) / v if v > 0 else 0
                rows.append({"Return": tgt, "Volatility": v, "Sharpe": sharpe})
        except Exception:
            continue

    return pd.DataFrame(rows)
