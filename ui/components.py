"""
aegisguard/ui/components.py
─────────────────────────────────────────────────────────────────────────────
Step 6 — Professional Bloomberg-style UI components

Every chart uses the shared DARK_TEMPLATE so the visual language is
consistent across the whole dashboard.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Shared theme
# ─────────────────────────────────────────────────────────────────────────────

BG_DEEP   = "#0A0E1A"
BG_PANEL  = "#0F1629"
BG_CARD   = "#141B2D"
ACCENT    = "#00D4FF"
GREEN     = "#00FF88"
YELLOW    = "#FFD600"
RED       = "#FF3860"
ORANGE    = "#FF6B35"
TEXT_PRI  = "#E8EAF0"
TEXT_SEC  = "#8892A4"
GRID      = "#1E2A3E"

DARK_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor = BG_DEEP,
        plot_bgcolor  = BG_PANEL,
        font          = dict(family="'JetBrains Mono', 'Courier New', monospace",
                             color=TEXT_PRI, size=12),
        xaxis         = dict(gridcolor=GRID, zerolinecolor=GRID,
                             tickfont=dict(color=TEXT_SEC)),
        yaxis         = dict(gridcolor=GRID, zerolinecolor=GRID,
                             tickfont=dict(color=TEXT_SEC)),
        legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_PRI)),
        margin        = dict(l=50, r=30, t=50, b=40),
        hoverlabel    = dict(bgcolor=BG_CARD, font_color=TEXT_PRI,
                             bordercolor=ACCENT),
    )
)


def _fig_base(title: str = "", height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=title, font=dict(color=ACCENT, size=14), x=0.01),
        height=height,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Status badges
# ─────────────────────────────────────────────────────────────────────────────

_BADGE_CSS = """
<style>
.ag-badge {
    display:inline-block; padding:4px 12px; border-radius:4px;
    font-family:'JetBrains Mono',monospace; font-size:0.78rem;
    font-weight:700; letter-spacing:0.08em; margin:2px 4px;
}
.ag-green   { background:#003320; color:#00FF88; border:1px solid #00FF88; }
.ag-red     { background:#2D0010; color:#FF3860; border:1px solid #FF3860; }
.ag-yellow  { background:#2D2200; color:#FFD600; border:1px solid #FFD600; }
.ag-blue    { background:#001A2D; color:#00D4FF; border:1px solid #00D4FF; }
.ag-orange  { background:#2D1500; color:#FF6B35; border:1px solid #FF6B35; }
.ag-kpi { background:#0F1629; border:1px solid #1E2A3E; border-radius:8px;
          padding:16px 20px; margin:4px; text-align:center; }
.ag-kpi-val { font-size:1.8rem; font-weight:700; font-family:monospace; }
.ag-kpi-lbl { font-size:0.72rem; color:#8892A4; letter-spacing:0.1em; margin-top:4px; }
</style>
"""


def inject_css() -> None:
    st.markdown(_BADGE_CSS, unsafe_allow_html=True)


def badge(label: str, style: str = "blue") -> str:
    """Return an HTML badge string. style: green | red | yellow | blue | orange"""
    return f'<span class="ag-badge ag-{style}">{label}</span>'


def zone_badge(zone: str) -> str:
    mapping = {
        "GREEN":     ("● GREEN ZONE",    "green"),
        "NO-GO":     ("✖ NO-GO ZONE",    "red"),
        "BLACKLIST": ("⬛ BLACKLISTED",   "orange"),
    }
    label, style = mapping.get(zone, (zone, "blue"))
    return badge(label, style)


def regime_badge(signal: str) -> str:
    mapping = {
        "RISK-ON":   ("▲ RISK-ON",          "green"),
        "CAUTION":   ("◆ CAUTION",           "yellow"),
        "RISK-OFF":  ("▼ RISK-OFF",          "red"),
        "DEFENSIVE": ("⚠ DEFENSIVE MODE",   "orange"),
        "BULL":      ("▲ BULL TREND",        "green"),
        "BEAR":      ("▼ BEAR TREND",        "red"),
        "NORMAL":    ("● VIX NORMAL",        "green"),
        "ELEVATED":  ("◆ VIX ELEVATED",      "yellow"),
    }
    label, style = mapping.get(signal, (signal, "blue"))
    return badge(label, style)


def kpi_card(value: str, label: str, color: str = ACCENT) -> str:
    return (
        f'<div class="ag-kpi">'
        f'<div class="ag-kpi-val" style="color:{color}">{value}</div>'
        f'<div class="ag-kpi-lbl">{label}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Price + 200-SMA with zone shading
# ─────────────────────────────────────────────────────────────────────────────

def chart_price_sma(
    price_series: pd.Series,
    ticker: str,
    sma_period: int = 200,
    height: int = 380,
) -> go.Figure:
    prices = price_series.dropna()
    sma    = prices.rolling(sma_period, min_periods=sma_period).mean()

    fig = _fig_base(f"{ticker} — Price vs {sma_period}-Day SMA", height)

    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        mode="lines", name="Price",
        line=dict(color=ACCENT, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sma.index, y=sma.values,
        mode="lines", name=f"{sma_period}-Day SMA",
        line=dict(color=YELLOW, width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>SMA: $%{y:,.2f}<extra></extra>",
    ))

    # Shade bear periods
    below_sma = prices < sma
    in_bear   = False
    start_idx = None
    dates     = prices.index.tolist()

    for i, d in enumerate(dates):
        if below_sma.iloc[i] and not in_bear:
            in_bear   = True
            start_idx = d
        elif not below_sma.iloc[i] and in_bear:
            fig.add_vrect(
                x0=start_idx, x1=d,
                fillcolor="rgba(255,56,96,0.08)",
                layer="below", line_width=0,
            )
            in_bear = False
    if in_bear:
        fig.add_vrect(
            x0=start_idx, x1=dates[-1],
            fillcolor="rgba(255,56,96,0.08)",
            layer="below", line_width=0,
        )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Volatility rolling window
# ─────────────────────────────────────────────────────────────────────────────

def chart_volatility(
    price_series: pd.Series,
    ticker: str,
    short_window: int = 20,
    long_window: int = 252,
    height: int = 320,
) -> go.Figure:
    log_ret    = np.log(price_series / price_series.shift(1)).dropna()
    vol_short  = log_ret.rolling(short_window).std() * np.sqrt(252) * 100
    vol_long   = log_ret.rolling(long_window).std()  * np.sqrt(252) * 100

    fig = _fig_base(f"{ticker} — Rolling Annualised Volatility (%)", height)

    fig.add_trace(go.Scatter(
        x=vol_short.index, y=vol_short.values,
        mode="lines", name=f"{short_window}-Day Vol",
        line=dict(color=ACCENT, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=vol_long.index, y=vol_long.values,
        mode="lines", name=f"{long_window}-Day Vol",
        line=dict(color=YELLOW, width=1.5, dash="dot"),
    ))

    # Mark spike threshold (1.5 × long vol)
    threshold = (vol_long * 1.5).dropna()
    fig.add_trace(go.Scatter(
        x=threshold.index, y=threshold.values,
        mode="lines", name="Spike Threshold (1.5×)",
        line=dict(color=RED, width=1, dash="dash"),
    ))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def chart_correlation_heatmap(corr_dict: dict, height: int = 450) -> go.Figure:
    corr_df = pd.DataFrame(corr_dict)
    tickers = list(corr_df.columns)

    fig = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=tickers, y=tickers,
        colorscale=[
            [0.0,  "#1a0a2e"],
            [0.35, "#0A0E1A"],
            [0.6,  "#002D3A"],
            [0.85, "#003344"],
            [1.0,  "#00D4FF"],
        ],
        zmin=-1, zmax=1,
        text=corr_df.round(2).values,
        texttemplate="%{text}",
        hovertemplate="%{x} / %{y}<br>ρ = %{z:.4f}<extra></extra>",
        colorbar=dict(
            tickfont=dict(color=TEXT_SEC),
            outlinecolor=GRID, outlinewidth=1,
        ),
    ))

    # Draw threshold border
    n = len(tickers)
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr_df.values[i][j]) > 0.85:
                fig.add_shape(
                    type="rect",
                    x0=j - 0.5, x1=j + 0.5,
                    y0=i - 0.5, y1=i + 0.5,
                    line=dict(color=RED, width=2),
                    layer="above",
                )

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text="Correlation Matrix (red = >0.85 threshold)", font=dict(color=ACCENT, size=14)),
        height=height,
        xaxis=dict(tickangle=-45),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Portfolio weights donut
# ─────────────────────────────────────────────────────────────────────────────

def chart_weights_donut(weights: dict[str, float], height: int = 380) -> go.Figure:
    labels  = list(weights.keys())
    values  = [w * 100 for w in weights.values()]
    colours = px.colors.qualitative.Bold[:len(labels)]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colours, line=dict(color=BG_DEEP, width=2)),
        hovertemplate="%{label}<br>%{value:.2f}%<extra></extra>",
        textinfo="label+percent",
        textfont=dict(color=TEXT_PRI, size=11),
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text="Optimal Portfolio Allocation", font=dict(color=ACCENT, size=14)),
        height=height,
        showlegend=False,
        annotations=[dict(
            text="MVO", x=0.5, y=0.5,
            font=dict(size=16, color=ACCENT, family="monospace"),
            showarrow=False,
        )],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Efficient frontier scatter
# ─────────────────────────────────────────────────────────────────────────────

def chart_efficient_frontier(
    frontier_df: pd.DataFrame,
    optimal_return: float,
    optimal_vol: float,
    height: int = 400,
) -> go.Figure:
    if frontier_df.empty:
        return _fig_base("Efficient Frontier — Insufficient Data", height)

    fig = _fig_base("Efficient Frontier", height)

    # Colour by Sharpe
    fig.add_trace(go.Scatter(
        x=frontier_df["Volatility"] * 100,
        y=frontier_df["Return"]     * 100,
        mode="lines+markers",
        name="Frontier",
        marker=dict(
            color=frontier_df["Sharpe"],
            colorscale=[[0, "#001A2D"], [0.5, ACCENT], [1, GREEN]],
            size=5,
            showscale=True,
            colorbar=dict(title="Sharpe", tickfont=dict(color=TEXT_SEC)),
        ),
        line=dict(color=ACCENT, width=1),
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
    ))

    # Mark optimal point
    fig.add_trace(go.Scatter(
        x=[optimal_vol * 100],
        y=[optimal_return * 100],
        mode="markers",
        name="Optimal (Max Sharpe)",
        marker=dict(color=GREEN, size=14, symbol="star",
                    line=dict(color=BG_DEEP, width=2)),
        hovertemplate=f"OPTIMAL<br>Vol: {optimal_vol*100:.2f}%<br>Return: {optimal_return*100:.2f}%<extra></extra>",
    ))

    fig.update_xaxes(title_text="Annualised Volatility (%)", title_font=dict(color=TEXT_SEC))
    fig.update_yaxes(title_text="Annualised Return (%)",     title_font=dict(color=TEXT_SEC))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Cumulative return comparison
# ─────────────────────────────────────────────────────────────────────────────

def chart_cumulative_returns(
    adj_close: pd.DataFrame,
    weights: dict[str, float],
    height: int = 400,
) -> go.Figure:
    fig = _fig_base("Cumulative Return — Portfolio vs Individual Assets", height)

    colours = px.colors.qualitative.Bold
    norm    = adj_close / adj_close.iloc[0] * 100

    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col],
            mode="lines", name=col,
            line=dict(color=colours[i % len(colours)], width=1, dash="dot"),
            opacity=0.55,
            hovertemplate=f"{col}: %{{y:.1f}}<extra></extra>",
        ))

    # Portfolio weighted line
    w_array  = np.array([weights.get(c, 0) for c in adj_close.columns])
    port_ret = (adj_close.pct_change().fillna(0) * w_array).sum(axis=1)
    port_cum = (1 + port_ret).cumprod() * 100

    fig.add_trace(go.Scatter(
        x=port_cum.index, y=port_cum.values,
        mode="lines", name="Portfolio (MVO)",
        line=dict(color=GREEN, width=2.5),
        hovertemplate="Portfolio: %{y:.1f}<extra></extra>",
    ))

    fig.update_yaxes(title_text="Index (Base=100)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart: VIX time series with threshold bands
# ─────────────────────────────────────────────────────────────────────────────

def chart_vix(vix_data: pd.DataFrame, height: int = 300) -> go.Figure:
    prices = vix_data["Adj Close"].dropna()
    fig    = _fig_base("CBOE Volatility Index (VIX)", height)

    fig.add_hrect(y0=30, y1=100, fillcolor="rgba(255,56,96,0.07)", layer="below", line_width=0)
    fig.add_hrect(y0=20, y1=30,  fillcolor="rgba(255,214,0,0.05)", layer="below", line_width=0)

    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        mode="lines", name="VIX",
        line=dict(color=ACCENT, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.05)",
    ))

    fig.add_hline(y=30, line_color=RED,    line_dash="dash", annotation_text="Defensive (30)",
                  annotation_font_color=RED)
    fig.add_hline(y=20, line_color=YELLOW, line_dash="dash", annotation_text="Elevated (20)",
                  annotation_font_color=YELLOW)

    return fig
