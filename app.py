"""
aegisguard/app.py  —  AegisGuard v3.0
─────────────────────────────────────────────────────────────────────────────
Global Institutional Portfolio Risk Dashboard
Run: streamlit run app.py
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from data.loader       import load_multiple, build_adj_close_matrix
from data.universe     import search_assets, get_regions, get_sectors
from data.alt_data     import fetch_headlines, aggregate_sentiment, compute_geopolitical_risk
from logic.filters     import run_all_filters
from logic.optimizer   import optimise_portfolio, efficient_frontier
from logic.regime      import detect_trend_regime, detect_vix_regime, calculate_exit_plans, RegimeReport
from logic.ai_analytics import detect_anomalies, score_predictive_risk, generate_prescriptions
from logic.compliance   import (build_regulatory_mapping, regulatory_coverage_stats,
                                audit_model_bias, ofac_screen_tickers, detect_sar_patterns)
from logic.scenarios    import run_scenario, run_all_scenarios, build_escalation_events
from logic.nlq          import build_portfolio_context, ask_portfolio, PROVIDERS, PROVIDER_MAP, NO_API_KEYS
from ui.components      import (
    inject_css, badge, zone_badge, regime_badge, kpi_card,
    chart_price_sma, chart_volatility, chart_correlation_heatmap,
    chart_weights_donut, chart_efficient_frontier, chart_cumulative_returns,
    chart_vix, GREEN, RED, YELLOW, ACCENT, ORANGE,
)
from docs.report_generator import generate_report_pdf
from logic.emailer         import (
    send_report, build_html_body, SMTP_PRESETS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AegisGuard | Global Risk Dashboard",
    page_icon="🛡️", layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────
def _authenticate() -> bool:
    try:    correct = st.secrets["auth"]["password"]
    except: correct = "Kayode"
    if st.session_state.get("authenticated"):
        return True
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;min-height:65vh">
    <div style="background:#0F1629;border:1px solid #1E2A3E;border-radius:12px;
                padding:48px 56px;max-width:420px;width:100%;text-align:center">
      <div style="font-size:2.8rem">🛡️</div>
      <div style="font-size:1.5rem;font-weight:700;color:#00D4FF;font-family:monospace;
                  letter-spacing:0.06em;margin:8px 0 4px">AEGISGUARD</div>
      <div style="font-size:0.7rem;color:#8892A4;letter-spacing:0.15em;margin-bottom:32px">
        GLOBAL INSTITUTIONAL RISK PLATFORM v3.0
      </div>
    </div></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        pwd = st.text_input("Access Code", type="password",
                            label_visibility="collapsed", placeholder="Enter access code…")
        if st.button("AUTHENTICATE", use_container_width=True, type="primary"):
            if pwd == correct:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid access code.")
    return False

if not _authenticate():
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px">
      <span style="font-size:1.4rem;font-weight:900;color:#00D4FF;font-family:monospace">
        🛡️ AEGISGUARD v3.0
      </span><br>
      <span style="font-size:0.62rem;color:#8892A4;letter-spacing:0.16em">
        GLOBAL RISK INTELLIGENCE
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🌍 Global Asset Search")
    rc, sc = st.columns(2)
    with rc: region_filter = st.selectbox("Region", get_regions(), label_visibility="collapsed")
    with sc: sector_filter = st.selectbox("Sector", get_sectors(), label_visibility="collapsed")
    search_q = st.text_input("Search", placeholder="Apple, NVDA, Tencent, GLD…")
    if search_q or region_filter != "All" or sector_filter != "All":
        hits = search_assets(search_q, region_filter, sector_filter, limit=10)
        for h in hits[:8]:
            st.markdown(
                f'<span style="color:#00D4FF;font-family:monospace;font-size:0.78rem">{h["ticker"]}</span>'
                f' <span style="color:#8892A4;font-size:0.7rem">{h["name"][:26]}</span>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Watchlist")
    raw_input = st.text_area(
        "Tickers (comma-separated)\nAny global exchange supported",
        value="AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, V, JNJ, XOM",
        height=110,
    )
    tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]

    st.markdown("---")
    st.markdown("### ⚙️ Risk Parameters")
    trade_size  = st.number_input("Firm Trade Size (USD)", 10_000, 50_000_000, 500_000, 50_000, format="%d")
    max_adv_pct = st.slider("Max % of ADV", 0.1, 5.0, 1.0, 0.1) / 100
    max_weight  = st.slider("Max Asset Weight (%)", 5, 50, 20, 5) / 100
    max_corr    = st.slider("Max Correlation", 0.50, 0.99, 0.85, 0.01)
    spike_mult  = st.slider("Vol Spike Multiplier", 1.0, 3.0, 1.5, 0.1)
    risk_free   = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.25) / 100

    st.markdown("---")
    st.markdown("### 🔌 Data & AI")
    try:
        pref_src  = st.secrets["api_keys"].get("preferred_source", "yfinance")
        av_key    = st.secrets["api_keys"].get("alpha_vantage", "")
        poly_key  = st.secrets["api_keys"].get("polygon", "")
        gnews_key = st.secrets["api_keys"].get("gnews", "")
    except:
        pref_src = "yfinance"; av_key = ""; poly_key = ""; gnews_key = ""

    src_map = {"yfinance":"Yahoo Finance","alphavantage":"AlphaVantage","polygon":"Polygon.io"}
    st.markdown(f'Source: {badge(src_map.get(pref_src, pref_src), "blue")}', unsafe_allow_html=True)
    enable_ai        = st.toggle("🤖 AI Prescriptions",  value=True)
    enable_sentiment = st.toggle("📰 News Sentiment NLP", value=True)
    enable_scenarios = st.toggle("📉 Scenario Modeling",  value=True)

    # ── AI Provider Picker ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 AI Analyst Provider")

    _no_api_labels  = [p[0] for p in PROVIDERS if not p[2]]
    _api_labels     = [p[0] for p in PROVIDERS if     p[2]]
    _provider_keys  = [p[1] for p in PROVIDERS]
    _provider_labels= [p[0] for p in PROVIDERS]

    st.markdown('<span style="font-size:0.72rem;color:#8892A4">🟢 No API Key Required</span>',
                unsafe_allow_html=True)
    for p in PROVIDERS:
        if not p[2]:
            st.markdown(
                f'<span style="font-size:0.68rem;color:#4A5568;padding-left:8px">• {p[0]}</span>'
                f'<span style="font-size:0.62rem;color:#2D3748;padding-left:4px">({p[4].split("—")[0].strip()})</span>',
                unsafe_allow_html=True)

    st.markdown('<span style="font-size:0.72rem;color:#8892A4">🔑 API Key Required</span>',
                unsafe_allow_html=True)
    for p in PROVIDERS:
        if p[2]:
            st.markdown(
                f'<span style="font-size:0.68rem;color:#4A5568;padding-left:8px">• {p[0]}</span>',
                unsafe_allow_html=True)

    ai_provider_label = st.selectbox(
        "Select AI Provider",
        options=_provider_labels,
        index=0,
        help="Local providers need no API key. API providers need a key below.",
        label_visibility="collapsed",
    )
    # Resolve selected provider key
    _sel_idx     = _provider_labels.index(ai_provider_label)
    ai_provider  = _provider_keys[_sel_idx]
    _prov_info   = PROVIDERS[_sel_idx]
    _needs_key   = _prov_info[2]
    _default_mdl = _prov_info[3]
    _hint        = _prov_info[4]

    st.caption(f"ℹ️ {_hint}")

    # Model override
    ai_model = st.text_input(
        "Model name (optional override)",
        value=_default_mdl,
        placeholder=_default_mdl,
        label_visibility="visible",
    )

    # API key field (hidden for local providers)
    ai_api_key = ""
    if _needs_key:
        # Try loading from secrets first
        try:
            _secret_key = st.secrets["ai_providers"].get(ai_provider, "")
        except:
            _secret_key = ""
        ai_api_key = st.text_input(
            f"{ai_provider_label} API Key",
            value=_secret_key,
            type="password",
            placeholder="Paste your API key here…",
            label_visibility="visible",
        )
        if not ai_api_key:
            st.warning("⚠️ No API key — responses will fail. Add key above or in secrets.toml.")
    else:
        st.success("✅ No API key needed — using local server")

    st.markdown("---")
    run_btn = st.button("▶  RUN FULL SCAN", use_container_width=True, type="primary")
    st.markdown(f'<div style="margin-top:12px;font-size:0.6rem;color:#4A5568;text-align:center">'
                f'v3.0.0 — {datetime.now().strftime("%Y-%m-%d %H:%M")} UTC</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_keys = ["price_dict","adj_close","filter_result","opt_result","frontier",
         "regime_report","anomalies","risk_scores","prescriptions",
         "sentiment","geo_risk","scenarios","escalation_events",
         "compliance_mappings","bias_reports","sanctions","sar_flags",
         "nlq_history","nlq_context","spx_data","vix_data","nlq_pending"]
for k in _keys:
    if k not in st.session_state:
        st.session_state[k] = None

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()

    prog = st.progress(0, "Fetching global market data…")

    price_dict, errors = load_multiple(
        tickers + ["^GSPC", "^VIX", "GLD", "USO"],
        period_days=450,
        preferred_source=pref_src, av_key=av_key, poly_key=poly_key,
    )
    for e in errors: st.warning(e)

    spx_data = price_dict.pop("^GSPC", None)
    vix_data = price_dict.pop("^VIX",  None)
    gld_data = price_dict.pop("GLD",   None)
    uso_data = price_dict.pop("USO",   None)

    if not price_dict:
        st.error("No price data fetched. Check tickers and data source.")
        st.stop()

    prog.progress(15, "Building price matrix…")
    adj_close = build_adj_close_matrix(price_dict)

    prog.progress(25, "Running risk filters…")
    filter_result = run_all_filters(
        price_dict, adj_close, trade_size, max_adv_pct, spike_mult, max_corr)

    prog.progress(38, "Optimising portfolio…")
    green_cols = [c for c in adj_close.columns if c in filter_result.green_zone]
    opt_result = frontier = None
    if green_cols:
        gp = adj_close[green_cols]
        opt_result = optimise_portfolio(gp, max_weight=max_weight, risk_free_rate=risk_free)
        frontier   = efficient_frontier(gp, max_weight=max_weight)

    prog.progress(50, "Detecting market regime…")
    trend  = detect_trend_regime(spx_data)
    vix_r  = detect_vix_regime(vix_data)
    exits  = calculate_exit_plans(
        price_dict, filter_result.green_zone,
        atr_multiplier=1.5 if vix_r.is_defensive else 2.5)
    regime_report = RegimeReport(trend=trend, vix=vix_r, exit_plans=exits)

    prog.progress(60, "Running AI anomaly detection…")
    anomalies   = detect_anomalies(price_dict)
    risk_scores = score_predictive_risk(price_dict)

    prog.progress(68, "AI prescriptive analysis…")
    prescriptions = None
    if enable_ai:
        prescriptions = generate_prescriptions(
            filter_result, opt_result, regime_report, risk_scores, anomalies,
            provider=ai_provider, model=ai_model, api_key=ai_api_key,
        )

    prog.progress(74, "Processing news sentiment…")
    sentiment = []
    geo_risk  = compute_geopolitical_risk(vix_r.vix_level, gld_data, uso_data)
    if enable_sentiment:
        headlines = fetch_headlines(filter_result.green_zone[:8], gnews_key=gnews_key)
        sentiment = aggregate_sentiment(headlines)

    prog.progress(82, "Scenario stress tests…")
    scenarios = []
    if enable_scenarios and opt_result and opt_result.status == "optimal":
        scenarios = run_all_scenarios(opt_result.weights, price_dict, exits, adj_close)

    prog.progress(88, "GRC & compliance checks…")
    compliance_mappings = build_regulatory_mapping()
    bias_reports        = audit_model_bias()
    sanctions           = ofac_screen_tickers(tickers)
    sar_flags           = detect_sar_patterns(price_dict)

    prog.progress(94, "Building escalation queue…")
    escalation_events = build_escalation_events(
        filter_result, regime_report, anomalies, sar_flags, sanctions)

    prog.progress(99, "Finalising NLQ context…")
    nlq_context = build_portfolio_context(
        filter_result, opt_result, regime_report,
        risk_scores, anomalies, sentiment, scenarios)

    st.session_state.update({
        "price_dict": price_dict, "adj_close": adj_close,
        "filter_result": filter_result, "opt_result": opt_result,
        "frontier": frontier, "regime_report": regime_report,
        "anomalies": anomalies, "risk_scores": risk_scores,
        "prescriptions": prescriptions, "sentiment": sentiment,
        "geo_risk": geo_risk, "scenarios": scenarios,
        "compliance_mappings": compliance_mappings, "bias_reports": bias_reports,
        "sanctions": sanctions, "sar_flags": sar_flags,
        "escalation_events": escalation_events,
        "nlq_context": nlq_context, "nlq_history": [],
        "spx_data": spx_data, "vix_data": vix_data,
    })
    prog.progress(100, "Scan complete ✅")
    prog.empty()
    st.success(f"✅ Scan complete — {len(price_dict)} assets | {datetime.now().strftime('%H:%M:%S UTC')}")

# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────
fr  = st.session_state.get("filter_result")
rr  = st.session_state.get("regime_report")
opt = st.session_state.get("opt_result")

if fr is None:
    st.markdown("""
    <div style="text-align:center;padding:80px 0;color:#4A5568">
      <div style="font-size:3.5rem">🛡️</div>
      <div style="font-size:1rem;margin-top:16px;font-family:monospace;letter-spacing:0.1em">
        ADD TICKERS — ANY STOCK, ETF OR INDEX WORLDWIDE<br>
        <span style="color:#00D4FF;font-size:1.1rem">▶ RUN FULL SCAN</span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────
dl_col, _ = st.columns([1, 5])
with dl_col:
    try:
        pdf_bytes = generate_report_pdf(
            fr, opt, rr,
            st.session_state.get("price_dict", {}),
            datetime.now().strftime("%Y-%m-%d  %H:%M UTC"),
        )
        st.download_button(
            "⬇  Download Report (PDF)", pdf_bytes,
            f"AegisGuard_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            "application/pdf", use_container_width=True,
        )
    except Exception as e:
        st.warning(f"PDF error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
(tab_brief, tab_overview, tab_filters, tab_optimize, tab_regime, tab_exits,
 tab_ai, tab_alt, tab_scenarios, tab_grc, tab_nlq) = st.tabs([
    "🎯 INTEL BRIEF", "📊 OVERVIEW", "🔍 RISK FILTERS", "⚖️ OPTIMIZATION",
    "🌐 MARKET REGIME", "🚪 EXIT PLAN", "🤖 AI INSIGHTS",
    "📰 ALT DATA", "📉 SCENARIOS", "🛡️ COMPLIANCE", "💬 AI ANALYST",
])

# ── Helper colours
_ZONE_COLOUR = {"GREEN": GREEN, "NO-GO": RED, "BLACKLIST": ORANGE}
_SIG_COLOUR  = {"RISK-ON": GREEN, "CAUTION": YELLOW, "RISK-OFF": RED,
                "BULL": GREEN, "BEAR": RED, "NORMAL": GREEN,
                "ELEVATED": YELLOW, "DEFENSIVE": RED}

# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — INTEL BRIEF  (aggregated single-page command centre)
# ════════════════════════════════════════════════════════════════════════════
with tab_brief:
    pd_store   = st.session_state.get("price_dict", {})
    rs_list    = st.session_state.get("risk_scores") or []
    anom_list  = st.session_state.get("anomalies")  or []
    presc      = st.session_state.get("prescriptions")
    sent_list  = st.session_state.get("sentiment")  or []
    sc_list    = st.session_state.get("scenarios")  or []
    esc_list   = st.session_state.get("escalation_events") or []
    san_list   = st.session_state.get("sanctions")  or []
    sar_list   = st.session_state.get("sar_flags")  or []
    geo        = st.session_state.get("geo_risk")
    eps        = rr.exit_plans if rr else []
    adj_close  = st.session_state.get("adj_close")

    # ── STATUS BANNER ────────────────────────────────────────────────────────
    stance_colour = _SIG_COLOUR.get(rr.overall_signal, ACCENT)
    crit_count    = len([e for e in esc_list if e.severity == "CRITICAL"])
    blk_count     = len(fr.blacklisted)
    banner_bg     = "#3D0000" if (crit_count or blk_count) else "#0A1628"
    st.markdown(f"""
    <div style="background:{banner_bg};border:2px solid {stance_colour};border-radius:12px;
                padding:20px 28px;margin-bottom:20px;display:flex;align-items:center;
                justify-content:space-between;flex-wrap:wrap;gap:12px">
      <div>
        <div style="font-size:0.65rem;color:#8892A4;letter-spacing:0.18em;font-family:monospace">
          AEGISGUARD · INTELLIGENCE BRIEF · {datetime.now().strftime("%Y-%m-%d  %H:%M UTC")}
        </div>
        <div style="font-size:1.6rem;font-weight:900;color:{stance_colour};
                    font-family:monospace;letter-spacing:0.06em;margin-top:4px">
          {rr.overall_signal}
        </div>
        <div style="font-size:0.75rem;color:#8892A4;margin-top:2px">
          {len(pd_store)} assets scanned · {len(fr.green_zone)} tradeable ·
          {len(fr.no_go_zone)} restricted · {len(fr.blacklisted)} blacklisted
        </div>
      </div>
      <div style="display:flex;gap:16px;flex-wrap:wrap">
        <div style="text-align:center">
          <div style="font-size:1.4rem;font-weight:900;color:{RED if crit_count else GREEN}">{crit_count}</div>
          <div style="font-size:0.6rem;color:#8892A4;letter-spacing:0.1em">CRITICAL ALERTS</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:1.4rem;font-weight:900;color:{RED if blk_count else GREEN}">{blk_count}</div>
          <div style="font-size:0.6rem;color:#8892A4;letter-spacing:0.1em">BLACKLISTED</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:1.4rem;font-weight:900;color:{YELLOW}">{rr.vix.vix_level:.1f}</div>
          <div style="font-size:0.6rem;color:#8892A4;letter-spacing:0.1em">VIX</div>
        </div>
        <div style="text-align:center">
          <div style="font-size:1.4rem;font-weight:900;color:{GREEN if opt and opt.status=='optimal' else YELLOW}">
            {f"{opt.sharpe_ratio:.2f}" if opt and opt.status=="optimal" else "—"}
          </div>
          <div style="font-size:0.6rem;color:#8892A4;letter-spacing:0.1em">SHARPE</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── CRITICAL ALERTS ───────────────────────────────────────────────────────
    crit_evts = [e for e in esc_list if e.severity == "CRITICAL"]
    if crit_evts:
        st.markdown("#### 🚨 Critical Alerts")
        for ev in crit_evts[:5]:
            st.error(f"**{ev.category}** — **{ev.ticker}**: {ev.message} | Action: {ev.action} | Deadline: {ev.deadline}")

    if fr.blacklisted:
        st.error(f"🔴 **Immediate Exit Required:** {', '.join(fr.blacklisted)} — liquidate within 4 hours per SOP")

    # ── 8-KPI ROW ─────────────────────────────────────────────────────────────
    st.markdown("#### 📊 Key Metrics")
    k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
    _kd = [
        (str(len(pd_store)),                                           "SCANNED",      ACCENT),
        (str(len(fr.green_zone)),                                      "TRADEABLE",    GREEN),
        (str(len(fr.no_go_zone)),                                      "RESTRICTED",   YELLOW),
        (str(len(fr.blacklisted)),                                     "BLACKLISTED",  RED if fr.blacklisted else GREEN),
        (f"{opt.expected_return*100:.1f}%" if opt and opt.status=="optimal" else "—", "EXP RETURN", GREEN),
        (f"{opt.volatility*100:.1f}%"      if opt and opt.status=="optimal" else "—", "PORTFOLIO VOL", YELLOW),
        (f"{rr.vix.vix_level:.1f}",                                    "VIX",          RED if rr.vix.is_defensive else YELLOW if rr.vix.is_elevated else GREEN),
        (f"{rr.vix.suggested_cash_pct:.0f}%",                          "CASH REC",     YELLOW),
    ]
    for col, (v, l, c) in zip([k1,k2,k3,k4,k5,k6,k7,k8], _kd):
        with col: st.markdown(kpi_card(v, l, c), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MAIN 3-COLUMN LAYOUT ──────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([1.1, 1.1, 0.9])

    # ─ Column A: Portfolio Zones & Weights ───────────────────────────────────
    with col_a:
        st.markdown("#### 🗂️ Portfolio Zones")
        for zone_name, zone_tickers, colour in [
            ("✅ Green Zone (Tradeable)", fr.green_zone, GREEN),
            ("⛔ No-Go Zone (Restricted)", fr.no_go_zone, YELLOW),
            ("🔴 Blacklisted (Exit Now)", fr.blacklisted, RED),
        ]:
            if zone_tickers:
                st.markdown(
                    f'<div style="margin-bottom:8px">'
                    f'<span style="font-size:0.72rem;color:{colour};font-weight:700">{zone_name}</span><br>'
                    f'{"  ".join(badge(t, "green" if colour==GREEN else "red" if colour==RED else "yellow") for t in zone_tickers)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if opt and opt.status == "optimal":
            st.markdown("#### ⚖️ Optimal Weights")
            top_w = sorted(opt.weights.items(), key=lambda x: -x[1])[:8]
            for tk, wt in top_w:
                pct = wt * 100
                bar_colour = GREEN if pct >= 15 else ACCENT if pct >= 8 else "#4A5568"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                    f'<span style="font-family:monospace;font-size:0.72rem;color:#CBD5E0;width:52px">{tk}</span>'
                    f'<div style="flex:1;background:#1A2035;border-radius:3px;height:8px">'
                    f'<div style="width:{pct*4:.0f}px;max-width:100%;height:8px;background:{bar_colour};border-radius:3px"></div>'
                    f'</div>'
                    f'<span style="font-size:0.7rem;color:{bar_colour};width:36px;text-align:right">{pct:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ─ Column B: Market Regime + Risk Scores ─────────────────────────────────
    with col_b:
        st.markdown("#### 🌐 Market Regime")
        trend = rr.trend; vix_r = rr.vix
        sc = _SIG_COLOUR.get(rr.overall_signal, ACCENT)
        st.markdown(
            f'<div style="background:#0F1629;border:1px solid {sc};border-radius:8px;padding:14px;margin-bottom:12px">'
            f'<div style="font-size:0.68rem;color:#8892A4">S&P 500 Trend</div>'
            f'<div style="font-size:1rem;font-weight:700;color:{GREEN if trend.is_bull else RED}">{trend.signal}</div>'
            f'<div style="font-size:0.68rem;color:#8892A4;margin-top:4px">'
            f'SPX ${trend.spx_price:,.0f} · SMA200 ${trend.sma_200:,.0f} · Dist {trend.distance_pct:+.2f}%'
            f'</div><hr style="border-color:#1E2A3E;margin:8px 0">'
            f'<div style="font-size:0.68rem;color:#8892A4">VIX Regime</div>'
            f'<div style="font-size:1rem;font-weight:700;color:{RED if vix_r.is_defensive else YELLOW if vix_r.is_elevated else GREEN}">'
            f'{vix_r.signal} ({vix_r.vix_level:.1f})</div>'
            f'<div style="font-size:0.68rem;color:#8892A4;margin-top:4px">{vix_r.description[:90]}…</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### 🎯 Top Risk Scores")
        if rs_list:
            for rs in rs_list[:6]:
                band_c = {"DANGER": RED, "ELEVATED": ORANGE, "WATCH": YELLOW, "SAFE": GREEN}.get(rs.band, ACCENT)
                bar_w  = int(rs.score * 1.4)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
                    f'<span style="font-family:monospace;font-size:0.7rem;color:#CBD5E0;width:48px">{rs.ticker}</span>'
                    f'<div style="flex:1;background:#1A2035;border-radius:3px;height:7px">'
                    f'<div style="width:{bar_w}px;max-width:140px;height:7px;background:{band_c};border-radius:3px"></div>'
                    f'</div>'
                    f'<span style="font-size:0.68rem;color:{band_c};width:70px;text-align:right">'
                    f'{rs.score:.0f}/100 {rs.band}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No risk scores computed yet.")

    # ─ Column C: Anomalies + Sentiment snapshot ───────────────────────────────
    with col_c:
        st.markdown("#### 🔔 Active Anomalies")
        if anom_list:
            for a in anom_list[:6]:
                colour = {"CRITICAL": RED, "HIGH": ORANGE, "MEDIUM": YELLOW}.get(a.severity, GREEN)
                st.markdown(
                    f'<div style="background:#0F1629;border-left:3px solid {colour};'
                    f'padding:7px 10px;border-radius:0 6px 6px 0;margin-bottom:6px">'
                    f'<span style="font-size:0.65rem;color:{colour};font-weight:700">{a.severity}</span> '
                    f'<span style="font-size:0.63rem;color:#8892A4">[{a.anomaly_type}]</span><br>'
                    f'<span style="font-size:0.67rem;color:#CBD5E0">{a.description[:90]}…</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No anomalies detected")

        st.markdown("#### 📰 News Sentiment")
        if sent_list:
            for s in sent_list[:5]:
                sig_c = {"BULLISH": GREEN, "BEARISH": RED, "NEUTRAL": YELLOW}.get(s.signal, ACCENT)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:4px 0;border-bottom:1px solid #1E2A3E">'
                    f'<span style="font-family:monospace;font-size:0.7rem;color:#CBD5E0">{s.ticker}</span>'
                    f'<span style="font-size:0.68rem;color:{sig_c};font-weight:700">{s.signal} ({s.avg_sentiment:+.2f})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Enable News Sentiment in sidebar.")

    st.markdown("---")

    # ── AI PRESCRIPTIONS ──────────────────────────────────────────────────────
    st.markdown(
        f'#### 🧠 AI Prescriptive Recommendations '
        f'<span style="font-size:0.72rem">{badge(ai_provider_label, "blue" if not _needs_key else "purple")}</span>',
        unsafe_allow_html=True,
    )
    if presc and (presc.hedging_ideas or presc.rebalance_ops or presc.alt_assets):
        st.info(presc.summary)
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown('<div style="background:#0F1629;border:1px solid #1E2A3E;border-radius:8px;padding:14px">'
                        '<div style="font-size:0.72rem;color:#00D4FF;font-weight:700;margin-bottom:8px">🛡️ HEDGING STRATEGIES</div>',
                        unsafe_allow_html=True)
            for h in presc.hedging_ideas:
                st.markdown(f"• {h}")
            st.markdown('</div>', unsafe_allow_html=True)
        with p2:
            st.markdown('<div style="background:#0F1629;border:1px solid #1E2A3E;border-radius:8px;padding:14px">'
                        '<div style="font-size:0.72rem;color:#FFD600;font-weight:700;margin-bottom:8px">⚖️ REBALANCING ACTIONS</div>',
                        unsafe_allow_html=True)
            for r in presc.rebalance_ops:
                st.markdown(f"• {r}")
            st.markdown('</div>', unsafe_allow_html=True)
        with p3:
            st.markdown('<div style="background:#0F1629;border:1px solid #1E2A3E;border-radius:8px;padding:14px">'
                        '<div style="font-size:0.72rem;color:#00FF88;font-weight:700;margin-bottom:8px">🔄 ALTERNATIVE ASSETS</div>',
                        unsafe_allow_html=True)
            for a in presc.alt_assets:
                st.markdown(f"• {a}")
            st.markdown('</div>', unsafe_allow_html=True)
        if presc.sharpe_outlook:
            st.markdown(f"**📈 Sharpe Outlook:** {presc.sharpe_outlook}")
    elif not st.session_state.get("filter_result"):
        st.info("Run a scan to generate AI recommendations.")
    else:
        st.info("Enable 'AI Prescriptions' in the sidebar and re-run scan to see recommendations here.")

    st.markdown("---")

    # ── SCENARIO STRESS RESULTS ───────────────────────────────────────────────
    b_left, b_right = st.columns(2)

    with b_left:
        st.markdown("#### 📉 Scenario Stress Results")
        if sc_list:
            for r in sc_list:
                plc = RED if r.portfolio_return < -0.15 else ORANGE if r.portfolio_return < -0.05 else GREEN
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:6px 0;border-bottom:1px solid #1E2A3E">'
                    f'<span style="font-size:0.72rem;color:#CBD5E0">{r.scenario_label}</span>'
                    f'<span style="font-size:0.72rem;color:{plc};font-weight:700">'
                    f'{r.portfolio_return*100:+.1f}%  ${r.portfolio_value_1m:+,.0f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Run scan with Green Zone assets to see stress results.")

    with b_right:
        st.markdown("#### 🚪 Exit Plan Summary")
        if eps:
            for ep in eps[:8]:
                act_c = RED if ep.action == "EXIT NOW" else GREEN
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:5px 0;border-bottom:1px solid #1E2A3E">'
                    f'<span style="font-family:monospace;font-size:0.7rem;color:#CBD5E0">{ep.ticker}</span>'
                    f'<span style="font-size:0.68rem;color:{act_c};font-weight:700">{ep.action}</span>'
                    f'<span style="font-size:0.66rem;color:#8892A4">Stop ${ep.stop_price:,.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No exit plans — ensure Green Zone assets have OHLC data.")

    st.markdown("---")

    # ── COMPLIANCE & SANCTIONS DIGEST ─────────────────────────────────────────
    st.markdown("#### 🛡️ Compliance Digest")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**🔎 Sanctions Screening**")
        if san_list:
            for s in san_list:
                sc2 = RED if s.risk_level == "BLOCKED" else YELLOW if s.risk_level == "WATCHLIST" else GREEN
                st.markdown(f'<span style="color:{sc2};font-weight:700">{s.ticker}</span>'
                            f'<span style="color:#8892A4;font-size:0.7rem"> — {s.risk_level}: {s.reason[:50]}</span>',
                            unsafe_allow_html=True)
        else:
            st.success("All tickers clear")
    with d2:
        st.markdown("**🚩 SAR Flags**")
        if sar_list:
            for s in sar_list:
                fn = st.error if s.severity == "HIGH" else st.warning
                fn(f"**{s.pattern}**: {s.description[:70]}")
        else:
            st.success("No suspicious activity patterns")
    with d3:
        st.markdown("**📋 Escalation Queue**")
        if esc_list:
            for e in esc_list[:5]:
                ec = RED if e.severity == "CRITICAL" else YELLOW if e.severity == "WARNING" else GREEN
                st.markdown(f'<span style="color:{ec};font-size:0.7rem;font-weight:700">{e.severity}</span> '
                            f'<span style="font-size:0.7rem;color:#CBD5E0">{e.ticker} — {e.message[:55]}</span>',
                            unsafe_allow_html=True)
        else:
            st.success("No escalations")

    st.markdown("---")

    # ── ALL IMPORTANT DETAILS TABLE ───────────────────────────────────────────
    st.markdown("#### 📋 Complete Asset Intelligence")
    rs_map  = {s.ticker: s for s in rs_list}
    rows_all = []
    for ticker in sorted(pd_store.keys()):
        zone  = fr.zone_of(ticker)
        liq   = fr.liquidity.details.get(ticker, {})
        vol   = fr.volatility.details.get(ticker, {})
        rs    = rs_map.get(ticker)
        ep    = next((e for e in eps if e.ticker == ticker), None)
        sent  = next((s for s in sent_list if s.ticker == ticker), None)
        san   = next((s for s in san_list  if s.ticker == ticker), None)
        rows_all.append({
            "Ticker":      ticker,
            "Zone":        zone,
            "ADV (USD)":   f"${liq.get('adv_usd',0):,.0f}"       if liq.get("adv_usd")       else "—",
            "% ADV":       f"{liq.get('pct_of_adv',0):.3f}%"     if liq.get("pct_of_adv")    else "—",
            "Vol 20D":     f"{vol.get('vol_20d_ann',0):.1f}%"     if vol.get("vol_20d_ann")   else "—",
            "Risk Score":  f"{rs.score:.0f}/100 ({rs.band})"      if rs                        else "—",
            "Opt Weight":  f"{opt.weights.get(ticker,0)*100:.1f}%" if opt and opt.status=="optimal" else "—",
            "Exit Action": ep.action                               if ep                        else "—",
            "Stop Price":  f"${ep.stop_price:,.2f}"               if ep                        else "—",
            "Sentiment":   sent.signal                             if sent                      else "—",
            "Sanctions":   san.risk_level                          if san                       else "CLEAR",
        })

    def _style_brief(v):
        if v in ("GREEN",): return f"color:{GREEN};font-weight:700"
        if v in ("NO-GO",): return f"color:{YELLOW};font-weight:700"
        if v in ("BLACKLIST",): return f"color:{RED};font-weight:700"
        if v in ("EXIT NOW",):  return f"color:{RED};font-weight:900"
        if v in ("HOLD",):      return f"color:{GREEN};font-weight:700"
        if v in ("DANGER",):    return f"color:{RED};font-weight:700"
        if v in ("ELEVATED",):  return f"color:{ORANGE}"
        if v in ("WATCH",):     return f"color:{YELLOW}"
        if v in ("SAFE",):      return f"color:{GREEN}"
        if v in ("BLOCKED",):   return f"color:{RED};font-weight:900"
        if v in ("WATCHLIST",): return f"color:{YELLOW}"
        if v in ("BULLISH",):   return f"color:{GREEN};font-weight:700"
        if v in ("BEARISH",):   return f"color:{RED};font-weight:700"
        return ""

    if rows_all:
        df_all = pd.DataFrame(rows_all)
        st.dataframe(
            df_all.style.map(_style_brief, subset=["Zone","Exit Action","Sentiment","Sanctions"]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # ── AI BRIEF GENERATOR ────────────────────────────────────────────────────
    st.markdown(
        f'#### 💡 AI Intelligence Brief '
        f'<span style="font-size:0.72rem">{badge(ai_provider_label, "blue" if not _needs_key else "purple")}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<small style="color:#8892A4">Click to generate a full narrative intelligence brief from all scan data using the selected AI provider.</small>',
                unsafe_allow_html=True)

    if st.button("⚡ Generate Full Intelligence Brief", type="primary", key="gen_brief"):
        nlq_ctx = st.session_state.get("nlq_context", "No data loaded.")
        brief_q = (
            "You are AegisGuard AI. Generate a comprehensive institutional intelligence brief covering: "
            "1) Executive Summary (3-4 sentences), "
            "2) Market Regime Analysis, "
            "3) Portfolio Risk Assessment with specific tickers, "
            "4) Key Recommendations (numbered list), "
            "5) Critical Risks to Monitor, "
            "6) Compliance & Regulatory Flags, "
            "7) Outlook for next 5-10 trading days. "
            "Use professional financial language. Be specific with numbers and tickers from the context."
        )
        with st.spinner(f"Generating brief via {ai_provider_label}…"):
            brief_resp = ask_portfolio(
                brief_q, nlq_ctx, [],
                provider=ai_provider, model=ai_model, api_key=ai_api_key,
            )
        st.markdown(
            f'<div style="background:#0A1628;border:1px solid #1E2A3E;border-radius:10px;'
            f'padding:24px;line-height:1.7;color:#CBD5E0;font-size:0.85rem">'
            f'{brief_resp.answer.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Generated via {ai_provider_label} · {brief_resp.timestamp}")

    st.markdown("---")

    # ── EMAIL DISPATCH PANEL ───────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#0F1629;border:1px solid #00D4FF;border-radius:10px;padding:20px 24px;margin-bottom:8px">
      <div style="font-size:1rem;font-weight:700;color:#00D4FF;font-family:monospace">📧 SEND INTELLIGENCE REPORT</div>
      <div style="font-size:0.7rem;color:#8892A4;margin-top:4px">
        Email this brief, AI narrative, and asset table directly to your team.
      </div>
    </div>""", unsafe_allow_html=True)

    # Load SMTP defaults from secrets
    try:
        _em = st.secrets["email"]
    except:
        _em = {}
    _smtp_host_def  = _em.get("smtp_host", "smtp.gmail.com")
    _smtp_port_def  = int(_em.get("smtp_port", 587))
    _smtp_user_def  = _em.get("smtp_user", "")
    _smtp_pass_def  = _em.get("smtp_pass", "")
    _from_def       = _em.get("from_addr", "")
    _to_def         = _em.get("default_to", "")
    _cc_def         = _em.get("default_cc", "")

    with st.expander("⚙️ Configure Email Settings", expanded=not bool(_smtp_user_def)):
        em_c1, em_c2 = st.columns(2)
        with em_c1:
            smtp_provider_name = st.selectbox(
                "Email Provider",
                list(SMTP_PRESETS.keys()),
                index=0,
                key="email_provider",
            )
            preset = SMTP_PRESETS[smtp_provider_name]
            smtp_host = st.text_input("SMTP Host",  value=preset["host"] or _smtp_host_def, key="smtp_host")
            smtp_port = st.number_input("SMTP Port", value=preset["port"] or _smtp_port_def,
                                        min_value=1, max_value=65535, key="smtp_port")
        with em_c2:
            smtp_user = st.text_input("SMTP Username (your email)", value=_smtp_user_def, key="smtp_user")
            smtp_pass = st.text_input("SMTP Password / App Password", value=_smtp_pass_def,
                                      type="password", key="smtp_pass")
            from_addr = st.text_input("From Address", value=_from_def or _smtp_user_def, key="from_addr")
        if smtp_provider_name == "Gmail":
            st.info("🔐 **Gmail users:** use an App Password, not your normal password. "
                    "Go to myaccount.google.com → Security → 2-Step → App Passwords")
        elif smtp_provider_name == "Outlook / Hotmail":
            st.info("🔐 **Outlook users:** enable SMTP AUTH in account settings first.")

    # Recipients
    re_c1, re_c2 = st.columns(2)
    with re_c1:
        to_field = st.text_input(
            "📨 To (comma-separated)",
            value=_to_def,
            placeholder="cio@yourfirm.com, riskteam@yourfirm.com",
            key="email_to",
        )
    with re_c2:
        cc_field = st.text_input(
            "📄 CC (comma-separated, optional)",
            value=_cc_def,
            placeholder="compliance@yourfirm.com",
            key="email_cc",
        )

    subj_default = (
        f"AegisGuard Intelligence Brief — {rr.overall_signal} — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    email_subject = st.text_input("📤 Subject", value=subj_default, key="email_subj")

    # Content toggles
    st.markdown("**Include in email:**")
    inc_c1, inc_c2, inc_c3, inc_c4 = st.columns(4)
    with inc_c1: inc_alerts    = st.checkbox("🚨 Critical Alerts",      value=True,  key="inc_alerts")
    with inc_c2: inc_brief     = st.checkbox("💡 AI Narrative Brief",  value=True,  key="inc_brief")
    with inc_c3: inc_portfolio = st.checkbox("📋 Asset Table",          value=True,  key="inc_portfolio")
    with inc_c4: inc_pdf       = st.checkbox("📅 PDF Report (attach)", value=False, key="inc_pdf")

    if st.button("📧  Send Report Email", type="primary", key="send_email_btn", use_container_width=True):
        # Validate basics
        _to_list = [x.strip() for x in to_field.split(",") if x.strip()]
        _cc_list = [x.strip() for x in cc_field.split(",") if x.strip()]

        if not _to_list:
            st.error("⚠️ Please enter at least one recipient in the To field.")
        elif not smtp_user or not smtp_pass:
            st.error("⚠️ SMTP Username and Password are required. Configure them in Email Settings above.")
        else:
            # Gather content
            _brief_text  = st.session_state.get("last_brief_text", "")
            _alert_lines = [
                f"{ev.category} — {ev.ticker}: {ev.message} | {ev.action} | Deadline: {ev.deadline}"
                for ev in esc_list if ev.severity == "CRITICAL"
            ]
            if fr.blacklisted:
                _alert_lines.insert(0,
                    f"IMMEDIATE EXIT: {', '.join(fr.blacklisted)} — liquidate within 4 hours per SOP")

            _pdf_bytes = None
            if inc_pdf:
                try:
                    _pdf_bytes = generate_report_pdf(
                        fr, opt, rr,
                        st.session_state.get("price_dict", {}),
                        datetime.now().strftime("%Y-%m-%d  %H:%M UTC"),
                    )
                except Exception as _pe:
                    st.warning(f"PDF generation failed (will send without it): {_pe}")

            _html = build_html_body(
                brief_text     = _brief_text if inc_brief else "",
                alert_lines    = _alert_lines if inc_alerts else [],
                portfolio_rows = rows_all if inc_portfolio else [],
                stance         = rr.overall_signal,
                scan_time      = datetime.now().strftime("%Y-%m-%d  %H:%M UTC"),
                include_alerts    = inc_alerts,
                include_brief     = inc_brief,
                include_portfolio = inc_portfolio,
            )

            with st.spinner("Sending email…"):
                result = send_report(
                    smtp_host    = smtp_host,
                    smtp_port    = int(smtp_port),
                    smtp_user    = smtp_user,
                    smtp_pass    = smtp_pass,
                    use_tls      = True,
                    from_addr    = from_addr or smtp_user,
                    to_addrs     = _to_list,
                    cc_addrs     = _cc_list,
                    subject      = email_subject,
                    html_body    = _html,
                    pdf_bytes    = _pdf_bytes,
                    pdf_filename = f"AegisGuard_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                )

            if result.success:
                st.success(f"✅ {result.message}  ({result.timestamp})")
            else:
                st.error(f"❌ Send failed: {result.message}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:

    pd_store = st.session_state.get("price_dict", {})
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi_data = [
        (str(len(pd_store)),           "SCANNED",     ACCENT),
        (str(len(fr.green_zone)),      "GREEN ZONE",  GREEN),
        (str(len(fr.no_go_zone)),      "NO-GO ZONE",  RED),
        (str(len(fr.blacklisted)),     "BLACKLISTED", ORANGE),
        (rr.overall_signal,            "STANCE",      _SIG_COLOUR.get(rr.overall_signal, ACCENT)),
        (f"{opt.sharpe_ratio:.2f}" if opt and opt.status == "optimal" else "—", "SHARPE", GREEN),
    ]
    for col, (v,l,c) in zip([c1,c2,c3,c4,c5,c6], kpi_data):
        with col: st.markdown(kpi_card(v,l,c), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Escalation alerts
    for ev in [e for e in (st.session_state.get("escalation_events") or []) if e.severity=="CRITICAL"][:3]:
        st.error(f"🚨 **{ev.category}** — {ev.ticker}: {ev.message} | {ev.action} | Deadline: {ev.deadline}")

    # Summary table
    rs_map = {s.ticker: s for s in (st.session_state.get("risk_scores") or [])}
    rows = []
    for ticker in sorted(pd_store.keys()):
        zone  = fr.zone_of(ticker)
        liq   = fr.liquidity.details.get(ticker, {})
        vol   = fr.volatility.details.get(ticker, {})
        rs    = rs_map.get(ticker)
        rows.append({
            "Ticker":     ticker,
            "Zone":       zone,
            "ADV (USD)":  f"${liq.get('adv_usd',0):,.0f}" if liq.get("adv_usd") else "—",
            "% ADV":      f"{liq.get('pct_of_adv',0):.3f}%" if liq.get("pct_of_adv") else "—",
            "Vol 20D":    f"{vol.get('vol_20d_ann',0):.1f}%" if vol.get("vol_20d_ann") else "—",
            "Risk Score": f"{rs.score:.0f}/100 ({rs.band})" if rs else "—",
        })

    def _sz(v): return {
        "GREEN":"color:#00FF88;font-weight:700",
        "NO-GO":"color:#FF3860;font-weight:700",
        "BLACKLIST":"color:#FF6B35;font-weight:700",
    }.get(v,"")
    st.dataframe(pd.DataFrame(rows).style.map(_sz, subset=["Zone"]),
                 use_container_width=True, hide_index=True)

    adj_close = st.session_state.get("adj_close")
    if opt and opt.weights and adj_close is not None and not adj_close.empty:
        valid = [c for c in adj_close.columns if c in opt.weights]
        if valid:
            st.plotly_chart(chart_cumulative_returns(adj_close[valid], opt.weights),
                            use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK FILTERS
# ════════════════════════════════════════════════════════════════════════════
with tab_filters:
    c1,c2,c3 = st.columns(3)
    for col, title, res in [
        (c1,"💧 Liquidity Gate",  fr.liquidity),
        (c2,"📈 Volatility Cap",  fr.volatility),
        (c3,"🔗 Correlation Gate",fr.correlation),
    ]:
        clr = GREEN if res.pass_rate > 0.7 else RED
        with col:
            st.markdown(f"#### {title}")
            st.markdown(kpi_card(f"{res.pass_rate*100:.0f}%",
                f"PASS RATE ({len(res.passed)}/{len(res.passed)+len(res.rejected)})", clr),
                unsafe_allow_html=True)
            st.markdown(" ".join(badge(t,"green") for t in res.passed) or "—", unsafe_allow_html=True)
            st.markdown(" ".join(badge(t,"red")   for t in res.rejected) or "—", unsafe_allow_html=True)

    st.markdown("---")
    corr_data = fr.correlation.details.get("correlation_matrix", {})
    if corr_data:
        st.plotly_chart(chart_correlation_heatmap(corr_data), use_container_width=True)

    flagged = fr.correlation.details.get("flagged_pairs", [])
    if flagged:
        st.markdown("#### ⚠️ Over-Correlated Pairs")
        st.dataframe(pd.DataFrame(flagged), use_container_width=True, hide_index=True)

    st.markdown("---")
    pd_store = st.session_state.get("price_dict", {})
    sel = st.selectbox("Deep-dive ticker", sorted(pd_store.keys()))
    if sel and sel in pd_store:
        v1,v2 = st.columns(2)
        with v1: st.plotly_chart(chart_volatility(pd_store[sel]["Adj Close"], sel, height=300), use_container_width=True)
        with v2: st.plotly_chart(chart_price_sma(pd_store[sel]["Adj Close"],  sel, sma_period=50, height=300), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════════
with tab_optimize:
    if not opt or opt.status != "optimal":
        st.warning("No Green Zone assets for optimisation, or insufficient data.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        for col,(v,l,c) in zip([c1,c2,c3,c4],[
            (f"{opt.expected_return*100:.2f}%","EXP. RETURN",GREEN),
            (f"{opt.volatility*100:.2f}%",     "VOLATILITY", YELLOW),
            (f"{opt.sharpe_ratio:.3f}",         "SHARPE",     GREEN if opt.sharpe_ratio>1 else YELLOW),
            (opt.solver_used.upper(),           "SOLVER",     ACCENT),
        ]):
            with col: st.markdown(kpi_card(v,l,c), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ch, tb = st.columns([1.2,1])
        with ch: st.plotly_chart(chart_weights_donut(opt.weights), use_container_width=True)
        with tb:
            st.markdown("#### Optimal Weights")
            st.dataframe(opt.as_dataframe(), use_container_width=True, hide_index=True)

        frontier_df = st.session_state.get("frontier")
        if frontier_df is not None and not frontier_df.empty:
            st.plotly_chart(chart_efficient_frontier(frontier_df, opt.expected_return, opt.volatility),
                            use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — MARKET REGIME
# ════════════════════════════════════════════════════════════════════════════
with tab_regime:
    trend = rr.trend; vix_r = rr.vix
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"#### S&P 500 Trend {regime_badge(trend.signal)}", unsafe_allow_html=True)
        for v,l,c in [
            (f"${trend.spx_price:,.0f}", "S&P 500 PRICE", ACCENT),
            (f"${trend.sma_200:,.0f}",   "200-DAY SMA",   GREEN if trend.is_bull else RED),
            (f"{trend.distance_pct:+.2f}%","SMA DISTANCE",GREEN if trend.distance_pct>0 else RED),
        ]:
            st.markdown(kpi_card(v,l,c), unsafe_allow_html=True)
        st.info(trend.description)
    with c2:
        st.markdown(f"#### VIX  {regime_badge(vix_r.signal)}", unsafe_allow_html=True)
        vc = RED if vix_r.is_defensive else YELLOW if vix_r.is_elevated else GREEN
        for v,l,c in [
            (f"{vix_r.vix_level:.1f}",         "VIX LEVEL",   vc),
            (f"{vix_r.suggested_cash_pct:.0f}%","CASH BUFFER", YELLOW),
        ]:
            st.markdown(kpi_card(v,l,c), unsafe_allow_html=True)
        (st.error if vix_r.is_defensive else st.warning if vix_r.is_elevated else st.success)(vix_r.description)

    sc = _SIG_COLOUR.get(rr.overall_signal, ACCENT)
    st.markdown(f"""
    <div style="background:#0F1629;border:1px solid {sc};border-radius:8px;
                padding:18px;text-align:center;margin:16px 0">
      <div style="font-size:1.3rem;font-weight:900;color:{sc};font-family:monospace">
        {rr.overall_signal}</div>
      <div style="font-size:0.72rem;color:#8892A4;margin-top:4px">
        COMPOSITE STANCE  |  Cash Rec: {rr.cash_recommendation_pct:.0f}%
      </div>
    </div>""", unsafe_allow_html=True)

    spx_data = st.session_state.get("spx_data")
    vix_data = st.session_state.get("vix_data")
    if spx_data is not None:
        st.plotly_chart(chart_price_sma(spx_data["Adj Close"],"S&P 500",sma_period=200),
                        use_container_width=True)
    if vix_data is not None:
        st.plotly_chart(chart_vix(vix_data), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXIT PLAN
# ════════════════════════════════════════════════════════════════════════════
with tab_exits:
    eps = rr.exit_plans
    if not eps:
        st.warning("No exit plans — ensure Green Zone assets have full OHLC data.")
    else:
        atr_mult = 1.5 if rr.vix.is_defensive else 2.5
        st.markdown(
            f'ATR Multiplier: {badge(f"{atr_mult}× Defensive" if rr.vix.is_defensive else f"{atr_mult}× Standard","orange" if rr.vix.is_defensive else "blue")}',
            unsafe_allow_html=True)
        edf = rr.exit_plan_df()
        if not edf.empty:
            def _sa(v): return f"color:{RED};font-weight:900" if v=="EXIT NOW" else f"color:{GREEN};font-weight:700"
            st.dataframe(edf.style.map(_sa, subset=["Action"]), use_container_width=True, hide_index=True)
        for ep in [e for e in eps if e.action=="EXIT NOW"]:
            st.error(f"🚨 **CRITICAL EXIT** — **{ep.ticker}** Current: ${ep.current_price:,.2f} | Stop: ${ep.stop_price:,.2f} | Liquidate within 4 hours")

        st.markdown("---")
        st.markdown("""
        #### 📋 SOP Quick Reference
        | Condition | Action | Deadline |
        |---|---|---|
        | Asset → NO-GO ZONE | Halt new orders; review exits | Before next trade |
        | Asset → BLACKLIST | Full liquidation | Within 4 hours |
        | VIX > 30 | Raise cash; tighten stops to 1.5× ATR | Same day |
        | BEAR TREND | Reduce equity exposure ≥25% | 2 trading days |
        | ATR Stop Breached | Full exit | Within 4 hours |
        """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab_ai:
    rs_list = st.session_state.get("risk_scores") or []
    if rs_list:
        st.markdown("#### 🎯 Predictive Risk Scores (0 = Safe → 100 = Critical)")
        rs_df = pd.DataFrame([{
            "Ticker":     s.ticker,
            "Score":      s.score,
            "Band":       s.band,
            "Momentum Z": s.momentum_z,
            "Vol Trend":  f"{s.vol_trend_z*100:+.1f}%",
            "Drawdown":   f"{s.drawdown_pct:.1f}%",
            "SMA Dist":   f"{s.trend_score:+.1f}%",
        } for s in rs_list])
        def _sb(v): return {
            "DANGER":"color:#FF3860;font-weight:700","ELEVATED":"color:#FF6B35;font-weight:700",
            "WATCH":"color:#FFD600","SAFE":"color:#00FF88",
        }.get(v,"")
        st.dataframe(rs_df.style.map(_sb, subset=["Band"]), use_container_width=True, hide_index=True)

    anom = st.session_state.get("anomalies") or []
    if anom:
        st.markdown("---")
        st.markdown("#### 🔔 Anomalies Detected")
        for a in anom[:10]:
            fn = {"CRITICAL":st.error,"HIGH":st.warning,"MEDIUM":st.info}.get(a.severity, st.success)
            fn(f"**{a.severity}** [{a.anomaly_type}] {a.description}")

    presc = st.session_state.get("prescriptions")
    if presc:
        st.markdown("---")
        st.markdown(
            f'#### 🧠 AI Prescriptive Recommendations '
            f'<span style="font-size:0.7rem">{badge(ai_provider_label, "blue" if not _needs_key else "purple")}'
            f'<span style="font-size:0.65rem;color:#4A5568;margin-left:6px">model: <code>{ai_model or _default_mdl}</code></span></span>',
            unsafe_allow_html=True,
        )
        st.info(presc.summary)
        p1,p2,p3 = st.columns(3)
        with p1:
            st.markdown("**🛡️ Hedging Strategies**")
            for h in presc.hedging_ideas: st.markdown(f"• {h}")
        with p2:
            st.markdown("**⚖️ Rebalancing Actions**")
            for r in presc.rebalance_ops: st.markdown(f"• {r}")
        with p3:
            st.markdown("**🔄 Alternative Assets**")
            for a in presc.alt_assets: st.markdown(f"• {a}")
        if presc.sharpe_outlook:
            st.markdown(f"**📈 Sharpe Outlook:** {presc.sharpe_outlook}")
    elif not enable_ai:
        st.info("Enable 'AI Prescriptions' in the sidebar to activate this feature.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — ALT DATA
# ════════════════════════════════════════════════════════════════════════════
with tab_alt:
    geo = st.session_state.get("geo_risk")
    if geo:
        st.markdown("#### 🌍 Geopolitical Risk Index")
        gc = RED if geo.band=="EXTREME" else ORANGE if geo.band=="HIGH" else YELLOW if geo.band=="MODERATE" else GREEN
        g1,g2,g3,g4 = st.columns(4)
        with g1: st.markdown(kpi_card(f"{geo.score:.0f}/100","GPR SCORE",gc),unsafe_allow_html=True)
        with g2: st.markdown(kpi_card(geo.band,"BAND",gc),unsafe_allow_html=True)
        with g3: st.markdown(kpi_card(f"{geo.vix_contrib:.0f}","VIX CONTRIB",ACCENT),unsafe_allow_html=True)
        with g4: st.markdown(kpi_card(f"{geo.gold_contrib:.0f}","GOLD CONTRIB",YELLOW),unsafe_allow_html=True)
        (st.error if geo.band in ("EXTREME","HIGH") else st.warning if geo.band=="MODERATE" else st.success)(geo.description)

    st.markdown("---")
    sent = st.session_state.get("sentiment") or []
    if sent:
        st.markdown("#### 📰 News Sentiment — NLP Scored via Claude")
        sent_df = pd.DataFrame([{
            "Ticker":       s.ticker,
            "Sentiment":    s.avg_sentiment,
            "Signal":       s.signal,
            "Articles":     s.n_articles,
            "Top Headline": s.top_headline[:70]+"…" if len(s.top_headline)>70 else s.top_headline,
        } for s in sent])
        def _ss(v): return {
            "BULLISH":"color:#00FF88;font-weight:700",
            "BEARISH":"color:#FF3860;font-weight:700",
            "NEUTRAL":"color:#FFD600",
        }.get(v,"")
        st.dataframe(sent_df.style.map(_ss, subset=["Signal"]), use_container_width=True, hide_index=True)
    else:
        st.info("Enable 'News Sentiment NLP' in the sidebar to activate this panel.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — SCENARIOS
# ════════════════════════════════════════════════════════════════════════════
with tab_scenarios:
    pd_store  = st.session_state.get("price_dict", {})
    adj_close = st.session_state.get("adj_close")
    eps       = rr.exit_plans

    if not opt or opt.status != "optimal":
        st.warning("Run a scan with Green Zone assets to enable scenario modeling.")
    else:
        sc_stored = st.session_state.get("scenarios") or []
        if sc_stored:
            st.markdown("#### 📉 Historical Stress Test Results")
            sc_df = pd.DataFrame([{
                "Scenario":       r.scenario_label,
                "Market Shock":   f"{r.equity_shock*100:.0f}%",
                "Portfolio P&L":  f"{r.portfolio_return*100:.1f}%",
                "$ Impact ($1M)": f"${r.portfolio_value_1m:+,.0f}",
                "Survival %":     f"{r.survival_rate*100:.0f}%",
                "Max DD Est.":    f"{r.max_drawdown_est*100:.1f}%",
            } for r in sc_stored])
            def _scp(v):
                try:
                    n = float(v.replace(",","").replace("$","").replace("%",""))
                    if n < -15: return f"color:{RED};font-weight:700"
                    if n < -5:  return f"color:{ORANGE}"
                    return f"color:{GREEN}"
                except: return ""
            st.dataframe(sc_df.style.map(_scp, subset=["Portfolio P&L","$ Impact ($1M)"]),
                         use_container_width=True, hide_index=True)
            for r in sc_stored:
                if abs(r.portfolio_return) > 0.15:
                    st.warning(f"⚠️ **{r.scenario_label}**: {r.recommendation}")

        st.markdown("---")
        st.markdown("#### 🔧 Custom What-If Scenario Builder")
        col1,col2,col3 = st.columns(3)
        with col1: custom_shock = st.slider("Equity Shock (%)", -80, 50, -20, 1) / 100
        with col2: custom_vol   = st.slider("Vol Multiplier",    0.5, 10.0, 2.0, 0.5)
        with col3: custom_rate  = st.slider("Rate Δ (bps)",      -300, 500, 0, 25) / 10000

        if st.button("▶ Run Custom Scenario", type="primary"):
            r = run_scenario("CUSTOM", opt.weights, pd_store, eps, adj_close,
                             custom_shock, custom_vol, custom_rate)
            c1,c2,c3 = st.columns(3)
            with c1: st.markdown(kpi_card(f"{r.portfolio_return*100:.2f}%","PORTFOLIO P&L",RED if r.portfolio_return<-0.1 else YELLOW),unsafe_allow_html=True)
            with c2: st.markdown(kpi_card(f"${r.portfolio_value_1m:+,.0f}","$ IMPACT $1M",RED if r.portfolio_value_1m<-100_000 else GREEN),unsafe_allow_html=True)
            with c3: st.markdown(kpi_card(f"{r.survival_rate*100:.0f}%","SURVIVAL RATE",GREEN if r.survival_rate>0.7 else RED),unsafe_allow_html=True)
            st.info(r.recommendation)
            if r.assets:
                adf = pd.DataFrame([{
                    "Ticker":  a.ticker,
                    "Current": f"${a.current_price:,.2f}",
                    "Shocked": f"${a.shocked_price:,.2f}",
                    "Change %":f"{a.pct_change:+.1f}%",
                    "Beta":    a.beta_used,
                    "Stop Breach":"⚠️ YES" if a.breaches_stop else "✓ NO",
                } for a in r.assets])
                st.dataframe(adf, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 9 — COMPLIANCE / GRC
# ════════════════════════════════════════════════════════════════════════════
with tab_grc:
    mappings = st.session_state.get("compliance_mappings") or []
    if mappings:
        stats = regulatory_coverage_stats(mappings)
        st.markdown("#### 🗂️ Regulatory Control Mapping")
        g1,g2,g3,g4 = st.columns(4)
        with g1: st.markdown(kpi_card(f"{stats['coverage_pct']}%","COVERAGE",GREEN),unsafe_allow_html=True)
        with g2: st.markdown(kpi_card(str(stats['nist_controls']),"NIST",ACCENT),unsafe_allow_html=True)
        with g3: st.markdown(kpi_card(str(stats['soc2_controls']),"SOC 2",ACCENT),unsafe_allow_html=True)
        with g4: st.markdown(kpi_card(str(stats['mifid_articles']),"MiFID II",ACCENT),unsafe_allow_html=True)

        map_df = pd.DataFrame([{
            "Control":     m.control_id,
            "Description": m.description,
            "NIST":        ", ".join(m.nist_refs),
            "SOC 2":       ", ".join(m.soc2_refs),
            "MiFID II":    ", ".join(m.mifid_refs),
            "Status":      m.status,
        } for m in mappings])
        st.dataframe(map_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🤖 Model Bias Audit (CFPB / FTC Readiness)")
    for br in (st.session_state.get("bias_reports") or []):
        cv = GREEN if br.verdict == "PASS" else YELLOW
        st.markdown(f'**{br.model_name}**  {badge(br.verdict, "green" if br.verdict=="PASS" else "yellow")}',
                    unsafe_allow_html=True)
        b1,b2,b3 = st.columns(3)
        with b1: st.markdown(kpi_card(f"{br.disparate_impact:.2f}","DISPARATE IMPACT",cv),unsafe_allow_html=True)
        with b2: st.markdown(kpi_card(f"{br.feature_stability:.2f}","STABILITY",ACCENT),unsafe_allow_html=True)
        with b3: st.markdown(kpi_card(f"{br.decision_coverage*100:.0f}%","EXPLAINABILITY",GREEN),unsafe_allow_html=True)
        st.caption(br.narrative)
        st.markdown("---")

    st.markdown("#### 🔎 OFAC Sanctions Screening")
    san = st.session_state.get("sanctions") or []
    if san:
        san_df = pd.DataFrame([{
            "Ticker":     s.ticker,
            "Risk Level": s.risk_level,
            "Reason":     s.reason,
            "Action":     s.action,
        } for s in san])
        def _ssl(v): return {
            "BLOCKED":   f"color:{RED};font-weight:900",
            "WATCHLIST": f"color:{YELLOW}",
            "CLEAR":     f"color:{GREEN}",
        }.get(v,"")
        st.dataframe(san_df.style.map(_ssl, subset=["Risk Level"]), use_container_width=True, hide_index=True)

    st.markdown("#### 🚩 SAR Pattern Detection")
    sars = st.session_state.get("sar_flags") or []
    if sars:
        for s in sars:
            (st.error if s.severity=="HIGH" else st.warning)(f"**{s.pattern}**: {s.description}")
    else:
        st.success("✅ No suspicious activity patterns detected.")

    st.markdown("#### 📋 Escalation Queue")
    esc = st.session_state.get("escalation_events") or []
    if esc:
        esc_df = pd.DataFrame([{
            "Severity": e.severity,
            "Category": e.category,
            "Ticker":   e.ticker,
            "Message":  e.message,
            "Action":   e.action,
            "Deadline": e.deadline,
            "Notify":   ", ".join(e.notify),
        } for e in esc])
        def _esev(v): return {
            "CRITICAL": f"color:{RED};font-weight:900",
            "WARNING":  f"color:{YELLOW};font-weight:700",
            "INFO":     f"color:{GREEN}",
        }.get(v,"")
        st.dataframe(esc_df.style.map(_esev, subset=["Severity"]), use_container_width=True, hide_index=True)

        with st.expander("📧 Auto-Generated Alert Emails"):
            for e in [ev for ev in esc if ev.email_draft]:
                st.code(e.email_draft, language=None)

# ════════════════════════════════════════════════════════════════════════════
# TAB 10 — AI ANALYST (NLQ)
# ════════════════════════════════════════════════════════════════════════════
with tab_nlq:
    st.markdown("#### 💬 AI Portfolio Analyst — Natural Language Interface")
    st.markdown('<small style="color:#8892A4">Ask anything about your portfolio. '
                'The AI reads all scan results and answers like a senior risk manager.</small>',
                unsafe_allow_html=True)

    for turn in (st.session_state.get("nlq_history") or []):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    st.markdown("**⚡ Quick questions:**")
    qcols = st.columns(3)
    quick_qs = [
        "What are the main risk drivers in my portfolio?",
        "Which assets should I reduce exposure to and why?",
        "How resilient is my portfolio to a 30% market crash?",
        "Explain the worst anomaly detected today.",
        "What hedges would protect against rising VIX?",
        "Summarise compliance risks for the CIO report.",
    ]
    for i, q in enumerate(quick_qs):
        with qcols[i % 3]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                st.session_state["nlq_pending"] = q

    user_q = st.chat_input("Ask the AI analyst…")
    if not user_q and st.session_state.get("nlq_pending"):
        user_q = st.session_state.pop("nlq_pending")

    # Show active provider badge
    st.markdown(
        f'<div style="margin-bottom:8px">'
        f'<span style="font-size:0.7rem;color:#8892A4">Active provider: </span>'
        f'{badge(ai_provider_label, "blue" if not _needs_key else "purple")}'
        f'<span style="font-size:0.68rem;color:#4A5568;margin-left:8px">model: <code>{ai_model or _default_mdl}</code></span>'
        f'</div>',
        unsafe_allow_html=True)

    if user_q:
        context = st.session_state.get("nlq_context","No portfolio data loaded.")
        history = st.session_state.get("nlq_history") or []
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner(f"Querying {ai_provider_label}…"):
                resp = ask_portfolio(
                    user_q, context, history,
                    provider=ai_provider,
                    model=ai_model,
                    api_key=ai_api_key,
                )
            st.markdown(resp.answer)
            if resp.provider:
                st.caption(f"via {ai_provider_label} · {resp.timestamp}")
        history += [{"role":"user","content":user_q},
                    {"role":"assistant","content":resp.answer}]
        st.session_state["nlq_history"] = history
        st.rerun()
