"""
Generate the AegisGuard Standard Operating Procedure PDF.
Run: python docs/generate_sop.py
Output: docs/AegisGuard_SOP.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units     import mm
from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
from reportlab.lib            import colors
from reportlab.platypus       import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums      import TA_CENTER, TA_LEFT, TA_RIGHT

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

C_DARK   = colors.HexColor("#0A0E1A")
C_PANEL  = colors.HexColor("#0F1629")
C_ACCENT = colors.HexColor("#00D4FF")
C_GREEN  = colors.HexColor("#00C06A")
C_YELLOW = colors.HexColor("#FFD600")
C_RED    = colors.HexColor("#FF3860")
C_TEXT   = colors.HexColor("#E8EAF0")
C_MUTED  = colors.HexColor("#8892A4")
C_BORDER = colors.HexColor("#1E2A3E")
C_WHITE  = colors.white

# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────

base   = getSampleStyleSheet()

def make_style(name, parent="Normal", **kwargs):
    return ParagraphStyle(name, parent=base[parent], **kwargs)

S_COVER_TITLE = make_style("CoverTitle", fontSize=28, textColor=C_ACCENT,
                            fontName="Courier-Bold", alignment=TA_CENTER,
                            spaceAfter=6)
S_COVER_SUB   = make_style("CoverSub",   fontSize=11, textColor=C_MUTED,
                            fontName="Courier",       alignment=TA_CENTER,
                            spaceAfter=4)
S_COVER_DATE  = make_style("CoverDate",  fontSize=9,  textColor=C_MUTED,
                            fontName="Courier",       alignment=TA_CENTER)

S_SECTION     = make_style("Section",    fontSize=13, textColor=C_ACCENT,
                            fontName="Courier-Bold",  spaceBefore=18, spaceAfter=6)
S_SUBSEC      = make_style("SubSec",     fontSize=10, textColor=C_TEXT,
                            fontName="Courier-Bold",  spaceBefore=10, spaceAfter=4)
S_BODY        = make_style("Body",       fontSize=9,  textColor=C_TEXT,
                            fontName="Courier",       leading=14, spaceAfter=4)
S_BULLET      = make_style("Bullet",     fontSize=9,  textColor=C_TEXT,
                            fontName="Courier",       leading=13,
                            leftIndent=14, spaceAfter=3,
                            bulletIndent=4)
S_WARNING     = make_style("Warning",    fontSize=9,  textColor=C_RED,
                            fontName="Courier-Bold",  spaceBefore=6, spaceAfter=6)
S_NOTE        = make_style("Note",       fontSize=8,  textColor=C_YELLOW,
                            fontName="Courier",       leading=12, spaceAfter=4)
S_FOOTER      = make_style("Footer",     fontSize=7,  textColor=C_MUTED,
                            fontName="Courier",       alignment=TA_CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rule(color=C_BORDER, thickness=0.5, space_before=4, space_after=8):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=space_after, spaceBefore=space_before)


def section_header(text: str):
    return [Spacer(1, 4*mm), Paragraph(text, S_SECTION), rule(C_ACCENT, 0.8)]


def table_dark(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),   C_PANEL),
        ("TEXTCOLOR",    (0, 0), (-1, 0),   C_ACCENT),
        ("FONTNAME",     (0, 0), (-1, 0),   "Courier-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),   8),
        ("BOTTOMPADDING",(0, 0), (-1, 0),   6),
        ("TOPPADDING",   (0, 0), (-1, 0),   6),
        ("BACKGROUND",   (0, 1), (-1, -1),  C_DARK),
        ("TEXTCOLOR",    (0, 1), (-1, -1),  C_TEXT),
        ("FONTNAME",     (0, 1), (-1, -1),  "Courier"),
        ("FONTSIZE",     (0, 1), (-1, -1),  8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_DARK, C_PANEL]),
        ("GRID",         (0, 0), (-1, -1),  0.4, C_BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1),  6),
        ("RIGHTPADDING", (0, 0), (-1, -1),  6),
        ("TOPPADDING",   (0, 1), (-1, -1),  5),
        ("BOTTOMPADDING",(0, 1), (-1, -1),  5),
        ("VALIGN",       (0, 0), (-1, -1),  "MIDDLE"),
    ]))
    return t


def alert_box(text: str, bg=None, border=None, text_color=None):
    bg     = bg     or C_PANEL
    border = border or C_RED
    tc     = text_color or C_RED
    style  = ParagraphStyle("alert", fontSize=9, fontName="Courier-Bold",
                             textColor=tc, leading=13, leftIndent=8, rightIndent=8)
    inner  = Paragraph(text, style)
    t      = Table([[inner]], colWidths=["100%"])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,0), bg),
        ("BOX",           (0,0),(0,0), 1.2, border),
        ("TOPPADDING",    (0,0),(0,0), 8),
        ("BOTTOMPADDING", (0,0),(0,0), 8),
        ("LEFTPADDING",   (0,0),(0,0), 10),
    ]))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Page canvas (header/footer on every page)
# ─────────────────────────────────────────────────────────────────────────────

def on_page(canvas, doc):
    W, H = A4
    canvas.saveState()

    # Top bar
    canvas.setFillColor(C_PANEL)
    canvas.rect(0, H - 20*mm, W, 20*mm, fill=1, stroke=0)
    canvas.setFillColor(C_ACCENT)
    canvas.rect(0, H - 20*mm, W, 0.8, fill=1, stroke=0)
    canvas.setFont("Courier-Bold", 9)
    canvas.setFillColor(C_ACCENT)
    canvas.drawString(15*mm, H - 13*mm, "🛡  AEGISGUARD")
    canvas.setFillColor(C_MUTED)
    canvas.setFont("Courier", 7)
    canvas.drawRightString(W - 15*mm, H - 13*mm,
                           "STANDARD OPERATING PROCEDURE — CONFIDENTIAL")

    # Footer
    canvas.setFillColor(C_PANEL)
    canvas.rect(0, 0, W, 12*mm, fill=1, stroke=0)
    canvas.setFillColor(C_BORDER)
    canvas.rect(0, 12*mm, W, 0.5, fill=1, stroke=0)
    canvas.setFont("Courier", 7)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(15*mm, 5*mm, "AegisGuard Institutional Risk Platform  |  v2.0.0")
    canvas.drawRightString(W - 15*mm, 5*mm, f"Page {doc.page}")

    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Document body builder
# ─────────────────────────────────────────────────────────────────────────────

def build_story() -> list:
    story = []

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    story += [Spacer(1, 42*mm)]
    story.append(Paragraph("🛡  AEGISGUARD", S_COVER_TITLE))
    story.append(Paragraph("INSTITUTIONAL PORTFOLIO RISK MANAGEMENT PLATFORM", S_COVER_SUB))
    story.append(Spacer(1, 4*mm))
    story.append(rule(C_ACCENT, 1.5))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("STANDARD OPERATING PROCEDURE", S_COVER_SUB))
    story.append(Paragraph("v2.0.0  |  RESTRICTED — INTERNAL USE ONLY", S_COVER_DATE))
    story.append(Spacer(1, 60*mm))

    cover_meta = [
        ["Document ID",    "AG-SOP-2024-001"],
        ["Classification", "CONFIDENTIAL — FIRM EMPLOYEES ONLY"],
        ["Review Cycle",   "Quarterly"],
        ["Owner",          "Chief Risk Officer"],
        ["Applies To",     "Portfolio Management, Risk, Compliance"],
    ]
    ct = Table(cover_meta, colWidths=[55*mm, 110*mm])
    ct.setStyle(TableStyle([
        ("TEXTCOLOR",    (0,0),(0,-1), C_ACCENT),
        ("TEXTCOLOR",    (1,0),(1,-1), C_TEXT),
        ("FONTNAME",     (0,0),(-1,-1), "Courier"),
        ("FONTSIZE",     (0,0),(-1,-1), 8),
        ("BACKGROUND",   (0,0),(-1,-1), C_PANEL),
        ("GRID",         (0,0),(-1,-1), 0.4, C_BORDER),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
    ]))
    story.append(ct)
    story.append(PageBreak())

    # ── SECTION 1: PURPOSE & SCOPE ────────────────────────────────────────────
    story += section_header("1.  PURPOSE AND SCOPE")
    story.append(Paragraph(
        "AegisGuard is an institutional-grade portfolio risk and optimisation platform "
        "designed to enforce systematic, rules-based decision-making across the full "
        "investment lifecycle — from asset screening to position exit. This SOP defines "
        "the exact workflow that authorised personnel must follow to operate the platform.",
        S_BODY,
    ))
    story.append(Spacer(1, 3*mm))
    scope_data = [
        ["Module",               "Purpose",                              "Who Uses It"],
        ["Risk Filter Engine",   "Screens assets for liquidity, vol, correlation", "Risk / PM"],
        ["MVO Optimizer",        "Maximises Sharpe Ratio within constraints",      "Portfolio Manager"],
        ["Regime Detector",      "Market trend + VIX stance detection",            "Risk / CIO"],
        ["Exit Plan Engine",     "ATR-based trailing stops per asset",             "Trader / PM"],
        ["Dashboard",            "Real-time Bloomberg-style monitoring UI",         "All Staff"],
    ]
    story.append(table_dark(scope_data, col_widths=[45*mm, 80*mm, 40*mm]))

    # ── SECTION 2: WEEKLY OPERATING SCHEDULE ─────────────────────────────────
    story += section_header("2.  WEEKLY OPERATING SCHEDULE")
    story.append(alert_box(
        "⏰  MANDATORY: Run the full scan every MONDAY at 09:00 AM (local market time) "
        "before any trading activity begins.",
        bg=C_PANEL, border=C_ACCENT, text_color=C_ACCENT,
    ))
    story.append(Spacer(1, 3*mm))

    sched_data = [
        ["Day / Time",      "Task",                                    "Owner",   "Deadline"],
        ["MON  09:00",      "Run full AegisGuard scan",                "Risk",    "09:15"],
        ["MON  09:15",      "Review Zone classifications",             "PM",      "09:30"],
        ["MON  09:30",      "Execute any required NO-GO liquidations", "Trader",  "13:30"],
        ["MON  10:00",      "Send weekly risk brief to CIO",           "Risk",    "10:30"],
        ["WED  09:00",      "Mid-week regime check (VIX + SMA)",       "Risk",    "09:30"],
        ["FRI  16:00",      "Review ATR stops; update exit prices",    "PM",      "16:30"],
        ["DAILY (intraday)","Monitor for CRITICAL EXIT alerts",        "Trader",  "Continuous"],
    ]
    story.append(table_dark(sched_data, col_widths=[30*mm, 80*mm, 28*mm, 25*mm]))

    # ── SECTION 3: ZONE CLASSIFICATION PROTOCOL ───────────────────────────────
    story += section_header("3.  ZONE CLASSIFICATION PROTOCOL")

    zone_intro = (
        "Each asset in the watchlist is automatically classified into one of three zones "
        "by the AegisGuard filter engine. The following rules govern actions in each zone."
    )
    story.append(Paragraph(zone_intro, S_BODY))
    story.append(Spacer(1, 3*mm))

    zones = [
        ("GREEN ZONE", C_GREEN,
         "Asset passes all three filters: Liquidity Gate, Volatility Cap, and Correlation "
         "Matrix. Eligible for MVO optimisation and new long positions. Monitor ATR stops weekly."),
        ("NO-GO ZONE", C_RED,
         "Asset fails one or more filters. NEW positions are strictly prohibited. "
         "Existing positions must be reviewed for liquidation within 4 hours of classification."),
        ("BLACKLIST", C_YELLOW,
         "Asset triggered the Volatility Cap (20-day vol > 1.5x annual average). Automatic "
         "NO-GO override. Full liquidation required within 4 hours. Asset remains blacklisted "
         "until vol normalises for 5 consecutive trading days."),
    ]

    for zone_name, zone_colour, zone_desc in zones:
        zt = Table(
            [[Paragraph(f"● {zone_name}", ParagraphStyle(
                f"z{zone_name}", fontSize=10, fontName="Courier-Bold",
                textColor=zone_colour)),
              Paragraph(zone_desc, S_BODY)]],
            colWidths=[40*mm, 125*mm],
        )
        zt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), C_PANEL),
            ("BOX",           (0,0),(-1,-1), 0.5, zone_colour),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("RIGHTPADDING",  (0,0),(-1,-1), 8),
            ("TOPPADDING",    (0,0),(-1,-1), 7),
            ("BOTTOMPADDING", (0,0),(-1,-1), 7),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        story += [zt, Spacer(1, 3*mm)]

    # ── SECTION 4: FILTER DEFINITIONS & THRESHOLDS ───────────────────────────
    story += section_header("4.  FILTER DEFINITIONS AND THRESHOLDS")

    story.append(Paragraph("4.1  Liquidity Gate", S_SUBSEC))
    story.append(Paragraph(
        "The Liquidity Gate prevents the firm from placing trades that would move the market "
        "against itself. An asset is rejected if:", S_BODY))
    story.append(Paragraph(
        "•  Trade Size USD  ÷  20-Day Average Daily Volume (USD)  >  1.0%", S_BULLET))
    story.append(Paragraph(
        "•  Default firm trade size: $500,000. Adjustable in the sidebar.", S_BULLET))
    story.append(Paragraph(
        "•  Rationale: A trade exceeding 1% of ADV creates measurable slippage, "
        "increasing the effective entry/exit cost.", S_NOTE))

    story.append(Paragraph("4.2  Volatility Cap", S_SUBSEC))
    story.append(Paragraph(
        "Protects the portfolio from entering assets undergoing abnormal stress regimes:", S_BODY))
    story.append(Paragraph(
        "•  Compute: 20-Day rolling annualised return standard deviation (sigma_short)", S_BULLET))
    story.append(Paragraph(
        "•  Compute: 252-Day rolling annualised standard deviation (sigma_long)", S_BULLET))
    story.append(Paragraph(
        "•  REJECT if: sigma_short  >  1.5 x sigma_long", S_BULLET))
    story.append(Paragraph(
        "•  Multiplier 1.5 is configurable. Lower values (1.2x) give earlier warnings; "
        "higher values (2.0x) tolerate more variance.", S_NOTE))

    story.append(Paragraph("4.3  Correlation Matrix", S_SUBSEC))
    story.append(Paragraph(
        "Prevents \"fake diversification\" — where two assets appear different but move together:", S_BODY))
    story.append(Paragraph(
        "•  Compute 252-day pairwise Pearson correlation on log-returns", S_BULLET))
    story.append(Paragraph(
        "•  When |rho(A, B)|  >  0.85, flag the lower-ranked asset for removal", S_BULLET))
    story.append(Paragraph(
        "•  The surviving asset is the one listed earlier in the watchlist (higher priority)", S_BULLET))
    story.append(Paragraph(
        "•  Threshold 0.85 is configurable. Institutional standard is 0.75–0.90.", S_NOTE))

    # ── SECTION 5: OPTIMIZATION PROTOCOL ─────────────────────────────────────
    story += section_header("5.  PORTFOLIO OPTIMIZATION PROTOCOL")

    story.append(Paragraph(
        "The MVO engine runs Mean-Variance Optimization on all Green Zone assets to find "
        "the weight vector w that maximises the Sharpe Ratio:", S_BODY))

    formula_style = ParagraphStyle("formula", fontSize=10, fontName="Courier-Bold",
                                   textColor=C_ACCENT, alignment=TA_CENTER,
                                   spaceBefore=8, spaceAfter=8)
    story.append(Paragraph("Sharpe = (w^T mu - rf)  /  sqrt(w^T Sigma w)", formula_style))

    constraints = [
        ["Constraint",           "Value",              "Rationale"],
        ["Sum of weights",       "= 1.0 (100%)",       "Fully invested mandate"],
        ["Individual max weight","<= 20% (default)",   "Concentration limit (MiFID II / ERISA)"],
        ["Short selling",        "Prohibited (w >= 0)","Long-only institutional mandate"],
        ["Risk-free rate",       "4.5% (default)",     "US T-Bill proxy; adjust to client jurisdiction"],
        ["Lookback window",      "252 trading days",   "1-year rolling parameter estimation"],
    ]
    story.append(table_dark(constraints, col_widths=[52*mm, 42*mm, 71*mm]))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "IMPORTANT: The optimizer uses CVXPY (preferred) with SciPy SLSQP as fallback. "
        "Always verify the 'Solver Used' field in the Optimization tab post-run. "
        "If status shows 'suboptimal', increase the asset universe or relax constraints.", S_NOTE))

    # ── SECTION 6: REGIME DETECTION & DEFENSIVE MODE ─────────────────────────
    story += section_header("6.  MARKET REGIME DETECTION")

    story.append(Paragraph("6.1  Trend Regime (S&P 500 SMA-200)", S_SUBSEC))
    trend_data = [
        ["Signal",  "Condition",                    "Required Action"],
        ["BULL",    "SPX Close  >  200-Day SMA",    "Normal risk-on positioning"],
        ["BEAR",    "SPX Close  <  200-Day SMA",    "Reduce gross equity exposure by >= 25% within 2 trading days"],
        ["UNKNOWN", "Data unavailable",             "Default to CAUTION posture; escalate to CIO"],
    ]
    story.append(table_dark(trend_data, col_widths=[25*mm, 55*mm, 85*mm]))

    story.append(Paragraph("6.2  VIX Regime (Volatility Index)", S_SUBSEC))
    vix_data_table = [
        ["VIX Level",  "Signal",    "Cash Buffer",  "Required Action"],
        ["< 20",       "NORMAL",    "5%",           "Standard allocation; hold cash reserve"],
        ["20 – 30",    "ELEVATED",  "15%",          "Monitor closely; tighten stop-losses"],
        ["> 30",       "DEFENSIVE", "30–50%",       "DEFENSIVE MODE: raise cash, no new longs, tighten ATR stops to 1.5x"],
    ]
    story.append(table_dark(vix_data_table, col_widths=[25*mm, 25*mm, 28*mm, 87*mm]))
    story.append(Spacer(1, 3*mm))
    story.append(alert_box(
        "⚠  DEFENSIVE MODE ACTIVATION: When VIX exceeds 30, the dashboard will display a "
        "red DEFENSIVE MODE banner. The ATR stop multiplier automatically tightens from "
        "2.5x to 1.5x. All new long positions are prohibited until VIX closes below 25 "
        "for three consecutive sessions.",
        bg=C_PANEL, border=C_RED, text_color=C_RED,
    ))

    # ── SECTION 7: EXIT PLAN & TRAILING STOPS ────────────────────────────────
    story += section_header("7.  EXIT PLAN AND TRAILING STOP PROTOCOL")

    story.append(Paragraph(
        "AegisGuard calculates a daily ATR-based exit price for every Green Zone asset. "
        "This is the firm's primary downside protection mechanism.", S_BODY))

    story.append(Paragraph("7.1  ATR Calculation", S_SUBSEC))
    story.append(Paragraph(
        "Average True Range (ATR) over 14 periods using Wilder smoothing:", S_BODY))
    story.append(Paragraph(
        "•  True Range  =  max( H-L,  |H - Close_prev|,  |L - Close_prev| )", S_BULLET))
    story.append(Paragraph(
        "•  ATR(14) smoothed with Wilder's EMA formula", S_BULLET))

    story.append(Paragraph("7.2  Stop Price Calculation", S_SUBSEC))
    story.append(Paragraph("Exit Price  =  Latest Close  -  (ATR_Multiplier  x  ATR14)", formula_style))
    story.append(table_dark([
        ["Mode",           "ATR Multiplier",  "Trigger"],
        ["Standard",       "2.5x",            "Normal regime (VIX < 30)"],
        ["Defensive",      "1.5x",            "VIX > 30 (auto-tightened)"],
    ], col_widths=[40*mm, 40*mm, 85*mm]))

    story.append(Paragraph("7.3  Mandatory Exit Protocol", S_SUBSEC))
    story.append(alert_box(
        "RULE: If an asset's current price falls at or below its calculated Exit Price, "
        "the AegisGuard dashboard will display a red 'CRITICAL EXIT' badge. "
        "The position MUST be fully liquidated within 4 hours of market open.",
        bg=C_PANEL, border=C_RED, text_color=C_RED,
    ))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Exit prices are recalculated daily. The trader must check the EXIT PLAN tab each "
        "morning before the first trade. Update the firm's order management system with "
        "the new stop levels immediately after the Monday scan.", S_BODY))

    # ── SECTION 8: ESCALATION MATRIX ─────────────────────────────────────────
    story += section_header("8.  ESCALATION AND RESPONSE MATRIX")

    esc_data = [
        ["Event",                         "Severity",  "Response",                          "Deadline",     "Escalate To"],
        ["Asset to NO-GO Zone",           "HIGH",      "Halt new positions; begin exit",    "4 hours",      "Head of Risk"],
        ["Asset to BLACKLIST",            "CRITICAL",  "Full immediate liquidation",         "4 hours",      "CIO + Compliance"],
        ["VIX > 30 (Defensive Mode)",     "CRITICAL",  "Raise cash; tighten stops",          "Same day",     "CIO"],
        ["BEAR Trend Detected",           "HIGH",      "Reduce gross exposure >= 25%",       "2 trading days","CIO + PM"],
        ["ATR Stop Breached",             "CRITICAL",  "Full exit within session",           "4 hours",      "Head Trader"],
        ["Correlation pair flagged",      "MEDIUM",    "Flag for Monday rebalancing",        "Next Monday",  "PM"],
        ["All data providers offline",    "HIGH",      "Use last cached data; notify Risk",  "Immediate",    "Head of Risk"],
        ["Optimizer returns 'failed'",    "MEDIUM",    "Check asset count; review constraints","2 hours",    "Quant Analyst"],
    ]
    story.append(table_dark(esc_data, col_widths=[50*mm, 20*mm, 50*mm, 24*mm, 21*mm]))

    # ── SECTION 9: DATA AND SYSTEM RELIABILITY ───────────────────────────────
    story += section_header("9.  DATA PIPELINE AND SYSTEM RELIABILITY")

    story.append(Paragraph("9.1  Provider Failover Order", S_SUBSEC))
    story.append(Paragraph(
        "AegisGuard uses a multi-provider data architecture. If the preferred source fails, "
        "the system automatically retries the next provider:", S_BODY))
    story.append(table_dark([
        ["Priority", "Provider",      "Cost",      "Rate Limit"],
        ["1",        "Yahoo Finance", "Free",      "~2,000 req/hr"],
        ["2",        "AlphaVantage",  "Freemium",  "5 req/min (free); 75/min (paid)"],
        ["3",        "Polygon.io",    "Paid",      "Unlimited (paid tier)"],
    ], col_widths=[20*mm, 50*mm, 35*mm, 60*mm]))

    story.append(Paragraph("9.2  Local Cache", S_SUBSEC))
    story.append(Paragraph(
        "All fetched data is cached locally in /data/_cache/ as Parquet files. "
        "Cache validity period: 4 hours. If a Monday 09:00 scan is run before market open, "
        "the system will serve Friday's cached data — this is expected behaviour. "
        "Force a fresh pull using the 'Force Refresh' toggle if available.", S_BODY))

    story.append(Paragraph("9.3  Deployment", S_SUBSEC))
    story.append(table_dark([
        ["Environment",      "Platform",                   "Access"],
        ["Production",       "Streamlit Community Cloud",  "Firm URL + password auth"],
        ["Staging",          "Local (streamlit run app.py)","Developer only"],
        ["Backup",           "AWS EC2 or Azure App Service","CTO discretion"],
    ], col_widths=[40*mm, 65*mm, 60*mm]))

    # ── SECTION 10: COMPLIANCE NOTES ─────────────────────────────────────────
    story += section_header("10.  COMPLIANCE AND AUDIT NOTES")

    story.append(Paragraph(
        "AegisGuard is a decision-support tool. It does not constitute investment advice "
        "and does not replace the judgment of qualified investment professionals. All "
        "outputs must be reviewed by an authorised person before execution.", S_WARNING))
    story.append(Spacer(1, 2*mm))
    compliance = [
        "•  Retain scan reports for a minimum of 7 years (MiFID II / SEC Rule 17a-4).",
        "•  All override decisions (ignoring a NO-GO flag) must be documented in writing "
           "and countersigned by the CIO within 24 hours.",
        "•  Access credentials must be rotated every 90 days.",
        "•  The risk-free rate used in Sharpe calculations must reflect the firm's "
           "jurisdiction (e.g., ECB rate for EUR-denominated funds).",
        "•  This document must be reviewed and re-approved by the CRO every quarter.",
    ]
    for line in compliance:
        story.append(Paragraph(line, S_BULLET))

    # ── SECTION 11: QUICK REFERENCE CARD ─────────────────────────────────────
    story += section_header("11.  QUICK REFERENCE — MORNING CHECKLIST")

    checklist = [
        ["#",  "Task",                                                      "Done?"],
        ["1",  "Log into AegisGuard dashboard (Monday 09:00)",              "[ ]"],
        ["2",  "Verify data source status (sidebar — green = live)",        "[ ]"],
        ["3",  "Enter/confirm watchlist tickers",                           "[ ]"],
        ["4",  "Click  RUN FULL SCAN",                                      "[ ]"],
        ["5",  "Review OVERVIEW tab — check zone counts",                   "[ ]"],
        ["6",  "Action all NO-GO / BLACKLIST assets (Section 3)",           "[ ]"],
        ["7",  "Check MARKET REGIME tab — note VIX and SMA-200 signal",     "[ ]"],
        ["8",  "Check EXIT PLAN tab — update stops in OMS",                 "[ ]"],
        ["9",  "Action any CRITICAL EXIT alerts within 4 hours",            "[ ]"],
        ["10", "Send weekly risk brief to CIO by 10:30",                    "[ ]"],
    ]
    story.append(table_dark(checklist, col_widths=[10*mm, 140*mm, 15*mm]))

    # ── SIGN-OFF ──────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph("DOCUMENT APPROVAL", S_SECTION))
    story.append(rule(C_ACCENT, 0.8))
    story.append(Spacer(1, 8*mm))

    signoff = [
        ["Role",              "Name",     "Signature",    "Date"],
        ["Chief Risk Officer","",         "",             ""],
        ["Portfolio Manager", "",         "",             ""],
        ["Compliance Officer","",         "",             ""],
        ["Chief Technology Officer","",   "",             ""],
    ]
    so_table = Table(signoff, colWidths=[55*mm, 45*mm, 45*mm, 20*mm])
    so_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),   C_PANEL),
        ("TEXTCOLOR",     (0,0),(-1,0),   C_ACCENT),
        ("FONTNAME",      (0,0),(-1,0),   "Courier-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1),  8),
        ("FONTNAME",      (0,1),(-1,-1),  "Courier"),
        ("TEXTCOLOR",     (0,1),(-1,-1),  C_TEXT),
        ("GRID",          (0,0),(-1,-1),  0.4, C_BORDER),
        ("BACKGROUND",    (0,1),(-1,-1),  C_DARK),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [C_DARK, C_PANEL]),
        ("TOPPADDING",    (0,0),(-1,-1),  10),
        ("BOTTOMPADDING", (0,0),(-1,-1),  10),
        ("LEFTPADDING",   (0,0),(-1,-1),  6),
    ]))
    story.append(so_table)
    story.append(Spacer(1, 12*mm))
    story.append(Paragraph(
        "This document is the property of the firm. Any reproduction or distribution "
        "outside the firm without written authorisation from the CRO is strictly prohibited.",
        S_FOOTER,
    ))

    return story


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "AegisGuard_SOP.pdf"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize    = A4,
        leftMargin  = 15*mm,
        rightMargin = 15*mm,
        topMargin   = 24*mm,
        bottomMargin= 16*mm,
        title       = "AegisGuard — Standard Operating Procedure",
        author      = "AegisGuard Risk Platform",
        subject     = "Institutional Portfolio Risk SOP",
    )

    story = build_story()
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"✅  SOP generated → {out_path}")
