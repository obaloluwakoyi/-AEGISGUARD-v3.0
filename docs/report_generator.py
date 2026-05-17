"""
aegisguard/docs/report_generator.py
─────────────────────────────────────────────────────────────────────────────
Generates a Bloomberg-style PDF scan report from live session data.
Called by app.py and returns raw bytes for st.download_button().
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units     import mm
from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
from reportlab.lib            import colors
from reportlab.platypus       import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── Colour palette (matches dashboard)
C_DARK   = colors.HexColor("#0A0E1A")
C_PANEL  = colors.HexColor("#0F1629")
C_ACCENT = colors.HexColor("#00D4FF")
C_GREEN  = colors.HexColor("#00C06A")
C_YELLOW = colors.HexColor("#FFD600")
C_RED    = colors.HexColor("#FF3860")
C_ORANGE = colors.HexColor("#FF6B35")
C_TEXT   = colors.HexColor("#E8EAF0")
C_MUTED  = colors.HexColor("#8892A4")
C_BORDER = colors.HexColor("#1E2A3E")

base = getSampleStyleSheet()

def _s(name, **kw):
    kw.setdefault("textColor", C_TEXT)
    kw.setdefault("fontName",  "Courier")
    return ParagraphStyle(name, parent=base["Normal"], **kw)

S_TITLE   = _s("T",  fontSize=20, textColor=C_ACCENT, fontName="Courier-Bold",
                alignment=TA_CENTER, spaceAfter=4)
S_SUB     = _s("Su", fontSize=8,  textColor=C_MUTED,  alignment=TA_CENTER, spaceAfter=2)
S_SEC     = _s("S",  fontSize=11, textColor=C_ACCENT, fontName="Courier-Bold",
                spaceBefore=14, spaceAfter=4)
S_BODY    = _s("B",  fontSize=8,  leading=13, spaceAfter=3)
S_BULLET  = _s("Bu", fontSize=8,  leading=12, leftIndent=12, spaceAfter=2)
S_WARN    = _s("W",  fontSize=8,  textColor=C_RED,    fontName="Courier-Bold", spaceAfter=4)
S_FOOTER  = _s("F",  fontSize=6,  textColor=C_MUTED,  alignment=TA_CENTER)


def _rule(c=C_BORDER, t=0.5):
    return HRFlowable(width="100%", thickness=t, color=c, spaceAfter=6, spaceBefore=2)


def _tbl(data, widths=None, header_color=C_ACCENT):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  C_PANEL),
        ("TEXTCOLOR",     (0,0),(-1,0),  header_color),
        ("FONTNAME",      (0,0),(-1,0),  "Courier-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 7.5),
        ("FONTNAME",      (0,1),(-1,-1), "Courier"),
        ("TEXTCOLOR",     (0,1),(-1,-1), C_TEXT),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_DARK, C_PANEL]),
        ("GRID",          (0,0),(-1,-1), 0.3, C_BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("RIGHTPADDING",  (0,0),(-1,-1), 5),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    return t


def _zone_color(zone: str) -> colors.Color:
    return {
        "GREEN":     C_GREEN,
        "NO-GO":     C_RED,
        "BLACKLIST": C_ORANGE,
    }.get(zone, C_MUTED)


def _signal_color(signal: str) -> colors.Color:
    return {
        "BULL": C_GREEN, "BEAR": C_RED,
        "NORMAL": C_GREEN, "ELEVATED": C_YELLOW, "DEFENSIVE": C_RED,
        "RISK-ON": C_GREEN, "CAUTION": C_YELLOW, "RISK-OFF": C_RED,
    }.get(signal, C_ACCENT)


def _on_page(canvas, doc, scan_time: str):
    W, H = A4
    canvas.saveState()
    # Header bar
    canvas.setFillColor(C_PANEL)
    canvas.rect(0, H - 18*mm, W, 18*mm, fill=1, stroke=0)
    canvas.setFillColor(C_ACCENT)
    canvas.rect(0, H - 18*mm, W, 0.7, fill=1, stroke=0)
    canvas.setFont("Courier-Bold", 8)
    canvas.setFillColor(C_ACCENT)
    canvas.drawString(14*mm, H - 11*mm, "AEGISGUARD")
    canvas.setFont("Courier", 6.5)
    canvas.setFillColor(C_MUTED)
    canvas.drawRightString(W - 14*mm, H - 11*mm,
                           f"RISK SCAN REPORT  |  {scan_time}  |  CONFIDENTIAL")
    # Footer
    canvas.setFillColor(C_PANEL)
    canvas.rect(0, 0, W, 11*mm, fill=1, stroke=0)
    canvas.setFillColor(C_BORDER)
    canvas.rect(0, 11*mm, W, 0.4, fill=1, stroke=0)
    canvas.setFont("Courier", 6)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(14*mm, 4*mm, "AegisGuard Institutional Risk Platform  |  v2.0.0")
    canvas.drawRightString(W - 14*mm, 4*mm, f"Page {doc.page}")
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def generate_report_pdf(
    filter_result,
    opt_result,
    regime_report,
    price_dict: dict,
    scan_time: str | None = None,
) -> bytes:
    """
    Build a full scan-results PDF and return it as raw bytes.
    Pass directly to st.download_button(data=...).
    """
    scan_time = scan_time or datetime.now().strftime("%Y-%m-%d  %H:%M UTC")
    buf       = io.BytesIO()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=14*mm, rightMargin=14*mm,
        topMargin=22*mm, bottomMargin=14*mm,
        title="AegisGuard Risk Scan Report",
    )

    story = []

    # ── COVER BLOCK ──────────────────────────────────────────────────────────
    story += [Spacer(1, 18*mm)]
    story.append(Paragraph("🛡  AEGISGUARD", S_TITLE))
    story.append(Paragraph("PORTFOLIO RISK SCAN REPORT", S_SUB))
    story.append(Paragraph(scan_time, S_SUB))
    story.append(Spacer(1, 3*mm))
    story.append(_rule(C_ACCENT, 1.2))
    story.append(Spacer(1, 4*mm))

    # Summary row
    n_scanned = len(price_dict)
    n_green   = len(filter_result.green_zone)
    n_nogo    = len(filter_result.no_go_zone)
    n_black   = len(filter_result.blacklisted)
    stance    = regime_report.overall_signal if regime_report else "—"
    sharpe    = f"{opt_result.sharpe_ratio:.3f}" if opt_result and opt_result.status == "optimal" else "—"

    cover_data = [
        ["METRIC",            "VALUE"],
        ["Assets Scanned",    str(n_scanned)],
        ["Green Zone",        str(n_green)],
        ["No-Go Zone",        str(n_nogo)],
        ["Blacklisted",       str(n_black)],
        ["Market Stance",     stance],
        ["Portfolio Sharpe",  sharpe],
    ]
    ct = _tbl(cover_data, widths=[80*mm, 80*mm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  C_PANEL),
        ("TEXTCOLOR",     (0,0),(-1,0),  C_ACCENT),
        ("FONTNAME",      (0,0),(-1,0),  "Courier-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("FONTNAME",      (0,1),(-1,-1), "Courier"),
        ("TEXTCOLOR",     (0,1),(0,-1),  C_MUTED),
        ("TEXTCOLOR",     (1,1),(1,-1),  C_TEXT),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_DARK, C_PANEL]),
        ("GRID",          (0,0),(-1,-1), 0.3, C_BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("ALIGN",         (1,0),(1,-1),  "CENTER"),
    ]))
    story.append(ct)

    # ── SECTION 1 — ZONE CLASSIFICATION ──────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("1.  ASSET ZONE CLASSIFICATION", S_SEC))
    story.append(_rule())

    zone_rows = [["TICKER", "ZONE", "ADV (USD)", "TRADE % ADV", "VOL 20D", "VOL 1Y"]]
    for ticker in sorted(price_dict.keys()):
        zone  = filter_result.zone_of(ticker)
        liq   = filter_result.liquidity.details.get(ticker, {})
        vol   = filter_result.volatility.details.get(ticker, {})
        zone_rows.append([
            ticker,
            zone,
            f"${liq.get('adv_usd', 0):,.0f}"      if liq.get("adv_usd")      else "—",
            f"{liq.get('pct_of_adv', 0):.3f}%"    if liq.get("pct_of_adv")   else "—",
            f"{vol.get('vol_20d_ann', 0):.1f}%"   if vol.get("vol_20d_ann")  else "—",
            f"{vol.get('vol_1yr_ann', 0):.1f}%"   if vol.get("vol_1yr_ann")  else "—",
        ])

    zt = _tbl(zone_rows, widths=[22*mm, 24*mm, 38*mm, 28*mm, 24*mm, 24*mm])

    # Colour-code Zone column
    zone_style = []
    for i, row in enumerate(zone_rows[1:], start=1):
        c = _zone_color(row[1])
        zone_style.append(("TEXTCOLOR", (1, i), (1, i), c))
        zone_style.append(("FONTNAME",  (1, i), (1, i), "Courier-Bold"))
    zt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  C_PANEL),
        ("TEXTCOLOR",     (0,0),(-1,0),  C_ACCENT),
        ("FONTNAME",      (0,0),(-1,0),  "Courier-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 7.5),
        ("FONTNAME",      (0,1),(-1,-1), "Courier"),
        ("TEXTCOLOR",     (0,1),(-1,-1), C_TEXT),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_DARK, C_PANEL]),
        ("GRID",          (0,0),(-1,-1), 0.3, C_BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        *zone_style,
    ]))
    story.append(zt)

    # ── SECTION 2 — FILTER SUMMARY ───────────────────────────────────────────
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("2.  RISK FILTER SUMMARY", S_SEC))
    story.append(_rule())

    for label, res in [
        ("Liquidity Gate",   filter_result.liquidity),
        ("Volatility Cap",   filter_result.volatility),
        ("Correlation Gate", filter_result.correlation),
    ]:
        pass_rate = f"{res.pass_rate*100:.0f}%"
        passed    = ", ".join(res.passed)  or "—"
        rejected  = ", ".join(res.rejected) or "—"
        fd = [
            ["FILTER",   "PASS RATE", "PASSED",  "REJECTED"],
            [label,      pass_rate,   passed,    rejected],
        ]
        ft = _tbl(fd, widths=[38*mm, 22*mm, 67*mm, 33*mm])
        story += [ft, Spacer(1, 2*mm)]

    # Flagged correlation pairs
    flagged = filter_result.correlation.details.get("flagged_pairs", [])
    if flagged:
        story.append(Paragraph("Over-Correlated Pairs (flagged for removal):", S_BODY))
        fp_rows = [["ASSET 1", "ASSET 2", "CORRELATION (rho)"]]
        for p in flagged:
            fp_rows.append([p["asset_1"], p["asset_2"], str(p["rho"])])
        story.append(_tbl(fp_rows, widths=[55*mm, 55*mm, 50*mm]))

    # ── SECTION 3 — OPTIMIZATION RESULTS ─────────────────────────────────────
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("3.  PORTFOLIO OPTIMIZATION (MVO — MAX SHARPE)", S_SEC))
    story.append(_rule())

    if opt_result and opt_result.status == "optimal":
        metrics = [
            ["METRIC",           "VALUE"],
            ["Expected Return",  f"{opt_result.expected_return*100:.2f}% p.a."],
            ["Volatility",       f"{opt_result.volatility*100:.2f}% p.a."],
            ["Sharpe Ratio",     f"{opt_result.sharpe_ratio:.4f}"],
            ["Solver Used",      opt_result.solver_used.upper()],
            ["Assets in Pool",   str(opt_result.diagnostics.get("n_assets", "—"))],
            ["Max Single Weight",f"{opt_result.diagnostics.get('max_weight', 0)*100:.0f}%"],
        ]
        story.append(_tbl(metrics, widths=[70*mm, 90*mm]))
        story.append(Spacer(1, 3*mm))

        # Weights table
        w_rows = [["TICKER", "WEIGHT (%)", "ALLOCATION (per $1M)"]]
        for ticker, w in sorted(opt_result.weights.items(), key=lambda x: -x[1]):
            w_rows.append([ticker, f"{w*100:.2f}%", f"${w*1_000_000:,.0f}"])
        story.append(_tbl(w_rows, widths=[50*mm, 50*mm, 60*mm]))
    else:
        story.append(Paragraph(
            "Optimization not available — insufficient Green Zone assets or data.",
            S_WARN))

    # ── SECTION 4 — MARKET REGIME ─────────────────────────────────────────────
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("4.  MARKET REGIME DETECTION", S_SEC))
    story.append(_rule())

    if regime_report:
        trend = regime_report.trend
        vix   = regime_report.vix

        reg_data = [
            ["INDICATOR",         "VALUE",                   "SIGNAL"],
            ["S&P 500 Price",     f"${trend.spx_price:,.0f}", ""],
            ["200-Day SMA",       f"${trend.sma_200:,.0f}",   trend.signal],
            ["SMA Distance",      f"{trend.distance_pct:+.2f}%", ""],
            ["VIX Level",         f"{vix.vix_level:.1f}",     vix.signal],
            ["Suggested Cash",    f"{vix.suggested_cash_pct:.0f}%", ""],
            ["Overall Stance",    "",                          regime_report.overall_signal],
        ]
        rt = _tbl(reg_data, widths=[55*mm, 50*mm, 55*mm])
        # Colour signal column
        sig_styles = []
        signal_map = {
            2: trend.signal,
            4: vix.signal,
            6: regime_report.overall_signal,
        }
        for row_i, sig in signal_map.items():
            c = _signal_color(sig)
            sig_styles += [
                ("TEXTCOLOR", (2, row_i), (2, row_i), c),
                ("FONTNAME",  (2, row_i), (2, row_i), "Courier-Bold"),
            ]
        rt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), C_PANEL),
            ("TEXTCOLOR",     (0,0),(-1,0), C_ACCENT),
            ("FONTNAME",      (0,0),(-1,0), "Courier-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("FONTNAME",      (0,1),(-1,-1), "Courier"),
            ("TEXTCOLOR",     (0,1),(-1,-1), C_TEXT),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_DARK, C_PANEL]),
            ("GRID",          (0,0),(-1,-1), 0.3, C_BORDER),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
            *sig_styles,
        ]))
        story.append(rt)
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(trend.description, S_BODY))
        story.append(Paragraph(vix.description,   S_BODY))

    # ── SECTION 5 — EXIT PLAN ─────────────────────────────────────────────────
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("5.  EXIT PLAN — ATR TRAILING STOPS", S_SEC))
    story.append(_rule())

    if regime_report and regime_report.exit_plans:
        atr_mult = 1.5 if regime_report.vix.is_defensive else 2.5
        story.append(Paragraph(
            f"ATR Multiplier: {atr_mult}x  "
            f"({'Defensive Mode — VIX > 30' if regime_report.vix.is_defensive else 'Standard Mode'})",
            S_BODY,
        ))

        ep_rows = [["TICKER", "CURRENT PRICE", "ATR-14", "EXIT PRICE", "BUFFER %", "ACTION"]]
        for ep in regime_report.exit_plans:
            ep_rows.append([
                ep.ticker,
                f"${ep.current_price:,.2f}",
                f"${ep.atr_14:,.3f}",
                f"${ep.stop_price:,.2f}",
                f"{ep.stop_pct_below:.2f}%",
                ep.action,
            ])

        et = _tbl(ep_rows, widths=[22*mm, 30*mm, 26*mm, 28*mm, 22*mm, 32*mm])

        action_styles = []
        for i, ep in enumerate(regime_report.exit_plans, start=1):
            c = C_RED if ep.action == "EXIT NOW" else C_GREEN
            action_styles += [
                ("TEXTCOLOR", (5, i), (5, i), c),
                ("FONTNAME",  (5, i), (5, i), "Courier-Bold"),
            ]
        et.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), C_PANEL),
            ("TEXTCOLOR",     (0,0),(-1,0), C_ACCENT),
            ("FONTNAME",      (0,0),(-1,0), "Courier-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("FONTNAME",      (0,1),(-1,-1), "Courier"),
            ("TEXTCOLOR",     (0,1),(-1,-1), C_TEXT),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_DARK, C_PANEL]),
            ("GRID",          (0,0),(-1,-1), 0.3, C_BORDER),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
            *action_styles,
        ]))
        story.append(et)

        # Critical exits callout
        critical = [ep for ep in regime_report.exit_plans if ep.action == "EXIT NOW"]
        if critical:
            story.append(Spacer(1, 3*mm))
            for ep in critical:
                story.append(Paragraph(
                    f"⚠  CRITICAL EXIT: {ep.ticker} — Current ${ep.current_price:,.2f} "
                    f"has breached stop ${ep.stop_price:,.2f}. Liquidate within 4 hours.",
                    S_WARN,
                ))
    else:
        story.append(Paragraph("No exit plan data available.", S_BODY))

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(_rule(C_BORDER, 0.3))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by AegisGuard, a decision-support tool. "
        "It does not constitute investment advice. All outputs must be reviewed by an "
        "authorised investment professional before execution. Retain for 7 years (MiFID II).",
        S_FOOTER,
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(
        story,
        onFirstPage  = lambda c, d: _on_page(c, d, scan_time),
        onLaterPages = lambda c, d: _on_page(c, d, scan_time),
    )
    return buf.getvalue()
