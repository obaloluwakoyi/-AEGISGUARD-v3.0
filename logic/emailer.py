"""
aegisguard/logic/emailer.py
─────────────────────────────────────────────────────────────────────────────
Email Dispatch Module

Supports:
  • SMTP with TLS (Gmail, Outlook / Hotmail, Yahoo, custom server)
  • HTML-formatted intelligence reports
  • PDF attachment (the generated AegisGuard report)
  • Plain-text critical-alert digest
  • Multiple recipients (To + CC)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass, field
from datetime    import datetime
from email                    import encoders
from email.mime.base          import MIMEBase
from email.mime.multipart     import MIMEMultipart
from email.mime.text          import MIMEText


# ─────────────────────────────────────────────────────────────────────────────
# Preset SMTP configurations for popular providers
# ─────────────────────────────────────────────────────────────────────────────

SMTP_PRESETS: dict[str, dict] = {
    "Gmail":           {"host": "smtp.gmail.com",        "port": 587, "tls": True},
    "Outlook / Hotmail": {"host": "smtp-mail.outlook.com", "port": 587, "tls": True},
    "Yahoo Mail":      {"host": "smtp.mail.yahoo.com",   "port": 587, "tls": True},
    "iCloud Mail":     {"host": "smtp.mail.me.com",      "port": 587, "tls": True},
    "Zoho Mail":       {"host": "smtp.zoho.com",         "port": 587, "tls": True},
    "Custom SMTP":     {"host": "",                       "port": 587, "tls": True},
}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SendResult:
    success:   bool
    message:   str
    recipients: list[str] = field(default_factory=list)
    timestamp:  str        = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M UTC"))


# ─────────────────────────────────────────────────────────────────────────────
# HTML template builder
# ─────────────────────────────────────────────────────────────────────────────

def build_html_body(
    brief_text:     str  = "",
    alert_lines:    list[str] | None = None,
    portfolio_rows: list[dict] | None = None,
    stance:         str  = "",
    scan_time:      str  = "",
    include_alerts:    bool = True,
    include_brief:     bool = True,
    include_portfolio: bool = True,
) -> str:
    """Build a styled HTML email body."""

    stance_colour = {
        "RISK-ON": "#00FF88", "CAUTION": "#FFD600", "RISK-OFF": "#FF3860",
        "BULL": "#00FF88",    "BEAR": "#FF3860",    "NORMAL": "#00FF88",
        "ELEVATED": "#FFD600","DEFENSIVE": "#FF3860",
    }.get(stance, "#00D4FF")

    rows_html = ""
    if include_portfolio and portfolio_rows:
        rows_html = "".join(
            f"<tr style='border-bottom:1px solid #1E2A3E'>"
            f"<td style='padding:6px 10px;font-family:monospace;color:#00D4FF'>{r.get('Ticker','')}</td>"
            f"<td style='padding:6px 10px'>{r.get('Zone','')}</td>"
            f"<td style='padding:6px 10px'>{r.get('Risk Score','')}</td>"
            f"<td style='padding:6px 10px'>{r.get('Opt Weight','')}</td>"
            f"<td style='padding:6px 10px'>{r.get('Exit Action','')}</td>"
            f"<td style='padding:6px 10px'>{r.get('Sentiment','')}</td>"
            f"</tr>"
            for r in portfolio_rows[:20]
        )

    alerts_html = ""
    if include_alerts and alert_lines:
        alerts_html = f"""
        <div style="margin:20px 0">
          <h3 style="color:#FF3860;font-family:monospace;letter-spacing:0.1em;border-bottom:1px solid #FF3860;padding-bottom:6px">
            🚨 CRITICAL ALERTS
          </h3>
          {"".join(f'<div style="background:#3D0000;border-left:4px solid #FF3860;padding:10px 14px;margin:6px 0;border-radius:0 6px 6px 0;font-size:13px">{a}</div>' for a in alert_lines[:10])}
        </div>"""

    brief_html = ""
    if include_brief and brief_text:
        brief_html = f"""
        <div style="margin:20px 0">
          <h3 style="color:#00D4FF;font-family:monospace;letter-spacing:0.1em;border-bottom:1px solid #00D4FF;padding-bottom:6px">
            💡 AI INTELLIGENCE BRIEF
          </h3>
          <div style="background:#0A1628;border:1px solid #1E2A3E;border-radius:8px;padding:18px;
                      font-size:13px;line-height:1.7;color:#CBD5E0;white-space:pre-wrap">{brief_text}</div>
        </div>"""

    portfolio_html = ""
    if include_portfolio and portfolio_rows:
        portfolio_html = f"""
        <div style="margin:20px 0">
          <h3 style="color:#00D4FF;font-family:monospace;letter-spacing:0.1em;border-bottom:1px solid #00D4FF;padding-bottom:6px">
            📋 ASSET INTELLIGENCE
          </h3>
          <table style="width:100%;border-collapse:collapse;font-size:12px;color:#CBD5E0">
            <thead>
              <tr style="background:#0F1629">
                <th style="padding:8px 10px;text-align:left;color:#8892A4">TICKER</th>
                <th style="padding:8px 10px;text-align:left;color:#8892A4">ZONE</th>
                <th style="padding:8px 10px;text-align:left;color:#8892A4">RISK SCORE</th>
                <th style="padding:8px 10px;text-align:left;color:#8892A4">WEIGHT</th>
                <th style="padding:8px 10px;text-align:left;color:#8892A4">EXIT</th>
                <th style="padding:8px 10px;text-align:left;color:#8892A4">SENTIMENT</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<body style="background:#060D1A;color:#CBD5E0;font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:0">
  <div style="max-width:720px;margin:0 auto;padding:32px 20px">

    <!-- Header -->
    <div style="background:#0F1629;border:2px solid {stance_colour};border-radius:12px;
                padding:24px 28px;margin-bottom:24px;text-align:center">
      <div style="font-size:11px;color:#8892A4;letter-spacing:0.2em;font-family:monospace">
        AEGISGUARD · INTELLIGENCE REPORT · {scan_time or datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
      </div>
      <div style="font-size:28px;font-weight:900;color:{stance_colour};
                  font-family:monospace;letter-spacing:0.06em;margin:10px 0 4px">
        {stance or "AEGISGUARD"}
      </div>
      <div style="font-size:11px;color:#8892A4">GLOBAL INSTITUTIONAL RISK PLATFORM v3.0</div>
    </div>

    {alerts_html}
    {brief_html}
    {portfolio_html}

    <!-- Footer -->
    <div style="margin-top:32px;padding-top:16px;border-top:1px solid #1E2A3E;
                font-size:10px;color:#4A5568;text-align:center">
      This report was generated automatically by AegisGuard v3.0.<br>
      All recommendations require human approval before execution per firm SOP.<br>
      {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
    </div>
  </div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Core send function
# ─────────────────────────────────────────────────────────────────────────────

def send_report(
    smtp_host:    str,
    smtp_port:    int,
    smtp_user:    str,
    smtp_pass:    str,
    use_tls:      bool,
    from_addr:    str,
    to_addrs:     list[str],
    cc_addrs:     list[str],
    subject:      str,
    html_body:    str,
    pdf_bytes:    bytes | None = None,
    pdf_filename: str = "AegisGuard_Report.pdf",
) -> SendResult:
    """
    Send an HTML email (with optional PDF attachment) via SMTP.

    Parameters
    ----------
    smtp_host   : SMTP server hostname
    smtp_port   : SMTP port (usually 587 for TLS)
    smtp_user   : SMTP login username (usually your email)
    smtp_pass   : SMTP password / app password
    use_tls     : True to use STARTTLS (recommended)
    from_addr   : Sender email address shown in From:
    to_addrs    : List of primary recipient addresses
    cc_addrs    : List of CC addresses (can be empty)
    subject     : Email subject line
    html_body   : Full HTML content of the email
    pdf_bytes   : Raw PDF bytes to attach (None = no attachment)
    pdf_filename: Name for the PDF attachment file
    """
    if not smtp_host or not smtp_user or not smtp_pass:
        return SendResult(False, "SMTP host, username, and password are required.")

    all_recipients = [a.strip() for a in to_addrs + cc_addrs if a.strip()]
    if not all_recipients:
        return SendResult(False, "At least one recipient email address is required.")

    # Build MIME message
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"]    = f"AegisGuard Risk Platform <{from_addr}>"
    msg["To"]      = ", ".join(to_addrs)
    if cc_addrs:
        msg["Cc"]  = ", ".join(cc_addrs)

    # HTML part
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # PDF attachment
    if pdf_bytes:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{pdf_filename}"')
        msg.attach(part)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            if use_tls:
                server.starttls(context=context)
            server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, all_recipients, msg.as_string())

        return SendResult(
            success=True,
            message=f"Email sent successfully to {len(all_recipients)} recipient(s).",
            recipients=all_recipients,
        )

    except smtplib.SMTPAuthenticationError:
        return SendResult(False,
            "Authentication failed. Check your email/password. "
            "For Gmail, use an App Password (myaccount.google.com → Security → App Passwords).")
    except smtplib.SMTPConnectError:
        return SendResult(False, f"Cannot connect to {smtp_host}:{smtp_port}. Check host and port.")
    except smtplib.SMTPRecipientsRefused as e:
        return SendResult(False, f"Recipients refused by server: {e}")
    except Exception as exc:
        return SendResult(False, f"Email send failed: {exc}")
