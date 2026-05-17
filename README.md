# 🛡️ AegisGuard v3.0 — Global Institutional Portfolio Risk Platform

> Full-stack SaaS risk dashboard: AI anomaly detection, MVO optimisation,
> 500+ global assets, NLP sentiment, scenario modeling, OFAC screening,
> GRC/compliance mapping, and a Claude-powered natural-language analyst.

---

## Quick Start
Live demo; https://awjxeianz3wrm2fpdyryzt.streamlit.app/
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml — set password and API keys
streamlit run app.py
```

Default password: **Kayode**

---

## Architecture

```
aegisguard/
├── app.py                         ← Streamlit entry point (10-tab dashboard)
├── requirements.txt
├── .streamlit/
│   ├── config.toml                ← Bloomberg dark theme
│   ├── secrets.toml               ← Auth + API keys (DO NOT COMMIT)
│   └── secrets.toml.example
│
├── data/
│   ├── loader.py                  ← API-agnostic: yfinance → AlphaVantage → Polygon
│   ├── universe.py                ← 500+ global instruments (US/EU/Asia/EM/ETF/Crypto)
│   └── alt_data.py                ← News NLP sentiment + Geopolitical Risk Index
│
├── logic/
│   ├── filters.py                 ← Liquidity Gate · Vol Cap · Correlation Matrix
│   ├── optimizer.py               ← MVO Max-Sharpe (CVXPY + SciPy fallback)
│   ├── regime.py                  ← SMA-200 trend · VIX regime · ATR trailing stops
│   ├── ai_analytics.py            ← Anomaly detection · Predictive risk scores · AI prescriptions
│   ├── compliance.py              ← NIST/SOC2/MiFID II mapping · Bias audit · OFAC · SAR
│   ├── scenarios.py               ← What-if stress tests · Escalation engine · Email drafts
│   └── nlq.py                     ← Natural-language query interface (Claude-powered)
│
├── ui/
│   └── components.py              ← 7 Plotly charts · KPI cards · Bloomberg badges
│
└── docs/
    ├── generate_sop.py            ← SOP PDF generator
    ├── report_generator.py        ← Scan results PDF (download button)
    └── AegisGuard_SOP.pdf         ← Pre-built Standard Operating Procedure
```

---

## 10 Dashboard Tabs

| Tab | What It Shows |
|---|---|
| 📊 Overview | KPI summary, zone table, cumulative return chart |
| 🔍 Risk Filters | Liquidity · Volatility · Correlation results + heatmap |
| ⚖️ Optimization | MVO weights, Sharpe, efficient frontier |
| 🌐 Market Regime | SMA-200, VIX level, composite market stance |
| 🚪 Exit Plan | ATR trailing stops, CRITICAL EXIT alerts |
| 🤖 AI Insights | Predictive risk scores, anomalies, AI prescriptions |
| 📰 Alt Data | News NLP sentiment, Geopolitical Risk Index |
| 📉 Scenarios | 5 historical stress tests + custom what-if builder |
| 🛡️ Compliance | GRC mapping, bias audit, OFAC screening, SAR flags, escalation queue |
| 💬 AI Analyst | Natural-language chat with your portfolio data |

---

## Global Asset Universe (500+ instruments)

| Region | Examples |
|---|---|
| US Equities | AAPL, MSFT, NVDA, JPM, XOM, LLY … |
| UK / Europe | SHEL.L, AZN.L, ASML.AS, SAP.DE, LVMH.PA … |
| Asia Pacific | 7203.T, 0700.HK, 005930.KS, TCS.NS, BHP.AX … |
| Canada / LatAm | SHOP.TO, RY.TO, VALE3.SA, PETR4.SA … |
| Africa / ME | NPN.JO, FSR.JO, SOL.JO … |
| ETFs | SPY, QQQ, EEM, GLD, TLT, IBIT, XLK … |
| Indices | ^GSPC, ^FTSE, ^N225, ^HSI, ^VIX … |

Any yfinance-supported ticker works — just type it in the sidebar.

---

## API Keys (Optional)

| Key | Purpose | Where to get |
|---|---|---|
| `alpha_vantage` | Backup data source | alphavantage.co |
| `polygon` | Tier-2 data source | polygon.io |
| `gnews` | Live news headlines | gnews.io |

Without keys, the system runs fully on Yahoo Finance (free).

---

## Deployment

**Streamlit Cloud:**
1. Push to GitHub (ensure `secrets.toml` is in `.gitignore`)
2. share.streamlit.io → New app → `app.py`
3. Paste secrets in Advanced Settings

**Docker / Azure / AWS:**
```bash
streamlit run app.py --server.port 8080 --server.headless true
```

---

## Regenerate SOP PDF

```bash
python docs/generate_sop.py
```

---

## License
Proprietary. All rights reserved. Licensed institutional use only.
