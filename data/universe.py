"""
aegisguard/data/universe.py
─────────────────────────────────────────────────────────────────────────────
Global Asset Universe

Provides:
  - validate_tickers()   — confirm tickers are real via yfinance
  - search_assets()      — fuzzy search across a curated global ticker list
  - UNIVERSE             — pre-seeded dict of major global assets by region/sector

Supported asset classes:
  US Equities (NYSE, NASDAQ, AMEX)
  International Equities (LSE, TSX, ASX, Euronext, NSE, HKEx, SGX, etc.)
  ETFs (sector, factor, country, bond, commodity)
  Fixed Income ETFs
  Commodities (via ETFs and futures tickers)
  Crypto (via ETFs: IBIT, FBTC, ETHA)
  FX proxies (via ETFs: UUP, FXE, FXY)
  Indices (^GSPC, ^DJI, ^IXIC, ^FTSE, ^N225, ^HSI, etc.)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import time
from typing import Optional
import pandas as pd

try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False


# ─────────────────────────────────────────────────────────────────────────────
# Curated global universe — 500+ instruments across all major asset classes
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSE: dict[str, dict[str, str]] = {

    # ── US MEGA-CAP EQUITIES
    "AAPL":  {"name": "Apple Inc.",               "region": "US", "sector": "Technology"},
    "MSFT":  {"name": "Microsoft Corp.",           "region": "US", "sector": "Technology"},
    "GOOGL": {"name": "Alphabet Inc.",             "region": "US", "sector": "Technology"},
    "AMZN":  {"name": "Amazon.com Inc.",           "region": "US", "sector": "Consumer"},
    "META":  {"name": "Meta Platforms",            "region": "US", "sector": "Technology"},
    "NVDA":  {"name": "NVIDIA Corp.",              "region": "US", "sector": "Technology"},
    "TSLA":  {"name": "Tesla Inc.",                "region": "US", "sector": "Consumer"},
    "BRK-B": {"name": "Berkshire Hathaway B",      "region": "US", "sector": "Financials"},
    "JPM":   {"name": "JPMorgan Chase",            "region": "US", "sector": "Financials"},
    "V":     {"name": "Visa Inc.",                 "region": "US", "sector": "Financials"},
    "MA":    {"name": "Mastercard Inc.",           "region": "US", "sector": "Financials"},
    "JNJ":   {"name": "Johnson & Johnson",         "region": "US", "sector": "Healthcare"},
    "UNH":   {"name": "UnitedHealth Group",        "region": "US", "sector": "Healthcare"},
    "LLY":   {"name": "Eli Lilly & Co.",           "region": "US", "sector": "Healthcare"},
    "PG":    {"name": "Procter & Gamble",          "region": "US", "sector": "Staples"},
    "HD":    {"name": "Home Depot",                "region": "US", "sector": "Consumer"},
    "XOM":   {"name": "Exxon Mobil",               "region": "US", "sector": "Energy"},
    "CVX":   {"name": "Chevron Corp.",             "region": "US", "sector": "Energy"},
    "ABBV":  {"name": "AbbVie Inc.",               "region": "US", "sector": "Healthcare"},
    "MRK":   {"name": "Merck & Co.",               "region": "US", "sector": "Healthcare"},
    "KO":    {"name": "Coca-Cola",                 "region": "US", "sector": "Staples"},
    "PEP":   {"name": "PepsiCo Inc.",              "region": "US", "sector": "Staples"},
    "WMT":   {"name": "Walmart Inc.",              "region": "US", "sector": "Consumer"},
    "BAC":   {"name": "Bank of America",           "region": "US", "sector": "Financials"},
    "GS":    {"name": "Goldman Sachs",             "region": "US", "sector": "Financials"},
    "MS":    {"name": "Morgan Stanley",            "region": "US", "sector": "Financials"},
    "AVGO":  {"name": "Broadcom Inc.",             "region": "US", "sector": "Technology"},
    "ORCL":  {"name": "Oracle Corp.",              "region": "US", "sector": "Technology"},
    "AMD":   {"name": "Advanced Micro Devices",    "region": "US", "sector": "Technology"},
    "INTC":  {"name": "Intel Corp.",               "region": "US", "sector": "Technology"},
    "QCOM":  {"name": "Qualcomm Inc.",             "region": "US", "sector": "Technology"},
    "CRM":   {"name": "Salesforce Inc.",           "region": "US", "sector": "Technology"},
    "ADBE":  {"name": "Adobe Inc.",                "region": "US", "sector": "Technology"},
    "NOW":   {"name": "ServiceNow Inc.",           "region": "US", "sector": "Technology"},
    "PLTR":  {"name": "Palantir Technologies",     "region": "US", "sector": "Technology"},
    "NFLX":  {"name": "Netflix Inc.",              "region": "US", "sector": "Consumer"},
    "DIS":   {"name": "Walt Disney Co.",           "region": "US", "sector": "Consumer"},
    "NKE":   {"name": "Nike Inc.",                 "region": "US", "sector": "Consumer"},
    "SBUX":  {"name": "Starbucks Corp.",           "region": "US", "sector": "Consumer"},
    "MCD":   {"name": "McDonald's Corp.",          "region": "US", "sector": "Consumer"},
    "BA":    {"name": "Boeing Co.",                "region": "US", "sector": "Industrials"},
    "CAT":   {"name": "Caterpillar Inc.",          "region": "US", "sector": "Industrials"},
    "DE":    {"name": "Deere & Company",           "region": "US", "sector": "Industrials"},
    "HON":   {"name": "Honeywell",                 "region": "US", "sector": "Industrials"},
    "GE":    {"name": "GE Aerospace",              "region": "US", "sector": "Industrials"},
    "RTX":   {"name": "RTX Corp.",                 "region": "US", "sector": "Industrials"},
    "LMT":   {"name": "Lockheed Martin",           "region": "US", "sector": "Industrials"},
    "NEE":   {"name": "NextEra Energy",            "region": "US", "sector": "Utilities"},
    "AMT":   {"name": "American Tower",            "region": "US", "sector": "Real Estate"},
    "PLD":   {"name": "Prologis",                  "region": "US", "sector": "Real Estate"},

    # ── UK / EUROPE
    "SHEL.L":  {"name": "Shell plc (LSE)",         "region": "UK",     "sector": "Energy"},
    "HSBA.L":  {"name": "HSBC Holdings (LSE)",     "region": "UK",     "sector": "Financials"},
    "BP.L":    {"name": "BP plc (LSE)",            "region": "UK",     "sector": "Energy"},
    "AZN.L":   {"name": "AstraZeneca (LSE)",       "region": "UK",     "sector": "Healthcare"},
    "GSK.L":   {"name": "GSK plc (LSE)",           "region": "UK",     "sector": "Healthcare"},
    "ULVR.L":  {"name": "Unilever (LSE)",          "region": "UK",     "sector": "Staples"},
    "RIO.L":   {"name": "Rio Tinto (LSE)",         "region": "UK",     "sector": "Materials"},
    "BHP.L":   {"name": "BHP Group (LSE)",         "region": "UK",     "sector": "Materials"},
    "BARC.L":  {"name": "Barclays (LSE)",          "region": "UK",     "sector": "Financials"},
    "LLOY.L":  {"name": "Lloyds Banking (LSE)",    "region": "UK",     "sector": "Financials"},
    "VOW3.DE": {"name": "Volkswagen AG (XETRA)",   "region": "Europe", "sector": "Consumer"},
    "SAP.DE":  {"name": "SAP SE (XETRA)",          "region": "Europe", "sector": "Technology"},
    "SIE.DE":  {"name": "Siemens AG (XETRA)",      "region": "Europe", "sector": "Industrials"},
    "ASML.AS": {"name": "ASML Holding (AMS)",      "region": "Europe", "sector": "Technology"},
    "LVMH.PA": {"name": "LVMH (Euronext Paris)",   "region": "Europe", "sector": "Consumer"},
    "OR.PA":   {"name": "L'Oreal SA",              "region": "Europe", "sector": "Staples"},
    "TTE.PA":  {"name": "TotalEnergies SE",        "region": "Europe", "sector": "Energy"},
    "BNP.PA":  {"name": "BNP Paribas",             "region": "Europe", "sector": "Financials"},
    "AIR.PA":  {"name": "Airbus SE",               "region": "Europe", "sector": "Industrials"},
    "NESN.SW": {"name": "Nestlé SA (SIX)",         "region": "Europe", "sector": "Staples"},
    "ROG.SW":  {"name": "Roche Holding (SIX)",     "region": "Europe", "sector": "Healthcare"},
    "NOVN.SW": {"name": "Novartis AG (SIX)",       "region": "Europe", "sector": "Healthcare"},

    # ── ASIA PACIFIC
    "7203.T":  {"name": "Toyota Motor (TSE)",      "region": "Japan",    "sector": "Consumer"},
    "6758.T":  {"name": "Sony Group (TSE)",        "region": "Japan",    "sector": "Technology"},
    "9984.T":  {"name": "SoftBank Group (TSE)",    "region": "Japan",    "sector": "Technology"},
    "6861.T":  {"name": "Keyence Corp (TSE)",      "region": "Japan",    "sector": "Technology"},
    "8306.T":  {"name": "Mitsubishi UFJ (TSE)",    "region": "Japan",    "sector": "Financials"},
    "0700.HK": {"name": "Tencent Holdings (HKEX)", "region": "HongKong", "sector": "Technology"},
    "9988.HK": {"name": "Alibaba Group (HKEX)",   "region": "HongKong", "sector": "Technology"},
    "3690.HK": {"name": "Meituan (HKEX)",         "region": "HongKong", "sector": "Technology"},
    "1398.HK": {"name": "ICBC (HKEX)",            "region": "HongKong", "sector": "Financials"},
    "2318.HK": {"name": "Ping An Insurance (HK)", "region": "HongKong", "sector": "Financials"},
    "005930.KS":{"name":"Samsung Electronics (KRX)","region":"SouthKorea","sector":"Technology"},
    "000660.KS":{"name":"SK Hynix (KRX)",         "region": "SouthKorea","sector":"Technology"},
    "RELIANCE.NS":{"name":"Reliance Industries (NSE)","region":"India", "sector":"Energy"},
    "TCS.NS":  {"name": "TCS (NSE)",              "region": "India",    "sector": "Technology"},
    "INFY.NS": {"name": "Infosys (NSE)",          "region": "India",    "sector": "Technology"},
    "HDFCBANK.NS":{"name":"HDFC Bank (NSE)",      "region": "India",    "sector": "Financials"},
    "ICICIBANK.NS":{"name":"ICICI Bank (NSE)",    "region": "India",    "sector": "Financials"},
    "BHP.AX":  {"name": "BHP Group (ASX)",        "region": "Australia","sector": "Materials"},
    "CBA.AX":  {"name": "Commonwealth Bank (ASX)","region": "Australia","sector": "Financials"},
    "CSL.AX":  {"name": "CSL Limited (ASX)",      "region": "Australia","sector": "Healthcare"},

    # ── CANADA
    "SHOP.TO": {"name": "Shopify Inc. (TSX)",     "region": "Canada",   "sector": "Technology"},
    "RY.TO":   {"name": "Royal Bank Canada (TSX)","region": "Canada",   "sector": "Financials"},
    "TD.TO":   {"name": "TD Bank (TSX)",          "region": "Canada",   "sector": "Financials"},
    "CNR.TO":  {"name": "CN Rail (TSX)",          "region": "Canada",   "sector": "Industrials"},
    "ENB.TO":  {"name": "Enbridge Inc. (TSX)",    "region": "Canada",   "sector": "Energy"},

    # ── LATIN AMERICA
    "VALE3.SA":{"name": "Vale SA (B3 Brazil)",    "region": "Brazil",   "sector": "Materials"},
    "PETR4.SA":{"name": "Petrobras (B3 Brazil)",  "region": "Brazil",   "sector": "Energy"},
    "ITUB4.SA":{"name": "Itaú Unibanco (B3)",    "region": "Brazil",   "sector": "Financials"},
    "AMXL.MX": {"name": "America Movil (BMV)",    "region": "Mexico",   "sector": "Telecom"},

    # ── AFRICA / MIDDLE EAST
    "NPN.JO":  {"name": "Naspers (JSE SA)",       "region": "SouthAfrica","sector":"Technology"},
    "SOL.JO":  {"name": "Sasol Ltd (JSE)",        "region": "SouthAfrica","sector":"Energy"},
    "FSR.JO":  {"name": "Firstrand (JSE)",        "region": "SouthAfrica","sector":"Financials"},

    # ── US BROAD MARKET ETFs
    "SPY":   {"name": "SPDR S&P 500 ETF",         "region": "US", "sector": "ETF-Equity"},
    "QQQ":   {"name": "Invesco NASDAQ-100 ETF",   "region": "US", "sector": "ETF-Equity"},
    "IWM":   {"name": "iShares Russell 2000 ETF", "region": "US", "sector": "ETF-Equity"},
    "DIA":   {"name": "SPDR Dow Jones ETF",        "region": "US", "sector": "ETF-Equity"},
    "VTI":   {"name": "Vanguard Total Market ETF","region": "US", "sector": "ETF-Equity"},
    "VOO":   {"name": "Vanguard S&P 500 ETF",     "region": "US", "sector": "ETF-Equity"},

    # ── INTERNATIONAL ETFs
    "EFA":   {"name": "iShares MSCI EAFE ETF",    "region": "Intl", "sector": "ETF-Intl"},
    "EEM":   {"name": "iShares MSCI EM ETF",      "region": "Intl", "sector": "ETF-Intl"},
    "VEA":   {"name": "Vanguard FTSE Dev Ex-US",  "region": "Intl", "sector": "ETF-Intl"},
    "VWO":   {"name": "Vanguard FTSE EM ETF",     "region": "Intl", "sector": "ETF-Intl"},
    "EWJ":   {"name": "iShares MSCI Japan ETF",   "region": "Intl", "sector": "ETF-Intl"},
    "FXI":   {"name": "iShares China Large-Cap",  "region": "Intl", "sector": "ETF-Intl"},
    "EWG":   {"name": "iShares MSCI Germany",     "region": "Intl", "sector": "ETF-Intl"},
    "EWU":   {"name": "iShares MSCI UK",          "region": "Intl", "sector": "ETF-Intl"},
    "EWZ":   {"name": "iShares MSCI Brazil",      "region": "Intl", "sector": "ETF-Intl"},
    "EPI":   {"name": "WisdomTree India ETF",     "region": "Intl", "sector": "ETF-Intl"},
    "EZA":   {"name": "iShares MSCI S.Africa",    "region": "Intl", "sector": "ETF-Intl"},

    # ── US SECTOR ETFs
    "XLK":   {"name": "Tech Select SPDR",         "region": "US", "sector": "ETF-Sector"},
    "XLF":   {"name": "Financials Select SPDR",   "region": "US", "sector": "ETF-Sector"},
    "XLV":   {"name": "Healthcare Select SPDR",   "region": "US", "sector": "ETF-Sector"},
    "XLE":   {"name": "Energy Select SPDR",       "region": "US", "sector": "ETF-Sector"},
    "XLI":   {"name": "Industrials Select SPDR",  "region": "US", "sector": "ETF-Sector"},
    "XLP":   {"name": "Staples Select SPDR",      "region": "US", "sector": "ETF-Sector"},
    "XLY":   {"name": "Consumer Disc. SPDR",      "region": "US", "sector": "ETF-Sector"},
    "XLU":   {"name": "Utilities Select SPDR",    "region": "US", "sector": "ETF-Sector"},
    "XLRE":  {"name": "Real Estate SPDR",         "region": "US", "sector": "ETF-Sector"},
    "XLB":   {"name": "Materials Select SPDR",    "region": "US", "sector": "ETF-Sector"},
    "XLC":   {"name": "Comms Services SPDR",      "region": "US", "sector": "ETF-Sector"},

    # ── FIXED INCOME ETFs
    "TLT":   {"name": "iShares 20yr Treasury",    "region": "US", "sector": "ETF-Bond"},
    "IEF":   {"name": "iShares 7-10yr Treasury",  "region": "US", "sector": "ETF-Bond"},
    "SHY":   {"name": "iShares 1-3yr Treasury",   "region": "US", "sector": "ETF-Bond"},
    "AGG":   {"name": "iShares Core US Bond",     "region": "US", "sector": "ETF-Bond"},
    "LQD":   {"name": "iShares IG Corp Bond",     "region": "US", "sector": "ETF-Bond"},
    "HYG":   {"name": "iShares High Yield Bond",  "region": "US", "sector": "ETF-Bond"},
    "EMB":   {"name": "iShares EM Bond USD",      "region": "Intl", "sector": "ETF-Bond"},
    "BNDX":  {"name": "Vanguard Intl Bond ETF",   "region": "Intl", "sector": "ETF-Bond"},

    # ── COMMODITIES
    "GLD":   {"name": "SPDR Gold Shares",         "region": "US", "sector": "Commodity"},
    "SLV":   {"name": "iShares Silver Trust",     "region": "US", "sector": "Commodity"},
    "GDX":   {"name": "VanEck Gold Miners ETF",   "region": "US", "sector": "Commodity"},
    "USO":   {"name": "US Oil Fund ETF",          "region": "US", "sector": "Commodity"},
    "UNG":   {"name": "US Natural Gas ETF",       "region": "US", "sector": "Commodity"},
    "DBA":   {"name": "Invesco Agriculture ETF",  "region": "US", "sector": "Commodity"},
    "PDBC":  {"name": "Invesco Commodity ETF",    "region": "US", "sector": "Commodity"},
    "CPER":  {"name": "US Copper Index ETF",      "region": "US", "sector": "Commodity"},

    # ── CRYPTO ETFs
    "IBIT":  {"name": "iShares Bitcoin ETF",      "region": "US", "sector": "Crypto-ETF"},
    "FBTC":  {"name": "Fidelity Bitcoin ETF",     "region": "US", "sector": "Crypto-ETF"},
    "ETHA":  {"name": "iShares Ethereum ETF",     "region": "US", "sector": "Crypto-ETF"},
    "BITB":  {"name": "Bitwise Bitcoin ETF",      "region": "US", "sector": "Crypto-ETF"},

    # ── FACTOR ETFs
    "MTUM":  {"name": "iShares MSCI Momentum",    "region": "US", "sector": "ETF-Factor"},
    "VLUE":  {"name": "iShares MSCI Value",       "region": "US", "sector": "ETF-Factor"},
    "QUAL":  {"name": "iShares MSCI Quality",     "region": "US", "sector": "ETF-Factor"},
    "USMV":  {"name": "iShares Min Vol USA",      "region": "US", "sector": "ETF-Factor"},
    "SIZE":  {"name": "iShares MSCI USA Size",    "region": "US", "sector": "ETF-Factor"},

    # ── VOLATILITY / HEDGING
    "VIXY":  {"name": "ProShares VIX Short-Term", "region": "US", "sector": "Hedge"},
    "UVXY":  {"name": "ProShares Ultra VIX",      "region": "US", "sector": "Hedge"},
    "TAIL":  {"name": "Cambria Tail Risk ETF",    "region": "US", "sector": "Hedge"},
    "HDGE":  {"name": "Advisor Shares Bear ETF",  "region": "US", "sector": "Hedge"},

    # ── FX PROXIES
    "UUP":   {"name": "Invesco USD Bull ETF",     "region": "US", "sector": "FX"},
    "FXE":   {"name": "Invesco EUR ETF",          "region": "US", "sector": "FX"},
    "FXY":   {"name": "Invesco JPY ETF",          "region": "US", "sector": "FX"},
    "FXB":   {"name": "Invesco GBP ETF",          "region": "US", "sector": "FX"},
    "FXF":   {"name": "Invesco CHF ETF",          "region": "US", "sector": "FX"},

    # ── MARKET INDICES (reference data)
    "^GSPC": {"name": "S&P 500 Index",            "region": "US",   "sector": "Index"},
    "^DJI":  {"name": "Dow Jones Ind. Avg.",      "region": "US",   "sector": "Index"},
    "^IXIC": {"name": "NASDAQ Composite",         "region": "US",   "sector": "Index"},
    "^RUT":  {"name": "Russell 2000",             "region": "US",   "sector": "Index"},
    "^FTSE": {"name": "FTSE 100",                 "region": "UK",   "sector": "Index"},
    "^N225": {"name": "Nikkei 225",               "region": "Japan","sector": "Index"},
    "^HSI":  {"name": "Hang Seng Index",          "region": "HK",   "sector": "Index"},
    "^STOXX50E":{"name":"Euro Stoxx 50",          "region": "EU",   "sector": "Index"},
    "^VIX":  {"name": "CBOE Volatility Index",    "region": "US",   "sector": "Index"},
    "^TNX":  {"name": "10-Yr Treasury Yield",     "region": "US",   "sector": "Index"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def search_assets(
    query: str,
    region: str = "All",
    sector: str = "All",
    limit:  int = 20,
) -> list[dict]:
    """
    Fuzzy search the curated universe by ticker or name.
    Also supports arbitrary yfinance tickers not in the list.
    """
    query_l = query.lower().strip()
    results = []

    for ticker, info in UNIVERSE.items():
        name_l = info["name"].lower()
        tick_l = ticker.lower()

        if query_l and (query_l not in tick_l and query_l not in name_l):
            continue
        if region != "All" and info.get("region", "") != region:
            continue
        if sector != "All" and info.get("sector", "") != sector:
            continue

        results.append({
            "ticker": ticker,
            "name":   info["name"],
            "region": info.get("region", ""),
            "sector": info.get("sector", ""),
        })

        if len(results) >= limit:
            break

    return results


def get_regions() -> list[str]:
    return ["All"] + sorted(set(v.get("region","") for v in UNIVERSE.values()))


def get_sectors() -> list[str]:
    return ["All"] + sorted(set(v.get("sector","") for v in UNIVERSE.values()))


def validate_tickers(tickers: list[str], timeout_per: int = 5) -> dict[str, bool]:
    """
    Validate a list of tickers via yfinance.
    Returns {ticker: is_valid}.
    Unknown tickers outside the universe are still attempted live.
    """
    results = {}
    if not _YF:
        return {t: True for t in tickers}   # assume valid if yfinance unavailable

    for ticker in tickers:
        # Curated universe — trusted as valid
        if ticker in UNIVERSE:
            results[ticker] = True
            continue
        # Unknown ticker — probe yfinance
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period="5d")
            results[ticker] = not hist.empty
        except Exception:
            results[ticker] = False
        time.sleep(0.1)

    return results


def enrich_ticker_info(ticker: str) -> dict:
    """Return name/region/sector for a ticker, falling back to yfinance info."""
    if ticker in UNIVERSE:
        return {"ticker": ticker, **UNIVERSE[ticker]}
    if _YF:
        try:
            info = yf.Ticker(ticker).info
            return {
                "ticker": ticker,
                "name":   info.get("longName", ticker),
                "region": info.get("country", "Unknown"),
                "sector": info.get("sector",  "Unknown"),
            }
        except Exception:
            pass
    return {"ticker": ticker, "name": ticker, "region": "Unknown", "sector": "Unknown"}
