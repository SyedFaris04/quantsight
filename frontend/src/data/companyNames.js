/**
 * frontend/src/data/companyNames.js
 * Static ticker → full name reference data for the 44 tickers this
 * project actually tracks (confirmed against GET /tickers). Public,
 * unchanging factual names — not a computed/fabricated metric, so this
 * is safe reference data rather than mock data.
 */

export const COMPANY_NAMES = {
  AAPL  : "Apple Inc.",
  ABBV  : "AbbVie Inc.",
  ADBE  : "Adobe Inc.",
  AMD   : "Advanced Micro Devices",
  AMZN  : "Amazon.com Inc.",
  BAC   : "Bank of America Corp.",
  C     : "Citigroup Inc.",
  COP   : "ConocoPhillips",
  COST  : "Costco Wholesale Corp.",
  CRM   : "Salesforce Inc.",
  CVS   : "CVS Health Corp.",
  CVX   : "Chevron Corp.",
  DIA   : "SPDR Dow Jones Industrial Average ETF",
  GLD   : "SPDR Gold Shares",
  GOOGL : "Alphabet Inc.",
  GS    : "Goldman Sachs Group Inc.",
  INTC  : "Intel Corp.",
  IWM   : "iShares Russell 2000 ETF",
  JNJ   : "Johnson & Johnson",
  JPM   : "JPMorgan Chase & Co.",
  MCD   : "McDonald's Corp.",
  META  : "Meta Platforms Inc.",
  MRNA  : "Moderna Inc.",
  MS    : "Morgan Stanley",
  MSFT  : "Microsoft Corp.",
  NFLX  : "Netflix Inc.",
  NKE   : "Nike Inc.",
  NVDA  : "NVIDIA Corp.",
  ORCL  : "Oracle Corp.",
  OXY   : "Occidental Petroleum Corp.",
  PFE   : "Pfizer Inc.",
  PYPL  : "PayPal Holdings Inc.",
  QQQ   : "Invesco QQQ Trust",
  SLB   : "Schlumberger Ltd.",
  SNAP  : "Snap Inc.",
  SPY   : "SPDR S&P 500 ETF Trust",
  TGT   : "Target Corp.",
  TLT   : "iShares 20+ Year Treasury Bond ETF",
  TSLA  : "Tesla Inc.",
  UBER  : "Uber Technologies Inc.",
  UNH   : "UnitedHealth Group Inc.",
  WFC   : "Wells Fargo & Co.",
  WMT   : "Walmart Inc.",
  XOM   : "Exxon Mobil Corp.",
};

export function companyName(ticker) {
  return COMPANY_NAMES[ticker?.toUpperCase()] || null;
}
