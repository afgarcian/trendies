from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cachetools import TTLCache
import time
import logging
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Caches
ohlcv_cache = TTLCache(maxsize=300, ttl=300)     # 5m OHLCV cache
exchange_cache = TTLCache(maxsize=50, ttl=3600)  # 1h exchange instances

last_update_time = None
FORCE_UPDATE_AFTER = timedelta(hours=24)

STATIC_DIR = Path("static")
STATIC_FILE = STATIC_DIR / "index.html"
STATIC_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)

# ---- Utility: get/cached exchange instance ----------------------------------

def get_exchange_instance(exchange_id: str):
    """Return a cached ccxt exchange instance with markets loaded."""
    def _create():
        ex = getattr(ccxt, exchange_id)()
        # Some exchanges require enableRateLimit to be polite
        ex.enableRateLimit = True
        ex.load_markets()
        return ex
    return get_cached_data(f"exchange::{exchange_id}", _create, cache_store=exchange_cache)

# ---- Utility: cached data wrapper -------------------------------------------

def get_cached_data(key, fetch_func, cache_store=ohlcv_cache):
    if key not in cache_store:
        try:
            cache_store[key] = fetch_func()
            time.sleep(0.2)  # gentle rate limiting
        except Exception as e:
            logging.exception(f"Error fetching data for {key}")
            return {"error": str(e)}
    return cache_store[key]

# ---- Math helpers ------------------------------------------------------------

def calculate_ema(data, periods):
    s = pd.Series(data)
    if len(s) < periods:
        return float('nan')
    return float(s.ewm(span=periods, adjust=False).mean().iloc[-1])

def calculate_performance(closes, days):
    try:
        if len(closes) > days:
            current_price = closes[-1]
            past_price = closes[-days-1]
            if past_price:
                return ((current_price - past_price) / past_price) * 100
    except Exception:
        pass
    return None

# ---- Routing preferences (extendable) ---------------------------------------

# Order matters; we’ll try each exchange until we find a market.
TOKEN_EXCHANGE_PREFERENCES = {
    # Blue chips often on Coinbase
    "BTC": ["coinbase", "binance", "kraken"],
    "ETH": ["coinbase", "binance", "kraken"],
    "SOL": ["coinbase", "binance", "kraken"],
    "LINK": ["coinbase", "binance", "kraken"],
    "ADA": ["coinbase", "binance", "kraken"],
    "LDO": ["coinbase", "binance", "kraken"],
    "ARB": ["coinbase", "binance", "kraken"],
    "WLD": ["coinbase", "binance", "kraken"],

    # Commonly on Binance/Bybit/Gate/MEXC/KuCoin
    "ENA": ["coinbase", "binance", "bybit", "gate", "mexc", "kucoin", "bitget"],
    "RAY": ["binance", "bybit", "gate", "mexc", "kucoin", "bitget"],
    "MNT": ["binance", "bybit", "gate", "mexc", "kucoin", "bitget"],
    "AERO": ["bybit", "gate", "mexc", "bitget", "kucoin"],
    "HYPE": ["gate", "mexc", "bybit", "bitget", "kucoin"],
    "PENDLE": ["coinbase", "binance", "bybit", "gate", "mexc", "kucoin", "bitget"],
    "PUMP": ["gate", "mexc", "bybit", "bitget", "kucoin"],
    "DRIFT": ["bybit", "gate", "mexc", "bitget", "kucoin"],
    "EDGE": ["gate", "mexc", "bybit", "bitget", "kucoin"],
    "CARDS": ["gate", "mexc", "bybit", "bitget", "kucoin"],

    # You already had these special ones
    "NATIX": ["kucoin", "gate", "mexc"],
    "VANA": ["mexc", "gate", "bitget"],

    # Non-crypto / likely unsupported as spot via ccxt
    "SPX": ["coinbase", "binance", "kraken"],  # will fail gracefully unless venue lists a proxy
}

DEFAULT_EXCHANGE_ORDER = ["coinbase", "binance", "kraken", "bybit", "bitget", "gate", "mexc", "kucoin"]

# ---- Market symbol resolver --------------------------------------------------

def find_market_symbol(exchange, base_symbol, preferred_quotes=("USD","USDT")):
    """
    Try to find a market like BASE/USD or BASE/USDT across various separators.
    Scans exchange.symbols for robustness.
    Returns the ccxt symbol string or None.
    """
    symbols = getattr(exchange, "symbols", []) or []
    # quick exact matches first
    candidates = []
    for q in preferred_quotes:
        candidates += [
            f"{base_symbol}/{q}",
            f"{base_symbol}-{q}",    # kucoin style
            f"{base_symbol}_{q}",    # gate style
        ]
    for c in candidates:
        if c in symbols:
            return c
    # fallback: scan by parsing
    for sym in symbols:
        try:
            base, quote = sym.replace("-", "/").replace("_", "/").split("/")
            if base == base_symbol and quote in preferred_quotes:
                return sym
        except Exception:
            continue
    return None

def resolve_exchange_and_symbol(base_symbol, preferred_exchange: str | None = None):
    """
    Decide which exchange to use for a token and resolve the correct trading pair.
    Returns (exchange, symbol_pair). May raise on complete failure.
    """
    # Choose routing list
    route = []
    if preferred_exchange:
        route = [preferred_exchange]
    elif base_symbol in TOKEN_EXCHANGE_PREFERENCES:
        route = TOKEN_EXCHANGE_PREFERENCES[base_symbol]
    else:
        route = DEFAULT_EXCHANGE_ORDER

    last_err = None
    for ex_id in route:
        try:
            ex = get_exchange_instance(ex_id)
            # Coinbase prefers USD pairs; others USDT or USD both OK
            quotes = ("USD","USDT") if ex.id == "coinbase" else ("USDT","USD")
            sym = find_market_symbol(ex, base_symbol, quotes)
            if sym:
                return ex, sym
        except Exception as e:
            last_err = e
            continue
    raise Exception(f"No market found for {base_symbol} on route {route}. Last error: {last_err}")

# ---- BTC price (USD) series for relative calc -------------------------------

def get_btc_usd_closes(limit=50):
    def fetch():
        ex = get_exchange_instance("coinbase")
        ohlcv = ex.fetch_ohlcv("BTC/USD", "1d", limit=limit)
        return [x[4] for x in ohlcv]
    return get_cached_data("btc_usd_closes", fetch)

# ---- Core trend analysis -----------------------------------------------------

def get_trend_analysis(base_symbol, quote_symbol="USD", chain=None, preferred_exchange=None):
    cache_key = f"{base_symbol}_{quote_symbol}"
    def fetch_analysis():
        try:
            exchange, token_market = resolve_exchange_and_symbol(base_symbol, preferred_exchange)
        except Exception as e:
            return {"symbol": f"{base_symbol}/{quote_symbol}", "error": str(e), "chain": chain}

        try:
            # Pull token OHLCV (use the resolved market)
            token_ohlcv = exchange.fetch_ohlcv(token_market, '1d', limit=50)
            if not token_ohlcv:
                return {
                    "symbol": token_market,
                    "error": f"No data available on {exchange.id}",
                    "chain": chain,
                    "exchange": exchange.id
                }

            token_closes = [x[4] for x in token_ohlcv]

            if quote_symbol == "BTC":
                # Build USD-relative ratio using BTC/USD from Coinbase
                btc_usd = get_btc_usd_closes(limit=len(token_closes))
                n = min(len(token_closes), len(btc_usd))
                if n == 0:
                    return {
                        "symbol": token_market,
                        "error": "Insufficient BTC/USD data for ratio",
                        "chain": chain,
                        "exchange": exchange.id
                    }
                closes = [token_closes[i] / btc_usd[i] for i in range(n)]
            else:
                closes = token_closes

            ema8 = calculate_ema(closes, 8)
            ema20 = calculate_ema(closes, 20)
            current_price = closes[-1] if closes else float('nan')

            perf_7d = calculate_performance(closes, 7)
            perf_14d = calculate_performance(closes, 14)

            return {
                "symbol": token_market if quote_symbol != "BTC" else f"{base_symbol}/BTC (via USD ratio)",
                "current_price": current_price,
                "ema8": ema8,
                "ema20": ema20,
                "is_uptrend": (ema8 > ema20) if (np.isfinite(ema8) and np.isfinite(ema20)) else None,
                "trend_text": "Uptrend" if (np.isfinite(ema8) and np.isfinite(ema20) and ema8 > ema20) else "Downtrend",
                "quote_currency": quote_symbol,
                "chain": chain,
                "exchange": exchange.id,
                "is_calculated": (quote_symbol == "BTC"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "perf_7d": perf_7d,
                "perf_14d": perf_14d
            }
        except Exception as e:
            return {
                "symbol": f"{base_symbol}/{quote_symbol}",
                "error": f"Error on {exchange.id if 'exchange' in locals() else 'resolver'}: {str(e)}",
                "chain": chain,
                "exchange": getattr(exchange, 'id', None)
            }
    return get_cached_data(cache_key, fetch_analysis)

# ---- HTTP endpoints ----------------------------------------------------------

@app.get("/update")
async def update_data(request: Request):
    try:
        logging.info("Starting data update...")
        ohlcv_cache.clear()
        exchange_cache.clear()
        logging.info("Caches cleared")

        # ---- YOUR NEW ASSET LIST ------------------------------------------------
        # Single list used as “portfolio”; feel free to split into watchlist if needed.
        symbols = [
            "BTC","ETH","SOL","ENA","DRIFT","PENDLE","LINK","PUMP","RAY","LDO",
            "SPX","ADA","HYPE","AERO","NATIX","VANA","ARB","MNT","WLD","EDGE","CARDS"
        ]
        portfolio_assets = [{"symbol": s, "chain": None} for s in symbols]

        portfolio_analysis = []
        for asset in portfolio_assets:
            sym = asset["symbol"]
            try:
                if sym == "BTC":
                    usdt_analysis = get_trend_analysis(sym, "USD", asset["chain"])
                    portfolio_analysis.append({
                        "asset": sym,
                        "chain": asset["chain"],
                        "usdt": usdt_analysis,
                        "btc": {"symbol": "BTC/BTC", "error": "Same asset"}
                    })
                else:
                    usdt_analysis = get_trend_analysis(sym, "USD", asset["chain"])
                    time.sleep(0.2)
                    btc_analysis = get_trend_analysis(sym, "BTC", asset["chain"])
                    portfolio_analysis.append({
                        "asset": sym,
                        "chain": asset["chain"],
                        "usdt": usdt_analysis,
                        "btc": btc_analysis
                    })
            except Exception as e:
                logging.exception(f"Error processing {sym}")
                portfolio_analysis.append({
                    "asset": sym,
                    "chain": asset["chain"],
                    "usdt": {"error": f"Failed: {str(e)}"},
                    "btc": {"error": f"Failed: {str(e)}"}
                })

        # If you still want a watchlist section, keep it; otherwise pass empty.
        watchlist_analysis = []

        # ---- Render HTML correctly (no .body.decode()) -------------------------
        html = templates.get_template("index.html").render({
            "request": request,
            "portfolio_analysis": portfolio_analysis,
            "watchlist_analysis": watchlist_analysis,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        STATIC_FILE.write_text(html, encoding="utf-8")
        logging.info("Data update completed successfully")
        return {"status": "success", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logging.exception("Error updating data")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    refresh = request.query_params.get('refresh', '').lower() == 'true'
    if refresh or not STATIC_FILE.exists():
        logging.info("Generating new data...")
        await update_data(request)

    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }
    return HTMLResponse(content=STATIC_FILE.read_text(encoding="utf-8"), headers=headers)
