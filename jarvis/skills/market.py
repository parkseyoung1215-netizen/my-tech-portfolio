"""주가·시장: yfinance → 야후 직접 → Stooq 순서로 시도"""
import datetime
import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request

from config import *
from jarvis.core import S, client, run_applescript, ssl_context, home_now, load_state, save_state, \
    press_media_key, get_volume, set_muted, is_muted
from jarvis.registry import register, execute, _tool, _s, _i
import http.cookiejar

YAHOO_HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
_yahoo_opener = None
STOOQ_MAP = {"^GSPC": "^spx", "^IXIC": "^ndq", "^DJI": "^dji", "^GDAXI": "^dax", "^KS11": "^kospi",
             "BTC-USD": "btcusd", "ETH-USD": "ethusd"}


def _yahoo():
    """야후는 쿠키 없는 요청을 막아서(429), 처음에 쿠키를 한 번 받아두고 계속 씀."""
    global _yahoo_opener
    if _yahoo_opener is None:
        jar = http.cookiejar.CookieJar()
        _yahoo_opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(jar),
            urllib.request.HTTPSHandler(context=ssl_context()))
        _yahoo_opener.addheaders = list(YAHOO_HEADERS.items())
        try:
            _yahoo_opener.open("https://fc.yahoo.com", timeout=10)
        except Exception:
            pass  # 404가 나도 쿠키는 받아짐
    return _yahoo_opener


def _get_json(url: str) -> dict:
    last = None
    for host in ("query1", "query2"):
        try:
            with _yahoo().open(url.replace("query1", host), timeout=10) as response:
                return json.loads(response.read())
        except Exception as e:
            last = e
    raise last


def stooq_symbol(symbol: str) -> str:
    sym = symbol.upper()
    if sym in STOOQ_MAP:
        return STOOQ_MAP[sym]
    if sym.endswith("=X"):                 # USDKRW=X -> usdkrw
        return sym[:-2].lower()
    if re.fullmatch(r"[A-Z.]{1,6}", sym):  # NVDA -> nvda.us
        return sym.lower().replace(".", "-") + ".us"
    return ""


def quote_stooq(symbol: str) -> str:
    code = stooq_symbol(symbol)
    if not code:
        raise ValueError("not available")
    request = urllib.request.Request(f"https://stooq.com/q/d/l/?s={urllib.parse.quote(code)}&i=d",
                                     headers=YAHOO_HEADERS)
    with urllib.request.urlopen(request, timeout=10, context=ssl_context()) as response:
        rows = response.read().decode().strip().splitlines()[1:]
    closes = [float(r.split(",")[4]) for r in rows[-23:] if r.count(",") >= 4]
    if not closes:
        raise ValueError("no data")
    price = closes[-1]

    def change(old):
        return f"{(price / old - 1) * 100:+.2f}%" if old else "n/a"

    last_date = rows[-1].split(",")[0]
    return (f"{symbol}: {price:,.2f} | 1d {change(closes[-2] if len(closes) > 1 else None)}, "
            f"5d {change(closes[-6] if len(closes) > 5 else None)}, 1mo {change(closes[0])} "
            f"| last close {last_date} (backup source)")


def find_ticker(name: str) -> str:
    """회사 이름 -> 티커 (예: 'samsung electronics' -> '005930.KS')."""
    data = _get_json("https://query1.finance.yahoo.com/v1/finance/search?quotesCount=3&newsCount=0&q="
                     + urllib.parse.quote(name))
    quotes = data.get("quotes") or []
    return quotes[0]["symbol"] if quotes else ""


def quote_one(symbol: str) -> str:
    """티커 하나의 가격과 변동률을 글자로."""
    data = _get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
                     "?range=1mo&interval=1d")
    result = (data.get("chart") or {}).get("result")
    if not result:
        raise ValueError("no data")
    meta = result[0]["meta"]
    closes = [c for c in result[0]["indicators"]["quote"][0].get("close", []) if c is not None]
    price = meta.get("regularMarketPrice") or (closes[-1] if closes else None)
    if price is None:
        raise ValueError("no price")

    def change(old):
        return f"{(price / old - 1) * 100:+.2f}%" if old else "n/a"

    day = change(closes[-2] if len(closes) >= 2 else meta.get("chartPreviousClose"))
    week = change(closes[-6] if len(closes) >= 6 else None)
    month = change(closes[0] if closes else None)
    name = meta.get("shortName") or meta.get("longName") or symbol
    when = datetime.datetime.fromtimestamp(meta.get("regularMarketTime", 0), datetime.timezone.utc)
    return (f"{name} ({symbol}): {price:,.2f} {meta.get('currency', '')} | 1d {day}, 5d {week}, 1mo {month} "
            f"| as of {when.strftime('%Y-%m-%d %H:%M')} UTC")


def quote_yfinance(symbol: str) -> str:
    """yfinance: 브라우저처럼 접속해서 야후의 로봇 차단을 피함 (pip install yfinance)."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1mo", interval="1d")
    closes = [float(c) for c in hist["Close"].dropna().tolist()]
    if not closes:
        # 이름으로 말한 경우 티커 찾기
        found = yf.Search(symbol, max_results=1).quotes
        if not found:
            raise ValueError("not found")
        return quote_yfinance(found[0]["symbol"])
    price = closes[-1]

    def change(old):
        return f"{(price / old - 1) * 100:+.2f}%" if old else "n/a"

    try:
        currency = ticker.fast_info["currency"]
    except Exception:
        currency = ""
    last = hist.index[-1].strftime("%Y-%m-%d %H:%M")
    return (f"{symbol}: {price:,.2f} {currency} | 1d {change(closes[-2] if len(closes) > 1 else None)}, "
            f"5d {change(closes[-6] if len(closes) > 5 else None)}, 1mo {change(closes[0])} | as of {last}")


def get_quotes(symbols: list) -> str:
    lines = []
    for raw in symbols[:8]:  # 한 번에 최대 8개
        sym = str(raw).strip()
        if not sym:
            continue
        errors = []
        try:
            lines.append(quote_yfinance(sym))  # 1순위: yfinance (가장 잘 안 막힘)
            continue
        except ImportError:
            errors.append("yfinance not installed")
        except Exception as e:
            errors.append(f"yfinance: {e}")
        try:
            lines.append(quote_one(sym))       # 2순위: 야후 직접
            continue
        except Exception as e:
            errors.append(f"yahoo: {e}")
            yahoo_error = e
        try:
            lines.append(quote_stooq(sym))     # 3순위: 예비 데이터
            continue
        except Exception as e:
            errors.append(f"stooq: {e}")
        print(f"(시세 실패 - {sym}: {' / '.join(errors)})")
        try:
            ticker = find_ticker(sym)          # 이름으로 말한 경우 티커 찾기
            lines.append(quote_one(ticker) if ticker else f"{sym}: not found")
        except Exception:
            lines.append(f"{sym}: failed ({yahoo_error})")
    return "\n".join(lines) or "no symbols"


def handle_get_quote(args):
    symbols = args.get("symbols") or []
    if isinstance(symbols, str):
        symbols = [symbols]
    return get_quotes(symbols)

# get_quote: 주가·지수·코인·환율 조회 (읽기만 함)
register(_tool("get_quote", "주가·지수·암호화폐·환율의 현재 가격과 1일/5일/1개월 변동률 가져오기. 여러 개를 한 번에 조회할 수 있다.",
          {"symbols": {"type": "array", "items": {"type": "string"},
                       "description": "야후 파이낸스 티커 또는 회사 이름. 예: NVDA, AAPL, TSLA, 005930.KS(삼성전자), "
                                      "^GSPC(S&P500), ^IXIC(나스닥), ^DJI(다우), ^KS11(코스피), ^GDAXI(독일 DAX), "
                                      "BTC-USD, ETH-USD, USDKRW=X, EURKRW=X, EURUSD=X. 티커를 모르면 회사 이름을 넣어도 된다."}},
          ["symbols"]),
         'auto', handle_get_quote)
