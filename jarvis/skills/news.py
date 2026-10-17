"""뉴스: Google News RSS (무료, 키 필요 없음)"""
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
import email.utils
import xml.etree.ElementTree as ET

last_news = []
NEWS_LANG = {"global": "hl=en-US&gl=US&ceid=US:en", "korea": "hl=ko&gl=KR&ceid=KR:ko"}


def fetch_rss(url: str) -> list:
    """RSS 주소에서 기사 목록 [(시간, 제목, 출처, 링크)]."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=10, context=ssl_context()) as response:
        root = ET.fromstring(response.read())
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source = (item.findtext("source") or "").strip()
        try:
            when = email.utils.parsedate_to_datetime(item.findtext("pubDate") or "")
        except Exception:
            when = None
        if title:
            items.append((when, title, source, link))
    return items


def get_news(region: str, topic: str, query: str) -> str:
    """최신 뉴스를 가져와서 번호 붙인 목록 글자로."""
    global last_news
    regions = ["global", "korea"] if region == "both" else [region if region in NEWS_LANG else "global"]
    collected = []
    for r in regions:
        lang = NEWS_LANG[r]
        if topic == "search" and query:
            url = (f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}+when:1d&{lang}")
        else:
            section = topic.upper() if topic in ("business", "technology", "world") else "BUSINESS"
            url = f"https://news.google.com/rss/headlines/section/topic/{section}?{lang}"
        try:
            items = fetch_rss(url)
        except Exception as e:
            print(f"(뉴스를 못 가져왔어요: {e})")
            continue
        now = datetime.datetime.now(datetime.timezone.utc)
        items.sort(key=lambda x: x[0] or now - datetime.timedelta(days=30), reverse=True)
        recent = [i for i in items if i[0] and now - i[0] <= datetime.timedelta(hours=NEWS_HOURS)]
        picked = (recent if len(recent) >= 5 else items)[:NEWS_MAX_PER_FEED]
        collected += [(r,) + i for i in picked]
    if not collected:
        return "no news could be fetched"
    last_news = collected
    now = datetime.datetime.now(datetime.timezone.utc)
    lines = []
    for n, (r, when, title, source, _link) in enumerate(collected, 1):
        ago = f"{int((now - when).total_seconds() // 3600)}h ago" if when else "time unknown"
        lines.append(f"{n}. [{r}, {ago}] {title}" + (f" ({source})" if source else ""))
    return ("Numbered articles (when you mention an article, use this same number so the user "
            "can say 'open number N'):\n" + "\n".join(lines))


def handle_get_news(args):
    return get_news(str(args.get("region", "both")), str(args.get("topic", "business")),
                    str(args.get("query", "")).strip())


def handle_open_article(args):
    try:
        n = int(args.get("number", 0))
    except (TypeError, ValueError):
        return False
    if not 1 <= n <= len(last_news):
        return "no such article number. Get the news first."
    link = last_news[n - 1][4]
    if not link.startswith("https://"):  # 기본법: https 주소만
        return False
    S.last_opened_url = link
    return subprocess.run(["open", link]).returncode == 0

# get_news: 최신 뉴스 가져오기 (읽기만 함)
register(_tool("get_news", "최신 뉴스 기사 목록 가져오기 (물어본 순간 새로 가져옴). 결과를 보고 중요한 3~5개를 골라 번호를 붙여 짧게 브리핑한다.",
          {"region": {"type": "string", "enum": ["global", "korea", "both"],
                      "description": "global=영어 해외뉴스, korea=한국 뉴스, both=둘 다. 사용자가 말 안 하면 both"},
           "topic": {"type": "string", "enum": ["business", "technology", "world", "search"],
                     "description": "경제·시장=business, 기술·AI=technology, 국제=world, 특정 회사·키워드=search"},
           "query": _s("topic이 search일 때 검색어 (예: Nvidia, 삼성전자, AI startup funding). 아니면 빈 문자열")},
          ["region", "topic", "query"]),
         'auto', handle_get_news)

# open_article: 방금 브리핑한 뉴스 중 하나를 브라우저로 열기
register(_tool("open_article", "방금 브리핑한 뉴스 중 번호로 골라서 브라우저로 열기 ('두 번째 거 열어줘')",
          {"number": _i("브리핑 때 말한 번호 (1부터)")}, ["number"]),
         'auto', handle_open_article)
