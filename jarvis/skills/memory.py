"""오래 기억하기: jarvis_memory.json (내 맥 안에만 저장)"""
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


def load_memories() -> list:
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_memories(memories: list) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)


def memory_prompt() -> str:
    memories = load_memories()
    if not memories:
        return "기억하고 있는 것: (아직 없음)"
    lines = [f"{n}. [{m.get('category', 'other')}] {m['fact']}"
             + (f" (symbol: {m['symbol']})" if m.get("symbol") else "")
             for n, m in enumerate(memories, 1)]
    return "기억하고 있는 것:\n" + "\n".join(lines)


def remember_fact(fact: str, category: str, symbol: str):
    fact = fact.strip()
    if not fact:
        return False
    memories = load_memories()
    if any(m["fact"] == fact for m in memories):
        return "already remembered"
    if len(memories) >= MEMORY_LIMIT:
        return "memory is full. Ask the user which old memory to forget."
    memories.append({"fact": fact, "category": category or "other", "symbol": symbol.strip(),
                     "added": home_now().strftime("%Y-%m-%d")})
    save_memories(memories)
    return f"remembered as #{len(memories)}"


def forget_fact(number: int):
    memories = load_memories()
    if not 1 <= number <= len(memories):
        return "no such memory number"
    with open(MEMORY_FILE.replace(".json", ".bak.json"), "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)   # 실수 대비 백업
    removed = memories.pop(number - 1)
    save_memories(memories)
    return f"forgot: {removed['fact']}"


def watchlist_symbols() -> list:
    return [m["symbol"] for m in load_memories() if m.get("category") == "watchlist" and m.get("symbol")]


def handle_remember_fact(args):
    return remember_fact(str(args.get("fact", "")), str(args.get("category", "other")),
                         str(args.get("symbol", "")))


def handle_forget_fact(args):
    try:
        return forget_fact(int(args.get("number", 0)))
    except (TypeError, ValueError):
        return False


# remember_fact: 나에 대한 것 기억하기 (내 맥 안 파일에만 저장)
register(_tool("remember_fact", "사용자에 대해 오래 기억할 것 저장 (사용자가 '기억해/알아둬'라고 하거나, 계속 유효한 자기 정보·취향·일정·관심 종목을 말할 때)",
          {"fact": _s("기억할 내용 한 문장. 사용자가 말한 언어 그대로. 예: '과제 마감은 매주 금요일'"),
           "category": {"type": "string", "enum": ["profile", "preference", "schedule", "watchlist", "other"],
                        "description": "profile=나에 대한 사실, preference=취향·말투, schedule=반복 일정·마감, watchlist=관심 종목"},
           "symbol": _s("watchlist일 때 티커 (예: NVDA, 005930.KS). 아니면 빈 문자열")},
          ["fact", "category", "symbol"]),
         'auto', handle_remember_fact)

# forget_fact: 기억 지우기 (지운 건 jarvis_memory.bak.json에 한 번 백업)
register(_tool("forget_fact", "기억 지우기 ('그거 잊어줘', '관심 종목에서 테슬라 빼줘'). 기억 목록의 번호로 지정.",
          {"number": _i("아래 '기억하고 있는 것' 목록의 번호")}, ["number"]),
         'auto', handle_forget_fact)
