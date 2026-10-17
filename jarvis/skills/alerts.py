"""먼저 말 거는 자비스: 한 번 알림 / 매일 알림 / 가격 알림 / 일정 직전 알림

- 알림은 ~/.jarvis_alerts.json에 저장돼서 자비스를 껐다 켜도 남아 있음
- 시간은 브레멘 시간(HOME_TZ) 기준 (노트북 시계가 한국 시간이어도 '9시'는 브레멘 9시)
- 대화 중이면 끝날 때까지 기다렸다가 말함 (말하는 도중에 끼어들지 않게)
- 알림은 말하기 + 맥 알림 + 화면 카드, 행동 기록에도 남음
- 알림 만들기·지우기는 되돌리기 가능. 알림이 하는 일은 '말하기'뿐이라 바로 실행(auto)
"""

import datetime
import json
import os
import re
import subprocess
import threading
import time
import uuid
from zoneinfo import ZoneInfo

from config import *
from jarvis import hud
from jarvis.core import home_now, log, run_applescript
from jarvis.registry import register, _tool, _s, _i
from jarvis.voice import speak

ALERTS_FILE = os.path.expanduser("~/.jarvis_alerts.json")
LATE_LIMIT = datetime.timedelta(hours=2)   # 자비스가 꺼져 있다 켜졌을 때, 이 안에 놓친 알림만 늦게라도 말함
PRICE_EVERY = 300                           # 가격 확인 간격(초)
WEEKDAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

_lock = threading.Lock()
_queue = []
_announced_events = set()
_last_price_check = 0.0


# ---------- 저장 ----------
def load_alerts() -> list:
    try:
        with open(ALERTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_alerts(alerts: list) -> None:
    with open(ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, ensure_ascii=False, indent=1)


def tz():
    try:
        return ZoneInfo(HOME_TZ)
    except Exception:
        return None


# ---------- 알림 만들기 ----------
def parse_hhmm(text: str):
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(text or ""))
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        return None
    return int(m.group(1)), int(m.group(2))


def describe_alert(a: dict) -> str:
    if a["kind"] == "once":
        when = datetime.datetime.fromtimestamp(a["at_ts"], tz()).strftime("%a %d %b %H:%M")
        return f'"{a["text"]}" once at {when} (Bremen time)'
    if a["kind"] == "daily":
        days = "every day" if not a.get("weekdays") else ", ".join(WEEKDAY_NAMES[d] for d in a["weekdays"])
        return f'"{a["text"]}" at {a["time"]} ({days}, Bremen time)'
    return f'when {a["symbol"]} goes {a["op"]} {a["price"]:g}' + (f' ("{a["text"]}")' if a.get("text") else "")


def make_alert(args: dict, now: datetime.datetime):
    """AI가 준 값으로 알림 하나 만들기. 이상하면 (None, 이유)."""
    kind = str(args.get("kind", "")).strip()
    text = str(args.get("text", "")).strip()[:200]
    alert = {"id": uuid.uuid4().hex[:8], "kind": kind, "text": text, "created": now.isoformat(timespec="seconds")}
    if kind == "once":
        minutes = int(args.get("in_minutes") or 0)
        if minutes > 0:
            at = now + datetime.timedelta(minutes=minutes)
        else:
            hm = parse_hhmm(args.get("time"))
            if not hm:
                return None, "need in_minutes or a time like 09:00"
            day = now.date() + datetime.timedelta(days=int(args.get("day_offset") or 0))
            at = datetime.datetime(day.year, day.month, day.day, hm[0], hm[1], tzinfo=now.tzinfo)
            if at <= now and not args.get("day_offset"):
                at += datetime.timedelta(days=1)          # 이미 지난 시각이면 내일
        if not text:
            return None, "need the reminder text"
        alert["at_ts"] = at.timestamp()
    elif kind == "daily":
        if not parse_hhmm(args.get("time")) or not text:
            return None, "need a time like 08:00 and the reminder text"
        alert["time"] = "%02d:%02d" % parse_hhmm(args.get("time"))
        days = [WEEKDAY_NAMES.index(d) for d in (args.get("weekdays") or []) if d in WEEKDAY_NAMES]
        alert["weekdays"] = sorted(set(days))
        alert["last_fired"] = ""
    elif kind == "price":
        symbol = str(args.get("symbol", "")).strip().upper()
        op = str(args.get("op", "")).strip()
        try:
            price = float(args.get("price"))
        except (TypeError, ValueError):
            return None, "need a price number"
        if not symbol or op not in ("above", "below"):
            return None, "need a symbol and op above/below"
        alert.update(symbol=symbol, op=op, price=price)
    else:
        return None, "kind must be once, daily or price"
    return alert, ""


# ---------- 때가 됐는지 판단 (순수 계산) ----------
def due_alerts(alerts: list, now: datetime.datetime, prices: dict = None):
    """지금 말해야 할 알림들과, 그 뒤에 남길 알림 목록."""
    fire, keep = [], []
    for a in alerts:
        if a["kind"] == "once":
            at = datetime.datetime.fromtimestamp(a["at_ts"], now.tzinfo)
            if now >= at:
                late = now - at > datetime.timedelta(minutes=2)
                if now - at <= LATE_LIMIT:
                    fire.append((a, "late" if late else ""))
                continue                                     # 한 번 알림은 지나면 지움 (너무 오래 지난 건 조용히)
        elif a["kind"] == "daily":
            h, m = parse_hhmm(a["time"])
            today = now.date().isoformat()
            at = now.replace(hour=h, minute=m, second=0, microsecond=0)
            weekday_ok = not a.get("weekdays") or now.weekday() in a["weekdays"]
            if weekday_ok and a.get("last_fired") != today and at <= now <= at + LATE_LIMIT:
                fire.append((a, "late" if now - at > datetime.timedelta(minutes=2) else ""))
                a = {**a, "last_fired": today}
        elif a["kind"] == "price" and prices is not None and a["symbol"] in prices:
            p = prices[a["symbol"]]
            if p is not None and ((a["op"] == "above" and p >= a["price"]) or (a["op"] == "below" and p <= a["price"])):
                fire.append(({**a, "now_price": p}, ""))
                continue                                     # 가격 알림은 한 번 울리면 끝
        keep.append(a)
    return fire, keep


def alert_message(a: dict, note: str) -> str:
    if re.search("[가-힣]", a.get("text") or ""):     # 한국어로 말한 알림은 한국어로 (유나 목소리)
        late = " (자비스가 꺼져 있어서 조금 늦었어요)" if note == "late" else ""
        if a["kind"] == "price":
            return f"{a['symbol']}가 지금 {a['now_price']:,.2f}로, 정해둔 {a['price']:g}을 넘었어요. {a['text']}{late}" \
                if a["op"] == "above" else f"{a['symbol']}가 지금 {a['now_price']:,.2f}로, 정해둔 {a['price']:g} 아래로 내려갔어요. {a['text']}{late}"
        return f"알림이에요. {a['text']}{late}"
    late = " (I was offline at the scheduled time)" if note == "late" else ""
    if a["kind"] == "price":
        return (f"Sir, {a['symbol']} is now {a['now_price']:,.2f}, {a['op']} your {a['price']:g} mark."
                + (f" {a['text']}" if a.get("text") else "") + late)
    return f"Sir, a reminder: {a['text']}{late}"


def parse_price(quote_line: str):
    m = re.search(r":\s*([\d,]+(?:\.\d+)?)", quote_line or "")
    return float(m.group(1).replace(",", "")) if m else None


# ---------- 일정 직전 알림 ----------
def upcoming_events() -> list:
    """앞으로 15분 안에 시작하는 일정: [(제목, 남은 초)]. 시간대와 상관없이 '남은 초'로 받음."""
    result = run_applescript([
        'set nowD to current date',
        'set out to ""',
        'tell application "Calendar"',
        'repeat with c in calendars',
        'try',
        'repeat with e in (every event of c whose start date > nowD and start date < (nowD + 15 * minutes))',
        'set out to out & (summary of e) & tab & ((start date of e) - nowD) & linefeed',
        'end repeat',
        'end try',
        'end repeat',
        'end tell',
        'return out'])
    events = []
    for line in (result.stdout or "").splitlines():
        if "\t" in line:
            title, secs = line.rsplit("\t", 1)
            try:
                events.append((title.strip(), int(float(secs))))
            except ValueError:
                pass
    return events


def event_heads_up(events: list, today: str) -> list:
    """일정 N분 전(±1분)에 한 번씩. 말할 문장 목록."""
    out = []
    for title, secs in events:
        key = (today, title)
        if key in _announced_events:
            continue
        if (EVENT_HEADS_UP - 1) * 60 <= secs <= (EVENT_HEADS_UP + 1) * 60:
            _announced_events.add(key)
            out.append(f"Sir, {title} starts in {round(secs / 60)} minutes.")
    return out


# ---------- 말하기 ----------
def deliver(message: str, record: str) -> None:
    with _lock:
        _queue.append((message, record))


def _speaker():
    """대화 중이 아닐 때만 말함."""
    while True:
        time.sleep(1)
        if hud.busy or not _queue:
            continue
        with _lock:
            message, record = _queue.pop(0)
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])
        run_applescript(['on run argv', 'display notification (item 1 of argv) with title "Jarvis"', 'end run'], [message])
        hud.set("done", caption=message, text="reminder")
        speak(message)
        log({"heard": "(alert)", "actions": [{"action": "alert", "args": {}, "result": record}],
             "reply": message, "result": "done"})


def _scheduler():
    global _last_price_check
    while True:
        try:
            now = home_now()
            prices = None
            alerts = load_alerts()
            if any(a["kind"] == "price" for a in alerts) and time.time() - _last_price_check > PRICE_EVERY:
                _last_price_check = time.time()
                from skills.market import get_quotes
                prices = {}
                for sym in {a["symbol"] for a in alerts if a["kind"] == "price"}:
                    prices[sym] = parse_price(get_quotes([sym]))
            with _lock:
                alerts = load_alerts()
                fire, keep = due_alerts(alerts, now, prices)
                if fire or keep != alerts:
                    save_alerts(keep)
            for a, note in fire:
                deliver(alert_message(a, note), describe_alert(a))
            if EVENT_HEADS_UP:
                for message in event_heads_up(upcoming_events(), now.date().isoformat()):
                    deliver(message, "calendar heads-up")
        except Exception as e:
            print(f"(알림 확인 중 문제: {e})")
        time.sleep(20)


def start() -> None:
    threading.Thread(target=_scheduler, daemon=True).start()
    threading.Thread(target=_speaker, daemon=True).start()


# ---------- 도구 ----------
def handle_add_alert(args):
    alert, why = make_alert(args, home_now())
    if not alert:
        return f"failed: {why}"
    with _lock:
        save_alerts(load_alerts() + [alert])
    return f"alert set: {describe_alert(alert)} [id {alert['id']}]"


def handle_list_alerts(args):
    alerts = load_alerts()
    if not alerts:
        return "no alerts"
    return "\n".join(f"{i}. {describe_alert(a)}" for i, a in enumerate(alerts, 1))


def handle_cancel_alert(args):
    try:
        n = int(args.get("number", 0))
    except (TypeError, ValueError):
        return False
    with _lock:
        alerts = load_alerts()
        if not 1 <= n <= len(alerts):
            return "failed: no alert with that number (list alerts first)"
        removed = alerts.pop(n - 1)
        save_alerts(alerts)
    return f"cancelled: {describe_alert(removed)}"


def remove_alert_id(alert_id: str) -> None:
    with _lock:
        save_alerts([a for a in load_alerts() if a["id"] != alert_id])


def restore_alert(alert: dict) -> None:
    with _lock:
        save_alerts(load_alerts() + [alert])


register(_tool("add_alert",
               "자비스가 정해진 때 먼저 말로 알려주기. "
               "once: '30분 뒤에 빨래 꺼내라고 해줘'(in_minutes=30), '내일 9시에 과제 내라고 알려줘'(time='09:00', day_offset=1). "
               "daily: '평일 아침 8시마다 수업 준비하라고 해줘'(time='08:00', weekdays=[mon..fri]). "
               "price: '애플 200 넘으면 알려줘'(symbol='AAPL', op='above', price=200). "
               "시간은 사용자가 말한 시각 그대로 24시간제로 (시간대 변환하지 말 것). "
               "미리 알림 앱에 할 일로 넣는 건 add_reminder, 단순 카운트다운은 set_timer.",
               {"kind": {"type": "string", "enum": ["once", "daily", "price"]},
                "text": _s("알려줄 내용 (사용자가 말한 언어 그대로). 가격 알림은 빈 문자열 가능"),
                "in_minutes": _i("once: 지금부터 몇 분 뒤. 시각으로 말했으면 0"),
                "time": _s("once/daily: 'HH:MM' 24시간제. 안 쓰면 빈 문자열"),
                "day_offset": _i("once: 0=오늘, 1=내일, 2=모레"),
                "weekdays": {"type": "array", "items": {"type": "string", "enum": WEEKDAY_NAMES},
                             "description": "daily: 요일들. 매일이면 빈 배열"},
                "symbol": _s("price: 티커 (예: AAPL, ^GSPC, BTC-USD). 아니면 빈 문자열"),
                "op": {"type": "string", "enum": ["above", "below", ""], "description": "price: 넘으면 above, 떨어지면 below"},
                "price": {"type": "number", "description": "price: 기준 가격. 아니면 0"}},
               ["kind", "text", "in_minutes", "time", "day_offset", "weekdays", "symbol", "op", "price"]),
         "auto", handle_add_alert)
register(_tool("list_alerts", "자비스가 먼저 알려주기로 한 알림 목록 보기 ('알림 뭐 있어?')"),
         "auto", handle_list_alerts)
register(_tool("cancel_alert", "알림 취소. list_alerts의 번호로 지정 (번호를 모르면 먼저 list_alerts)",
               {"number": _i("알림 번호")}, ["number"]),
         "auto", handle_cancel_alert)
