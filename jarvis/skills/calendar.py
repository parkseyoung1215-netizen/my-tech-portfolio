"""일정과 할 일: 캘린더, 미리 알림"""
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


def get_reminders() -> str:
    """끝내지 않은 미리 알림 (최대 5개)."""
    result = run_applescript([
        'if application "Reminders" is not running then return ""',
        'tell application "Reminders"',
        'set out to ""',
        'set n to 0',
        'repeat with r in (reminders whose completed is false)',
        'set out to out & (name of r) & linefeed',
        'set n to n + 1',
        'if n ≥ 5 then exit repeat',
        'end repeat',
        'return out',
        'end tell'])
    return result.stdout.strip() if result.returncode == 0 else ""


def handle_today_events(args):
    result = run_applescript([
        'set startDay to current date',
        'set time of startDay to 0',
        'set endDay to startDay + 1 * days',
        'set out to ""',
        'tell application "Calendar"',
        'repeat with c in calendars',
        'repeat with e in (every event of c whose start date ≥ startDay and start date < endDay)',
        'set out to out & (summary of e) & " | " & (time string of (start date of e)) & linefeed',
        'end repeat',
        'end repeat',
        'end tell',
        'return out'])
    if result.returncode != 0:
        return False
    return result.stdout.strip() or "(no events today)"


def handle_add_reminder(args):
    text = str(args.get("text", "")).strip()
    if not text:
        return False
    return run_applescript([
        'on run argv',
        'tell application "Reminders" to make new reminder with properties {name:(item 1 of argv)}',
        'end run'], [text]).returncode == 0


def handle_add_event(args):
    title = str(args.get("title", "")).strip()
    try:
        start = datetime.datetime.strptime(str(args.get("start", "")), "%Y-%m-%d %H:%M")
        minutes = max(5, min(24 * 60, int(args.get("minutes", 60))))
    except (TypeError, ValueError):
        return False
    if not title:
        return False
    return run_applescript([
        'on run argv',
        'set d to current date',
        'set day of d to 1',
        'set year of d to (item 2 of argv) as integer',
        'set month of d to (item 3 of argv) as integer',
        'set day of d to (item 4 of argv) as integer',
        'set hours of d to (item 5 of argv) as integer',
        'set minutes of d to (item 6 of argv) as integer',
        'set seconds of d to 0',
        'tell application "Calendar"',
        'set cal to first calendar whose writable is true',
        'make new event at end of events of cal with properties {summary:(item 1 of argv), start date:d, end date:(d + ((item 7 of argv) as integer) * minutes)}',
        'end tell',
        'end run'],
        [title, start.year, start.month, start.day, start.hour, start.minute, minutes]
    ).returncode == 0


# today_events: 오늘 일정 읽기 (읽기만 함)
register(_tool("today_events", "오늘 캘린더 일정 가져오기. 결과를 보고 요약해서 말해준다."),
         'auto', handle_today_events)

# add_reminder: 미리 알림 추가 (기록이 남는 일이라 허락 필요)
register(_tool("add_reminder", "미리 알림 추가 (프로그램이 사용자에게 허락을 받음)", {"text": _s("내용. 사용자가 말한 언어 그대로")}, ["text"]),
         'confirm', handle_add_reminder)

# add_event: 캘린더 일정 추가 (기록이 남는 일이라 허락 필요)
register(_tool("add_event", "캘린더 일정 추가 (프로그램이 사용자에게 허락을 받음)",
          {"title": _s("일정 이름. 사용자가 말한 언어 그대로"), "start": _s("YYYY-MM-DD HH:MM"),
           "minutes": _i("일정 길이(분). 모르면 60")}, ["title", "start", "minutes"]),
         'confirm', handle_add_event)
