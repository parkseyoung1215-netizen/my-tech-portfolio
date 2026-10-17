"""앱과 웹: 앱 열기·끄기, 웹페이지, 구글 검색, 탭 닫기"""
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


def handle_open_app(args):
    app = str(args.get("app", "")).strip()
    return bool(app) and subprocess.run(["open", "-a", app]).returncode == 0


def handle_open_url(args):
    url = str(args.get("url", "")).strip()
    if not url.startswith("https://"):  # 기본법: https 주소만
        return False
    S.last_opened_url = url
    return subprocess.run(["open", url]).returncode == 0


def handle_web_search(args):
    query = str(args.get("query", "")).strip()
    if not query:
        return False
    url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
    S.last_opened_url = url
    return subprocess.run(["open", url]).returncode == 0


def handle_close_tab(args):
    keyword = str(args.get("keyword", "")).strip().lower()
    if len(keyword) < 3:  # 기본법: 너무 짧은 단어로 이것저것 닫지 않게
        return False
    result = run_applescript([
        'on run argv',
        'set k to item 1 of argv',
        'set closed to 0',
        'if application "Safari" is running then',
        'try',
        'tell application "Safari"',
        'repeat with w in windows',
        'set closed to closed + (count of (every tab of w whose URL contains k))',
        'close (every tab of w whose URL contains k)',
        'end repeat',
        'end tell',
        'end try',
        'end if',
        'if application "Google Chrome" is running then',
        'try',
        'tell application "Google Chrome"',
        'repeat with w in windows',
        'set closed to closed + (count of (every tab of w whose URL contains k))',
        'close (every tab of w whose URL contains k)',
        'end repeat',
        'end tell',
        'end try',
        'end if',
        'return closed as text',
        'end run'], [keyword])
    try:
        return int(result.stdout.strip()) > 0
    except ValueError:
        return False


def handle_quit_app(args):
    app = str(args.get("app", "")).strip()
    if not app or app.lower() in PROTECTED_APPS:  # 기본법: 자비스가 돌아가는 앱은 못 끔
        return False
    return run_applescript([
        'on run argv',
        'if application (item 1 of argv) is running then',
        'tell application (item 1 of argv) to quit',
        'end if',
        'end run'], [app]).returncode == 0


# open_app: 앱 열기
register(_tool("open_app", "맥 앱 열기", {"app": _s("맥 앱의 정확한 영문 이름. 예: Claude, Safari, Spotify, Notes, Calendar")}, ["app"]),
         'auto', handle_open_app)

# open_url: 웹페이지 열기
register(_tool("open_url", "웹페이지 열기", {"url": _s("https:// 로 시작하는 주소")}, ["url"]),
         'auto', handle_open_url)

# web_search: 구글 검색 결과 열기
register(_tool("web_search", "구글 검색 결과 페이지 열기", {"query": _s("검색어")}, ["query"]),
         'auto', handle_web_search)

# close_tab: 브라우저 탭 닫기 (실수해도 ⌘⇧T로 되살릴 수 있음)
register(_tool("close_tab", "브라우저 탭 닫기 (멈추기가 아니라 끄기)", {"keyword": _s("닫을 탭 주소에 들어있는 단어. 예: youtube, google")}, ["keyword"]),
         'auto', handle_close_tab)

# quit_app: 앱 끄기 (저장 안 한 작업이 날아갈 수 있어서 허락 필요)
register(_tool("quit_app", "앱 종료 (프로그램이 사용자에게 허락을 받음)", {"app": _s("앱의 정확한 영문 이름")}, ["app"]),
         'confirm', handle_quit_app)
