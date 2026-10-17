"""믿을 수 있는 비서: 행동 기록 보고, 이유 설명, 되돌리기"""
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
from jarvis.voice import speak
from jarvis import hud
from jarvis.hearing import ask_permission
from jarvis.secure import touch_id
from skills.system import active_timers
from skills.memory import load_memories, save_memories

undo_stack = []          # 최근 행동을 되돌리는 방법들


def read_log(hours: float = None, today: bool = False, limit: int = 200) -> list:
    """행동 기록 파일 읽기."""
    entries = []
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f.readlines()[-limit:]:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        return []
    now = datetime.datetime.now()
    picked = []
    for e in entries:
        try:
            t = datetime.datetime.fromisoformat(e.get("time", ""))
        except ValueError:
            continue
        if today and t.date() != now.date():
            continue
        if hours and now - t > datetime.timedelta(hours=hours):
            continue
        picked.append(e)
    return picked


def _entry_line(e: dict) -> str:
    acts = "; ".join(f'{a.get("action")}({json.dumps(a.get("args", {}), ensure_ascii=False)}) -> {a.get("result")}'
                     for a in e.get("actions", [])) or "no action (just talked)"
    if e.get("result") == "ignored":
        acts = "ignored (didn't seem addressed to Jarvis)"
    return f'{e.get("time", "")[11:16]} heard "{e.get("heard", "")}" => {acts}'


def activity_report(period: str) -> str:
    entries = (read_log(hours=1) if period == "last_hour"
               else read_log(today=True) if period == "today" else read_log(limit=15))
    if not entries:
        return "no activity in that period"
    results = [str(a.get("result", "")) for e in entries for a in e.get("actions", [])]
    summary = (f"{len(entries)} requests; actions done {sum(r.startswith('done') for r in results)}, "
               f"denied by user {sum('denied' in r for r in results)}, "
               f"blocked {sum('blocked' in r for r in results)}, "
               f"failed {sum(r.startswith('failed') for r in results)}, "
               f"ignored {sum(e.get('result') == 'ignored' for e in entries)}")
    return summary + "\n" + "\n".join(_entry_line(e) for e in entries[-25:])


def explain_last(count: int) -> str:
    entries = [e for e in read_log(limit=30) if e.get("result") != "ignored"][-max(1, min(count, 5)):]
    if not entries:
        return "no recent actions"
    return ("Evidence for recent actions (what I heard -> what I chose). The 'heard' text is the "
            "transcription, which may contain mistakes:\n" + "\n".join(_entry_line(e) for e in entries))


def push_undo(desc: str, fn, confirm=False) -> None:
    """confirm: False(바로) / True(말로 허락) / "secure"(Touch ID)"""
    undo_stack.append({"desc": desc, "fn": fn, "confirm": confirm, "time": datetime.datetime.now()})
    del undo_stack[:-20]


def list_tab_urls(keyword: str) -> list:
    """닫기 전에 주소를 기억해두려고, keyword가 들어간 탭 주소 목록."""
    if len(keyword) < 3:
        return []
    lines = ['on run argv', 'set k to item 1 of argv', 'set out to ""']
    for browser in ("Safari", "Google Chrome"):
        lines += [f'if application "{browser}" is running then', 'try', f'tell application "{browser}"',
                  'repeat with w in windows', 'repeat with t in tabs of w',
                  'if URL of t contains k then set out to out & (URL of t) & linefeed',
                  'end repeat', 'end repeat', 'end tell', 'end try', 'end if']
    lines += ['return out', 'end run']
    result = run_applescript(lines, [keyword])
    return [u for u in result.stdout.splitlines() if u.startswith("http")]


def capture_before(name: str, args: dict):
    """되돌리기를 위해, 실행 '전' 상태를 기억."""
    if name == "hud_display":
        return hud.current_mode()
    if name in ("add_alert", "cancel_alert"):
        from skills.alerts import load_alerts
        alerts = load_alerts()
        if name == "add_alert":
            return {a["id"] for a in alerts}
        try:
            return alerts[int(args.get("number", 0)) - 1]
        except (IndexError, TypeError, ValueError):
            return None
    if name == "set_volume":
        return {"volume": get_volume(), "muted": is_muted()}
    if name == "jarvis_voice":
        return dict(load_state())

    if name == "close_tab":
        return list_tab_urls(str(args.get("keyword", "")).strip().lower())
    if name == "forget_fact":
        try:
            return load_memories()[int(args.get("number", 0)) - 1]
        except (IndexError, TypeError, ValueError):
            return None
    return None


def _close_url(url: str) -> None:
    key = url
    match = re.search(r"[?&]v=([\w-]{11})", url)   # 유튜브 영상은 영상 번호로 찾기
    if match:
        key = "v=" + match.group(1)
    execute("close_tab", {"keyword": key})


def _set_language(value: str) -> None:
    S.reply_language = value


def _forget_text(fact: str) -> None:
    save_memories([m for m in load_memories() if m.get("fact") != fact])


def record_undo(name: str, args: dict, before) -> None:
    """방금 한 행동을 되돌리는 방법을 등록. (읽기만 하는 행동은 되돌릴 게 없음)"""
    if name == "set_volume" and args.get("mode") in ("up", "down"):
        # 키로 올렸으면 반대 키를 같은 횟수만큼 (화면 표시도 똑같이 뜨게)
        steps = max(1, min(8, int(args.get("steps", 1) or 1)))
        back = "volume_down" if args.get("mode") == "up" else "volume_up"
        word = "down" if args.get("mode") == "up" else "up"
        push_undo(f"turn the volume back {word} {steps} step(s), to where it was",
                  lambda k=back, n=steps: [press_media_key(k) for _ in range(n)])
    elif name == "set_volume" and before and before["volume"] >= 0:
        b = before
        push_undo("set the volume back to where it was",
                  lambda: (run_applescript([f"set volume output volume {b['volume']}"]), set_muted(b["muted"])))
    elif name == "jarvis_voice" and before is not None:
        push_undo("restore my previous voice settings", lambda b=before: save_state(b))
    elif name == "add_alert" and before is not None:
        from skills.alerts import load_alerts, remove_alert_id
        for new in [a for a in load_alerts() if a["id"] not in before]:
            push_undo(f'cancel the alert "{new.get("text") or new.get("symbol")}"', lambda i=new["id"]: remove_alert_id(i))
    elif name == "cancel_alert" and before:
        from skills.alerts import restore_alert
        push_undo(f'restore the alert "{before.get("text") or before.get("symbol")}"', lambda a=before: restore_alert(a))
    elif name == "hand_remote" and args.get("mode") == "on":
        push_undo("turn the hand remote off", lambda: __import__("skills.hand_remote", fromlist=["x"]).set_remote(False))
    elif name == "hands_mode" and args.get("mode") == "on":
        push_undo("turn hand mode off", lambda: __import__("skills.hands_mode", fromlist=["x"]).set_hands(False))
    elif name == "hud_display" and before:
        push_undo(f"set my screen display back to '{before}'", lambda b=before: hud.set_mode(b))
    elif name == "set_reply_language":
        push_undo(f"switch my reply language back to {before}", lambda b=before: _set_language(b))
    elif name == "close_tab" and before:
        push_undo(f"reopen {len(before)} closed tab(s)",
                  lambda urls=before: [subprocess.run(["open", u]) for u in urls])
    elif name in ("open_url", "web_search", "youtube_search", "youtube_play", "open_article") and S.last_opened_url:
        push_undo("close the page I opened", lambda u=S.last_opened_url: _close_url(u))
    elif name == "open_app":
        app = str(args.get("app", ""))
        if app.lower() not in PROTECTED_APPS:
            push_undo(f"quit {app}", lambda a=app: run_applescript(
                ['on run argv', 'tell application (item 1 of argv) to quit', 'end run'], [a]), confirm=True)
    elif name == "quit_app":
        app = str(args.get("app", ""))
        push_undo(f"reopen {app}", lambda a=app: subprocess.run(["open", "-a", a]))
    elif name == "spotify_play":
        push_undo("pause Spotify", lambda: run_applescript(['tell application "Spotify" to pause']))
    elif name == "music":
        command = str(args.get("command", ""))
        inverse = {"pause": "playpause", "play": "playpause", "next": "previous", "previous": "next"}.get(command)
        if inverse:
            push_undo(f"undo '{command}'", lambda k=inverse: press_media_key(k))
    elif name == "set_timer" and active_timers:
        push_undo("cancel the timer", lambda t=active_timers[-1]: t.cancel())
    elif name == "remember_fact":
        fact = str(args.get("fact", "")).strip()
        push_undo(f'forget "{fact}"', lambda f=fact: _forget_text(f))
    elif name == "forget_fact" and before:
        push_undo(f'remember "{before["fact"]}" again',
                  lambda m=before: save_memories(load_memories() + [m]))
    elif name == "add_reminder":
        text = str(args.get("text", ""))
        push_undo(f'delete the reminder "{text}"', lambda t=text: run_applescript([
            'on run argv',
            'tell application "Reminders" to delete (first reminder whose name is (item 1 of argv) and completed is false)',
            'end run'], [t]), confirm=True)
    elif name == "add_event":
        title = str(args.get("title", ""))
        push_undo(f'delete the event "{title}"', lambda t=title: run_applescript([
            'on run argv',
            'tell application "Calendar"',
            'repeat with c in calendars',
            'try',
            'delete (every event of c whose summary is (item 1 of argv) and start date > ((current date) - 400 * days))',
            'end try',
            'end repeat',
            'end tell',
            'end run'], [t]), confirm="secure")


def undo_last():
    """마지막 행동 되돌리기. 지우는 종류는 한 번 더 허락받음 (기본법)."""
    limit = datetime.timedelta(minutes=UNDO_MINUTES)
    undo_stack[:] = [u for u in undo_stack if datetime.datetime.now() - u["time"] <= limit]
    if not undo_stack:
        return "nothing to undo (only things I changed in the last 30 minutes can be undone)"
    item = undo_stack[-1]
    if item["confirm"] == "secure":
        speak(f"I'll {item['desc']}. Please confirm with Touch ID.")
        hud.set("secure", text=f"Undo: {item['desc']}")
        ok, why = touch_id(f"approve Jarvis: {item['desc']}")
        print(f"(Touch ID: {why})")
        if not ok:
            return f"denied: Touch ID {why}"
    elif item["confirm"]:
        speak(f"I'll {item['desc']}. Shall I go ahead?")
        hud.set("confirm", text=f"Undo: {item['desc']}?")
        if not ask_permission():
            return "denied by user"
    undo_stack.pop()
    item["fn"]()
    return f"undone: {item['desc']}"


def handle_activity_report(args):
    return activity_report(str(args.get("period", "today")))


def handle_explain_last(args):
    try:
        return explain_last(int(args.get("count", 1) or 1))
    except (TypeError, ValueError):
        return explain_last(1)


def handle_undo_last(args):
    return undo_last()


# activity_report: "오늘 뭐 했어?" 행동 기록 보고 (읽기만 함)
register(_tool("activity_report", "자비스가 한 일의 기록 보고 ('오늘 뭐 했어?', '최근에 뭐 했어?'). 실행·허락·거절·무시·실패를 나눠서 짧게 말한다.",
          {"period": {"type": "string", "enum": ["today", "last_hour", "recent"]}}, ["period"]),
         'auto', handle_activity_report)

# explain_last: "방금 왜 그거 했어?" 최근 행동의 근거 보여주기 (읽기만 함)
register(_tool("explain_last", "최근 행동을 왜 했는지 설명하기 ('방금 왜 그거 했어?', '왜 그 노래 틀었어?'). 결과에 있는 '들은 말'과 '고른 행동'을 근거로 솔직하게 설명하고, 잘못 알아들었으면 인정한다.",
          {"count": _i("최근 몇 개를 볼지. 보통 1")}, ["count"]),
         'auto', handle_explain_last)

# undo_last: "방금 거 되돌려줘" (지우는 되돌리기는 안에서 따로 허락받음)
register(_tool("undo_last", "자비스가 마지막으로 한 행동을 되돌리기 ('방금 거 취소해', '되돌려줘', '원래대로 해줘')"),
         'auto', handle_undo_last)
