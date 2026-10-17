"""손 모드: 말로 켜고 끄는 손짓 조작 화면 (hud/hands.html)

기본법 (카메라)
- 내가 켜라고 할 때만 켜짐. "꺼줘" 하면 카메라와 화면이 같이 꺼짐
- 손 모드 화면 서버는 이 컴퓨터 안(127.0.0.1)에서만 열리고, 주소가 localhost가 아니면 거부
- 화면에 보여줄 파일은 hands.html 하나뿐 (폴더 안 다른 파일은 안 보여줌)
- 화면이 자비스에게 보낼 수 있는 건 "되돌려줘" 요청 하나뿐 (실제로 할지는 기본법대로: 지우는 종류는 지문·말로 다시 허락)
- 자비스가 꺼지면 화면이 연결 끊김을 알아채고 카메라를 끔
"""

import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import *
from jarvis import hud
from jarvis.registry import register, _tool

PAGE = os.path.join(BASE_DIR, "hud", "hands.html")
_server = None
_on = False
_today = {"text": {}, "at": 0}


def _refresh_today():
    """오늘 패널 내용 (날씨·일정·관심 종목). 손 모드가 켜져 있는 동안 10분마다."""
    from skills.briefing import get_weather
    from skills.market import get_quotes
    from skills.memory import watchlist_symbols
    from jarvis.registry import execute
    while _on:
        text = {}
        try:
            text["weather"] = get_weather()
        except Exception:
            text["weather"] = "weather unavailable"
        try:
            events = execute("today_events", {})
            text["events"] = events if isinstance(events, str) else "(couldn't read the calendar)"
        except Exception:
            text["events"] = "(couldn't read the calendar)"
        try:
            symbols = watchlist_symbols()
            text["quotes"] = get_quotes(symbols[:4]) if symbols else "(no watchlist yet)"
        except Exception:
            text["quotes"] = "(quotes unavailable)"
        _today.update(text=text, at=time.time())
        for _ in range(600):
            if not _on:
                return
            time.sleep(1)


def state_json() -> dict:
    from jarvis.trust import read_log, undo_stack
    recent = []
    for e in read_log(limit=40)[-30:]:
        if e.get("result") == "ignored":
            continue
        acts = [f'{a.get("action")} → {str(a.get("result", ""))[:40]}' for a in e.get("actions", [])]
        recent.append({"time": e.get("time", "")[11:16], "heard": e.get("heard", "")[:60], "acts": acts[:2]})
    return {"on": _on, "hud": hud.last, "recent": recent[-5:][::-1],
            "undo": undo_stack[-1]["desc"] if undo_stack else "", "today": _today["text"]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _local(self) -> bool:
        # 다른 웹사이트가 몰래 이 주소를 부르는 걸 막기 (주소 이름이 localhost일 때만 허용)
        return self.headers.get("Host", "").split(":")[0] in ("localhost", "127.0.0.1")

    def _send(self, code, body: bytes, kind="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._local():
            return self._send(403, b"{}")
        path = self.path.split("?")[0]
        if path in ("/", "/hands.html"):
            with open(PAGE, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/state":
            return self._send(200, json.dumps(state_json(), ensure_ascii=False).encode("utf-8"))
        self._send(404, b"{}")

    def do_POST(self):
        # 되돌리기 요청만. 우리 화면에서 보낸 것만 (다른 사이트는 이 표시를 붙일 수 없음)
        if not self._local() or self.path != "/api/undo" or self.headers.get("X-Jarvis") != "1":
            return self._send(403, b"{}")
        hud.undo_clicked.set()
        self._send(200, b'{"ok": true}')


def start_server() -> bool:
    global _server
    if _server:
        return True
    try:
        _server = ThreadingHTTPServer(("127.0.0.1", HANDS_PORT), Handler)
    except OSError as e:
        print(f"(손 모드 화면을 못 열었어요: {e})")
        return False
    threading.Thread(target=_server.serve_forever, daemon=True).start()
    return True


def stop_server_later(seconds: float = 6) -> None:
    """화면이 '꺼짐'을 읽고 카메라를 끌 시간을 준 뒤 서버 닫기."""
    def stop():
        global _server
        time.sleep(seconds)
        if not _on and _server:
            _server.shutdown()
            _server.server_close()
            _server = None
    threading.Thread(target=stop, daemon=True).start()


def set_hands(on: bool):
    global _on
    if on:
        if not start_server():
            return False
        _on = True
        threading.Thread(target=_refresh_today, daemon=True).start()
        subprocess.run(["open", f"http://localhost:{HANDS_PORT}/hands.html"])
        return "hand mode is on: the panel screen opened and the camera starts (video is not saved or sent)"
    _on = False
    stop_server_later()
    return "hand mode is off: the camera is turned off"


def handle_hands_mode(args):
    mode = str(args.get("mode", "")).strip()
    if mode not in ("on", "off"):
        return False
    return set_hands(mode == "on")


# hands_mode: 손 모드 켜고 끄기 (카메라를 켜는 거라 내가 말할 때만, 되돌리기 가능)
register(_tool("hands_mode",
               "손 모드(웹캠 손짓으로 3D 패널 조작 화면) 켜기·끄기. "
               "'손 모드 켜줘/손으로 할래' → on, '손 모드 꺼줘/카메라 꺼줘' → off. 사용자가 직접 말할 때만 켠다.",
               {"mode": {"type": "string", "enum": ["on", "off"], "description": "on / off"}}, ["mode"]),
         "auto", handle_hands_mode)
