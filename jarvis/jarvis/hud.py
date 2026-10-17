"""화면(HUD)과 주고받기

- 자비스 → 화면: 지금 상태(대기/듣는 중/생각 중/말하는 중/허락 대기/완료/차단)를 보냄
- 화면 → 자비스: "되돌리기" 버튼을 눌렀다는 신호만 받음
- 이 컴퓨터 안(127.0.0.1)에서만 주고받고, 화면이 꺼져 있어도 자비스는 그냥 계속 일함
- 화면은 보여주기만 함. 실제 실행·허락·차단은 전부 자비스 본체(기본법)가 결정
"""

import atexit
import json
import os
import socket
import subprocess
import sys
import threading

from config import *
from jarvis.core import load_state, save_state

MODES = ("everywhere", "auto", "off")   # 항상 보이기 / 부를 때만 나타나기 / 완전히 끄기
busy = False                            # 자비스가 지금 대화 중인지 (대화 밖에서 말하면 끝나고 대기로)

_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
undo_clicked = threading.Event()     # 화면에서 되돌리기 버튼을 누르면 켜짐
_hud_process = None
last = {"state": "idle"}                # 마지막 상태 (손 모드 화면이 읽어감)


def set(state: str, **data) -> None:
    """화면 상태 바꾸기. 화면이 없거나 꺼져 있으면 조용히 무시."""
    global last
    caption = data.get("caption", last.get("caption", "") if state != "idle" else "")
    last = {"state": state, **data, "caption": caption}
    if not HUD_ENABLED:
        return
    try:
        _out.sendto(json.dumps({"state": state, **data}, ensure_ascii=False).encode("utf-8"),
                    ("127.0.0.1", HUD_PORT))
    except OSError:
        pass


def level(value: float) -> None:
    """마이크 소리 크기(0~1)를 보내서 원이 목소리에 맞춰 움직이게."""
    if not HUD_ENABLED:
        return
    try:
        _out.sendto(json.dumps({"level": round(max(0.0, min(1.0, value)), 3)}).encode(),
                    ("127.0.0.1", HUD_PORT))
    except OSError:
        pass


def current_mode() -> str:
    mode = load_state().get("hud_mode", HUD_MODE)
    return mode if mode in MODES else "everywhere"


def set_mode(mode: str) -> None:
    """화면 보이기 방식을 바꾸고 기억 (다음에 켤 때도 그대로)."""
    state = load_state()
    state["hud_mode"] = mode
    save_state(state)
    try:
        _out.sendto(json.dumps({"mode": mode}).encode(), ("127.0.0.1", HUD_PORT))
    except OSError:
        pass


def _listen_for_buttons() -> None:
    """화면 버튼 신호 받기. 'undo' 한 단어만 받아들이고 나머지는 전부 무시."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", HUD_PORT + 1))
    except OSError as e:
        print(f"(화면 버튼 연결 실패, 버튼만 안 될 뿐 나머지는 정상: {e})")
        return
    while True:
        data, _ = sock.recvfrom(64)
        if data.strip() == b"undo":
            undo_clicked.set()


def start() -> None:
    """화면 프로그램을 따로 띄움 (화면이 멈추거나 꺼져도 자비스는 영향 없음)."""
    global _hud_process
    if not HUD_ENABLED:
        return
    threading.Thread(target=_listen_for_buttons, daemon=True).start()
    app = os.path.join(BASE_DIR, "hud", "hud_app.py")
    try:
        import WebKit  # noqa: F401  (맥 화면 부품이 깔려 있는지 확인만)
    except ImportError:
        print("(화면을 띄우려면 터미널에서: pip install pyobjc-framework-WebKit  — 없어도 자비스는 정상 작동)")
        return
    _hud_process = subprocess.Popen([sys.executable, app, str(HUD_PORT), HUD_POSITION, current_mode()])
    atexit.register(stop)


def stop() -> None:
    if _hud_process and _hud_process.poll() is None:
        _hud_process.terminate()
