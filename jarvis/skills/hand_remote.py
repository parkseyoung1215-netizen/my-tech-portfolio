"""손 리모컨 켜고 끄기 + 손짓을 받아서 실행

기본법
- 카메라는 내가 "손 리모컨 켜줘"라고 할 때만 (손 리모컨 프로그램이 따로 돌고, 끄면 카메라도 꺼짐)
- 손짓으로 할 수 있는 건 아래 목록뿐: 재생/멈춤, 다음, 이전, 볼륨 올리기·내리기
  (목록에 없는 신호는 무시. 허락이 필요한 행동은 손짓으로 절대 실행 안 됨)
- 재생·다음·이전은 행동 기록에 남음 (볼륨 한 칸씩은 너무 많아서 안 남김)
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time

from config import *
from jarvis import hud
from jarvis.core import press_media_key, log
from jarvis.registry import register, _tool

GESTURES = {   # 손짓 → (누를 키, 화면에 보일 말)
    "playpause": ("playpause", "✋ play / pause"),
    "next": ("next", "👉 next"),
    "previous": ("previous", "👈 previous"),
    "volume_up": ("volume_up", "🤏 volume up"),
    "volume_down": ("volume_down", "🤏 volume down"),
}
DIAL_MARKS = {"dial_start": "🎚 volume dial on — move left/right, pinch to finish",
              "dial_done": "🎚 volume set"}   # 다이얼 켜짐·끝: 키를 누르지 않고 화면 표시만
_process = None
_listener_started = False


def _show(text: str) -> None:
    """대화 중이 아닐 때만 화면에 잠깐 표시."""
    if hud.busy:
        return
    hud.set("done", caption=text, text="hand gesture")
    def back():
        time.sleep(1.5)
        if not hud.busy:
            hud.set("idle")
    threading.Thread(target=back, daemon=True).start()


def handle_gesture(name: str) -> bool:
    """손짓 하나 실행. 목록에 없으면 무시."""
    global _process
    if name == "_auto_off":
        _process = None
        log({"heard": "(hand remote)", "actions": [{"action": "hand_remote", "args": {"mode": "off"},
             "result": "auto off: no hands for a long time"}], "reply": "", "result": "done"})
        return True
    if name in DIAL_MARKS and remote_on():
        _show(DIAL_MARKS[name])
        return True
    if name not in GESTURES or not remote_on():
        return False
    key, label = GESTURES[name]
    ok = press_media_key(key)
    _show(label)
    if not name.startswith("volume"):
        log({"heard": "(hand gesture)", "actions": [{"action": f"gesture:{name}", "args": {},
             "result": "done" if ok else "failed"}], "reply": "", "result": "done"})
    return ok


def _listen() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", HAND_REMOTE_PORT))
    except OSError as e:
        print(f"(손 리모컨 신호를 받을 수 없어요: {e})")
        return
    while True:
        data, _ = sock.recvfrom(256)
        try:
            handle_gesture(str(json.loads(data.decode("utf-8")).get("gesture", "")))
        except Exception:
            continue


def remote_on() -> bool:
    return _process is not None and _process.poll() is None


def set_remote(on: bool):
    global _process, _listener_started
    if on:
        if remote_on():
            return "hand remote is already on"
        if not _listener_started:
            threading.Thread(target=_listen, daemon=True).start()
            _listener_started = True
        script = os.path.join(BASE_DIR, "hud", "hand_remote.py")
        _process = subprocess.Popen([sys.executable, script, str(HAND_REMOTE_PORT)])
        time.sleep(2)
        if not remote_on():
            _process = None
            return False
        return ("hand remote is on (camera on): open palm = play/pause, "
                "volume dial: L-shape (thumb and index wide apart) held, then move left/right, pinch fully to finish")
    if remote_on():
        _process.terminate()
    _process = None
    return "hand remote is off: the camera is turned off"


def _stop_on_exit():
    if remote_on():
        _process.terminate()


import atexit  # noqa: E402
atexit.register(_stop_on_exit)


def handle_hand_remote(args):
    mode = str(args.get("mode", "")).strip()
    if mode not in ("on", "off"):
        return False
    return set_remote(mode == "on")


# hand_remote: 손 리모컨 켜기·끄기 (내가 말할 때만, 되돌리기 가능)
register(_tool("hand_remote",
               "손 리모컨(웹캠 손짓으로 어디서든 재생·멈춤, 볼륨) 켜기·끄기. "
               "'손 리모컨 켜줘/손으로 조종할래' → on, '손 리모컨 꺼줘/카메라 꺼줘' → off. 사용자가 직접 말할 때만 켠다.",
               {"mode": {"type": "string", "enum": ["on", "off"], "description": "on / off"}}, ["mode"]),
         "auto", handle_hand_remote)
