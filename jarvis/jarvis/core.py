"""공통 바탕: API 키, AI 연결, AppleScript, 상태 파일, 기록, 맥 소리 제어"""

import datetime
import json
import os
import ssl
import subprocess
import sys
from zoneinfo import ZoneInfo

from openai import OpenAI

from config import *


class SharedState:
    """여러 모듈이 같이 읽고 바꾸는 값."""
    reply_language = "english"   # 자비스 대답 언어 (말로 바꿀 수 있음)
    last_opened_url = ""         # 자비스가 마지막으로 연 주소 (되돌리기로 닫기)


S = SharedState()
history = []             # 최근 대화 (짧은 기억)


def load_keys() -> None:
    """이 파일과 같은 폴더의 .env 파일에서 키를 읽어온다 (KEY=값 형식).
    터미널에서 export로 넣은 값이 있으면 그걸 우선한다."""
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and not os.environ.get(key):
                os.environ[key] = value


load_keys()
client = OpenAI()  # .env 파일 또는 환경변수의 OPENAI_API_KEY 사용


def run_applescript(lines: list, argv: list = None) -> subprocess.CompletedProcess:
    """AppleScript 실행. 사용자 말(글자)은 코드에 섞지 않고 argv로만 넘긴다 (기본법: 명령 주입 방지)."""
    cmd = ["osascript"]
    for line in lines:
        cmd += ["-e", line]
    return subprocess.run(cmd + [str(a) for a in (argv or [])],
                          capture_output=True, text=True, timeout=30)


def ssl_context():
    """맥용 파이썬은 인증서가 비어 있는 경우가 많아서, openai와 같이 설치된 certifi 인증서를 씀."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def home_now() -> datetime.datetime:
    try:
        return datetime.datetime.now(ZoneInfo(HOME_TZ))
    except Exception:
        return datetime.datetime.now()


def load_state() -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


MEDIA_KEYS = {"playpause": 16, "next": 17, "previous": 18, "volume_up": 0, "volume_down": 1}


def press_media_key(name: str) -> bool:
    """키보드의 ⏯ ⏭ ⏮ 키를 누른 것처럼 신호를 보냄. 유튜브(브라우저), 스포티파이, 음악 앱 모두 반응.
    필요: pip install pyobjc-framework-Quartz, 그리고 손쉬운 사용 권한."""
    try:
        import Quartz
        from AppKit import NSEvent
    except ImportError:
        print("(미디어 키를 쓰려면 터미널에서: pip install pyobjc-framework-Quartz)")
        return False
    key = MEDIA_KEYS[name]
    try:
        for down in (True, False):
            state = 0xA if down else 0xB
            event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                14, (0, 0), state << 8, 0, 0, None, 8, (key << 16) | (state << 8), -1)
            Quartz.CGEventPost(0, event.CGEvent())
        return True
    except Exception as e:
        print(f"(미디어 키 실패: {e})")
        return False


def get_volume() -> int:
    result = run_applescript(["output volume of (get volume settings)"])
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def set_muted(muted: bool) -> None:
    """맥 소리 끄기/켜기 (볼륨 크기는 그대로 기억됨)."""
    run_applescript([f"set volume {'with' if muted else 'without'} output muted"])


def is_muted() -> bool:
    return "true" in run_applescript(["output muted of (get volume settings)"]).stdout


def remember(entry: dict) -> None:
    """방금 대화를 기억에 추가 (오래된 건 버림)."""
    history.append({**entry, "time": datetime.datetime.now()})
    del history[:-MEMORY_TURNS]


def log(entry: dict) -> None:
    """기본법: 모든 실행 기록을 남긴다. (대화 기억에도 같이 넣음)"""
    remember(entry)
    entry["time"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
