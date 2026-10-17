"""맥 설정: 볼륨, 자비스 목소리, 타이머, 대답 언어"""
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

active_timers = []       # 돌고 있는 타이머 (되돌리기로 취소)


def timer_done(label: str) -> None:
    """타이머가 끝나면 소리 + 알림 + 말하기."""
    name = label or "Timer"
    for _ in range(3):
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])
    run_applescript(['on run argv',
                     'display notification "Time is up" with title (item 1 of argv)',
                     'end run'], [name])
    speak(f"Sir, your {label} timer is done." if label else "Sir, your timer is done.")


def handle_set_volume(args):
    mode = str(args.get("mode", "set"))
    if mode in ("mute", "unmute"):
        set_muted(mode == "mute")
        return f"{mode}d"
    try:
        if mode == "set":
            level = int(args.get("level", 50))
        else:
            steps = max(1, min(8, int(args.get("steps", 1) or 1)))
            # 키보드 볼륨 키(F11/F12)를 진짜 누른 것처럼 (화면에 볼륨 표시도 뜸)
            key = "volume_up" if mode == "up" else "volume_down"
            if all(press_media_key(key) or False for _ in range(steps)):
                time.sleep(0.2)
                return f"pressed volume {mode} {steps} time(s), volume is now {get_volume()}"
            step = 100 / 16 * steps          # 맥 볼륨 키 한 번 = 16칸 중 한 칸
            current = get_volume()
            if current < 0:
                return False
            level = current + (step if mode == "up" else -step)
        level = max(0, min(100, round(level)))
    except (TypeError, ValueError):
        return False
    ok = run_applescript([f"set volume output volume {level}"]).returncode == 0
    return f"volume is now {level}" if ok else False


def handle_jarvis_voice(args):
    state = load_state()
    vol = float(state.get("voice_volume", 1.0))
    rate = int(state.get("voice_rate", 0))           # 0 = 기본 빠르기
    if args.get("volume") == "up":
        vol = min(1.0, vol + 0.2)
    elif args.get("volume") == "down":
        vol = max(0.2, vol - 0.2)
    speed = args.get("speed")
    if speed == "normal":
        rate = 0
    elif speed in ("faster", "slower"):
        base = rate or 185
        rate = max(120, min(260, base + (25 if speed == "faster" else -25)))
    state["voice_volume"], state["voice_rate"] = round(vol, 2), rate
    save_state(state)
    return f"Jarvis voice volume {int(vol * 100)}%, speed {'default' if not rate else rate}"


def handle_set_timer(args):
    try:
        seconds = int(args.get("seconds", 0))
    except (TypeError, ValueError):
        return False
    if not 1 <= seconds <= 24 * 3600:  # 기본법: 하루 넘는 타이머는 안 함
        return False
    label = str(args.get("label", "")).strip()
    timer = threading.Timer(seconds, timer_done, args=[label])
    timer.start()
    active_timers.append(timer)
    return True


def handle_set_reply_language(args):
    language = str(args.get("language", "english"))
    if language not in ("english", "korean"):
        return False
    S.reply_language = language
    return f"reply language is now {language}"


# set_volume: 맥 볼륨 조절 (숫자로 / 한 칸씩 / 음소거)
register(_tool("set_volume", "맥 전체 볼륨 조절. '30으로'는 set, '한 칸 낮춰/조금 올려'는 down/up, '음소거/소리 꺼'는 mute",
          {"mode": {"type": "string", "enum": ["set", "up", "down", "mute", "unmute"]},
           "level": _i("mode가 set일 때 0~100, 아니면 0"),
           "steps": _i("up/down일 때 몇 칸 (한 칸=맥 볼륨 키 한 번). '조금'=1, '많이'=3, 아니면 1")},
          ["mode", "level", "steps"]),
         'auto', handle_set_volume)

# jarvis_voice: 자비스 목소리 크기·빠르기 조절
register(_tool("jarvis_voice", "자비스 자기 목소리만 조절 (음악·맥 전체 볼륨은 그대로). '네 목소리 좀 작게', '좀 천천히 말해'",
          {"volume": {"type": "string", "enum": ["up", "down", "same"]},
           "speed": {"type": "string", "enum": ["faster", "slower", "same", "normal"]}},
          ["volume", "speed"]),
         'auto', handle_jarvis_voice)

# set_timer: 타이머
register(_tool("set_timer", "타이머 맞추기", {"seconds": _i("초"), "label": _s("타이머 이름, 없으면 빈 문자열")}, ["seconds", "label"]),
         'auto', handle_set_timer)

# set_reply_language: 대답 언어 바꾸기 (영어/한국어)
register(_tool("set_reply_language", "앞으로의 대답 언어를 바꾸기 ('계속 한국어로 해줘', '다시 영어로')",
          {"language": {"type": "string", "enum": ["english", "korean"]}}, ["language"]),
         'auto', handle_set_reply_language)
