"""말하기: 맥 목소리(Ava/유나), OpenAI·ElevenLabs 목소리"""

import json
import os
import re
import subprocess
import tempfile
import urllib.request

from config import *
from jarvis.core import client, load_state, ssl_context
from jarvis import hud

_voice_cache = {}


def eleven_tts(text: str) -> bytes:
    """ElevenLabs에 글을 보내 mp3 소리를 받아온다."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY가 없어요")
    request = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
        "?output_format=mp3_44100_128",
        data=json.dumps({"text": text, "model_id": ELEVEN_MODEL}).encode("utf-8"),
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30, context=ssl_context()) as response:
        return response.read()


def resolve_voice(name: str) -> str:
    """'Ava' -> 'Ava (Premium)'처럼 맥에 설치된 가장 좋은 버전의 정확한 이름 찾기."""
    if not name:
        return ""
    if name in _voice_cache:
        return _voice_cache[name]
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    names = [m.group(1).strip() for m in re.finditer(r"^(.+?)\s{2,}[a-z]{2}_[A-Z]{2}", out, re.M)]
    matches = [n for n in names if n == name or n.startswith(name + " (")]
    best = ""
    for tier in ("Premium", "프리미엄", "Enhanced", "고급"):
        best = next((n for n in matches if tier in n), "")
        if best:
            break
    best = best or (matches[0] if matches else "")
    if not best:
        print(f"(목소리 '{name}'을 못 찾아서 기본 목소리로 말해요)")
    _voice_cache[name] = best
    return best


def speak(text: str) -> None:
    """대답을 소리로 말하기. AI 목소리가 실패하면 맥 목소리로 대신."""
    print(f"{ASSISTANT_NAME}: {text}")
    hud.set("speaking", caption=text)
    try:
        _say(text)
    finally:
        if not hud.busy:          # 타이머 알림처럼 대화 밖에서 말한 경우엔 끝나고 대기로
            hud.set("idle")


def _say(text: str) -> None:
    if VOICE_ENGINE == "elevenlabs":
        try:
            audio = eleven_tts(text)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio)
                path = f.name
            try:
                subprocess.run(["afplay", path])
            finally:
                os.remove(path)
            return
        except Exception as e:
            print(f"(ElevenLabs 목소리 실패, OpenAI 목소리로 대신: {e})")
    if VOICE_ENGINE in ("openai", "elevenlabs"):
        try:
            response = client.audio.speech.create(
                model="gpt-4o-mini-tts", voice=OPENAI_VOICE, input=text,
                instructions=VOICE_STYLE, response_format="mp3")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(response.content)
                path = f.name
            try:
                subprocess.run(["afplay", path])
            finally:
                os.remove(path)
            return
        except Exception as e:
            print(f"(AI 목소리 실패, 맥 목소리로 대신: {e})")
    # 한글이 있으면 한국어 목소리, 아니면 영어 목소리
    voice = resolve_voice(KOREAN_VOICE if re.search("[가-힣]", text) else MAC_VOICE)
    state = load_state()
    vol = float(state.get("voice_volume", 1.0))
    rate = int(state.get("voice_rate", 0))
    extra = ["-r", str(rate)] if rate else []
    voice_args = ["-v", voice] if voice else []
    if vol < 1.0:
        # 프리미엄 목소리는 크기 조절 명령을 무시해서, 파일로 만든 뒤 작은 소리로 재생
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            path = f.name
        try:
            made = subprocess.run(["say", *voice_args, *extra, "-o", path, text]).returncode == 0
            if made and subprocess.run(["afplay", "-v", str(vol), path]).returncode == 0:
                return
        finally:
            os.remove(path)
    if voice and subprocess.run(["say", *voice_args, *extra, text]).returncode == 0:
        return
    subprocess.run(["say", *extra, text])
