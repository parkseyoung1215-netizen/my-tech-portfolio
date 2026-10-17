"""듣기: 호출어, 목소리 잠금, 말 끝 감지, 받아쓰기, 말로 허락받기"""

import os
import select
import subprocess
import sys
import tempfile
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from config import *
from jarvis.core import client, get_volume, run_applescript, set_muted, is_muted
from jarvis.voice import speak
from jarvis import hud

wake_model = None
try:
    import openwakeword
    from openwakeword.model import Model
    openwakeword.utils.download_models(["hey_jarvis"])  # 처음 한 번만 내려받음
    wake_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    if VOICE_LOCK and os.path.exists(VERIFIER_PATH):
        try:
            name = list(wake_model.models.keys())[0]
            wake_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx",
                               custom_verifier_models={name: VERIFIER_PATH},
                               custom_verifier_threshold=VERIFY_FROM)
            print("(목소리 잠금 켜짐: 등록된 목소리에만 반응해요)")
        except Exception as e:
            print(f"(목소리 잠금을 켜지 못했어요, 일반 모드로 실행: {e})")
    elif VOICE_LOCK:
        print("(목소리 미등록: 누구 목소리든 반응해요. enroll_voice.py로 등록하세요)")
except Exception as e:
    print(f"(호출어 기능을 켜지 못해서 Enter 방식으로 실행합니다: {e})")

HAS_KEYBOARD = sys.stdin is not None and sys.stdin.isatty()
noise_floor = 150.0
HALLUCINATIONS = ("시청해주셔서감사합니다", "구독과좋아요", "구독좋아요", "MBC뉴스", "KBS뉴스",
                  "다음영상에서만나요", "Thanksforwatching", "Pleasesubscribe",
                  "자막제공", "자막by", "Subtitlesby")
UNDO_BUTTON = "<화면 되돌리기 버튼>"   # 화면 버튼으로 되돌리기를 요청했다는 표시
YES_WORDS = ("yes", "yeah", "yep", "sure", "goahead", "doit", "okay", "ok", "please",
             "응", "어", "네", "예", "그래", "좋아", "해", "해줘", "진행", "ㅇㅇ")
NO_WORDS = ("no", "nope", "cancel", "stop", "don't", "dont", "nevermind",
            "아니", "아니야", "아니요", "취소", "하지마", "됐어", "그만")


def wait_for_trigger() -> str:
    """호출어("헤이 자비스")가 들릴 때까지 기다린다. 호출어를 못 켰으면 Enter로 대신.
    화면의 되돌리기 버튼을 누르면 "undo"를 돌려준다."""
    hud.busy = False
    hud.set("idle")
    if wake_model is None:
        if not HAS_KEYBOARD:
            # 창 없이(백그라운드로) 돌 때는 Enter를 누를 수 없으니 호출어가 꼭 필요함
            print("(호출어 기능이 없어서 백그라운드로는 실행할 수 없어요. 종료합니다)")
            sys.exit(0)
        input(f"\n[Enter]를 누르고 {RECORD_SECONDS}초 동안 말하세요...")
        return "wake"
    print(f'\n"헤이 {ASSISTANT_NAME}"라고 부르거나, 이 터미널에서 Enter를 누르세요...')
    wake_model.reset()
    hud.undo_clicked.clear()
    soft_count = 0
    best = 0.0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=WAKE_FRAME) as stream:
        while True:
            # Enter로도 깨울 수 있게 (음악이 시끄러워 호출어가 안 들릴 때)
            if HAS_KEYBOARD and select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break
            if hud.undo_clicked.is_set():      # 화면에서 되돌리기 버튼을 누름
                hud.undo_clicked.clear()
                return "undo"
            frame, _ = stream.read(WAKE_FRAME)
            track_noise(frame)
            # 작은 목소리도 잡히도록 소리를 키워서 넣기 (찢어지지 않게 최대값에서 자름)
            boosted = np.clip(frame.flatten().astype(np.float32) * WAKE_GAIN,
                              -32768, 32767).astype(np.int16)
            scores = wake_model.predict(boosted)
            score = max(scores.values())
            if score >= 0.05:
                best = max(best, score)
            elif best > 0:  # 한 번 부르고 끝난 뒤 최고 점수만 한 줄로 보여줌
                if SHOW_WAKE_SCORE:
                    print(f"(호출어 최고 점수: {best:.2f} / 기준 {WAKE_THRESHOLD})")
                best = 0.0
            soft_count = soft_count + 1 if score >= WAKE_SOFT else 0
            if score >= WAKE_THRESHOLD or soft_count >= WAKE_SOFT_FRAMES:
                print(f"(깨어남: 점수 {score:.2f})")
                break
    subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"])
    return "wake"


def track_noise(frame) -> None:
    """조용한 순간들의 소리 크기를 천천히 따라가며 주변 소음 수준을 기억."""
    global noise_floor
    level = float(np.abs(frame).mean())
    if level < noise_floor * 2:
        noise_floor = noise_floor * 0.98 + level * 0.02


def speech_level() -> float:
    return min(MAX_SPEECH_LEVEL, max(MIN_SPEECH_LEVEL, noise_floor * 2.5))


def listen(skip_wake: bool = False) -> str:
    """깨우면 음악 소리를 잠깐 줄이고, 말이 끝날 때까지 듣고 글로 바꾼다.
    skip_wake=True면 호출어 없이 바로 이어서 듣는다 (대화 이어가기)."""
    if skip_wake:
        print(f"\n(이어서 말하세요... {FOLLOW_UP_WAIT:.0f}초)")
    elif wait_for_trigger() == "undo":
        hud.busy = True
        return UNDO_BUTTON
    hud.busy = True
    hud.set("listening", caption="")
    # 듣는 동안 스피커 소리를 꺼서 노래가 같이 녹음되지 않게 (원래 꺼져 있었으면 그대로 둠)
    mute = MUTE_WHILE_LISTENING and not is_muted()
    if mute:
        set_muted(True)
    try:
        return record_and_transcribe(FOLLOW_UP_WAIT if skip_wake else START_WAIT)
    finally:
        if mute:
            set_muted(False)


def record_until_silence(start_wait: float):
    """말을 시작할 때까지 기다렸다가, 말이 끝나면(잠깐 조용해지면) 녹음을 멈춘다.
    말을 안 하면 None."""
    chunk = int(SAMPLE_RATE * 0.1)  # 0.1초 단위
    threshold = speech_level()
    pre_roll = []      # 말 시작 직전 0.3초도 같이 담기 (첫 글자 잘림 방지)
    frames = []
    started = False
    silent_time = 0.0
    waited = 0.0
    print("듣는 중...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=chunk) as stream:
        while True:
            data, _ = stream.read(chunk)
            level = float(np.abs(data).mean())
            hud.level(level / (threshold * 3))
            if not started:
                pre_roll = (pre_roll + [data.copy()])[-3:]
                waited += 0.1
                if level >= threshold:
                    started = True
                    frames = list(pre_roll)
                elif waited >= start_wait:
                    print("(말소리가 없어서 건너뜀)")
                    hud.level(0)
                    return None
                continue
            frames.append(data.copy())
            silent_time = silent_time + 0.1 if level < threshold * 0.6 else 0.0
            if silent_time >= END_SILENCE or len(frames) * 0.1 >= MAX_LISTEN:
                break
    print(f"(들은 시간: {len(frames) * 0.1:.1f}초)")
    hud.level(0)
    return np.concatenate(frames)


def record_and_transcribe(start_wait: float) -> str:
    """말이 끝날 때까지 녹음해서 글로 바꾼다. 말이 없으면 빈 글자."""
    audio = record_until_silence(start_wait)
    if audio is None:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, SAMPLE_RATE, audio)
        path = f.name
    try:
        with open(path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL, file=audio_file,
                prompt=TRANSCRIBE_HINT)
        return result.text.strip()
    finally:
        os.remove(path)


def _tokens(text: str) -> list:
    """대답을 단어 목록으로 (문장부호 제거, 소문자)."""
    cleaned = "".join(c if c.isalnum() or c in " '" else " " for c in text.lower())
    return cleaned.split()


def is_no(text: str) -> bool:
    joined = "".join(_tokens(text))
    return any(t in NO_WORDS for t in _tokens(text)) or any(
        w in joined for w in NO_WORDS if not w.isascii())


def is_yes(text: str) -> bool:
    tokens = _tokens(text)
    joined = "".join(tokens)
    return any(t in YES_WORDS for t in tokens) or any(
        joined.startswith(w) for w in YES_WORDS)


def ask_permission() -> bool:
    """말로 허락을 받는다. 확실한 '예'가 아니면 전부 '아니오'로 본다 (기본법: 애매하면 실행 안 함)."""
    for attempt in range(2):
        hud.set("confirm")          # 화면에 허락 카드 다시 보이기 (질문 내용은 그대로)
        subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"])
        mute = MUTE_WHILE_LISTENING and not is_muted()
        if mute:
            set_muted(True)
        try:
            answer = record_and_transcribe(CONFIRM_SECONDS)
        finally:
            if mute:
                set_muted(False)
        print(f"나: {answer or '(대답 없음)'}")
        if is_no(answer):
            return False
        if is_yes(answer):
            return True
        if attempt == 0:
            speak("Sorry, was that a yes or a no?")
    speak("I'll take that as a no.")
    return False


def check_microphone() -> None:
    """마이크 점검: 권한이 없으면 맥이 소리 대신 0만 보내줌.
    (처음엔 허용 창에 대답할 시간을 주려고 최대 20초까지 다시 확인)"""
    for _ in range(20):
        test = sd.rec(int(0.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        if int(np.abs(test).max()) > 0:
            break
        time.sleep(0.5)
    if int(np.abs(test).max()) == 0:
        print("(마이크 소리가 전혀 안 들어와요. 마이크 권한을 확인하세요)")
        speak("I can't hear anything. Please check the microphone permission.")
    else:
        print(f"(마이크 정상: 주변 소리 크기 {float(np.abs(test).mean()):.0f})")
