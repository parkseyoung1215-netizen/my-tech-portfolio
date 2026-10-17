"""
내 목소리 등록 (자비스 목소리 잠금)

하는 일:
  1) 내가 "헤이 자비스"라고 말하는 걸 10번 녹음  -> "이 목소리면 깨어나라"
  2) 내가 다른 말을 하는 걸 5번 녹음              -> "호출어가 아니면 깨지 마라"
  3) 맥의 다른 목소리들이 "Hey Jarvis"라고 하는 걸 스피커로 틀어서 녹음
                                                  -> "다른 사람이 부르면 깨지 마라"
  4) 위 녹음으로 내 목소리 전용 검증 모델(voice/verifier.pkl)을 만들고 점검

실행 전: 백그라운드 자비스를 꺼두세요 (./jarvis.sh off). 안 그러면 녹음 중에 자비스가 깨어나요.
실행:    python enroll_voice.py
"""

import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SR = 16000
GAIN = 3.0            # assistant.py의 WAKE_GAIN과 같게 (실제로 들을 때와 같은 조건으로 녹음)
N_POSITIVE = 10
N_MY_OTHER_SPEECH = 5
MAX_OTHER_VOICES = 10

DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(DIR, "voice")
POS_DIR = os.path.join(VOICE_DIR, "positive")
NEG_DIR = os.path.join(VOICE_DIR, "negative")
VERIFIER_PATH = os.path.join(VOICE_DIR, "verifier.pkl")


def beep():
    subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"])


def boosted(audio):
    return np.clip(audio.astype(np.float32) * GAIN, -32768, 32767).astype(np.int16)


def record(seconds):
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    return audio.flatten()


def save(path, audio):
    wavfile.write(path, SR, boosted(audio))


def level(audio):
    return float(np.abs(audio).mean())


def record_my_clips(folder, count, seconds, prompt, tips):
    os.makedirs(folder, exist_ok=True)
    i = 0
    while i < count:
        tip = tips[i] if i < len(tips) else ""
        input(f"\n[{i + 1}/{count}] Enter를 누르고, 띵 소리 뒤에 {prompt}  {tip}")
        beep()
        audio = record(seconds)
        if level(audio) < 150:
            print("  (소리가 너무 작아요. 다시 해볼게요)")
            continue
        save(os.path.join(folder, f"me_{i:02d}.wav"), audio)
        print("  저장했어요.")
        i += 1


def english_voices():
    """맥에 있는 영어·한국어 목소리 이름 목록."""
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    names = []
    for line in out.splitlines():
        parts = line.split("#")[0].split()
        if len(parts) < 2:
            continue
        lang = parts[-1]
        name = " ".join(parts[:-1])
        if lang.startswith(("en_", "ko_")) and name not in names:
            names.append(name)
    return names[:MAX_OTHER_VOICES]


def record_other_voices():
    """다른 목소리들이 'Hey Jarvis'라고 하는 걸 스피커로 틀고 마이크로 녹음."""
    os.makedirs(NEG_DIR, exist_ok=True)
    voices = english_voices()
    if not voices:
        print("  (맥에 쓸 수 있는 다른 목소리가 없어서 이 단계는 건너뛰어요)")
        return
    print(f"\n이제 맥의 다른 목소리 {len(voices)}개가 'Hey Jarvis'라고 말하는 걸 녹음할게요.")
    print("스피커 소리를 적당히 키우고, 이 동안은 조용히 있어주세요.")
    input("준비되면 Enter...")
    for i, voice in enumerate(voices):
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            tts = f.name
        try:
            if subprocess.run(["say", "-v", voice, "-o", tts, "Hey Jarvis"]).returncode != 0:
                continue
            recording = sd.rec(int(2.5 * SR), samplerate=SR, channels=1, dtype="int16")
            subprocess.run(["afplay", tts])
            sd.wait()
            save(os.path.join(NEG_DIR, f"other_{i:02d}.wav"), recording.flatten())
            print(f"  {voice} 녹음했어요.")
        finally:
            os.remove(tts)
    # 주변 소리만 있는 것도 하나
    print("  주변 소리 녹음 중 (3초)...")
    save(os.path.join(NEG_DIR, "room.wav"), record(3))


def max_score(model, path):
    preds = model.predict_clip(path)
    return max(max(p.values()) for p in preds) if preds else 0.0


def main():
    print(__doc__)
    try:
        import openwakeword
        from openwakeword.model import Model
        import sklearn  # noqa: F401 (검증 모델 만들 때 필요)
    except ImportError as e:
        print(f"필요한 부품이 없어요: {e}\n터미널에서: pip install scikit-learn")
        sys.exit(1)

    # "python enroll_voice.py train" 이면 녹음은 건너뛰고 이미 있는 녹음으로 모델만 다시 만듦
    train_only = len(sys.argv) > 1 and sys.argv[1] == "train"
    if train_only and glob.glob(os.path.join(POS_DIR, "*.wav")):
        print("이미 있는 녹음으로 모델만 다시 만들게요.")
    else:
        record_all()

    train_and_check()


def record_all():
    # 다시 등록하는 경우 예전 녹음 지우기
    for f in glob.glob(os.path.join(POS_DIR, "*.wav")) + glob.glob(os.path.join(NEG_DIR, "*.wav")):
        os.remove(f)

    print("\n=== 1단계: '헤이 자비스' 10번 ===")
    print("평소에 부르는 것처럼 자연스럽게 말해주세요. 매번 조금씩 다르게 하면 더 정확해져요.")
    tips = ["(보통 목소리)"] * 4 + ["(조금 작게)"] * 2 + ["(조금 빠르게)"] * 2 + ["(노트북에서 조금 떨어져서)"] * 2
    record_my_clips(POS_DIR, N_POSITIVE, 2.0, "'헤이 자비스'라고 말하세요", tips)

    print("\n=== 2단계: 다른 말 5번 ===")
    print("'헤이 자비스'가 아닌 아무 문장이나 말해주세요. (예: 오늘 날씨 어때, 밥 먹었어?)")
    record_my_clips(os.path.join(NEG_DIR), N_MY_OTHER_SPEECH, 3.0, "아무 문장이나 말하세요", [])

    print("\n=== 3단계: 다른 목소리 ===")
    record_other_voices()



def train_and_check():
    import openwakeword
    from openwakeword.model import Model
    print("\n=== 4단계: 내 목소리 전용 모델 만들기 ===")
    positives = sorted(glob.glob(os.path.join(POS_DIR, "*.wav")))
    negatives = sorted(glob.glob(os.path.join(NEG_DIR, "*.wav")))
    openwakeword.utils.download_models(["hey_jarvis"])
    # 맥에서는 tflite가 없어서 onnx 모델 파일을 직접 지정
    onnx_path = "hey_jarvis"
    try:
        paths = openwakeword.get_pretrained_model_paths(inference_framework="onnx")
        onnx_path = next(p for p in paths if "hey_jarvis" in os.path.basename(p))
    except Exception:
        pass
    attempts = [
        dict(model_name=onnx_path, inference_framework="onnx"),
        dict(model_name=onnx_path),
    ]
    last_error = None
    for extra in attempts:
        try:
            openwakeword.train_custom_verifier(
                positive_reference_clips=positives,
                negative_reference_clips=negatives,
                output_path=VERIFIER_PATH,
                **extra,
            )
            last_error = None
            break
        except Exception as e:
            last_error = e
    if last_error:
        print(f"모델 만들기에 실패했어요: {last_error}")
        print("녹음은 저장돼 있으니, 고친 뒤에는 'python enroll_voice.py train'으로 녹음 없이 다시 만들 수 있어요.")
        sys.exit(1)
    print(f"만들었어요: {VERIFIER_PATH}")

    print("\n=== 5단계: 점검 ===")
    try:
        base = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        name = list(base.models.keys())[0]
        locked = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx",
                       custom_verifier_models={name: VERIFIER_PATH},
                       custom_verifier_threshold=0.1)
        mine = [max_score(locked, p) for p in positives]
        others = [max_score(locked, p) for p in glob.glob(os.path.join(NEG_DIR, "other_*.wav"))]
        print(f"  내 '헤이 자비스' 점수: 평균 {np.mean(mine):.2f} (최저 {min(mine):.2f})")
        if others:
            print(f"  다른 목소리 점수:     평균 {np.mean(others):.2f} (최고 {max(others):.2f})")
        print("  내 점수는 높고(0.5 이상), 다른 목소리 점수는 낮을수록(0.2 이하) 잘 된 거예요.")
    except Exception as e:
        print(f"  (점검은 건너뛰었어요: {e})")

    print("\n끝! 이제 자비스를 다시 켜세요:  ./jarvis.sh on")
    print("마음에 안 들면 이 프로그램을 다시 실행해서 새로 등록하면 돼요.")


if __name__ == "__main__":
    main()
