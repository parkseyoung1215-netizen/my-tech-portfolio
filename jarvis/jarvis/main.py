"""자비스 실행: 듣기 → 판단 → 실행 → 대답을 계속 반복"""

from config import *
from jarvis.core import history, log
from jarvis.voice import speak
from jarvis.hearing import HAS_KEYBOARD, HALLUCINATIONS, UNDO_BUTTON, check_microphone, listen
from jarvis.brain import run_agent, run_tool, strip_filler
from jarvis import hud

# 기능(스킬)들: 여기서 불러오면 도구 등록부에 자동으로 등록됨
import skills.apps_web      # noqa: F401
import skills.media         # noqa: F401
import skills.system        # noqa: F401
import skills.calendar      # noqa: F401
import skills.news          # noqa: F401
import skills.market        # noqa: F401
import skills.memory        # noqa: F401
import skills.briefing      # noqa: F401
import skills.hud_control   # noqa: F401
import skills.hands_mode    # noqa: F401
import skills.hand_remote   # noqa: F401
import skills.alerts        # noqa: F401
import skills.lookup        # noqa: F401
import jarvis.trust         # noqa: F401
from jarvis.trust import undo_stack
from skills.briefing import maybe_morning_briefing


def show_result(done: list, undo_before: int) -> None:
    """한 일을 화면 카드로: 차단됐으면 빨강, 실행했으면 초록 (+이번에 되돌릴 게 생겼으면 버튼)."""
    blocked = [d for d in done if str(d["result"]).startswith("blocked")]
    acted = [d for d in done if str(d["result"]).startswith(("done", "undone"))
             and d["action"] != "hud_display"]      # 화면 숨기기 자체는 카드로 안 띄움
    if blocked:
        hud.set("blocked", text=blocked[-1]["result"])
    elif acted:
        text = " · ".join(f'{d["action"]} → {str(d["result"]).removeprefix("done").strip(": ") or "ok"}'
                          for d in acted[-3:])
        hud.set("done", text=text[:160], undo=len(undo_stack) > undo_before)


def main() -> None:
    hud.start()
    check_microphone()
    speak("Jarvis online. How can I help?")
    skills.alerts.start()      # 먼저 말 거는 알림 (놓친 알림이 있으면 지금 알려줌)
    maybe_morning_briefing()   # 아침에 켜졌으면 바로 브리핑
    follow_up = False
    while True:
        in_follow_up = follow_up
        heard = listen(skip_wake=follow_up)
        follow_up = False
        if not heard:
            continue
        if heard == UNDO_BUTTON:   # 화면의 되돌리기 버튼 = "방금 거 되돌려줘"라고 말한 것과 같음
            print("(화면 버튼: 되돌리기)")
            before = len(undo_stack)
            result = run_tool("undo_last", {})
            speak("Done, I've undone that." if "undone:" in result else
                  "Okay, I left it as it is." if "denied" in result else "There's nothing to undo.")
            log({"heard": "(HUD undo button)", "actions": [{"action": "undo_last", "args": {}, "result": result[:80]}],
                 "reply": "", "result": "done"})
            show_result([{"action": "undo_last", "result": result}], before)
            continue
        # 받아쓰기가 조용한 소리에서 자주 지어내는 문장은 무시
        if any(p in heard.replace(" ", "") for p in HALLUCINATIONS):
            print(f"(무시: 받아쓰기 착각으로 보임 - {heard})")
            continue
        print(f"나: {heard}")
        hud.set("thinking", caption=heard)
        # 종료 단어만 짧게 말했을 때만 종료 ("과제 끝나면 알려줘" 같은 문장은 종료 안 함)
        cleaned = heard.strip(" .!?~").replace(" ", "").lower()
        if cleaned in QUIT_WORDS:
            if QUIT_EXITS_PROGRAM or HAS_KEYBOARD:
                speak("Shutting down. Goodbye.")
                break
            speak("Alright. I'll be here if you need me.")
            history.clear()      # 대화 기억도 비우고 처음 상태로
            follow_up = False
            continue

        undo_before = len(undo_stack)
        reply, done, ignored = run_agent(heard)
        entry = {"heard": heard, "actions": done, "reply": reply}

        if ignored:
            print("(무시: 비서에게 한 말이 아닌 것 같아요)")
            log({**entry, "result": "ignored"})
            continue

        reply = strip_filler(reply)
        if reply:
            speak(reply)
        log({**entry, "result": "done"})
        show_result(done, undo_before)
        maybe_morning_briefing()   # 오늘 아침 첫 대화였으면 이어서 브리핑

        # 대답·질문만 한 경우엔 호출어 없이 바로 이어서 말할 수 있게
        # (음악·영상을 튼 뒤에는 그 소리를 명령으로 착각할 수 있어서 안 함)
        played = any(d["action"] in ("youtube_play", "spotify_play", "music") for d in done)
        # 이어 말하기는 진짜로 되물었을 때만, 그리고 한 번만 (계속 묻는 고리 방지)
        follow_up = FOLLOW_UP and not played and not in_follow_up and reply.endswith("?")
