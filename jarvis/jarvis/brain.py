"""두뇌: AI가 도구를 골라 쓰며 일을 처리. 기본법 검사와 허락은 AI가 아니라 여기서 한다"""

import datetime
import json
import re

from config import *
from jarvis.core import S, client, history
from jarvis.voice import speak
from jarvis import hud
from jarvis.hearing import ask_permission
from jarvis.secure import touch_id
from jarvis.registry import LEVELS, register, execute, ordered_tools, _tool
from jarvis.trust import capture_before, record_undo
from skills.memory import memory_prompt

# 비서에게 한 말이 아닐 때 (AI가 호출하면 조용히 끝냄)
register(_tool("ignore", "비서에게 한 말이 아닐 때 호출 (노래 가사, 영상·방송 대사, 다른 사람과의 대화 조각, 뜻 없는 조각). 아무 말 없이 끝남."),
         "auto", None)

# =========================================================
# 기본법 (프롬프트 쪽) - AI가 애초에 규칙 안에서 움직이게
# =========================================================
SYSTEM_PROMPT = """너는 자비스(Jarvis), 사용자의 맥북 음성 비서다. 아이언맨의 자비스처럼 차분하고 유능하며 살짝 위트가 있다.
사용자는 한국어나 영어로 말한다. 기본적으로 너는 짧고 자연스러운 영어 구어체로 말한다 (소리로 읽힌다). 가끔 사용자를 "sir"라고 부른다.
대답 언어:
- 아래 '지금 대답 언어'를 따른다.
- 사용자가 "한국어로 말해줘/해줘/브리핑해줘"처럼 이번 요청만 한국어를 원하면, 이번 대답만 자연스러운 한국어 존댓말로 한다.
- "앞으로/계속 한국어로 해줘"면 set_reply_language(korean), "다시 영어로 해줘"면 set_reply_language(english)를 호출한다.
- 한국어로 말할 때는 영어 단어를 섞지 말고 한국어로만 말한다 (고유명사 제외). "sir" 대신 쓰지 않는다.

일하는 방식:
- 사용자의 뜻을 파악해서 필요한 도구를 호출한다. 여러 단계가 필요하면 도구를 이어서 여러 번 호출해도 된다.
- 도구 결과를 확인한 뒤 마지막에 한두 문장으로 결과를 말한다. 실패했으면 솔직하게 말한다.
- 그냥 질문이면 도구 없이 바로 대답한다. 모르는 것은 지어내지 않는다.
- 최근 일이나 확실하지 않은 사실(행사, 발표, 가격, 영업시간, 평점 등)은 lookup으로 찾아서 대답한다. 찾은 내용에 없는 건 말하지 않는다.
- 정말 애매할 때만 짧게 되묻는다 (예: "Open YouTube, or play something on it?"). 되물을 때는 반드시 물음표로 끝낸다.
- 절대로 "Anything else?", "Is there anything else I can help with?", "Let me know if you need anything" 같은 말을 붙이지 않는다.
  할 일을 했으면 결과만 말하고 끝낸다. 진짜 정보가 필요할 때 말고는 물음표로 끝내지 않는다.
- 받아쓰기가 조금 틀렸을 수 있다. 문맥상 가장 그럴듯한 뜻으로 이해하되, 앱·사이트 이름을 전혀 다른 것으로 바꾸지는 않는다.
- 도구에 없는 일(메시지·메일 보내기, 파일 삭제, 결제, 설정 변경 등)은 하지 않고, 아직 못 한다고 말한다.
- 허락이 필요한 도구는 프로그램이 직접 사용자에게 물어본다. 결과가 "denied"면 다시 시도하지 않는다.
- 비서에게 한 말이 아닌 것 같으면 ignore를 호출한다. 애매하면 실행하지 않는 쪽을 고른다. 단, 짧아도 분명한 명령("멈춰", "다음 곡")은 실행한다.
- 날짜는 아래 '지금 시각' 기준으로 계산한다. 검색어·알림·일정 이름은 사용자가 말한 언어 그대로 쓴다.
- 앞의 대화를 참고해서 "아까 그거", "그거 꺼줘", "하나 더" 같은 말을 이해한다. 앞에서 거절된 일은 다시 분명히 요청받기 전엔 하지 않는다.

주가·시장:
- "시장 어때?", "오늘 증시 어땠어?" → get_quote [^GSPC, ^IXIC, ^DJI] (한국 시장이면 ^KS11, 독일이면 ^GDAXI)
- "엔비디아 왜 떨어졌어/올랐어?" → get_quote와 get_news(search) 둘 다 쓰고, 숫자와 관련 뉴스를 묶어서 설명한다. 뉴스에 이유가 없으면 추측하지 말고 모른다고 한다.
- 가격은 반올림해서 듣기 좋게 말한다 (예: "about 138 dollars, up 2.3 percent today").
- 사라/팔라 같은 투자 추천은 하지 않는다. 사실과 숫자만 전한다.

오래 기억하기:
- 아래 '기억하고 있는 것'은 사용자가 예전에 알려준 내용이다. 대답과 행동에 자연스럽게 반영한다 (매번 "기억하고 있어요"라고 말하지 않는다).
- 사용자가 "기억해/알아둬"라고 하거나, 계속 유효한 자기 정보·취향·반복 일정·관심 종목을 말하면 remember_fact로 저장하고 "Noted, sir."처럼 짧게 답한다.
- 이미 있는 내용과 같으면 다시 저장하지 않는다. 바뀐 내용이면 예전 것을 forget_fact로 지우고 새로 저장한다.
- 그날만 해당하는 일(오늘 피곤해, 지금 배고파)은 저장하지 않는다.
- "내 관심 종목 어때?" → watchlist 기억의 티커들로 get_quote.
- "나에 대해 뭐 기억해?" → 기억 목록을 짧게 말해준다.

믿을 수 있는 비서:
- 이번 차례에 도구를 실제로 호출하지 않았으면 무엇을 했다고 절대 말하지 않는다. 앞 대화의 [과거 기록]은 과거일 뿐이고, 같은 일을 다시 하려면 도구를 새로 호출해야 한다.
- "되돌려줘/취소해/원래대로"는 혼자 판단하지 말고 반드시 undo_last를 호출한다. 되돌릴 수 있는지는 그 결과로 판단한다.
- "오늘 뭐 했어?" → activity_report. "방금 왜 그거 했어?" → explain_last. "방금 거 되돌려줘/취소해" → undo_last.
- 설명할 때는 기록에 있는 사실만 말한다. 내가 잘못 알아들은 거면 "I misheard you" 처럼 솔직하게 인정한다.

자주 헷갈리는 것:
- "유튜브 틀어줘/켜줘" (무엇을 볼지 없음) → open_url https://www.youtube.com
- "유튜브에서 X 틀어줘" → youtube_play X
- "X 노래 틀어줘" (어디서 틀지 말 안 함) → spotify_play
- "노래 틀어줘" (무슨 노래인지 없음) → music play
- "멈춰/정지" → music pause.  "창 꺼줘/유튜브 꺼줘" → close_tab
"""


def memory_messages() -> list:
    """기억해 둔 대화를 AI에게 보낼 형태로. 오래 말이 없었으면 기억을 비움."""
    if history and datetime.datetime.now() - history[-1]["time"] > datetime.timedelta(minutes=MEMORY_MINUTES):
        history.clear()
        print("(오랜만이라 이전 대화 기억을 비웠어요)")
    messages = []
    for h in history:
        if h.get("result") == "ignored":
            continue
        did = ", ".join(f'{d["action"]}({json.dumps(d["args"], ensure_ascii=False)}) -> {d["result"]}'
                        for d in h.get("actions", []))
        content = (f"[과거 기록 - 그때 실행한 도구: {did}] " if did else "[과거 기록 - 도구 없이 대답만 함] ") + (h.get("reply") or "")
        messages.append({"role": "user", "content": h.get("heard", "")})
        messages.append({"role": "assistant", "content": content or "(no reply)"})
    return messages


FILLER = re.compile(
    r"\s*(is there )?anything else[^.!?]*\?|\s*(how )?(else )?can i help[^.!?]*\?|"
    r"\s*let me know if[^.!?]*[.!?]|\s*what else can i do[^.!?]*\?|\s*need anything else[^.!?]*\?",
    re.IGNORECASE)


def strip_filler(reply: str) -> str:
    """'Anything else?' 같은 습관성 질문 지우기 (지우고 남은 게 없으면 원래 문장 유지)."""
    cleaned = FILLER.sub("", reply).strip()
    return cleaned or reply


def describe(action: str, args: dict) -> str:
    """허락받을 때 말할 설명."""
    if action == "add_reminder":
        return f'add a reminder: "{args.get("text", "")}"'
    if action == "add_event":
        return f'add "{args.get("title", "")}" to your calendar at {args.get("start", "")}'
    if action == "quit_app":
        return f'quit {args.get("app", "")}'
    if action == "forget_fact":
        return f'delete memory number {args.get("number", "")}'
    shown = ", ".join(f"{k}: {v}" for k, v in args.items() if v not in ("", None, [], 0))[:80]
    return f"run {action.replace('_', ' ')}" + (f" ({shown})" if shown else "")


# 읽기만 하는 도구 (웹 결과를 읽은 뒤에도 그대로 바로 실행해도 안전)
READ_ONLY = {"lookup", "get_news", "get_quote", "today_events", "briefing", "list_alerts",
             "activity_report", "explain_last"}


def run_tool(name: str, args: dict, tainted: bool = False) -> str:
    """기본법 검사 -> (필요하면 허락) -> 실행. 결과를 AI에게 돌려줄 글자로.
    tainted: 이번 차례에 웹 결과를 읽었는지. 웹 페이지에 숨은 지시를 따르지 않게,
    그 뒤의 행동은 원래 바로 실행되는 것이라도 말로 허락받음 (기본법)."""
    if name not in LEVELS:  # 기본법: 목록에 없는 건 무조건 거부
        return "blocked: this action is not allowed"
    if name == "quit_app" and str(args.get("app", "")).strip().lower() in PROTECTED_APPS:
        return f"blocked: {args.get('app')} is a protected app (Jarvis runs on it)"
    if LEVELS[name] == "secure":    # 기본법: 말로는 안 되고 Touch ID로만
        what = describe(name, args)
        speak(f"I'm about to {what}. Please confirm with Touch ID.")
        hud.set("secure", text=what.capitalize())
        ok, why = touch_id(f"approve Jarvis: {what}")
        print(f"(Touch ID: {why})")
        if not ok:
            return f"denied: Touch ID {why}"
    elif tainted and LEVELS[name] == "auto" and name not in READ_ONLY:
        speak(f"I just read web pages, so let me double-check. I'm about to {describe(name, args)}. Shall I go ahead?")
        hud.set("confirm", text=describe(name, args).capitalize() + "? (after reading the web)")
        if not ask_permission():
            return "denied by user (asked because web content was read this turn)"
    elif LEVELS[name] == "confirm":
        speak(f"I'm about to {describe(name, args)}. Shall I go ahead?")
        hud.set("confirm", text=describe(name, args).capitalize() + "?")
        if not ask_permission():
            return "denied by user"
    try:
        before = capture_before(name, args)
        result = execute(name, args)
    except Exception as e:
        return f"failed: {e}"
    if result is not False and name != "undo_last":
        try:
            record_undo(name, args, before)
        except Exception as e:
            print(f"(되돌리기 정보 저장 실패: {e})")
    if result is False:
        return "failed"
    if result is True:
        return "done"
    return f"done: {result}"


CLAIMS_ACTION = re.compile(
    r"\b(turned|turning|opened|opening|playing|played|closed|closing|added|adding|paused|stopped|"
    r"started|muted|unmuted|undid|undone|reverted|lowered|raised|set the|cancelled|canceled|done)\b"
    r"|했어요|했습니다|켰어요|껐어요|열었|틀었|낮췄|올렸|멈췄|되돌렸|추가했",
    re.IGNORECASE)


UNDO_REQUEST = re.compile(r"되돌|취소|원래대로|undo|revert", re.IGNORECASE)


def run_agent(heard: str):
    """AI가 도구를 골라 쓰며 일을 끝까지 처리. (대답, 한 일 목록, 무시했는지)를 돌려줌."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n지금 시각: "
         + datetime.datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
         + f"\n지금 대답 언어: {S.reply_language}"
         + "\n\n" + memory_prompt()},
        *memory_messages(),
        {"role": "user", "content": heard},
    ]
    done = []
    nudged = False
    tainted = False        # 이번 차례에 웹 결과를 읽었는지
    wants_undo = bool(UNDO_REQUEST.search(heard))
    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=BRAIN_MODEL, messages=messages, tools=ordered_tools(), temperature=0.3)
        msg = response.choices[0].message
        if not msg.tool_calls:
            reply = (msg.content or "").strip()
            # 기본법: 도구를 안 썼는데 "했다"고 하거나, 되돌리기를 도구 없이 거절하면 -> 한 번 바로잡기
            if not done and not nudged and (CLAIMS_ACTION.search(reply) or wants_undo):
                nudged = True
                print(f"(거짓 보고 방지: 도구 없이 '{reply[:60]}'라고 해서 다시 판단)")
                messages.append({"role": "system", "content":
                    "You did not call any tool in this turn, so nothing has actually happened. "
                    "If the user wants an action (including undo, which must use undo_last), call the right tool now. "
                    "Otherwise answer without claiming you did anything."})
                continue
            return reply, done, False
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"(행동: {name} {args})")
            if name == "ignore":
                return "", done, True
            result = run_tool(name, args, tainted)
            if name == "lookup" and result.startswith("done"):
                tainted = True
            print(f"  -> {result[:120]}")
            done.append({"action": name, "args": args, "result": result[:80]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result[:3000]})
    return "Sorry, that got a bit complicated. Could you say that again?", done, False
