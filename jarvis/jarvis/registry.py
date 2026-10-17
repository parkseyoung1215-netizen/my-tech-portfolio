"""도구 등록부: 기능(스킬)마다 도구를 여기에 등록한다. 등록 안 된 도구는 AI가 쓸 방법이 없다 (기본법)

허락 수준 3단계
  auto    바로 실행
  confirm 말로 허락 ("yes")
  secure  Touch ID로만 허락 (말로는 안 됨)
"""

from config import SECURE_TOOLS

LEVEL_RANK = {"auto": 0, "confirm": 1, "secure": 2}


TOOLS = []      # AI에게 보여줄 도구 설명
LEVELS = {}     # 도구 이름 -> "auto" / "confirm" / "secure"
HANDLERS = {}   # 도구 이름 -> 실행 함수

# AI에게 보여줄 순서 (원래 순서 유지)
TOOL_ORDER = ['ignore', 'open_app', 'open_url', 'web_search', 'youtube_play', 'youtube_search', 'spotify_play', 'music', 'set_volume', 'jarvis_voice', 'set_timer', 'today_events', 'get_news', 'open_article', 'activity_report', 'explain_last', 'undo_last', 'remember_fact', 'forget_fact', 'get_quote', 'set_reply_language', 'briefing', 'close_tab', 'quit_app', 'add_reminder', 'add_event']


def _tool(name, desc, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {},
                       "required": required or [], "additionalProperties": False}}}


def _s(desc):
    return {"type": "string", "description": desc}


def _i(desc):
    return {"type": "integer", "description": desc}


def register(tool: dict, level: str, handler) -> None:
    name = tool["function"]["name"]
    assert level in LEVEL_RANK, f"{name}: level must be auto, confirm or secure"
    if name in SECURE_TOOLS:          # 설정에서 Touch ID로 올린 도구 (올리기만 하고 내리지는 않음)
        level = "secure"
    TOOLS.append(tool)
    LEVELS[name] = level
    HANDLERS[name] = handler


def ordered_tools() -> list:
    rank = {n: i for i, n in enumerate(TOOL_ORDER)}
    return sorted(TOOLS, key=lambda t: rank.get(t["function"]["name"], len(rank)))


def execute(action: str, args: dict):
    """도구 실행. 성공하면 True/결과 글자, 실패하면 False."""
    handler = HANDLERS.get(action)
    if handler is None:
        return False
    return handler(args)
