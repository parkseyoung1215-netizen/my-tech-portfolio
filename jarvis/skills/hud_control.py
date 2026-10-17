"""화면(HUD) 보이기 방식: 항상 / 부를 때만 / 끄기"""

from jarvis import hud
from jarvis.registry import register, _tool


def handle_hud_display(args):
    mode = str(args.get("mode", "")).strip()
    if mode not in hud.MODES:
        return False
    hud.set_mode(mode)
    return {"everywhere": "the screen orb now stays visible on every desktop, even over full-screen videos",
            "auto": "the screen orb is hidden and will only appear while the user is talking to Jarvis",
            "off": "the screen orb is turned off completely"}[mode]


# hud_display: 화면 원 보이기 방식 (보여주기만 하는 화면이라 바로 실행, 되돌리기 가능)
register(_tool("hud_display",
               "Jarvis 화면 원(HUD) 보이기 방식 바꾸기. "
               "'숨겨줘/치워줘/안 보이게' → auto (평소엔 숨고, 부를 때만 잠깐 나타남). "
               "'다시 보여줘/항상 띄워줘' → everywhere (유튜브·전체화면 위에도 항상). "
               "'화면 완전히 꺼줘' → off.",
               {"mode": {"type": "string", "enum": ["everywhere", "auto", "off"],
                         "description": "everywhere / auto / off"}}, ["mode"]),
         "auto", handle_hud_display)
