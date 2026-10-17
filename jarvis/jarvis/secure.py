"""Touch ID 허락 (기본법: 중요한 행동은 말이 아니라 내 손가락으로)

- 말로 받는 허락은 옆 사람이나 유튜브 소리가 "yes"라고 해도 통과될 수 있음
- 그래서 지우기처럼 되돌리기 어려운 행동은 Touch ID(또는 Mac 암호)로만 허락
- 확인이 안 되면 무조건 거절 (부품이 없거나, 시간이 지나거나, 취소하거나, 에러가 나도 전부 거절)
"""

import threading

from config import *


def touch_id(reason: str) -> tuple:
    """맥의 Touch ID 창을 띄워서 확인. (허락됐는지, 이유)를 돌려줌."""
    try:
        import LocalAuthentication as LA
    except ImportError:
        return False, "Touch ID part is not installed (pip install pyobjc-framework-LocalAuthentication)"
    try:
        context = LA.LAContext.alloc().init()
        # 지문, 지문이 안 되면(노트북 덮개를 닫고 쓸 때 등) Mac 로그인 암호
        policy = LA.LAPolicyDeviceOwnerAuthentication
        can, error = context.canEvaluatePolicy_error_(policy, None)
        if not can:
            return False, f"Touch ID is not available ({error})"
        finished = threading.Event()
        result = {"ok": False, "why": "no answer"}

        def reply(success, err):
            result["ok"] = bool(success)
            result["why"] = "confirmed" if success else f"not confirmed ({err.localizedDescription() if err else 'cancelled'})"
            finished.set()

        context.evaluatePolicy_localizedReason_reply_(policy, reason, reply)
        if not finished.wait(SECURE_TIMEOUT):
            context.invalidate()          # 창 닫기
            return False, f"no fingerprint within {SECURE_TIMEOUT} seconds"
        return result["ok"], result["why"]
    except Exception as e:
        return False, f"Touch ID error ({e})"
