"""출처(provenance) 기반 방어.

모델에 들어간 모든 텍스트에 출처 라벨을 붙여 기록하고,
위험한 도구는 "컨텍스트에 신뢰 낮은 출처가 하나라도 섞여 있으면" 실행하지 않는다.

이 방어는 문장 내용을 보지 않는다. 오직 "누가 쓴 글이 모델 안에 들어와 있는가"만 본다.
"""
from __future__ import annotations

# 출처별 신뢰 수준 (클수록 믿을 수 있음)
SOURCE_TRUST = {"user": 2, "web": 0}

# 도구마다 요구되는 최소 신뢰 수준
TOOL_REQUIRED_TRUST = {"fetch_url": 0, "send_email": 2, "delete_file": 2}


class ProvenanceGuard:
    def __init__(self):
        self.sources: list[str] = []

    def add(self, source: str) -> None:
        """모델 컨텍스트에 이 출처의 텍스트가 들어왔다고 기록한다."""
        self.sources.append(source)

    @property
    def context_trust(self) -> int:
        """컨텍스트 전체의 신뢰 수준 = 섞여 들어온 출처 중 가장 낮은 것."""
        return min((SOURCE_TRUST[s] for s in self.sources), default=SOURCE_TRUST["user"])

    def check(self, tool: str) -> tuple[bool, str]:
        """(허용 여부, 이유)를 돌려준다. 모르는 도구는 가장 엄격하게 취급."""
        required = TOOL_REQUIRED_TRUST.get(tool, max(SOURCE_TRUST.values()))
        if self.context_trust >= required:
            return True, "ok"
        low = sorted({s for s in self.sources if SOURCE_TRUST[s] < required})
        return False, (
            f"{tool} requires trust {required}, but the context contains "
            f"lower-trust sources: {', '.join(low)}"
        )
