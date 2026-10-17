"""손 리모컨: 어떤 앱 위에서든 손짓으로 재생·멈춤, 다음·이전, 볼륨

자비스 본체가 "손 리모컨 켜줘"라고 하면 따로 띄움. 점검만 하고 싶으면:
    python hud/hand_remote.py --check      (손짓을 인식하면 화면에 글자로만 보여주고 아무것도 안 함)

손짓 (카메라에서 1~2m 정도)
    손바닥 펴고 0.6초 가만히            → 재생/멈춤
    볼륨 다이얼 (v6.10)
      1) 검지 펴고 엄지를 크게 벌려 ㄴ자(L자), 나머지 세 손가락 접기 → 0.35초 유지하면 켜짐
      2) 그대로 손을 내 기준 오른쪽으로 → 볼륨 올라감 / 왼쪽으로 → 내려감
      3) 엄지·검지를 완전히 붙이면 → 그 볼륨으로 끝
    (옆으로 휙 → 다음/이전은 쓸 데가 적어서 꺼둠. SWIPE_ENABLED = True로 켤 수 있음)

카메라 기본법
- 이 프로그램이 켜져 있는 동안만 카메라를 씀 (맥 초록 불이 켜짐). 끄면 카메라도 꺼짐
- 영상은 한 장씩 손 위치만 읽고 바로 버림. 저장·전송 없음. 손 인식은 맥에 들어 있는 Apple Vision으로 노트북 안에서
- 손짓으로 하는 일은 자비스에게 '손짓 이름'만 보내고, 실제 실행은 자비스가 정해진 목록(재생·다음·이전·볼륨)에서만 함
- 손이 오래 안 보이면(기본 30분) 스스로 꺼짐. 자비스가 꺼져도 2초 안에 같이 꺼짐
"""

import json
import math
import os
import socket
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8767
CHECK = "--check" in sys.argv
IDLE_OFF_SECONDS = 30 * 60
FPS = 15
SWIPE_ENABLED = False      # 옆으로 휙 → 다음/이전 (v6.9에서 꺼둠)


# =========================================================
# 손짓 판단 (카메라·맥과 상관없는 순수 계산 → 따로 점검 가능)
# 점 좌표: 0~1, 왼쪽 위가 (0,0). 카메라 원본 기준(거울 아님)
# =========================================================
FINGERS = (("index_tip", "index_pip"), ("middle_tip", "middle_pip"),
           ("ring_tip", "ring_pip"), ("little_tip", "little_pip"))


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def read_hand(p: dict) -> dict:
    """점 몇 개로 손 모양 읽기: 손 크기, 집기 정도, 편 손가락 수, 손 중심."""
    wrist = p["wrist"]
    size = _d(wrist, p["middle_mcp"]) or 1e-6
    extended = sum(1 for tip, pip in FINGERS if _d(p[tip], wrist) > _d(p[pip], wrist) * 1.15)
    pinch = _d(p["thumb_tip"], p["index_tip"]) / size
    # 나머지 세 손가락(가운데·약지·새끼)이 펴졌는지: OK 사인과 주먹을 구분하는 데 씀
    others = sum(1 for tip, pip in FINGERS[1:] if _d(p[tip], wrist) > _d(p[pip], wrist) * 1.15)
    index_ext = _d(p["index_tip"], wrist) > _d(p["index_pip"], wrist) * 1.15
    center = ((wrist[0] + p["middle_mcp"][0]) / 2, (wrist[1] + p["middle_mcp"][1]) / 2)
    return {"size": size, "pinch": pinch, "extended": extended, "others": others,
            "index_ext": index_ext, "center": center}


class GestureEngine:
    PALM_HOLD = 0.6        # 손바닥을 이만큼 가만히 → 재생/멈춤
    PALM_STILL = 0.06      # 이만큼보다 덜 움직여야 '가만히'
    SWIPE_DIST = 0.22      # 화면 폭의 22% 이상 옆으로
    SWIPE_TIME = 0.45      # 이 시간 안에
    VOL_STEP = 0.06        # 다이얼이 켜진 채 손을 이만큼 옆으로 움직일 때마다 볼륨 한 칸
    DIAL_ARM = 0.35        # 다이얼 준비 모양을 이만큼 유지해야 켜짐
    DIAL_STILL = 0.05      # 준비하는 동안 손이 이만큼보다 덜 움직여야 함
    LOST_GRACE = 0.5       # 손이 잠깐 안 잡혀도 이 시간까지는 다이얼 유지
    COOLDOWN = 1.2

    def __init__(self):
        self.palm_since = None
        self.palm_anchor = None
        self.palm_latched = False      # 한 번 멈춘 뒤 손을 내리기 전엔 또 안 함
        self.track = []                # (시간, x, y) 최근 손 중심
        self.dial = None               # None / {"phase": "arming"|"active", ...}
        self.dial_latched = False      # 붙여서 끝낸 뒤, 손 모양을 한 번 풀기 전엔 다시 안 켜짐
        self.closed_frames = 0
        self.away_since = None
        self.last_seen = 0.0
        self.last_vol = 0.0
        self.cooldown_until = 0.0

    def dial_volume(self, h, cx, cy, now, events) -> bool:
        """볼륨 다이얼. 다이얼이 켜져 있거나 준비 중이면 True (다른 손짓은 건너뜀)."""
        # v6.11: 켤 때는 엄지·검지를 크게 벌린 L자 (힘 뺀 손·검지만 편 손과 확실히 구분)
        ready = h["index_ext"] and h["others"] <= 1 and h["pinch"] >= 0.55
        closed = h["pinch"] < 0.25        # 켤 때와 차이가 커서 붙이기 기준을 너그럽게 (카메라 오차 대비)
        if self.dial and self.dial["phase"] == "active":
            # 3) 엄지·검지 완전히 붙이기 → 끝 (두 장면 연속이어야: 순간적인 오인식 거르기)
            self.closed_frames = self.closed_frames + 1 if closed else 0
            if self.closed_frames >= 2:
                self.dial, self.dial_latched, self.closed_frames = None, True, 0
                events.append("dial_done")
                return True
            # 손바닥을 펴거나 손 모양이 완전히 바뀌면 0.5초 뒤 그냥 끝 (그때까지 바꾼 볼륨은 유지)
            if h["extended"] >= 3 or (not h["index_ext"] and not closed):
                self.away_since = self.away_since or now
                if now - self.away_since > self.LOST_GRACE:
                    self.dial = None
                    events.append("dial_done")
                return True
            self.away_since = None
            # 2) 옆으로 → 볼륨. 카메라는 나를 마주 봐서 좌우 반대: 내 기준 오른쪽 = 화면 x 감소
            moved = self.dial["x"] - cx
            if abs(moved) >= self.VOL_STEP and now - self.last_vol >= 0.1:
                events.append("volume_up" if moved > 0 else "volume_down")
                self.dial["x"] -= math.copysign(self.VOL_STEP, moved)
                self.last_vol = now
            return True
        if not ready and not closed:
            self.dial_latched = False                      # 손 모양을 풀었으니 다시 켤 수 있음
        if not ready or self.dial_latched:
            self.dial = None
            return False
        # 1) 준비 모양을 가만히 0.35초 → 켜짐
        if not self.dial or _d(self.dial["anchor"], (cx, cy)) > self.DIAL_STILL:
            self.dial = {"phase": "arming", "since": now, "anchor": (cx, cy)}
        elif now - self.dial["since"] >= self.DIAL_ARM:
            self.dial = {"phase": "active", "x": cx}
            self.closed_frames, self.away_since = 0, None
            events.append("dial_start")
        return True

    def update(self, hand, now: float) -> list:
        """한 장면의 손(없으면 None) → 일어난 손짓 목록."""
        events = []
        if hand is None:
            self.palm_since = None
            self.palm_latched = False
            self.track.clear()
            if now - self.last_seen > self.LOST_GRACE:      # 손이 잠깐 안 잡힌 건 봐줌
                self.dial = None
                self.dial_latched = False
            return events
        self.last_seen = now
        h = read_hand(hand)
        cx, cy = h["center"]
        if self.dial_volume(h, cx, cy, now, events):
            return events

        # --- 옆으로 휙 → 다음/이전 ---
        self.track.append((now, cx, cy))
        self.track = [t for t in self.track if now - t[0] <= self.SWIPE_TIME]
        if SWIPE_ENABLED and now >= self.cooldown_until and h["extended"] >= 2 and len(self.track) >= 3:
            dx = cx - self.track[0][1]
            dy = cy - self.track[0][2]
            if abs(dx) >= self.SWIPE_DIST and abs(dy) < abs(dx) * 0.5:
                # 카메라는 나를 마주 보고 있어서 좌우가 반대: 내 기준 오른쪽으로 휙 = 화면 좌표 x 감소
                events.append("next" if dx < 0 else "previous")
                self.cooldown_until = now + self.COOLDOWN
                self.track.clear()
                self.palm_since = None
                return events

        # --- 손바닥 펴고 가만히 → 재생/멈춤 ---
        open_palm = h["extended"] == 4 and h["pinch"] > 0.6
        if not open_palm:
            self.palm_since = None
            self.palm_latched = False
            return events
        if self.palm_since is None or _d(self.palm_anchor, (cx, cy)) > self.PALM_STILL:
            self.palm_since, self.palm_anchor = now, (cx, cy)
        elif (not self.palm_latched and now >= self.cooldown_until
              and now - self.palm_since >= self.PALM_HOLD):
            events.append("playpause")
            self.palm_latched = True
            self.cooldown_until = now + self.COOLDOWN
        return events


# =========================================================
# 카메라 + Apple Vision (맥에서만)
# =========================================================
JOINTS = {
    "wrist": "VNHumanHandPoseObservationJointNameWrist",
    "thumb_tip": "VNHumanHandPoseObservationJointNameThumbTip",
    "index_tip": "VNHumanHandPoseObservationJointNameIndexTip",
    "index_pip": "VNHumanHandPoseObservationJointNameIndexPIP",
    "middle_tip": "VNHumanHandPoseObservationJointNameMiddleTip",
    "middle_pip": "VNHumanHandPoseObservationJointNameMiddlePIP",
    "middle_mcp": "VNHumanHandPoseObservationJointNameMiddleMCP",
    "ring_tip": "VNHumanHandPoseObservationJointNameRingTip",
    "ring_pip": "VNHumanHandPoseObservationJointNameRingPIP",
    "little_tip": "VNHumanHandPoseObservationJointNameLittleTip",
    "little_pip": "VNHumanHandPoseObservationJointNameLittlePIP",
}


def find_hand(Vision, Foundation, cv2, frame):
    """사진 한 장에서 가장 큰 손 하나의 점들. 없으면 None."""
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    data = Foundation.NSData.dataWithBytes_length_(jpg.tobytes(), len(jpg))
    request = Vision.VNDetectHumanHandPoseRequest.alloc().init()
    request.setMaximumHandCount_(1)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    success, _ = handler.performRequests_error_([request], None)
    if not success or not request.results():
        return None
    observation = request.results()[0]
    points = {}
    for key, const in JOINTS.items():
        point, _ = observation.recognizedPointForJointName_error_(getattr(Vision, const), None)
        if point is None or point.confidence() < 0.3:
            return None
        loc = point.location()
        points[key] = (loc.x, 1.0 - loc.y)     # Vision은 왼쪽 아래가 (0,0) → 위아래 뒤집기
    return points


def main():
    try:
        import cv2
        import Foundation
        import Vision
    except ImportError as e:
        print(f"(손 리모컨: 부품이 없어요 - {e}. pip install opencv-python pyobjc-framework-Vision)", flush=True)
        sys.exit(1)
    camera = cv2.VideoCapture(0)
    for _ in range(15):                 # 처음엔 권한 창에 '허용'을 누를 시간을 줌 (최대 15초)
        if camera.isOpened():
            break
        time.sleep(1)
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("(손 리모컨: 카메라를 열 수 없어요. 카메라 권한을 확인하세요)", flush=True)
        sys.exit(1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    engine = GestureEngine()
    parent = os.getppid()
    last_hand = time.time()
    last_print = 0.0
    print("(손 리모컨 켜짐 — 손바닥 0.6초: 재생/멈춤 / 볼륨 다이얼: 엄지·검지 크게 벌려 L자 0.35초 → 좌우로 조절 → 엄지·검지 붙이면 끝)", flush=True)
    try:
        while True:
            start = time.time()
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.2)
                continue
            hand = find_hand(Vision, Foundation, cv2, frame)
            del frame                                   # 영상은 바로 버림
            now = time.time()
            if hand:
                last_hand = now
            if CHECK and hand and now - last_print > 0.5:   # 점검 모드: 손 모양을 숫자로 보여줘서 왜 안 되는지 알 수 있게
                last_print = now
                h = read_hand(hand)
                shape = ("L자(다이얼 켜기 모양)" if h["index_ext"] and h["others"] <= 1 and h["pinch"] >= 0.55
                         else "붙임" if h["pinch"] < 0.25 else "손바닥" if h["extended"] == 4 else "기타")
                print(f"  [손] 엄지·검지 벌림 {h['pinch']:.2f} (L자는 0.55 이상, 붙임은 0.25 미만) | "
                      f"검지 {'폄' if h['index_ext'] else '굽음'} | 나머지 편 손가락 {h['others']}개 → {shape}", flush=True)
            for event in engine.update(hand, now):
                if CHECK:
                    print(f"  손짓: {event}", flush=True)
                else:
                    out.sendto(json.dumps({"gesture": event}).encode(), ("127.0.0.1", PORT))
            if now - last_hand > IDLE_OFF_SECONDS:
                print("(손 리모컨: 손이 오래 안 보여서 꺼요)", flush=True)
                out.sendto(json.dumps({"gesture": "_auto_off"}).encode(), ("127.0.0.1", PORT))
                break
            if not CHECK and os.getppid() != parent:   # 자비스가 꺼짐
                break
            time.sleep(max(0.0, 1 / FPS - (time.time() - start)))
    finally:
        camera.release()                                # 카메라 끄기 (초록 불 꺼짐)


if __name__ == "__main__":
    main()
