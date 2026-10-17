"""자비스 화면(HUD) 프로그램

자비스 본체가 자동으로 띄움. 따로 실행해서 모양만 보고 싶으면:
    python hud/hud_app.py

- 맥의 '패널' 창으로 띄움: 모든 데스크탑, 그리고 다른 앱의 전체화면(유튜브 등) 위에도 뜰 수 있는 종류
  (v6.2까지는 일반 창이라 전체화면 위에는 못 들어갔음)
- 클릭해도 지금 보던 앱(유튜브 등)에서 포커스를 뺏지 않음
- 보이기 방식 3가지 (자비스에게 말로 바꿈)
    everywhere: 어디서든 항상
    auto:       평소엔 숨어 있다가 부를 때(듣기·생각·말하기·허락)만 나타남
    off:        완전히 숨김
- 자비스가 보내는 상태를 받아서 그리기만 함. 여기서는 아무것도 실행하지 않음
- 되돌리기 버튼을 누르면 자비스에게 "undo" 신호만 보냄 (실제로 할지는 자비스가 기본법대로 판단)
"""

import json
import os
import socket
import sys
import threading
import time

import AppKit
import Foundation
import WebKit
import objc
from PyObjCTools import AppHelper

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
POSITION = sys.argv[2] if len(sys.argv) > 2 else "bottom-right"
START_MODE = sys.argv[3] if len(sys.argv) > 3 else "everywhere"
WIDTH, HEIGHT = 340, 500
HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hud.html")

# 맥 창 설정값
ALL_SPACES = 1 << 0            # 모든 데스크탑(스페이스)에 보이기
STATIONARY = 1 << 4            # 데스크탑 넘길 때 같이 미끄러지지 않고 제자리
IGNORES_CYCLE = 1 << 6         # Cmd+` 창 전환에 안 끼어들기
FULLSCREEN_AUX = 1 << 8        # 다른 앱의 전체화면 위에도 보이기
BORDERLESS = 0
NONACTIVATING_PANEL = 1 << 7   # 눌러도 다른 앱의 포커스를 안 뺏는 패널

panel = None
web = None


def on_main(fn):
    """맥 창은 메인 스레드에서만 만질 수 있음."""
    AppHelper.callAfter(fn)


def run_js(code: str):
    on_main(lambda: web.evaluateJavaScript_completionHandler_(code, None))


def send_undo():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(b"undo", ("127.0.0.1", PORT + 1))
    s.close()


try:
    _PROTOCOLS = [objc.protocolNamed("WKScriptMessageHandler")]
except Exception:
    _PROTOCOLS = []


class Bridge(AppKit.NSObject, protocols=_PROTOCOLS):
    """화면(HTML)에서 오는 신호: 'undo'(되돌리기 버튼)와 끌어서 옮기기만 받음."""

    def userContentController_didReceiveScriptMessage_(self, controller, message):
        body = message.body()
        if body == "undo":
            send_undo()
            return
        try:
            dx, dy = float(body["dx"]), float(body["dy"])
        except Exception:
            return
        frame = panel.frame()
        panel.setFrameOrigin_((frame.origin.x + dx, frame.origin.y - dy))


def start_frame():
    screen = AppKit.NSScreen.mainScreen().visibleFrame()   # 맥 좌표: 왼쪽 아래가 (0,0)
    margin = 24
    left = screen.origin.x + margin
    right = screen.origin.x + screen.size.width - WIDTH - margin
    bottom = screen.origin.y + margin
    top = screen.origin.y + screen.size.height - HEIGHT - margin
    x = left if "left" in POSITION else right
    y = top if "top" in POSITION else bottom
    if POSITION == "center":
        x = screen.origin.x + (screen.size.width - WIDTH) / 2
        y = screen.origin.y + (screen.size.height - HEIGHT) / 2
    return ((x, y), (WIDTH, HEIGHT))


def build_window():
    global panel, web
    rect = start_frame()
    panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        rect, BORDERLESS | NONACTIVATING_PANEL, AppKit.NSBackingStoreBuffered, False)
    panel.setLevel_(AppKit.NSStatusWindowLevel)
    panel.setCollectionBehavior_(ALL_SPACES | STATIONARY | IGNORES_CYCLE | FULLSCREEN_AUX)
    panel.setFloatingPanel_(True)
    panel.setHidesOnDeactivate_(False)
    panel.setBecomesKeyOnlyIfNeeded_(True)
    panel.setOpaque_(False)
    panel.setHasShadow_(False)
    panel.setBackgroundColor_(AppKit.NSColor.clearColor())

    config = WebKit.WKWebViewConfiguration.alloc().init()
    controller = config.userContentController()
    controller.addScriptMessageHandler_name_(Bridge.alloc().init(), "jarvis")
    # 실제 화면 모드라는 표시 (투명 배경 + 시연 버튼 숨김)
    controller.addUserScript_(WebKit.WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_(
        "window.JARVIS_LIVE = true;", WebKit.WKUserScriptInjectionTimeAtDocumentStart, True))

    web = WebKit.WKWebView.alloc().initWithFrame_configuration_(((0, 0), (WIDTH, HEIGHT)), config)
    web.setValue_forKey_(False, "drawsBackground")          # 웹 화면 배경 투명
    panel.setContentView_(web)
    url = Foundation.NSURL.fileURLWithPath_(HTML)
    web.loadFileURL_allowingReadAccessToURL_(url, url.URLByDeletingLastPathComponent())


def show_window(visible: bool):
    def apply():
        if visible:
            panel.setCollectionBehavior_(ALL_SPACES | STATIONARY | IGNORES_CYCLE | FULLSCREEN_AUX)
            panel.orderFrontRegardless()
        else:
            panel.orderOut_(None)
    on_main(apply)


class Visibility:
    """보이기 방식(mode)과 자비스 상태에 따라 창을 보이거나 숨김."""

    def __init__(self, mode):
        self.mode = mode
        self.last = "idle"
        self.timer = None
        self.shown = None

    def _set(self, visible):
        if self.timer:
            self.timer.cancel()
            self.timer = None
        if visible != self.shown:
            self.shown = visible
            show_window(visible)

    def _hide_later(self, seconds):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(seconds, lambda: self._set(False))
        self.timer.daemon = True
        self.timer.start()

    def apply(self, state=None):
        prev, self.last = self.last, (state or self.last)
        if self.mode == "everywhere":
            self._set(True)
        elif self.mode == "off":
            self._set(False)
        elif self.last != "idle":          # auto: 대화 중엔 보이기
            self._set(True)
        elif self.shown is not False:      # auto: 대기로 돌아가면 잠시 뒤 숨기기 (결과 카드는 좀 더 오래)
            self._hide_later(10.5 if prev in ("done", "blocked") else 1.5)

    def set_mode(self, mode):
        if mode in ("everywhere", "auto", "off"):
            self.mode = mode
            self.shown = None              # 지금 상태와 상관없이 다시 적용
            self.apply()


def receive(vis):
    """자비스가 보내는 상태를 받아서 화면에 전달."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", PORT))
    while True:
        data, _ = sock.recvfrom(65536)
        try:
            msg = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if "level" in msg:
            run_js(f"jarvisHUD.level({float(msg['level'])})")
        elif "mode" in msg:
            vis.set_mode(str(msg["mode"]))
        elif "state" in msg:
            state = str(msg.pop("state"))
            run_js(f"jarvisHUD.set({json.dumps(state)}, {json.dumps(msg)})")
            vis.apply(state)


def watch_parent():
    """자비스 본체가 꺼지면 화면도 같이 꺼짐 (화면만 혼자 남지 않게)."""
    parent = os.getppid()
    while True:
        time.sleep(2)
        if os.getppid() != parent:
            on_main(lambda: AppKit.NSApp().terminate_(None))
            return


def main():
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)   # 독 아이콘 없음
    build_window()
    vis = Visibility(START_MODE)
    vis.apply("idle")
    print("(HUD: 패널 창으로 시작 — 모든 데스크탑·전체화면 위)", flush=True)
    threading.Thread(target=receive, args=(vis,), daemon=True).start()
    threading.Thread(target=watch_parent, daemon=True).start()
    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
