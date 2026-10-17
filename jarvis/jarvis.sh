#!/bin/bash
# 자비스를 창 없이 뒤에서 돌리는 스위치 (v3: AppleScript로 만든 진짜 맥 앱 'Jarvis.app'으로 감싸서 마이크 권한을 받음)
#   ./jarvis.sh on     : 켜기 + 맥에 로그인할 때마다 자동으로 켜지게 등록
#   ./jarvis.sh off    : 끄기 + 자동 실행 해제
#   ./jarvis.sh status : 지금 켜져 있는지 확인
#   ./jarvis.sh log    : 자비스가 듣고 판단한 기록을 실시간으로 보기 (Ctrl + C로 나가기)

DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$DIR/Jarvis.app"
RUNNER="$DIR/.jarvis_run.sh"
LOG="$HOME/Library/Logs/jarvis.log"
OLD_PLIST="$HOME/Library/LaunchAgents/com.jarvis.assistant.plist"

stop_all() {
  # 예전 방식(launchd)으로 켠 자비스가 있으면 정리
  if [ -f "$OLD_PLIST" ]; then
    launchctl bootout "gui/$(id -u)" "$OLD_PLIST" 2>/dev/null
    rm -f "$OLD_PLIST"
  fi
  pkill -f "$DIR/assistant.py" 2>/dev/null
  pkill -f "Jarvis.app/Contents/MacOS" 2>/dev/null
  sleep 1
}

build_app() {
  # 1) 파이썬을 켜는 작은 실행 파일
  {
    echo '#!/bin/bash'
    echo "cd \"$DIR\" || exit 1"
    echo 'if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then'
    echo "  /usr/bin/arch -arm64 \"$DIR/venv/bin/python\" -u \"$DIR/assistant.py\" >> \"$LOG\" 2>&1"
    echo 'else'
    echo "  \"$DIR/venv/bin/python\" -u \"$DIR/assistant.py\" >> \"$LOG\" 2>&1"
    echo 'fi'
  } > "$RUNNER"
  chmod +x "$RUNNER"

  # 2) AppleScript로 진짜 맥 앱을 만들고, 그 앱이 1)을 '자식'으로 실행
  #    -> 맥이 마이크 권한을 파이썬이 아니라 'Jarvis' 앱 이름으로 물어봄
  rm -rf "$APP"
  osacompile -o "$APP" -e "do shell script quoted form of \"$RUNNER\"" || return 1

  # 3) 앱 정보: Dock에 안 보이게, 마이크·카메라 사용 이유 적기
  PL="$APP/Contents/Info.plist"
  plutil -replace CFBundleIdentifier -string "com.jarvis.assistant" "$PL"
  plutil -replace CFBundleName -string "Jarvis" "$PL"
  plutil -replace LSUIElement -bool true "$PL"
  plutil -replace NSMicrophoneUsageDescription -string "자비스가 '헤이 자비스'와 명령을 듣기 위해 마이크가 필요해요." "$PL"
  plutil -replace NSCameraUsageDescription -string "자비스 손 리모컨·손 모드가 손짓을 보기 위해 카메라가 필요해요. 영상은 저장하거나 보내지 않아요." "$PL"
  plutil -replace NSAppleEventsUsageDescription -string "자비스가 음악, 캘린더, 브라우저를 조종하기 위해 필요해요." "$PL"
  codesign --force --deep -s - "$APP" 2>/dev/null
}

case "$1" in
  on)
    mkdir -p "$HOME/Library/Logs"
    stop_all
    : > "$LOG"   # 예전 기록 지우고 새로 시작
    build_app || { echo "앱 만들기에 실패했어요."; exit 1; }
    # 예전에 꼬인 마이크 권한 기록이 있으면 초기화 (허용 창이 다시 뜨게)
    tccutil reset Microphone com.jarvis.assistant >/dev/null 2>&1
    # 로그인할 때 자동 실행 (중복 등록 방지 후 등록)
    osascript -e 'tell application "System Events" to delete (every login item whose name is "Jarvis")' >/dev/null 2>&1
    osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$APP\", hidden:true}" >/dev/null 2>&1
    open "$APP"
    echo "자비스를 켰어요. (창 없음, 로그인할 때마다 자동 실행)"
    echo "'Jarvis가 마이크에 접근하려고 합니다' 창이 뜨면 20초 안에 [허용]을 눌러주세요."
    ;;
  off)
    stop_all
    osascript -e 'tell application "System Events" to delete (every login item whose name is "Jarvis")' >/dev/null 2>&1
    echo "자비스를 끄고 자동 실행도 해제했어요."
    ;;
  status)
    if pgrep -f "$DIR/assistant.py" >/dev/null; then
      echo "자비스: 켜져 있음 (대기 중)"
    else
      echo "자비스: 꺼져 있음"
    fi
    ;;
  log)
    touch "$LOG"
    tail -n 30 -f "$LOG"
    ;;
  *)
    echo "사용법: ./jarvis.sh on | off | status | log"
    ;;
esac
