#!/bin/bash
# 자비스 자동 실행 파일
# 더블클릭하거나, 로그인 항목에 넣어두면 맥이 켜질 때 자비스가 자동으로 대기 상태로 켜짐

cd "$(dirname "$0")" || exit 1
source venv/bin/activate
python assistant.py
