# ACS 실행·재현 안내

현재 실험: `data/acs_16a7001279c947c0acb730672deb1a97/`.

```bash
# 중단 후 같은 실험 재개: 완료된 요청은 다시 모델로 실행하지 않음
.alsr-venv/bin/python -B -u -m src.acs_experiment data/acs_16a7001279c947c0acb730672deb1a97
# 고정된 문제/설정으로 새 폴더에서 실제 모델 전체 재실행
.alsr-venv/bin/python -B -u -m src.acs_replay data/acs_16a7001279c947c0acb730672deb1a97/MANIFEST.json
# 완료된 체크포인트 분석
.venv/bin/python -B -m src.acs_analysis <실행 폴더>
.venv/bin/python -B -m src.acs_content_audit <생성된 ANALYSIS 폴더>
.venv/bin/python -B -m src.acs_report <생성된 ANALYSIS 폴더> "<실제 테스트 결과>"
```

재현은 기존 고정 평가 데이터를 다시 쓰는 것이며 새 미사용 문제 실험이 아니다. 최초 생성 명령은 기록된 문제와 중복되면 중단한다. 프로파일·코어 소스·패키지가 달라지면 재개를 거부한다. 새 연구 버전에서 명시적으로 분기해야 하며 기존 결과를 지워 재실행하지 않는다.

모델은 기존 로컬 Qwen2.5 1.5B BF16 고정 체크포인트다. Metal 권한이 필요하고 유료 API는 호출하지 않는다. 요청별 JSON/fsync 체크포인트와 새 EXPORT CSV를 사용한다. 동시 재개는 파일 lock으로 차단한다.

내용 정확도·형식 준수율·전체 과제 성공률을 별도로 본다. 문장 약어/사실 보존을 검사하는 새 점수와 기존 점수를 나란히 저장한다. 실제 token probability는 정답/완성 confidence가 아니다. EOS mass는 현재 token 이전의 경쟁 확률이지 미래 EOS 예측이 아니다. 비용은 실제 wall/CPU와 추정 model matmul FLOPs를 구분한다.

추가 감사는 doctor/physician 동의어 누락으로 가려진 사실 손실을 별도 CSV에 기록한다. 사후의 제한적인 사실 보존 검사이며 완전한 의미 정확도/추가 사실 없는 충실도 판정이 아니다. 원점수·통계·정책·모델 답변을 변경하지 않는다.
