# Stable1 Runtime 실험 — 재개 안내

현재 실행 폴더:
`data/stable_runtime_7437bcc4afc44d6d958ba98ed2f41628/`

```bash
# GPU를 호출하지 않는 상태 확인
.venv/bin/python -B -m src.stable_runtime_status data/stable_runtime_7437bcc4afc44d6d958ba98ed2f41628
# 중단된 실험을 동일 코드·패키지·모델·문제로 재개
.alsr-venv/bin/python -B -u -m src.stable_runtime_experiment data/stable_runtime_7437bcc4afc44d6d958ba98ed2f41628
```

완료된 요청은 다시 모델로 실행하지 않는다. 각 요청의 JSON은 새 attempt 파일로 저장하며 불완전한 파일도 보존한다. 동시 실행은 파일 lock으로 막는다. 소스나 패키지 hash가 달라지면 같은 실험의 재개를 거부한다. 완료 후 재개는 모델 재측정이 아니라 새 EXPORT 파일 생성이다. 누적 기록 수를 새로운 모델 실행 수로 해석하지 않는다.

새 실험 생성 명령은 이미 기록된 문제와 중복되면 중단한다. 이번 3회 반복을 다시 실행하고 싶다면 기존 결과를 지우는 대신, 명시적인 별도 실험 버전과 데이터 재사용 표기가 필요하다.

분석은 완료된 체크포인트에서 새 ANALYSIS 폴더를 생성한다.

```bash
.venv/bin/python -B -m src.stable_runtime_analysis data/stable_runtime_7437bcc4afc44d6d958ba98ed2f41628
.venv/bin/python -B -m src.stable_runtime_report <새 ANALYSIS 폴더> "<실제 테스트 결과>"
```

`legacy`와 `lean`은 같은 출력 종료 조건에서 비교한다. `lean_option`은 세 정책에 동일한 명시적 선택지 완료 규칙을 적용한 별도 조건이다. 그 속도 개선을 계층 생략만의 효과로 해석하지 않는다. 상세 native 호출 계측은 별도 profile이며 primary latency에 포함하지 않는다. 모델은 무료 로컬 HF 고정 revision을 사용하며 유료 API는 호출하지 않는다.
