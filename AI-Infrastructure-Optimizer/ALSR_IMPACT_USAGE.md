# ALSR 계층 영향 연구 실행 안내

주 결과는 `data/alsr_impact_e8eea7cd19db4d8796ea54cfe6f57b2c/CANONICAL_aca75ae9b3014233ba48c949c0186bb4/`에 저장한다. 상위 폴더는 최초 실행의 원시 로그다. 어느 쪽도 삭제하거나 덮어쓰지 않는다.

최초 진단에서 발견한 BF16 동점 정렬 문제는 `src/alsr_impact_ties.py`가 실제 greedy argmax 규칙으로 수정한다. `alsr_impact_run` 및 초기 `alsr_impact_replay`는 원래 실행을 보존한 진입점이다. 재현할 때는 **수정된 진입점**을 사용한다.

```bash
# 고정 학습/보정/평가를 명시적으로 재사용해 모든 진단과 생성을 실제 재계산
.alsr-venv/bin/python -B -u -m src.alsr_impact_fixed_run data/alsr_impact_e8eea7cd19db4d8796ea54cfe6f57b2c/MANIFEST.json
.alsr-venv/bin/python -B -m src.alsr_impact_analysis <새 출력 폴더>
.venv/bin/python -B -m src.alsr_impact_cpu <새 출력 폴더>
.alsr-venv/bin/python -B -u -m src.alsr_impact_smoke <새 출력 폴더>
.venv/bin/python -B -m src.alsr_impact_report <새 출력 폴더> "<실제 테스트 결과>"
```

이 명령은 새 미사용 평가셋 실험이 아니라 저장된 데이터의 재현이다. 모델은 기존 로컬 HF 고정 revision을 사용하며 유료 API를 호출하지 않는다. Metal 권한이 필요하다. CSV/JSON/보고서는 기존 파일이 있으면 중단한다.

원래 실험과 수정 후 실험의 시간을 합쳐 알고리즘 절감률을 계산하지 않는다. 최종 평균에는 수정 후 동일 조건의 세 반복만 사용하고, 수정 전후 표는 감사 자료로 별도 제공한다. 강제 임계값을 사용하는 gate 분기 smoke는 소프트웨어 검증이며 평가 성능으로 사용하지 않는다.
