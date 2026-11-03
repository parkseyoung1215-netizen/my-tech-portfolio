# 실행 안내

프로젝트 루트에서 질문을 입력하거나 비교 실험을 실행합니다.

```bash
.venv/bin/python -B src/routing_runner.py
.venv/bin/python -B src/routing_runner.py --question '4 + 1'
.venv/bin/python -B -u src/routing_runner.py --compare
```

`--min-quality` 기본값은 0.8이며 `--max-latency`는 초 단위 평균 지연 상한입니다.
실험 데이터는 `--benchmark`로 지정할 수 있습니다. 기존 라우터의 산술식 인식 범위를 유지합니다.
적합한 모델이 없으면 답변 생성 없이 이유를 표시합니다. API 실패도 별도로 기록합니다.

비교는 기존 30문제를 두 방식 모두에 제공하고 문제마다 실행 순서를 교대합니다.
서버와 모델을 확인한 뒤 두 모델을 워밍업하며, 워밍업은 측정에서 제외합니다.
고정 방식은 정확도 조건과 관계없이 항상 1.5B를 호출합니다.
자동 방식은 라우터가 거절할 수 있습니다. 두 방식에 같은 프롬프트와 생성 옵션을 사용합니다.

상세·요약 CSV는 `data/routing_comparison_<고유값>.csv`,
`data/routing_summary_<고유값>.csv`로 배타적으로 생성합니다.
비용은 미측정이며 절감률을 추정하지 않습니다.

- `accuracy`: 전체 요청 기준 정답률. 거절과 API 오류도 분모에 포함합니다.
- `answered_accuracy`: 실제 응답을 받은 요청만의 정답률. 추출 실패는 오답입니다.
- `mean_request_seconds`: 거절·실패·라우팅을 포함한 평균 요청 처리 시간.
- `mean_generation_seconds`: 응답을 받은 Ollama 호출의 평균 시간.
- `total_processing_seconds`: 해당 방식의 요청 처리 시간 합계. 거절로 처리량이 달라질 수 있습니다.
- `model_selection_ratios`: 전체 요청 기준 모델 선택 비율. 미선택은 `no_selection`입니다.
- `experiment_seconds`: 두 방식 전체 실험 루프의 경과 시간. 워밍업과 CSV 저장은 제외합니다.

현재 비교 문제는 라우팅 기준을 산출한 기존 문제와 같습니다. 이는 실행 흐름 재현 실험이며,
독립된 새 문제에 대한 일반화 성능 검증이 아닙니다. 난이도별 관측 정확도는 개별 답변의 정답을 보장하지 않습니다.

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```
