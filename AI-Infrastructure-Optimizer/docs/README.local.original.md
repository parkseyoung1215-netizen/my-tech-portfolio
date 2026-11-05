# AI Infrastructure Optimizer

AI 모델의 정확도, 응답 시간, 비용을 비교하는 실험용 프로젝트입니다.

## 실측 결과 기반 모델 선택

```python
from src.model_router import route_request

result = route_request("4 + 1", min_quality=0.8, max_latency=1.0)
print(result)
```

기본 데이터는 `data/benchmark_v2_f2be12734ce5444397f2ca02a62c1869.csv`입니다.
다른 실험은 `benchmark_path`로 명시합니다. CSV를 읽기만 하며 Ollama를 호출하지 않습니다.

- `min_quality`: 0~1 범위의 최소 정확도. `max_latency`: 평균 응답 시간 기준 초 단위 상한.
- `suitable=False`이면 선택하지 않으며 `reason`과 `assessments`에 판단 근거를 반환합니다.
- `quality`는 추출 실패를 포함한 전체 응답 중 정답 비율입니다.
- 난이도는 벤치마크와 같은 형태·숫자 범위의 산술식만 인식합니다.
  그 밖의 요청은 `uncertain=True`이며, 모든 난이도 중 최소 정확도와 최대 평균 응답 시간을 사용합니다.
  이는 일반 서술형 요청의 품질을 측정했다는 뜻이 아닙니다.
- CSV에 비용 정보가 없으므로 `cost=None`, `cost_status="unmeasured"`를 반환합니다.
  별도로 측정한 해당 요청의 예상 USD 비용이 있으면 `measured_costs={모델명: 비용}`으로 전달할 수 있습니다.
  모든 적합 후보의 비용이 알려진 경우 비용 우선, 동률이면 시간 순으로 선택합니다.
  하나라도 미측정이면 후보 전체를 응답 시간으로 비교합니다.
- 기존 `select_model(min_quality, max_latency)`와 `MODELS`는 가상 수치를 사용하는 데모 호환용입니다.
  실측 기반 선택에는 `route_request()`를 사용합니다.

현재 데이터는 모델별 30개 산술 문제의 1회 측정입니다. 관측 정확도는 미래 요청의 정확도 보장이 아니며,
CSV에 없는 요청 실패나 반복 실험의 변동성은 반영하지 않습니다.

## 테스트

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

테스트는 Ollama를 호출하지 않으며 기존 CSV를 변경하지 않습니다.
