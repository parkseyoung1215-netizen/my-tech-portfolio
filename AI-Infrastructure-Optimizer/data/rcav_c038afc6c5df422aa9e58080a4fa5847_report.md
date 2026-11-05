# Risk-Cost Adaptive Verification (RCAV)

실행 ID `c038afc6c5df422aa9e58080a4fa5847`. 실제 로컬 모델 실행. 모든 값은 평균 [최소, 최대].

## 사전 설계

새 보정100문제와 새 평가150문제(5유형×30)를 과거 질문 줄기·교환 산술·선택지 재배열 중복과 분리했다. 세 번 같은150문제를 반복했으며 독립450문제로 보지 않는다. 각 모델은 별도로 보정했고 평가 전에 프로필을 고정했다. 온도0,seed42,출력64토큰,동일 해결/검증 프롬프트·내용 파서·산술 도구를 적용했다.

RCAV는 문제를 해결한 뒤 검증 전에 요청 유형·난이도·답변 해석 상태로 오류위험p를 추정한다. 보정의 오답 수정확률r와 정답 악화확률h를 사용해 예상 순오류 감소 Δ=p×r−(1−p)×h를 계산한다. 유형/형식 층화 표본8개 미만은 유형10개 이상/전체 통계로 후퇴하고, 전역Beta(1,1)과 사전강도10의 경험적 수축을 사용한다. 해석 불능은 유효답 실패위험1, 안전 산술 도구 결과는 위험0으로 다룬다. 이러한 수치는 확률 보장/모델 자체 confidence가 아니다.

검증 입력 문자수와 보정에서 실제 측정한 prefill+generation초를 비음수 기울기 선형회귀로 연결한다. 예측값은 보정 최소 관측시간으로 하한을 두며 비용 기준Cref는 보정 중앙값이다. Δ>λ×예상시간/Cref일 때만 검증한다. 주 RCAV λ=0.05, 사전 감도분석 λ=0와0.15를 모두 실행·공개했다. 평가로 λ를 고르거나 조정하지 않았다.

Baseline은 전수 검증하는 연구용 대조군, Selective는 기존 최소20보정·쉬운 유형·수정≤악화이면 검증 생략 규칙 그대로다. 같은 새 보정100개로 각 모델의 Selective 통계도 구했으므로 과거 프로필의 수치를 그대로 쓴 것은 아니다. 원본 기존 코드/실험은 변경하지 않았다. 검증 실패 시 원래 답을 유지하며 추가 재시도는 없다.

## 전체 측정

| 모델 | 방식 | 내용 정확도 % | 답변율 % | 호출 수 | 입력/출력 토큰 | 처리시간(초) | 모델 처리시간(초) |
|---|---|---:|---:|---:|---|---|---|
| qwen2.5:1.5b | baseline | 62.67 [62.67, 62.67] | 100.00 [100.00, 100.00] | 270 [270, 270] | 28618 [28618, 28618]/652 [652, 652] | 18.526 [17.672, 19.028] | 17.354 [16.572, 17.791] |
| qwen2.5:1.5b | selective | 73.33 [73.33, 73.33] | 100.00 [100.00, 100.00] | 180 [180, 180] | 19543 [19543, 19543]/361 [361, 361] | 13.474 [13.126, 13.988] | 12.514 [12.198, 13.008] |
| qwen2.5:1.5b | rcav_zero | 74.67 [74.67, 74.67] | 100.00 [100.00, 100.00] | 150 [150, 150] | 15535 [15535, 15535]/300 [300, 300] | 9.191 [8.715, 9.892] | 8.415 [7.958, 9.082] |
| qwen2.5:1.5b | rcav | 75.33 [75.33, 75.33] | 100.00 [100.00, 100.00] | 120 [120, 120] | 11707 [11707, 11707]/240 [240, 241] | 5.588 [5.149, 6.404] | 5.006 [4.586, 5.811] |
| qwen2.5:1.5b | rcav_high | 75.33 [75.33, 75.33] | 100.00 [100.00, 100.00] | 120 [120, 120] | 11707 [11707, 11707]/240 [240, 240] | 4.621 [4.160, 5.333] | 4.113 [3.665, 4.794] |
| qwen2.5:0.5b | baseline | 48.00 [48.00, 48.00] | 98.00 [98.00, 98.00] | 270 [270, 270] | 29088 [29085, 29094]/1227 [1224, 1234] | 11.319 [11.243, 11.458] | 10.444 [10.361, 10.568] |
| qwen2.5:0.5b | selective | 48.67 [48.67, 48.67] | 98.00 [98.00, 98.00] | 210 [210, 210] | 22926 [22926, 22926]/975 [975, 975] | 9.663 [9.484, 9.776] | 8.977 [8.822, 9.073] |
| qwen2.5:0.5b | rcav_zero | 49.33 [49.33, 49.33] | 98.00 [98.00, 98.00] | 201 [200, 201] | 21778 [21688, 21823]/953 [945, 957] | 8.982 [8.908, 9.081] | 8.328 [8.256, 8.429] |
| qwen2.5:0.5b | rcav | 47.33 [47.33, 47.33] | 98.00 [98.00, 98.00] | 137 [136, 137] | 14127 [14037, 14172]/762 [754, 766] | 6.465 [6.312, 6.674] | 6.019 [5.862, 6.224] |
| qwen2.5:0.5b | rcav_high | 44.67 [44.67, 44.67] | 88.89 [88.67, 89.33] | 120 [120, 120] | 11707 [11707, 11707]/716 [709, 719] | 5.157 [4.952, 5.320] | 4.773 [4.585, 4.933] |

| 모델 | 방식 | 평균 서비스(ms) | 평균 완료 지연(ms) | Ollama CPU(초) | RSS 경계최대(MiB) |
|---|---|---|---|---|---|
| qwen2.5:1.5b | baseline | 123.50 [117.81, 126.85] | 303.56 [288.10, 313.86] | 17.400 [16.830, 17.900] | 2110.7 [2096.5, 2139.0] |
| qwen2.5:1.5b | selective | 89.82 [87.51, 93.25] | 220.69 [216.44, 227.22] | 11.677 [11.610, 11.720] | 2110.7 [2096.5, 2139.0] |
| qwen2.5:1.5b | rcav_zero | 61.27 [58.10, 65.94] | 150.00 [143.31, 162.35] | 9.180 [9.040, 9.380] | 2110.7 [2096.5, 2139.0] |
| qwen2.5:1.5b | rcav | 37.25 [34.32, 42.69] | 93.26 [84.83, 107.49] | 6.770 [6.670, 6.920] | 2110.7 [2096.5, 2139.0] |
| qwen2.5:1.5b | rcav_high | 30.81 [27.73, 35.55] | 76.51 [69.89, 88.62] | 6.167 [5.930, 6.430] | 2109.8 [2093.8, 2139.0] |
| qwen2.5:0.5b | baseline | 75.46 [74.95, 76.39] | 178.59 [176.72, 182.04] | 18.033 [17.970, 18.100] | 2019.5 [1821.3, 2241.4] |
| qwen2.5:0.5b | selective | 64.42 [63.23, 65.17] | 150.82 [147.81, 152.96] | 14.923 [14.480, 15.210] | 2018.3 [1822.9, 2238.2] |
| qwen2.5:0.5b | rcav_zero | 59.88 [59.38, 60.54] | 137.81 [136.18, 139.91] | 14.117 [13.830, 14.330] | 2018.9 [1822.9, 2238.2] |
| qwen2.5:0.5b | rcav | 43.09 [42.08, 44.49] | 97.62 [94.18, 101.51] | 10.970 [10.850, 11.130] | 2018.9 [1822.9, 2238.2] |
| qwen2.5:0.5b | rcav_high | 34.38 [33.01, 35.47] | 77.57 [74.32, 80.63] | 9.470 [9.200, 9.760] | 2018.0 [1820.1, 2238.2] |

서비스 시간은 해당 요청의 생성·의사결정·검증을 포함한다. 완료 지연은 블록 시작 동시도착에서 앞 요청 대기를 포함한다. 처리시간은 방식별 블록 벽시계 합계이며 모델 처리시간은 서버 prefill+generation합계다. CPU는 Ollama 프로세스군 실제 카운터 차다. RSS는 양끝 표본 최대로 피크/전용 메모리가 아니고 다른 상주 모델도 포함할 수 있다.

## 기존 Selective 대비 관측 판정

| 모델 | RCAV 조건 | 품질 유지 반복 수 | 처리시간 감소 | 모델시간 감소 | CPU 감소 |
|---|---|---:|---:|---:|---:|
| qwen2.5:1.5b | rcav_zero | 3/3 | 3/3 | 3/3 | 3/3 |
| qwen2.5:1.5b | rcav | 3/3 | 3/3 | 3/3 | 3/3 |
| qwen2.5:1.5b | rcav_high | 3/3 | 3/3 | 3/3 | 3/3 |
| qwen2.5:0.5b | rcav_zero | 3/3 | 3/3 | 3/3 | 3/3 |
| qwen2.5:0.5b | rcav | 0/3 | 3/3 | 3/3 | 3/3 |
| qwen2.5:0.5b | rcav_high | 0/3 | 3/3 | 3/3 | 3/3 |

품질 유지는 같은 반복의 정확도·답변율 이상이라는 관측 기준이며 통계적 비열등성 보장이 아니다. paired.csv는 Baseline/Selective 각각에 대한 문항별 손실도 기록한다. λ=0은 비용항이 없는 절감 효과 대조군이므로 비용항이 없는 변형과 주 RCAV의 행동이 같으면 비용항의 추가 효용을 입증한 것이 아니다.

## 관측 Pareto 관계

```json
[
  {
    "model": "qwen2.5:1.5b",
    "method": "baseline",
    "cost_axis": "model_processing_seconds",
    "accuracy": 0.6266666666666667,
    "answer_rate": 1.0,
    "measured_mean_cost": 17.354441333333334,
    "on_observed_frontier": false,
    "dominated_by": [
      "selective",
      "rcav_zero",
      "rcav",
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "baseline",
    "cost_axis": "processing_seconds",
    "accuracy": 0.6266666666666667,
    "answer_rate": 1.0,
    "measured_mean_cost": 18.526231723139063,
    "on_observed_frontier": false,
    "dominated_by": [
      "selective",
      "rcav_zero",
      "rcav",
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "baseline",
    "cost_axis": "ollama_cpu_seconds",
    "accuracy": 0.6266666666666667,
    "answer_rate": 1.0,
    "measured_mean_cost": 17.399999999999984,
    "on_observed_frontier": false,
    "dominated_by": [
      "selective",
      "rcav_zero",
      "rcav",
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "selective",
    "cost_axis": "model_processing_seconds",
    "accuracy": 0.7333333333333333,
    "answer_rate": 1.0,
    "measured_mean_cost": 12.514224332,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav_zero",
      "rcav",
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "selective",
    "cost_axis": "processing_seconds",
    "accuracy": 0.7333333333333333,
    "answer_rate": 1.0,
    "measured_mean_cost": 13.474175665294752,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav_zero",
      "rcav",
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "selective",
    "cost_axis": "ollama_cpu_seconds",
    "accuracy": 0.7333333333333333,
    "answer_rate": 1.0,
    "measured_mean_cost": 11.67666666666665,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav_zero",
      "rcav",
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "rcav_zero",
    "cost_axis": "model_processing_seconds",
    "accuracy": 0.7466666666666667,
    "answer_rate": 1.0,
    "measured_mean_cost": 8.414886999,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav",
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "rcav_zero",
    "cost_axis": "processing_seconds",
    "accuracy": 0.7466666666666667,
    "answer_rate": 1.0,
    "measured_mean_cost": 9.190688154427335,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav",
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "rcav_zero",
    "cost_axis": "ollama_cpu_seconds",
    "accuracy": 0.7466666666666667,
    "answer_rate": 1.0,
    "measured_mean_cost": 9.18000000000002,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav",
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "rcav",
    "cost_axis": "model_processing_seconds",
    "accuracy": 0.7533333333333333,
    "answer_rate": 1.0,
    "measured_mean_cost": 5.006192333333333,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "rcav",
    "cost_axis": "processing_seconds",
    "accuracy": 0.7533333333333333,
    "answer_rate": 1.0,
    "measured_mean_cost": 5.5878522507846355,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "rcav",
    "cost_axis": "ollama_cpu_seconds",
    "accuracy": 0.7533333333333333,
    "answer_rate": 1.0,
    "measured_mean_cost": 6.769999999999972,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav_high"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "rcav_high",
    "cost_axis": "model_processing_seconds",
    "accuracy": 0.7533333333333333,
    "answer_rate": 1.0,
    "measured_mean_cost": 4.113200999666667,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "rcav_high",
    "cost_axis": "processing_seconds",
    "accuracy": 0.7533333333333333,
    "answer_rate": 1.0,
    "measured_mean_cost": 4.621454873665546,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:1.5b",
    "method": "rcav_high",
    "cost_axis": "ollama_cpu_seconds",
    "accuracy": 0.7533333333333333,
    "answer_rate": 1.0,
    "measured_mean_cost": 6.166666666666713,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "baseline",
    "cost_axis": "model_processing_seconds",
    "accuracy": 0.48,
    "answer_rate": 0.98,
    "measured_mean_cost": 10.443686663,
    "on_observed_frontier": false,
    "dominated_by": [
      "selective",
      "rcav_zero"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "baseline",
    "cost_axis": "processing_seconds",
    "accuracy": 0.48,
    "answer_rate": 0.98,
    "measured_mean_cost": 11.318831583174566,
    "on_observed_frontier": false,
    "dominated_by": [
      "selective",
      "rcav_zero"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "baseline",
    "cost_axis": "ollama_cpu_seconds",
    "accuracy": 0.48,
    "answer_rate": 0.98,
    "measured_mean_cost": 18.03333333333333,
    "on_observed_frontier": false,
    "dominated_by": [
      "selective",
      "rcav_zero"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "selective",
    "cost_axis": "model_processing_seconds",
    "accuracy": 0.4866666666666667,
    "answer_rate": 0.98,
    "measured_mean_cost": 8.976966316,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav_zero"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "selective",
    "cost_axis": "processing_seconds",
    "accuracy": 0.4866666666666667,
    "answer_rate": 0.98,
    "measured_mean_cost": 9.663320736416305,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav_zero"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "selective",
    "cost_axis": "ollama_cpu_seconds",
    "accuracy": 0.4866666666666667,
    "answer_rate": 0.98,
    "measured_mean_cost": 14.923333333333359,
    "on_observed_frontier": false,
    "dominated_by": [
      "rcav_zero"
    ],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "rcav_zero",
    "cost_axis": "model_processing_seconds",
    "accuracy": 0.49333333333333335,
    "answer_rate": 0.98,
    "measured_mean_cost": 8.327523649666666,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "rcav_zero",
    "cost_axis": "processing_seconds",
    "accuracy": 0.49333333333333335,
    "answer_rate": 0.98,
    "measured_mean_cost": 8.982023960367465,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "rcav_zero",
    "cost_axis": "ollama_cpu_seconds",
    "accuracy": 0.49333333333333335,
    "answer_rate": 0.98,
    "measured_mean_cost": 14.116666666666601,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "rcav",
    "cost_axis": "model_processing_seconds",
    "accuracy": 0.47333333333333333,
    "answer_rate": 0.98,
    "measured_mean_cost": 6.019129988,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "rcav",
    "cost_axis": "processing_seconds",
    "accuracy": 0.47333333333333333,
    "answer_rate": 0.98,
    "measured_mean_cost": 6.46455075001965,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "rcav",
    "cost_axis": "ollama_cpu_seconds",
    "accuracy": 0.47333333333333333,
    "answer_rate": 0.98,
    "measured_mean_cost": 10.970000000000068,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "rcav_high",
    "cost_axis": "model_processing_seconds",
    "accuracy": 0.44666666666666666,
    "answer_rate": 0.888888888888889,
    "measured_mean_cost": 4.773163663,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "rcav_high",
    "cost_axis": "processing_seconds",
    "accuracy": 0.44666666666666666,
    "answer_rate": 0.888888888888889,
    "measured_mean_cost": 5.157038460408027,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  },
  {
    "model": "qwen2.5:0.5b",
    "method": "rcav_high",
    "cost_axis": "ollama_cpu_seconds",
    "accuracy": 0.44666666666666666,
    "answer_rate": 0.888888888888889,
    "measured_mean_cost": 9.469999999999953,
    "on_observed_frontier": true,
    "dominated_by": [],
    "scope": "Observed means only; not significance-tested or a global Pareto frontier"
  }
]
```

정확도 최대·측정시간/CPU 최소·답변율 비감소 기준의 다섯 관측점이다. 표본 오차/타이밍 잡음에 따른 극소 차이로 frontier가 달라질 수 있으며 전역 최적화나 독립 일반화의 증거가 아니다. 가중치 중 유리한 조건만 선택해 성공이라고 하지 않는다.

## 추정과 실측·한계

decision의 risk_estimate/expected_error_reduction/estimated_verification_model_seconds는 보정 데이터 기반 **추정값**이다. 실제 정확도·토큰·서버시간·클라이언트 벽시계·CPU·RSS는 실행 로그의 **측정값**이다. 호출 감소와 총 연산량 감소를 동일시하지 않는다. **금전비용·GPU FLOPs·전력은 미측정이며 추정 절감률을 만들지 않았다.**

보정 비용은 calibration_summary.csv에 분리했으며 초기 비용 상각을 가정하지 않았다. 평가 전체 루프 345.505초는 계측·사후채점·파일저장도 포함한다. 워밍업은 별도 호출 로그에 저장했다. 모델마다 연속 실행해 서버/캐시/발열/메모리 상태가 다를 수 있다. 모델 간 절대속도 인과 비교에는 한계가 있다.

작은 층화 표본·회귀 비용의 캐시 민감도·효과 크기의 점추정·분포 이동 때문에 위험/비용/검증 효과가 틀릴 수 있다. CPU/GPU별 비용 모델이나 신뢰상한을 학습한 시스템이 아니다. 임계값의 물리적·경제적 최적성을 주장하지 않는다. 영어 객관식, 합성 논리/지문과 코드 출력 예측이며 의미상 템플릿 독립을 보장하지 않는다. 내용 추출은 정답과 무관한 명확한 표기만 해석하며 자유서술의 의미 정답을 판정하지 않는다.

## 파일

- [manifest.json](rcav_c038afc6c5df422aa9e58080a4fa5847_manifest.json)
- [profiles.json](rcav_c038afc6c5df422aa9e58080a4fa5847_profiles.json)
- [calibration.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_calibration.csv)
- [calibration_calls.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_calibration_calls.csv)
- [calibration_summary.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_calibration_summary.csv)
- [warmup_calls.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_warmup_calls.csv)
- [evaluation.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_evaluation.csv)
- [calls.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_calls.csv)
- [blocks.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_blocks.csv)
- [repeat_summary.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_repeat_summary.csv)
- [aggregate.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_aggregate.csv)
- [paired.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_paired.csv)
- [pareto.csv](rcav_c038afc6c5df422aa9e58080a4fa5847_pareto.csv)
- [metadata.json](rcav_c038afc6c5df422aa9e58080a4fa5847_metadata.json)
