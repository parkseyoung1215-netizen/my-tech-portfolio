# 불필요한 추론·검증 생략 연구

실행 ID `7a7cc6bdb72e495ab9af664ba5b15db7`.

## 사전에 고정한 설계

Baseline은 기존 Hybrid의 해결 경로(산술 도구, 비산술1.5B)에 전수1.5B 재검증을 추가한 연구용 대조군이다. 이것이 이전 Hybrid보다 우수한 생산 기준선이라는 뜻은 아니다. 검증 결과가 파싱되면 최종 답으로 채택하고, 검증 형식이 실패하면 원래 답을 유지한다. 검증이 오히려 정답을 오답으로 바꿀 수 있다.

- Early Exit: 실행 전 안전 산술식임을 확인하면 도구 답 이후 모델 재검증을 생략한다. 기존 도구 방식의 통제 실험이며 새 발명으로 주장하지 않는다.
- Selective: 안전 도구 증명은 생략하고, 쉬운 유형의 보정20개 이상에서 검증 수정건수≤악화건수이면 검증을 생략한다. 어려운 요청·근거 부족은 검증한다.
- 가설1(Selective): 기본 답의 자신감이 아니라 검증의 예상 순이득을 기준으로 판단하면 개선하지 못하는 검증을 줄일 수 있다. 보정의 점추정 규칙으로 정확도 보장이 아니다.
- 가설2(Budget Cap): 객관식의 첫 모델 출력 상한을64에서8토큰으로 줄이면 호출 수가 같아도 출력 연산을 줄일 수 있다. 검증 예산은 그대로다. 잘림 위험을 기록한다.
- 가설3(Batch Verify): 해결 전 유형·입력 길이만으로 최대4개·2600자 묶음을 고정해 검증을 한 호출로 합친다. 호출 감소가 문맥·출력·대기 증가를 상쇄할지는 실제 측정으로 판단한다.

새150문제는 유형별30개다. 보정은 별도 저장된100문제이며 현재 평가와 과거 모든 질문 줄기·교환 중복을 제외했다. 보정·정책·출력 계약·묶음계획은 평가 전에 고정했다. 생략 함수는 질문과 보정 통계만 받고 답안·기대 정답·유형 라벨을 받지 않는다.

## 실제 결과

| 방식 | 정확도 | 답변율 | 모델 호출 | 입력/출력 토큰 | 처리 벽시계 합계(초) | 평균 완료 지연(ms) | 생성 시간(초) |
|---|---:|---:|---:|---|---:|---:|---:|
| baseline | 58.00% | 99.33% | 270 | 27532/639 | 27.660 | 439.508 | 6.035 |
| early_exit | 66.67% | 99.33% | 240 | 25284/489 | 29.223 | 473.262 | 4.747 |
| selective | 64.67% | 97.33% | 210 | 22421/428 | 24.370 | 394.738 | 3.915 |
| budget_cap | 58.67% | 99.33% | 270 | 27532/639 | 31.687 | 511.426 | 6.172 |
| batch_verify | 67.33% | 97.33% | 160 | 22796/2099 | 59.741 | 1538.758 | 31.839 |

모든 요청은 해당 블록 시작에 도착한 것으로 정의했다. 순차 방식도 앞 요청 대기를 포함하고 묶음 방식은 같은 묶음 검증 종료를 기다린다. 처리 시간은 메서드별 블록의 실제 벽시계 합계이며, 겹치는 요청 지연을 더한 값과 다르다. 공유 검증 호출의 토큰·시간을 요청별로 복제하거나 임의 분할하지 않았다.

## 실제 자원과 검증 효과

| 방식 | Ollama CPU(초) | Python CPU(초) | RSS 경계최대(MiB) | 검증 요청/생략 요청 | 결과별 검증 효과 |
|---|---:|---:|---:|---|---|
| baseline | 19.590 | 0.277 | 1494.6 | 150/0 | {'beneficial': 13, 'redundant_correct_by_outcome': 74, 'harmful': 28, 'still_wrong_or_unresolved': 35} |
| early_exit | 15.610 | 0.220 | 1500.5 | 120/30 | {'beneficial': 14, 'redundant_correct_by_outcome': 56, 'not_executed': 30, 'still_wrong_or_unresolved': 35, 'harmful': 15} |
| selective | 13.840 | 0.192 | 1500.5 | 90/60 | {'not_executed': 60, 'redundant_correct_by_outcome': 30, 'still_wrong_or_unresolved': 34, 'harmful': 15, 'beneficial': 11} |
| budget_cap | 19.770 | 0.246 | 1501.2 | 150/0 | {'beneficial': 14, 'redundant_correct_by_outcome': 74, 'harmful': 27, 'still_wrong_or_unresolved': 35} |
| batch_verify | 56.520 | 0.160 | 1488.6 | 150/0 | {'still_wrong_or_unresolved': 49, 'redundant_correct_by_outcome': 101} |

redundant_correct_by_outcome은 사후에 원래 정답과 최종 정답이 모두 맞았던 검증을 뜻한다. 실행 시 정답을 알고 생략한 것이 아니며, 이런 검증이 모든 목적에서 무가치하다고 단정하지 않는다. correct→wrong 악화와 wrong→correct 개선, 파싱 실패를 문항별로 분리했다.

## 기준선 대비 판정

```json
[
  {
    "method": "early_exit",
    "accuracy_delta": 0.08666666666666667,
    "answer_rate_delta": 0.0,
    "model_call_delta": -30,
    "input_token_delta": -2248,
    "output_token_delta": -150,
    "processing_seconds_delta": 1.562545750522986,
    "mean_latency_delta": 0.03375414523761722,
    "correct_to_wrong": 0,
    "wrong_to_correct": 13,
    "quality_non_degradation_observed": true,
    "lower_latency_observed": false,
    "lower_processing_time_observed": false,
    "interpretation": "Calls and tokens are measured workload indicators, not FLOPs or money"
  },
  {
    "method": "selective",
    "accuracy_delta": 0.06666666666666665,
    "answer_rate_delta": -0.019999999999999907,
    "model_call_delta": -60,
    "input_token_delta": -5111,
    "output_token_delta": -211,
    "processing_seconds_delta": -3.2903075399808586,
    "mean_latency_delta": -0.04477061389790227,
    "correct_to_wrong": 3,
    "wrong_to_correct": 13,
    "quality_non_degradation_observed": false,
    "lower_latency_observed": true,
    "lower_processing_time_observed": true,
    "interpretation": "Calls and tokens are measured workload indicators, not FLOPs or money"
  },
  {
    "method": "budget_cap",
    "accuracy_delta": 0.00666666666666671,
    "answer_rate_delta": 0.0,
    "model_call_delta": 0,
    "input_token_delta": 0,
    "output_token_delta": 0,
    "processing_seconds_delta": 4.027215207694098,
    "mean_latency_delta": 0.07191808556827411,
    "correct_to_wrong": 0,
    "wrong_to_correct": 1,
    "quality_non_degradation_observed": true,
    "lower_latency_observed": false,
    "lower_processing_time_observed": false,
    "interpretation": "Calls and tokens are measured workload indicators, not FLOPs or money"
  },
  {
    "method": "batch_verify",
    "accuracy_delta": 0.09333333333333338,
    "answer_rate_delta": -0.019999999999999907,
    "model_call_delta": -110,
    "input_token_delta": -4736,
    "output_token_delta": 1460,
    "processing_seconds_delta": 32.08043000055477,
    "mean_latency_delta": 1.099250026877659,
    "correct_to_wrong": 13,
    "wrong_to_correct": 27,
    "quality_non_degradation_observed": false,
    "lower_latency_observed": false,
    "lower_processing_time_observed": false,
    "interpretation": "Calls and tokens are measured workload indicators, not FLOPs or money"
  }
]
```

모델 호출이 감소해도 입력/출력 토큰·실제 생성 시간·CPU·처리 시간까지 감소했는지는 각각 확인해야 한다. 이 값들은 작업량과 시간의 대리 지표이며 GPU FLOPs나 총 하드웨어 연산량의 직접 측정이 아니다. 정확도가 떨어지면 무조건 우수하다고 하지 않는다. 같은 답변율의 단일 실행 관측값으로 일반적 우위를 확정하지 않는다.

## 보정과 계측 한계

```json
{
  "version": "skipwork-v1",
  "min_samples": 20,
  "skip_rule": "easy and calibration corrected <= harmed; exact tool proof always skips",
  "categories": {
    "arithmetic": {
      "samples": 20,
      "primary_correct": 20,
      "final_correct": 16,
      "corrected": 0,
      "harmed": 4
    },
    "knowledge": {
      "samples": 20,
      "primary_correct": 20,
      "final_correct": 18,
      "corrected": 0,
      "harmed": 2
    },
    "comprehension": {
      "samples": 20,
      "primary_correct": 9,
      "final_correct": 11,
      "corrected": 3,
      "harmed": 1
    },
    "coding": {
      "samples": 20,
      "primary_correct": 15,
      "final_correct": 16,
      "corrected": 2,
      "harmed": 1
    },
    "logic": {
      "samples": 20,
      "primary_correct": 13,
      "final_correct": 7,
      "corrected": 1,
      "harmed": 7
    }
  }
}
```

보정 벽시계 17.978초, 평가 전체 루프 187.779초. 보정 호출·토큰은 calibration_calls.csv에 분리했다. 최초 보정 비용을 회수했다고 가정하지 않는다. 점추정 기반 검증 순이득이 다른 문제로 일반화될 보장은 없으며 어려움 판정은 요청의 문법/길이 휴리스틱이다.

Ollama 프로세스군의 CPU 카운터 차·RSS 경계 표본과 Python process_time을 분리해 실제 수집했다. RSS는 상주·공유 메모리의 표본으로 전용/피크 메모리가 아니다. GPU·전력·FLOPs·금전 비용은 미측정이다. 입출력 토큰은 서버 기록이며 캐시를 제외한 실제 prefill 연산량을 의미하지 않는다. 자원 수집·사후 채점·워밍업은 처리 시간에서 제외하고 전체 루프에는 모니터·채점도 포함한다.

검증 묶음은 JSON 출력과 정확한 id 집합을 요구했다. 파싱 실패면 원래 답을 유지하고 추가 모델 호출로 숨겨 복구하지 않았다. 호출 감소가 실패한 검증이나 단순 답안 보존에서 나온 경우도 효과 분석에 남겼다. 같은 문제를 반복하는 로컬 측정이라 prefix cache·발열·배경 부하 영향이 있고 블록 순서만 균형화했다.

문제는 영어 객관식·코드 출력 예측·합성 논리/지문 템플릿이며 지식은 수도 문제에 편중된다. 지식 보정은 다른 주제이므로 분포 이동 한계도 있다. 새 질문 줄기는 중복이 없지만 개념·템플릿 계열이 모두 독립적이지 않다. 원본 파일·고정 명세·프로필 해시를 검사했고, 평가 라벨은 결과 생성 후 채점에만 사용했다. 유료 API나 모델 설치는 사용하지 않았다.

## 파일

- [manifest.json](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_manifest.json)
- [profile.json](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_profile.json)
- [plans.json](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_plans.json)
- [calibration.csv](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_calibration.csv)
- [calibration_calls.csv](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_calibration_calls.csv)
- [evaluation.csv](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_evaluation.csv)
- [calls.csv](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_calls.csv)
- [blocks.csv](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_blocks.csv)
- [summary.csv](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_summary.csv)
- [comparison.csv](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_comparison.csv)
- [reference_checks.csv](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_reference_checks.csv)
- [metadata.json](skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_metadata.json)
