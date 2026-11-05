# ORACLE_EXPERIMENT_REPORT

**최종 판정: C — 실패.** 실제 무료 로컬 Qwen2.5-1.5B-Instruct BF16 실험 `oracle_8ef05974726245a1b273f178fc668cb7`. 기존 실험·코드·데이터 보존, 유료 API·GitHub 업로드 없음.

판정의 의미는 사전에 정한 **정확도/형식 비감소·개별 정답 손실 없음·동일열 평균5% 이상·CI/반복 방향**의 동시 충족 여부다. C를 단순히 물리적인 속도 이득이 전혀 없거나 모든 DLR이 불가능하다는 뜻으로 사용하지 않는다. 5%는 이번 실험의 사전 실용 기준이며 그보다 작은 관측 이득도 아래에 숨기지 않고 보고한다.

**중요한 범위:** 이 Oracle은 평가 정답과 미래 Baseline 활성값을 사용해 후보3개 중 최선의 결정표를 고른다. 실제 서비스 알고리즘이 아니다. 수학적으로 최적인 Oracle도 아니므로, 부정적 결과가 모든 DLR의 불가능성을 증명하지 않는다. 긍정적 결과도 배포 가능한 판단 기준의 발견이나 모집단 정확도 보존 증거가 아니다.

## 1. 실험의 질문과 기존 구현

기존 DLR은8/12/16/24번 MLP에서 최신 fresh 입력 대비 상대 L2가0.5이하면 이전 MLP residual을 재사용한다. Attention과 KV는 매 토큰 갱신한다. Device-Resident는 GPU 조건 분기로 projection MAC 루프를 실제 우회하지만 fresh GEMV 수치 순서가 native와 달랐다. Lightweight는 이미 읽은 token ID의 반복을 신호로 썼으며 문맥 변화 및 EOS 손상이 발생했다.

이번에는 모든 비교의 실제 MLP를 기존 native MLX 연산으로 실행해 custom GEMV 수치 오차 요인을 제거했다. 생략 시 gate/up/down projection과 activation을 호출하지 않고 최신 fresh MLP 출력을 residual로 사용한다. zero residual Layer Skipping이 아니며 KV caching과도 다르다.

## 2. Oracle과 기준선 정의

- Baseline: 기존 정상 native 추론.
- Ideal Oracle: 미리 고정한 표 조회만 수행, runtime 거리 계산/판단 없음.
- Random Skip: 같은 토큰별 quota/cooldown에서 무작위 MLP 재사용.
- Existing DLR: 기존4후보·0.5기준 그대로. 목표 생략률별 재튜닝하지 않았다.
- Oracle+Actual Overhead: Ideal과 같은 표를 적용하면서 기존4후보의 실제 거리/eval/item 계산과 입력 anchor 관리를 추가하고, 그 판단은 버린다.

Ideal/Random/Overhead는 최대28개 MLP를 대상으로 하지만 Existing DLR은4개 후보 제한을 유지한다. 따라서 Existing을50% matched-rate 기법이라고 비교하지 않는다. 한 번 재사용 뒤 fresh 계산을 요구하므로 기존4후보의 긴 decode 이론상 평균 상한도 약2/28=7.14%다(유한 짧은 출력에서는 달라짐). 이는 구조적 호출 상한이며 실측 FLOPs/시간 감소율이 아니다. Baseline/Existing은 조건당 한 번 측정했고 그래프의 수평선은 동일 측정값이다. 여섯 생략률에서 독립 측정한 것으로 과장하지 않았다.

Random은 질문/rate별 독립 seed의 표 한 개를 고정해5회 반복했다. 따라서 반복 CI는 서로 다른 random plan 다섯 개의 불확실성을 포함하지 않는다. Oracle은 정답으로3후보를 선택하고 Random은 한 표만 쓰므로, 둘의 품질 차이를 실용적인 예측기의 우수성으로 해석하지 않는다.

Zero-overhead는 **판단 계산을 뺀 비교군**이라는 뜻이다. 표 생성·로드·경로 준비 비용은 따로 있고, runtime 표 인덱싱/분기와 output cache 관리는 남는다. Ideal/Random/Overhead는0%에서도28개 output anchor를 관리한다. 0%와 Baseline 차이에는 cache/lookup뿐 아니라 새 loop·계수·timer dispatch 차이도 있어 순수 memory-copy 비용으로 해석하지 않는다. Overhead는4개 입력 anchor를 더 보관한다. 이를 완전한 모든 overhead0으로 부르지 않는다.

생략률0/10/20/30/40/50%는 decode의28개 MLP 호출 기준이다. Prefill은 항상 전체 계산하고 한 층을 연속2번 재사용하지 않는다. 정수 quota는 누적 floor로 맞췄다. 실제 decode 비율, 전체 layer-token 비율, 전체 모델 FLOPs 추정 감소율을 각각 구분한다. 특히50% target은 전체 추론 연산량50% 감소가 아니다.

## 3. 사전 결정표 생성과 특권 정보

정상 모델을 공통 최대 길이까지 실행해 각 토큰·계층의 실제 MLP 출력을 수집했다. EOS 이후에도 미래 활성값 확보를 위해 계속 생성한 부분이 있다. 그 출력을 최근 fresh anchor와 비교한 최소 drift 표1개와 random 표2개를 만들고 실제 자연생성으로 평가했다. 선택 기준은 내용 정답, 형식, Baseline token열 일치, 길이 차이, 후보 index 순서로 사전에 고정했다. 평가 정답을 사용한 의도적 특권 선택이며 정보 누출 없는 실용 모델로 표현하지 않는다. Random 비교군은 별도 독립 seed다.

오프라인 trace 모델 호출 20회, 후보 생성 호출 300회. 표 생성 wall 합계 179.555초, CPU 합계 35.540초. 여기에는 활성값 수집·후보 생성·채점·파일 저장이 포함된다. 모든 후보와 실패는 tables 하위 TRACE/CANDIDATE 파일에 보존했다. 원시 activation 전체를 장기 보관하지 않고 shape/hash와 표를 저장했다.

이 오프라인 비용을 공짜로 보아 전체 서비스 비용이 감소했다고 주장하지 않는다. 성공 판정은 이미 표가 주어진 온라인 추론 단계의 가능성에 한정한다. 새 질문마다 표를 새로 만들면 이 비용이 다시 발생하며, 동일 질문의 표를 반복 사용하는 상황은 최종 답변 캐싱과도 구분해야 한다.

각 표의 SHA256를 반복 실행 전 기록하고 분석 시 검증했다. runtime은 질문·표·토큰예산을 받으며 정답/미래 활성값은 직접 받지 않지만, 표에 이미 그 정보가 반영되어 있다. 신규 문제의 정답도 Oracle 표 선택에 썼으므로 Oracle 정확도는 독립 holdout 일반화 측정이 아니다.

## 4. 공정 조건과 측정

기존8문제(객관식6/JSON2), 신규8문제(객관식4/JSON2/color2), 신규 긴 복사4문제(반복2/다양단어2), 총20문제다. 각 조건5회, 자연생성 및 동일 토큰열 각각2000건, 주 비교4000건을 실제 실행했다. seed2026110606. 동일 prompt/chat template/greedy/EOS, short32/long64토큰 cap. 쉬운/어려운 실패를 제외하지 않았다. 합성 exact stem 중복만 검사했으며 개념적 유사성은 남는다.

가중치 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 패키지 `{"mlx": "0.32.4", "mlx-lm": "0.32.2", "numpy": "2.5.4", "transformers": "5.22.1"}`. 기존 로컬 파일 해시 확인. M4 16GiB, 권장 working-set80%/MLX allocator cache128MiB 제한을 유지했다.

워밍업 후 질문별 조건 순서를 섞고5회 반복했다. 자연생성이 모두 끝난 뒤 Baseline1회차의 실제 토큰열(EOS 포함)을 모든 방식에 동일하게 입력하는 fixed-stream을 실행했다. 후자는 실제 forward 비용 측정이며 강제 출력의 정답률을 독립적인 생성 품질로 해석하지 않는다. 실제 예측 token도 가능한 경로에서 별도 보존한다.

요청 wall은 기존 DLR과 같은 내부 측정 범위로, tokenization 전부터 decode 및 최종 MLX 동기화까지다. 이후 무결성 검사/일부 metadata 조립/함수 반환·최종 참조 해제/저장은 별도여서 전체 controller 시간과 같지 않다. 루프 안 cache 갱신·반환값 준비에 필요한 GPU 작업·판단 비용은 포함한다. COMPLETE 로그의 controller elapsed는 표 생성/로드/워밍업/저장도 포함하므로 방식별 추론 절감률의 분모로 사용하지 않았다.

내용과 형식을 기존 채점 함수로 분리했다. JSON은 중복키/type혼동을 거부하고, color는 기존 제한 문법, copy는24단어 전체 순서 일치를 요구했다. 부분 prefix만 맞은 불완전 응답은 실패다. 일반 자연어 의미/환각 전체를 평가하는 지표가 아니다.

## 5. 자연생성 전체 결과

### 전체20문제 합계

| 방식 | 목표% | 내용% | 형식% | Baseline token 일치% | wall 평균±SD(s) | CPU(s) | 출력토큰 |
|---|---|---|---|---|---|---|---|
| Baseline | 고유 | 75.00 | 30.00 | 100.00 | 11.060 ± 0.414 | 3.140 | 261 |
| Existing DLR | 고유 | 75.00 | 30.00 | 100.00 | 11.655 ± 0.496 | 3.407 | 261 |
| Ideal Oracle | 0 | 75.00 | 30.00 | 100.00 | 11.032 ± 0.466 | 2.954 | 261 |
| Ideal Oracle | 10 | 70.00 | 45.00 | 75.00 | 10.458 ± 0.468 | 2.966 | 259 |
| Ideal Oracle | 20 | 50.00 | 40.00 | 50.00 | 9.557 ± 0.287 | 2.503 | 249 |
| Ideal Oracle | 30 | 45.00 | 15.00 | 15.00 | 9.634 ± 0.404 | 2.861 | 268 |
| Ideal Oracle | 40 | 45.00 | 10.00 | 10.00 | 10.808 ± 0.570 | 3.233 | 340 |
| Ideal Oracle | 50 | 10.00 | 0.00 | 0.00 | 9.534 ± 1.032 | 2.729 | 302 |
| Random Skip | 0 | 75.00 | 30.00 | 100.00 | 11.283 ± 0.596 | 3.136 | 261 |
| Random Skip | 10 | 60.00 | 20.00 | 40.00 | 11.442 ± 0.742 | 3.097 | 283 |
| Random Skip | 20 | 35.00 | 15.00 | 20.00 | 10.791 ± 0.683 | 3.017 | 281 |
| Random Skip | 30 | 30.00 | 5.00 | 10.00 | 12.825 ± 0.424 | 3.801 | 386 |
| Random Skip | 40 | 20.00 | 5.00 | 0.00 | 12.240 ± 0.698 | 3.537 | 394 |
| Random Skip | 50 | 0.00 | 0.00 | 0.00 | 10.362 ± 0.562 | 3.177 | 354 |
| Oracle+Actual Overhead | 0 | 75.00 | 30.00 | 100.00 | 12.064 ± 0.844 | 3.537 | 261 |
| Oracle+Actual Overhead | 10 | 70.00 | 45.00 | 75.00 | 11.143 ± 0.540 | 3.331 | 259 |
| Oracle+Actual Overhead | 20 | 50.00 | 40.00 | 50.00 | 10.191 ± 0.493 | 2.985 | 249 |
| Oracle+Actual Overhead | 30 | 45.00 | 15.00 | 15.00 | 10.080 ± 0.459 | 2.946 | 268 |
| Oracle+Actual Overhead | 40 | 45.00 | 10.00 | 10.00 | 11.400 ± 0.707 | 3.218 | 340 |
| Oracle+Actual Overhead | 50 | 10.00 | 0.00 | 0.00 | 9.791 ± 0.649 | 2.991 | 302 |

전체 정확도는 집합 평균을 단순 평균하지 않고 실제 문제 수8/8/4로 가중했다. 전체 표의95%CI는 고정20문제의5반복 합계 bootstrap이고, paired 질문/반복 비교 CI는 아래 별도 표다.

### replay8

| 방식 | 목표 decode% | 실제 decode% | 내용% | 형식% | Baseline 일치% | wall 평균±SD(s) | CPU(s) | 출력토큰 | MLX peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 고유 설정 | 0.00 | 87.50 | 12.50 | 100.00 | 3.502 ± 0.202 | 0.959 | 76 | 3077.90 |
| Existing DLR | 고유 설정 | 0.16 | 87.50 | 12.50 | 100.00 | 3.721 ± 0.209 | 1.134 | 76 | 3080.31 |
| Ideal Oracle | 0 | 0.00 | 87.50 | 12.50 | 100.00 | 3.428 ± 0.178 | 0.806 | 76 | 3087.18 |
| Ideal Oracle | 10 | 9.78 | 87.50 | 37.50 | 75.00 | 3.192 ± 0.223 | 0.799 | 73 | 3087.18 |
| Ideal Oracle | 20 | 19.81 | 62.50 | 25.00 | 50.00 | 3.030 ± 0.156 | 0.730 | 72 | 3087.18 |
| Ideal Oracle | 30 | 29.72 | 62.50 | 12.50 | 12.50 | 2.799 ± 0.108 | 0.777 | 70 | 3087.18 |
| Ideal Oracle | 40 | 39.80 | 62.50 | 0.00 | 0.00 | 2.870 ± 0.228 | 0.748 | 78 | 3087.18 |
| Ideal Oracle | 50 | 50.00 | 25.00 | 0.00 | 0.00 | 2.979 ± 0.131 | 0.824 | 88 | 3087.36 |
| Random Skip | 0 | 0.00 | 87.50 | 12.50 | 100.00 | 3.470 ± 0.226 | 0.894 | 76 | 3087.25 |
| Random Skip | 10 | 9.78 | 87.50 | 25.00 | 50.00 | 3.302 ± 0.126 | 0.821 | 77 | 3087.18 |
| Random Skip | 20 | 19.86 | 50.00 | 25.00 | 12.50 | 3.091 ± 0.231 | 0.759 | 74 | 3087.05 |
| Random Skip | 30 | 29.81 | 50.00 | 12.50 | 25.00 | 2.957 ± 0.162 | 0.769 | 77 | 3087.18 |
| Random Skip | 40 | 39.87 | 37.50 | 0.00 | 0.00 | 3.365 ± 0.279 | 0.880 | 99 | 3087.18 |
| Random Skip | 50 | 50.00 | 0.00 | 0.00 | 0.00 | 3.685 ± 0.154 | 1.106 | 120 | 3087.18 |
| Oracle+Actual Overhead | 0 | 0.00 | 87.50 | 12.50 | 100.00 | 3.809 ± 0.494 | 0.977 | 76 | 3088.21 |
| Oracle+Actual Overhead | 10 | 9.78 | 87.50 | 37.50 | 75.00 | 3.377 ± 0.273 | 0.942 | 73 | 3088.21 |
| Oracle+Actual Overhead | 20 | 19.81 | 62.50 | 25.00 | 50.00 | 3.152 ± 0.271 | 0.814 | 72 | 3088.66 |
| Oracle+Actual Overhead | 30 | 29.72 | 62.50 | 12.50 | 12.50 | 2.879 ± 0.169 | 0.735 | 70 | 3088.21 |
| Oracle+Actual Overhead | 40 | 39.80 | 62.50 | 0.00 | 0.00 | 3.006 ± 0.235 | 0.862 | 78 | 3088.21 |
| Oracle+Actual Overhead | 50 | 50.00 | 25.00 | 0.00 | 0.00 | 3.105 ± 0.258 | 0.858 | 88 | 3088.21 |

### fresh8

| 방식 | 목표 decode% | 실제 decode% | 내용% | 형식% | Baseline 일치% | wall 평균±SD(s) | CPU(s) | 출력토큰 | MLX peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 고유 설정 | 0.00 | 75.00 | 37.50 | 100.00 | 3.319 ± 0.138 | 1.054 | 74 | 3072.54 |
| Existing DLR | 고유 설정 | 0.11 | 75.00 | 37.50 | 100.00 | 3.501 ± 0.212 | 1.069 | 74 | 3075.01 |
| Ideal Oracle | 0 | 0.00 | 75.00 | 37.50 | 100.00 | 3.300 ± 0.177 | 0.961 | 74 | 3081.40 |
| Ideal Oracle | 10 | 9.62 | 75.00 | 62.50 | 75.00 | 2.890 ± 0.169 | 0.847 | 67 | 3081.40 |
| Ideal Oracle | 20 | 19.75 | 50.00 | 62.50 | 50.00 | 2.918 ± 0.089 | 0.805 | 72 | 3081.40 |
| Ideal Oracle | 30 | 29.78 | 50.00 | 25.00 | 25.00 | 3.167 ± 0.191 | 0.927 | 85 | 3081.40 |
| Ideal Oracle | 40 | 39.86 | 50.00 | 25.00 | 25.00 | 2.880 ± 0.163 | 0.841 | 83 | 3081.40 |
| Ideal Oracle | 50 | 50.00 | 0.00 | 0.00 | 0.00 | 2.989 ± 0.434 | 0.932 | 88 | 3081.58 |
| Random Skip | 0 | 0.00 | 75.00 | 37.50 | 100.00 | 3.320 ± 0.145 | 0.965 | 74 | 3081.84 |
| Random Skip | 10 | 9.71 | 62.50 | 25.00 | 50.00 | 3.143 ± 0.314 | 0.851 | 72 | 3081.56 |
| Random Skip | 20 | 19.88 | 37.50 | 12.50 | 37.50 | 3.048 ± 0.158 | 0.887 | 77 | 3081.58 |
| Random Skip | 30 | 29.81 | 25.00 | 0.00 | 0.00 | 3.235 ± 0.113 | 0.939 | 89 | 3081.40 |
| Random Skip | 40 | 39.88 | 12.50 | 12.50 | 0.00 | 3.156 ± 0.216 | 0.887 | 92 | 3081.49 |
| Random Skip | 50 | 50.00 | 0.00 | 0.00 | 0.00 | 2.269 ± 0.164 | 0.650 | 66 | 3081.40 |
| Oracle+Actual Overhead | 0 | 0.00 | 75.00 | 37.50 | 100.00 | 3.503 ± 0.226 | 1.082 | 74 | 3082.65 |
| Oracle+Actual Overhead | 10 | 9.62 | 75.00 | 62.50 | 75.00 | 3.065 ± 0.159 | 0.915 | 67 | 3082.39 |
| Oracle+Actual Overhead | 20 | 19.75 | 50.00 | 62.50 | 50.00 | 3.106 ± 0.141 | 0.955 | 72 | 3082.39 |
| Oracle+Actual Overhead | 30 | 29.78 | 50.00 | 25.00 | 25.00 | 3.324 ± 0.195 | 1.004 | 85 | 3082.39 |
| Oracle+Actual Overhead | 40 | 39.86 | 50.00 | 25.00 | 25.00 | 3.078 ± 0.237 | 0.888 | 83 | 3082.39 |
| Oracle+Actual Overhead | 50 | 50.00 | 0.00 | 0.00 | 0.00 | 3.066 ± 0.187 | 1.024 | 88 | 3082.46 |

### long4

| 방식 | 목표 decode% | 실제 decode% | 내용% | 형식% | Baseline 일치% | wall 평균±SD(s) | CPU(s) | 출력토큰 | MLX peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 고유 설정 | 0.00 | 50.00 | 50.00 | 100.00 | 4.239 ± 0.084 | 1.128 | 111 | 3052.51 |
| Existing DLR | 고유 설정 | 3.27 | 50.00 | 50.00 | 100.00 | 4.432 ± 0.107 | 1.204 | 111 | 3054.20 |
| Ideal Oracle | 0 | 0.00 | 50.00 | 50.00 | 100.00 | 4.303 ± 0.174 | 1.187 | 111 | 3059.20 |
| Ideal Oracle | 10 | 9.97 | 25.00 | 25.00 | 75.00 | 4.376 ± 0.162 | 1.320 | 119 | 3059.37 |
| Ideal Oracle | 20 | 19.94 | 25.00 | 25.00 | 50.00 | 3.609 ± 0.158 | 0.968 | 105 | 3059.20 |
| Ideal Oracle | 30 | 29.95 | 0.00 | 0.00 | 0.00 | 3.667 ± 0.123 | 1.157 | 113 | 3059.20 |
| Ideal Oracle | 40 | 39.96 | 0.00 | 0.00 | 0.00 | 5.058 ± 0.210 | 1.645 | 179 | 3059.20 |
| Ideal Oracle | 50 | 50.00 | 0.00 | 0.00 | 0.00 | 3.566 ± 0.630 | 0.972 | 126 | 3059.20 |
| Random Skip | 0 | 0.00 | 50.00 | 50.00 | 100.00 | 4.494 ± 0.284 | 1.278 | 111 | 3059.20 |
| Random Skip | 10 | 9.95 | 0.00 | 0.00 | 0.00 | 4.998 ± 0.376 | 1.425 | 134 | 3059.20 |
| Random Skip | 20 | 19.95 | 0.00 | 0.00 | 0.00 | 4.653 ± 0.579 | 1.371 | 130 | 3059.55 |
| Random Skip | 30 | 29.98 | 0.00 | 0.00 | 0.00 | 6.633 ± 0.189 | 2.093 | 220 | 3059.20 |
| Random Skip | 40 | 39.97 | 0.00 | 0.00 | 0.00 | 5.719 ± 0.268 | 1.771 | 203 | 3059.55 |
| Random Skip | 50 | 50.00 | 0.00 | 0.00 | 0.00 | 4.408 ± 0.285 | 1.421 | 168 | 3059.20 |
| Oracle+Actual Overhead | 0 | 0.00 | 50.00 | 50.00 | 100.00 | 4.752 ± 0.202 | 1.478 | 111 | 3060.00 |
| Oracle+Actual Overhead | 10 | 9.97 | 25.00 | 25.00 | 75.00 | 4.701 ± 0.153 | 1.474 | 119 | 3060.05 |
| Oracle+Actual Overhead | 20 | 19.94 | 25.00 | 25.00 | 50.00 | 3.933 ± 0.142 | 1.216 | 105 | 3059.95 |
| Oracle+Actual Overhead | 30 | 29.95 | 0.00 | 0.00 | 0.00 | 3.878 ± 0.130 | 1.207 | 113 | 3059.95 |
| Oracle+Actual Overhead | 40 | 39.96 | 0.00 | 0.00 | 0.00 | 5.317 ± 0.299 | 1.468 | 179 | 3059.95 |
| Oracle+Actual Overhead | 50 | 50.00 | 0.00 | 0.00 | 0.00 | 3.620 ± 0.242 | 1.109 | 126 | 3059.95 |

시간/토큰은 각 집합 전체 문제의 합계 평균이다. 토큰당 지연은prefill 포함 wall/output수이며 decode-only 시간과 다르다. 모든 평균·SD·최소·최대·95%CI는 ORACLE_RESULTS.csv, 반복별 값은 ORACLE_REPEATS.csv에 있다.

## 6. 동일 토큰 수의 실제 비용 비교

### replay8

| 방식 | 목표% | wall 감소% [95%CI] | CPU 감소% | 모델 matmul 추정 감소% | 5회 wall 비율 |
|---|---|---|---|---|---|
| Existing DLR | 고유 | -7.11 [-11.89, -2.07] | -15.14 | 0.013 | 1.112, 1.046, 1.077, 1.002, 1.123 |
| Ideal Oracle | 0 | -0.91 [-10.57, 4.56] | 7.36 | 0.000 | 1.008, 0.997, 1.016, 1.018, 1.006 |
| Ideal Oracle | 10 | 4.62 [-0.36, 8.83] | 9.80 | 0.799 | 0.991, 0.952, 0.956, 0.907, 0.965 |
| Ideal Oracle | 20 | 10.59 [5.12, 15.92] | 11.68 | 1.611 | 0.903, 0.881, 0.915, 0.854, 0.921 |
| Ideal Oracle | 30 | 11.65 [3.19, 16.12] | 5.96 | 2.422 | 0.885, 0.865, 0.879, 0.870, 0.920 |
| Ideal Oracle | 40 | 19.66 [9.46, 26.32] | 27.95 | 3.234 | 0.817, 0.797, 0.819, 0.758, 0.828 |
| Ideal Oracle | 50 | 22.89 [9.74, 30.54] | 20.37 | 4.067 | 0.772, 0.743, 0.794, 0.732, 0.818 |
| Random Skip | 0 | 1.96 [-1.85, 6.87] | 8.12 | 0.000 | 0.988, 0.995, 0.996, 0.927, 1.000 |
| Random Skip | 10 | 5.81 [1.47, 10.73] | 9.99 | 0.799 | 0.956, 0.924, 0.968, 0.909, 0.954 |
| Random Skip | 20 | 7.37 [-4.19, 13.45] | 13.23 | 1.611 | 0.923, 0.898, 0.957, 0.922, 0.933 |
| Random Skip | 30 | 13.47 [3.89, 19.57] | 12.39 | 2.422 | 0.862, 0.841, 0.876, 0.859, 0.891 |
| Random Skip | 40 | 18.97 [9.58, 25.60] | 20.81 | 3.234 | 0.832, 0.808, 0.826, 0.756, 0.833 |
| Random Skip | 50 | 22.73 [11.23, 29.32] | 19.84 | 4.067 | 0.779, 0.763, 0.792, 0.742, 0.790 |
| Oracle+Actual Overhead | 0 | -8.63 [-14.45, -2.83] | -14.21 | 0.000 | 1.159, 1.058, 1.052, 1.055, 1.107 |
| Oracle+Actual Overhead | 10 | -7.56 [-19.66, -0.74] | -7.47 | 0.799 | 1.030, 1.033, 1.047, 1.182, 1.082 |
| Oracle+Actual Overhead | 20 | 2.89 [-4.19, 8.66] | 2.85 | 1.611 | 0.979, 0.965, 0.962, 0.962, 0.987 |
| Oracle+Actual Overhead | 30 | 7.29 [1.16, 15.27] | 1.82 | 2.422 | 0.966, 0.893, 0.951, 0.862, 0.968 |
| Oracle+Actual Overhead | 40 | 7.66 [-14.61, 20.84] | 8.13 | 3.234 | 0.888, 0.860, 1.178, 0.766, 0.940 |
| Oracle+Actual Overhead | 50 | 16.81 [8.51, 22.42] | 4.37 | 4.067 | 0.858, 0.801, 0.824, 0.813, 0.865 |

### fresh8

| 방식 | 목표% | wall 감소% [95%CI] | CPU 감소% | 모델 matmul 추정 감소% | 5회 wall 비율 |
|---|---|---|---|---|---|
| Existing DLR | 고유 | -9.66 [-13.42, -3.51] | -27.84 | 0.010 | 1.071, 1.097, 1.069, 1.098, 1.148 |
| Ideal Oracle | 0 | -2.45 [-5.60, 1.26] | -5.09 | 0.000 | 1.059, 1.030, 1.006, 1.013, 1.016 |
| Ideal Oracle | 10 | 3.44 [0.25, 5.81] | -5.79 | 0.875 | 0.993, 0.961, 0.939, 0.940, 0.997 |
| Ideal Oracle | 20 | 7.48 [1.71, 10.77] | 2.03 | 1.790 | 0.911, 0.992, 0.905, 0.915, 0.902 |
| Ideal Oracle | 30 | 11.87 [4.28, 15.12] | -0.96 | 2.690 | 0.878, 0.893, 0.871, 0.874, 0.890 |
| Ideal Oracle | 40 | 18.74 [7.08, 22.91] | 21.84 | 3.605 | 0.816, 0.805, 0.813, 0.815, 0.813 |
| Ideal Oracle | 50 | 23.31 [10.84, 27.87] | 18.04 | 4.519 | 0.755, 0.782, 0.762, 0.768, 0.767 |
| Random Skip | 0 | -2.78 [-6.32, 0.81] | -17.66 | 0.000 | 1.027, 1.054, 1.002, 0.997, 1.059 |
| Random Skip | 10 | 2.80 [0.01, 5.19] | -5.47 | 0.875 | 0.978, 0.962, 0.973, 0.946, 1.001 |
| Random Skip | 20 | 6.68 [2.20, 9.97] | -4.69 | 1.790 | 0.981, 0.954, 0.893, 0.905, 0.935 |
| Random Skip | 30 | 12.97 [5.26, 16.34] | 3.68 | 2.690 | 0.872, 0.858, 0.851, 0.860, 0.912 |
| Random Skip | 40 | 19.07 [9.46, 22.80] | 18.13 | 3.605 | 0.828, 0.803, 0.793, 0.803, 0.821 |
| Random Skip | 50 | 11.28 [-28.07, 26.72] | 6.32 | 4.519 | 0.759, 1.354, 0.753, 0.772, 0.791 |
| Oracle+Actual Overhead | 0 | -9.49 [-14.98, -0.83] | -35.19 | 0.000 | 1.079, 1.117, 1.059, 1.065, 1.156 |
| Oracle+Actual Overhead | 10 | -2.94 [-6.90, 1.36] | -12.02 | 0.875 | 1.062, 1.026, 1.013, 1.014, 1.032 |
| Oracle+Actual Overhead | 20 | 3.25 [-1.27, 6.65] | -4.80 | 1.790 | 0.963, 0.977, 0.947, 0.980, 0.970 |
| Oracle+Actual Overhead | 30 | 8.16 [2.26, 11.80] | 2.62 | 2.690 | 0.908, 0.923, 0.907, 0.923, 0.931 |
| Oracle+Actual Overhead | 40 | 12.71 [4.99, 16.62] | 6.12 | 3.605 | 0.866, 0.865, 0.904, 0.866, 0.863 |
| Oracle+Actual Overhead | 50 | 17.74 [6.26, 22.58] | 10.24 | 4.519 | 0.800, 0.832, 0.808, 0.826, 0.847 |

### long4

| 방식 | 목표% | wall 감소% [95%CI] | CPU 감소% | 모델 matmul 추정 감소% | 5회 wall 비율 |
|---|---|---|---|---|---|
| Existing DLR | 고유 | -4.79 [-9.35, -0.32] | -3.13 | 0.718 | 1.068, 1.029, 1.081, 1.014, 1.048 |
| Ideal Oracle | 0 | -0.61 [-4.25, 1.52] | 2.36 | 0.000 | 1.005, 0.986, 1.040, 1.004, 0.998 |
| Ideal Oracle | 10 | 5.35 [3.27, 7.18] | 7.80 | 2.184 | 0.937, 0.952, 0.924, 0.942, 0.977 |
| Ideal Oracle | 20 | 10.43 [5.39, 14.20] | 9.88 | 4.384 | 0.878, 0.879, 0.924, 0.861, 0.939 |
| Ideal Oracle | 30 | 17.65 [11.71, 21.17] | 20.24 | 6.575 | 0.823, 0.823, 0.869, 0.800, 0.805 |
| Ideal Oracle | 40 | 22.82 [18.53, 25.18] | 25.31 | 8.774 | 0.758, 0.757, 0.816, 0.754, 0.777 |
| Ideal Oracle | 50 | 28.11 [25.16, 30.36] | 20.56 | 10.981 | 0.713, 0.712, 0.738, 0.722, 0.711 |
| Random Skip | 0 | -0.89 [-4.53, 2.20] | -2.81 | 0.000 | 0.986, 0.978, 1.051, 1.009, 1.025 |
| Random Skip | 10 | 4.13 [-0.34, 5.89] | -3.99 | 2.184 | 0.950, 0.953, 1.006, 0.944, 0.943 |
| Random Skip | 20 | 11.24 [8.45, 13.39] | 9.51 | 4.384 | 0.862, 0.904, 0.874, 0.888, 0.909 |
| Random Skip | 30 | 15.69 [11.71, 18.97] | 11.86 | 6.575 | 0.831, 0.868, 0.854, 0.804, 0.858 |
| Random Skip | 40 | 24.70 [22.28, 27.11] | 33.95 | 8.774 | 0.757, 0.742, 0.755, 0.748, 0.762 |
| Random Skip | 50 | 24.75 [6.27, 30.54] | 22.37 | 10.981 | 0.715, 0.710, 0.741, 0.706, 0.892 |
| Oracle+Actual Overhead | 0 | -9.90 [-13.57, -6.94] | -23.52 | 0.000 | 1.076, 1.095, 1.084, 1.099, 1.141 |
| Oracle+Actual Overhead | 10 | -1.62 [-3.90, 0.67] | -5.23 | 2.184 | 1.012, 1.034, 0.985, 1.013, 1.035 |
| Oracle+Actual Overhead | 20 | 1.93 [-3.26, 6.07] | -8.03 | 4.384 | 0.941, 1.002, 1.016, 0.955, 0.991 |
| Oracle+Actual Overhead | 30 | 8.40 [3.71, 12.79] | -0.49 | 6.575 | 0.922, 0.958, 0.912, 0.866, 0.920 |
| Oracle+Actual Overhead | 40 | 12.93 [0.55, 19.39] | 6.58 | 8.774 | 0.849, 1.001, 0.816, 0.791, 0.891 |
| Oracle+Actual Overhead | 50 | 16.27 [-13.64, 26.24] | 13.02 | 10.981 | 0.769, 0.763, 0.890, 0.750, 1.019 |

양의 감소율은 절감, 음수는 증가다. CI는 고유 질문 및 반복 축 paired bootstrap이다. 다수 rate/집합 탐색에 대한 동시 신뢰구간 보정은 하지 않았으므로 탐색적 nominal95%CI다. 작은20문제와 deterministic 반복을 독립100개 문제로 과장하지 않는다. 정확도 차이0 또는 bootstrap[0,0]만으로 모집단 비열등성을 입증하지 않는다.

### 모델·판단·cache·실제 생략 계수

| 집합 | 방식 | 목표% | 모델 step(s) | gate(s) | cache Python(s) | 표조회(s) | wall ms/토큰 | 실제 생략 MLP | 전체 layer-token 감소% |
|---|---|---|---|---|---|---|---|---|---|
| replay8 | Baseline | 고유 | 3.570 | 0.000 | 미측정 | 미측정 | 47.115 | 0 | 0.000 |
| replay8 | Existing DLR | 고유 | 3.826 | 0.121 | 미측정 | 미측정 | 50.464 | 3 | 0.015 |
| replay8 | Ideal Oracle | 0 | 3.604 | 0.000 | 0.009 | 0.000 | 47.544 | 0 | 0.000 |
| replay8 | Ideal Oracle | 10 | 3.406 | 0.000 | 0.009 | 0.000 | 44.936 | 187 | 0.928 |
| replay8 | Ideal Oracle | 20 | 3.192 | 0.000 | 0.008 | 0.000 | 42.126 | 377 | 1.870 |
| replay8 | Ideal Oracle | 30 | 3.153 | 0.000 | 0.009 | 0.000 | 41.624 | 567 | 2.812 |
| replay8 | Ideal Oracle | 40 | 2.867 | 0.000 | 0.006 | 0.000 | 37.853 | 757 | 3.755 |
| replay8 | Ideal Oracle | 50 | 2.751 | 0.000 | 0.006 | 0.000 | 36.331 | 952 | 4.722 |
| replay8 | Random Skip | 0 | 3.501 | 0.000 | 0.009 | 0.000 | 46.190 | 0 | 0.000 |
| replay8 | Random Skip | 10 | 3.363 | 0.000 | 0.009 | 0.000 | 44.376 | 187 | 0.928 |
| replay8 | Random Skip | 20 | 3.307 | 0.000 | 0.008 | 0.000 | 43.644 | 377 | 1.870 |
| replay8 | Random Skip | 30 | 3.088 | 0.000 | 0.008 | 0.000 | 40.768 | 567 | 2.812 |
| replay8 | Random Skip | 40 | 2.892 | 0.000 | 0.007 | 0.000 | 38.178 | 757 | 3.755 |
| replay8 | Random Skip | 50 | 2.757 | 0.000 | 0.006 | 0.000 | 36.406 | 952 | 4.722 |
| replay8 | Oracle+Actual Overhead | 0 | 3.880 | 0.125 | 0.013 | 0.000 | 51.183 | 0 | 0.000 |
| replay8 | Oracle+Actual Overhead | 10 | 3.836 | 0.125 | 0.021 | 0.004 | 50.676 | 187 | 0.928 |
| replay8 | Oracle+Actual Overhead | 20 | 3.468 | 0.088 | 0.010 | 0.000 | 45.752 | 377 | 1.870 |
| replay8 | Oracle+Actual Overhead | 30 | 3.310 | 0.088 | 0.010 | 0.000 | 43.680 | 567 | 2.812 |
| replay8 | Oracle+Actual Overhead | 40 | 3.295 | 0.077 | 0.022 | 0.001 | 43.507 | 757 | 3.755 |
| replay8 | Oracle+Actual Overhead | 50 | 2.969 | 0.069 | 0.008 | 0.000 | 39.193 | 952 | 4.722 |
| fresh8 | Baseline | 고유 | 3.289 | 0.000 | 미측정 | 미측정 | 44.555 | 0 | 0.000 |
| fresh8 | Existing DLR | 고유 | 3.607 | 0.111 | 미측정 | 미측정 | 48.860 | 2 | 0.011 |
| fresh8 | Ideal Oracle | 0 | 3.370 | 0.000 | 0.008 | 0.000 | 45.645 | 0 | 0.000 |
| fresh8 | Ideal Oracle | 10 | 3.175 | 0.000 | 0.008 | 0.000 | 43.023 | 179 | 1.018 |
| fresh8 | Ideal Oracle | 20 | 3.043 | 0.000 | 0.013 | 0.000 | 41.223 | 366 | 2.081 |
| fresh8 | Ideal Oracle | 30 | 2.897 | 0.000 | 0.008 | 0.000 | 39.264 | 550 | 3.128 |
| fresh8 | Ideal Oracle | 40 | 2.672 | 0.000 | 0.005 | 0.000 | 36.205 | 737 | 4.191 |
| fresh8 | Ideal Oracle | 50 | 2.521 | 0.000 | 0.005 | 0.000 | 34.170 | 924 | 5.255 |
| fresh8 | Random Skip | 0 | 3.380 | 0.000 | 0.009 | 0.000 | 45.793 | 0 | 0.000 |
| fresh8 | Random Skip | 10 | 3.197 | 0.000 | 0.009 | 0.000 | 43.306 | 179 | 1.018 |
| fresh8 | Random Skip | 20 | 3.069 | 0.000 | 0.008 | 0.000 | 41.578 | 366 | 2.081 |
| fresh8 | Random Skip | 30 | 2.862 | 0.000 | 0.007 | 0.000 | 38.776 | 550 | 3.128 |
| fresh8 | Random Skip | 40 | 2.661 | 0.000 | 0.005 | 0.000 | 36.058 | 737 | 4.191 |
| fresh8 | Random Skip | 50 | 2.914 | 0.000 | 0.008 | 0.000 | 39.530 | 924 | 5.255 |
| fresh8 | Oracle+Actual Overhead | 0 | 3.602 | 0.113 | 0.013 | 0.000 | 48.783 | 0 | 0.000 |
| fresh8 | Oracle+Actual Overhead | 10 | 3.386 | 0.093 | 0.010 | 0.000 | 45.864 | 179 | 1.018 |
| fresh8 | Oracle+Actual Overhead | 20 | 3.182 | 0.084 | 0.009 | 0.000 | 43.106 | 366 | 2.081 |
| fresh8 | Oracle+Actual Overhead | 30 | 3.021 | 0.071 | 0.008 | 0.000 | 40.921 | 550 | 3.128 |
| fresh8 | Oracle+Actual Overhead | 40 | 2.870 | 0.066 | 0.007 | 0.000 | 38.890 | 737 | 4.191 |
| fresh8 | Oracle+Actual Overhead | 50 | 2.703 | 0.057 | 0.006 | 0.000 | 36.650 | 924 | 5.255 |
| long4 | Baseline | 고유 | 4.317 | 0.000 | 미측정 | 미측정 | 38.952 | 0 | 0.000 |
| long4 | Existing DLR | 고유 | 4.525 | 0.132 | 미측정 | 미측정 | 40.818 | 98 | 0.858 |
| long4 | Ideal Oracle | 0 | 4.343 | 0.000 | 0.013 | 0.000 | 39.188 | 0 | 0.000 |
| long4 | Ideal Oracle | 10 | 4.086 | 0.000 | 0.012 | 0.000 | 36.868 | 298 | 2.609 |
| long4 | Ideal Oracle | 20 | 3.866 | 0.000 | 0.012 | 0.000 | 34.889 | 598 | 5.235 |
| long4 | Ideal Oracle | 30 | 3.554 | 0.000 | 0.010 | 0.000 | 32.078 | 897 | 7.852 |
| long4 | Ideal Oracle | 40 | 3.331 | 0.000 | 0.009 | 0.000 | 30.063 | 1197 | 10.478 |
| long4 | Ideal Oracle | 50 | 3.102 | 0.000 | 0.008 | 0.000 | 28.004 | 1498 | 13.113 |
| long4 | Random Skip | 0 | 4.356 | 0.000 | 0.014 | 0.000 | 39.299 | 0 | 0.000 |
| long4 | Random Skip | 10 | 4.138 | 0.000 | 0.014 | 0.000 | 37.342 | 298 | 2.609 |
| long4 | Random Skip | 20 | 3.831 | 0.000 | 0.012 | 0.000 | 34.573 | 598 | 5.235 |
| long4 | Random Skip | 30 | 3.639 | 0.000 | 0.011 | 0.000 | 32.841 | 897 | 7.852 |
| long4 | Random Skip | 40 | 3.250 | 0.000 | 0.007 | 0.000 | 29.329 | 1197 | 10.478 |
| long4 | Random Skip | 50 | 3.246 | 0.000 | 0.037 | 0.000 | 29.312 | 1498 | 13.113 |
| long4 | Oracle+Actual Overhead | 0 | 4.745 | 0.181 | 0.019 | 0.000 | 42.809 | 0 | 0.000 |
| long4 | Oracle+Actual Overhead | 10 | 4.388 | 0.150 | 0.015 | 0.000 | 39.582 | 298 | 2.609 |
| long4 | Oracle+Actual Overhead | 20 | 4.234 | 0.144 | 0.015 | 0.000 | 38.202 | 598 | 5.235 |
| long4 | Oracle+Actual Overhead | 30 | 3.954 | 0.124 | 0.013 | 0.000 | 35.678 | 897 | 7.852 |
| long4 | Oracle+Actual Overhead | 40 | 3.758 | 0.119 | 0.030 | 0.000 | 33.915 | 1197 | 10.478 |
| long4 | Oracle+Actual Overhead | 50 | 3.613 | 0.132 | 0.014 | 0.000 | 32.613 | 1498 | 13.113 |

## 7. 실제 판단 비용과 cache 비용

| 집합 | 목표% | Overhead/Ideal wall 비율 [95%CI] | 추가 CPU 비율 |
|---|---|---|---|
| replay8 | 0 | 1.077 [0.971,1.142] | 1.233 |
| replay8 | 10 | 1.128 [1.035,1.305] | 1.191 |
| replay8 | 20 | 1.086 [1.050,1.130] | 1.100 |
| replay8 | 30 | 1.049 [0.976,1.109] | 1.044 |
| replay8 | 40 | 1.149 [1.024,1.410] | 1.275 |
| replay8 | 50 | 1.079 [1.008,1.134] | 1.201 |
| fresh8 | 0 | 1.069 [1.006,1.121] | 1.286 |
| fresh8 | 10 | 1.066 [1.012,1.109] | 1.059 |
| fresh8 | 20 | 1.046 [1.010,1.078] | 1.070 |
| fresh8 | 30 | 1.042 [0.991,1.088] | 0.965 |
| fresh8 | 40 | 1.074 [1.009,1.126] | 1.201 |
| fresh8 | 50 | 1.073 [1.029,1.113] | 1.095 |
| long4 | 0 | 1.092 [1.046,1.139] | 1.265 |
| long4 | 10 | 1.074 [1.055,1.088] | 1.141 |
| long4 | 20 | 1.095 [1.036,1.149] | 1.199 |
| long4 | 30 | 1.112 [1.066,1.161] | 1.260 |
| long4 | 40 | 1.128 [1.036,1.304] | 1.251 |
| long4 | 50 | 1.165 [1.034,1.579] | 1.095 |

Overhead/Ideal은 실제로 gate 연산을 실행한 비교다. gate_wall은 입력 준비 후 거리 계산·GPU eval·item 회수를 포함하고, pre_gate_sync는 앞선 모델 연산 대기를 포함한다. 후자를 전부 추가 overhead로 해석하거나 두 값을 전체 wall에 다시 더하지 않는다. cache_python_wall/route_lookup은 Python·MLX graph dispatch 시간이지 GPU memcpy 시간이 아니다.

추가 gate는 기존 거리/eval/item 계산과 입력 anchor 관리를 실제 수행하지만, event dict 로깅·age bookkeeping까지 원래 구현과 byte 단위로 같지는 않다. 따라서 E−Ideal을 원래 DLR의 모든 Python/로깅 비용의 정확한 인과 분해로 표현하지 않는다.

Overhead/Ideal의 자연생성600쌍에서 token열 차이 0개, 동일열 실제 예측600쌍에서 차이 0개를 기록했다. 강제된 최종 문자열 일치만으로 연산 동일성을 주장하지 않았다.

GPU activation을 host로 내보내는 경로는 주 추론에 없다. output anchor는 logical84KiB, 추가4 input anchor는12KiB다. slice/contiguous가 backing storage를 유지하거나 필요 시 복사할 수 있어 실제 메모리 이동량은 별도 미측정이다. MLX peak는 실제 allocator 관측이지만 장치 전체 memory가 아니고 RSS는 프로세스 lifetime high-water라 방식별 독립 peak가 아니다.

모델 matmul FLOPs는 MAC=2의 분석 추정이고 실제 hardware FLOPs가 아니다. 실제 native MLP 호출 우회 수는 별도로 검증했다. gate 산술약5*d+4를 별도 저장하되 norm/activation/lookup/cache/메모리traffic/실제 instruction 전체를 계산한 것은 아니다. native projection call 수를 GPU kernel launch 수로 바꾸지 않는다. 현재 개발 환경에서 xcrun --find xctrace가 실패해 해당 도구를 사용할 수 없었다. 실제 GPU kernel count/GPU timestamp·전력·금전 비용은 미측정이다.

## 8. 별도 침습적 병목 프로파일

조건별1회 워밍업 후 측정한 보조 프로파일 `PROFILE_67704bf49cff41f88b7c42d8cf63e8fd`를 사용했다. 첫 프로파일의 cold-start 가능성 자료도 보존했지만 아래 표에는 혼합하지 않았다. 주4000건과 별도 실행이다.

| 집합 | 방식 | 목표% | 동기화 wall(s) | Attention 구간(s) | MLP 구간(s) | cache 동기화 구간(s) |
|---|---|---|---|---|---|---|
| replay8 | Baseline | 고유 | 0.388 | 0.069 | 0.252 | 0.000 |
| replay8 | Ideal Oracle | 0 | 0.282 | 0.049 | 0.193 | 0.008 |
| replay8 | Ideal Oracle | 50 | 0.194 | 0.037 | 0.121 | 0.007 |
| replay8 | Oracle+Actual Overhead | 50 | 0.255 | 0.047 | 0.167 | 0.009 |
| replay8 | Existing DLR | 고유 | 0.212 | 0.037 | 0.137 | 미측정 |
| long4 | Baseline | 고유 | 1.634 | 0.307 | 0.886 | 0.000 |
| long4 | Ideal Oracle | 0 | 1.657 | 0.309 | 0.887 | 0.047 |
| long4 | Ideal Oracle | 50 | 1.174 | 0.310 | 0.491 | 0.026 |
| long4 | Oracle+Actual Overhead | 50 | 1.197 | 0.312 | 0.489 | 0.026 |
| long4 | Existing DLR | 고유 | 1.610 | 0.304 | 0.840 | 미측정 |

각 Attention/MLP/cache에 동기화를 추가한 wall 구간이다. 정상 pipeline overlap을 바꾸므로 주 성능값·GPU 실행시간·원인별 정확한 기여율로 취급하지 않는다. Existing wrapper의 Attention은 inner 모듈만, 다른 profile 경로는 input norm/attention/residual add를 포함하므로 그 bucket을 직접 동일 범위로 비교하지 않는다. MLP 구간은 post-norm 입력 준비 뒤 측정한다. 작은 모델의 MLP 절감 여지는 matched-stream 곡선에서 직접 보고, Python/cache/Attention/head의 잔여 비용은 이 프로파일을 보조로 사용한다.

## 9. 품질 손상과 판정

| 집합 | 목표% | Oracle 관측 기회 | 정답 손실 문제 | 동일열 시간 감소% | 동일열 CI 상한 | 5회 모두 빨라짐 |
|---|---|---|---|---|---|---|
| replay8 | 10 | False | 0 | 4.62 | 1.004 | True |
| replay8 | 20 | False | 2 | 10.59 | 0.949 | True |
| replay8 | 30 | False | 2 | 11.65 | 0.968 | True |
| replay8 | 40 | False | 2 | 19.66 | 0.905 | True |
| replay8 | 50 | False | 5 | 22.89 | 0.903 | True |
| fresh8 | 10 | False | 0 | 3.44 | 0.998 | True |
| fresh8 | 20 | False | 2 | 7.48 | 0.983 | True |
| fresh8 | 30 | False | 2 | 11.87 | 0.957 | True |
| fresh8 | 40 | False | 2 | 18.74 | 0.929 | True |
| fresh8 | 50 | False | 6 | 23.31 | 0.892 | True |
| long4 | 10 | False | 1 | 5.35 | 0.967 | True |
| long4 | 20 | False | 1 | 10.43 | 0.946 | True |
| long4 | 30 | False | 2 | 17.65 | 0.883 | True |
| long4 | 40 | False | 2 | 22.82 | 0.815 | True |
| long4 | 50 | False | 2 | 28.11 | 0.748 | True |

기회 판정은 평균5% 이상 matched-stream 감소·CI 상한<1·5회 모두 개선·관측 내용/형식 비감소·개별 Baseline 정답 손실 없음·실제 MLP우회를 동시에 요구한다. Baseline 정답률0인 집합은 품질 성공으로 처리하지 않는다. 이는 소규모 관측 기준이며 일반화나 상용 정책의 정확도 유지 증명이 아니다.

### replay8/Oracle20%/13315d7efd
- 질문: Encode this as JSON with exactly label and enabled: label is unit30200; enabled is false. Return only JSON.
- 기준: {'label': 'unit30200', 'enabled': False}
- Baseline: '```json\n{\n  "label": "unit30200",\n  "enabled": false\n}\n```'
- Oracle: '``````json\n\n``````jsonaskell\n\n```json\n{\n}\n```json\n\n``````json'; 출력24토큰, 완료 사유 eos.

### replay8/Oracle20%/3e2df56dff
- 질문: Encode this as JSON with exactly label and enabled: label is unit30203; enabled is true. Return only JSON.
- 기준: {'label': 'unit30203', 'enabled': True}
- Baseline: '```json\n{\n  "label": "unit30203",\n  "enabled": true\n}\n```'
- Oracle: '```jsonjson\n{\n"label": " unit300203",\n"enabled": true\n}\n```'; 출력25토큰, 완료 사유 eos.

### replay8/Oracle30%/13315d7efd
- 질문: Encode this as JSON with exactly label and enabled: label is unit30200; enabled is false. Return only JSON.
- 기준: {'label': 'unit30200', 'enabled': False}
- Baseline: '```json\n{\n  "label": "unit30200",\n  "enabled": false\n}\n```'
- Oracle: '```{\njsonjson\n label: "unit3300",\n enabled: false false\n\n``````'; 출력24토큰, 완료 사유 eos.

### replay8/Oracle30%/3e2df56dff
- 질문: Encode this as JSON with exactly label and enabled: label is unit30203; enabled is true. Return only JSON.
- 기준: {'label': 'unit30203', 'enabled': True}
- Baseline: '```json\n{\n  "label": "unit30203",\n  "enabled": true\n}\n```'
- Oracle: '``````jsonjson\njson1000:label":": unit3000000; enabledtruefalse```'; 출력28토큰, 완료 사유 eos.

### replay8/Oracle40%/13315d7efd
- 질문: Encode this as JSON with exactly label and enabled: label is unit30200; enabled is false. Return only JSON.
- 기준: {'label': 'unit30200', 'enabled': False}
- Baseline: '```json\n{\n  "label": "unit30200",\n  "enabled": false\n}\n```'
- Oracle: '``````jsonjson\nlabel ""enabled":" "falseenabled\n"``````'; 출력19토큰, 완료 사유 eos.

### replay8/Oracle40%/3e2df56dff
- 질문: Encode this as JSON with exactly label and enabled: label is unit30203; enabled is true. Return only JSON.
- 기준: {'label': 'unit30203', 'enabled': True}
- Baseline: '```json\n{\n  "label": "unit30203",\n  "enabled": true\n}\n```'
- Oracle: '```{\n{\n""{\n"label,\n"enabled"\n\n,\ntruetrue\n\n\n"\n"\n"\n"\n"\n"\n"\n"\n"\n"\n"\n"\n"\n'; 출력32토큰, 완료 사유 max_tokens.

### replay8/Oracle50%/a19a8fbe27
- 질문: Python snippet 30103: / ```python / x = [11, 54, 67] / x.append(2) / print(len(x)) / ``` / What is printed? / A) 4 / B) 5 / C) Error / D) 3 / Return only the option letter.
- 기준: 'A'
- Baseline: 'A)'
- Oracle: 'AB'; 출력3토큰, 완료 사유 eos.

### replay8/Oracle50%/0a8d2cf244
- 질문: Memo 30104: Nemi30104 owns a silver compass. Nemi30104 lends the compass to Pavo30104 for one day; ownership does not change. Who owns the compass during that day? / A) Nobody / B) Nemi30104 / C) Selu30104 / D) Pavo30104 / Return only the option letter.
- 기준: 'B'
- Baseline: 'B) Nemi30104'
- Oracle: 'BD)'; 출력4토큰, 완료 사유 eos.

### replay8/Oracle50%/4bead11d7a
- 질문: Log 30107: A package arrives on Tuesday. Nemi30107 stores it until Friday, when Pavo30107 collects it. On which day is it collected? / A) Tuesday / B) Friday / C) Wednesday / D) Saturday / Return only the option letter.
- 기준: 'B'
- Baseline: 'B) Friday'
- Oracle: 'BB)) Friday'; 출력6토큰, 완료 사유 eos.

### replay8/Oracle50%/13315d7efd
- 질문: Encode this as JSON with exactly label and enabled: label is unit30200; enabled is false. Return only JSON.
- 기준: {'label': 'unit30200', 'enabled': False}
- Baseline: '```json\n{\n  "label": "unit30200",\n  "enabled": false\n}\n```'
- Oracle: '``````{\njson<{\n\n{\n{\n"   "labellabel":": " "unit330",\n",\n{\n{\n"label"\n":":": "'; 출력32토큰, 완료 사유 max_tokens.

### replay8/Oracle50%/3e2df56dff
- 질문: Encode this as JSON with exactly label and enabled: label is unit30203; enabled is true. Return only JSON.
- 기준: {'label': 'unit30203', 'enabled': True}
- Baseline: '```json\n{\n  "label": "unit30203",\n  "enabled": true\n}\n```'
- Oracle: '``````{\njson<{\n\n{\n" label\n\n{\nenabled"{\n}\nenabled}\n}\n}\n}\n}\n`````````'; 출력27토큰, 완료 사유 eos.

### fresh8/Oracle20%/b4d17639b2
- 질문: Return only a JSON object with keys code and quantity. Values: code="oracle79200", quantity=41.
- 기준: {'code': 'oracle79200', 'quantity': 41}
- Baseline: '```json\n{\n  "code": "oracle79200",\n  "quantity": 41\n}\n```'
- Oracle: '```{\njson\n{\n"code": "oracle792000",\n"quantity": 411\n}\n}\n```'; 출력29토큰, 완료 사유 eos.

모든 응답·오답·개별 전환·순서는 ORACLE_CASES/FAILURES CSV에 보존했다. 위 예시나 가장 유리한 rate만으로 결론을 정하지 않았다.

## 10. 한계와 다음 연구 권고

**최종 C: 실패.** 공통 통과 rate는 [], 통과 집합/rate 수는 0다. 판단은 이 native 구현/표/소규모 synthetic 데이터에만 해당한다. throughput만 빨라지고 정확도가 나빠진 rate는 성공이 아니다.

더 큰 모델에서는 MLP weight traffic 대비 작은 lookup 비용의 비율이 유리해질 가능성이 있으나 다른 모델은 실행하지 않아 미검증이다. 긴 출력은 decode 비중이 커질 수 있지만 재사용 오차 누적과 EOS 오류도 커질 수 있다. 메모리나 전력 절감을 실측하지 않고 주장하지 않는다.

다음 단계는 이 결과에서 품질을 유지한 표가 존재하는 경우에만 그 선택을 재현하는 저비용 사전 신호를 별도 학습/평가 데이터로 검증하는 것이다. 조건이 없으면 gate 최적화를 확대하기 전에 full-model native kernel/compiled execution과 다른 비용 절감 방향을 우선 검토한다. 이번에 새 서비스 알고리즘이나 독립적 기술을 발견했다고 주장하지 않는다.

## 11. 재현 명령과 보존

자동 테스트: 전체 snapshot256개 중252개 통과, NumPy 환경 사유4개 건너뜀(76.046초). 해당4개 수치 검사와 새 Oracle 테스트를 포함한 별도 MLX/NumPy 환경13개는 전부 통과(0.077초). 실제 모델 smoke는0% native 출력 일치·생략 계수·MLP trap 우회·fresh positive control을 확인했다.

```sh
.alsr-venv/bin/python -B -m src.oracle_experiment ./data/oracle_8ef05974726245a1b273f178fc668cb7
.venv/bin/python -B -m src.oracle_analysis ./data/oracle_8ef05974726245a1b273f178fc668cb7
.alsr-venv/bin/python -B -m src.oracle_profile ./data/oracle_8ef05974726245a1b273f178fc668cb7
.venv/bin/python -B -m src.oracle_report <새 ANALYSIS 경로>
.venv/bin/python -B -m unittest tests.test_oracle
.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json
```

기존 10308개 파일 SHA256를 보존했다. checkpoints는 append-only이며 재시작은 완료 요청을 재생성하지 않는다. 표 생성·모델 로드·워밍업·별도 profile은 주 비교 밖에서 기록한다. 새 데이터는 최초 저장 후 다시 미사용 문제로 주장하지 않는다.

[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [synchronize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html). 공식 문서는 비동기/동기화 해석의 근거이며 본 실험 성능의 근거는 원본 측정 파일이다.

전체 실제 모델 재실행은 다음 명령으로 새 UUID에 생성한다. 같은 문제를 재사용하므로 신규 평가라고 주장하지 않는다.
```sh
.alsr-venv/bin/python -B -m src.oracle_replay ./data/oracle_8ef05974726245a1b273f178fc668cb7/MANIFEST.json
```
