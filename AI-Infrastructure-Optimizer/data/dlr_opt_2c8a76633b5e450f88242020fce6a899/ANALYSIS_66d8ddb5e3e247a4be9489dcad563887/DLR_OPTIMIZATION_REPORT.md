# DLR_OPTIMIZATION_REPORT

**최종 결론: C — 최적화 실패.** 실행 `dlr_opt_2c8a76633b5e450f88242020fce6a899`. 무료 로컬 실제 추론. 단일 M4 16GiB, Qwen2.5-1.5B-Instruct BF16. 기존 파일/가중치/데이터 보존; 유료 API 및 GitHub 업로드 없음.

최소 관측 기준을 충족한 제한적 조건 수: 0. 작은 표본, 단일 장치, synthetic 문제, Device fresh-kernel 수치 차이 때문에 모집단 무손실·일반적인 전체 연산 감소를 입증하지 않는다. 관측 기준을 통과하지 못한 방식/집합도 모두 아래에 포함한다.

## 1. 기존 DLR의 병목

기존 src/dlr_runtime.py는 8/12/16/24번층에서 Attention 및 KV cache를 정상 계산한 다음, post-attention RMSNorm의 현재 입력과 최근 fresh MLP anchor 입력의 상대 L2 거리가0.5이하이면 gate/up/down projection을 포함한 MLP residual 전체를 이전 값으로 대체한다. 최대1토큰 연속 재사용, prefill 재사용 없음. KV Cache와 달리 새 토큰의 FFN 근사값을 사용한다.

각 후보 판단 시 mx.eval(z)로 앞선 GPU 작업을 기다리고, 거리 reduction을 mx.eval한 뒤 distance.item()으로 Python 분기한다. 이전 보고서에서 자연생성 평균 DLR 재사용은 기존32문제8회/신규32문제16회였고, 모델 matmul 추정 감소는0.00858%/0.01972%였는데 wall은5.9%/7.8% 증가했다. 이는 이번 새 측정과 섞지 않은 과거 결과다. gate당 대략5*d+4개 scalar arithmetic 대비, 재사용하면 MLP3projection의6*d*I FLOPs를 생략한다. Qwen d=1536,I=8960이면 gate약7684ops, MLP약82575360FLOPs다. 하지만 낮은 재사용 빈도, host round-trip, pipeline serialization 및 cache/선택 비용 때문에 이 산술 비율만으로 속도를 예측할 수 없다.

특히 pre_gate_sync 시간에는 원래 필요한 Attention/앞선 모델 연산 대기가 포함된다. 이를 전부 추가 오버헤드로 더하거나 wall에서 빼지 않는다. gate_wall과 pre_gate_sync는 서로 다른 범위이며 둘 모두 전체 wall에 포함돼 있다. 단일 profiler 분해 없이 원인별 정확한 인과 비중은 식별하지 못한다.

## 2. 실제 구현과 가능성

A Baseline/B 기존 DLR은 원래 DLRRuntime을 그대로 호출해 이전 정책·native 커널을 유지했다. C Device는 동일 거리 기준을 MLX array로 계산하고, 자체 Metal GEMV 커널의 uniform if(reuse) 분기가 weight 읽기와 MAC loop 이전에 return하도록 구현했다. gate/up과 down의 GPU 출력 counter는 fresh일 때 합3, reuse일 때0이다. mx.where(fresh_mlp,cached)로 양쪽 projection을 모두 계산하는 가짜 생략 방식이 아니다. 재사용 중에도 커널 launch·중간 zero write·cached output copy는 발생한다.

설치된 MLX에는 mx.cond가 없었지만 custom Metal kernel은 가능했다. 따라서 device 방식 자체가 불가능하다고 처리하지 않았다. 동적 선택/anchor 갱신에는 mx.where를 쓰되, 고비용 projection 자체는 커널 내부에서 조건부 실행한다. 세부 소스는 dlr_opt_kernels.py다.

D Lightweight는 이미 argmax 과정에서 읽은 현재 입력 token ID가 최근 fresh anchor의 ID와 같고 age=0일 때 네 후보 MLP 출력을 재사용한다. 추가 LLM 호출·hidden-state 거리 계산·GPU scalar read가 없다. 단, 동일 token ID라도 attention 문맥과 hidden state는 다르므로 안전성 보장은 없고 실패는 실제 출력으로 판정한다. 후보4층과 최대연속1회 제한은 유지했다.

Device/Lightweight forward에는 .item 또는 mx.eval로 gate를 읽는 코드가 없다. 일반 greedy token 읽기(토큰당1회)와 요청 시작/끝 동기화는 여전히 있다. Device branch counter는 요청 종료 시1 batch로 회수하며 그 시간도 wall에 포함한다. 따라서 모든 호스트 동기화를 제거했다고 주장하지 않는다.

Device fresh GEMV는 stock MLX GEMV와 reduction/rounding 순서가 다르다. 개발 random activation에서 native MLP 최대 절대 차이0.125, 상대 L2약0.00388을 관측했다. 개발용 두 prompt의 forced-fresh 생성 token은 Baseline과 같았지만, 모든 평가 질문에서 fresh-kernel parity가 입증된 것은 아니다. Device와 기존 DLR 차이는 sync뿐 아니라 kernel 구현 및 수치 오차가 섞인다.

별도 사후 fresh-kernel 점검은 고정된 부분집합 8문제를3회, 총24건 실행했다. 재사용을 끈 Device의 Baseline token열 일치는 24/24건이었다. 이를 학습/임계값 조정에 쓰지 않았으며 모든 평가 문제의 수치 동일성을 증명하지 않는다. 주 비교와 실행 순서가 달라 이 추가 점검의 latency는 주 성능 비교에서 제외했다.

개발 파일럿에서는 반복 hello8회를 요청했을 때 Baseline/Lightweight는8개, 기존 DLR/Device는9개를 출력했다. 이는 실제 재사용의 품질 실패 사례이며 버리지 않았다. 최초 Metal 컴파일은 scalar reuse 입력을 pointer처럼 사용해 실패했고, 입력을 길이1 array로 수정한 뒤 통과했다. 실패한 protocol 폴더도 보존했다. 평가 중 정책/커널/채점을 조정하지 않았다.

[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [where](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.where.html), [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html). 기존 [activation caching 연구](https://arxiv.org/abs/2512.16843)와 겹치는 로컬 엔지니어링 실험이며 새 발명의 증거가 아니다.

## 3. 공정 조건과 평가 범위

기존32문제와 신규32문제, 반복복사 long_high4문제와 서로 다른 단어 복사 long_low4문제를 사용했다. high/low는 사전 입력 설계 가설이며 결과를 보고 유리한 문제를 선별한 것이 아니다. 전체72문제×4방식×3회 자연생성864건, 동일 Baseline token stream을 teacher forcing으로 실제 실행하는864건, 총1728건이다. 시드2026110605. 같은 system/user template, greedy EOS, short32/long96토큰 상한. 모든 방식에 동일하게 적용했다. 정답은 scoring에만 사용한다. 복사 과제의 원문이 prompt에 포함된 것은 과제 정의이며, 채점 결과나 정답 여부를 gate에 제공한 것은 아니다.

워밍업/컴파일 후 측정하고 질문 및 방식 순서를 무작위 교차했다. schedule·sequence·position·session·원본 체크포인트를 보존한다. 저장·모델 로드·개발 audit·warmup 시간은 요청 latency 밖에서 별도 기록했다. 각 요청은 fresh KV/cache 상태에서 시작한다. 장치 온도/OS 부하를 완전히 통제하지 못했으며 반복3회는 독립적 모델 출력 표본3개가 아니다.

신규 exact stem 중복은 과거 JSON/CSV를 재귀 검사했다. 합성 템플릿/문제 개념의 유사성은 남아 외부 일반화를 증명하지 않는다. MCQ 선택지 추출, typed JSON, 사전 제한한 color 표현, 40단어 전체 literal copying으로 채점했다. 형식과 내용은 분리한다. 긴 답변의 일부 prefix만 맞아도 정답으로 처리하지 않는다. 열린 문장의 의미·환각 전반을 평가하는 데이터는 아니다.

가중치 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 패키지 `{"mlx": "0.32.4", "mlx-lm": "0.32.2", "numpy": "2.5.4", "transformers": "5.22.1"}`. 모든 pinned 가중치/설정 파일 해시를 로드 시 확인했다.

## 4. 전체 자연생성 비교

### all72

| 방식 | 정답률 | 형식 준수 | 출력 token열 일치 | wall 평균±SD(s) | 모델시간(s) | CPU 평균±SD(s) | 입력/출력토큰 | ms/출력토큰 | MLX peak MiB | 실제재사용/반복 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 62.50% | 36.11% | 100.00% | 34.072 ± 0.792 | 34.017 | 7.319 ± 1.422 | 5704/854 | 39.897 | 3083.87 | 0 |
| 기존 DLR | 61.11% | 34.72% | 97.22% | 35.594 ± 0.981 | 35.539 | 7.762 ± 1.810 | 5704/856 | 41.581 | 3086.12 | 333 |
| Device-Resident | 61.11% | 34.72% | 97.22% | 34.319 ± 0.752 | 34.154 | 7.783 ± 1.383 | 5704/856 | 40.092 | 3086.41 | 333 |
| Lightweight | 62.50% | 36.11% | 95.83% | 35.967 ± 0.989 | 35.911 | 8.098 ± 1.620 | 5704/911 | 39.481 | 3085.21 | 436 |

### replay32

| 방식 | 정답률 | 형식 준수 | 출력 token열 일치 | wall 평균±SD(s) | 모델시간(s) | CPU 평균±SD(s) | 입력/출력토큰 | ms/출력토큰 | MLX peak MiB | 실제재사용/반복 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 59.38% | 18.75% | 100.00% | 12.154 ± 0.356 | 12.130 | 2.577 ± 0.694 | 2615/288 | 42.201 | 3083.87 | 0 |
| 기존 DLR | 59.38% | 18.75% | 100.00% | 12.759 ± 0.442 | 12.734 | 2.705 ± 0.970 | 2615/288 | 44.301 | 3086.12 | 8 |
| Device-Resident | 59.38% | 18.75% | 100.00% | 12.338 ± 0.280 | 12.274 | 2.761 ± 0.639 | 2615/288 | 42.839 | 3086.41 | 8 |
| Lightweight | 59.38% | 18.75% | 100.00% | 12.180 ± 0.345 | 12.155 | 2.604 ± 0.689 | 2615/288 | 42.293 | 3085.21 | 4 |

### fresh32

| 방식 | 정답률 | 형식 준수 | 출력 token열 일치 | wall 평균±SD(s) | 모델시간(s) | CPU 평균±SD(s) | 입력/출력토큰 | ms/출력토큰 | MLX peak MiB | 실제재사용/반복 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 71.88% | 53.12% | 100.00% | 12.147 ± 0.323 | 12.125 | 2.456 ± 0.249 | 2331/301 | 40.354 | 3077.04 | 0 |
| 기존 DLR | 71.88% | 53.12% | 100.00% | 12.772 ± 0.216 | 12.750 | 2.606 ± 0.183 | 2331/301 | 42.432 | 3081.38 | 15 |
| Device-Resident | 71.88% | 53.12% | 100.00% | 12.320 ± 0.313 | 12.262 | 2.632 ± 0.281 | 2331/301 | 40.931 | 3080.87 | 15 |
| Lightweight | 71.88% | 53.12% | 100.00% | 12.090 ± 0.178 | 12.070 | 2.396 ± 0.164 | 2331/301 | 40.167 | 3079.64 | 12 |

### long_high

| 방식 | 정답률 | 형식 준수 | 출력 token열 일치 | wall 평균±SD(s) | 모델시간(s) | CPU 평균±SD(s) | 입력/출력토큰 | ms/출력토큰 | MLX peak MiB | 실제재사용/반복 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 50.00% | 50.00% | 100.00% | 5.994 ± 0.113 | 5.989 | 1.495 ± 0.285 | 364/164 | 36.549 | 3066.76 | 0 |
| 기존 DLR | 25.00% | 25.00% | 50.00% | 6.034 ± 0.175 | 6.029 | 1.540 ± 0.286 | 364/166 | 36.349 | 3067.94 | 302 |
| Device-Resident | 25.00% | 25.00% | 50.00% | 5.828 ± 0.143 | 5.801 | 1.537 ± 0.280 | 364/166 | 35.107 | 3067.84 | 302 |
| Lightweight | 50.00% | 50.00% | 25.00% | 7.925 ± 0.474 | 7.917 | 2.258 ± 0.386 | 364/221 | 35.859 | 3066.69 | 420 |

### long_low

| 방식 | 정답률 | 형식 준수 | 출력 token열 일치 | wall 평균±SD(s) | 모델시간(s) | CPU 평균±SD(s) | 입력/출력토큰 | ms/출력토큰 | MLX peak MiB | 실제재사용/반복 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 25.00% | 25.00% | 100.00% | 3.778 ± 0.198 | 3.774 | 0.791 ± 0.349 | 394/101 | 37.401 | 3068.56 | 0 |
| 기존 DLR | 25.00% | 25.00% | 100.00% | 4.029 ± 0.294 | 4.025 | 0.911 ± 0.532 | 394/101 | 39.891 | 3070.75 | 8 |
| Device-Resident | 25.00% | 25.00% | 100.00% | 3.833 ± 0.161 | 3.817 | 0.853 ± 0.401 | 394/101 | 37.955 | 3071.00 | 8 |
| Lightweight | 25.00% | 25.00% | 100.00% | 3.772 ± 0.185 | 3.768 | 0.840 ± 0.435 | 394/101 | 37.348 | 3069.81 | 0 |

wall·CPU·모델시간·토큰은 각 집합의 전체 문제 합계다. 출력 토큰 수는 실제 실행한 EOS 생성 단계도 포함한다. 토큰당 시간은 prefill을 포함한 전체 wall/output 수이며 decode-only 지연이 아니다. RSS는 프로세스 lifetime peak라 방식별 독립 peak가 아니고, MLX peak도 장치 전체 메모리가 아니다.

## 5. 동일 토큰 수·동일 입력 위치 비교

자연생성 1회차 Baseline의 토큰열을 고정해 모든 방식에서 실제 forward를 실행했다. 고정 응답의 채점을 독립적인 생성 정답률로 이용하지 않는다. 입력·출력 토큰 수와 prefix 위치가 일치하는지 감사했다.

### 고정열 비교 대상: Baseline

| 집합 | 방식 | wall 감소율 [95%CI] | CPU 감소율 | 모델 matmul 추정 감소 | gate 포함 산술 proxy 감소 | 3회 wall 비율 |
|---|---|---|---|---|---|---|
| replay32 | 기존 DLR | -7.33% [-8.95, -5.29] | -9.97% | 0.0086% | 0.0085% | 1.0815, 1.0630, 1.0755 |
| replay32 | Device-Resident | -1.35% [-2.59, 0.05] | 1.31% | 0.0086% | 0.0085% | 1.0124, 1.0096, 1.0187 |
| replay32 | Lightweight | 0.23% [-0.92, 1.82] | 2.49% | 0.0043% | 0.0043% | 0.9971, 0.9920, 1.0041 |
| fresh32 | 기존 DLR | -5.19% [-8.49, -1.57] | -10.32% | 0.0177% | 0.0176% | 1.0530, 1.0809, 1.0217 |
| fresh32 | Device-Resident | -0.37% [-2.01, 1.70] | -1.87% | 0.0177% | 0.0176% | 1.0068, 1.0123, 0.9921 |
| fresh32 | Lightweight | 1.07% [-0.29, 2.44] | 1.15% | 0.0142% | 0.0142% | 0.9923, 0.9918, 0.9836 |
| long_high | 기존 DLR | -1.63% [-4.38, 0.95] | -10.27% | 1.6986% | 1.6984% | 1.0119, 1.0362, 1.0002 |
| long_high | Device-Resident | 1.69% [0.07, 3.85] | -7.01% | 1.6986% | 1.6983% | 0.9877, 0.9914, 0.9700 |
| long_high | Lightweight | 3.23% [0.71, 5.28] | 2.02% | 1.7439% | 1.7439% | 0.9556, 0.9698, 0.9776 |
| long_low | 기존 DLR | -16.43% [-61.72, -6.40] | -23.02% | 0.0492% | 0.0490% | 1.3538, 1.0728, 1.0771 |
| long_low | Device-Resident | -1.68% [-6.41, 0.71] | -6.98% | 0.0492% | 0.0490% | 1.0256, 0.9932, 1.0329 |
| long_low | Lightweight | -0.18% [-9.05, 3.36] | 6.14% | 0.0000% | 0.0000% | 1.0081, 0.9926, 1.0052 |

### 고정열 비교 대상: 기존 DLR

| 집합 | 방식 | wall 감소율 [95%CI] | CPU 감소율 | 모델 matmul 추정 감소 | gate 포함 산술 proxy 감소 | 3회 wall 비율 |
|---|---|---|---|---|---|---|
| replay32 | Device-Resident | 5.57% [4.03, 7.06] | 10.25% | 0.0000% | -0.0000% | 0.9361, 0.9498, 0.9471 |
| replay32 | Lightweight | 7.04% [5.59, 8.37] | 11.34% | -0.0043% | -0.0042% | 0.9220, 0.9331, 0.9336 |
| fresh32 | Device-Resident | 4.58% [2.42, 6.77] | 7.66% | 0.0000% | -0.0000% | 0.9562, 0.9365, 0.9711 |
| fresh32 | Lightweight | 5.96% [2.76, 8.37] | 10.39% | -0.0035% | -0.0034% | 0.9424, 0.9176, 0.9628 |
| long_high | Device-Resident | 3.26% [2.34, 4.77] | 2.95% | 0.0000% | -0.0002% | 0.9761, 0.9568, 0.9698 |
| long_high | Lightweight | 4.78% [2.30, 7.15] | 11.14% | 0.0461% | 0.0463% | 0.9444, 0.9359, 0.9773 |
| long_low | Device-Resident | 12.67% [2.25, 36.37] | 13.04% | 0.0000% | -0.0000% | 0.7576, 0.9258, 0.9589 |
| long_low | Lightweight | 13.96% [3.53, 35.76] | 23.70% | -0.0492% | -0.0490% | 0.7446, 0.9252, 0.9332 |

양의 감소율은 감소, 음수는 증가다. CI는 질문과 반복 축을 함께 재표집한 paired bootstrap이며, 3회 평균·SD는 RESULTS에 모두 저장했다. long은 각 4문제로 특히 불확실성이 크다. 정확도 차이 0 또는 CI[0,0]으로 모집단 비열등성을 증명하지 않는다.

## 6. 판단 추가 비용과 실제 생략

| 집합 | 방식 | host gate 읽기/반복 | pre-gate eval/반복 | 거리 GPU 평가/반복 | ID 비교/반복 | 기존 gate 시간(s) | device graph 구성 CPU(s) | device gate GPU 시간 | 추정 gate 산술 ops |
|---|---|---|---|---|---|---|---|---|---|
| replay32 | Baseline | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 미측정 | 0 |
| replay32 | 기존 DLR | 1016 | 1016 | 1016 | 0 | 0.3109 | 0.0000 | 미측정 | 7806944 |
| replay32 | Device-Resident | 0 | 0 | 1024 | 0 | 0.0000 | 0.0172 | 미측정 | 7868416 |
| replay32 | Lightweight | 0 | 0 | 0 | 256 | 0.0010 | 0.0000 | 미측정 | 0 |
| fresh32 | Baseline | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 미측정 | 0 |
| fresh32 | 기존 DLR | 1061 | 1061 | 1061 | 0 | 0.3169 | 0.0000 | 미측정 | 8152724 |
| fresh32 | Device-Resident | 0 | 0 | 1076 | 0 | 0.0000 | 0.0160 | 미측정 | 8267984 |
| fresh32 | Lightweight | 0 | 0 | 0 | 269 | 0.0010 | 0.0000 | 미측정 | 0 |
| long_high | Baseline | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 미측정 | 0 |
| long_high | 기존 DLR | 352 | 352 | 352 | 0 | 0.1175 | 0.0000 | 미측정 | 2704768 |
| long_high | Device-Resident | 0 | 0 | 648 | 0 | 0.0000 | 0.0120 | 미측정 | 4979232 |
| long_high | Lightweight | 0 | 0 | 0 | 217 | 0.0010 | 0.0000 | 미측정 | 0 |
| long_low | Baseline | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 미측정 | 0 |
| long_low | 기존 DLR | 380 | 380 | 380 | 0 | 0.1211 | 0.0000 | 미측정 | 2919920 |
| long_low | Device-Resident | 0 | 0 | 388 | 0 | 0.0000 | 0.0060 | 미측정 | 2981392 |
| long_low | Lightweight | 0 | 0 | 0 | 97 | 0.0004 | 0.0000 | 미측정 | 0 |
| all72 | Baseline | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 미측정 | 0 |
| all72 | 기존 DLR | 2809 | 2809 | 2809 | 0 | 0.8664 | 0.0000 | 미측정 | 21584356 |
| all72 | Device-Resident | 0 | 0 | 3136 | 0 | 0.0000 | 0.0513 | 미측정 | 24097024 |
| all72 | Lightweight | 0 | 0 | 0 | 839 | 0.0034 | 0.0000 | 미측정 | 0 |

위 표의 기존 gate wall과 Device graph 구성 CPU는 범위가 달라 직접 차감하지 않는다. Device gate GPU 단독 시간은 추가 동기화 없이 분리할 수 없어 미측정이며, 계산·대기는 wall/model 시간에 포함된다. Lightweight ID 정책 점검 비용은 gate_wall/gate_cpu에 기록했다. id_comparisons는 decode 단계별 정책 점검 수이며 age 단락 조건이면 ID equality 자체는 실행하지 않는다. CPU 정수 비교를 GPU FLOPs로 표현하지 않았다. counter batch 읽기, 선택·anchor 갱신, 커널 launch도 실제 wall에 포함한다.

FLOPs는 dense MAC=2의 분석적 추정이며 GPU 실측값이 아니다. gate 산술 약5*d+4, 선택 element 수는 별도 저장했다. norm·activation·비교·메모리 이동을 포함한 완전한 GPU 명령 계수가 아니므로 제한적 산술 proxy를 전체 AI 연산량으로 부르지 않는다. 실제 MAC 루프 우회 counter와 추정 FLOPs를 구분한다. GPU 전력·금전 비용·GPU 실행시간은 미측정이다.

## 7. 품질 실패와 개별 변화

| 집합 | 방식 | Correct→Wrong/반복 | Wrong→Correct/반복 | 출력 불일치/전체요청 |
|---|---|---|---|---|
| replay32 | 기존 DLR | 0.00 | 0.00 | 0/96 |
| replay32 | Device-Resident | 0.00 | 0.00 | 0/96 |
| replay32 | Lightweight | 0.00 | 0.00 | 0/96 |
| fresh32 | 기존 DLR | 0.00 | 0.00 | 0/96 |
| fresh32 | Device-Resident | 0.00 | 0.00 | 0/96 |
| fresh32 | Lightweight | 0.00 | 0.00 | 0/96 |
| long_high | 기존 DLR | 1.00 | 0.00 | 6/12 |
| long_high | Device-Resident | 1.00 | 0.00 | 6/12 |
| long_high | Lightweight | 1.00 | 1.00 | 9/12 |
| long_low | 기존 DLR | 0.00 | 0.00 | 0/12 |
| long_low | Device-Resident | 0.00 | 0.00 | 0/12 |
| long_low | Lightweight | 0.00 | 0.00 | 0/12 |

### long_high/기존 DLR/1e25765168
- 질문: Literal transcription check 74603. Copy exactly the following 40 words in the same order, separated by single spaces. Add no explanation: / go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go
- 기준: 'go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go'
- Baseline: 'go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go'
- 비교 출력: 'go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go'; 전환 Correct->Wrong, 재사용74회.

### long_high/Device-Resident/1e25765168
- 질문: Literal transcription check 74603. Copy exactly the following 40 words in the same order, separated by single spaces. Add no explanation: / go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go
- 기준: 'go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go'
- Baseline: 'go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go'
- 비교 출력: 'go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go'; 전환 Correct->Wrong, 재사용74회.

### long_high/Lightweight/1e25765168
- 질문: Literal transcription check 74603. Copy exactly the following 40 words in the same order, separated by single spaces. Add no explanation: / go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go
- 기준: 'go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go'
- Baseline: 'go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go'
- 비교 출력: 'go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go go'; 전환 Correct->Wrong, 재사용188회.

### long_high/기존 DLR/a130f5ef24
- 질문: Literal transcription check 74600. Copy exactly the following 40 words in the same order, separated by single spaces. Add no explanation: / echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo
- 기준: 'echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo'
- Baseline: 'echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo'
- 비교 출력: 'echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo'; 전환 Wrong->Wrong, 재사용81회.

### long_high/Device-Resident/a130f5ef24
- 질문: Literal transcription check 74600. Copy exactly the following 40 words in the same order, separated by single spaces. Add no explanation: / echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo
- 기준: 'echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo'
- Baseline: 'echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo'
- 비교 출력: 'echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo'; 전환 Wrong->Wrong, 재사용81회.

### long_high/Lightweight/a130f5ef24
- 질문: Literal transcription check 74600. Copy exactly the following 40 words in the same order, separated by single spaces. Add no explanation: / echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo
- 기준: 'echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo'
- Baseline: 'echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo'
- 비교 출력: 'echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo echo'; 전환 Wrong->Wrong, 재사용80회.

### long_high/Lightweight/dda38ca147
- 질문: Literal transcription check 74601. Copy exactly the following 40 words in the same order, separated by single spaces. Add no explanation: / ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha
- 기준: 'ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha'
- Baseline: 'ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha'
- 비교 출력: 'ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha'; 전환 Wrong->Correct, 재사용76회.

전체 응답·채점·실패·순서는 CASES/FAILURES/GROUPS CSV에 보존했다. 위 예시만으로 결론을 선택하지 않는다. Baseline이 원래 전부 실패한 긴 생성 조건에서 같은 0%를 얻은 것을 품질 유지 성공으로 처리하지 않는다.

## 8. 최소 성공 기준과 결론

| 집합 | 방식 | 동일 관측 정답률 | 정답 손실 없음 | 5% 동일열 감소 | 3회 모두5%감소 | 순산술proxy감소 | 최소 관측 통과 |
|---|---|---|---|---|---|---|---|
| replay32 | 기존 DLR | True | True | False | False | True | False |
| replay32 | Device-Resident | True | True | False | False | True | False |
| replay32 | Lightweight | True | True | False | False | True | False |
| fresh32 | 기존 DLR | True | True | False | False | True | False |
| fresh32 | Device-Resident | True | True | False | False | True | False |
| fresh32 | Lightweight | True | True | False | False | True | False |
| long_high | 기존 DLR | False | False | False | False | True | False |
| long_high | Device-Resident | False | False | False | False | True | False |
| long_high | Lightweight | True | False | False | False | True | False |
| long_low | 기존 DLR | True | True | False | False | True | False |
| long_low | Device-Resident | True | True | False | False | True | False |
| long_low | Lightweight | True | True | False | False | False | False |

**최종 분류 C: 최적화 실패.** 관측 통과 조건만으로 무손실 일반화나 신규 기술 발견을 주장하지 않는다. CPU 시간도 반드시 따로 보고, wall이 줄어도 CPU가 늘면 CPU 절감이라고 표현하지 않는다. 메모리가 줄지 않으면 메모리 절감 성공이 아니다.

이번 반복 단어 집합에서 Baseline 정답률50%가 기존 DLR/Device에서는25%로 떨어졌다. Lightweight는50%로 같았지만 Correct→Wrong1개와 Wrong→Correct1개가 상쇄됐으며, go 복사는40단어 대신 상한96단어까지 생성했다. 집합 정답률만 보면 이 손상을 놓치므로 토큰열 일치율과 개별 전환을 별도 판정했다. 동일열 Lightweight의 평균 속도 이득은약3.23%로 목표5% 미만이다.

## 자동 테스트와 실제 실행 검증

기존 전체 snapshot 테스트251개:247개 통과, NumPy 환경 사유4개 건너뜀(73.387초). 격리 MLX/NumPy 환경의 관련17개 테스트는 전부 통과(0.210초)했으며 건너뛴4개 수치 검사를 포함한다. 추가 실제 모델 MLP tap은4개 forward에서 후보별 실제 MLP3회와 재사용1회를 확인했다. 작은 poisoned-weight GPU 제어 실험에서도 reuse 분기는 finite cached output/MAC counter0, fresh 분기는 NaN/counter3으로 확인했다. 이 소프트웨어 제어 실험을 실제 과제 성능으로 보고하지 않았다.

## 9. 질문별 답변

1. **기존 DLR이 느려진 원인:** 낮은 재사용률에 비해 후보별 host 결과 회수와 직렬화가 많았다. 추가 거리 산술은 작지만 latency 비용은 FLOPs만으로 설명되지 않는다. 이번 표의 pre-gate eval·host read·고정열 시간을 함께 보며, pre-gate 대기 전체를 추가 overhead로 오인하지 않는다.

2. **동기화 제거가 도움 됐는가:** Device의 gate item()/eval는 실제로0이며 요청 종료 counter1 batch와 greedy token sync는 남는다. 기존 DLR 대비 시간비는고정열 표에 있다. 새 GEMV kernel/수치 차이가 섞여 순수 sync 인과효과만은 식별하지 못한다.

3. **경량화하면 얼마나 빨라지는가:** Lightweight의 기존 DLR 대비 감소율과 Baseline 대비 감소율을 모두 위 표에서 구분했다. hidden-state distance 비용은0이나 동일토큰 문맥 변화 위험은 남는다. 재사용0인 요청은 연산 절감의 증거가 아니다.

4. **큰 모델에 유리할 가능성:** MLP 절감량은d*I, 거리gate는d 또는ID O(1)이므로 산술적 여지는 있다. 그러나 memory bandwidth, cache, prefill 비중, 모델별 hidden-state 변화, custom kernel 효율이 달라져 속도/품질 우위를 외삽하지 않는다. 다른 모델은 이번에 미실험이다.

5. **정확도와 시간을 함께 만족한 조건:** 위 최소 관측 통과 0개를 명시했다. 작은 표본과 반복 상관이 남아 모집단 정확도 유지가 입증된 것은 아니다.

6. **기존 activation reuse와 차이:** 이전 MLP residual 재사용 자체는 기존 계열이다. 이번 차이는 이 로컬 Qwen 구현에서 GPU 조건부 projection과 이미 관측된 token ID proxy를 비교한 점이다. 독립적 새로운 알고리즘/발명/일반적인 SOTA 근거는 없다.

## 10. 재현·보존

원래8526개 파일의 해시 보존 검사 통과. 모든 raw request는 append-only checkpoints, 원본 CSV는EXPORT 폴더다. 재시작은 완료 요청을 재생성하지 않고 source hash를 대조한다.

```sh
.alsr-venv/bin/python -B -m src.dlr_opt_experiment ./data/dlr_opt_2c8a76633b5e450f88242020fce6a899
.venv/bin/python -B -m src.dlr_opt_analysis ./data/dlr_opt_2c8a76633b5e450f88242020fce6a899
.venv/bin/python -B -m src.dlr_opt_report <새 ANALYSIS 폴더>
.venv/bin/python -B -m unittest tests.test_dlr_optimization
.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json
```

그래프는 기존 /private/tmp/vvp_plot_deps_20261105의 plotting 전용 의존성을 사용하며 모델 환경에 패키지를 설치하지 않는다. 모델 로컬 경로/revision/패키지/seed는manifest에 있다. 신규 문제는 한번 저장하면 다시 미사용 문제라고 주장하지 않는다.
