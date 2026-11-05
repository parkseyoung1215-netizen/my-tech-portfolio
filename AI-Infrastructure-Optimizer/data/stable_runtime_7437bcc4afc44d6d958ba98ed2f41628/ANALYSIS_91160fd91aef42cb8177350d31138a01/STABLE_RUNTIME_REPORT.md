# Stable1 Runtime Attribution Report



실행 `stable_runtime_7437bcc4afc44d6d958ba98ed2f41628`. 실제 로컬 Qwen2.5 1.5B 모델 실측. **재현된 관측 효율 후보: [('lean_option_full', 'lean_full')]**. 이 후보 표기는 모집단 무손실·독립 신기술 성공 판정이 아니다.

**Stable1 계층 생략 개선은 성공하지 못했다.** 요청이 요구한 선택지 한 글자를 생성한 뒤 끝내는 Full 조건은 두 셋에서 내용 점수를 유지하며 시간·CPU·실제 블록 호출을 줄였지만, 이는 층 생략 성공이나 일반 서비스의 무손실·비용 절감률이 아니다.

replay64: Stable1 내용 점수 차이 +0.00%p, 자연 wall +13.76%, 동일16-token wall -3.83%. 실행기만 바꾼 Stable1 wall 감소 0.08%의 CI는 [-2.10, +1.59]%로0을 포함한다. Full의 선택지 완료 종료는 내용 점수 59.38%를 유지하고 wall 18.93%, CPU 22.56%, 추정 matmul FLOPs 2.65% 감소했다. 실제 블록 호출은 15288 → 11536회/회다.

fresh64: Stable1 내용 점수 차이 -3.12%p, 자연 wall +10.99%, 동일16-token wall -4.30%. 실행기만 바꾼 Stable1 wall 감소 1.14%의 CI는 [-0.77, +2.21]%로0을 포함한다. Full의 선택지 완료 종료는 내용 점수 85.94%를 유지하고 wall 19.94%, CPU 21.94%, 추정 matmul FLOPs 3.11% 감소했다. 실제 블록 호출은 17500 → 13160회/회다.



## 1. 질문과 공정한 비교

이전 Stable1은 총 성공률은 같았지만 정답1개를 잃고1개를 얻었고 형식 준수율도 떨어졌다. 그러므로 출발점부터 전반적 출력 품질이 유지됐다고 가정하지 않았다. 이번에는 출력 길이와 실행기 비용, 같은 token 부하의 비용을 분리한다.

모델 `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16, Apple M4 16GiB, MLX0.32.4/MLX-LM0.32.2, Apache-2.0 공개 가중치. 기존 Ollama Q4 또는 다른 모델과의 비교가 아니다. 모델/프롬프트/greedy/EOS/MCQ16·open48 상한은 공통이다. last-position LM head 경로도 모두 공통이며 native full-logits 정합성을 확인했다.

Full, Stable1(0-based24), 기존 frozen Adaptive-Sketch를 legacy/lean에서 비교한다. 모델 학습·계층 재선택·평가 임계값 조정은 없다. replay64와 별도 fresh64를 각3회 실행하고128문제/방식 순서를 무작위화했다. fresh64는 새 템플릿과 값이지만 합성 유형 간 의미적 상관까지 배제한 실서비스 표본은 아니다.

lean_option은 별도 출력 조건이다. 질문이 명시적으로 한 글자 선택지만 요구하고 현재 생성 prefix가 A-D 한 글자일 때만 종료한다. 모델이 만든 token을 그대로 반환하고 정답을 보지 않는다. JSON/문장은 원래 EOS/상한을 유지한다. 세 정책에 같은 규칙을 적용했다. Option Stable1 vs Option Full과 Lean Stable1 vs Lean Full을 비교하며, 출력 조건의 효과를 층 생략 효과로 귀속하지 않는다.



## 2. 측정 범위와 실행기 개선

legacy는 기존 Qwen native blocks·attention/MLP wrapper counters·per-token set/list/dict trace·logits materialize 뒤 argmax를 유지한다. lean은 같은 native attention/RMS/MLP/잔차 연산 순서를 사용하며, 실행 tuple을 미리 구성하고 decode 때 재사용한다. attention/MLP 디버그 wrapper를 우회하고 실제 진입 층 counter1개를 남긴다. argmax까지 함께 materialize하고 그 device token을 다음 입력으로 사용하여 host 토큰 tensor 재생성을 줄였다.

가중치 복사·양자화·hidden 보정·새 gating/학습·다른 모델은 사용하지 않았다. 원래 경로도 큰 logits 전체를 CPU로 복사하지 않으므로 이를 제거했다고 주장하지 않는다. 실제 줄인 것은 host-created token tensor와 Python 추적/dispatch 및 별도 materialization 경계다. Mac unified-memory에서 실제 물리 복사 byte·bandwidth를 측정한 것은 아니다.

legacy는 연구용 추적이 포함된 custom 실행기다. 그 비용을 제거해 빨라져도 공식 MLX-LM/llama.cpp/Ollama 같은 production runtime보다 우수함을 증명하지 않는다. 새 legacy에도 공통 구간 timer와 integrity 검사가 포함되어 지난 보고서의 absolute wall과 직접 비교해 개선율을 계산하지 않는다. 같은 새 실험 안에서 비교한다. 세 host 변경을 개별적으로 ablate하지 않아 각각의 기여 비율을 단정할 수 없다.

model_step은 forward+다음token 확정+동기화를 합친 실제 wall이다. gate는 이 구간 안의 중첩 timer이며 앞선 두 층의 queued GPU 작업 대기가 섞일 수 있다. model_step에서 gate를 뺀 값은 코드 영역별 시간일 뿐 순수 GPU 연산 시간이 아니다. Stable1은 정적 경로여서 per-request 동적 gate가 없으며 경로 초기화 비용은 load/setup에 포함된다.

host_bookkeeping은 기록한 구간을 제외한 루프 잔차이며 tensor 생성 등 MLX native 호출이 포함될 수 있다. process CPU time에는 Python 이외의 런타임/driver 작업도 포함된다. lean의 argmax wait는 아직 실행 중인 forward 대기까지 포함하고 legacy는 이미 logits를 materialize했으므로 이 둘을 argmax kernel 시간으로 직접 비교하면 안 된다.

별도16-token workload는 replay/fresh 각16문제에서6기본변형을 같은 baseline token prefix로 실행했다. 매 step에서 logits/argmax를 실제 계산하며 고정 token 때문에 계산 그래프를 생략하지 않았다. 별도의8문제×6변형×3회 heavy profile은 실제 native attention/MLP Tap와 첫 logits hash·dispatch timer를 포함하여 primary latency에서 제외한다.

GPU 경과시간·GPU FLOPs 하드웨어 counter·전력·금전 비용은 **미측정**. FLOPs는 실제 route/context/query 토큰 수에 기반한 dense matmul MAC×2 추정이며 norm/softmax/비선형 및 물리 memory traffic은 제외한다.



## 3. 실제 결과: 평균·표준편차·변동 범위

### replay64

| 방식 | 내용/과제 성공률 | 객관식 정확도 | 형식 준수율 | 평균 생략층 | wall s: 평균±SD [min,max] | CPU s: 평균±SD | 출력 token/회 | MLX peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 기존 Full | 59.38% | 45.83% | 32.81% | 0.00 | 25.387±1.069 [24.707,26.618] | 8.690±0.093 | 546.0 | 2.984 |
| 기존 Stable1 | 59.38% | 45.83% | 21.88% | 1.00 | 28.879±1.013 [28.147,30.036] | 9.845±0.235 | 668.0 | 2.984 |
| 기존 Sketch | 56.25% | 41.67% | 31.25% | 0.91 | 25.367±1.007 [24.755,26.529] | 8.700±0.262 | 568.0 | 2.984 |
| Lean Full | 59.38% | 45.83% | 32.81% | 0.00 | 24.941±0.914 [24.377,25.995] | 8.622±0.245 | 546.0 | 2.984 |
| Lean Stable1 | 59.38% | 45.83% | 21.88% | 1.00 | 28.856±1.284 [27.968,30.328] | 9.897±0.197 | 668.0 | 2.984 |
| Lean Sketch | 56.25% | 41.67% | 31.25% | 0.91 | 24.995±0.800 [24.486,25.917] | 8.411±0.290 | 568.0 | 2.984 |
| Option Full | 59.38% | 45.83% | 87.50% | 0.00 | 20.220±1.167 [19.496,21.566] | 6.677±0.204 | 412.0 | 2.984 |
| Option Stable1 | 59.38% | 45.83% | 87.50% | 1.00 | 19.628±0.768 [18.874,20.410] | 6.461±0.208 | 416.0 | 2.984 |
| Option Sketch | 56.25% | 41.67% | 87.50% | 0.91 | 19.444±0.872 [18.843,20.444] | 6.521±0.179 | 416.0 | 2.984 |


한 회64문제의 합계/비율을3회 반복한 통계다. 정확도와 형식을 구분하며 JSON은 타입까지 검사한다. 문장 품질은 이름·장소·한 문장 제약에 한정되고 사실성/문체의 완전한 평가는 아니다. MLX peak는 같은 프로세스·공통 상주 모델에서 요청별 allocator peak의 평균이며 process lifetime RSS를 방법별 peak로 사용하지 않았다.

| 방식 | 동일16-token wall s | 동일 부하 추정 FLOPs | 블록 실제 호출/회 | 층-token 평가/회 | 동적 gate s/64요청 | host input tensor/회 |
| --- | --- | --- | --- | --- | --- | --- |
| 기존 Full | 10.827 | 4.231 T | 15288 | 163772 | 0.000 | 546.0 |
| 기존 Stable1 | 10.413 | 4.084 T | 18036 | 161217 | 0.000 | 668.0 |
| 기존 Sketch | 10.690 | 4.114 T | 15184 | 159183 | 0.553 | 568.0 |
| Lean Full | 10.840 | 4.231 T | 15288 | 163772 | 0.000 | 64.0 |
| Lean Stable1 | 10.367 | 4.084 T | 18036 | 161217 | 0.000 | 64.0 |
| Lean Sketch | 10.336 | 4.114 T | 15184 | 159183 | 0.541 | 64.0 |
| Option Full | 미측정/정의 불가 | 해당 없음 | 11536 | 160020 | 0.000 | 64.0 |
| Option Stable1 | 미측정/정의 불가 | 해당 없음 | 11232 | 154413 | 0.000 | 64.0 |
| Option Sketch | 미측정/정의 불가 | 해당 없음 | 11046 | 155045 | 0.546 | 64.0 |


### fresh64

| 방식 | 내용/과제 성공률 | 객관식 정확도 | 형식 준수율 | 평균 생략층 | wall s: 평균±SD [min,max] | CPU s: 평균±SD | 출력 token/회 | MLX peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 기존 Full | 85.94% | 81.25% | 26.56% | 0.00 | 27.984±0.865 [27.340,28.968] | 9.676±0.343 | 625.0 | 2.983 |
| 기존 Stable1 | 82.81% | 77.08% | 18.75% | 1.00 | 31.060±0.904 [30.390,32.088] | 10.855±0.186 | 730.0 | 2.983 |
| 기존 Sketch | 67.19% | 72.92% | 23.44% | 0.94 | 25.796±1.126 [25.042,27.091] | 8.691±0.089 | 582.0 | 2.983 |
| Lean Full | 85.94% | 81.25% | 26.56% | 0.00 | 27.815±0.954 [27.221,28.916] | 9.505±0.184 | 625.0 | 2.983 |
| Lean Stable1 | 82.81% | 77.08% | 18.75% | 1.00 | 30.706±1.131 [29.880,31.995] | 10.485±0.205 | 730.0 | 2.983 |
| Lean Sketch | 67.19% | 72.92% | 23.44% | 0.94 | 25.331±0.748 [24.869,26.194] | 8.671±0.137 | 582.0 | 2.983 |
| Option Full | 85.94% | 81.25% | 82.81% | 0.00 | 22.268±0.671 [21.729,23.020] | 7.419±0.154 | 470.0 | 2.983 |
| Option Stable1 | 82.81% | 77.08% | 78.12% | 1.00 | 23.259±0.945 [22.629,24.345] | 7.859±0.195 | 522.0 | 2.983 |
| Option Sketch | 67.19% | 72.92% | 82.81% | 0.94 | 19.178±0.601 [18.739,19.862] | 6.237±0.125 | 404.0 | 2.983 |


한 회64문제의 합계/비율을3회 반복한 통계다. 정확도와 형식을 구분하며 JSON은 타입까지 검사한다. 문장 품질은 이름·장소·한 문장 제약에 한정되고 사실성/문체의 완전한 평가는 아니다. MLX peak는 같은 프로세스·공통 상주 모델에서 요청별 allocator peak의 평균이며 process lifetime RSS를 방법별 peak로 사용하지 않았다.

| 방식 | 동일16-token wall s | 동일 부하 추정 FLOPs | 블록 실제 호출/회 | 층-token 평가/회 | 동적 gate s/64요청 | host input tensor/회 |
| --- | --- | --- | --- | --- | --- | --- |
| 기존 Full | 10.855 | 4.127 T | 17500 | 161028 | 0.000 | 625.0 |
| 기존 Stable1 | 10.389 | 3.984 T | 19710 | 158112 | 0.000 | 730.0 |
| 기존 Sketch | 10.502 | 4.007 T | 15607 | 154521 | 0.546 | 582.0 |
| Lean Full | 10.647 | 4.127 T | 17500 | 161028 | 0.000 | 64.0 |
| Lean Stable1 | 10.264 | 3.984 T | 19710 | 158112 | 0.000 | 64.0 |
| Lean Sketch | 10.383 | 4.007 T | 15607 | 154521 | 0.529 | 64.0 |
| Option Full | 미측정/정의 불가 | 해당 없음 | 13160 | 156688 | 0.000 | 64.0 |
| Option Stable1 | 미측정/정의 불가 | 해당 없음 | 14094 | 152496 | 0.000 | 64.0 |
| Option Sketch | 미측정/정의 불가 | 해당 없음 | 10771 | 149685 | 0.537 | 64.0 |


## 4. paired 정확도·실제 시간·추정 계산량 차이

시간 감소율은1−target/control이다. CI는 질문 cluster와 반복 축의 paired resampling이다. 같은 질문3회를192개의 독립 질문으로 취급하지 않는다. CI가0개선/비율1을 포함하면 명확한 시간 개선으로 주장하지 않는다. 정확도 점추정이 하락하면 통계적 유의성과 무관하게 성공으로 판정하지 않는다.

| 셋 | 대상 / 대조 | 정답률 차이 pp [95% CI] | wall 감소율 % [95% CI] | CPU 감소율 % [CI] | 자연 추정 FLOPs 감소율 % | fixed wall 감소율 % [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| replay64 | 기존 Stable1 / 기존 Full | +0.00 [-4.69,+4.69] | -13.76 [-25.78,-5.57] | -13.29 [-27.43,-3.33] | +1.17 | +3.83 [+1.89,+7.05] |
| replay64 | 기존 Sketch / 기존 Full | -3.12 [-7.81,+0.00] | +0.08 [-6.21,+5.27] | -0.11 [-8.24,+6.98] | +2.69 | +1.26 [-2.95,+3.89] |
| replay64 | Lean Full / 기존 Full | +0.00 [+0.00,+0.00] | +1.76 [+0.53,+3.41] | +0.78 [-2.84,+4.29] | +0.00 | -0.12 [-5.34,+3.85] |
| replay64 | Lean Stable1 / 기존 Stable1 | +0.00 [+0.00,+0.00] | +0.08 [-2.10,+1.59] | -0.53 [-4.23,+2.60] | +0.00 | +0.44 [-2.11,+2.36] |
| replay64 | Lean Sketch / 기존 Sketch | +0.00 [+0.00,+0.00] | +1.46 [-0.06,+3.68] | +3.31 [+0.67,+6.16] | +0.00 | +3.32 [+1.08,+7.97] |
| replay64 | Lean Stable1 / Lean Full | +0.00 [-4.69,+4.69] | -15.70 [-27.80,-7.58] | -14.79 [-29.08,-5.11] | +1.17 | +4.36 [+1.46,+9.21] |
| replay64 | Lean Sketch / Lean Full | -3.12 [-7.81,+0.00] | -0.22 [-6.39,+5.07] | +2.44 [-5.76,+10.05] | +2.69 | +4.65 [+1.70,+10.41] |
| replay64 | Lean Stable1 / 기존 Full | +0.00 [-4.69,+4.69] | -13.67 [-25.90,-5.51] | -13.89 [-27.66,-4.49] | +1.17 | +4.25 [+2.57,+6.90] |
| replay64 | Option Full / Lean Full | +0.00 [+0.00,+0.00] | +18.93 [+11.67,+28.08] | +22.56 [+15.13,+32.95] | +2.65 | 조건 다름/해당 없음 |
| replay64 | Option Stable1 / Option Full | +0.00 [-4.69,+4.69] | +2.93 [-1.38,+6.62] | +3.24 [-0.73,+9.05] | +3.45 | 조건 다름/해당 없음 |
| replay64 | Option Sketch / Option Full | -3.12 [-7.81,+0.00] | +3.84 [+1.34,+7.38] | +2.33 [-2.62,+8.10] | +3.06 | 조건 다름/해당 없음 |
| replay64 | Option Stable1 / Lean Stable1 | +0.00 [+0.00,+0.00] | +31.98 [+22.63,+42.33] | +34.72 [+24.19,+46.56] | +4.90 | 조건 다름/해당 없음 |
| fresh64 | 기존 Stable1 / 기존 Full | -3.12 [-7.81,+0.00] | -10.99 [-19.87,-4.52] | -12.18 [-21.59,-4.55] | +1.46 | +4.30 [+2.34,+7.94] |
| fresh64 | 기존 Sketch / 기존 Full | -18.75 [-28.12,-9.38] | +7.82 [-0.13,+14.50] | +10.18 [+1.20,+17.97] | +4.09 | +3.25 [+1.31,+6.42] |
| fresh64 | Lean Full / 기존 Full | +0.00 [+0.00,+0.00] | +0.61 [-0.79,+1.64] | +1.77 [-1.93,+5.22] | +0.00 | +1.92 [-0.47,+4.98] |
| fresh64 | Lean Stable1 / 기존 Stable1 | +0.00 [+0.00,+0.00] | +1.14 [-0.77,+2.21] | +3.40 [+0.83,+5.98] | +0.00 | +1.21 [-0.16,+3.37] |
| fresh64 | Lean Sketch / 기존 Sketch | +0.00 [+0.00,+0.00] | +1.80 [+0.15,+4.29] | +0.23 [-3.51,+3.56] | +0.00 | +1.13 [-0.18,+2.96] |
| fresh64 | Lean Stable1 / Lean Full | -3.12 [-7.81,+0.00] | -10.39 [-19.21,-3.56] | -10.31 [-21.46,-2.15] | +1.46 | +3.60 [+1.54,+6.75] |
| fresh64 | Lean Sketch / Lean Full | -18.75 [-28.12,-9.38] | +8.93 [+1.16,+15.61] | +8.78 [-0.17,+16.33] | +4.09 | +2.47 [-0.05,+5.40] |
| fresh64 | Lean Stable1 / 기존 Full | -3.12 [-7.81,+0.00] | -9.73 [-18.45,-3.05] | -8.37 [-18.21,-0.57] | +1.46 | +5.45 [+3.35,+9.73] |
| fresh64 | Option Full / Lean Full | +0.00 [+0.00,+0.00] | +19.94 [+12.16,+29.33] | +21.94 [+13.64,+32.51] | +3.11 | 조건 다름/해당 없음 |
| fresh64 | Option Stable1 / Option Full | -3.12 [-7.81,+0.00] | -4.45 [-15.66,+2.86] | -5.93 [-19.66,+3.64] | +2.47 | 조건 다름/해당 없음 |
| fresh64 | Option Sketch / Option Full | -18.75 [-28.12,-9.38] | +13.88 [+6.25,+20.26] | +15.93 [+7.54,+22.59] | +4.61 | 조건 다름/해당 없음 |
| fresh64 | Option Stable1 / Lean Stable1 | +0.00 [+0.00,+0.00] | +24.25 [+16.67,+34.05] | +25.04 [+16.16,+36.34] | +4.11 | 조건 다름/해당 없음 |


실행기만 바꾼 동일 정책 비교에서 native token/mask가 같으면 Transformer matmul 자체의 수는 같고 host scheduling 비용이 달라진 것이다. 반대로 Stable1 vs Full은 실제 층 수가 달라지지만 자연 출력 길이도 달라진다. Option vs EOS의 절감은 주로 불필요한 후속 token 생성을 끝내는 효과이며 새 계층 생략 알고리즘으로 포장하지 않는다.



## 5. 병목 원인과 길이 영향

| 셋 | 실행 조건 | Stable1 추가 token/회 | 추가 wall/회 s | 관측 Δtoken-Δwall 상관 | 회귀 slope s/token |
| --- | --- | --- | --- | --- | --- |
| replay64 | legacy | 122.0 | 3.493 | 0.985 | 0.0374 |
| replay64 | lean | 122.0 | 3.915 | 0.989 | 0.0371 |
| replay64 | lean_option | 4.0 | -0.592 | 0.440 | 0.0614 |
| fresh64 | legacy | 105.0 | 3.076 | 0.993 | 0.0370 |
| fresh64 | lean | 105.0 | 2.891 | 0.980 | 0.0370 |
| fresh64 | lean_option | 52.0 | 0.991 | 0.957 | 0.0364 |


이는 동일 문제의 관측 결과 차이에 대한 사후 진단이며 출력 길이의 독립적인 인과 계수나 GPU token당 kernel 시간을 뜻하지 않는다. 동일16-token 통제가 빨라지고 자연 생성이 느리면 생성 길이가 구조적 절감을 상쇄했다는 근거다. 그 반대/불확실한 결과도 표에 그대로 남긴다.

| 셋 | 방식 | forward+token s | gate s(중첩) | tokenize s | decode s | 형식검사 s | host 잔차 s | reset/cache/check s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay64 | 기존 Full | 25.295 | 0.000 | 0.046 | 0.006 | 0.000 | 0.016 | 0.023 |
| replay64 | 기존 Stable1 | 28.782 | 0.000 | 0.045 | 0.006 | 0.000 | 0.023 | 0.024 |
| replay64 | 기존 Sketch | 25.275 | 0.553 | 0.045 | 0.006 | 0.000 | 0.017 | 0.023 |
| replay64 | Lean Full | 24.850 | 0.000 | 0.046 | 0.006 | 0.000 | 0.013 | 0.026 |
| replay64 | Lean Stable1 | 28.740 | 0.000 | 0.049 | 0.009 | 0.000 | 0.031 | 0.026 |
| replay64 | Lean Sketch | 24.907 | 0.541 | 0.045 | 0.006 | 0.000 | 0.013 | 0.024 |
| replay64 | Option Full | 20.121 | 0.000 | 0.045 | 0.005 | 0.004 | 0.020 | 0.026 |
| replay64 | Option Stable1 | 19.529 | 0.000 | 0.052 | 0.006 | 0.004 | 0.011 | 0.026 |
| replay64 | Option Sketch | 19.356 | 0.546 | 0.044 | 0.003 | 0.004 | 0.010 | 0.027 |
| fresh64 | 기존 Full | 27.889 | 0.000 | 0.048 | 0.006 | 0.000 | 0.018 | 0.024 |
| fresh64 | 기존 Stable1 | 30.962 | 0.000 | 0.047 | 0.006 | 0.000 | 0.020 | 0.025 |
| fresh64 | 기존 Sketch | 25.688 | 0.546 | 0.059 | 0.006 | 0.000 | 0.016 | 0.026 |
| fresh64 | Lean Full | 27.723 | 0.000 | 0.044 | 0.006 | 0.000 | 0.016 | 0.025 |
| fresh64 | Lean Stable1 | 30.574 | 0.000 | 0.045 | 0.007 | 0.000 | 0.055 | 0.025 |
| fresh64 | Lean Sketch | 25.205 | 0.529 | 0.045 | 0.006 | 0.000 | 0.047 | 0.027 |
| fresh64 | Option Full | 22.154 | 0.000 | 0.049 | 0.004 | 0.009 | 0.013 | 0.040 |
| fresh64 | Option Stable1 | 23.146 | 0.000 | 0.049 | 0.004 | 0.016 | 0.014 | 0.030 |
| fresh64 | Option Sketch | 19.069 | 0.537 | 0.046 | 0.004 | 0.010 | 0.024 | 0.026 |


게이트 열은 forward 열에 이미 포함되므로 합산하지 않는다. 그림의 분해에서는 forward−gate와 gate를 합쳐 중복을 제거했다. 메모리 복사비/순수 Python 실행시간의 완전한 분리는 하지 못했다. 세 정책 공통의 scalar token read와 종료 확인이 남아 있고 비전용 노트북 열·kernel cache 영향을 완전히 통제하지 못했다.



### 이미 계산된 hidden에서 분리한 Sketch 판단 진단

| 진단 | warm 관측 수 | wall ms 평균±SD | CPU ms 평균 |
| --- | --- | --- | --- |
| ISOLATED_GATE_c4cc734c03ab4e56b230f11551f1dd69.json | 60 | 0.335±0.123 | 0.122 |


학습용4입력의 첫2층 hidden을 완전히 materialize한 후 같은 hidden에서3회×5번씩 판단했다. 이는 캐시가 따뜻한 microbenchmark이며60개의 독립 문제나 production latency가 아니다. 본 실험의 Sketch gate 포함 timer는 약8ms/요청이지만 앞선 계층 대기가 섞여 있으므로 이를 순수 판단 비용이라고 해석할 수 없다. 두 시간을 빼서 정확한 GPU 비용을 추정하지 않는다.

## 6. 개별 정답과 출력 품질 변화

| 셋 | 방식 | 대조 | 문제 | 전환 | 기준 응답 | 응답 |
| --- | --- | --- | --- | --- | --- | --- |
| replay64 | 기존 Stable1 | 기존 Full | Queue 30110: Nemi30110 stands immediately before Pavo30110, and Pavo30110 immediately before Selu30110. The queue contains only these three people. Who is in the middle? A) Selu30110 B) Pavo30110 C) Nemi30110 D) Cannot be determined Return only the option letter. | Wrong -> Wrong | C | C) Nemi30110 |
| replay64 | Lean Stable1 | Lean Full | Queue 30110: Nemi30110 stands immediately before Pavo30110, and Pavo30110 immediately before Selu30110. The queue contains only these three people. Who is in the middle? A) Selu30110 B) Pavo30110 C) Nemi30110 D) Cannot be determined Return only the option letter. | Wrong -> Wrong | C | C) Nemi30110 |
| replay64 | 기존 Stable1 | 기존 Full | Queue 30104: Nemi30104 stands immediately before Pavo30104, and Pavo30104 immediately before Selu30104. The queue contains only these three people. Who is in the middle? A) Nemi30104 B) Selu30104 C) Pavo30104 D) Cannot be determined Return only the option letter. | Wrong -> Wrong | B | B) |
| replay64 | Lean Stable1 | Lean Full | Queue 30104: Nemi30104 stands immediately before Pavo30104, and Pavo30104 immediately before Selu30104. The queue contains only these three people. Who is in the middle? A) Nemi30104 B) Selu30104 C) Pavo30104 D) Cannot be determined Return only the option letter. | Wrong -> Wrong | B | B) |
| replay64 | 기존 Stable1 | 기존 Full | Rules for set 30103: All daxes are merins. No merin is blue. Object Nemi30103 is a dax. Which conclusion follows? A) Nemi30103 is not blue B) Nothing follows C) Nemi30103 is not a merin D) Nemi30103 is blue Return only the option letter. | Correct -> Correct | A | A) Nemi30103 is not blue |
| replay64 | 기존 Sketch | 기존 Full | Rules for set 30103: All daxes are merins. No merin is blue. Object Nemi30103 is a dax. Which conclusion follows? A) Nemi30103 is not blue B) Nothing follows C) Nemi30103 is not a merin D) Nemi30103 is blue Return only the option letter. | Correct -> Correct | A | A) Nemi30103 is not blue |
| replay64 | Lean Stable1 | Lean Full | Rules for set 30103: All daxes are merins. No merin is blue. Object Nemi30103 is a dax. Which conclusion follows? A) Nemi30103 is not blue B) Nothing follows C) Nemi30103 is not a merin D) Nemi30103 is blue Return only the option letter. | Correct -> Correct | A | A) Nemi30103 is not blue |
| replay64 | Lean Sketch | Lean Full | Rules for set 30103: All daxes are merins. No merin is blue. Object Nemi30103 is a dax. Which conclusion follows? A) Nemi30103 is not blue B) Nothing follows C) Nemi30103 is not a merin D) Nemi30103 is blue Return only the option letter. | Correct -> Correct | A | A) Nemi30103 is not blue |
| replay64 | 기존 Stable1 | 기존 Full | Queue 30102: Nemi30102 stands immediately before Pavo30102, and Pavo30102 immediately before Selu30102. The queue contains only these three people. Who is in the middle? A) Pavo30102 B) Nemi30102 C) Selu30102 D) Cannot be determined Return only the option letter. | Wrong -> Wrong | B | B) Nemi30102 |
| replay64 | Lean Stable1 | Lean Full | Queue 30102: Nemi30102 stands immediately before Pavo30102, and Pavo30102 immediately before Selu30102. The queue contains only these three people. Who is in the middle? A) Pavo30102 B) Nemi30102 C) Selu30102 D) Cannot be determined Return only the option letter. | Wrong -> Wrong | B | B) Nemi30102 |
| replay64 | 기존 Stable1 | 기존 Full | Parcel 30107 contains 5 packs of 53 cards and 29 loose cards. What is the total? A) 410 B) 236 C) 294 D) 82 Return only the option letter. | Correct -> Wrong | C) | A) 410 |
| replay64 | Lean Stable1 | Lean Full | Parcel 30107 contains 5 packs of 53 cards and 29 loose cards. What is the total? A) 410 B) 236 C) 294 D) 82 Return only the option letter. | Correct -> Wrong | C) | A) 410 |
| replay64 | Option Stable1 | Option Full | Parcel 30107 contains 5 packs of 53 cards and 29 loose cards. What is the total? A) 410 B) 236 C) 294 D) 82 Return only the option letter. | Correct -> Wrong | C | A |
| replay64 | 기존 Sketch | 기존 Full | Rules for set 30111: All daxes are merins. No merin is blue. Object Nemi30111 is a dax. Which conclusion follows? A) Nemi30111 is blue B) Nothing follows C) Nemi30111 is not a merin D) Nemi30111 is not blue Return only the option letter. | Correct -> Wrong | D) Nemi30111 is not blue | C) Nemi30111 is not a merin |
| replay64 | Lean Sketch | Lean Full | Rules for set 30111: All daxes are merins. No merin is blue. Object Nemi30111 is a dax. Which conclusion follows? A) Nemi30111 is blue B) Nothing follows C) Nemi30111 is not a merin D) Nemi30111 is not blue Return only the option letter. | Correct -> Wrong | D) Nemi30111 is not blue | C) Nemi30111 is not a merin |
| replay64 | Option Sketch | Option Full | Rules for set 30111: All daxes are merins. No merin is blue. Object Nemi30111 is a dax. Which conclusion follows? A) Nemi30111 is blue B) Nothing follows C) Nemi30111 is not a merin D) Nemi30111 is not blue Return only the option letter. | Correct -> Wrong | D | C |
| replay64 | 기존 Sketch | 기존 Full | Memo 30100: Nemi30100 owns a silver compass. Nemi30100 lends the compass to Pavo30100 for one day; ownership does not change. Who owns the compass during that day? A) Nemi30100 B) Selu30100 C) Nobody D) Pavo30100 Return only the option letter. | Correct -> Wrong | A) Nemi30100 | D |
| replay64 | Lean Sketch | Lean Full | Memo 30100: Nemi30100 owns a silver compass. Nemi30100 lends the compass to Pavo30100 for one day; ownership does not change. Who owns the compass during that day? A) Nemi30100 B) Selu30100 C) Nobody D) Pavo30100 Return only the option letter. | Correct -> Wrong | A) Nemi30100 | D |
| replay64 | Option Sketch | Option Full | Memo 30100: Nemi30100 owns a silver compass. Nemi30100 lends the compass to Pavo30100 for one day; ownership does not change. Who owns the compass during that day? A) Nemi30100 B) Selu30100 C) Nobody D) Pavo30100 Return only the option letter. | Correct -> Wrong | A | D |
| replay64 | 기존 Stable1 | 기존 Full | Rules for set 30109: All daxes are merins. No merin is blue. Object Nemi30109 is a dax. Which conclusion follows? A) Nemi30109 is not a merin B) Nothing follows C) Nemi30109 is not blue D) Nemi30109 is blue Return only the option letter. | Wrong -> Wrong | A | A) Nemi30109 is not a merin |
| replay64 | 기존 Sketch | 기존 Full | Rules for set 30109: All daxes are merins. No merin is blue. Object Nemi30109 is a dax. Which conclusion follows? A) Nemi30109 is not a merin B) Nothing follows C) Nemi30109 is not blue D) Nemi30109 is blue Return only the option letter. | Wrong -> Wrong | A | A) Nemi30109 is not a merin |
| replay64 | Lean Stable1 | Lean Full | Rules for set 30109: All daxes are merins. No merin is blue. Object Nemi30109 is a dax. Which conclusion follows? A) Nemi30109 is not a merin B) Nothing follows C) Nemi30109 is not blue D) Nemi30109 is blue Return only the option letter. | Wrong -> Wrong | A | A) Nemi30109 is not a merin |
| replay64 | Lean Sketch | Lean Full | Rules for set 30109: All daxes are merins. No merin is blue. Object Nemi30109 is a dax. Which conclusion follows? A) Nemi30109 is not a merin B) Nothing follows C) Nemi30109 is not blue D) Nemi30109 is blue Return only the option letter. | Wrong -> Wrong | A | A) Nemi30109 is not a merin |
| replay64 | 기존 Stable1 | 기존 Full | Rules for set 30105: All daxes are merins. No merin is blue. Object Nemi30105 is a dax. Which conclusion follows? A) Nemi30105 is not a merin B) Nemi30105 is not blue C) Nemi30105 is blue D) Nothing follows Return only the option letter. | Wrong -> Wrong | A | A) Nemi30105 is not a merin |


전체 실패는 FAILURES.csv, 모든 문제/반복의 정답 유지·파괴·복구와 원문은 TRANSITIONS.csv에 보존한다. 같은 총점이 개별 답변 보존을 의미하지 않는다. 선택지 종료로 부적절한 후속 설명이 사라져 parser 점수가 변할 수도 있다. 실제 생성 prefix 여부를 검사하고 출력 조건이 다른 결과는 분리했다.

| 셋 | 종료 조건 비교 정책 | EOS 정답→option 오답/회 | EOS 오답→option 정답/회 | 단어 prefix 가능 관측 |
| --- | --- | --- | --- | --- |
| replay64 | full | 0.0 | 0.0 | 0 |
| replay64 | stable1 | 0.0 | 0.0 | 0 |
| replay64 | sketch | 0.0 | 0.0 | 0 |
| fresh64 | full | 0.0 | 0.0 | 0 |
| fresh64 | stable1 | 0.0 | 0.0 | 0 |
| fresh64 | sketch | 0.0 | 0.0 | 0 |


이번 Full 종료 조건에서는 두 셋 모두 개별 내용 점수 손실/복구가0이었다. supplementary initial_choice_label_accuracy도 RESULTS.csv에 남겼다. 이 형식 완료 규칙은 일반 자유서술·코딩 생성·JSON에 적용하지 않았으며 더 넓은 출력 형태에 안전하다고 주장하지 않는다. STOP_TRANSITIONS.csv에 종료 조건 변경 전후 전 사례를 보존한다.

| 셋 | 변형 | 유형 | 성공률 | 형식 준수율 | 출력 token/회 |
| --- | --- | --- | --- | --- | --- |
| replay64 | 기존 Full | arithmetic | 25.00% | 0.00% | 44.0 |
| replay64 | 기존 Full | coding | 41.67% | 0.00% | 36.0 |
| replay64 | 기존 Full | logic | 25.00% | 91.67% | 35.0 |
| replay64 | 기존 Full | reading | 91.67% | 16.67% | 67.0 |
| replay64 | 기존 Full | structured_output | 100.00% | 0.00% | 200.0 |
| replay64 | 기존 Full | text_quality | 100.00% | 100.00% | 164.0 |
| replay64 | 기존 Stable1 | arithmetic | 16.67% | 0.00% | 69.0 |
| replay64 | 기존 Stable1 | coding | 41.67% | 8.33% | 56.0 |
| replay64 | 기존 Stable1 | logic | 25.00% | 25.00% | 113.0 |
| replay64 | 기존 Stable1 | reading | 100.00% | 16.67% | 62.0 |
| replay64 | 기존 Stable1 | structured_output | 100.00% | 0.00% | 200.0 |
| replay64 | 기존 Stable1 | text_quality | 100.00% | 100.00% | 168.0 |
| replay64 | 기존 Sketch | arithmetic | 25.00% | 0.00% | 45.0 |
| replay64 | 기존 Sketch | coding | 41.67% | 0.00% | 36.0 |
| replay64 | 기존 Sketch | logic | 16.67% | 75.00% | 61.0 |
| replay64 | 기존 Sketch | reading | 83.33% | 25.00% | 58.0 |
| replay64 | 기존 Sketch | structured_output | 100.00% | 0.00% | 200.0 |
| replay64 | 기존 Sketch | text_quality | 100.00% | 100.00% | 168.0 |
| replay64 | Lean Full | arithmetic | 25.00% | 0.00% | 44.0 |
| replay64 | Lean Full | coding | 41.67% | 0.00% | 36.0 |
| replay64 | Lean Full | logic | 25.00% | 91.67% | 35.0 |
| replay64 | Lean Full | reading | 91.67% | 16.67% | 67.0 |
| replay64 | Lean Full | structured_output | 100.00% | 0.00% | 200.0 |
| replay64 | Lean Full | text_quality | 100.00% | 100.00% | 164.0 |
| replay64 | Lean Stable1 | arithmetic | 16.67% | 0.00% | 69.0 |
| replay64 | Lean Stable1 | coding | 41.67% | 8.33% | 56.0 |
| replay64 | Lean Stable1 | logic | 25.00% | 25.00% | 113.0 |
| replay64 | Lean Stable1 | reading | 100.00% | 16.67% | 62.0 |
| replay64 | Lean Stable1 | structured_output | 100.00% | 0.00% | 200.0 |
| replay64 | Lean Stable1 | text_quality | 100.00% | 100.00% | 168.0 |
| replay64 | Lean Sketch | arithmetic | 25.00% | 0.00% | 45.0 |
| replay64 | Lean Sketch | coding | 41.67% | 0.00% | 36.0 |
| replay64 | Lean Sketch | logic | 16.67% | 75.00% | 61.0 |
| replay64 | Lean Sketch | reading | 83.33% | 25.00% | 58.0 |
| replay64 | Lean Sketch | structured_output | 100.00% | 0.00% | 200.0 |
| replay64 | Lean Sketch | text_quality | 100.00% | 100.00% | 168.0 |
| replay64 | Option Full | arithmetic | 25.00% | 100.00% | 12.0 |
| replay64 | Option Full | coding | 41.67% | 100.00% | 12.0 |
| replay64 | Option Full | logic | 25.00% | 100.00% | 12.0 |
| replay64 | Option Full | reading | 91.67% | 100.00% | 12.0 |
| replay64 | Option Full | structured_output | 100.00% | 0.00% | 200.0 |
| replay64 | Option Full | text_quality | 100.00% | 100.00% | 164.0 |
| replay64 | Option Stable1 | arithmetic | 16.67% | 100.00% | 12.0 |
| replay64 | Option Stable1 | coding | 41.67% | 100.00% | 12.0 |
| replay64 | Option Stable1 | logic | 25.00% | 100.00% | 12.0 |
| replay64 | Option Stable1 | reading | 100.00% | 100.00% | 12.0 |
| replay64 | Option Stable1 | structured_output | 100.00% | 0.00% | 200.0 |
| replay64 | Option Stable1 | text_quality | 100.00% | 100.00% | 168.0 |
| replay64 | Option Sketch | arithmetic | 25.00% | 100.00% | 12.0 |
| replay64 | Option Sketch | coding | 41.67% | 100.00% | 12.0 |
| replay64 | Option Sketch | logic | 16.67% | 100.00% | 12.0 |
| replay64 | Option Sketch | reading | 83.33% | 100.00% | 12.0 |
| replay64 | Option Sketch | structured_output | 100.00% | 0.00% | 200.0 |
| replay64 | Option Sketch | text_quality | 100.00% | 100.00% | 168.0 |
| fresh64 | 기존 Full | arithmetic | 41.67% | 0.00% | 87.0 |
| fresh64 | 기존 Full | coding | 83.33% | 0.00% | 39.0 |
| fresh64 | 기존 Full | logic | 100.00% | 25.00% | 90.0 |
| fresh64 | 기존 Full | reading | 100.00% | 50.00% | 32.0 |
| fresh64 | 기존 Full | structured_output | 100.00% | 0.00% | 216.0 |
| fresh64 | 기존 Full | text_quality | 100.00% | 100.00% | 161.0 |
| fresh64 | 기존 Stable1 | arithmetic | 33.33% | 0.00% | 137.0 |
| fresh64 | 기존 Stable1 | coding | 75.00% | 0.00% | 61.0 |
| fresh64 | 기존 Stable1 | logic | 100.00% | 8.33% | 94.0 |
| fresh64 | 기존 Stable1 | reading | 100.00% | 25.00% | 54.0 |
| fresh64 | 기존 Stable1 | structured_output | 100.00% | 0.00% | 216.0 |
| fresh64 | 기존 Stable1 | text_quality | 100.00% | 100.00% | 168.0 |
| fresh64 | 기존 Sketch | arithmetic | 33.33% | 0.00% | 93.0 |
| fresh64 | 기존 Sketch | coding | 83.33% | 0.00% | 42.0 |
| fresh64 | 기존 Sketch | logic | 75.00% | 33.33% | 80.0 |
| fresh64 | 기존 Sketch | reading | 100.00% | 25.00% | 56.0 |
| fresh64 | 기존 Sketch | structured_output | 0.00% | 0.00% | 149.0 |
| fresh64 | 기존 Sketch | text_quality | 100.00% | 100.00% | 162.0 |
| fresh64 | Lean Full | arithmetic | 41.67% | 0.00% | 87.0 |
| fresh64 | Lean Full | coding | 83.33% | 0.00% | 39.0 |
| fresh64 | Lean Full | logic | 100.00% | 25.00% | 90.0 |
| fresh64 | Lean Full | reading | 100.00% | 50.00% | 32.0 |
| fresh64 | Lean Full | structured_output | 100.00% | 0.00% | 216.0 |
| fresh64 | Lean Full | text_quality | 100.00% | 100.00% | 161.0 |
| fresh64 | Lean Stable1 | arithmetic | 33.33% | 0.00% | 137.0 |
| fresh64 | Lean Stable1 | coding | 75.00% | 0.00% | 61.0 |
| fresh64 | Lean Stable1 | logic | 100.00% | 8.33% | 94.0 |
| fresh64 | Lean Stable1 | reading | 100.00% | 25.00% | 54.0 |
| fresh64 | Lean Stable1 | structured_output | 100.00% | 0.00% | 216.0 |
| fresh64 | Lean Stable1 | text_quality | 100.00% | 100.00% | 168.0 |
| fresh64 | Lean Sketch | arithmetic | 33.33% | 0.00% | 93.0 |
| fresh64 | Lean Sketch | coding | 83.33% | 0.00% | 42.0 |
| fresh64 | Lean Sketch | logic | 75.00% | 33.33% | 80.0 |
| fresh64 | Lean Sketch | reading | 100.00% | 25.00% | 56.0 |
| fresh64 | Lean Sketch | structured_output | 0.00% | 0.00% | 149.0 |
| fresh64 | Lean Sketch | text_quality | 100.00% | 100.00% | 162.0 |
| fresh64 | Option Full | arithmetic | 41.67% | 75.00% | 57.0 |
| fresh64 | Option Full | coding | 83.33% | 100.00% | 12.0 |
| fresh64 | Option Full | logic | 100.00% | 100.00% | 12.0 |
| fresh64 | Option Full | reading | 100.00% | 100.00% | 12.0 |
| fresh64 | Option Full | structured_output | 100.00% | 0.00% | 216.0 |
| fresh64 | Option Full | text_quality | 100.00% | 100.00% | 161.0 |
| fresh64 | Option Stable1 | arithmetic | 33.33% | 50.00% | 102.0 |
| fresh64 | Option Stable1 | coding | 75.00% | 100.00% | 12.0 |
| fresh64 | Option Stable1 | logic | 100.00% | 100.00% | 12.0 |
| fresh64 | Option Stable1 | reading | 100.00% | 100.00% | 12.0 |
| fresh64 | Option Stable1 | structured_output | 100.00% | 0.00% | 216.0 |
| fresh64 | Option Stable1 | text_quality | 100.00% | 100.00% | 168.0 |
| fresh64 | Option Sketch | arithmetic | 33.33% | 75.00% | 57.0 |
| fresh64 | Option Sketch | coding | 83.33% | 100.00% | 12.0 |
| fresh64 | Option Sketch | logic | 75.00% | 100.00% | 12.0 |
| fresh64 | Option Sketch | reading | 100.00% | 100.00% | 12.0 |
| fresh64 | Option Sketch | structured_output | 0.00% | 0.00% | 149.0 |
| fresh64 | Option Sketch | text_quality | 100.00% | 100.00% | 162.0 |


## 7. 보존·누출·실행 증거

| 검사 | 결과 |
| --- | --- |
| existing_files_verified | 829 |
| fresh_exact_stem_past_overlap | 0 |
| replay_and_fresh_disjoint | True |
| model_policy_sources_frozen | True |
| no_training_or_threshold_tuning | True |
| actual_layer_counts_and_cache_verified | True |
| actual_attention_mlp_audit_verified | True |
| native_logits_smoke_exact | True |
| legacy_lean_token_disagreements | 0 |
| legacy_lean_mask_disagreements | 0 |
| option_prefix_disagreements | 0 |
| replay_vs_previous_experiment_token_disagreements | 0 |
| primary_requests | 3456 |
| fixed_requests | 576 |
| heavy_profile_requests | 144 |
| checkpoint_complete_records | 4209 |
| GPU_elapsed_power_money | unmeasured |
| FLOPs | analytical dense matmul estimate; physical copy bandwidth unmeasured |
| profile_first_logit_hash_disagreements | 0 |
| option_possible_word_prefix_observations | 0 |


새로운 학습이나 threshold 탐색은 하지 않았다. 생성기 함수는 question과 고정 정책/출력 조건만 받으며 reference/정답 여부를 받지 않는다. fixed prefix는 benchmark 실행기만 가지며 Sketch 판단에는 전달하지 않는다. 새로운 문제의 exact stem은 과거 기록과 겹치지 않지만 의미적/템플릿 독립성을 과장하지 않는다.

legacy/lean의 natural token·mask 일치 여부를 PARITY.csv에, 옵션 종료 prefix 일치 여부를 STOP_PREFIX_PARITY.csv에 모두 남겼다. 실제 native attention/MLP 호출은 별도 Tap profile 및 소프트웨어 양성/음성 대조로 확인했다. 상세 계측의 비용을 primary 비교에 섞지 않았다.

자동 테스트: **전체 unittest 234개 중230통과·4개 NumPy 환경 skip (75.828초); 격리 MLX 환경 신규4개+관련12개 모두 통과 (16개,0.124초); 실제 native logits/토큰/attention·MLP/KV 대조 통과; 완료 재개 시 추가 생성0 검증**. 테스트 모의 응답을 실측표에 사용하지 않았다.



## 8. 판정·기술적 의미와 다음 가설

| 셋 | 대상/대조 | 내용·형식 점추정 유지 | 시간 개선 CI | 추정 자연 연산 감소 | 관측 후보 | 모집단 무손실 확정 |
| --- | --- | --- | --- | --- | --- | --- |
| replay64 | Lean Stable1/기존 Stable1 | True | False | False | False | False |
| replay64 | Lean Stable1/Lean Full | False | False | True | False | False |
| replay64 | Option Full/Lean Full | True | True | True | True | False |
| replay64 | Option Stable1/Option Full | True | False | True | False | False |
| fresh64 | Lean Stable1/기존 Stable1 | True | False | False | False | False |
| fresh64 | Lean Stable1/Lean Full | False | False | True | False | False |
| fresh64 | Option Full/Lean Full | True | True | True | True | False |
| fresh64 | Option Stable1/Option Full | False | False | True | False | False |


새로운 기술: 실행계획 재사용, host 추적 감소, lazy materialization, device token 재사용, 요청 형식에 맞춘 종료는 알려진 런타임/생성 제어 아이디어의 로컬 조합이다. 이 실험으로 독립적인 신기술·일반적인 무손실 layer skipping을 주장할 수 없다. [MLX lazy evaluation 문서](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [공식 MLX-LM generate](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), [custom StoppingCriteria 문서](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.StoppingCriteria), [ShortGPT](https://arxiv.org/abs/2403.03853)와 범위를 구분한다. 공식 문서는 개념의 근거이며 이번 수치의 근거는 로컬 로그다.

다음 가장 유망한 검증 가설은 **모든 층을 유지한 공식 MLX-LM 실행기에서도 명확한 출력 계약의 완료 종료가 정확도를 유지하며 시간을 줄이는가**다. Full의 선택지 종료만 두 셋에 재현됐으므로 새로운 계층 생략 정책보다 production baseline·독립 holdout·다른 출력 길이에서 이 효과를 먼저 확인할 가치가 있다. JSON/자유서술로 확대하는 안전성은 아직 검증하지 않았다. 현재64+64 합성 문제에서 관측한 무손실은 모집단 무손실 증명이 아니며, 일반 AI 서비스 전체에19~20% 절감률을 적용할 수 없다. 추가 개발이나 새 정책 튜닝은 이번 결과에 맞춰 수행하지 않았다.



## 9. 재개·재현·비용 범위

완료 세션 수 2, 완료 세션 wall 합계 1860.884s. 이는 import/manifest 생성 이전은 제외하며 여러 세션일 때 중단 공백을 포함한 연속 wall이 아니다. 요청 latency 합계에는 모델 응답 처리 비용을, 세션 wall에는 load/warmup·smoke·prep·heavy profile·오프라인 채점·체크포인트 쓰기·export를 포함한다. 단위별 benchmark 파일 쓰기 시간은 별도로 계측하지 않았고 세션 잔차를 pure Python으로 해석하지 않는다. 중단된 세션은 end 기록이 없어 이 합계에 포함되지 않을 수 있다.

요청 완료 JSON은 새 attempt 파일로 fsync하며 재개 시 완료 행을 재사용한다. 불완전 JSON도 삭제하지 않는다. 같은 key의 완료 행이 중복되면 중단한다. 프로파일/소스/패키지 hash가 바뀌면 재개를 거부한다. 재개로 달라진 세션/캐시 상태는 session ID로 추적한다.

완료 후 실제 재개 감사: {'checkpoint_file_hashes_unchanged': True, 'complete_records_before': 4209, 'complete_records_after': 4209, 'new_session': 'db13370563444951a84488a561e6bc00', 'additional_generated_checkpoint_requests': 0, 'new_model_load_record': False, 'resume_export_completed': True, 'counter_note': 'Session end new_complete_records is a cumulative total in this frozen runner; actual additional records are derived from checkpoint session IDs.'}. frozen runner의 session end 필드 new_complete_records는 완료 누계이며, 새 실행 수는 checkpoint의 session ID로 따로 계산한다. 재개 감사는 추가 생성0과 새 model-load 기록 없음을 확인했다.

```bash
# 새 실행
.alsr-venv/bin/python -B -u -m src.stable_runtime_experiment
# 중단 후 이 실행 재개
.alsr-venv/bin/python -B -u -m src.stable_runtime_experiment ./data/stable_runtime_7437bcc4afc44d6d958ba98ed2f41628
# GPU 호출 없는 상태 확인
.venv/bin/python -B -m src.stable_runtime_status ./data/stable_runtime_7437bcc4afc44d6d958ba98ed2f41628
# 완료 후 새 ANALYSIS 폴더에 분석
.venv/bin/python -B -m src.stable_runtime_analysis ./data/stable_runtime_7437bcc4afc44d6d958ba98ed2f41628
.venv/bin/python -B -m src.stable_runtime_report <생성된 ANALYSIS 폴더> "<테스트 결과>"
```

[결과](RESULTS.csv) · [반복](REPEATS.csv) · [비교 CI](COMPARISONS.csv) · [길이 원인 분석](LENGTH_ATTRIBUTION.csv) · [실패](FAILURES.csv) · [호출 상세 profile](PROFILE.csv) · [그래프](STABLE_RUNTIME_COMPARISON.png) · [분해 그래프](STABLE_RUNTIME_BREAKDOWN.png)
