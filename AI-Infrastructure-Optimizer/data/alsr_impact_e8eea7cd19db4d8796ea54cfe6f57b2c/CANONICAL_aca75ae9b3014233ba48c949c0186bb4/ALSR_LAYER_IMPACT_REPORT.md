# ALSR Layer Impact Research Report



실행 `CANONICAL_aca75ae9b3014233ba48c949c0186bb4`. **실제 로컬 모델 실측**. 사전 설계: [ALSR_IMPACT_DESIGN.md](../../../ALSR_IMPACT_DESIGN.md). 기존 파일과 지난 ALSR 결과를 보존했다.



## 1. 핵심 판정

확대 조건을 충족한 후보: **없음**. 성공을 가정하거나 평가에서 임계값을 재조정하지 않았다. 계층 번호는 모두 **0부터 시작**한다.

**새 판단 신호 판정:** 동일8층/첫 위치에서 AUROC와 MAE 모두의 CI가 기존 Sketch보다 유리하다. 다만 학습 조건 차이가 있고 이것만으로 실제 연산 절감은 입증하지 못한다.

새 gate는 학습 데이터에서 고른 층 **24** 하나만 대상으로 한다. 보정에서 고정한 위험 임계값은 **0**다. 추정 위험은 KL+top1-flip 회귀값이며 확률로 보정된 오류 확률이 아니다.

Full 과제 성공률 59.38%, 새 gate 59.38%. 각 방법의 개별 정답 파괴/복구를 아래에 따로 보고한다. 같은 총점이 같은 응답을 의미하지 않는다.



## 동점 처리 오류와 재측정 공개

새 진단 코드의 argpartition 기반 top2 정렬이 BF16 동점에서 실제 greedy argmax와 다른 top1을 선택했다. full top1 기록 84행, flip 라벨 89/6272행이 달랐다. 과거 ALSR의 kl_effect는 이미 np.argmax를 사용하므로 이 오류를 과거 ALSR 전체의 오류로 확대하지 않는다.

원래 로그·프로파일·생성/시간 기록은 보존했다. top10에 모든 최대값 동점 토큰이 포함됨을 확인해 최소 token ID라는 greedy 규칙으로 라벨을 복원했다. signed margin 복원이 불가능했던 10건은 실제 재실행하여 원래 logits SHA 일치를 확인했다. KL/JS/top2 margin 원측정은 그대로다. 최초 실제 MLX 생성의 첫 token도 NumPy greedy 선택과 모두 같음을 확인했다.

수정 라벨의 학습32/보정16만으로 다시 fit했다. grid·KL 기준·후보 선택 규칙은 변경하지 않았다. 수정 후에도 후보24/threshold0이다. 단, 회귀 계수가 달라져7가지 방식 전체를 같은 순서 조건으로 다시3회 실측했다. **아래 주 결과는 수정 후3회만**이며 원래3회를 독립적인 추가3회로 합치거나 유리한 성능으로 골라 쓰지 않았다.

이 평가셋은 처음 연구 시에는 새로운 holdout이었지만 오류 감사 후 재사용됐으므로 수정본에 대해 완전히 새롭고 눈가림된 외부 검증이라고 주장하지 않는다. 평가 라벨/효과로 policy 임계값을 조정하지는 않았다.

진단 분포·고정 token 시퀀스·참조 NLL은 원래 실측을 재사용했다. 참조 NLL의 모든 마스크가 수정 후 생성의 마스크와 같음을 검사했다. 이를 새 NLL 반복 측정으로 표현하지 않는다. 준비/진단의 원래 비용과 재측정 비용은 구분한다.

| 방식 | 수정 전 성공률 | 수정 후 성공률 | 응답 변화(192관측) | 수정 전 wall/회 | 수정 후 wall/회 |
| --- | --- | --- | --- | --- | --- |
| Full | 59.38% | 59.38% | 0 | 24.197 | 25.038 |
| 기존 Sketch | 56.25% | 56.25% | 0 | 23.853 | 24.738 |
| 기존 Static4 | 18.75% | 18.75% | 0 | 29.719 | 30.134 |
| Random4 | 16.67% | 16.67% | 0 | 31.613 | 31.833 |
| Stable1 | 59.38% | 59.38% | 0 | 27.499 | 28.161 |
| Random1 | 43.75% | 43.75% | 0 | 25.216 | 26.061 |
| PreLayerGate | 59.38% | 59.38% | 0 | 24.179 | 24.966 |


위 두 phase의 시간 차이는 kernel cache·실행 시점·원래1회차의 NLL 진단 교차 실행 등과 섞여 있다. 알고리즘 절감 효과로 해석하지 않는다. 상세: ORIGINAL_PHASE_RESULTS.csv, PHASE_COMPARISON.csv, TIE_LABEL_CHANGES.csv.

## 2. 실행 환경과 공정성

Apple M4 / 16GiB Mac, 기존 공개 HF 모델 `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Apache-2.0. MLX 0.32.4/MLX-LM0.32.2, 실제 BF16, 28층(hidden1536, FFN8960, attention12/KV2). 기존 Ollama Q4와의 속도 비교가 아니다. Ollama 계층 생략 API를 가정하지 않았다.

모든 방식은 같은 chat template, greedy, MCQ 최대16/open최대48 생성토큰 및 EOS 규칙이다. 3회, 각 문제 안에서 방식 순서와 문제 순서를 무작위화하고 MANIFEST에 기록했다. 같은 문제를 3회 실행한 것을 독립된 192문제로 해석하지 않는다. 전용 장비/완전한 열·캐시 통제는 아니다.

기존 ALSR과 같은 last-position LM head 경로를 모든 방식에 공통 적용했다. native 전체-position logits와 수동 전체-position logits 정합 검사를 통과했다. BF16의 head 행렬 shape에 따른 작은 수치 차이는 SMOKE.json에 별도로 기록하며 공통 head 최적화의 효과를 새 생략 효과에 포함하지 않는다.

자연 생성 wall은 tokenization·실제 prefill/decode·gate·동기화·detokenize를 포함한다. 별도16문제는 동일한 prompt와 full이 만든 동일16-token 시퀀스를 모든 방식에 제공하여 출력 길이 차이를 제거했다. 미래 token은 benchmark 실행기만 가지며 gate는 현재 prompt hidden만 본다.

각 방식 중립 입력 워밍업 및 KV 신규 할당. 한 prompt의 첫 prefill에서 정한 skip mask를 끝까지 유지한다. 생략 캐시offset0, 나머지 offset 정합, 실제 블록/attention/MLP 미호출을 확인한다. 가중치를 제거하지 않아 대부분 메모리는 계속 상주한다.



## 3. 데이터 및 정보 누출 방지

새 학습32/보정16/평가64. 평가: 산술·논리·코딩·독해 각12, JSON8, 문장제약8. 학습/보정에도 열린 출력 유형을 포함했다. 과거 exact stem 중복0, split 교집합0. 합성 템플릿과 기본 능력은 공유하므로 의미적 중복·실서비스 다양성을 완전히 배제한 데이터는 아니다.

학습만으로 unembedding shortlist·각 층 회귀·후보 층을 결정하고, 보정의 분포 변화만으로 임계값을 선택했다. 정답, 정답 여부, 실제 생략 결과, 기준 미래 token은 live gate에 주지 않는다. 평가 ablation 결과가 생성 전에 존재하지만 PROFILE hash와 재구성 검사로 평가 결과를 policy에 사용하지 않았음을 확인했다.

기존 ALSR의 고정 parser를 유지하고 내용과 형식을 별도 측정한다. 오프라인 감사에서 Python의 true==1 비교로 JSON boolean을 숫자와 혼동하지 않도록 모든 방식에 동일한 정확한 타입 검사를 추가했다. 원시 GENERATION의 기존 grade는 수정하지 않았고 LEGACY_SCORE_COMPARISON.csv에 전후 점수를 보존한다. 문장 제약 통과는 전반적인 사실성·문체 품질 보장이 아니다. 참조 NLL은 각 방식/문제1회이며 비용은 진단 항목으로 분리했다. 64문제로 모집단 정확도 손실1%p 이내를 증명할 수 없다.

JSON 타입 감사로 판정이 달라진 반복 관측 수: **0**. 상세는 STRICT_SCORE_AUDIT.csv. 이 오프라인 채점 검사는 policy 학습·보정·생략 판단에 사용하지 않는다.



## 4. 단일 계층 인과 진단

모든28층을 하나씩 제거하여 두 context에서 측정했다: 첫 답변 직전(position0), full 모델의 첫 생성 token 뒤(position1). 첫 token이 EOS면 두 번째 context는 제외한다. 두 번째 context는 오프라인 고정-prefix 진단이다. 자연 생성은 각 방식 자신의 token 이력을 따른다.

전체 어휘 분포로 KL(P_full || P_skip), JS divergence, top-1 변화, 양쪽 top2 logit margin, full top1의 skip 분포 내 signed margin을 계산했다. CSV에는 top10 확률과 전체 float32 logits SHA를 기록한다. top10만으로 KL을 근사하지 않았다. feature 수집 baseline과 일반 baseline logits는 모든 context에서 정확히 같았다.

### position0 평균 KL이 낮은5개 층

| 층 | 평균 KL | p90 KL | 최대 KL | top1 일치율 |
| --- | --- | --- | --- | --- |
| 24 | 0.066 | 0.160 | 0.963 | 96.88% |
| 25 | 0.066 | 0.175 | 0.801 | 90.62% |
| 23 | 0.096 | 0.225 | 1.986 | 95.31% |
| 27 | 0.105 | 0.218 | 0.519 | 96.88% |
| 26 | 0.387 | 1.522 | 1.783 | 78.12% |


### position0 평균 KL이 높은5개 층

| 층 | 평균 KL | p90 KL | 최대 KL | top1 일치율 |
| --- | --- | --- | --- | --- |
| 0 | 11.793 | 14.345 | 18.694 | 0.00% |
| 1 | 3.088 | 5.934 | 7.178 | 23.44% |
| 20 | 2.416 | 5.134 | 8.505 | 40.62% |
| 21 | 2.094 | 4.317 | 14.105 | 56.25% |
| 14 | 1.454 | 4.692 | 8.286 | 60.94% |


위 순위는 평가 진단 결과이며 새 gate의 후보 선택에 사용하지 않았다. 낮은 평균 KL이 모든 입력·전체 생성에서 안전함을 의미하지 않는다. 전체28층/두 위치 지표는 [LAYER_SUMMARY.csv](LAYER_SUMMARY.csv), 원시 측정은 [LAYER_ABLATIONS.csv](LAYER_ABLATIONS.csv).



단일 층의 효과를 합쳐4층 동시 생략의 효과로 간주하지 않는다. 층 사이 보상·상호작용 때문에 Static4/Random4의 결과는 별도 실제 생성으로 검증했다.

### 학습에서 선택한 층의 두 위치 비교

| 후보 층 | 위치 | 평균 KL | top1 일치율 | full margin | skip margin |
| --- | --- | --- | --- | --- | --- |
| 24 | 0 | 0.066 | 96.88% | 4.836 | 4.344 |
| 24 | 1 | 0.413 | 85.94% | 4.688 | 5.105 |


position1은 동일 baseline token을 붙여 context 전체를 다시 prefill한 비교다. 실제 incremental KV decode와 수치적으로 완전히 동일하다는 보장은 하지 않는다. 유형별 층 효과는 LAYER_CATEGORY_SUMMARY.csv, margin과 KL의 관계는 LAYER_MARGIN_RELATION.csv에 기록한다. 최종 full logit margin은 오프라인 설명 변수이며 live 신호가 아니다.



## 5. 저비용 입력 신호의 예측력

현재 층 입력의8차원 고정 투영·RMS·길이, 직전 층 변화, 학습 전용 작은 LM-head shortlist margin/entropy를 사용한다. 현재 층 출력이나 full final logits를 live에서 계산하지 않는다. Shortlist는 전체 어휘에 대한 중간층 confidence가 아니며 잘 보정된 확률로 취급하지 않는다.

층마다 ridge로 KL+flip을 예측한다. layer-mean은 입력을 보지 않는 깊이 기준선이다. 기존 Sketch는 원래 ALSR profile을 그대로 고정하고 같은8층/position0에서 비교했다. 기존 Sketch와 새 신호의 학습 표본·특징 시점이 다르므로 예측력 차이를 신호 하나만의 인과 효과로 주장할 수 없다.

### 보정에서 고정한 생략 기준

| 임계값 | 허용된 문제 | top1 flip | 허용군 평균 KL | 조건 통과 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.00000 | True |
| 0.01 | 3 | 0 | 0.04940 | False |
| 0.025 | 4 | 0 | 0.04758 | False |
| 0.05 | 5 | 0 | 0.09281 | False |
| 0.1 | 9 | 0 | 0.05617 | False |
| 0.2 | 9 | 0 | 0.05617 | False |
| 0.5 | 16 | 0 | 0.07005 | False |
| 1 | 16 | 0 | 0.07005 | False |
| 2 | 16 | 0 | 0.07005 | False |


보정에서 후보 층의 모든 문제는 top1을 유지했지만 비어 있지 않은 모든 허용군이 mean KL<=.02 조건을 넘었다. 따라서 임계값0은 top1 분류 실패 때문이라고 단정할 수 없으며 엄격한 분포 보존 기준과 위험 회귀의 선택력에 함께 기인한다. 이 기준을 평가에서 완화하지 않았다.

임계값0이면 gate를 상수 Full로 접어 추가 probe를 제거할 수 있다. 이번 실측은 판단 오버헤드를 확인하기 위해 probe를 유지했다. 상수 접기를 해도 Baseline과 같아질 뿐 실제 계층 연산 절감은 생기지 않는다. 이 비용을 정책의 본질적인 필수 비용이라고 과장하지 않는다.

| 범위 | 위치 | 신호 | flip AUROC | 위험 MAE | 위험 Spearman |
| --- | --- | --- | --- | --- | --- |
| all28 | 0 | prelayer_ridge | 0.818 | 1.187 | 0.611 |
| all28 | 0 | layer_mean | 0.707 | 1.422 | 0.425 |
| old_sketch_pool8 | 0 | prelayer_ridge | 0.798 | 1.280 | 0.608 |
| old_sketch_pool8 | 0 | layer_mean | 0.684 | 1.634 | 0.324 |
| old_sketch_pool8 | 0 | frozen_old_sketch | 0.582 | 3.052 | 0.106 |
| candidate1 | 0 | prelayer_ridge | 0.726 | 0.133 | 0.295 |
| candidate1 | 0 | layer_mean | 0.500 | 0.096 | 미측정/정의 불가 |
| all28 | 1 | prelayer_ridge | 0.872 | 1.040 | 0.805 |
| all28 | 1 | layer_mean | 0.717 | 1.641 | 0.413 |
| old_sketch_pool8 | 1 | prelayer_ridge | 0.878 | 1.049 | 0.798 |
| old_sketch_pool8 | 1 | layer_mean | 0.701 | 1.737 | 0.322 |
| candidate1 | 1 | prelayer_ridge | 0.909 | 0.529 | 0.732 |
| candidate1 | 1 | layer_mean | 0.500 | 0.642 | 미측정/정의 불가 |


### 신호 비교의 질문 단위 bootstrap 95% CI

| 범위 | 대조 | AUROC 차이 [CI] | MAE 차이 [CI] |
| --- | --- | --- | --- |
| all28 | layer_mean | 0.112 [0.064, 0.156] | -0.235 [-0.367, -0.108] |
| candidate1 | layer_mean | 0.226 [-0.119, 0.500] | 0.038 [0.016, 0.059] |
| old_sketch_pool8 | layer_mean | 0.114 [0.049, 0.175] | -0.354 [-0.519, -0.185] |
| old_sketch_pool8 | frozen_old_sketch | 0.217 [0.123, 0.306] | -1.772 [-2.459, -1.147] |


AUROC 차이 양수/MAE 차이 음수가 새 신호에 유리하다. CI가0을 포함하면 우월성은 불확실하다. 전체 층 pooled AUROC는 깊이 식별만으로도 높아질 수 있으므로 [SIGNAL_METRICS.csv](SIGNAL_METRICS.csv)의 층별 평가와 candidate1을 함께 본다. 모든 회귀/원시 단일 신호는 탐색적 비교이며 다중 비교 보정된 기술 우월성 증거가 아니다.



공통8층/position0의 AUROC는 기존 0.582 → 새 신호 0.798다. 하지만 실제 선택 후보 층의 MAE는 layer-mean보다 +0.0377 크고 CI [+0.0160, +0.0586]다. 따라서 전체 pool에서의 예측 개선을 저위험 허용 구간의 정확한 보정으로 간주할 수 없다.

## 6. 실제 생성 결과와 계산 비용

표는64문제 한 회의 합계/비율을3회 평균한 값이다. Static4/Random4와 1층 방식의 예산은 다르다. Stable1/Random1은 1층 기준선이며 gate가 거절한 경우 gate의 평균 생략량은 더 작다. 이를 동일 연산 예산 비교로 포장하지 않는다.

기존 Sketch/Static4/Random4의 후보는8개 중간층, 새 Stable1/gate의 학습 탐색과 Random1은 내부25층(2..26)이다. 후보24는 기존8개 pool 밖에 있다. 후보 범위·생략 예산·학습 표본이 동시에 다르므로 정책 결과의 차이를 새 입력 신호만의 효과로 해석하지 않는다. 신호 자체는 공통8층 비교를 따로 제공한다.

| 방식 | 과제 성공률 | 객관식 정답률 | 형식 준수율 | 평균 생략층 | 생성 wall s | CPU s | 출력 토큰 | gate ms/요청 | MLX peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full | 59.38% | 45.83% | 32.81% | 0.00 | 25.038 | 3.767 | 546.0 | 0.000 | 2.985 |
| 기존 Sketch | 56.25% | 41.67% | 31.25% | 0.91 | 24.738 | 3.553 | 568.0 | 9.724 | 2.985 |
| 기존 Static4 | 18.75% | 4.17% | 26.56% | 4.00 | 30.134 | 4.417 | 827.0 | 0.000 | 2.984 |
| Random4 | 16.67% | 8.33% | 17.71% | 4.00 | 31.833 | 4.682 | 889.3 | 0.000 | 2.984 |
| Stable1 | 59.38% | 45.83% | 21.88% | 1.00 | 28.161 | 4.272 | 668.0 | 0.000 | 2.985 |
| Random1 | 43.75% | 27.08% | 34.90% | 1.00 | 26.061 | 3.860 | 604.7 | 0.000 | 2.985 |
| PreLayerGate | 59.38% | 45.83% | 32.81% | 0.00 | 24.966 | 3.534 | 546.0 | 0.998 | 2.984 |


CPU는 이 Python 프로세스의 실제 CPU 시간이며 GPU 전체 연산량이 아니다. MLX peak는 요청별 allocator peak의 평균이다. 프로세스 lifetime max RSS는 원시 로그에 따로 남겼고 방식별 peak로 해석하지 않는다. GPU 경과 시간·FLOPs 하드웨어 카운터·전력·금전 비용은 **미측정**이다.

방식들을 같은 프로세스에서 교차 실행하므로 모델과 작은 두 probe의 가중치는 공통 상주한다. 이 메모리 표는 공통 실행기 안에서 요청의 활성화/KV 포함 peak를 비교한 것이며 각 방식을 독립 프로세스로 배포한 최소 메모리 비교가 아니다.

새 gate timer는 앞선 hidden 계산을 materialize한 뒤 feature+prediction 시간을 잰다. 이때 생기는 동기화의 전체 latency 영향은 요청 wall에 포함된다. 기존 Sketch timer에는 앞서 queue된 GPU 계산 대기가 섞일 수 있어 둘의 gate timer만 직접 비교해 우월성을 주장하지 않는다.

### 동일16-token workload 및 분석 matmul FLOPs

| 방식 | fixed wall s | fixed CPU s | FLOPs/Full | 자연 생성 FLOPs/Full | 참조 NLL |
| --- | --- | --- | --- | --- | --- |
| Full | 10.324 | 1.502 | 1.0000 | 1.0000 | 2.595 |
| 기존 Sketch | 10.098 | 1.441 | 0.9722 | 0.9731 | 2.559 |
| 기존 Static4 | 8.994 | 1.333 | 0.8612 | 0.9084 | 2.053 |
| Random4 | 9.049 | 1.288 | 0.8612 | 0.9192 | 3.083 |
| Stable1 | 9.915 | 1.469 | 0.9653 | 0.9883 | 2.397 |
| Random1 | 9.942 | 1.467 | 0.9653 | 0.9761 | 2.522 |
| PreLayerGate | 10.240 | 1.527 | 1.0000 | 1.0000 | 2.595 |


FLOPs는 dense Q/K/V/O+MLP+attention matmul+LM head와 probe matmul의 MAC×2 분석 추정이다. norms/softmax/비선형/메모리 전송·kernel fusion 효과 등은 제외한다. 실제 미호출 검사와 결합해 구조적 연산 감소를 확인하지만 GPU FLOPs를 측정한 것으로 표현하지 않는다. 자연 생성 FLOPs에는 응답 길이 변화가 섞이므로 동일 token 부하를 별도 제시한다.

### Full 대비 paired 95% CI

| 방식 | 성공률 차이 pp [CI] | 자연 wall 비율 [CI] | fixed wall 비율 [CI] |
| --- | --- | --- | --- |
| 기존 Sketch | -3.12 [-7.81, 0.00] | 0.988 [0.942, 1.048] | 0.978 [0.941, 1.038] |
| 기존 Static4 | -40.62 [-53.12, -28.12] | 1.204 [1.072, 1.366] | 0.871 [0.840, 0.892] |
| Random4 | -42.71 [-54.17, -30.73] | 1.271 [1.131, 1.454] | 0.877 [0.860, 0.909] |
| Stable1 | 0.00 [-4.69, 4.69] | 1.125 [1.049, 1.232] | 0.960 [0.911, 0.991] |
| Random1 | -15.62 [-25.00, -7.29] | 1.041 [0.950, 1.148] | 0.963 [0.922, 0.992] |
| PreLayerGate | 0.00 [0.00, 0.00] | 0.997 [0.977, 1.014] | 0.992 [0.950, 1.014] |


CI는 질문 cluster와 반복 축을 재표본화한다. 3회 반복은 길이·입력 다양성에 대한 독립 표본을 늘리지 않는다.0차이 CI도1%p 모집단 비열등의 증명이 아니므로 보수적 exact 하한을 COMPARISONS.csv에 병기했다. 반복 평균/표준편차/최소/최대는 RESULTS.csv, 개별 반복은 REPEAT_METRICS.csv. 실행 순서 상관은 ORDER_DIAGNOSTICS.csv의 기술통계이며 캐시·열에 대한 인과 통제가 아니다.



### CPU 시간의 paired 비율과95% CI

| 방식 | 부하 | CPU/Full [CI] |
| --- | --- | --- |
| 기존 Sketch | natural | 0.943 [0.820, 1.067] |
| PreLayerGate | natural | 0.938 [0.811, 1.079] |
| Random4 | natural | 1.243 [1.033, 1.525] |
| Random1 | natural | 1.025 [0.867, 1.249] |
| Stable1 | natural | 1.134 [0.965, 1.343] |
| 기존 Static4 | natural | 1.173 [0.942, 1.459] |
| 기존 Sketch | fixed | 0.959 [0.855, 1.040] |
| PreLayerGate | fixed | 1.017 [0.915, 1.170] |
| Random4 | fixed | 0.857 [0.759, 0.949] |
| Random1 | fixed | 0.977 [0.945, 1.025] |
| Stable1 | fixed | 0.978 [0.915, 1.151] |
| 기존 Static4 | fixed | 0.887 [0.851, 0.940] |


새 gate의 자연 생성 CPU 시간 점추정은 줄었지만 CI가1을 포함한다. fixed 부하에서는 CPU 점추정도 줄지 않았다. 실제 계층 생략0이며 dense matmul 추정량도 감소하지 않아 이 CPU 변동을 전체 모델 추론 연산 절감으로 해석하지 않는다.

Static4는 Full보다 평균 참조 NLL이 낮으면서도 과제 성공률은 크게 나빠졌다. NLL·첫 token 안정성·형식·내용 정답률은 서로 다른 지표이며 어느 하나의 개선만으로 품질 유지라고 판단하지 않는다.

## 7. 총점에 가려진 개별 변화

| 방식 | 정답→오답 / 회 | 오답→정답 / 회 | 응답 변화 / 회 |
| --- | --- | --- | --- |
| Full | 0.00 | 0.00 | 0.00 |
| 기존 Sketch | 2.00 | 0.00 | 10.00 |
| 기존 Static4 | 27.00 | 1.00 | 61.00 |
| Random4 | 29.00 | 1.67 | 61.00 |
| Stable1 | 1.00 | 1.00 | 32.00 |
| Random1 | 12.33 | 2.33 | 45.00 |
| PreLayerGate | 0.00 | 0.00 | 0.00 |


Stable1은 Full과 총 성공률이 같지만 매회 정답 1개를 잃고 1개를 얻었다. 형식 준수율은 32.81% → 21.88%다. 출력 token은 546 → 668개/회로 증가했다. 같은 token 부하에서는 약 3.96% 빠르지만 자연 생성은 12.47% 느리다. 총점 보존을 출력 품질 보존으로 해석하면 안 된다.

| 문제 | 정답 | Full 응답 | Stable1 응답 | 전환 |
| --- | --- | --- | --- | --- |
| Log 30111: A package arrives on Tuesday. Nemi30111 stores it until Friday, when Pavo30111 collects it. On which day is it collected? A) Friday B) Wednesday C) Tuesday D) Saturday Return only the option letter. | A | B) Wednesday | A) | 오답→정답 |
| Parcel 30107 contains 5 packs of 53 cards and 29 loose cards. What is the total? A) 410 B) 236 C) 294 D) 82 Return only the option letter. | C | C) | A) 410 | 정답→오답 |


### 첫 top1이 같아도 전체 생성이 달라지는가?

| 방식 | 관측 수(반복 포함) | 고유 문제 | 응답 변화 | 정답 파괴 | 오답 복구 |
| --- | --- | --- | --- | --- | --- |
| Stable1 | 186 | 62 | 90 | 0 | 0 |
| Random1 | 126 | 57 | 69 | 5 | 0 |
| PreLayerGate | 0 | 0 | 0 | 0 | 0 |


위 표는 실제1층을 생략했고 최초 top1이 같았던 경우만 집계한다. 모든 생략 정책에 일반화한 비율이 아니다. 조건부 표본 및 반복 수를 고유 문제 수와 분리했다. [TRANSITIONS.csv](TRANSITIONS.csv)는 모든 문제/방식/반복의 원문과 전환을 보존한다.

Random1에서는 최초 top1이 같아도 정답을 잃은 관측이5건(고유4문제) 있었다. 첫 token이 구조 기호/답변 접두어일 수도 있어 최초 token 안정성은 의미 내용이나 전체 생성 안전성의 충분조건이 아니다. Stable1의 최초 top1 유지군에서는 정답 파괴가 없었지만 응답 변화가 발생했다. 이 제한된 조건부 결과를 모든 층에 일반화하지 않는다.

### 유형별 과제 성공률

| 유형 | Full | 기존 Sketch | 기존 Static4 | Random4 | Stable1 | Random1 | PreLayerGate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arithmetic | 25.00% | 25.00% | 0.00% | 2.78% | 16.67% | 11.11% | 25.00% |
| coding | 41.67% | 41.67% | 8.33% | 5.56% | 41.67% | 33.33% | 41.67% |
| logic | 25.00% | 16.67% | 0.00% | 11.11% | 25.00% | 8.33% | 25.00% |
| reading | 91.67% | 83.33% | 8.33% | 13.89% | 100.00% | 55.56% | 91.67% |
| structured_output | 100.00% | 100.00% | 37.50% | 25.00% | 100.00% | 87.50% | 100.00% |
| text_quality | 100.00% | 100.00% | 87.50% | 58.33% | 100.00% | 100.00% | 100.00% |


실패/응답 변화 사례의 prompt·reference·기준/수정 원문·생략층은 [FAILURE_CASES.csv](FAILURE_CASES.csv)에 전부 기록한다. 생성 길이 상한에 따른 형식 실패와 내용 오류를 구분한다.



## 8. 기존 기술과의 관계

* [ShortGPT](https://arxiv.org/abs/2403.03853)는 층 입출력 유사성으로 Block Influence를 정의하고 정적 pruning을 연구했다. 본 연구는 단일 층 제거의 최종 분포 인과 효과를 분석하고 현재 층 실행 전 특징을 진단한다. ShortGPT 전체 재현이나 우월성 증명이 아니다.

* [Tuned Lens](https://arxiv.org/abs/2303.08112)는 중간 hidden에서 최종 분포를 읽는 학습 affine probe를 연구했다. 여기서는 작은 shortlist와 ridge를 쓰며 전체 tuned-lens 학습을 재현하지 않았다. 중간 표현 probe라는 발상은 기존 기법이다.

* [LayerSkip](https://arxiv.org/abs/2404.16710)는 layer dropout/early-exit loss 학습과 self-speculative decoding을 결합한다. 여기서는 모델 가중치를 학습하지 않고 원래 Qwen에 branch-before-block을 적용했다.

* [SkipDecode](https://arxiv.org/abs/2307.02628)는 token별 exit와 batching/KV 제약을 다룬다. 여기서는 첫 prefill의 마스크를 고정하므로 동적 decode 재진입이나 batch 최적화를 주장하지 않는다.

문헌 검토는 대표 기법과의 범위를 구분한 것으로 전 세계 새로움을 확정하는 전수 조사가 아니다. 새로운 발명/독립 기술이라고 주장하지 않는다.



## 9. 판정과 한계

| 방식 | 품질 점추정 목표 | 10% latency CI 목표 | 실제 skip/추정 연산 감소 | 확대 후보 |
| --- | --- | --- | --- | --- |
| 기존 Sketch | False | False | True | False |
| 기존 Static4 | False | False | True | False |
| Random4 | False | False | True | False |
| Stable1 | False | False | True | False |
| Random1 | False | False | True | False |
| PreLayerGate | True | False | False | False |


가중치 학습 없이 raw single-layer skip은 downstream 계산 및 여러 token의 오차를 바꾼다. 단일 next-token KL/flip을 잘 예측하는 것과 전체 정답·형식·길이를 보존하는 것은 별개다. 단일 안정 층, 작은 보정셋, 합성 영어 단문, BF16/Qwen1.5B 한 모델,3회, Python 계측 및 host-GPU 동기화 비용이라는 한계가 있다.

10% 지연 감소는 앞선 ALSR에서 이어온 확대 목표다. 이번 1층 gate는 구조상 전체28층 중 최대1층만 생략하므로 큰 속도 향상을 기대하기 어렵다. 10% 미달 자체를 모든 작은 효과가 없다는 증거로 해석하지 않으며 실제 비율/CI를 그대로 제시한다. 신호의 예측력, 안정 층의 존재, 실용적인 전체 연산 절감은 서로 다른 판정이다.

새 신호가 예측 지표에서 일부 좋아도 실제 품질-지연 동시 조건을 충족하지 못하면 연산 절감 성공으로 판단하지 않는다. 실패를 숨기거나 test 기준으로 정책을 보정하지 않는다. 다음 단계는 평가 결과를 다시 학습시키는 것이 아니라 별도 학습 데이터에서 다중-token/완성답변 변화와 단일-token 위험의 연결을 검증하는 것이다.



## 10. 비용 범위·보존·재현

실측 모델 단계 wall 합계: 2439.692s. 원래 모델 단계 1640.693s + 수정 재측정 단계 798.999s의 합이며 중간 감사·테스트·보고서 작성 공백을 포함한 연속 경과시간이 아니다. 수정 fit 비용은 FIT_COST.json, margin 재검사 비용은 TIE_MARGIN_RECHECK.csv. 오프라인 원래 단계 비용: OFFLINE_COSTS.csv; 원래 shortlist/fixed-sequence 준비: PREPARATION.csv; 이번 load: LOAD.json; 이번 워밍업: WARMUPS.csv. 원래 fit 비용은 별도 미측정이며 원래 job wall에 포함된다. 오프라인 연구비와 live inference비를 혼합해 절감률을 만들지 않았다.

기존 729개 파일 해시 확인. 실제 호출/캐시/토큰/마스크 검사, probe baseline parity, profile train/validation 재구성, reference gate 호출0. 자동 테스트: **전체 unittest 230개: 226통과·4개 NumPy 환경 skip (74.330초); 격리 MLX 환경 신규12개 전부 통과 (0.075초); 실제 GPU gate 양성·음성 대조 통과**. 테스트 모의 응답을 실측표에 사용하지 않았다.

GATE_BRANCH_SMOKE.json은 중립 입력으로 실제 모델의 gate 생략 분기를 강제한 소프트웨어 검사다. 후보 블록에 trap을 설치해 생략 시 미호출과 비생략 시 예외라는 양성 대조를 확인하고 원래 프로파일을 보존한다. 강제 임계값의 답변·속도를 평가 성능이나 성공 사례로 사용하지 않는다.

```bash
# 동점 수정을 진단 수집 전부터 적용하고 모든 진단/생성을 실제 재계산
.alsr-venv/bin/python -B -u -m src.alsr_impact_fixed_run ./data/alsr_impact_e8eea7cd19db4d8796ea54cfe6f57b2c/MANIFEST.json
# 분석/보고서는 기존 산출물이 있으면 덮어쓰지 않음
.alsr-venv/bin/python -B -m src.alsr_impact_analysis <새 실행 폴더>
.venv/bin/python -B -m src.alsr_impact_cpu <새 실행 폴더>
.alsr-venv/bin/python -B -u -m src.alsr_impact_smoke <새 실행 폴더>
.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json
.alsr-venv/bin/python -B -m unittest discover -s tests -p "test_alsr_impact*.py"
.venv/bin/python -B -m src.alsr_impact_report <새 실행 폴더> "<실제 테스트 결과>"
```

고정 HF revision 캐시가 있어야 하며 MLX 실제 실행에는 Metal 권한이 필요하다. seed와 모든 실행 순서는 MANIFEST에 저장된다.



[결과](RESULTS.csv) · [층별 효과](LAYER_SUMMARY.csv) · [신호 비교 CI](SIGNAL_PAIRED.csv) · [전환](TRANSITIONS.csv) · [비교 그림](LAYER_IMPACT_COMPARISON.png) · [효율 그림](LAYER_IMPACT_EFFICIENCY.png) · [감사](AUDIT.json)
