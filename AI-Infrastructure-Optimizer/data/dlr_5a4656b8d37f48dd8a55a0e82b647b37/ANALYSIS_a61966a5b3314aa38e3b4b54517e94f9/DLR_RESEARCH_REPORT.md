# Dynamic Layer Reuse (DLR) 연구 보고서

실행 `dlr_5a4656b8d37f48dd8a55a0e82b647b37`. 무료 로컬 모델 실측. 기존 파일·실험 보존. GitHub 업로드 없음.

**결론: 이번 DLR은 효율 개선에 실패했다.** 실제 MLP 우회는 확인했지만 매우 드물었고, 판단 오버헤드를 포함한 실행시간이 늘었다. Conservative는 재사용0회로 model matmul을 줄이지 못했다. DLR의 정답 손실이 이 표본에서 관측되지 않았다는 사실과 일반적인 정확도 유지 입증은 다르다.

replay32: 자연 wall +5.93%, CPU +10.13%, 모델 matmul 추정 감소 0.00858%; fresh32: 자연 wall +7.77%, CPU +16.57%, 모델 matmul 추정 감소 0.01972%

동일 토큰열에서도 replay32 wall +5.52%, 추정 matmul 감소 0.01407%; fresh32 wall +8.20%, 추정 matmul 감소 0.03769%. 출력 길이를 통제해도 시간이 증가했다. 메모리 감소도 입증하지 못했다.

## 1. 구현과 선행 기법

Qwen2.5-1.5B-Instruct BF16(28층), 기존 MLX-LM 환경과 고정된 로컬 가중치를 사용했다. Ollama의 공개 생성 API에는 계층별 MLP 실행을 바꾸는 훅이 없어 MLX에서 직접 실행했다. 원본 가중치·설치 패키지는 변경하지 않았다. Apache-2.0 모델이며 모델 파일 SHA256를 로드 시 대조했다. Apple M4 16GiB, MLX 메모리 제한은 권장 working set의80%, allocator cache 상한128MiB다.

Baseline은 전체 연산이다. Static은 이전 Stable1의24번층 전체를 생략한다(0-based). DLR은8·12·16·24번층에서 현재 토큰의 Attention과 KV 갱신을 정상 실행한 뒤, post-attention RMSNorm 입력의 이전 fresh-anchor 대비 상대 L2 거리가0.5이하이면 MLP 출력만 재사용한다. Conservative는0.2이하다. 연속 재사용은1회로 제한하고 다음 토큰에서는 새로 계산한다. prefill은 재사용하지 않는다. 각 요청마다 캐시를 비우며 서로 다른 질문의 활성값을 공유하지 않는다. 추가 LLM 판별 호출은 없다.

이는 **시간축 MLP activation reuse의 소규모 재현 실험**이다. KV Cache는 과거 토큰의 정확한 K/V 보관인 반면 이 방식은 새 토큰의 MLP 결과를 과거 근사값으로 대체한다. Layer Skipping은 residual branch 전체를0으로 처리하지만 DLR은 이전 nonzero MLP residual을 사용하고 Attention은 갱신한다. Token Pruning은 토큰 위치 자체를 제거하며 이번 실험은 위치를 제거하지 않는다. 토큰을 적게 생성하는 ACS와도 다르다. Activation reuse와 본질적으로 겹치므로 독립적인 새 발명이라고 주장하지 않는다.

[LLMCache](https://arxiv.org/abs/2512.16843)는 중간 활성값 캐싱을 다룬다. [dLLM-Cache](https://arxiv.org/abs/2506.06295)는 diffusion 모델의 반복 denoising 활성값 재사용으로, 서로 다른 새 토큰을 처리하는 이 실험과 조건이 다르다. [SkipDecode](https://arxiv.org/abs/2307.02628)는 token-level early exit 비교 맥락이다. 문헌 확인은 독창성의 완전한 조사나 성능 재현을 뜻하지 않는다. [모델 카드](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Ollama API](https://docs.ollama.com/api/generate).

## 2. 사전 설계와 측정 범위

가중치 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. 패키지: `{"mlx": "0.32.4", "mlx-lm": "0.32.2", "numpy": "2.5.4", "transformers": "5.22.1"}`.

초기 소프트웨어 검증에서는 native 참조와 수동 경로의 normalization/head 평가 경계가 달라 exact-logit 검사가 실패했다. 이를 성능 데이터로 채택하지 않았다. 실패한 파일럿 PROTOCOL 폴더도 보존했다. 수동 경로와 참조 모두 전체 hidden 정규화 후 마지막 위치 head 조건을 맞춘 뒤 max-abs logit 차이0을 확인하고 개발 파일럿·본 실험을 진행했다. 평가를 시작한 뒤 런타임이나 임계값을 바꾸지는 않았다.

무작위 시드 2026110604. 기존32문제(객관식24, JSON8), 신규32문제(객관식16, JSON8, 짧은 색상 사실답8). 방식별3회, 자연 생성768회. 각 집합8문제에서는 Baseline 1회차의 실제 토큰열을 모든 방식에 동일하게 입력하는 fixed-stream192회를 추가했다. 합계960개 실제 비교 요청이다. 모든 방식의 프롬프트·greedy·EOS·32토큰 상한이 동일하며 early stop 정책은 추가하지 않았다. 코드 해시와 정책을 manifest에 동결했다. 개발용8문제와 소프트웨어 positive control은 평가에서 제외했다. 개발 파일럿에서 재사용0회였지만 임계값을 평가 데이터에 맞춰 완화하지 않았다.

신규 문제는 기존 JSON/CSV의 question stem을 재귀적으로 수집해 정확 중복을 검사했다. 단, 합성 템플릿·개념의 유사성은 남으며 외부 일반화를 증명하지 않는다. 학습·임계값 튜닝을 하지 않았다. runtime은 질문·방식·토큰예산만 받고 정답·채점·미래 검증 결과는 받지 않는다. 평가용 expected_object/reference는 별도 오프라인 채점에서만 사용된다.

내용 정확도와 형식 준수율을 분리한다. 객관식은 관찰 가능한 선택지 표식을 파싱하고, JSON은 중복키·bool/정수 혼동을 거부하는 typed equality, 색상 답은 사전 고정한 제한적 문법 변형을 인정한다. 이전 doctor/physician 문제를 피하기 위해 자유 문장 재작성은 이번 평가에서 제외했다. 즉 본 정확도는 이 폐쇄형 과제 범위에 대한 값이며 일반 문장 의미·환각 품질을 대신하지 않는다. 32토큰 상한으로 잘린 결과도 그대로 채점한다.

wall은 매 요청 시작/종료 MLX 동기화 및 판단 대기를 포함한다. CPU는 process_time으로 측정한 호스트 프로세스 CPU이며 GPU 시간이나 전체 칩 연산량이 아니다. 판단 이전 입력 준비 동기화 대기는 pre_gate_sync로 따로 기록하되 모델/wall에서 빠지지 않는다. gate 시간은 입력 준비 뒤 float32 거리·GPU reduction·host branch 비용이다. 둘은 중복 합산하지 않는다. 저장/모델 로드/워밍업은 요청 wall 밖이며 별도 로그다. 실제 call 카운터·KV offset 검증과 MLP 트랩으로 재사용 분기가 실행을 우회함을 확인했다.

FLOPs는 dense matmul MAC=2 추정치다. Q/K/V/O, Attention QK/AV, MLP3projection, 마지막 위치 vocabulary head를 포함한다. norm·activation·메모리 이동·Python·거리 계산의 elementwise 연산은 포함하지 않는다. gate 처리 element 수는 별도이며 gate 포함 실제 시간은 보고한다. GPU FLOPs·전력·금전 비용은 **미측정**이다. MLX allocator peak는 실제 관측이나 장치 전체 메모리와 다르다. RSS는 프로세스 lifetime high water여서 방식별 독립 peak가 아니다. 캐시 payload는 논리적 활성값 크기이며 slice backing storage 및 임시 메모리는 allocator peak에 포함될 수 있다.

## 3. 자연 생성 결과

### replay32

| 방식 | 내용 정확도 | 형식 준수 | wall 합계 평균±SD(s) | CPU 합계 평균±SD(s) | 입력/출력 토큰 | 실제 MLP 재사용/회 | gate(s) | MLX peak(MiB) |
|---|---|---|---|---|---|---|---|---|
| baseline | 59.38% | 18.75% | 12.570 ± 0.194 | 3.723 ± 0.255 | 2615/288 | 0 | 0.000 | 3083.74 |
| static | 59.38% | 9.38% | 14.009 ± 0.153 | 4.258 ± 0.249 | 2615/342 | 0 | 0.000 | 3083.36 |
| dlr | 59.38% | 18.75% | 13.314 ± 0.140 | 4.100 ± 0.140 | 2615/288 | 8 | 0.400 | 3086.86 |
| conservative | 59.38% | 18.75% | 13.233 ± 0.197 | 3.923 ± 0.269 | 2615/288 | 0 | 0.386 | 3086.12 |

### fresh32

| 방식 | 내용 정확도 | 형식 준수 | wall 합계 평균±SD(s) | CPU 합계 평균±SD(s) | 입력/출력 토큰 | 실제 MLP 재사용/회 | gate(s) | MLX peak(MiB) |
|---|---|---|---|---|---|---|---|---|
| baseline | 84.38% | 50.00% | 12.315 ± 0.233 | 3.528 ± 0.601 | 2228/296 | 0 | 0.000 | 3073.30 |
| static | 90.62% | 46.88% | 12.657 ± 0.323 | 3.742 ± 0.572 | 2228/314 | 0 | 0.000 | 3072.67 |
| dlr | 84.38% | 50.00% | 13.273 ± 0.407 | 4.113 ± 0.749 | 2228/296 | 16 | 0.414 | 3075.12 |
| conservative | 84.38% | 50.00% | 13.133 ± 0.561 | 3.810 ± 0.951 | 2228/296 | 0 | 0.392 | 3074.99 |

## 4. 동일 토큰열과 비용 분해

fixed-stream은 각 문제의 동일한 실제 Baseline 토큰열을 재생해 생성 길이를 통제한다. 실제 forward를 모두 실행하며 응답을 캐시에서 재생해 성능으로 간주한 것이 아니다. 다만 teacher forcing의 채점 정확도는 독립적인 생성 정확도로 해석하지 않는다. 모든 방식의 입력·출력 토큰 수 일치 검사를 통과했다.

### natural

| 집합 | 방식 | 정확도차(pp) | wall 감소율 [95%CI] | CPU 감소율 | matmul 추정 감소율 | 출력 토큰 감소율 | 손실 문제/고유 |
|---|---|---|---|---|---|---|---|
| replay32 | static | +0.00 | -11.45% [-26.29, -1.79] | -14.36% | 1.398% | -18.75% | 0 |
| replay32 | dlr | +0.00 | -5.93% [-8.13, -3.50] | -10.13% | 0.009% | 0.00% | 0 |
| replay32 | conservative | +0.00 | -5.28% [-7.24, -2.68] | -5.36% | 0.000% | 0.00% | 0 |
| fresh32 | static | +6.25 | -2.78% [-11.08, 2.27] | -6.06% | 2.690% | -6.08% | 0 |
| fresh32 | dlr | +0.00 | -7.77% [-12.05, -4.68] | -16.57% | 0.020% | 0.00% | 0 |
| fresh32 | conservative | +0.00 | -6.64% [-9.97, -3.65] | -7.99% | 0.000% | 0.00% | 0 |

### fixed

| 집합 | 방식 | 정확도차(pp) | wall 감소율 [95%CI] | CPU 감소율 | matmul 추정 감소율 | 출력 토큰 감소율 | 손실 문제/고유 |
|---|---|---|---|---|---|---|---|
| replay32 | static | +0.00 | 2.37% [-0.24, 4.59] | 5.73% | 3.465% | 0.00% | 0 |
| replay32 | dlr | +0.00 | -5.52% [-9.35, -2.23] | 2.34% | 0.014% | 0.00% | 0 |
| replay32 | conservative | +0.00 | -4.59% [-7.57, -1.16] | 4.34% | 0.000% | 0.00% | 0 |
| fresh32 | static | +0.00 | 2.19% [-2.98, 6.40] | -0.23% | 3.449% | 0.00% | 0 |
| fresh32 | dlr | +0.00 | -8.20% [-12.08, -3.88] | -24.90% | 0.038% | 0.00% | 0 |
| fresh32 | conservative | +0.00 | -6.87% [-10.39, -3.65] | -19.02% | 0.000% | 0.00% | 0 |

양의 감소율은 감소, 음수는 증가다. paired CI는 고유 질문과 반복 축을 함께 재표집한 bootstrap이다. 3회 반복 greedy 출력을 독립적인 새 답변3개로 과장하지 않는다. 정확도차의 보수적 구간은 DLR_COMPARISONS.csv에 있으며, 차이0이나 bootstrap CI[0,0]만으로 모집단 비열등성을 확정하지 않는다. CP 기반1%p 비열등성 여부도 따로 기록했다. fixed8문제/집합은 특히 작아 확정적인 지연 일반화를 주장할 수 없다.

## 5. 재사용·오버헤드·순서 효과

- replay32/dlr: 실제 재사용 24/3048회(판단 중 0.79%). 재사용이 있었던 요청 중 내용 정답 24/24. 이는 반사실적 무손실 확률이나 각 연산의 안전 확률이 아니다. 거리 범위 0.4592–1.3010.
- replay32/conservative: 실제 재사용 0/3072회(판단 중 0.00%). 재사용이 있었던 요청 중 내용 정답 0/0. 이는 반사실적 무손실 확률이나 각 연산의 안전 확률이 아니다. 거리 범위 0.4592–1.3010.
- fresh32/dlr: 실제 재사용 48/3120회(판단 중 1.54%). 재사용이 있었던 요청 중 내용 정답 24/24. 이는 반사실적 무손실 확률이나 각 연산의 안전 확률이 아니다. 거리 범위 0.4308–1.3258.
- fresh32/conservative: 실제 재사용 0/3168회(판단 중 0.00%). 재사용이 있었던 요청 중 내용 정답 0/0. 이는 반사실적 무손실 확률이나 각 연산의 안전 확률이 아니다. 거리 범위 0.4020–1.3258.

각 후보 계층에서 host가 GPU reduction 결과를 읽어야 실제 MLP를 건너뛸 수 있다. 이 동기화는 GPU 파이프라인을 끊을 수 있으며, 재사용0이어도 판단 비용은 든다. 후보4층만 대상으로 하고 prefill을 재사용하지 않아 최대 절감 여지도 작다. 모델 가중치는 그대로 상주하고 KV cache도 줄지 않아, activation reuse가 곧 메모리 절감이라는 보장은 없다.

순서별 평균은 DLR_ORDER_EFFECTS.csv에 기록했다. 각 문제 안에서 방식 순서를 섞었지만, 위치마다 문제 구성이 다르므로 이 단순 평균으로 순서 인과효과를 식별할 수 없다. OS 부하·온도·Metal allocator/cache를 완전히 통제하지 못했다. 모델 로드/워밍업과 평가 실행순서는 원본 로그와 schedule로 보존한다.

## 6. 개별 실패와 품질 변화

| 집합 | 방식 | Correct→Wrong(평균) | Wrong→Correct(평균) | 다른 토큰열 요청/전체 |
|---|---|---|---|---|
| replay32 | baseline | 0.00 | 0.00 | 0/96 |
| replay32 | static | 0.00 | 0.00 | 48/96 |
| replay32 | dlr | 0.00 | 0.00 | 0/96 |
| replay32 | conservative | 0.00 | 0.00 | 0/96 |
| fresh32 | baseline | 0.00 | 0.00 | 0/96 |
| fresh32 | static | 0.00 | 2.00 | 15/96 |
| fresh32 | dlr | 0.00 | 0.00 | 0/96 |
| fresh32 | conservative | 0.00 | 0.00 | 0/96 |

이 표본에서 Baseline 대비 Correct→Wrong은 관측되지 않았다. 이는 일반적인 정확도 보존 증거가 아니다.

전체 오답·참조·응답·전환·토큰 변화는 DLR_FAILURES.csv와 DLR_CASES.csv에 있다. 유형별 정확도는 DLR_GROUPS.csv에 있다. aggregate 정답 수가 같아도 개별 손실/개선이 서로 상쇄되는지 확인한다.

## 7. 성공 판정과 다음 가설

실제 평가 재사용 이벤트는 총105회다. 동일 토큰열에서 정확도·실측시간·추정 matmul 기준을 동시에 만족한 제한적 DLR 후보 조건은0개다. 이는 확증 성공 수가 아니다. 자연 생성의 정확도 보존, 두 집합 재현, CPU/메모리와 CI까지 함께 봐야 한다.

| 방식 | 관측 정확도 양셋 비감소 | 1%p 비열등성 입증 | 동일열 추정 matmul 감소/양셋 | 동일열 wall CI 감소/양셋 | 동일열 CPU CI 감소/양셋 | 엄격 성공 입증 |
|---|---|---|---|---|---|---|
| static | True | False | True | False | False | False |
| dlr | True | False | True | False | False | False |
| conservative | True | False | False | False | False | False |

**최종 판단: 새로운 연산 절감 기술을 발견했다는 주장은 하지 않는다.** 임계값을 엄격하게 유지할 때 재사용이 없으면 계산을 줄인 것이 아니며, 재사용해도 실제 시간이 늘면 실패다. 작은 표본·제한적 채점으로 정확도 유지가 입증됐다고 판단하지 않는다. Static의 성능도 DLR의 발견으로 돌리지 않는다.

다음에 검증할 가장 유망한 가설 하나는 **여러 후보의 gate를 GPU 내부에 융합해 host scalar 동기화를 없애면, 동일한 재사용 규칙의 판단 오버헤드를 줄일 수 있는가**이다. 단, 단순 mx.where로 full MLP 양쪽을 계산하면 연산 절감이 아니므로 실제 projection kernel이 실행되지 않는 GPU 조건 분기가 필요하다. 그 구현·별도 개발용 신호 보정이 가능하기 전에는 임계값을 완화한 품질 손실을 성능 성공으로 포장하지 않는다.

## 8. 재현과 보존

960건 저장 뒤 새 분석 파일의 verdict 출력을 수정해 snapshot 해시 검사가 export를 중단했다. 새로 추가한 분석 파일을 동결본으로 복원하고 확장판을 dlr_analysis_final.py로 분리했다. 기존 연구 파일 및 모든 측정 체크포인트는 보존했고 resume의 신규 모델 호출은0회다. 정책·런타임·데이터·채점은 평가 중 변경하지 않았다. EXECUTION_RECOVERY.json에 이 과정을 기록했다.

새 실험 생성은 새 UUID를 만들며, 같은 디렉터리 인자는 동결 코드·패키지 해시를 확인하고 완료된 요청은 재생성하지 않는다. 보고서·분석도 기존 파일을 덮어쓰지 않는다. 신규 문제는 한번 저장하면 미사용 문제로 다시 주장할 수 없으며 freshness 검사로 동일 새 실험 생성을 거부한다. 기존 실행을 재시작하려면 아래 경로를 사용한다.

```sh
.alsr-venv/bin/python -B -m src.dlr_experiment ./data/dlr_5a4656b8d37f48dd8a55a0e82b647b37
.venv/bin/python -B -m src.dlr_analysis_final ./data/dlr_5a4656b8d37f48dd8a55a0e82b647b37
.alsr-venv/bin/python -B -m src.dlr_report <새로 출력된 ANALYSIS 경로>
.venv/bin/python -B -m unittest tests.test_dlr
.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json
```

기존 7536개 파일의 SHA256 보존 검사를 통과했다. 모델 로드마다 고정 가중치 해시도 확인했다. 데이터/런타임/정책 변경 없이 resume 가능하다. 원본 CSV는 EXPORT 폴더, per-request append-only JSON은 checkpoints 폴더에 있다. 정답/참조는 런타임 의사결정과 분리되어 있다.

- DLR_RESULTS.csv: 평균·SD·범위·반복 bootstrap CI
- DLR_REPEATS.csv: 반복별 원래 합계
- DLR_COMPARISONS.csv: paired 시간/CPU/토큰/FLOPs 비율 및 정확도 불확실성
- DLR_REUSE_EVENTS.csv: 실제 거리와 재사용 결정
- DLR_CASES.csv / DLR_FAILURES.csv: 모든 문제별 답변·오류
- DLR_PARETO.csv: 관측점 Pareto (통계적 우위 아님)
- DLR_ORDER_EFFECTS.csv: 순서 위치별 기술통계
- DLR_COMPARISON.png / DLR_PARETO.png: 비교 그래프
