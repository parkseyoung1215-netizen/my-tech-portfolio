# Adaptive Layer Skipping Research (ALSR)

## 결론
**계층 계산을 직접 생략하는 환경은 구현했지만, 품질 유지와 지연10% 감소를 함께 충족하는 확대 대상은 확인하지 못했다. 유리한 사례만 골라 성공으로 표현하지 않는다.**

실험 `alsr_1c1ea4a321cd4b78859d3ac33f4aba5f`. 두 모델, 8방식, 신규32문제, 3회 실제 반복으로 생성 1536건을 측정했다. 동일16토큰 부하의 별도8문제 하위 실험 384건, 참조 NLL 512건(문제/방식당1회)을 분리했다. 전체 MLX 작업 벽시계는 미측정(최종 메타데이터 저장 전 오류)이다. 요청별 생성/고정 부하 CPU·wall 실측 로그는 모두 보존되어 있다. 유료 추론 API는 사용하지 않았다.

## 1. 환경·구현 가능성
실제 장비는 **Apple M4 / 통합 메모리16GiB / macOS15.7.4 / arm64**다. MLX가 보고한 권장 작업 메모리와 물리 메모리를 확인하고, 모델을 하나씩 로드했다. 기존 `.venv`는 변경하지 않고 `.alsr-venv`를 새로 만들었다. 공개 HF 체크포인트는 `/private/tmp/alsr_models_20261105`에 별도 다운로드했으며 기존 Ollama 모델을 대체하거나 삭제하지 않았다.

로컬 GGUF 메타데이터를 직접 읽어 두 모델이 Qwen2.5 Instruct / Qwen2ForCausalLM 계열임을 확인했다. Ollama의 [공개 generate API](https://docs.ollama.com/api/generate)와 [Options/Runner 소스](https://github.com/ollama/ollama/blob/main/api/types.go)에서 계층 생략 제어를 확인하지 못해 지원한다고 가정하지 않았다. `num_gpu` 같은 배치/장치 옵션을 계산 생략으로 해석하지 않는다. 읽기 전용 MLX GGUF 로드는 `gguf_tensor_to_f16 failed`로 실패해 Q4 파일을 임의 변환하지 않았다.

대신 공식 HF Instruct 가중치를 MLX로 로드했다. 실제 dtype은 아래와 같다. 따라서 **Ollama Q4와 HF BF16의 속도·메모리를 직접 비교하지 않는다.** 모든 ALSR 방식은 같은 BF16 체크포인트를 공유한다. 모델은 [0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 모두 모델 카드의 Apache-2.0이고, 기존 GGUF 내 라이선스도 같은 표제를 확인했다. [MLX-LM](https://github.com/ml-explore/mlx-lm)은 MIT 라이선스다.

| 모델 | HF revision | 층 | hidden | FFN | attention/KV heads | 실행 dtype | 로드 활성 MLX GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | 7ae557604adf67be50417f59c2c2f167def9a775 | 24 | 896 | 4864 | 14/2 | mlx.core.bfloat16 | 0.920 |
| Qwen/Qwen2.5-1.5B-Instruct | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 | 28 | 1536 | 8960 | 12/2 | mlx.core.bfloat16 | 2.875 |

패키지: mlx=0.32.4, mlx-lm=0.32.2, transformers=5.22.1, huggingface-hub=2.4.0, numpy=2.5.4. MLX 메모리 한도 8.533GiB, allocator cache128MiB를 동일하게 적용했다. 모델 가중치는 학습·수정하지 않았다.


기준 구현 정합성 관측(복구 실행에서는 보조 재검사):
| 모델 | 전체 logits max 차이 | 마지막 위치 head max 차이 | 마지막 위치 top1 일치 | 실제 생략 trap/정상 호출 대조 |
| --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | 0.0 | 0.015625 | True | True |
| Qwen/Qwen2.5-1.5B-Instruct | 0.0 | 0.03125 | True | True |

모든 시간 비교 방식은 공통으로 마지막 위치만 출력 head에 투영한다. native 전체 logits 경로와의 정합성은 스모크에서 확인했으나 BF16의 head 행렬 크기에 따른 반올림 차이는 위처럼 존재하므로 모든 입력의 bitwise 동등을 주장하지 않는다.


### 최종 저장 오류와 보존 복구
모든3회 반복이 끝난 뒤 보정 trace의 학습행/보정행 열이 달라 legacy CSV writer가 중단했다. 원래1536개 생성·384개 고정 부하·512개 참조 평가와 모델별 profile/고정 token sequence는 이미 저장돼 있었다. 이 성능 출력은 재생성하거나 덮어쓰지 않았다. 부분 보정 CSV도 보존했다.

원래 모델 로드·학습/보정·warmup/준비 비용과 전체 job 종료 clock은 누락돼 **미측정**으로 남겼다. 별도의 postrun 보조 재검사에서 같은 모델/학습12/보정8로 원래 profile을 정확히 재현했고 native/trap/cache를 재검사했다. 해당 trace와 비용은 `_recheck` 파일에 분리했으며 원래 비용인 것처럼 대체하지 않았다. 위 로드 활성 메모리/dtype 표는 이 보조 재검사의 관측값이다. 새 `alsr_run` 진입점은 합집합 열을 만드는 exporter로 같은 저장 오류를 수정하며 원래 모델 코드와 파일을 변경하지 않는다.

| 모델 | 후속 보조 재검사 prefill 수 | 재검사 CPU초 | 재검사 wall초 | 원래 학습/보정 비용 |
| --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | 204 | 4.427 | 9.763 | 미측정 |
| Qwen/Qwen2.5-1.5B-Instruct | 204 | 4.843 | 25.324 | 미측정 |

## 2. 선행 기법과 차이·새로움의 범위
| 기법 | 선행 내용 | 이번 구현과의 관계 |
| --- | --- | --- |
| Early exit | 중간 층에서 출력하고 나머지 suffix를 생략 | 이번에는 마지막 층/출력 head를 남기고 중간 블록을 생략 |
| [ShortGPT](https://arxiv.org/abs/2403.03853) | 입출력 표현 유사성의 Block Influence로 정적 제거 | BI-Static4는 학습12입력·후보8층만 쓰는 부분 기준선, 논문 전체 재현 아님 |
| [LayerSkip](https://arxiv.org/abs/2404.16710) | layer dropout/early-exit 학습 및 self-speculative decoding | 이번 Qwen은 해당 학습을 하지 않았고 full-model correction도 없음 |
| [SkipDecode](https://arxiv.org/abs/2307.02628) | 토큰별 early exit와 batching/KV cache 정합성 문제 | 이번 계획은 입력별 고정이며 토큰별 재진입을 구현하지 않음 |
| [CLaSp](https://arxiv.org/abs/2505.24196) | 완전한 검증 단계 hidden states를 이용한 self-speculative layer 계획 | 이번에는 현재 입력의 첫2층과 과거 학습 통계만 사용 |
| [SkipGPT](https://proceedings.mlr.press/v267/zhao25ab.html) | token-aware routing, attention/MLP 분리 정책과 학습 | 이번에는 전체 블록 단위 생략, 모델 fine-tuning 없음 |
| [GateSkip](https://github.com/Thiggel/GateSkip) | 학습된 residual-stream gate 등 유사한 경량 동적 제어 | 가벼운 게이트라는 발상 자체의 새로움을 주장하지 않음 |
| [2026 Qwen 감사 preprint](https://arxiv.org/abs/2608.28846) | 동일 Qwen2.5 규모에서도 skipping/early-exit 비교가 이미 존재 | 모델 크기나 연구 방향 자체도 새로운 발명이 아님 |

아래 두 규칙은 프로젝트 내 추가 가설이다. 제한된 문헌 조사로 정확히 같은 조합의 선행 부재를 증명할 수 없으며 **전 세계적으로 새로운 기술이라는 근거는 확보하지 못했다.** 성능 효과 역시 실험 전 가정하지 않았다.


## 3. 두 판단 규칙과 실제 생략 구현
**Adaptive-Sketch:** 학습12입력의 단일 층 제거 실험에서 최종 next-token 분포 KL과 top1 변경을 측정한다. 첫2층 마지막 prompt 토큰의8차원 고정 random sketch, 길이·RMS·짧은 출력 후보의 margin으로 층별 위험을 예측하는 ridge 회귀를 맞춘다. 온라인에는 추가 모델 호출이 없고 후보 전체 층을 먼저 계산해 본 뒤 버리지 않는다.

**Adaptive-Leverage:** 학습 시 얻은 후보층의 평균 residual을 작은 출력 head shortlist에 투영해 둔다. 온라인 첫2층 probe의 상위2후보 margin과 RMS로 정규화해 해당 층의 예상 출력 결정 영향도를 추정한다. 전체 vocab projection 대신 학습에서 고정한 작은 shortlist를 사용한다. 이는 실제 출력 기여의 보장된 상한이나 인과적 증명이 아니다.

두 방식 모두 raw hidden 변화량 하나의 임계값과는 다르지만, 출력 민감도/학습된 라우터라는 기존 사고방식의 조합이다. 학습은 모델 가중치가 아닌 경량 회귀·통계만 수행한다. 학습12개와 보정8개는 모두 객관식이고, 평가의 JSON/문장8개는 출력 유형까지 이동하는 OOD 조건이다. 이 한계는 작은 보정 집합으로 다양한 생성 품질을 추정하는 데 불리하다.

시드 **88112026**, 후보8개 중 최대4층. 앞2층은 항상 실행해 probe를 얻고, 그 뒤 **프롬프트별 생략 계획을 고정**한다. 남은 prefill과 모든 decode 토큰에 같은 계획을 사용해 skipped KV offset0, 실행된 cache의 offset 정합성을 유지한다. 토큰마다 다시 켜는 cache backfill이나 KV 근사 복사를 했다고 주장하지 않는다.

보정 임계값0/.05/.15/.5/2 중8개 보정 입력의 다음 토큰 top1이 모두 유지되고 평균 KL<=.05인 설정에서 평균 생략 수가 최대인 값을 선택했다. 동률은 작은 임계값을 택했다. **첫 토큰 보존이 전체 생성 품질 보존을 뜻하지 않는다.** 최종 평가 정답이나 평가 결과로 설정을 조절하지 않았다.

전체 블록 호출 전 분기하므로 생략 층의 attention·MLP 자체를 호출하지 않는다. 실제 모듈 wrapper의 호출 카운터, retained graph의 `mx.eval`, GPU 동기화, cache offset 검사, 호출 시 예외가 나는 trap과 정상 호출 양성 대조를 사용했다. 이 카운터는 GPU의 하드웨어 instruction counter와는 다르다. 계측용 Python wrapper가 모든 방식에 들어간 연구용 구현이므로 최적화된 배포 서버의 절대 성능과 동일하지 않다.


## 4. 비교 조건·데이터·계측
8방식: Baseline, 사전 Static4, Random4, 학습 BI-Static4, 적응형2종, 각 적응형과 동일 생략량의 matched random2종이다. Matched random도 같은 adaptive probe를 실제 계산해 생략 수를 얻고 그 overhead를 부담한다. 단순히 무작위 기준선에 공짜 생략량을 주지 않았다.

새 학습12·보정8·평가32를 분리했고 과거 정규화 stem 중복을 검사했다. 평가32개는 산술/논리/코딩/읽기 각6개 객관식과 JSON4개·문장 제약4개다. **계산 도구는 사용하지 않고 모델이 모든 답변을 직접 생성**했다. 전체32개 과제 성공률과 객관식24개 정확도를 분리한다. 문장 제약 성공은 완전한 의미·사실성 평가가 아니다.

모든 방식은 같은 chat template, greedy decoding, EOS, 객관식 최대16·나머지 최대48토큰을 사용했다. 반복3회에 문제/방식 순서를 무작위화해 저장했다. 모델 순서는 메모리를 고려해0.5B→1.5B로 고정했으므로 모델 간 절대 속도를 열/순서 영향과 완전히 분리하지 못했다. 자연 생성 서비스 시간에는 tokenization·gate·모든 모델 forward·decode·동기화가 포함된다. 모델 로드/학습/채점/로그 저장은 단계별로 분리했다. CPU는 Python process_time이며 모델의 모든 GPU 시간을 포함하지 않는다.

추가8입력(객관식4·JSON2·문장2)에는 미리 기록한 Baseline token sequence를 EOS로 padding해 **동일16토큰** 부하를 모든 방식에 입력했다. 이는 자연 생성 품질 시험이 아닌 같은 일의 계산 비용 비교다. 계획은 prompt만으로 정하고 미래 고정 토큰을 보지 않으며 자연 생성과 같은 mask인지 검사했다. 짧은 출력/EOS 때문에 빨라진 것을 순수 layer 효율로 해석하지 않는다.

참조 NLL은 사람 작성 reference의 teacher forcing으로 문제/방식당1회만 계산했다. 자연 생성의 prompt-only mask를 그대로 고정하고 이 단계에서는 gate 호출이0인지 검사했다. 참조 토큰이 계획 선택에 들어가지 않는다. NLL 채점 비용은 서비스 지연에서 제외하고 별도로 보존했다. 동등한 다른 정답 표현의 NLL도 달라질 수 있어 단독 의미 품질 판정이 아니다.

GPU elapsed counter를 얻지 못했다. MLX 동기화 wall은 GPU+host의 실제 경과 시간이며 GPU 시간으로 바꾸어 쓰지 않았다. `xcrun --find xctrace`는 이 환경에서 도구를 찾지 못했다. MLX peak allocated memory는 요청마다 reset해 관측했으며 프로세스 RSS 전후와 lifetime 최대 RSS는 별도 저장했다. lifetime RSS를 방식별 peak처럼 비교하지 않는다. **전력·금전 비용은 미측정**이다.


## 5. 주요 결과: 3회 평균
전체 성공률에는 JSON/문장 제약도 포함하므로 객관식 정확도와 동일한 지표가 아니다. 시간은32요청 합계, 고정 부하는8요청 합계다.
| 모델 | 방식 | 평균 생략층 | 전체 성공률32 | 객관식 정확도24 | 형식 준수 | 자연 생성초 | 고정16토큰초 | CPU초 | MLX peak GiB | 참조 NLL¹ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5B-Instruct | Baseline | 0.00 | 59.38% | 45.83% | 75.00% | 3.635 | 1.774 | 1.130 | 1.051 | 0.983 |
| 0.5B-Instruct | Static4 | 4.00 | 0.00% | 0.00% | 9.38% | 6.862 | 1.550 | 2.150 | 1.048 | 3.368 |
| 0.5B-Instruct | Random4 | 4.00 | 3.12% | 2.78% | 6.25% | 6.688 | 1.617 | 2.029 | 1.048 | 4.022 |
| 0.5B-Instruct | BI-Static4 | 4.00 | 0.00% | 0.00% | 0.00% | 5.128 | 1.543 | 1.559 | 1.048 | 3.751 |
| 0.5B-Instruct | Adaptive-Sketch | 1.12 | 43.75% | 37.50% | 31.25% | 4.326 | 1.704 | 1.310 | 1.050 | 1.169 |
| 0.5B-Instruct | MatchedRandom-S | 1.12 | 44.79% | 38.89% | 56.25% | 4.442 | 1.720 | 1.336 | 1.049 | 1.854 |
| 0.5B-Instruct | Adaptive-Leverage | 0.00 | 59.38% | 45.83% | 75.00% | 3.638 | 1.788 | 1.129 | 1.050 | 0.983 |
| 0.5B-Instruct | MatchedRandom-L | 0.00 | 59.38% | 45.83% | 75.00% | 3.631 | 1.754 | 1.133 | 1.050 | 0.983 |
| 1.5B-Instruct | Baseline | 0.00 | 78.12% | 70.83% | 43.75% | 12.805 | 5.471 | 3.586 | 3.012 | 1.443 |
| 1.5B-Instruct | Static4 | 4.00 | 3.12% | 4.17% | 18.75% | 15.061 | 5.003 | 4.171 | 3.011 | 1.733 |
| 1.5B-Instruct | Random4 | 4.00 | 16.67% | 6.94% | 12.50% | 17.663 | 4.799 | 5.326 | 3.011 | 2.997 |
| 1.5B-Instruct | BI-Static4 | 4.00 | 21.88% | 0.00% | 9.38% | 22.191 | 4.790 | 6.566 | 3.011 | 3.098 |
| 1.5B-Instruct | Adaptive-Sketch | 1.16 | 78.12% | 70.83% | 50.00% | 12.766 | 5.296 | 3.444 | 3.012 | 1.484 |
| 1.5B-Instruct | MatchedRandom-S | 1.16 | 57.29% | 45.83% | 40.62% | 14.569 | 5.453 | 4.089 | 3.012 | 1.819 |
| 1.5B-Instruct | Adaptive-Leverage | 1.12 | 71.88% | 62.50% | 40.62% | 12.437 | 5.293 | 3.521 | 3.012 | 1.349 |
| 1.5B-Instruct | MatchedRandom-L | 1.12 | 55.21% | 41.67% | 40.62% | 13.367 | 5.257 | 3.813 | 3.012 | 1.851 |

¹ NLL은 문제별1회, 총32개의 평균이며3회 반복 측정이 아니다.


## 6. 계산량·지연 비교와 불확실성
FLOPs는 dense matmul의 MAC×2 분석값이다. 층당 attention projection `T*(4d²+4d*d_kv)`, MLP `6T*d*I`, attention QK/AV `4T*S*d`, 출력 head `2*d*V*출력위치수`를 더했다. 작은 gate matmul도 추정에 포함한다. norm·softmax·RoPE·비선형 함수와 하드웨어 fusion/causal 효율은 빠져 있어 실제 GPU FLOP counter가 아니다. 레이어 수 비율을 전체 FLOPs 비율로 단순 대체하지 않았다.

| 모델 | 방식 | 성공률 차이 pp | question95 CI pp | 자연 wall비 | 자연95 CI | 고정 wall비 | 고정95 CI | 고정 분석 FLOPs비 | NLL 차이 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5B-Instruct | Static4 | -59.375 | [-75.00,-40.62] | 1.888 | [1.509,2.421] | 0.874 | [0.843,0.893] | 0.844 | 2.385 |
| 0.5B-Instruct | Random4 | -56.250 | [-72.92,-38.54] | 1.840 | [1.434,2.426] | 0.911 | [0.859,1.013] | 0.844 | 3.038 |
| 0.5B-Instruct | BI-Static4 | -59.375 | [-75.00,-40.62] | 1.411 | [1.145,1.853] | 0.870 | [0.827,0.897] | 0.844 | 2.768 |
| 0.5B-Instruct | Adaptive-Sketch | -15.625 | [-28.12,-3.12] | 1.190 | [1.063,1.389] | 0.960 | [0.915,0.993] | 0.963 | 0.186 |
| 0.5B-Instruct | MatchedRandom-S | -14.583 | [-26.04,-2.08] | 1.222 | [1.046,1.498] | 0.969 | [0.930,1.006] | 0.963 | 0.870 |
| 0.5B-Instruct | Adaptive-Leverage | 0.000 | [0.00,0.00] | 1.001 | [0.958,1.023] | 1.008 | [0.962,1.074] | 1.000 | 0.000 |
| 0.5B-Instruct | MatchedRandom-L | 0.000 | [0.00,0.00] | 0.999 | [0.959,1.023] | 0.989 | [0.947,1.014] | 1.000 | 0.000 |
| 1.5B-Instruct | Static4 | -75.000 | [-90.62,-59.38] | 1.176 | [0.973,1.477] | 0.914 | [0.868,1.018] | 0.861 | 0.289 |
| 1.5B-Instruct | Random4 | -61.458 | [-78.12,-43.72] | 1.379 | [1.143,1.699] | 0.877 | [0.851,0.908] | 0.861 | 1.553 |
| 1.5B-Instruct | BI-Static4 | -56.250 | [-71.88,-37.50] | 1.733 | [1.437,2.143] | 0.875 | [0.845,0.917] | 0.861 | 1.655 |
| 1.5B-Instruct | Adaptive-Sketch | 0.000 | [-9.38,9.38] | 0.997 | [0.919,1.116] | 0.968 | [0.946,0.993] | 0.960 | 0.041 |
| 1.5B-Instruct | MatchedRandom-S | -20.833 | [-33.36,-7.29] | 1.138 | [0.997,1.331] | 0.997 | [0.961,1.038] | 0.960 | 0.375 |
| 1.5B-Instruct | Adaptive-Leverage | -6.250 | [-15.62,0.00] | 0.971 | [0.946,1.003] | 0.967 | [0.949,0.988] | 0.959 | -0.094 |
| 1.5B-Instruct | MatchedRandom-L | -22.917 | [-35.42,-11.46] | 1.044 | [0.935,1.198] | 0.961 | [0.941,0.986] | 0.959 | 0.408 |

고정 token 부하에서는 실제 실행 블록이 줄었는지를 확인하고, 자연 생성은 토큰 수가 달라지는 효과를 함께 본다. 모든 방식은 같은 수의 생성 요청을 처리했으므로 모델 호출 수 감소를 성과로 삼지 않았다. 모델 가중치는 계속 상주하며 layer skipping이 parameter memory 감소를 뜻하지 않는다.

평균·표본 SD·반복 CI는 CSV에 있다. 정확도 차이는32문제 단위의 반복 묶음 bootstrap4,000회와 보수적 전이 하한, 속도비는 문제와 반복을 대응시킨 bootstrap4,000회를 사용했다.3회 반복을96개 새 문제처럼 세지 않는다.32문제에서 한 문제는3.125%p, 객관식24개에서는4.167%p로1%p 목표를 정밀하게 확인하기 어렵다. 고정 부하도8입력의 작은 하위 실험이다.


동일 생략량/동일 probe overhead의 무작위 대조군과 비교:
| 모델 | 적응형 | 대조군 | 성공률 차이 pp | 95 CI pp | 자연 wall비 | 고정 wall비 | NLL 차이 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5B-Instruct | Adaptive-Sketch | MatchedRandom-S | -1.042 | [-14.58,12.50] | 0.974 | 0.991 | -0.685 |
| 0.5B-Instruct | Adaptive-Leverage | MatchedRandom-L | 0.000 | [0.00,0.00] | 1.002 | 1.019 | 0.000 |
| 1.5B-Instruct | Adaptive-Sketch | MatchedRandom-S | 20.833 | [10.42,31.25] | 0.876 | 0.971 | -0.335 |
| 1.5B-Instruct | Adaptive-Leverage | MatchedRandom-L | 16.667 | [4.17,29.17] | 0.930 | 1.007 | -0.502 |

탐색적 관측: 1.5B의 Adaptive-Sketch는 같은 생략량의 무작위 대조군보다 전체 과제 성공률이20.83%p 높았고 문제 bootstrap95 CI는[10.42,31.25]%p였다. 그러나 다중 비교 보정이 없고 보수적 전이 하한은 통과하지 못했으며0.5B에서는 같은 이점이 재현되지 않았다. 이 신호를 새로운 기술이나 확증된 일반화 우위로 부르지 않는다.


## 7. 유형별 품질·실패 사례
| 모델 | 방식 | 유형 | 고유 문제 수 | 성공률 | 형식 준수 | 평균 생략층 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5B-Instruct | Baseline | arithmetic | 6.0 | 16.67% | 100.00% | 0.00 |
| 0.5B-Instruct | Baseline | coding | 6.0 | 66.67% | 100.00% | 0.00 |
| 0.5B-Instruct | Baseline | logic | 6.0 | 33.33% | 100.00% | 0.00 |
| 0.5B-Instruct | Baseline | reading | 6.0 | 66.67% | 33.33% | 0.00 |
| 0.5B-Instruct | Baseline | structured_output | 4.0 | 100.00% | 0.00% | 0.00 |
| 0.5B-Instruct | Baseline | text_quality | 4.0 | 100.00% | 100.00% | 0.00 |
| 0.5B-Instruct | Static4 | arithmetic | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | Static4 | coding | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | Static4 | logic | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | Static4 | reading | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | Static4 | structured_output | 4.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | Static4 | text_quality | 4.0 | 0.00% | 75.00% | 4.00 |
| 0.5B-Instruct | Random4 | arithmetic | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | Random4 | coding | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | Random4 | logic | 6.0 | 11.11% | 0.00% | 4.00 |
| 0.5B-Instruct | Random4 | reading | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | Random4 | structured_output | 4.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | Random4 | text_quality | 4.0 | 8.33% | 50.00% | 4.00 |
| 0.5B-Instruct | BI-Static4 | arithmetic | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | BI-Static4 | coding | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | BI-Static4 | logic | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | BI-Static4 | reading | 6.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | BI-Static4 | structured_output | 4.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | BI-Static4 | text_quality | 4.0 | 0.00% | 0.00% | 4.00 |
| 0.5B-Instruct | Adaptive-Sketch | arithmetic | 6.0 | 16.67% | 0.00% | 2.00 |
| 0.5B-Instruct | Adaptive-Sketch | coding | 6.0 | 50.00% | 100.00% | 1.00 |
| 0.5B-Instruct | Adaptive-Sketch | logic | 6.0 | 33.33% | 33.33% | 0.67 |
| 0.5B-Instruct | Adaptive-Sketch | reading | 6.0 | 50.00% | 16.67% | 1.17 |
| 0.5B-Instruct | Adaptive-Sketch | structured_output | 4.0 | 100.00% | 0.00% | 1.00 |
| 0.5B-Instruct | Adaptive-Sketch | text_quality | 4.0 | 25.00% | 25.00% | 0.75 |
| 0.5B-Instruct | MatchedRandom-S | arithmetic | 6.0 | 11.11% | 33.33% | 2.00 |
| 0.5B-Instruct | MatchedRandom-S | coding | 6.0 | 50.00% | 77.78% | 1.00 |
| 0.5B-Instruct | MatchedRandom-S | logic | 6.0 | 55.56% | 88.89% | 0.67 |
| 0.5B-Instruct | MatchedRandom-S | reading | 6.0 | 38.89% | 44.44% | 1.17 |
| 0.5B-Instruct | MatchedRandom-S | structured_output | 4.0 | 58.33% | 0.00% | 1.00 |
| 0.5B-Instruct | MatchedRandom-S | text_quality | 4.0 | 66.67% | 83.33% | 0.75 |
| 0.5B-Instruct | Adaptive-Leverage | arithmetic | 6.0 | 16.67% | 100.00% | 0.00 |
| 0.5B-Instruct | Adaptive-Leverage | coding | 6.0 | 66.67% | 100.00% | 0.00 |
| 0.5B-Instruct | Adaptive-Leverage | logic | 6.0 | 33.33% | 100.00% | 0.00 |
| 0.5B-Instruct | Adaptive-Leverage | reading | 6.0 | 66.67% | 33.33% | 0.00 |
| 0.5B-Instruct | Adaptive-Leverage | structured_output | 4.0 | 100.00% | 0.00% | 0.00 |
| 0.5B-Instruct | Adaptive-Leverage | text_quality | 4.0 | 100.00% | 100.00% | 0.00 |
| 0.5B-Instruct | MatchedRandom-L | arithmetic | 6.0 | 16.67% | 100.00% | 0.00 |
| 0.5B-Instruct | MatchedRandom-L | coding | 6.0 | 66.67% | 100.00% | 0.00 |
| 0.5B-Instruct | MatchedRandom-L | logic | 6.0 | 33.33% | 100.00% | 0.00 |
| 0.5B-Instruct | MatchedRandom-L | reading | 6.0 | 66.67% | 33.33% | 0.00 |
| 0.5B-Instruct | MatchedRandom-L | structured_output | 4.0 | 100.00% | 0.00% | 0.00 |
| 0.5B-Instruct | MatchedRandom-L | text_quality | 4.0 | 100.00% | 100.00% | 0.00 |
| 1.5B-Instruct | Baseline | arithmetic | 6.0 | 50.00% | 0.00% | 0.00 |
| 1.5B-Instruct | Baseline | coding | 6.0 | 66.67% | 16.67% | 0.00 |
| 1.5B-Instruct | Baseline | logic | 6.0 | 66.67% | 50.00% | 0.00 |
| 1.5B-Instruct | Baseline | reading | 6.0 | 100.00% | 100.00% | 0.00 |
| 1.5B-Instruct | Baseline | structured_output | 4.0 | 100.00% | 0.00% | 0.00 |
| 1.5B-Instruct | Baseline | text_quality | 4.0 | 100.00% | 100.00% | 0.00 |
| 1.5B-Instruct | Static4 | arithmetic | 6.0 | 0.00% | 0.00% | 4.00 |
| 1.5B-Instruct | Static4 | coding | 6.0 | 0.00% | 0.00% | 4.00 |
| 1.5B-Instruct | Static4 | logic | 6.0 | 16.67% | 33.33% | 4.00 |
| 1.5B-Instruct | Static4 | reading | 6.0 | 0.00% | 0.00% | 4.00 |
| 1.5B-Instruct | Static4 | structured_output | 4.0 | 0.00% | 0.00% | 4.00 |
| 1.5B-Instruct | Static4 | text_quality | 4.0 | 0.00% | 100.00% | 4.00 |
| 1.5B-Instruct | Random4 | arithmetic | 6.0 | 5.56% | 5.56% | 4.00 |
| 1.5B-Instruct | Random4 | coding | 6.0 | 16.67% | 5.56% | 4.00 |
| 1.5B-Instruct | Random4 | logic | 6.0 | 0.00% | 0.00% | 4.00 |
| 1.5B-Instruct | Random4 | reading | 6.0 | 5.56% | 0.00% | 4.00 |
| 1.5B-Instruct | Random4 | structured_output | 4.0 | 41.67% | 8.33% | 4.00 |
| 1.5B-Instruct | Random4 | text_quality | 4.0 | 50.00% | 75.00% | 4.00 |
| 1.5B-Instruct | BI-Static4 | arithmetic | 6.0 | 0.00% | 0.00% | 4.00 |
| 1.5B-Instruct | BI-Static4 | coding | 6.0 | 0.00% | 0.00% | 4.00 |
| 1.5B-Instruct | BI-Static4 | logic | 6.0 | 0.00% | 0.00% | 4.00 |
| 1.5B-Instruct | BI-Static4 | reading | 6.0 | 0.00% | 0.00% | 4.00 |
| 1.5B-Instruct | BI-Static4 | structured_output | 4.0 | 100.00% | 0.00% | 4.00 |
| 1.5B-Instruct | BI-Static4 | text_quality | 4.0 | 75.00% | 75.00% | 4.00 |
| 1.5B-Instruct | Adaptive-Sketch | arithmetic | 6.0 | 66.67% | 33.33% | 1.83 |
| 1.5B-Instruct | Adaptive-Sketch | coding | 6.0 | 50.00% | 16.67% | 0.17 |
| 1.5B-Instruct | Adaptive-Sketch | logic | 6.0 | 66.67% | 50.00% | 1.17 |
| 1.5B-Instruct | Adaptive-Sketch | reading | 6.0 | 100.00% | 100.00% | 1.00 |
| 1.5B-Instruct | Adaptive-Sketch | structured_output | 4.0 | 100.00% | 0.00% | 2.00 |
| 1.5B-Instruct | Adaptive-Sketch | text_quality | 4.0 | 100.00% | 100.00% | 1.00 |
| 1.5B-Instruct | MatchedRandom-S | arithmetic | 6.0 | 27.78% | 38.89% | 1.83 |
| 1.5B-Instruct | MatchedRandom-S | coding | 6.0 | 50.00% | 22.22% | 0.17 |
| 1.5B-Instruct | MatchedRandom-S | logic | 6.0 | 38.89% | 22.22% | 1.17 |
| 1.5B-Instruct | MatchedRandom-S | reading | 6.0 | 66.67% | 61.11% | 1.00 |
| 1.5B-Instruct | MatchedRandom-S | structured_output | 4.0 | 83.33% | 8.33% | 2.00 |
| 1.5B-Instruct | MatchedRandom-S | text_quality | 4.0 | 100.00% | 100.00% | 1.00 |
| 1.5B-Instruct | Adaptive-Leverage | arithmetic | 6.0 | 50.00% | 0.00% | 1.00 |
| 1.5B-Instruct | Adaptive-Leverage | coding | 6.0 | 33.33% | 0.00% | 1.00 |
| 1.5B-Instruct | Adaptive-Leverage | logic | 6.0 | 66.67% | 50.00% | 1.00 |
| 1.5B-Instruct | Adaptive-Leverage | reading | 6.0 | 100.00% | 100.00% | 1.00 |
| 1.5B-Instruct | Adaptive-Leverage | structured_output | 4.0 | 100.00% | 0.00% | 1.00 |
| 1.5B-Instruct | Adaptive-Leverage | text_quality | 4.0 | 100.00% | 100.00% | 2.00 |
| 1.5B-Instruct | MatchedRandom-L | arithmetic | 6.0 | 16.67% | 33.33% | 1.00 |
| 1.5B-Instruct | MatchedRandom-L | coding | 6.0 | 44.44% | 33.33% | 1.00 |
| 1.5B-Instruct | MatchedRandom-L | logic | 6.0 | 38.89% | 22.22% | 1.00 |
| 1.5B-Instruct | MatchedRandom-L | reading | 6.0 | 66.67% | 61.11% | 1.00 |
| 1.5B-Instruct | MatchedRandom-L | structured_output | 4.0 | 100.00% | 0.00% | 1.00 |
| 1.5B-Instruct | MatchedRandom-L | text_quality | 4.0 | 91.67% | 100.00% | 2.00 |

첫 반복에서 기준 답변이 맞았으나 수정 모델이 실패한 사후 예시(학습/임계값 조정에 쓰지 않음):
| 모델 | 방식 | ID | 질문 | 참조 | 기준 출력 | 생략 출력 | 생략층 | 오류 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5B-Instruct | Static4 | 739df947f255 | Note 19104: Mira sent the letter; Navo translated it; Oren archived it. Who translated the letter? A) Oren B) Mira C) Navo D) Nobody Return only the option letter. | C | C) Navo | You are a helpful assistant in my previous questions, but I have a question about | [6, 10, 15, 19] | format_or_unresolved |
| 0.5B-Instruct | Static4 | 1b8c8886bbef | Rewrite as one concise sentence preserving the person and room: "The visitor is Lira19304. The visitor waits in Room 5." Do not add facts. | Lira19304 waits in Room 5. | Lira19304 waits in Room 5. | I'm not sure how to respond to the original text. Instead of using a sentence to rewrite, I've created a new sentence to generate a new sentence. I can create a sentence to describe the person in a way that is not related | [6, 10, 15, 19] | constraint_failure |
| 0.5B-Instruct | BI-Static4 | 739df947f255 | Note 19104: Mira sent the letter; Navo translated it; Oren archived it. Who translated the letter? A) Oren B) Mira C) Navo D) Nobody Return only the option letter. | C | C) Navo | The option letter A) Oren is the correct answer. | [10, 13, 17, 19] | format_or_unresolved |
| 0.5B-Instruct | BI-Static4 | 1b8c8886bbef | Rewrite as one concise sentence preserving the person and room: "The visitor is Lira19304. The visitor waits in Room 5." Do not add facts. | Lira19304 waits in Room 5. | Lira19304 waits in Room 5. | The visitor is Lira19304. The visitor waits in Room 5. | [10, 13, 17, 19] | constraint_failure |
| 0.5B-Instruct | Adaptive-Sketch | 739df947f255 | Note 19104: Mira sent the letter; Navo translated it; Oren archived it. Who translated the letter? A) Oren B) Mira C) Navo D) Nobody Return only the option letter. | C | C) Navo | B) Mira | [17] | content |
| 0.5B-Instruct | Adaptive-Sketch | 15d83d9c4414 | Rewrite as one concise sentence preserving the person and room: "The visitor is Lira19301. The visitor waits in Room 2." Do not add facts. | Lira19301 waits in Room 2. | Lira19301 is the visitor, waiting in Room 2. | The visitor is Lira19301. The visitor waits in Room 2. | [8] | constraint_failure |
| 1.5B-Instruct | Static4 | b9c9d5e1afc3 | Puzzle 19106: The list from youngest to oldest is Rena19106, Somu19106, Tavi19106, Ula19106. Who is position 2 in that order? A) Tavi19106 B) Somu19106 C) Rena19106 D) Ula19106 Return only the option letter. | B | B | Tavi1900101.01.01. | [7, 12, 18, 23] | format_or_unresolved |
| 1.5B-Instruct | Static4 | 739df947f255 | Note 19104: Mira sent the letter; Navo translated it; Oren archived it. Who translated the letter? A) Oren B) Mira C) Navo D) Nobody Return only the option letter. | C | C | A) Mira. | [7, 12, 18, 23] | format_or_unresolved |
| 1.5B-Instruct | BI-Static4 | b9c9d5e1afc3 | Puzzle 19106: The list from youngest to oldest is Rena19106, Somu19106, Tavi19106, Ula19106. Who is position 2 in that order? A) Tavi19106 B) Somu19106 C) Rena19106 D) Ula19106 Return only the option letter. | B | B | Rena19106, Somu19106, | [12, 15, 18, 23] | format_or_unresolved |
| 1.5B-Instruct | BI-Static4 | 739df947f255 | Note 19104: Mira sent the letter; Navo translated it; Oren archived it. Who translated the letter? A) Oren B) Mira C) Navo D) Nobody Return only the option letter. | C | C | A) Oren B) Navo C) Nobody D) | [12, 15, 18, 23] | format_or_unresolved |
| 1.5B-Instruct | Adaptive-Sketch | d13f632e9cc3 | Python case 19106: ```python a = [33, 36, 41] print(a[-2]) ``` What is printed? A) 41 B) 36 C) 33 D) Error Return only the option letter. | B | B) | A) 41 | [4] | content |
| 1.5B-Instruct | Adaptive-Leverage | d13f632e9cc3 | Python case 19106: ```python a = [33, 36, 41] print(a[-2]) ``` What is printed? A) 41 B) 36 C) 33 D) Error Return only the option letter. | B | B) | C) | [23] | content |
| 1.5B-Instruct | Adaptive-Leverage | db6f936b9c91 | Python case 19104: ```python a = [14, 17, 22] print(a[-2]) ``` What is printed? A) 22 B) Error C) 14 D) 17 Return only the option letter. | D | D | C) | [23] | content |

MCQ는 정답을 보지 않는 고정 추출 규칙으로 letter/해당 보기 본문/명시적인 답 표현을 인식하고 형식 준수는 별도 측정했다. JSON은 객체 일치와 중복 키를 검사하며 문장4개는 이름·방과 한 문장 제약만 검사한다. 보수적인 parser가 더 긴 정답 설명을 미해결로 분류할 수 있고, 제약 통과가 사실성·문체의 완전한 보장은 아니다. 참조 NLL과 원문 실패 사례를 함께 제공한다.

실패 가능 요인은 학습/보정 표본이 작음, 단일 토큰 logit 효과를 전체 생성 품질의 대리목표로 사용함, 학습에 없던 열린 출력 유형, 여러 층을 동시에 생략할 때의 상호작용, 짧은 요청에서의 GPU-host 동기화 overhead다. 이는 설계와 관측에 근거한 진단이며 모델 내부 원인을 확정하거나 실패를 일반 법칙으로 주장하지 않는다.


## 8. 파일럿 판정과 확대 여부
사전 목표는 성공률/형식 손실1%p 이내, NLL 증가 .05nats 이내, 자연 생성과 고정 부하 각각10% 이상 지연 감소, 실제 생략·분석 계산량 감소다. 아래는 점추정 품질 조건과 속도 CI를 함께 확인한 **확대 검토 여부**이며 신기술 성공 판정이 아니다.
| 모델 | 방식 | 확대 검토 후보 | 사전 파일럿 조건 | 일반화 품질1pp 비열등 확정 |
| --- | --- | --- | --- | --- |
| 0.5B-Instruct | Static4 | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": true, "real_skip_and_lower_estimated_compute": true} | False |
| 0.5B-Instruct | Random4 | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": true} | False |
| 0.5B-Instruct | BI-Static4 | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": true, "real_skip_and_lower_estimated_compute": true} | False |
| 0.5B-Instruct | Adaptive-Sketch | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": true} | False |
| 0.5B-Instruct | MatchedRandom-S | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": true} | False |
| 0.5B-Instruct | Adaptive-Leverage | False | {"quality_point_within1pp": true, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": false} | False |
| 0.5B-Instruct | MatchedRandom-L | False | {"quality_point_within1pp": true, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": false} | False |
| 1.5B-Instruct | Static4 | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": true} | False |
| 1.5B-Instruct | Random4 | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": true} | False |
| 1.5B-Instruct | BI-Static4 | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": true} | False |
| 1.5B-Instruct | Adaptive-Sketch | False | {"quality_point_within1pp": true, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": true} | False |
| 1.5B-Instruct | MatchedRandom-S | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": true} | False |
| 1.5B-Instruct | Adaptive-Leverage | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": true} | False |
| 1.5B-Instruct | MatchedRandom-L | False | {"quality_point_within1pp": false, "natural_speed10_confident": false, "fixed_speed10_confident": false, "real_skip_and_lower_estimated_compute": true} | False |

동시 조건을 충족하는 후보가 없어 대규모 실험으로 확대하지 않았다. 설정을 평가 결과에 맞춰 바꾸거나 유리한 데이터만 선택하지 않았다.


## 9. 마지막 세 질문을 분리한 결론
1. **실제 계산량 감소에 성공했는가?** 일부 방식에서는 그렇다. 실제 블록·attention·MLP 미호출과 KV 정합성 검사를 통과했고 동일 token 부하의 분석 matmul FLOPs도 감소했다. 다만 GPU 하드웨어 FLOPs·전력·금전 비용 감소를 직접 측정한 것은 아니다.
2. **품질을 유지하면서 속도도 개선했는가?** 이번 파일럿에서 두 조건을 함께 충족하는 확대 후보는 확인하지 못했다. 단순히 출력이 짧아졌거나 생략 층 수가 늘어난 것만으로 성공을 선언하지 않는다.
3. **기존 기술과 구별되는 새로운 발견이 있었는가?** 독립적인 새 기술·전 세계적 새로움·기존 대표 기법 대비 우위를 주장할 근거는 없다. 두 판단 규칙의 로컬 적용과 실제 호출 계측을 추가한 파일럿이다.


## 10. 보존·테스트·재현
실행 전 파일 **672개 해시 유지**. 과거 stem 중복0, 학습/보정/평가 분리, profile hash 및 보정 선택 재구성, 모든 블록/attention/MLP 호출·KV offset·matched random 생략량·NLL gate0 검사 통과. 생략0인데 기준 token 응답과 달랐던 adaptive 기록은 0개다(수치적 평가 순서 가능성도 있어 생략 효과로 단정하지 않음). 기존 Ollama GGUF의 SHA256도 종료 시 확인했다.

자동 테스트: **218개 통과 (76.150초, 역사 입력 스냅샷의 전체 unittest; 실제 GPU 정합성/호출 검사 별도)**. 원래 환경에는 MLX를 설치하지 않았고 기존 테스트는 고정 역사 입력의 임시 스냅샷으로 실행해 실제 데이터를 지우지 않았다. 실측표는 테스트 모의 응답으로 만들지 않았다. GPU native parity/trap/cache smoke와 실제 반복 로그는 별도 증거다.

재현 명령(프로젝트 루트, Metal 접근 필요):

```bash
.alsr-venv/bin/python -B -u -m src.alsr_prepare
.alsr-venv/bin/python -B -u -m src.alsr_run ./data/alsr_setup_1a6f378749694d5d9fd1f51937e952c0.json
.venv/bin/python -B -m src.alsr_analysis data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_reanalysis
.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json
.venv/bin/python -B -m src.alsr_report data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f "<실제 테스트 결과>" data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_reanalysis
```

새 동일 데이터 생성은 이미 저장된 stem이 있으면 중단한다. 원래 split·모델 revision·프롬프트·threshold·seed·schedule은 manifest/profile에 보존했다. 분석 출력은 새 폴더를 사용하며 기존 결과를 덮어쓰지 않는다. 임시 HF 캐시가 없어졌으면 setup의 고정 revision으로 다시 받아야 한다.

[결과](ALSR_RESULTS.csv) · [반복 지표](ALSR_REPEAT_METRICS.csv) · [기준 비교](ALSR_COMPARISONS.csv) · [동일 생략량 비교](ALSR_MATCHED_COMPARISONS.csv) · [실패 사례](ALSR_FAILURE_CASES.csv) · [감사](ALSR_AUDIT.json) · [비교 그림](ALSR_COMPARISON.png) · [계산량/메모리 그림](ALSR_EFFICIENCY.png)

원시 실제 실행: `../alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_generation.csv`, `../alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_fixed_work.csv`, `../alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_quality.csv`. 보조 trace: `../alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_calibration_recheck.csv`, `../alsr_1c1ea4a321cd4b78859d3ac33f4aba5f_native_parity_recheck.csv`. 원래 부분 calibration CSV도 그대로 보존돼 있다.

한계: 영어 합성32문제, 보정8개, 일부 출력 유형 OOD, parser 의존, 정확한 전력/GPU 시간 미측정, Python 계측기 overhead, 비전용 노트북 자원/열/캐시, greedy/BF16/Qwen2.5 두 모델만의 결과다. 다른 모델·장문·실서비스로 일반화하거나 논문의 성능을 재현했다고 주장하지 않는다.

