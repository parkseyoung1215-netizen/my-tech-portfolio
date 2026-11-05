# ALSR Layer Impact Study — 사전 설계

이 문서는 새 평가 실행 전에 작성한다. 기존 ALSR 산출물은 수정하지 않는다.

* 모델: 기존 고정 revision Qwen2.5-1.5B-Instruct, MLX BF16, greedy, 기존 chat template. 유료 API 없음.
* 데이터: 새 학습32 / 보정16 / 평가64. 평가에는 객관식48, JSON8, 문장 제약8. 정확한 stem 중복은 과거 기록과 검사하되 합성 템플릿의 의미적 유사성은 별도 한계로 명시한다.
* 인과 진단: 0-based 28개 층을 하나씩 전체 context에서 제거한다. 원래 prompt, 그리고 full 모델 첫 생성 토큰을 붙인 context에서 full-vocabulary KL(P_full || P_skip), Jensen-Shannon, top-1, 양쪽 top-2 logit margin, full top-1의 skip logit margin을 기록한다. EOS를 생성한 prompt에는 두 번째 위치를 만들지 않는다. 기준 미래 토큰은 오프라인 진단 전용이다.
* 계층 실행 전 특징: 해당 층 입력의 8차원 고정 무작위 투영, RMS, 길이, 직전 층의 상대 변화·cosine, 학습에서만 고른 작은 unembedding shortlist의 margin/entropy. 현재 층 출력, 최종 logits, 정답은 live gate에 주지 않는다.
* 위험 학습: 각 층별 ridge로 KL+top1-flip을 예측한다. 학습32개만 사용. layer-mean, 기존 early-state Sketch와 held-out 예측력을 비교한다. 깊이에 의한 상관을 구분하기 위해 층별 지표도 기록한다.
* 최소 실험 gate: 학습의 position0 평균 KL+flip이 가장 낮은 내부 층(2..26) 하나를 후보로 고정한다. 보정 position0에서 예측 임계값(0,.01,.025,.05,.1,.2,.5,1,2) 중 flip0 및 mean KL<=.02를 만족하면서 생략률 최대인 것을 선택한다. 평가에서 변경하지 않는다. 보정 품질 통과는 일반화 보장이 아니다.
* 생성 비교: Full, 기존 frozen Adaptive-Sketch, 기존 Static4, Random4, 학습 고정 Stable1, Random1, PreLayerGate. 기존 4층과 새 1층 방법은 같은 예산이 아님을 명시한다. Stable1/Random1은 1층 기준선이다. 첫 prefill에만 gate, 이후 같은 마스크를 유지하여 KV 정합성을 보장한다.
* 평가: 64문제 x 7방식 x 3회. 문제/방식 순서 무작위 기록. 같은 최대 생성16(MCQ)/48(open), EOS 준수. 별도 사전 고정16문제에서 동일16-token workload, 생성·gate·CPU·MLX peak·FLOPs 추정 기록. 기준 token 시퀀스는 fixed-work 실행기만 알고 gate는 보지 못한다.
* 출력 품질: 내용/형식 별도, 문제별 정답→오답 및 오답→정답, 응답 불일치, 참조 NLL(각 방식 1회, 비용 별도). 진단 top1 유지와 전체 답변 유지의 관계를 분석한다.
* 성공: 품질 손실<=1%p, 자연 생성과 동일 token workload 각각 latency10% 이상 감소, 실제 블록 미호출과 추정 matmul 감소. paired question-cluster CI 제시. 64개는 1%p 모집단 비열등을 증명하기에 부족하다. 약한/부정적 결과를 성공으로 해석하지 않는다.
* CPU=Python process CPU, GPU elapsed/전력/금전 비용 미측정. FLOPs는 기존 ALSR matmul 분석식이며 norms/softmax 등은 제외. MLX allocator peak와 process lifetime RSS를 구분한다.
* ShortGPT(https://arxiv.org/abs/2403.03853), Tuned Lens(https://arxiv.org/abs/2303.08112), LayerSkip(https://arxiv.org/abs/2404.16710), SkipDecode(https://arxiv.org/abs/2307.02628)와 비교한다. probe/ridge/층 중요도는 기존 아이디어이며 새 발명이라고 주장하지 않는다.
