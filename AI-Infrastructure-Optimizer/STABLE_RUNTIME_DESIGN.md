# Stable1 Runtime Attribution — 사전 실험 설계

목표는 실제 층 생략, 응답 길이, 실행기 비용을 분리한다. 기존 실험은 변경하지 않는다.

* Qwen2.5-1.5B-Instruct 기존 고정 HF revision, BF16/MLX. Full, Stable1(0-based24), 기존 frozen Adaptive-Sketch. 학습/임계값 재조정 없음.
* replay64 + 미사용 fresh64(산술/논리/코딩/독해 각12, JSON8, 문장제약8). fresh는 새 템플릿/값을 쓰되 합성 문제의 의미적 유사성을 배제했다고 주장하지 않는다. 과거 exact stem 및 split 중복 검사. 정답은 오프라인 채점에만 사용.
* legacy와 lean의2실행기 x3정책, 각128문제 x3회, 문제/방식 순서를 무작위화. 같은 chat template/greedy/EOS/MCQ16/open48 상한. legacy는 기존 native blocks·카운터·로깅·동기화 경로. lean은 공식 Qwen2 block과 같은 attention/RMS/MLP/잔차 연산 순서, native module 참조 실행계획을 prompt당1회 생성, 카운터1개/층, 상세 dict 생성 제거, argmax까지 한 번에 materialize, device token을 다음 decode 입력으로 사용한다. 가중치 복사·양자화·모델/레이어 선택 변경 없음.
* 별도 조건 lean_option: 세 정책 모두 명시적 요청 `Return only the option letter.`에 한하여 생성 prefix가 오직 A-D 한 글자가 되면 종료한다. 정답·선택지 본문·score_kind를 판단에 사용하지 않는다. JSON/문장은 기존 EOS/상한. 이 정지 조건은 순수 실행기 최적화와 분리하며 같은 출력 조건 내에서만 정책을 비교한다. 기존 원문을 사후 정답 기반으로 수정하지 않는다.
* replay/fresh 각16개의 동일16-token workload를6기본변형에서 반복한다. baseline으로 준비한 동일 token prefix를 사용하고 모든 step에서 logits/argmax까지 실제 계산한다. gate는 미래 token 배열을 보지 않는다. fixed 출력은 품질 점수로 사용하지 않는다.
* 분해: tokenization/cache 준비, forward+next-token 동기화, gate(그 안에 앞선 queued 작업 대기가 섞일 수 있음), detokenize, 선택지 완료 검사, 나머지 host bookkeeping, CPU process time, 전체 wall, MLX allocator peak. forward 내부 dispatch와 wait는 별도 상세 진단에서만 계측한다. host dispatch는 Python+MLX/native enqueue이며 pure Python/GPU시간이 아니다. GPU elapsed/전력/금전 비용 미측정.
* 실제 블록/attention/MLP 미호출과 KV offset, native logits 및 legacy/lean token·mask 일치를 검사한다. 정답→오답/오답→정답·형식·출력 변화·토큰 수를 모두 기록한다. 클래스별/단일 next-token NLL만으로 품질 보존을 주장하지 않는다.
* 단계 로그는 완료 JSON을 요청별 새 파일에 저장하고 fsync한다. 재개 시 유효한 완료 행은 재실행하지 않으며 불완전한 파일도 보존한다. 패키지·모델·프로파일·실험 source hash가 달라지면 재개를 거부한다. 재개 session과 중단 시간을 기록해 연속 wall과 요청 시간 합계를 혼동하지 않는다.
* 효율 후보는 같은 조건에서 관측 내용/형식 점수가 하락하지 않고 paired wall ratio의95% 상한<1, 실제 생략 및 추정 matmul 감소를 만족해야 한다. 정확도 하락/시간 개선 불확실이면 성공으로 판정하지 않는다. 동일 점수라도 개별 정답 교환을 공개한다. 유한 합성셋의 점수 유지는 모집단 무손실 증명이 아니다.
* 평균/표준편차/범위 및 질문·반복 paired bootstrap CI. 원래64는 replay로, fresh64는 별도로 보고한다. 템플릿 상관·노트북 열/캐시·3반복의 한계를 명시한다. 기존 학습/평가 데이터를 재학습에 쓰지 않는다.
