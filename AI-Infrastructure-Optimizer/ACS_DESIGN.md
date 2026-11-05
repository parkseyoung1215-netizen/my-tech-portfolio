# Adaptive Computation Stop — 사전 설계

기존 파일을 보존하고 별도 ACS 모듈/실행 폴더를 추가한다. 모델은 기존 Qwen2.5-1.5B-Instruct BF16/MLX 고정 revision이며 모든28층을 실행한다. 유료 API/추가 모델 설치 없음.

* replay64는 이전 Stable Runtime 실험의 replay64, fresh64는 새32객관식(산술/논리/코딩/독해 각8)+16JSON+16문장이다. exact stem 과거중복0, 의미적 템플릿 유사성은 한계. 정책 학습/평가 임계값 조정 없음.
* 동일 prompt/greedy/cache/채점, 3회, 데이터셋과 방식 실행 순서 무작위화. 기존 기본 종료는 선택지 요청16/그 외48 상한+EOS. 고정 상한8이 주대조,16은 사전 고정 민감도 대조. 경계/확률/Adaptive도 기본16/48 안전 상한을 공유한다.
* Answer Boundary: 명시적 선택지 요청에서 순수 A-D 한 글자, 요청된 키가 존재하는 완전한 중복 키 없는 JSON 객체, 명시적 한 문장 요청에서 종결 문장부호를 감지한다. 그 외는 EOS/상한.
* Confidence: 같은 경계 + 현재 생성 토큰의 실제 softmax 확률>=.90. 확률은 경계 후보에서만 계산하고 비용을 포함한다. 보정된 정답 확률이 아니다.
* Adaptive: 선택지 .75 / JSON .60 / 문장 .95, 입력 token>128이면 .05 상향(최대.99). 문장은 최소6단어와 약어 Dr./Mr./Mrs./Ms./Prof./e.g./i.e. 보호를 추가한다. 정답/평가 성공 여부/미래 token은 사용하지 않는다. 임계값은 이 문서에서 고정한 가설이며 최적값으로 주장하지 않는다.
* 실제 selected-token probability와 EOS competitor mass를 logits에서 계산할 수 있다. EOS mass는 현재 토큰을 생성하기 전 분포이며 다음 시점 EOS 확률이라고 표현하지 않는다. 다음 분포 예측/추가 LLM confidence 호출은 사용하지 않는다.
* 측정: 내용/형식/불완전 종료, wall·process CPU·입출력 token·end-to-end TPS·decode TPS·실제 모델 요청/forward/블록 호출·MLX allocator peak. GPU elapsed/전력/금전 비용 미측정. dense matmul FLOPs는 추정; confidence용 logsumexp/reduction 추가 비용은 wall에 포함하되 matmul 추정식에는 미포함임을 명시한다.
* 모든 방법은 같은 Full 경로. 공통 baseline 출력과 prefix/mask 정합 검사. 확률/정지 비용 및 토큰별 이벤트 저장. 실제 native attention/MLP 카운터는 별도 소규모 감사에서 확인하고 primary 계측에 섞지 않는다.
* 내용 채점은 기존 고정 parser+JSON 정확한 타입 검사. 새로운 문장에는 요구 사실3개 보존 및 한 문장 제약을 둔다. 제약 통과는 완전한 의미/문체 평가가 아니다. baseline 정답→오답/오답→정답, cap·구조적 미완성·완성됐지만 오답을 구분한다.
* 난이도는 생성 시 지정한 작업 단계/스키마 복잡도와 기존 요청 특징 기반 coarse label이다. baseline 출력 길이와 reference token 길이에 따른 사후 그룹은 분석에만 사용한다.
* 같은 정답률 점추정만으로 무손실을 단정하지 않는다. 질문 cluster paired bootstrap CI와 보수적 유한표본 하한을 보고한다. 양셋에서 내용/형식 점추정 하락0, wall CI 상한<1, 실제 연산 경로 감소를 만족하면 관측 후보로만 표시한다. strict population 무손실은 별도 판정. Adaptive와 fixed8/fixed16/boundary/confidence 모두 비교하며 유리한 기준선만 선택하지 않는다.
* 요청별 append-only/fsync 체크포인트와 hash 고정 manifest. 중단 후 완료 행 재실행 없이 재개. 논문/독립 신기술 주장은 하지 않으며 기존 custom stopping criteria와 관계를 보고한다.
