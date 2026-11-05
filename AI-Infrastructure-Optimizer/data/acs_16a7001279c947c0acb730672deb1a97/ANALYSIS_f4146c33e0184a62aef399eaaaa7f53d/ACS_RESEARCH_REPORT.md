# Adaptive Computation Stop (ACS) Research Report



실행 `acs_16a7001279c947c0acb730672deb1a97`. 실제 무료 로컬 모델 실측. 제한적 사실 감사 후 양셋의 관측 후보: **Adaptive Stop**. 작은 표본·사후 채점 감사이며 일반적인 무손실/정답 완성 인식/새 기술 발견을 입증하지 않는다.

replay64: Adaptive의 제한적 요구사실 보존 감사값 59.38%, Baseline 대비 wall 16.65% 감소 [95% CI 10.85, 23.99]%, process CPU 17.77%, 모델 matmul 추정 2.44% 감소. 의미적 완전 정답이나 전체 GPU 연산 감소를 입증한 값은 아니다.

fresh64: Adaptive의 제한적 요구사실 보존 감사값 82.81%, Baseline 대비 wall 8.22% 감소 [95% CI 4.56, 12.77]%, process CPU 7.43%, 모델 matmul 추정 1.87% 감소. 의미적 완전 정답이나 전체 GPU 연산 감소를 입증한 값은 아니다.



## 0. 최종 판단에 중요한 채점 감사

사전 문자열 채점은 Baseline의 doctor→physician 패러프레이즈8개를 오답으로 처리했다. 그 결과 Boundary/Confidence가 Dr.만 출력해 사실을 잃어도 Wrong→Wrong으로 분류하여 손실을 가렸다. 원래 점수·통계·모델 답변·시간은 그대로 두고 별도 CONTENT_AUDIT 파일을 추가했다. 동의어 인정은 사후의 제한적 평가 보정이며 정책·임계값·모델 학습에는 사용하지 않았다.

요구 사실 보존 감사는 doctor/physician을 동의 역할로 인정한다. 완전한 의미 평가가 아니며 새로운 점수를 무조건 진짜 정확도라고 부르지 않는다. Baseline과 Adaptive는 dedicated/diligently, 일부 환자·지역사회 관련 미제공 사실도 추가했다. 이 표현들은 별도 충실도 문제로 남아 있다. 따라서 요구 사실을 보존했다는 것이 추가 사실 없는 완전한 정답을 의미하지 않는다.

| 셋 | 방식 | 원문자열 내용점수 | 동의어 사실보존 감사 | 형식 준수율 | 식별된 추가표현율 | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| replay64 | Baseline | 59.38% | 59.38% | 32.81% | 0.00% | 24.888 |
| replay64 | Fixed Cap 8 | 26.56% | 26.56% | 20.31% | 0.00% | 16.342 |
| replay64 | Fixed Cap 16 | 34.38% | 34.38% | 20.31% | 0.00% | 21.336 |
| replay64 | Answer Boundary | 59.38% | 59.38% | 87.50% | 0.00% | 20.009 |
| replay64 | Confidence-Aware | 59.38% | 59.38% | 65.62% | 0.00% | 22.433 |
| replay64 | Adaptive Stop | 59.38% | 59.38% | 76.56% | 0.00% | 20.743 |
| fresh64 | Baseline | 70.31% | 82.81% | 48.44% | 12.50% | 39.687 |
| fresh64 | Fixed Cap 8 | 28.12% | 28.12% | 23.44% | 0.00% | 18.315 |
| fresh64 | Fixed Cap 16 | 32.81% | 32.81% | 23.44% | 12.50% | 27.431 |
| fresh64 | Answer Boundary | 70.31% | 70.31% | 62.50% | 0.00% | 28.124 |
| fresh64 | Confidence-Aware | 70.31% | 70.31% | 51.56% | 0.00% | 29.437 |
| fresh64 | Adaptive Stop | 70.31% | 82.81% | 68.75% | 12.50% | 36.424 |


| 셋 | 대상/대조 | 감사 사실보존 차이 pp [CI] | wall 감소율 % [CI] |
| --- | --- | --- | --- |
| replay64 | Fixed Cap 8/Baseline | -32.81 [-45.31,-21.88] | +34.34 [+24.77,+42.28] |
| replay64 | Fixed Cap 16/Baseline | -25.00 [-35.94,-14.06] | +14.27 [+8.52,+18.97] |
| replay64 | Answer Boundary/Baseline | +0.00 [+0.00,+0.00] | +19.60 [+13.40,+27.54] |
| replay64 | Confidence-Aware/Baseline | +0.00 [+0.00,+0.00] | +9.86 [+6.14,+14.38] |
| replay64 | Adaptive Stop/Baseline | +0.00 [+0.00,+0.00] | +16.65 [+10.85,+23.99] |
| replay64 | Adaptive Stop/Fixed Cap 8 | +32.81 [+21.88,+45.31] | -26.93 [-51.69,-2.72] |
| replay64 | Adaptive Stop/Fixed Cap 16 | +25.00 [+14.06,+35.94] | +2.78 [-8.55,+15.56] |
| replay64 | Adaptive Stop/Answer Boundary | +0.00 [+0.00,+0.00] | -3.67 [-7.70,-0.65] |
| replay64 | Adaptive Stop/Confidence-Aware | +0.00 [+0.00,+0.00] | +7.53 [+1.83,+14.97] |
| fresh64 | Fixed Cap 8/Baseline | -54.69 [-67.19,-42.19] | +53.85 [+45.98,+59.70] |
| fresh64 | Fixed Cap 16/Baseline | -50.00 [-62.50,-37.50] | +30.88 [+24.27,+36.18] |
| fresh64 | Answer Boundary/Baseline | -12.50 [-20.31,-4.69] | +29.14 [+18.63,+41.89] |
| fresh64 | Confidence-Aware/Baseline | -12.50 [-20.31,-4.69] | +25.83 [+14.86,+38.70] |
| fresh64 | Adaptive Stop/Baseline | +0.00 [+0.00,+0.00] | +8.22 [+4.56,+12.77] |
| fresh64 | Adaptive Stop/Fixed Cap 8 | +54.69 [+42.19,+67.19] | -98.87 [-134.10,-63.92] |
| fresh64 | Adaptive Stop/Fixed Cap 16 | +50.00 [+37.50,+62.50] | -32.78 [-47.80,-16.72] |
| fresh64 | Adaptive Stop/Answer Boundary | +12.50 [+4.69,+20.31] | -29.51 [-57.44,-12.56] |
| fresh64 | Adaptive Stop/Confidence-Aware | +12.50 [+4.69,+20.31] | -23.74 [-49.76,-7.68] |


| 셋 | 방식 | 감사 품질 점추정 조건 | 제한적 관측 후보 | 모집단 무손실 입증 |
| --- | --- | --- | --- | --- |
| replay64 | Fixed Cap 8 | False | False | False |
| replay64 | Fixed Cap 16 | False | False | False |
| replay64 | Answer Boundary | True | True | False |
| replay64 | Confidence-Aware | True | True | False |
| replay64 | Adaptive Stop | True | True | False |
| fresh64 | Fixed Cap 8 | False | False | False |
| fresh64 | Fixed Cap 16 | False | False | False |
| fresh64 | Answer Boundary | False | False | False |
| fresh64 | Confidence-Aware | False | False | False |
| fresh64 | Adaptive Stop | True | True | False |


Boundary/Confidence는 신규셋에서 Baseline이 담았던 요구 사실8개를 잃어 실패다. Adaptive는 이 제한적 감사에서 같은 요구 사실 보존을 보였지만, 명시적으로 설계한8개 약어 스트레스 사례의 보호 규칙이 작동한 결과다. 무작위 실서비스의 의미적 완료 판단으로 일반화하지 않는다. 아래 원래 지표들도 감사 전후 비교를 위해 모두 제공한다.



| 문제 ID | 방식 | 원문 응답 | 감사 사실보존 | 식별된 추가표현 |
| --- | --- | --- | --- | --- |
| 369cbf9f2f09 | Baseline | Dr. Leno53112, a dedicated physician, diligently works at Studio 43. | True | ['dedicated', 'diligently'] |
| 3ce908804e9c | Baseline | Dr. Leno53109, a dedicated physician, diligently works at Studio 40, where they provide exceptional medical care to patients in the community. | True | ['dedicated', 'diligently', 'exceptional', 'patients', 'community'] |
| 369cbf9f2f09 | Answer Boundary | Dr. | False | [] |
| 3ce908804e9c | Answer Boundary | Dr. | False | [] |
| 369cbf9f2f09 | Confidence-Aware | Dr. | False | [] |
| 3ce908804e9c | Confidence-Aware | Dr. | False | [] |
| 369cbf9f2f09 | Adaptive Stop | Dr. Leno53112, a dedicated physician, diligently works at Studio 43. | True | ['dedicated', 'diligently'] |
| 3ce908804e9c | Adaptive Stop | Dr. Leno53109, a dedicated physician, diligently works at Studio 40, where they provide exceptional medical care to patients in the community. | True | ['dedicated', 'diligently', 'exceptional', 'patients', 'community'] |


## 1. 설계·모델·데이터

Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Apache-2.0, BF16, Apple M4/16GiB, MLX0.32.4/MLX-LM0.32.2. 기존 로컬 HF 체크포인트를 사용했으며 유료 API/모델 설치는 없다. Full28층을 모든 방식에서 실행하고 종료 시점만 다르게 한다. 가중치·sampling 분포·프롬프트·KV 조건은 공통이다.

replay64는 앞선 Stable Runtime의 기존64문제다. fresh64는 새 객관식32(산술/논리/코딩/독해 각8), JSON16, 한 문장16이다. 이전75% MCQ 편중을 줄였지만 합성 평가이며 실제 서비스 분포는 아니다. fresh 문장 중8개는 Dr. 약어가 있는 의도적 경계 스트레스 사례다. Adaptive의 약어 보호 가설을 검증하는 구성이라 무작위 외부 검증으로 과장하지 않는다.

정책과 임계값은 ACS_DESIGN.md 및 runtime source hash로 평가 전에 고정했다. 학습/평가 임계값 조정은 없고 정답은 채점에만 사용했다. 모델 생성 함수는 question·method만 받으며 reference나 expected값은 받지 않는다. 요청 JSON 키는 입력에 명시된 형식 계약으로만 사용한다.

128문제와 방식 순서를 섞어 각3회 실행했다. 같은 질문의3회 출력을192개의 독립 표본으로 보지 않는다. 새 데이터 exact stem 중복0이나 의미적/템플릿 유사성을 완전히 배제하지 않았다. 기존64의 난이도는 미지정, fresh 난이도는 생성 단계/스키마에 따른 사전 coarse label이며 정답률로 사후 지정하지 않았다.



## 2. 사전 고정한 방식

| 방식 | 규칙 |
| --- | --- |
| Baseline | 기존 기본 EOS, 명시적 선택지 요청16/나머지48 안전 상한 |
| Fixed Cap8 /16 | 기본 상한 내 최대8 또는16; 양쪽을 모두 보고하며 유리한 상한만 고르지 않음 |
| Answer Boundary | 순수 선택지 A-D, 요청 키를 갖춘 완전 JSON, 종결 문장부호 |
| Confidence-Aware | 같은 경계 + 현재 선택 token의 실제 softmax 확률>=.90 |
| Adaptive Stop | 선택지 .75 / JSON .60 / 문장 .95; 긴 입력>128token이면 .05 상향(.99상한); 문장 최소6단어와 흔한 약어 보호 |


지원되는 신호: 현재 실제 logits의 selected-token probability와 EOS competitor mass. EOS mass는 현재 token 이전 분포이며 다음 시점 EOS 확률이 아니다. 정답 확률·미래 완성 여부·외부 판정 모델의 confidence는 구현하지 않았다. 선택 token 확률이 높아도 정답/완성을 보장하지 않는다. 확률은 경계 후보에서만 계산하고 그 logsumexp·readback 비용을 wall/CPU에 포함했다.

경계 parser와 유형 인식에 실패하면 EOS/안전 상한으로 계속한다. 자유서술 전체의 의미적 완료를 판정하는 시스템은 아니다. JSON은 따옴표 안 괄호/중복키를 처리하며 fenced JSON은 성급하게 중단하지 않는다.



## 3. 원문자열 자동지표 비교: 3회 평균·표준편차·범위 (최종 사실 감사는 위0절)

### replay64

| 방식 | 내용 정확도 | 형식 준수율 | 전체 과제 성공률 | 미완성 비율 | wall s 평균±SD [min,max] | CPU s 평균±SD | 입력/출력 token/회 | TPS | 모델호출/회 | MLX peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 59.38% | 32.81% | 18.75% | 12.50% | 24.888±0.814 [24.094,25.721] | 6.377±1.870 | 5367/546 | 21.95 | 64 | 2.984 |
| Fixed Cap 8 | 26.56% | 20.31% | 6.25% | 25.00% | 16.342±1.281 [15.124,17.678] | 3.806±0.942 | 5367/297 | 18.25 | 64 | 2.984 |
| Fixed Cap 16 | 34.38% | 20.31% | 6.25% | 25.00% | 21.336±0.946 [20.245,21.934] | 5.341±1.549 | 5367/438 | 20.56 | 64 | 2.984 |
| Answer Boundary | 59.38% | 87.50% | 46.88% | 12.50% | 20.009±0.960 [19.018,20.935] | 4.985±1.610 | 5367/404 | 20.22 | 64 | 2.984 |
| Confidence-Aware | 59.38% | 65.62% | 32.81% | 12.50% | 22.433±0.563 [21.896,23.020] | 5.622±1.741 | 5367/473 | 21.09 | 64 | 2.984 |
| Adaptive Stop | 59.38% | 76.56% | 40.62% | 12.50% | 20.743±0.847 [19.858,21.547] | 5.244±1.588 | 5367/423 | 20.42 | 64 | 2.984 |


| 방식 | 실제 forward/회 | 실제 블록 호출/회 | 모델 matmul 추정 TFLOPs/회 | 종료판단 s/회 | 확률판단 s/회(중첩) | 실제 확률query/회 | decode TPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 546 | 15288 | 15.671 | 0.000 | 0.000 | 0.0 | 28.64 |
| Fixed Cap 8 | 297 | 8316 | 14.900 | 0.000 | 0.000 | 0.0 | 28.51 |
| Fixed Cap 16 | 438 | 12264 | 15.336 | 0.000 | 0.000 | 0.0 | 28.39 |
| Answer Boundary | 404 | 11312 | 15.230 | 0.035 | 0.000 | 0.0 | 28.64 |
| Confidence-Aware | 473 | 13244 | 15.445 | 0.194 | 0.155 | 56.0 | 28.87 |
| Adaptive Stop | 423 | 11844 | 15.289 | 0.075 | 0.037 | 56.0 | 28.69 |


### fresh64

| 방식 | 내용 정확도 | 형식 준수율 | 전체 과제 성공률 | 미완성 비율 | wall s 평균±SD [min,max] | CPU s 평균±SD | 입력/출력 token/회 | TPS | 모델호출/회 | MLX peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 70.31% | 48.44% | 29.69% | 37.50% | 39.687±1.239 [38.758,41.094] | 9.931±1.609 | 4860/1000 | 25.21 | 64 | 2.980 |
| Fixed Cap 8 | 28.12% | 23.44% | 17.19% | 50.00% | 18.315±1.376 [17.288,19.878] | 4.214±0.890 | 4860/381 | 20.88 | 64 | 2.980 |
| Fixed Cap 16 | 32.81% | 23.44% | 17.19% | 50.00% | 27.431±0.924 [26.561,28.402] | 6.636±1.276 | 4860/644 | 23.49 | 64 | 2.980 |
| Answer Boundary | 70.31% | 62.50% | 45.31% | 37.50% | 28.124±1.159 [27.322,29.453] | 6.460±0.934 | 4860/677 | 24.10 | 64 | 2.980 |
| Confidence-Aware | 70.31% | 51.56% | 37.50% | 37.50% | 29.437±1.324 [28.563,30.960] | 7.017±1.364 | 4860/707 | 24.05 | 64 | 2.980 |
| Adaptive Stop | 70.31% | 68.75% | 40.62% | 37.50% | 36.424±1.430 [35.366,38.052] | 9.193±1.449 | 4860/905 | 24.87 | 64 | 2.980 |


| 방식 | 실제 forward/회 | 실제 블록 호출/회 | 모델 matmul 추정 TFLOPs/회 | 종료판단 s/회 | 확률판단 s/회(중첩) | 실제 확률query/회 | decode TPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1000 | 28000 | 15.736 | 0.000 | 0.000 | 0.0 | 28.74 |
| Fixed Cap 8 | 381 | 10668 | 13.817 | 0.000 | 0.000 | 0.0 | 28.82 |
| Fixed Cap 16 | 644 | 18032 | 14.632 | 0.000 | 0.000 | 0.0 | 28.70 |
| Answer Boundary | 677 | 18956 | 14.734 | 0.054 | 0.000 | 0.0 | 29.25 |
| Confidence-Aware | 707 | 19796 | 14.827 | 0.086 | 0.026 | 48.0 | 28.91 |
| Adaptive Stop | 905 | 25340 | 15.441 | 0.103 | 0.027 | 48.0 | 28.81 |


입출력·wall·CPU는64요청 한 회의 합계를3회 반복한 값이다. TPS는 출력 token 합계/전체 wall이며 prefill과 판단을 포함한다. decode TPS는 첫 step 이후 model-step 시간만 사용하므로 end-to-end TPS와 다르다. EOS도 생성 token에 포함한다. 더 짧은 답이 TPS가 낮아도 실제 latency는 줄 수 있어 TPS만으로 성공을 판정하지 않는다.

MLX peak는 요청별 allocator peak의 평균이고 동일 프로세스의 공통 상주 모델을 포함한다. process lifetime RSS를 방식별 peak로 사용하지 않았다. GPU elapsed/전력/금전 비용은 **미측정**이다. 실제 forward·블록 진입 수는 로그/네이티브 감사로 확인했으나 GPU hardware FLOPs는 측정하지 않았다.

모델 matmul FLOPs는 actual query/context/층 수에 기반한 dense MAC×2 추정이다. confidence의 softmax logsumexp·reduction 비용은 wall에는 들어가지만 matmul 추정에는 미포함이다. probability_elements_scanned에 실제 처리 어휘 원소 수를 따로 저장했으며 이를 전체 GPU FLOPs로 바꾸지 않는다.



## 4. 원문자열 지표 차이·실제 절감률·신뢰구간 (사실 손실은0절 감사 기준)

| 셋 | 대상/대조 | 내용 차이 pp [95% CI] | wall 감소율 % [CI] | CPU 감소율 % [CI] | 모델 matmul 추정 감소율 % |
| --- | --- | --- | --- | --- | --- |
| replay64 | Fixed Cap 8/Baseline | -32.81 [-45.31,-21.88] | +34.34 [+24.77,+42.28] | +40.32 [+29.83,+47.69] | 4.92 |
| replay64 | Fixed Cap 16/Baseline | -25.00 [-35.94,-14.06] | +14.27 [+8.52,+18.97] | +16.26 [+10.24,+21.60] | 2.14 |
| replay64 | Answer Boundary/Baseline | +0.00 [+0.00,+0.00] | +19.60 [+13.40,+27.54] | +21.83 [+13.31,+33.67] | 2.81 |
| replay64 | Confidence-Aware/Baseline | +0.00 [+0.00,+0.00] | +9.86 [+6.14,+14.38] | +11.85 [+6.98,+18.31] | 1.45 |
| replay64 | Adaptive Stop/Baseline | +0.00 [+0.00,+0.00] | +16.65 [+10.85,+23.99] | +17.77 [+9.18,+28.48] | 2.44 |
| replay64 | Adaptive Stop/Fixed Cap 8 | +32.81 [+21.88,+45.31] | -26.93 [-51.69,-2.72] | -37.78 [-68.08,-4.79] | -2.62 |
| replay64 | Adaptive Stop/Fixed Cap 16 | +25.00 [+14.06,+35.94] | +2.78 [-8.55,+15.56] | +1.81 [-12.19,+18.75] | 0.31 |
| replay64 | Adaptive Stop/Answer Boundary | +0.00 [+0.00,+0.00] | -3.67 [-7.70,-0.65] | -5.19 [-15.07,+0.08] | -0.39 |
| replay64 | Adaptive Stop/Confidence-Aware | +0.00 [+0.00,+0.00] | +7.53 [+1.83,+14.97] | +6.72 [-1.98,+16.38] | 1.01 |
| fresh64 | Fixed Cap 8/Baseline | -42.19 [-54.69,-31.25] | +53.85 [+45.98,+59.70] | +57.57 [+50.63,+62.99] | 12.19 |
| fresh64 | Fixed Cap 16/Baseline | -37.50 [-50.00,-26.56] | +30.88 [+24.27,+36.18] | +33.18 [+25.82,+39.44] | 7.01 |
| fresh64 | Answer Boundary/Baseline | +0.00 [+0.00,+0.00] | +29.14 [+18.63,+41.89] | +34.95 [+21.06,+49.57] | 6.37 |
| fresh64 | Confidence-Aware/Baseline | +0.00 [+0.00,+0.00] | +25.83 [+14.86,+38.70] | +29.35 [+16.24,+44.50] | 5.77 |
| fresh64 | Adaptive Stop/Baseline | +0.00 [+0.00,+0.00] | +8.22 [+4.56,+12.77] | +7.43 [+1.73,+13.58] | 1.87 |
| fresh64 | Adaptive Stop/Fixed Cap 8 | +42.19 [+31.25,+54.69] | -98.87 [-134.10,-63.92] | -118.18 [-158.10,-79.59] | -11.75 |
| fresh64 | Adaptive Stop/Fixed Cap 16 | +37.50 [+26.56,+50.00] | -32.78 [-47.80,-16.72] | -38.53 [-56.62,-20.56] | -5.53 |
| fresh64 | Adaptive Stop/Answer Boundary | +0.00 [+0.00,+0.00] | -29.51 [-57.44,-12.56] | -42.31 [-82.97,-17.51] | -4.80 |
| fresh64 | Adaptive Stop/Confidence-Aware | +0.00 [+0.00,+0.00] | -23.74 [-49.76,-7.68] | -31.02 [-67.27,-9.42] | -4.14 |


CI는 질문 cluster와 반복 축의 paired bootstrap이다. 각64문제에서1문제는1.5625%p다. 차이0의 bootstrap CI가0이어도 모집단의 엄격한 무손실은 증명할 수 없다. 보수적인 유한 표본 하한도 COMPARISONS.csv에 저장했다. 기준을 만족해도 관측 후보로만 표시하고 일반적인 정확도 유지를 단정하지 않는다.

### Adaptive의 고정 상한 대비 추가 가치 (동의어 사실 감사 기준)

| 셋 | 대조 | 사실보존 차이 pp | 시간 비율 Adaptive/대조 [CI] | 해석 |
| --- | --- | --- | --- | --- |
| replay64 | Fixed Cap 8 | 32.81 | 1.269 [1.027,1.517] | 품질·시간 동시 우위 미확인/교환관계 |
| replay64 | Fixed Cap 16 | 25.00 | 0.972 [0.844,1.085] | 품질·시간 동시 우위 미확인/교환관계 |
| fresh64 | Fixed Cap 8 | 54.69 | 1.989 [1.639,2.341] | 품질·시간 동시 우위 미확인/교환관계 |
| fresh64 | Fixed Cap 16 | 50.00 | 1.328 [1.167,1.478] | 품질·시간 동시 우위 미확인/교환관계 |


고정 cap보다 정확하지만 느린 결과는 품질-비용 trade-off이며 엄밀한 동시 우위가 아니다. cap8/16 두 개만으로 모든 최적 고정 상한을 이겼다고 주장하지 않는다. PARETO.csv는 원문자열 지표, AUDITED_PARETO.csv와 그림은 제한적 사실 감사의 비지배 점추정이며 통계적 Pareto 우월성 증명은 아니다.



## 5. 문제별 실패·불완전 종료

| 셋 | 방식 | 기준 정답→오답/회 | 기준 오답→정답/회 | 중단했으나 내용오답/회 | 미완성/회 |
| --- | --- | --- | --- | --- | --- |
| replay64 | Baseline | 0.0 | 0.0 | 0.0 | 8.0 |
| replay64 | Fixed Cap 8 | 21.0 | 0.0 | 0.0 | 16.0 |
| replay64 | Fixed Cap 16 | 16.0 | 0.0 | 0.0 | 16.0 |
| replay64 | Answer Boundary | 0.0 | 0.0 | 26.0 | 8.0 |
| replay64 | Confidence-Aware | 0.0 | 0.0 | 20.0 | 8.0 |
| replay64 | Adaptive Stop | 0.0 | 0.0 | 23.0 | 8.0 |
| fresh64 | Baseline | 0.0 | 0.0 | 0.0 | 24.0 |
| fresh64 | Fixed Cap 8 | 27.0 | 0.0 | 0.0 | 32.0 |
| fresh64 | Fixed Cap 16 | 24.0 | 0.0 | 0.0 | 32.0 |
| fresh64 | Answer Boundary | 0.0 | 0.0 | 19.0 | 24.0 |
| fresh64 | Confidence-Aware | 0.0 | 0.0 | 16.0 | 24.0 |
| fresh64 | Adaptive Stop | 0.0 | 0.0 | 14.0 | 24.0 |


| 셋 | 방식 | 전환 | 종료 | 질문 | 기준 응답 | 중단 응답 | 오류 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| replay64 | Baseline | Correct -> Correct | eos | Encode this as JSON with exactly label and enabled: label is unit30200; enabled is false. Return only JSON. | ```json {   "label": "unit30200",   "enabled": false } ``` | ```json {   "label": "unit30200",   "enabled": false } ``` | format_only |
| replay64 | Baseline | Correct -> Correct | eos | Encode this as JSON with exactly label and enabled: label is unit30202; enabled is false. Return only JSON. | ```json {   "label": "unit30202",   "enabled": false } ``` | ```json {   "label": "unit30202",   "enabled": false } ``` | format_only |
| replay64 | Fixed Cap 8 | Correct -> Wrong | max_tokens | Combine into one concise sentence without adding facts: "Vera30204 is a courier. Vera30204 is waiting at Gate 15." | Vera30204, a courier, is currently waiting at Gate 15. | Vera30204, | incomplete_or_missing_facts |
| replay64 | Fixed Cap 8 | Correct -> Wrong | max_tokens | Encode this as JSON with exactly label and enabled: label is unit30200; enabled is false. Return only JSON. | ```json {   "label": "unit30200",   "enabled": false } ``` | ```json {   "label": | incomplete_or_missing_facts |
| replay64 | Fixed Cap 16 | Correct -> Wrong | max_tokens | Combine into one concise sentence without adding facts: "Vera30204 is a courier. Vera30204 is waiting at Gate 15." | Vera30204, a courier, is currently waiting at Gate 15. | Vera30204, a courier, is currently waiting at Gate | incomplete_or_missing_facts |
| replay64 | Fixed Cap 16 | Correct -> Wrong | max_tokens | Encode this as JSON with exactly label and enabled: label is unit30200; enabled is false. Return only JSON. | ```json {   "label": "unit30200",   "enabled": false } ``` | ```json {   "label": "unit30200",  | incomplete_or_missing_facts |
| replay64 | Answer Boundary | Correct -> Correct | eos | Encode this as JSON with exactly label and enabled: label is unit30200; enabled is false. Return only JSON. | ```json {   "label": "unit30200",   "enabled": false } ``` | ```json {   "label": "unit30200",   "enabled": false } ``` | format_only |
| replay64 | Answer Boundary | Correct -> Correct | eos | Encode this as JSON with exactly label and enabled: label is unit30202; enabled is false. Return only JSON. | ```json {   "label": "unit30202",   "enabled": false } ``` | ```json {   "label": "unit30202",   "enabled": false } ``` | format_only |
| replay64 | Confidence-Aware | Correct -> Correct | eos | Encode this as JSON with exactly label and enabled: label is unit30200; enabled is false. Return only JSON. | ```json {   "label": "unit30200",   "enabled": false } ``` | ```json {   "label": "unit30200",   "enabled": false } ``` | format_only |
| replay64 | Confidence-Aware | Correct -> Correct | eos | Encode this as JSON with exactly label and enabled: label is unit30202; enabled is false. Return only JSON. | ```json {   "label": "unit30202",   "enabled": false } ``` | ```json {   "label": "unit30202",   "enabled": false } ``` | format_only |
| replay64 | Adaptive Stop | Correct -> Correct | eos | Encode this as JSON with exactly label and enabled: label is unit30200; enabled is false. Return only JSON. | ```json {   "label": "unit30200",   "enabled": false } ``` | ```json {   "label": "unit30200",   "enabled": false } ``` | format_only |
| replay64 | Adaptive Stop | Correct -> Correct | eos | Encode this as JSON with exactly label and enabled: label is unit30202; enabled is false. Return only JSON. | ```json {   "label": "unit30202",   "enabled": false } ``` | ```json {   "label": "unit30202",   "enabled": false } ``` | format_only |
| fresh64 | Baseline | Correct -> Correct | eos | Return only a JSON object with keys code, quantity and status. Values: code="acs52109", quantity=12, status="ready". | ```json {   "code": "acs52109",   "quantity": 12,   "status": "ready" } ``` | ```json {   "code": "acs52109",   "quantity": 12,   "status": "ready" } ``` | format_only |
| fresh64 | Baseline | Correct -> Correct | eos | Return only a JSON object with keys code and quantity. Values: code="acs52106", quantity=9. | ```json {   "code": "acs52106",   "quantity": 9 } ``` | ```json {   "code": "acs52106",   "quantity": 9 } ``` | format_only |
| fresh64 | Fixed Cap 8 | Correct -> Wrong | max_tokens | Use exactly one sentence to preserve all these facts without adding facts: "Leno53107 is a artist. Leno53107 works at Studio 38." | Leno53107 is an artist who works at Studio 38. | Leno53107 is | incomplete_or_missing_facts |
| fresh64 | Fixed Cap 8 | Correct -> Wrong | max_tokens | Return only a JSON object with keys code, quantity and status. Values: code="acs52109", quantity=12, status="ready". | ```json {   "code": "acs52109",   "quantity": 12,   "status": "ready" } ``` | ```json {   "code": | incomplete_or_missing_facts |
| fresh64 | Fixed Cap 16 | Correct -> Wrong | max_tokens | Use exactly one sentence to preserve all these facts without adding facts: "Leno53107 is a artist. Leno53107 works at Studio 38." | Leno53107 is an artist who works at Studio 38. | Leno53107 is an artist who works at Studio 3 | incomplete_or_missing_facts |
| fresh64 | Fixed Cap 16 | Correct -> Wrong | max_tokens | Return only a JSON object with keys code, quantity and status. Values: code="acs52109", quantity=12, status="ready". | ```json {   "code": "acs52109",   "quantity": 12,   "status": "ready" } ``` | ```json {   "code": "acs52109",  | incomplete_or_missing_facts |
| fresh64 | Answer Boundary | Correct -> Correct | eos | Return only a JSON object with keys code, quantity and status. Values: code="acs52109", quantity=12, status="ready". | ```json {   "code": "acs52109",   "quantity": 12,   "status": "ready" } ``` | ```json {   "code": "acs52109",   "quantity": 12,   "status": "ready" } ``` | format_only |
| fresh64 | Answer Boundary | Correct -> Correct | eos | Return only a JSON object with keys code and quantity. Values: code="acs52106", quantity=9. | ```json {   "code": "acs52106",   "quantity": 9 } ``` | ```json {   "code": "acs52106",   "quantity": 9 } ``` | format_only |
| fresh64 | Confidence-Aware | Correct -> Correct | eos | Return only a JSON object with keys code, quantity and status. Values: code="acs52109", quantity=12, status="ready". | ```json {   "code": "acs52109",   "quantity": 12,   "status": "ready" } ``` | ```json {   "code": "acs52109",   "quantity": 12,   "status": "ready" } ``` | format_only |
| fresh64 | Confidence-Aware | Correct -> Correct | eos | Return only a JSON object with keys code and quantity. Values: code="acs52106", quantity=9. | ```json {   "code": "acs52106",   "quantity": 9 } ``` | ```json {   "code": "acs52106",   "quantity": 9 } ``` | format_only |
| fresh64 | Adaptive Stop | Correct -> Correct | eos | Return only a JSON object with keys code, quantity and status. Values: code="acs52109", quantity=12, status="ready". | ```json {   "code": "acs52109",   "quantity": 12,   "status": "ready" } ``` | ```json {   "code": "acs52109",   "quantity": 12,   "status": "ready" } ``` | format_only |
| fresh64 | Adaptive Stop | Correct -> Correct | eos | Return only a JSON object with keys code and quantity. Values: code="acs52106", quantity=9. | ```json {   "code": "acs52106",   "quantity": 9 } ``` | ```json {   "code": "acs52106",   "quantity": 9 } ``` | format_only |


예시는 셋·방식별 최대2개이며 Baseline 정답 파괴와 미완성을 우선해 기계적으로 고른다. 유리한 사례만 고르지 않으며 전체 실패 원문은 별도 CSV에 보존한다.

CASES.csv는 모든 문제/3반복/방식의 원문·정답·시간·완성 상태를 보존한다. FAILURES.csv는 실패 전부다. 유효한 선택지/JSON이라도 내용 오답일 수 있고, 경계 도달과 정답 완성은 다르다. 상한으로 불완전한 JSON·누락 사실이 생기면 실패다. 문장 내용은 명시된 사실 문자열 보존 surrogate이며 완전한 의미 동등성·문체 평가는 아니다.



## 6. 실제 토큰 확률 신호와 한계

| 셋 | 방식 | token 확률 bin | 고유 문제 | 반복 관측 | 중단 답변 사실보존 감사 | 중단 미완성 감사 |
| --- | --- | --- | --- | --- | --- | --- |
| replay64 | Confidence-Aware | 0.9-0.99 | 12 | 36 | 41.67% | 0.00% |
| replay64 | Confidence-Aware | 0.99-1.000001 | 29 | 87 | 55.17% | 0.00% |
| replay64 | Adaptive Stop | 0.75-0.9 | 8 | 24 | 62.50% | 0.00% |
| replay64 | Adaptive Stop | 0.9-0.99 | 12 | 36 | 41.67% | 0.00% |
| replay64 | Adaptive Stop | 0.99-1.000001 | 29 | 87 | 55.17% | 0.00% |
| fresh64 | Confidence-Aware | 0.9-0.99 | 16 | 48 | 62.50% | 0.00% |
| fresh64 | Confidence-Aware | 0.99-1.000001 | 24 | 72 | 58.33% | 33.33% |
| fresh64 | Adaptive Stop | 0.75-0.9 | 4 | 12 | 50.00% | 0.00% |
| fresh64 | Adaptive Stop | 0.9-0.99 | 12 | 36 | 50.00% | 0.00% |
| fresh64 | Adaptive Stop | 0.99-1.000001 | 19 | 57 | 89.47% | 0.00% |


표는 실제로 중단한 경계만 포함한다. 거절된 경계의 prefix 내용은 따로 채점하지 않았으므로 그 확률을 최종 답변 라벨에 억지로 매핑하지 않았다. 같은 token 확률은 next-token likelihood이며 답변 정답/완성 확률로 보정된 것이 아니다. PROBABILITY_EVENTS.csv에 현재 위치·token확률·EOS competitor mass·임계값·종료 여부를 전부 보존한다.



## 7. 길이·난이도·유형 (제한적 사실 감사)

| 셋 | 방식 | 그룹 기준 | 그룹 | 고유문제 | 사실보존 감사 | 형식 준수율 | 미완성 감사 | wall/회 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay64 | Baseline | category | arithmetic | 12 | 25.00% | 0.00% | 0.00% | 2.720 |
| replay64 | Baseline | category | coding | 12 | 41.67% | 0.00% | 0.00% | 2.561 |
| replay64 | Baseline | category | logic | 12 | 25.00% | 91.67% | 0.00% | 2.721 |
| replay64 | Baseline | category | reading | 12 | 91.67% | 16.67% | 0.00% | 3.582 |
| replay64 | Baseline | category | structured_output | 8 | 100.00% | 0.00% | 100.00% | 7.189 |
| replay64 | Baseline | category | text_quality | 8 | 100.00% | 100.00% | 0.00% | 6.115 |
| replay64 | Baseline | difficulty | unrated | 64 | 59.38% | 32.81% | 12.50% | 24.888 |
| replay64 | Baseline | baseline_length_bin | 1-4 | 40 | 42.50% | 32.50% | 0.00% | 8.326 |
| replay64 | Baseline | baseline_length_bin | 17-32 | 16 | 100.00% | 50.00% | 50.00% | 13.303 |
| replay64 | Baseline | baseline_length_bin | 5-16 | 8 | 62.50% | 0.00% | 0.00% | 3.259 |
| replay64 | Baseline | reference_length_bin | 1-4 | 48 | 45.83% | 27.08% | 0.00% | 11.584 |
| replay64 | Baseline | reference_length_bin | 17-32 | 8 | 100.00% | 100.00% | 0.00% | 6.115 |
| replay64 | Baseline | reference_length_bin | 5-16 | 8 | 100.00% | 0.00% | 100.00% | 7.189 |
| replay64 | Fixed Cap 8 | category | arithmetic | 12 | 25.00% | 0.00% | 0.00% | 2.745 |
| replay64 | Fixed Cap 8 | category | coding | 12 | 41.67% | 0.00% | 0.00% | 2.518 |
| replay64 | Fixed Cap 8 | category | logic | 12 | 16.67% | 91.67% | 0.00% | 2.587 |
| replay64 | Fixed Cap 8 | category | reading | 12 | 58.33% | 16.67% | 0.00% | 3.353 |
| replay64 | Fixed Cap 8 | category | structured_output | 8 | 0.00% | 0.00% | 100.00% | 2.535 |
| replay64 | Fixed Cap 8 | category | text_quality | 8 | 0.00% | 0.00% | 100.00% | 2.604 |
| replay64 | Fixed Cap 8 | difficulty | unrated | 64 | 26.56% | 20.31% | 25.00% | 16.342 |
| replay64 | Fixed Cap 8 | baseline_length_bin | 1-4 | 40 | 42.50% | 32.50% | 0.00% | 8.396 |
| replay64 | Fixed Cap 8 | baseline_length_bin | 17-32 | 16 | 0.00% | 0.00% | 100.00% | 5.140 |
| replay64 | Fixed Cap 8 | baseline_length_bin | 5-16 | 8 | 0.00% | 0.00% | 0.00% | 2.807 |
| replay64 | Fixed Cap 8 | reference_length_bin | 1-4 | 48 | 35.42% | 27.08% | 0.00% | 11.203 |
| replay64 | Fixed Cap 8 | reference_length_bin | 17-32 | 8 | 0.00% | 0.00% | 100.00% | 2.604 |
| replay64 | Fixed Cap 8 | reference_length_bin | 5-16 | 8 | 0.00% | 0.00% | 100.00% | 2.535 |
| replay64 | Fixed Cap 16 | category | arithmetic | 12 | 25.00% | 0.00% | 0.00% | 2.717 |
| replay64 | Fixed Cap 16 | category | coding | 12 | 41.67% | 0.00% | 0.00% | 2.625 |
| replay64 | Fixed Cap 16 | category | logic | 12 | 25.00% | 91.67% | 0.00% | 2.756 |
| replay64 | Fixed Cap 16 | category | reading | 12 | 91.67% | 16.67% | 0.00% | 3.624 |
| replay64 | Fixed Cap 16 | category | structured_output | 8 | 0.00% | 0.00% | 100.00% | 4.747 |
| replay64 | Fixed Cap 16 | category | text_quality | 8 | 0.00% | 0.00% | 100.00% | 4.867 |
| replay64 | Fixed Cap 16 | difficulty | unrated | 64 | 34.38% | 20.31% | 25.00% | 21.336 |
| replay64 | Fixed Cap 16 | baseline_length_bin | 1-4 | 40 | 42.50% | 32.50% | 0.00% | 8.410 |
| replay64 | Fixed Cap 16 | baseline_length_bin | 17-32 | 16 | 0.00% | 0.00% | 100.00% | 9.614 |
| replay64 | Fixed Cap 16 | baseline_length_bin | 5-16 | 8 | 62.50% | 0.00% | 0.00% | 3.311 |
| replay64 | Fixed Cap 16 | reference_length_bin | 1-4 | 48 | 45.83% | 27.08% | 0.00% | 11.722 |
| replay64 | Fixed Cap 16 | reference_length_bin | 17-32 | 8 | 0.00% | 0.00% | 100.00% | 4.867 |
| replay64 | Fixed Cap 16 | reference_length_bin | 5-16 | 8 | 0.00% | 0.00% | 100.00% | 4.747 |
| replay64 | Answer Boundary | category | arithmetic | 12 | 25.00% | 100.00% | 0.00% | 1.574 |
| replay64 | Answer Boundary | category | coding | 12 | 41.67% | 100.00% | 0.00% | 1.673 |
| replay64 | Answer Boundary | category | logic | 12 | 25.00% | 100.00% | 0.00% | 1.924 |
| replay64 | Answer Boundary | category | reading | 12 | 91.67% | 100.00% | 0.00% | 1.700 |
| replay64 | Answer Boundary | category | structured_output | 8 | 100.00% | 0.00% | 100.00% | 7.267 |
| replay64 | Answer Boundary | category | text_quality | 8 | 100.00% | 100.00% | 0.00% | 5.871 |
| replay64 | Answer Boundary | difficulty | unrated | 64 | 59.38% | 87.50% | 12.50% | 20.009 |
| replay64 | Answer Boundary | baseline_length_bin | 1-4 | 40 | 42.50% | 100.00% | 0.00% | 5.775 |
| replay64 | Answer Boundary | baseline_length_bin | 17-32 | 16 | 100.00% | 50.00% | 50.00% | 13.138 |
| replay64 | Answer Boundary | baseline_length_bin | 5-16 | 8 | 62.50% | 100.00% | 0.00% | 1.096 |
| replay64 | Answer Boundary | reference_length_bin | 1-4 | 48 | 45.83% | 100.00% | 0.00% | 6.871 |
| replay64 | Answer Boundary | reference_length_bin | 17-32 | 8 | 100.00% | 100.00% | 0.00% | 5.871 |
| replay64 | Answer Boundary | reference_length_bin | 5-16 | 8 | 100.00% | 0.00% | 100.00% | 7.267 |
| replay64 | Confidence-Aware | category | arithmetic | 12 | 25.00% | 41.67% | 0.00% | 2.326 |
| replay64 | Confidence-Aware | category | coding | 12 | 41.67% | 100.00% | 0.00% | 1.851 |
| replay64 | Confidence-Aware | category | logic | 12 | 25.00% | 91.67% | 0.00% | 2.318 |
| replay64 | Confidence-Aware | category | reading | 12 | 91.67% | 50.00% | 0.00% | 2.932 |
| replay64 | Confidence-Aware | category | structured_output | 8 | 100.00% | 0.00% | 100.00% | 7.199 |
| replay64 | Confidence-Aware | category | text_quality | 8 | 100.00% | 100.00% | 0.00% | 5.806 |
| replay64 | Confidence-Aware | difficulty | unrated | 64 | 59.38% | 65.62% | 12.50% | 22.433 |
| replay64 | Confidence-Aware | baseline_length_bin | 1-4 | 40 | 42.50% | 80.00% | 0.00% | 6.595 |
| replay64 | Confidence-Aware | baseline_length_bin | 17-32 | 16 | 100.00% | 50.00% | 50.00% | 13.005 |
| replay64 | Confidence-Aware | baseline_length_bin | 5-16 | 8 | 62.50% | 25.00% | 0.00% | 2.832 |
| replay64 | Confidence-Aware | reference_length_bin | 1-4 | 48 | 45.83% | 70.83% | 0.00% | 9.428 |
| replay64 | Confidence-Aware | reference_length_bin | 17-32 | 8 | 100.00% | 100.00% | 0.00% | 5.806 |
| replay64 | Confidence-Aware | reference_length_bin | 5-16 | 8 | 100.00% | 0.00% | 100.00% | 7.199 |
| replay64 | Adaptive Stop | category | arithmetic | 12 | 25.00% | 58.33% | 0.00% | 2.099 |
| replay64 | Adaptive Stop | category | coding | 12 | 41.67% | 100.00% | 0.00% | 1.690 |
| replay64 | Adaptive Stop | category | logic | 12 | 25.00% | 100.00% | 0.00% | 1.919 |
| replay64 | Adaptive Stop | category | reading | 12 | 91.67% | 83.33% | 0.00% | 1.926 |
| replay64 | Adaptive Stop | category | structured_output | 8 | 100.00% | 0.00% | 100.00% | 7.277 |
| replay64 | Adaptive Stop | category | text_quality | 8 | 100.00% | 100.00% | 0.00% | 5.832 |
| replay64 | Adaptive Stop | difficulty | unrated | 64 | 59.38% | 76.56% | 12.50% | 20.743 |
| replay64 | Adaptive Stop | baseline_length_bin | 1-4 | 40 | 42.50% | 85.00% | 0.00% | 6.287 |
| replay64 | Adaptive Stop | baseline_length_bin | 17-32 | 16 | 100.00% | 50.00% | 50.00% | 13.109 |
| replay64 | Adaptive Stop | baseline_length_bin | 5-16 | 8 | 62.50% | 87.50% | 0.00% | 1.347 |
| replay64 | Adaptive Stop | reference_length_bin | 1-4 | 48 | 45.83% | 85.42% | 0.00% | 7.634 |
| replay64 | Adaptive Stop | reference_length_bin | 17-32 | 8 | 100.00% | 100.00% | 0.00% | 5.832 |
| replay64 | Adaptive Stop | reference_length_bin | 5-16 | 8 | 100.00% | 0.00% | 100.00% | 7.277 |
| fresh64 | Baseline | category | arithmetic | 8 | 25.00% | 0.00% | 0.00% | 2.608 |
| fresh64 | Baseline | category | coding | 8 | 87.50% | 50.00% | 0.00% | 1.480 |
| fresh64 | Baseline | category | logic | 8 | 50.00% | 87.50% | 0.00% | 1.570 |
| fresh64 | Baseline | category | reading | 8 | 100.00% | 50.00% | 0.00% | 2.409 |
| fresh64 | Baseline | category | structured_output | 16 | 100.00% | 0.00% | 100.00% | 17.551 |
| fresh64 | Baseline | category | text_quality | 16 | 100.00% | 100.00% | 0.00% | 14.070 |
| fresh64 | Baseline | difficulty | easy | 32 | 78.12% | 56.25% | 25.00% | 17.280 |
| fresh64 | Baseline | difficulty | moderate | 32 | 87.50% | 40.62% | 25.00% | 22.407 |
| fresh64 | Baseline | baseline_length_bin | 1-4 | 21 | 76.19% | 71.43% | 0.00% | 4.040 |
| fresh64 | Baseline | baseline_length_bin | 17-32 | 20 | 100.00% | 60.00% | 40.00% | 16.579 |
| fresh64 | Baseline | baseline_length_bin | 33+ | 12 | 100.00% | 33.33% | 66.67% | 15.041 |
| fresh64 | Baseline | baseline_length_bin | 5-16 | 11 | 45.45% | 0.00% | 0.00% | 4.027 |
| fresh64 | Baseline | reference_length_bin | 1-4 | 32 | 65.62% | 46.88% | 0.00% | 8.067 |
| fresh64 | Baseline | reference_length_bin | 17-32 | 24 | 100.00% | 66.67% | 33.33% | 24.018 |
| fresh64 | Baseline | reference_length_bin | 5-16 | 8 | 100.00% | 0.00% | 100.00% | 7.602 |
| fresh64 | Fixed Cap 8 | category | arithmetic | 8 | 25.00% | 0.00% | 0.00% | 2.685 |
| fresh64 | Fixed Cap 8 | category | coding | 8 | 87.50% | 50.00% | 0.00% | 1.498 |
| fresh64 | Fixed Cap 8 | category | logic | 8 | 50.00% | 87.50% | 0.00% | 1.607 |
| fresh64 | Fixed Cap 8 | category | reading | 8 | 62.50% | 50.00% | 0.00% | 2.338 |
| fresh64 | Fixed Cap 8 | category | structured_output | 16 | 0.00% | 0.00% | 100.00% | 5.026 |
| fresh64 | Fixed Cap 8 | category | text_quality | 16 | 0.00% | 0.00% | 100.00% | 5.161 |
| fresh64 | Fixed Cap 8 | difficulty | easy | 32 | 21.88% | 31.25% | 50.00% | 9.188 |
| fresh64 | Fixed Cap 8 | difficulty | moderate | 32 | 34.38% | 15.62% | 50.00% | 9.127 |
| fresh64 | Fixed Cap 8 | baseline_length_bin | 1-4 | 21 | 76.19% | 71.43% | 0.00% | 4.249 |
| fresh64 | Fixed Cap 8 | baseline_length_bin | 17-32 | 20 | 0.00% | 0.00% | 100.00% | 6.346 |
| fresh64 | Fixed Cap 8 | baseline_length_bin | 33+ | 12 | 0.00% | 0.00% | 100.00% | 3.841 |
| fresh64 | Fixed Cap 8 | baseline_length_bin | 5-16 | 11 | 18.18% | 0.00% | 0.00% | 3.880 |
| fresh64 | Fixed Cap 8 | reference_length_bin | 1-4 | 32 | 56.25% | 46.88% | 0.00% | 8.128 |
| fresh64 | Fixed Cap 8 | reference_length_bin | 17-32 | 24 | 0.00% | 0.00% | 100.00% | 7.679 |
| fresh64 | Fixed Cap 8 | reference_length_bin | 5-16 | 8 | 0.00% | 0.00% | 100.00% | 2.508 |
| fresh64 | Fixed Cap 16 | category | arithmetic | 8 | 25.00% | 0.00% | 0.00% | 2.635 |
| fresh64 | Fixed Cap 16 | category | coding | 8 | 87.50% | 50.00% | 0.00% | 1.502 |
| fresh64 | Fixed Cap 16 | category | logic | 8 | 50.00% | 87.50% | 0.00% | 1.598 |
| fresh64 | Fixed Cap 16 | category | reading | 8 | 100.00% | 50.00% | 0.00% | 2.637 |
| fresh64 | Fixed Cap 16 | category | structured_output | 16 | 0.00% | 0.00% | 100.00% | 9.449 |
| fresh64 | Fixed Cap 16 | category | text_quality | 16 | 0.00% | 0.00% | 100.00% | 9.609 |
| fresh64 | Fixed Cap 16 | difficulty | easy | 32 | 28.12% | 31.25% | 50.00% | 13.529 |
| fresh64 | Fixed Cap 16 | difficulty | moderate | 32 | 37.50% | 15.62% | 50.00% | 13.902 |
| fresh64 | Fixed Cap 16 | baseline_length_bin | 1-4 | 21 | 76.19% | 71.43% | 0.00% | 4.103 |
| fresh64 | Fixed Cap 16 | baseline_length_bin | 17-32 | 20 | 0.00% | 0.00% | 100.00% | 11.832 |
| fresh64 | Fixed Cap 16 | baseline_length_bin | 33+ | 12 | 0.00% | 0.00% | 100.00% | 7.227 |
| fresh64 | Fixed Cap 16 | baseline_length_bin | 5-16 | 11 | 45.45% | 0.00% | 0.00% | 4.269 |
| fresh64 | Fixed Cap 16 | reference_length_bin | 1-4 | 32 | 65.62% | 46.88% | 0.00% | 8.372 |
| fresh64 | Fixed Cap 16 | reference_length_bin | 17-32 | 24 | 0.00% | 0.00% | 100.00% | 14.367 |
| fresh64 | Fixed Cap 16 | reference_length_bin | 5-16 | 8 | 0.00% | 0.00% | 100.00% | 4.692 |
| fresh64 | Answer Boundary | category | arithmetic | 8 | 25.00% | 100.00% | 0.00% | 1.065 |
| fresh64 | Answer Boundary | category | coding | 8 | 87.50% | 100.00% | 0.00% | 1.056 |
| fresh64 | Answer Boundary | category | logic | 8 | 50.00% | 100.00% | 0.00% | 1.267 |
| fresh64 | Answer Boundary | category | reading | 8 | 100.00% | 100.00% | 0.00% | 1.287 |
| fresh64 | Answer Boundary | category | structured_output | 16 | 100.00% | 0.00% | 100.00% | 17.210 |
| fresh64 | Answer Boundary | category | text_quality | 16 | 50.00% | 50.00% | 50.00% | 6.238 |
| fresh64 | Answer Boundary | difficulty | easy | 32 | 78.12% | 75.00% | 25.00% | 15.163 |
| fresh64 | Answer Boundary | difficulty | moderate | 32 | 62.50% | 50.00% | 50.00% | 12.961 |
| fresh64 | Answer Boundary | baseline_length_bin | 1-4 | 21 | 76.19% | 100.00% | 0.00% | 3.158 |
| fresh64 | Answer Boundary | baseline_length_bin | 17-32 | 20 | 80.00% | 40.00% | 60.00% | 13.253 |
| fresh64 | Answer Boundary | baseline_length_bin | 33+ | 12 | 66.67% | 0.00% | 100.00% | 10.195 |
| fresh64 | Answer Boundary | baseline_length_bin | 5-16 | 11 | 45.45% | 100.00% | 0.00% | 1.518 |
| fresh64 | Answer Boundary | reference_length_bin | 1-4 | 32 | 65.62% | 100.00% | 0.00% | 4.675 |
| fresh64 | Answer Boundary | reference_length_bin | 17-32 | 24 | 66.67% | 33.33% | 66.67% | 15.984 |
| fresh64 | Answer Boundary | reference_length_bin | 5-16 | 8 | 100.00% | 0.00% | 100.00% | 7.465 |
| fresh64 | Confidence-Aware | category | arithmetic | 8 | 25.00% | 50.00% | 0.00% | 1.912 |
| fresh64 | Confidence-Aware | category | coding | 8 | 87.50% | 62.50% | 0.00% | 1.296 |
| fresh64 | Confidence-Aware | category | logic | 8 | 50.00% | 100.00% | 0.00% | 1.273 |
| fresh64 | Confidence-Aware | category | reading | 8 | 100.00% | 100.00% | 0.00% | 1.238 |
| fresh64 | Confidence-Aware | category | structured_output | 16 | 100.00% | 0.00% | 100.00% | 17.460 |
| fresh64 | Confidence-Aware | category | text_quality | 16 | 50.00% | 50.00% | 50.00% | 6.258 |
| fresh64 | Confidence-Aware | difficulty | easy | 32 | 78.12% | 68.75% | 25.00% | 15.703 |
| fresh64 | Confidence-Aware | difficulty | moderate | 32 | 62.50% | 34.38% | 50.00% | 13.734 |
| fresh64 | Confidence-Aware | baseline_length_bin | 1-4 | 21 | 76.19% | 85.71% | 0.00% | 3.334 |
| fresh64 | Confidence-Aware | baseline_length_bin | 17-32 | 20 | 80.00% | 40.00% | 60.00% | 13.402 |
| fresh64 | Confidence-Aware | baseline_length_bin | 33+ | 12 | 66.67% | 0.00% | 100.00% | 10.315 |
| fresh64 | Confidence-Aware | baseline_length_bin | 5-16 | 11 | 45.45% | 63.64% | 0.00% | 2.385 |
| fresh64 | Confidence-Aware | reference_length_bin | 1-4 | 32 | 65.62% | 78.12% | 0.00% | 5.719 |
| fresh64 | Confidence-Aware | reference_length_bin | 17-32 | 24 | 66.67% | 33.33% | 66.67% | 16.117 |
| fresh64 | Confidence-Aware | reference_length_bin | 5-16 | 8 | 100.00% | 0.00% | 100.00% | 7.601 |
| fresh64 | Adaptive Stop | category | arithmetic | 8 | 25.00% | 87.50% | 0.00% | 1.302 |
| fresh64 | Adaptive Stop | category | coding | 8 | 87.50% | 62.50% | 0.00% | 1.300 |
| fresh64 | Adaptive Stop | category | logic | 8 | 50.00% | 100.00% | 0.00% | 1.246 |
| fresh64 | Adaptive Stop | category | reading | 8 | 100.00% | 100.00% | 0.00% | 1.246 |
| fresh64 | Adaptive Stop | category | structured_output | 16 | 100.00% | 0.00% | 100.00% | 17.402 |
| fresh64 | Adaptive Stop | category | text_quality | 16 | 100.00% | 100.00% | 0.00% | 13.928 |
| fresh64 | Adaptive Stop | difficulty | easy | 32 | 78.12% | 71.88% | 25.00% | 15.549 |
| fresh64 | Adaptive Stop | difficulty | moderate | 32 | 87.50% | 65.62% | 25.00% | 20.875 |
| fresh64 | Adaptive Stop | baseline_length_bin | 1-4 | 21 | 76.19% | 85.71% | 0.00% | 3.349 |
| fresh64 | Adaptive Stop | baseline_length_bin | 17-32 | 20 | 100.00% | 60.00% | 40.00% | 16.461 |
| fresh64 | Adaptive Stop | baseline_length_bin | 33+ | 12 | 100.00% | 33.33% | 66.67% | 14.869 |
| fresh64 | Adaptive Stop | baseline_length_bin | 5-16 | 11 | 45.45% | 90.91% | 0.00% | 1.744 |
| fresh64 | Adaptive Stop | reference_length_bin | 1-4 | 32 | 65.62% | 87.50% | 0.00% | 5.094 |
| fresh64 | Adaptive Stop | reference_length_bin | 17-32 | 24 | 100.00% | 66.67% | 33.33% | 23.815 |
| fresh64 | Adaptive Stop | reference_length_bin | 5-16 | 8 | 100.00% | 0.00% | 100.00% | 7.515 |


길이 그룹은 같은 질문의Baseline 출력 평균 길이 또는 reference token 길이이며 사후 분석 전용이다. 짧게 중단한 방법의 결과 길이로 그룹을 다시 나누어 유리한 질문만 비교하지 않았다. reference/difficulty 라벨은 runtime에 전달하지 않는다.



## 8. 기존 점수 보존과 감사

| 셋 | 방식 | 기존 점수 | 새 내용 지표 | 기존 형식 | 새 형식 | 완전 과제 성공 |
| --- | --- | --- | --- | --- | --- | --- |
| replay64 | Baseline | 59.38% | 59.38% | 32.81% | 32.81% | 18.75% |
| replay64 | Fixed Cap 8 | 26.56% | 26.56% | 32.81% | 20.31% | 6.25% |
| replay64 | Fixed Cap 16 | 34.38% | 34.38% | 32.81% | 20.31% | 6.25% |
| replay64 | Answer Boundary | 59.38% | 59.38% | 87.50% | 87.50% | 46.88% |
| replay64 | Confidence-Aware | 59.38% | 59.38% | 65.62% | 65.62% | 32.81% |
| replay64 | Adaptive Stop | 59.38% | 59.38% | 76.56% | 76.56% | 40.62% |
| fresh64 | Baseline | 70.31% | 70.31% | 35.94% | 48.44% | 29.69% |
| fresh64 | Fixed Cap 8 | 28.12% | 28.12% | 48.44% | 23.44% | 17.19% |
| fresh64 | Fixed Cap 16 | 32.81% | 32.81% | 48.44% | 23.44% | 17.19% |
| fresh64 | Answer Boundary | 70.31% | 70.31% | 75.00% | 62.50% | 45.31% |
| fresh64 | Confidence-Aware | 70.31% | 70.31% | 64.06% | 51.56% | 37.50% |
| fresh64 | Adaptive Stop | 70.31% | 70.31% | 56.25% | 68.75% | 40.62% |


기존 CSV나 원래 점수를 수정하지 않았다. 문장 약어 Dr.를 독립 문장으로 세는 오류를 보호하고, 명시된 직업 사실도 보존하는지 확인한다. 내용과 형식을 나눴으므로 기존 correct와 값이 달라질 수 있다. 새 점수를 기존 정확도 개선으로 포장하지 않는다. 모델 프롬프트/정답은 변경하지 않았다.

| 검사 | 결과 |
| --- | --- |
| existing_files_unchanged | 5097 |
| new_stem_past_overlap | 0 |
| replay_fresh_disjoint | True |
| frozen_policy_and_no_eval_tuning | True |
| all_model_calls_one_per_request | True |
| all_28_layers_and_cache_verified | True |
| native_attention_mlp_audit_verified | True |
| actual_probability_queries | 624 |
| probabilities_in_range | True |
| confidence_is_not_calibrated_correctness_probability | True |
| prefix_disagreements | 0 |
| requests | 2304 |
| unique_questions | 128 |
| repeats | 3 |
| old_replay_difficulty_unrated | True |
| fresh_challenge_abbreviation_cases | 8 |
| source_score_preserved | True |
| GPU_elapsed_power_money | unmeasured |
| explicit_replay | False |
| freshness_scope | Original pre-ACS holdout provenance; explicit replays are not newly unused problems. |


자동 테스트: **전체 unittest242개:238통과·4개 NumPy 환경skip (75.269초); 격리 MLX 환경 신규/관련24개 모두 통과 (0.074초); 실제 Full·attention/MLP·cache 감사 통과; 완료 재개 추가생성0 확인**. 테스트 모의 값으로 성능표를 만들지 않았다.



## 9. 원문자열 판정 보존·기존 기술과 차이·다음 연구

아래는 원문자열 자동판정이며 동의어 누락으로 Boundary/Confidence를 잘못 후보로 남겼던 결과다. 최종 판정에는0절의 사후 사실 감사를 사용하며 사후 감사라는 한계도 함께 적용한다.

| 셋 | 방식 | 원품질 점추정 조건 | 시간 CI 개선 | 실제 모델작업 감소 | 원관측 후보 | 모집단 무손실 입증 |
| --- | --- | --- | --- | --- | --- | --- |
| replay64 | Fixed Cap 8 | False | True | True | False | False |
| replay64 | Fixed Cap 16 | False | True | True | False | False |
| replay64 | Answer Boundary | True | True | True | True | False |
| replay64 | Confidence-Aware | True | True | True | True | False |
| replay64 | Adaptive Stop | True | True | True | True | False |
| fresh64 | Fixed Cap 8 | False | True | True | False | False |
| fresh64 | Fixed Cap 16 | False | True | True | False | False |
| fresh64 | Answer Boundary | True | True | True | True | False |
| fresh64 | Confidence-Aware | True | True | True | True | False |
| fresh64 | Adaptive Stop | True | True | True | True | False |


ACS는 기존 custom stopping criteria에 요청 형식 parser·현재 실제 token 확률·유형별 고정 규칙을 조합한 실험이다. 확률분포/가중치를 바꾸지 않고 생성 종료만 바꾼다. 사전 정의된 규칙이 실제 정답 완성 시점을 의미적으로 알아낸다고 주장할 수 없다. [Hugging Face stopping criteria](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.StoppingCriteria), [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)와 범위를 구분한다. 독립적인 새로운 기술이라는 근거는 없다.

다음은 고정 상한의 더 넓은 사전 고정 sweep, 작성자가 규칙을 예상하지 못하는 독립적인 자연 요청, 다양한 출력 길이/JSON스키마/약어/실제 코드 생성에서 실패율을 확인하는 연구가 우선이다. 이 연구는 모델1개·합성64+64·3반복이며 전체 서비스에 절감률을 일반화하지 않는다. 사후 결과에 맞춘 임계값 보정이나 추가 최적화는 수행하지 않았다.



## 10. 재현과 재개

```bash
# 현재 실행 재개: 완료 요청은 재실행하지 않음
.alsr-venv/bin/python -B -u -m src.acs_experiment ./data/acs_16a7001279c947c0acb730672deb1a97
# 같은 고정 데이터로 새 폴더에서 실제 모델 전체 재실행
.alsr-venv/bin/python -B -u -m src.acs_replay ./data/acs_16a7001279c947c0acb730672deb1a97/MANIFEST.json
# 완료 후 새 분석 폴더 생성 및 별도 내용 감사
.venv/bin/python -B -m src.acs_analysis <실행 폴더>
.venv/bin/python -B -m src.acs_content_audit <ANALYSIS 폴더>
.venv/bin/python -B -m src.acs_report <ANALYSIS 폴더> "<실제 테스트 결과>"
```

재현은 같은 문제를 재사용하며 새 미사용 평가 데이터라고 부르지 않는다. 코어 소스/프로파일/패키지가 달라지면 재개를 거부한다. 요청 JSON은 새파일+fsync로 저장하고 부분 파일도 보존한다. 실제 원본 CSV는 각 EXPORT 폴더, session load/워밍업·native 감사는 sessions에 저장된다. 요청 latency에는 모델·종료판단·tokenize/decode를 포함하며 오프라인 채점/체크포인트 쓰기/모델 load는 별도 연구 단계다.



[사실 감사 결과](AUDITED_RESULTS.csv) · [원자동점수](RESULTS.csv) · [반복 원본](REPEATS.csv) · [감사 CI](AUDITED_COMPARISONS.csv) · [전체 감사](AUDITED_TRANSITIONS.csv) · [감사 실패](AUDITED_FAILURES.csv) · [확률](PROBABILITY_EVENTS.csv) · [그래프](ACS_COMPARISON.png) · [정확도-시간](ACS_PARETO.png)
