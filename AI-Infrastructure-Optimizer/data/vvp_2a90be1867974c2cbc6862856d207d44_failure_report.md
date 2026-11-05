# VVP 실패 사례 분석

주 예산 25%는 두 모델 모두 No Verification 대비 순정답 개선 0이다. 개별 사례를 반복 행으로 독립 표본처럼 세지 않는다.

| 모델 | 예산 | 수정 평균 | 훼손 평균 | 놓친 개선 평균 | 미추출 복구 평균 | 추출된 오답 복구 평균 |
| --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | 0 | 0 | 0 | 22 | 0 | 0 |
| qwen2.5:1.5b | 10 | 0 | 0 | 22 | 0 | 0 |
| qwen2.5:1.5b | 25 | 0 | 0 | 22 | 0 | 0 |
| qwen2.5:1.5b | 50 | 0 | 0 | 22 | 0 | 0 |
| qwen2.5:1.5b | 75 | 0 | 0 | 22 | 0 | 0 |
| qwen2.5:1.5b | 100 | 0 | 0 | 22 | 0 | 0 |
| qwen2.5:0.5b | 0 | 0 | 0 | 3 | 0 | 0 |
| qwen2.5:0.5b | 10 | 2 | 0 | 1 | 2 | 0 |
| qwen2.5:0.5b | 25 | 3 | 3 | 0 | 2 | 1 |
| qwen2.5:0.5b | 50 | 3 | 3 | 0 | 2 | 1 |
| qwen2.5:0.5b | 75 | 3 | 3 | 0 | 2 | 1 |
| qwen2.5:0.5b | 100 | 3 | 3 | 0 | 2 | 1 |

1.5B의 EV가 양수인 후보가 없어 모두 생략하는 퇴화가 발생했다. 0.5B에서는 기존 채점상 복구와 훼손이 상쇄되는 조건을 관찰했다. 임계값을 평가 결과에 맞춰 변경하지 않았다. 미추출 복구와 이미 추출된 오답 복구는 위 표와 원본 CSV에 분리했다. 미추출 복구가 모두 형식만의 개선인 것은 아니다. [별도 내용 감사](vvp_2a90be1867974c2cbc6862856d207d44_content_audit_report.md)에서는 복구 2건·훼손 4건으로 순정답 2개를 잃었다.



## 예시: 첫 반복의 유형별 최대 3건
사후 설명용 사례이며 학습·선택에 사용하지 않았다. 정답 기회 기준인 Full100은 별도 시간에 실행한 검증이다.

| 모델 | 사건 | ID | 유형 | 질문 | 정답 | 최초 답 | VVP 답 | Full100 답 | 예측 EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | 놓친 복구 | a1c654995378 | coding | What does this Python program output? ```python text = 'violet5108 blue green amber' print(len(text.split())) ``` A) Error B) 3 C) 5 D) 4 | D | B | B | D | -0.064 |
| qwen2.5:1.5b | 놓친 복구 | f132dbe32995 | coding | What does this Python program output? ```python text = 'violet5105 blue green amber' print(len(text.split())) ``` A) 5 B) 3 C) Error D) 4 | D | A | A | D | -0.064 |
| qwen2.5:1.5b | 놓친 복구 | 5003a4467186 | coding | What does this Python program output? ```python text = 'violet5104 blue green amber' print(len(text.split())) ``` A) Error B) 3 C) 5 D) 4 | D | B | B | D | -0.064 |
| qwen2.5:0.5b | 정답 훼손 | 727657aefbe2 | coding | What does this Python program output? ```python a = [227, 230, 235, 241] print(a[::-1][2]) ``` A) 235 B) 230 C) Error D) 241 | B | B | A | A | 0.028 |
| qwen2.5:0.5b | 정답 훼손 | fe4696522d65 | coding | What does this Python program output? ```python d = {'n': 224, 'x': 6} print(d.get('missing', 13) + d['x']) ``` A) 230 B) 19 C) 13 D) Error | B | B | A | A | 0.023 |
| qwen2.5:0.5b | 정답 훼손 | b19aebd6121b | coding | What does this Python program output? ```python d = {'n': 232, 'x': 6} print(d.get('missing', 13) + d['x']) ``` A) 238 B) 19 C) 13 D) Error | B | B | A | A | 0.023 |
| qwen2.5:0.5b | 오답 복구 | b5e57fe2f41f | coding | What does this Python program output? ```python print([n-2 for n in range(225, 231) if n % 2 == 1]) ``` A) [] B) Error C) [223, 225, 227] D) [225, 227, 229] | C | None | C | C | 0.090 |
| qwen2.5:0.5b | 오답 복구 | b50395f427c3 | coding | What does this Python program output? ```python print([n-2 for n in range(221, 227) if n % 2 == 1]) ``` A) [221, 223, 225] B) [] C) Error D) [219, 221, 223] | D | None | D | D | 0.081 |
| qwen2.5:0.5b | 오답 복구 | e25509c7e7a9 | coding | What does this Python program output? ```python print([n-2 for n in range(227, 233) if n % 2 == 1]) ``` A) [225, 227, 229] B) [] C) Error D) [227, 229, 231] | A | B | A | A | 0.027 |
| qwen2.5:0.5b | 검증했으나 정오 유지 | 93d3142e8378 | coding | What does this Python program output? ```python a = [232, 235, 240, 246] print(a[::-1][2]) ``` A) 240 B) Error C) 246 D) 235 | D | None | A | A | 0.125 |
| qwen2.5:0.5b | 검증했으나 정오 유지 | cdd74b0495f6 | coding | What does this Python program output? ```python a = [221, 224, 229, 235] print(a[::-1][2]) ``` A) 235 B) Error C) 229 D) 224 | D | None | A | A | 0.125 |
| qwen2.5:0.5b | 검증했으나 정오 유지 | 82f929fb98a5 | coding | What does this Python program output? ```python a = [222, 225, 230, 236] print(a[::-1][2]) ``` A) Error B) 236 C) 225 D) 230 | C | None | A | A | 0.125 |

## CPU 예산 예측 오차
다음은 실제/예산 비율이 큰 조건들이다. 예산은 선택된 항목 예상 비용의 합이 아닌 전체 허용량이므로 이 비율만으로 항목별 회귀 오차를 추정하지 않는다. 실제 총 CPU에는 판단 비용이 포함된다.

| 모델 | 방식 | 예산 | 반복 | 호출 | 예측 허용 CPU초 | 실제 총 CPU초 | 비율 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:0.5b | vvp | 25 | 4 | 75 | 4.953 | 17.293 | 3.491 |
| qwen2.5:0.5b | vvp | 25 | 5 | 75 | 4.953 | 15.919 | 3.214 |
| qwen2.5:0.5b | vvp | 25 | 3 | 75 | 4.953 | 15.737 | 3.177 |
| qwen2.5:0.5b | vvp | 25 | 2 | 75 | 4.953 | 15.717 | 3.173 |
| qwen2.5:0.5b | vvp | 25 | 1 | 75 | 4.953 | 15.315 | 3.092 |
| qwen2.5:0.5b | selective | 10 | 2 | 30 | 1.981 | 5.561 | 2.807 |
| qwen2.5:0.5b | full | 10 | 2 | 29 | 1.981 | 5.185 | 2.617 |
| qwen2.5:0.5b | vvp | 50 | 4 | 150 | 9.906 | 23.238 | 2.346 |
| qwen2.5:0.5b | selective | 10 | 5 | 30 | 1.981 | 4.626 | 2.335 |
| qwen2.5:0.5b | selective | 25 | 5 | 75 | 4.953 | 11.338 | 2.289 |
| qwen2.5:0.5b | vvp | 50 | 2 | 150 | 9.906 | 22.662 | 2.288 |
| qwen2.5:0.5b | full | 25 | 2 | 75 | 4.953 | 11.249 | 2.271 |

개별 정답·응답·선택: [vvp_2a90be1867974c2cbc6862856d207d44_outcomes.csv](vvp_2a90be1867974c2cbc6862856d207d44_outcomes.csv). VVP 오답 행: [vvp_2a90be1867974c2cbc6862856d207d44_failure_cases.csv](vvp_2a90be1867974c2cbc6862856d207d44_failure_cases.csv). CPU 예측 오차: [vvp_2a90be1867974c2cbc6862856d207d44_budget_actual.csv](vvp_2a90be1867974c2cbc6862856d207d44_budget_actual.csv). 전체 해석: [연구 보고서](../VVP_RESEARCH_REPORT.md).
