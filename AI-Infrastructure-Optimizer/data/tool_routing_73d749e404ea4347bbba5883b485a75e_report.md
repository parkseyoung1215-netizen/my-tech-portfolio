# Tool-Assisted Routing 실제 비교

실행 ID: `73d749e404ea4347bbba5883b485a75e`.

기존 고정 1.5B·Adaptive·Coverage 코드는 변경하지 않고 같은 실행에서 실제 재측정했습니다. Coverage 실험의 기존 200문제와 당시 신규 100문제를 그대로 사용합니다. 이번에 처음 만든 홀드아웃은 아니며 모델 재보정·임계값 변경·평가 정답의 런타임 전달은 없습니다.

도구는 입력 전체가 명확한 ASCII 정수 +,-,* 및 괄호 수식인 경우에만 직접 반환합니다. AST 노드 허용 목록과 길이·깊이·노드·리터럴·중간 결과 제한을 사용하며 Python 코드를 실행하지 않습니다. 계산값은 모델로 재생성하지 않습니다. 그 외 요청은 원래 Adaptive 모델 라우터로 전달하며, 그 라우터가 적합한 모델을 찾지 못하면 거절할 수 있습니다.

## replay200

| 방식 | 전체 정확도 | 답변 비율 | 답변 정확도 | 평균 지연(ms) | 처리 시간 합계(초) | 0.5B 호출 | 1.5B 호출 | 직접 계산 호출 | 검증용 계산 호출 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 고정 1.5B | 55.00% | 97.50% | 56.41% | 232.051498 | 46.410300 | 0 | 200 | 0 | 0 |
| Adaptive | 49.50% | 50.00% | 99.00% | 44.381059 | 8.876212 | 100 | 0 | 0 | 0 |
| Coverage | 64.50% | 64.50% | 100.00% | 259.613676 | 51.922735 | 100 | 175 | 0 | 274 |
| Tool-Assisted | 100.00% | 100.00% | 100.00% | 0.107714 | 0.021543 | 0 | 0 | 200 | 0 |

## unseen100

| 방식 | 전체 정확도 | 답변 비율 | 답변 정확도 | 평균 지연(ms) | 처리 시간 합계(초) | 0.5B 호출 | 1.5B 호출 | 직접 계산 호출 | 검증용 계산 호출 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 고정 1.5B | 33.00% | 98.00% | 33.67% | 344.279142 | 34.427914 | 0 | 100 | 0 | 0 |
| Adaptive | 20.00% | 20.00% | 100.00% | 22.756709 | 2.275671 | 20 | 0 | 0 | 0 |
| Coverage | 44.00% | 44.00% | 100.00% | 525.496117 | 52.549612 | 20 | 137 | 0 | 157 |
| Tool-Assisted | 100.00% | 100.00% | 100.00% | 0.116766 | 0.011677 | 0 | 0 | 100 | 0 |

평균 지연과 처리 시간 합계에는 선택·계산·검증·실패·거절·fallback이 포함됩니다. 워밍업과 사후 채점·파일 저장은 제외합니다. 호출은 재시도를 포함한 실제 횟수입니다. Coverage의 기존 산술 검증도 계산을 수행하므로 직접 답변 도구와 별도 열로 기록했습니다. 검증용 계산 열은 추출 실패로 계산까지 진행하지 못한 검증을 제외합니다.

## 성공 판정 및 같은 답변 범위의 속도

성공은 각 세트·각 기준선 대비 **전체 정확도와 답변 비율이 모두 엄격하게 증가한 경우에만** 판정합니다.

| 세트 | 기준선 | 정확도 증가 | 답변 비율 증가 | 성공 | 공통 답변 | 같은 범위 시간: 기준선 / 도구(초) | 잃은 답변 |
|---|---|---|---|---|---:|---|---:|
| replay200 | 고정 1.5B | True | True | True | 195 | 44.308331 / 0.020943 | 0 |
| replay200 | Adaptive | True | True | True | 100 | 8.806156 / 0.010408 | 0 |
| replay200 | Coverage | True | True | True | 129 | 15.384078 / 0.013899 | 0 |
| unseen100 | 고정 1.5B | True | True | True | 98 | 33.315576 / 0.011493 | 0 |
| unseen100 | Adaptive | True | True | True | 20 | 2.213578 / 0.002241 | 0 |
| unseen100 | Coverage | True | True | True | 44 | 9.622067 / 0.004960 | 0 |

사전 정의한 두 지표의 동시 개선 조건: **충족**. 거절로 인한 시간 감소를 성공 근거로 삼지 않습니다.

## 도구 오적용·fallback 점검

주 평가의 도구 오답/오적용 0건, 도구→모델 라우터 fallback 0건. 주 평가가 모두 지원 산술식이므로 별도 점검을 함께 실행했습니다.

| 점검 사례 | 도구 허용 예상 | 실제 답변 출처 | fallback 원인 | 실제 모델 호출 | 오적용 |
|---|---|---|---|---:|---|
| precedence | True | tool |  | 0 | False |
| signed_parentheses | True | tool |  | 0 | False |
| natural_language | False | model_fallback | not_integer_expression | 0 | False |
| text_with_expression | False | model_fallback | not_integer_expression | 0 | False |
| units | False | model_fallback | not_integer_expression | 0 | False |
| function_call | False | model_fallback | not_integer_expression | 0 | False |
| power | False | model_fallback | unsupported_operation | 0 | False |
| division | False | model_fallback | not_integer_expression | 0 | False |
| decimal | False | model_fallback | not_integer_expression | 0 | False |
| plain_number | False | model_fallback | no_binary_operation | 0 | False |
| leading_zero_model_fallback | False | model_fallback | invalid_syntax | 1 | False |
| depth_limit | False | model_fallback | depth_limit | 0 | False |
| result_limit | False | model_fallback | result_limit | 0 | False |
| length_limit | False | model_fallback | length_limit | 0 | False |

점검 사례는 주 평가의 정확도·지연·호출 합계에 넣지 않았습니다. 구문/안전 제한으로 도구가 거부한 경우에는 기존 라우터를 실제 호출했습니다. 자연어·미지원 수식을 원래 라우터도 처리하지 못하면 모델 호출 없이 거절합니다. 선행 0이 있는 식은 Python 정수 구문으로 허용하지 않아 원래 모델 경로를 점검하는 사례로 사용했습니다.

## 무결성·비용·한계

원본 소스·데이터 해시 보존: True. 독립 추출기 불일치: 0건. 전체 루프 경과 시간: 196.545086초(진행 출력·채점 포함, 워밍업·저장 제외).

문제 명세와 프로필·소스 해시를 실행 전 고정했고, 외부 AST 채점기로 기준 정답을 검사했습니다. 계산 도구·런타임에는 기대 정답을 전달하지 않았습니다. 실제 모델 응답과 직접 도구 결과를 출처별로 기록했습니다.

금전 비용·전력·CPU·메모리·FLOPs는 이번 실험에서 미측정입니다. 호출 수 감소를 금전 절감률로 바꾸지 않았습니다. Ollama 서버는 계속 실행 중이므로 모델 호출이 없어도 상주 메모리나 유휴 비용이 사라졌다는 뜻은 아닙니다.

이 결과는 지원하는 산술식 영역에 한정됩니다. 소수·나눗셈·거듭제곱·단위·일반 문장·일반 AI 작업의 성능은 개선됐다고 주장하지 않습니다. 100문제 세트도 이전에 평가한 데이터의 재실행이며 완전히 새로운 홀드아웃은 아닙니다. 단일 실행의 짧은 도구 지연은 반복 실험·부하 검증이 필요하고 모델과의 속도 차이를 모든 작업에 일반화할 수 없습니다.

## 파일

- [manifest.json](tool_routing_73d749e404ea4347bbba5883b485a75e_manifest.json)
- [evaluation.csv](tool_routing_73d749e404ea4347bbba5883b485a75e_evaluation.csv)
- [summary.csv](tool_routing_73d749e404ea4347bbba5883b485a75e_summary.csv)
- [paired.csv](tool_routing_73d749e404ea4347bbba5883b485a75e_paired.csv)
- [tool_events.csv](tool_routing_73d749e404ea4347bbba5883b485a75e_tool_events.csv)
- [tool_misapplications.csv](tool_routing_73d749e404ea4347bbba5883b485a75e_tool_misapplications.csv)
- [fallbacks.csv](tool_routing_73d749e404ea4347bbba5883b485a75e_fallbacks.csv)
- [probes.csv](tool_routing_73d749e404ea4347bbba5883b485a75e_probes.csv)
- [metadata.json](tool_routing_73d749e404ea4347bbba5883b485a75e_metadata.json)
