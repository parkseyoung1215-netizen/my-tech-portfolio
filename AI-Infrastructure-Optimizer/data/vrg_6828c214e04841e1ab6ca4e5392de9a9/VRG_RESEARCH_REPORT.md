# Verification Revision Guard (VRG) 연구 보고서

## 결론
**사전 지정 성공 조건을 통과한 Guard는 없다.** 정답 훼손 감소와 정확도–비용 우월성은 별개이며 검증을 이미 수행한 뒤 판단하는 VRG가 검증 연산을 절약한다고 주장하지 않는다.

실험 `vrg_6828c214e04841e1ab6ca4e5392de9a9`: 새 최종 평가 300문제, 모델 2개, 실제 실행 방식 5개, 각 5회 반복이다. 실제 검증 10,500회, 최초 모델 생성 480회, 새 보정 수집 360회, 실제 평가 행 15,000개다. 검증 루프 전체 벽시계 시간은 1344.563초(언로드·워밍업·기록 포함)이며 학습/공통 최초 생성은 별도다. Oracle 3,000행은 별도 사후 상한선이며 추가 실행이나 무료 시스템이 아니다. 워밍업 호출은 원시 파일에 별도 보존했다.

## 1. 원인 분석: 기존 실험
기존 VVP 결과를 모델·방식·유형별 CC/CW/WC/WW로 나눴다. 원본 점수는 보존하고 이전 내용 감사 규칙을 적용한 값도 별도 기록했다. 아래 표는 **기존** 내용 기준 전이의 5반복 평균이며, 같은 최초 답변을 반복한 1,500행을 독립 문제 1,500개로 보지 않는다.

| 모델 | 기존 방식 | CC | CW | WC | WW | 초기 정답 중 파괴율 | 초기 오답 중 복구율 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | selective | 169.0 | 14.0 | 18.0 | 99.0 | 7.65% | 15.38% |
| qwen2.5:1.5b | rcav | 183.0 | 0.0 | 0.0 | 117.0 | 0.00% | 0.00% |
| qwen2.5:1.5b | improved | 179.0 | 4.0 | 2.0 | 115.0 | 2.19% | 1.71% |
| qwen2.5:1.5b | vvp | 183.0 | 0.0 | 0.0 | 117.0 | 0.00% | 0.00% |
| qwen2.5:1.5b | none | 183.0 | 0.0 | 0.0 | 117.0 | 0.00% | 0.00% |
| qwen2.5:1.5b | full | 149.0 | 34.0 | 22.0 | 95.0 | 18.58% | 18.80% |
| qwen2.5:0.5b | rcav | 150.0 | 1.0 | 1.0 | 148.0 | 0.66% | 0.67% |
| qwen2.5:0.5b | none | 151.0 | 0.0 | 0.0 | 149.0 | 0.00% | 0.00% |
| qwen2.5:0.5b | improved | 150.0 | 1.0 | 2.0 | 147.0 | 0.66% | 1.34% |
| qwen2.5:0.5b | selective | 140.0 | 11.0 | 2.0 | 147.0 | 7.28% | 1.34% |
| qwen2.5:0.5b | vvp | 147.0 | 4.0 | 2.0 | 147.0 | 2.65% | 1.34% |
| qwen2.5:0.5b | full | 139.0 | 12.0 | 2.0 | 147.0 | 7.95% | 1.34% |

기존 전수 검증의 유형별 평균:
| 모델 | 유형 | CC | CW | WC | WW | 전체 중 CW율 | 전체 중 WC율 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | coding | 5.0 | 4.0 | 7.0 | 44.0 | 6.67% | 11.67% |
| qwen2.5:1.5b | knowledge | 46.0 | 2.0 | 1.0 | 11.0 | 3.33% | 1.67% |
| qwen2.5:1.5b | logic | 18.0 | 10.0 | 11.0 | 21.0 | 16.67% | 18.33% |
| qwen2.5:1.5b | comprehension | 33.0 | 5.0 | 3.0 | 19.0 | 8.33% | 5.00% |
| qwen2.5:1.5b | arithmetic | 47.0 | 13.0 | 0.0 | 0.0 | 21.67% | 0.00% |
| qwen2.5:0.5b | coding | 17.0 | 4.0 | 2.0 | 37.0 | 6.67% | 3.33% |
| qwen2.5:0.5b | knowledge | 31.0 | 7.0 | 0.0 | 22.0 | 11.67% | 0.00% |
| qwen2.5:0.5b | logic | 13.0 | 0.0 | 0.0 | 47.0 | 0.00% | 0.00% |
| qwen2.5:0.5b | comprehension | 18.0 | 1.0 | 0.0 | 41.0 | 1.67% | 0.00% |
| qwen2.5:0.5b | arithmetic | 60.0 | 0.0 | 0.0 | 0.0 | 0.00% | 0.00% |

기존 로그는 자기 검증이 정답을 뒤집을 수 있음을 보여주지만 내부 추론의 심리적 원인을 입증하지 않는다. 도구 정답을 모델로 재계산하는 불필요한 수정, 추출 실패와 실제 내용 오류, 모델별 복구/훼손 비대칭을 구분했다. 원본 및 내용 기준 모든 유형/방식 전이는 `../vrg_6828c214e04841e1ab6ca4e5392de9a9_historical_transitions.csv`에 있다. 기존 Full은 미추출 수정안을 최초 답변으로 fallback했으므로 이번 Always Accept와 정확히 같은 정의가 아니다. 과거/새 수치를 동일 실험처럼 직접 합치지 않는다.


## 2. 실험 설계와 알고리즘
시드 **68112026**. 모델은 기존 `qwen2.5:1.5b`, `qwen2.5:0.5b`이며 서버 태그·모델 digest·양자화·플랫폼은 manifest에 저장했다. temperature=0, seed=42, 최대 출력64토큰, 기존 최초 생성 및 검증 프롬프트를 유지했다. 산술·일반 지식·논리·코딩·문장 이해 각60문제이고, 같은 평가를 다섯 번 실행했다. 산술60개는 모든 방식의 공통 기존 계산 도구 경로를 사용해 최초 모델 호출은240회/모델이다. 산술 도구의 정확성을 VRG 자체 성과로 계산하지 않는다.

학습은 과거 **학습 전용**200문제/모델의 원시 최초·수정 답변만 재사용했다. 기존 평가 및 VVP 보정 문제는 학습에 넣지 않았다. 새 보정100문제는 정규화 .01/.1/1과 온도 .75/1/1.5의 log loss 선택에만 썼다. 새 평가300개에서는 학습·임계값 변경을 하지 않았다. 문제 줄기·보기 순서 교체·산술 정규화 기준 중복을 배제했으나 의미적·템플릿 독립까지 보증하지 않는다. 각 유형을 반씩 나눈 A/B 각150개는 방향 재현용으로만 쓰며 외부 독립 데이터셋이 아니다.

모델별 최초 답변은 한 번 생성해 동결했고 모든 방식·반복이 이를 공유했다. 기존 추출기와 이전 VVP 내용 감사 규칙을 **실행 전부터 고정**해 적용했다. 형식 준수는 별도 지표다. 내용이 미해결이면 오답으로 채점하며 자연어의 모든 의미를 이해하는 채점기는 아니다.

- **No Verification:** 최초 답변 그대로 반환.
- **Always Accept:** 미추출 수정안까지 원문 그대로 채택. 이를 fallback해 정답 훼손을 숨기지 않는다.
- **Selective:** 기존 게이트와 미추출 수정안 fallback 유지.
- **VRG-Heuristic:** 수정안 미해결이면 거절, 같은 내용은 허용, 안전한 계산 도구의 정답은 보존, 최초 답변 미해결일 때만 유효한 변경을 허용. 최초의 잘못된 유효 답변도 보존할 수 있는 보수적 규칙이다.
- **VRG-Learned:** 최초+수정안의 관측 가능한 길이·형식·파싱·유형·AST 크기·내용 일치 등으로 복구/훼손/정오 유지의 다항 로지스틱 회귀를 학습한다. 예상 복구확률−훼손확률>0인 변경을 채택하고 무효 후보/도구 정답 보호를 적용한다. 모델의 주장된 확신도를 정답 확률로 사용하지 않는다.
- **Oracle:** Always의 실제 최초/수정안 중 정답을 사후 선택한다. 선택 API와 실행 가능한 방식 목록에 포함하지 않았으며 비용을 0으로 표시하지 않는다.

Guard는 검증 후 판단하므로 두 Guard 모두 매 반복300회 검증을 실제로 실행한다. 같은 최초·도착 순서를 공유하지만 각 실제 방식은 별도로 모델을 호출했다. 무작위 기본 방식 순서를 반복마다 회전해 각 위치에 한 번씩 배치했고 모델 순서는 반복마다 교대했다. 모델 언로드·중립 워밍업과 실행 위치를 기록했다. OS/하드웨어 캐시를 완전히 제거하지는 못했다.


## 3. 실제 평가 결과: 5회 평균
CW는 최초 정답이 최종 오답이 된 수, WC는 반대다. 아래 시간·CPU·토큰은 최초 생성 이후 **판단+검증 단계**이다. Oracle은 품질 상한선이며 비용은 미측정이다.

| 모델 | 방식 | 최초 정확도 | 최종 정확도 | CW/WC | 답변 비율 | 형식 준수 | 검증 호출 | 입력/출력 토큰 | 단계 CPU초 | 단계 벽시계초 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | No Verification | 59.33% | 59.33% | 0/0 | 98.67% | 98.67% | 0.0 | 0.0/0.0 | 0.000 | 0.000 |
| qwen2.5:1.5b | Always Accept | 59.33% | 51.67% | 48/25 | 98.00% | 98.00% | 300.0 | 34755.0/962.0 | 30.352 | 50.897 |
| qwen2.5:1.5b | Selective | 59.33% | 55.67% | 21/10 | 98.67% | 98.67% | 120.0 | 15799.0/240.0 | 9.329 | 17.221 |
| qwen2.5:1.5b | VRG-Heuristic | 59.33% | 59.33% | 0/0 | 98.67% | 98.67% | 300.0 | 34755.0/962.0 | 31.187 | 49.977 |
| qwen2.5:1.5b | VRG-Learned | 59.33% | 59.33% | 0/0 | 98.67% | 98.67% | 300.0 | 34755.0/962.0 | 31.185 | 50.955 |
| qwen2.5:1.5b | Oracle Upper Bound | 59.33% | 67.67% | 0/25 | 미측정 | 미측정 | 미측정 | 미측정/미측정 | 미측정 | 미측정 |
| qwen2.5:0.5b | No Verification | 51.67% | 51.67% | 0/0 | 94.00% | 78.33% | 0.0 | 0.0/0.0 | 0.001 | 0.001 |
| qwen2.5:0.5b | Always Accept | 51.67% | 43.67% | 30/6 | 92.33% | 82.67% | 300.0 | 36084.0/2013.0 | 38.939 | 26.415 |
| qwen2.5:0.5b | Selective | 51.67% | 44.67% | 23/2 | 94.67% | 85.00% | 180.0 | 23725.0/1445.0 | 25.956 | 18.325 |
| qwen2.5:0.5b | VRG-Heuristic | 51.67% | 51.67% | 0/0 | 94.67% | 83.33% | 300.0 | 36084.0/2013.0 | 39.585 | 26.515 |
| qwen2.5:0.5b | VRG-Learned | 51.67% | 50.33% | 6/2 | 94.67% | 87.33% | 300.0 | 36084.0/2013.0 | 39.185 | 26.276 |
| qwen2.5:0.5b | Oracle Upper Bound | 51.67% | 53.67% | 0/6 | 미측정 | 미측정 | 미측정 | 미측정/미측정 | 미측정 | 미측정 |

전체 전이와 분모를 구분한 비율:
| 모델 | 방식 | CC | CW | WC | WW | 전체 중 CW | 초기 정답 중 CW | 전체 중 WC | 초기 오답 중 WC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | No Verification | 178.0 | 0.0 | 0.0 | 122.0 | 0.00% | 0.00% | 0.00% | 0.00% |
| qwen2.5:1.5b | Always Accept | 130.0 | 48.0 | 25.0 | 97.0 | 16.00% | 26.97% | 8.33% | 20.49% |
| qwen2.5:1.5b | Selective | 157.0 | 21.0 | 10.0 | 112.0 | 7.00% | 11.80% | 3.33% | 8.20% |
| qwen2.5:1.5b | VRG-Heuristic | 178.0 | 0.0 | 0.0 | 122.0 | 0.00% | 0.00% | 0.00% | 0.00% |
| qwen2.5:1.5b | VRG-Learned | 178.0 | 0.0 | 0.0 | 122.0 | 0.00% | 0.00% | 0.00% | 0.00% |
| qwen2.5:1.5b | Oracle Upper Bound | 178.0 | 0.0 | 25.0 | 97.0 | 0.00% | 0.00% | 8.33% | 20.49% |
| qwen2.5:0.5b | No Verification | 155.0 | 0.0 | 0.0 | 145.0 | 0.00% | 0.00% | 0.00% | 0.00% |
| qwen2.5:0.5b | Always Accept | 125.0 | 30.0 | 6.0 | 139.0 | 10.00% | 19.35% | 2.00% | 4.14% |
| qwen2.5:0.5b | Selective | 132.0 | 23.0 | 2.0 | 143.0 | 7.67% | 14.84% | 0.67% | 1.38% |
| qwen2.5:0.5b | VRG-Heuristic | 155.0 | 0.0 | 0.0 | 145.0 | 0.00% | 0.00% | 0.00% | 0.00% |
| qwen2.5:0.5b | VRG-Learned | 149.0 | 6.0 | 2.0 | 143.0 | 2.00% | 3.87% | 0.67% | 1.38% |
| qwen2.5:0.5b | Oracle Upper Bound | 155.0 | 0.0 | 6.0 | 139.0 | 0.00% | 0.00% | 2.00% | 4.14% |

## 4. 비용 계측과 학습 비용
CPU는 Ollama 관련 프로세스 누적 CPU 차이와 Python CPU의 합이다. CPU 시간은 GPU 연산량이 아니다. 메모리는 Ollama 프로세스 그룹 RSS를100ms 간격으로 관측한 최대값으로, 연속 진짜 peak나 방식별 독점 점유량이 아니다. 모델이 이미 상주해 있어 No Verification에도 RSS가 관측될 수 있다.

| 모델 | 최초 생성 호출 | 최초 입력/출력 토큰 | 최초 CPU초 | 최초 벽시계초 |
| --- | --- | --- | --- | --- |
| qwen2.5:1.5b | 240 | 23944/492 | 18.576 | 26.852 |
| qwen2.5:0.5b | 240 | 23944/1791 | 25.158 | 17.332 |

최초 생성과 실행 단계의 실측 구성요소 합산:
| 모델 | 방식 | 합산 모델 호출 | 합산 입력/출력 토큰 | 합산 CPU초 | 합산 벽시계초 | Guard CPU초 | 관측 최대 RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | No Verification | 240.0 | 23944.0/492.0 | 18.577 | 26.852 | 0.000000 | 1511.8 |
| qwen2.5:1.5b | Always Accept | 540.0 | 58699.0/1454.0 | 48.928 | 77.749 | 0.003188 | 1610.9 |
| qwen2.5:1.5b | Selective | 360.0 | 39743.0/732.0 | 27.906 | 44.073 | 0.001419 | 1606.2 |
| qwen2.5:1.5b | VRG-Heuristic | 540.0 | 58699.0/1454.0 | 49.763 | 76.829 | 0.012427 | 1598.6 |
| qwen2.5:1.5b | VRG-Learned | 540.0 | 58699.0/1454.0 | 49.761 | 77.807 | 0.150794 | 1567.6 |
| qwen2.5:1.5b | Oracle Upper Bound | 미측정 | 미측정/미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| qwen2.5:0.5b | No Verification | 240.0 | 23944.0/1791.0 | 25.159 | 17.332 | 0.000000 | 796.7 |
| qwen2.5:0.5b | Always Accept | 540.0 | 60028.0/3804.0 | 64.097 | 43.746 | 0.002205 | 857.5 |
| qwen2.5:0.5b | Selective | 420.0 | 47669.0/3236.0 | 51.114 | 35.656 | 0.001522 | 828.1 |
| qwen2.5:0.5b | VRG-Heuristic | 540.0 | 60028.0/3804.0 | 64.743 | 43.846 | 0.007411 | 863.5 |
| qwen2.5:0.5b | VRG-Learned | 540.0 | 60028.0/3804.0 | 64.343 | 43.608 | 0.114163 | 850.0 |
| qwen2.5:0.5b | Oracle Upper Bound | 미측정 | 미측정/미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

이 합계는 동일 최초 생성의 비용을 각 정책에 더한 **구성요소 회계**이며, 매번 최초 생성을 다시 한 독립 end-to-end 실행이 아니다. 최초 생성 변동을 반복 측정한 것으로 해석하지 않는다. 공통 워밍업·언로드·감사 비용은 방식별 표에 포함하지 않았으며 로그에 별도 기록했다.


| 모델 | 기존 학습행 | 신규 보정행 | 새 수집 호출 | 수집 입력/출력 토큰 | 수집 CPU초 | 수집 벽시계초 | 피팅 CPU초 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | 200 | 100 | 180 | 19562/485 | 15.445 | 19.450 | 1.456 |
| qwen2.5:0.5b | 200 | 100 | 180 | 19986/1143 | 17.973 | 11.345 | 1.574 |

기존 학습 출력 수집은 이미 지출한 VVP 비용이며 이번에 다시 생성하지 않았다. 이를 무료 데이터라고 보지 않는다. 과거 전체 최초 학습 생성 CPU를 완전히 별도 계측하지 않았으므로 생애주기 총 CPU 절감을 주장하지 않는다. 새 보정 수집과 피팅은 별도 실제 측정값이다. **GPU FLOPs·전력·금전 비용은 미측정**이다. 서버 duration 값은 보조 원시 로그이며 실제 CPU와 교체하지 않는다.


## 5. 반복 변동과 통계 비교
평균·표본 표준편차·최소/최대·bootstrap95% CI는 VRG_RESULTS.csv 및 VRG_COST_ANALYSIS.csv에 저장했다. 반복 평균 CI는 같은 고정 문제/최초 답변 조건부 변동이며 모집단 정확도의 확실성을 의미하지 않는다. 정확도 차이는 문제별5회 차이를 묶은 paired bootstrap4,000회 및 보수적 전이 하한을 사용했다. CPU 차이는 대응 반복5개 차이의 bootstrap이다. 5개 반복은 꼬리 위험과 시간 드리프트 추정에 작고, 별도 다중 비교 보정을 하지 않았으므로 탐색적 하위 비교만으로 성공을 주장하지 않는다.

| 모델 | Guard | 대조군 | 정확도 차이 pp | 95% CI pp | 보수 하한 pp | CPU 차이초 | CPU 95% CI초 | CW 감소 pp | CW 감소 95% CI pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | heuristic | selective | 3.667 | [0.000, 7.667] | -1.659 | 21.857 | [21.519, 22.197] | 7.000 | [4.000, 10.000] |
| qwen2.5:1.5b | heuristic | always | 7.667 | [2.000, 13.333] | -0.017 | 0.835 | [-0.154, 1.563] | 16.000 | [12.000, 20.333] |
| qwen2.5:1.5b | heuristic | none | 0.000 | [0.000, 0.000] | -1.222 | 31.186 | [30.757, 31.616] | 0.000 | [0.000, 0.000] |
| qwen2.5:1.5b | learned | selective | 3.667 | [0.000, 7.667] | -1.659 | 21.856 | [21.042, 22.702] | 7.000 | [4.000, 10.000] |
| qwen2.5:1.5b | learned | always | 7.667 | [2.000, 13.333] | -0.017 | 0.833 | [0.168, 1.331] | 16.000 | [12.000, 20.333] |
| qwen2.5:1.5b | learned | none | 0.000 | [0.000, 0.000] | -1.222 | 31.185 | [30.280, 32.062] | 0.000 | [0.000, 0.000] |
| qwen2.5:0.5b | heuristic | selective | 7.000 | [4.000, 10.333] | 2.535 | 13.629 | [12.998, 14.160] | 7.667 | [4.667, 11.000] |
| qwen2.5:0.5b | heuristic | always | 8.000 | [4.000, 12.000] | 2.547 | 0.646 | [-0.054, 1.343] | 10.000 | [6.667, 13.667] |
| qwen2.5:0.5b | heuristic | none | 0.000 | [0.000, 0.000] | -1.222 | 39.584 | [38.843, 40.461] | 0.000 | [0.000, 0.000] |
| qwen2.5:0.5b | learned | selective | 5.667 | [3.000, 8.667] | 2.113 | 13.229 | [12.378, 14.077] | 5.667 | [3.000, 8.667] |
| qwen2.5:0.5b | learned | always | 6.667 | [3.333, 10.333] | 1.814 | 0.246 | [-1.257, 1.650] | 8.000 | [5.000, 11.333] |
| qwen2.5:0.5b | learned | none | -1.333 | [-3.333, 0.333] | -4.221 | 39.184 | [38.204, 40.054] | -2.000 | [-3.667, -0.667] |

반복별 변동 요약:
| 모델 | 방식 | 정확도 평균±SD | 단계 CPU 평균±SD초 | CPU 95% CI초 | 단계 벽시계 평균±SD초 |
| --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | none | 59.33% ± 0.00% | 0.000 ± 0.000 | [0.000, 0.001] | 0.000 ± 0.000 |
| qwen2.5:1.5b | always | 51.67% ± 0.00% | 30.352 ± 0.961 | [29.681, 31.183] | 50.897 ± 3.777 |
| qwen2.5:1.5b | selective | 55.67% ± 0.00% | 9.329 ± 0.265 | [9.148, 9.551] | 17.221 ± 2.167 |
| qwen2.5:1.5b | heuristic | 59.33% ± 0.00% | 31.187 ± 0.577 | [30.757, 31.616] | 49.977 ± 4.196 |
| qwen2.5:1.5b | learned | 59.33% ± 0.00% | 31.185 ± 1.149 | [30.280, 32.063] | 50.955 ± 6.816 |
| qwen2.5:1.5b | oracle | 67.67% ± 0.00% | 미측정 ± 미측정 | [미측정, 미측정] | 미측정 ± 미측정 |
| qwen2.5:0.5b | none | 51.67% ± 0.00% | 0.001 ± 0.000 | [0.001, 0.001] | 0.001 ± 0.000 |
| qwen2.5:0.5b | always | 43.67% ± 0.00% | 38.939 ± 1.701 | [37.720, 40.431] | 26.415 ± 1.877 |
| qwen2.5:0.5b | selective | 44.67% ± 0.00% | 25.956 ± 1.026 | [25.165, 26.867] | 18.325 ± 1.921 |
| qwen2.5:0.5b | heuristic | 51.67% ± 0.00% | 39.585 ± 1.047 | [38.844, 40.462] | 26.515 ± 1.921 |
| qwen2.5:0.5b | learned | 50.33% ± 0.00% | 39.185 ± 1.159 | [38.205, 40.054] | 26.276 ± 0.939 |
| qwen2.5:0.5b | oracle | 53.67% ± 0.00% | 미측정 ± 미측정 | [미측정, 미측정] | 미측정 ± 미측정 |

## 6. H1–H5와 성공 판정
H2는 Always 복구의80% 이상 유지라는 사전 지정 설명적 기준을 적용했다. 이는 자체적인 통계적 비열등성 증명은 아니다. 정확도 개선의 불확실성, 비용 증가, 평가 누출, Selective 대비 실질 이점 부재 중 하나라도 있으면 성공으로 보지 않는다. 각 Guard는 두 모델 모두에서 조건을 충족해야 한다.

| 모델 | Guard | Always 대비 복구 유지율 | 사전 기준 | 성공 |
| --- | --- | --- | --- | --- |
| qwen2.5:1.5b | heuristic | 0.00% | {"H1_harm_reduction": true, "H2_repair_retention80": false, "H3_accuracy_strict": false, "H3_CPU_not_increased": false, "H4_net_post_verification_savings": false, "H5_cohorts_direction": true} | False |
| qwen2.5:1.5b | learned | 0.00% | {"H1_harm_reduction": true, "H2_repair_retention80": false, "H3_accuracy_strict": false, "H3_CPU_not_increased": false, "H4_net_post_verification_savings": false, "H5_cohorts_direction": true} | False |
| qwen2.5:0.5b | heuristic | 0.00% | {"H1_harm_reduction": true, "H2_repair_retention80": false, "H3_accuracy_strict": true, "H3_CPU_not_increased": false, "H4_net_post_verification_savings": false, "H5_cohorts_direction": true} | False |
| qwen2.5:0.5b | learned | 33.33% | {"H1_harm_reduction": true, "H2_repair_retention80": false, "H3_accuracy_strict": true, "H3_CPU_not_increased": false, "H4_net_post_verification_savings": false, "H5_cohorts_direction": true} | False |

H4: 두 Guard가 검증 호출을 줄이지 않는 설계이므로 추가 판단 CPU만으로 이미 지출한 검증 비용을 회수할 수 없다. 하위 시스템의 재시도 절감 등 측정하지 않은 미래 이익을 가정하지 않았다. 캐시/순서로 발생한 CPU 차이를 Guard의 연산 절감으로 설명하지 않는다. Heuristic이 처음부터 바꾸지 않을 유효 답변을 검증하는 일은 실제로 불필요할 수 있다. 이를 호출 전 생략으로 바꾸면 기존 verification gating의 문제가 되므로 이번 사후 채택 효과와 혼동하지 않았다.


관측 정확도–CPU 점들의 Pareto 관계(통계적 지배 증명은 아님):
| 모델 | 방식 | 관측 frontier | 지배하는 대안 |
| --- | --- | --- | --- |
| qwen2.5:1.5b | none | True |  |
| qwen2.5:1.5b | always | False | none, selective |
| qwen2.5:1.5b | selective | False | none |
| qwen2.5:1.5b | heuristic | False | none, learned |
| qwen2.5:1.5b | learned | False | none |
| qwen2.5:0.5b | none | True |  |
| qwen2.5:0.5b | always | False | none, selective |
| qwen2.5:0.5b | selective | False | none |
| qwen2.5:0.5b | heuristic | False | none |
| qwen2.5:0.5b | learned | False | none |

## 7. 유형별 결과와 실패 사례
| 모델 | 방식 | 유형 | 최종 정확도 | CW/WC 평균 |
| --- | --- | --- | --- | --- |
| qwen2.5:1.5b | none | arithmetic | 100.00% | 0/0 |
| qwen2.5:1.5b | none | knowledge | 85.00% | 0/0 |
| qwen2.5:1.5b | none | logic | 41.67% | 0/0 |
| qwen2.5:1.5b | none | coding | 28.33% | 0/0 |
| qwen2.5:1.5b | none | comprehension | 41.67% | 0/0 |
| qwen2.5:1.5b | always | arithmetic | 73.33% | 16/0 |
| qwen2.5:1.5b | always | knowledge | 78.33% | 4/0 |
| qwen2.5:1.5b | always | logic | 38.33% | 8/6 |
| qwen2.5:1.5b | always | coding | 13.33% | 13/4 |
| qwen2.5:1.5b | always | comprehension | 55.00% | 7/15 |
| qwen2.5:1.5b | selective | arithmetic | 100.00% | 0/0 |
| qwen2.5:1.5b | selective | knowledge | 85.00% | 0/0 |
| qwen2.5:1.5b | selective | logic | 38.33% | 8/6 |
| qwen2.5:1.5b | selective | coding | 13.33% | 13/4 |
| qwen2.5:1.5b | selective | comprehension | 41.67% | 0/0 |
| qwen2.5:1.5b | heuristic | arithmetic | 100.00% | 0/0 |
| qwen2.5:1.5b | heuristic | knowledge | 85.00% | 0/0 |
| qwen2.5:1.5b | heuristic | logic | 41.67% | 0/0 |
| qwen2.5:1.5b | heuristic | coding | 28.33% | 0/0 |
| qwen2.5:1.5b | heuristic | comprehension | 41.67% | 0/0 |
| qwen2.5:1.5b | learned | arithmetic | 100.00% | 0/0 |
| qwen2.5:1.5b | learned | knowledge | 85.00% | 0/0 |
| qwen2.5:1.5b | learned | logic | 41.67% | 0/0 |
| qwen2.5:1.5b | learned | coding | 28.33% | 0/0 |
| qwen2.5:1.5b | learned | comprehension | 41.67% | 0/0 |
| qwen2.5:1.5b | oracle | arithmetic | 100.00% | 0/0 |
| qwen2.5:1.5b | oracle | knowledge | 85.00% | 0/0 |
| qwen2.5:1.5b | oracle | logic | 51.67% | 0/6 |
| qwen2.5:1.5b | oracle | coding | 35.00% | 0/4 |
| qwen2.5:1.5b | oracle | comprehension | 66.67% | 0/15 |
| qwen2.5:0.5b | none | arithmetic | 100.00% | 0/0 |
| qwen2.5:0.5b | none | knowledge | 73.33% | 0/0 |
| qwen2.5:0.5b | none | logic | 20.00% | 0/0 |
| qwen2.5:0.5b | none | coding | 38.33% | 0/0 |
| qwen2.5:0.5b | none | comprehension | 26.67% | 0/0 |
| qwen2.5:0.5b | always | arithmetic | 98.33% | 1/0 |
| qwen2.5:0.5b | always | knowledge | 36.67% | 22/0 |
| qwen2.5:0.5b | always | logic | 20.00% | 0/0 |
| qwen2.5:0.5b | always | coding | 31.67% | 6/2 |
| qwen2.5:0.5b | always | comprehension | 31.67% | 1/4 |
| qwen2.5:0.5b | selective | arithmetic | 100.00% | 0/0 |
| qwen2.5:0.5b | selective | knowledge | 45.00% | 17/0 |
| qwen2.5:0.5b | selective | logic | 20.00% | 0/0 |
| qwen2.5:0.5b | selective | coding | 31.67% | 6/2 |
| qwen2.5:0.5b | selective | comprehension | 26.67% | 0/0 |
| qwen2.5:0.5b | heuristic | arithmetic | 100.00% | 0/0 |
| qwen2.5:0.5b | heuristic | knowledge | 73.33% | 0/0 |
| qwen2.5:0.5b | heuristic | logic | 20.00% | 0/0 |
| qwen2.5:0.5b | heuristic | coding | 38.33% | 0/0 |
| qwen2.5:0.5b | heuristic | comprehension | 26.67% | 0/0 |
| qwen2.5:0.5b | learned | arithmetic | 100.00% | 0/0 |
| qwen2.5:0.5b | learned | knowledge | 73.33% | 0/0 |
| qwen2.5:0.5b | learned | logic | 20.00% | 0/0 |
| qwen2.5:0.5b | learned | coding | 31.67% | 6/2 |
| qwen2.5:0.5b | learned | comprehension | 26.67% | 0/0 |
| qwen2.5:0.5b | oracle | arithmetic | 100.00% | 0/0 |
| qwen2.5:0.5b | oracle | knowledge | 73.33% | 0/0 |
| qwen2.5:0.5b | oracle | logic | 20.00% | 0/0 |
| qwen2.5:0.5b | oracle | coding | 41.67% | 0/2 |
| qwen2.5:0.5b | oracle | comprehension | 33.33% | 0/4 |

계산 도구를 제외하고 AI 모델이 직접 최초 답변을 생성한 **240문제**의 사후 하위 분석(별도 성공 기준이나 재학습에 쓰지 않음):
| 모델 | 방식 | 초기 정확도 | 최종 정확도 | CW/WC 평균 |
| --- | --- | --- | --- | --- |
| qwen2.5:1.5b | none | 49.17% | 49.17% | 0/0 |
| qwen2.5:1.5b | always | 49.17% | 46.25% | 32/25 |
| qwen2.5:1.5b | selective | 49.17% | 44.58% | 21/10 |
| qwen2.5:1.5b | heuristic | 49.17% | 49.17% | 0/0 |
| qwen2.5:1.5b | learned | 49.17% | 49.17% | 0/0 |
| qwen2.5:1.5b | oracle | 49.17% | 59.58% | 0/25 |
| qwen2.5:0.5b | none | 39.58% | 39.58% | 0/0 |
| qwen2.5:0.5b | always | 39.58% | 30.00% | 29/6 |
| qwen2.5:0.5b | selective | 39.58% | 30.83% | 23/2 |
| qwen2.5:0.5b | heuristic | 39.58% | 39.58% | 0/0 |
| qwen2.5:0.5b | learned | 39.58% | 37.92% | 6/2 |
| qwen2.5:0.5b | oracle | 39.58% | 42.08% | 0/6 |

첫 반복의 사후 설명용 사례(선택/학습에 사용하지 않음):
| 모델 | Guard | 사건 | ID | 유형 | 질문 | 정답 | 최초 | 수정 | 최종 | 판정 이유 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:1.5b | heuristic | 복구 거절 | f3e3635d9209 | comprehension | Read the passage: Sensor 9111 uses channel red except during maintenance when it uses blue. Maintenance is active. Which channel is used? A) Red B) No channel C) Green D) Blue | D | B | D | B | preserve_resolved_initial |
| qwen2.5:1.5b | heuristic | 복구 거절 | 717d1c0e04bf | comprehension | Read the passage: Sensor 9101 uses channel red except during maintenance when it uses blue. Maintenance is active. Which channel is used? A) Green B) No channel C) Red D) Blue | D | B | D | B | preserve_resolved_initial |
| qwen2.5:1.5b | heuristic | 오답 유지 | c4f2e6262e2f | comprehension | Read the passage: Ticket 9112 moved from open to pending, then closed. A later note says it reopened. What is the latest ticket state? A) Deleted B) Closed C) Open D) Pending | C | B | B | B | same_content |
| qwen2.5:1.5b | heuristic | 오답 유지 | 10f93bb7f19b | comprehension | Read the passage: Ticket 9115 moved from open to pending, then closed. A later note says it reopened. What is the latest ticket state? A) Deleted B) Closed C) Pending D) Open | D | B | B | B | same_content |
| qwen2.5:1.5b | learned | 복구 거절 | f3e3635d9209 | comprehension | Read the passage: Sensor 9111 uses channel red except during maintenance when it uses blue. Maintenance is active. Which channel is used? A) Red B) No channel C) Green D) Blue | D | B | D | B | preserve_resolved_initial |
| qwen2.5:1.5b | learned | 복구 거절 | 717d1c0e04bf | comprehension | Read the passage: Sensor 9101 uses channel red except during maintenance when it uses blue. Maintenance is active. Which channel is used? A) Green B) No channel C) Red D) Blue | D | B | D | B | preserve_resolved_initial |
| qwen2.5:1.5b | learned | 오답 유지 | c4f2e6262e2f | comprehension | Read the passage: Ticket 9112 moved from open to pending, then closed. A later note says it reopened. What is the latest ticket state? A) Deleted B) Closed C) Open D) Pending | C | B | B | B | positive_expected_gain |
| qwen2.5:1.5b | learned | 오답 유지 | 10f93bb7f19b | comprehension | Read the passage: Ticket 9115 moved from open to pending, then closed. A later note says it reopened. What is the latest ticket state? A) Deleted B) Closed C) Pending D) Open | D | B | B | B | positive_expected_gain |
| qwen2.5:0.5b | heuristic | 복구 거절 | 8641ea0e1e7f | comprehension | Read the passage: Sensor 9102 uses channel red except during maintenance when it uses blue. Maintenance is active. Which channel is used? A) No channel B) Blue C) Red D) Green | B | A | B | A | preserve_resolved_initial |
| qwen2.5:0.5b | heuristic | 복구 거절 | c0377bb429e1 | comprehension | Read the passage: Sensor 9104 uses channel red except during maintenance when it uses blue. Maintenance is active. Which channel is used? A) No channel B) Blue C) Red D) Green | B | A | B | A | preserve_resolved_initial |
| qwen2.5:0.5b | heuristic | 오답 유지 | 269a1e6f0bb3 | comprehension | Read the passage: Ticket 9106 moved from open to pending, then closed. A later note says it reopened. What is the latest ticket state? A) Closed B) Open C) Pending D) Deleted | B | A | A | A | same_content |
| qwen2.5:0.5b | heuristic | 오답 유지 | 81a6937ffde1 | logic | Reason about integer constraints: among integers from 9109 through 9120, including both ends, how many are divisible by 3? A) 6 B) 3 C) 5 D) 4 | D | None | None | None | revision_unresolved |
| qwen2.5:0.5b | learned | 정답 훼손 | d6582d9acae9 | coding | What does this Python program output? ```python d = {'a': 407, 'b': 7} x = d['b'] del d['b'] print(x + len(d)) ``` A) 414 B) 8 C) Error D) 7 | B | B | A | A | positive_expected_gain |
| qwen2.5:0.5b | learned | 정답 훼손 | 723daaed13ad | coding | What does this Python program output? ```python print(sum(x % 4 for x in range(403, 408))) ``` A) 9 B) 0 C) 10 D) Error | A | A | D | D | positive_expected_gain |
| qwen2.5:0.5b | learned | 복구 거절 | 8641ea0e1e7f | comprehension | Read the passage: Sensor 9102 uses channel red except during maintenance when it uses blue. Maintenance is active. Which channel is used? A) No channel B) Blue C) Red D) Green | B | A | B | A | preserve_resolved_initial |
| qwen2.5:0.5b | learned | 복구 거절 | c0377bb429e1 | comprehension | Read the passage: Sensor 9104 uses channel red except during maintenance when it uses blue. Maintenance is active. Which channel is used? A) No channel B) Blue C) Red D) Green | B | A | B | A | preserve_resolved_initial |
| qwen2.5:0.5b | learned | 오답 유지 | 269a1e6f0bb3 | comprehension | Read the passage: Ticket 9106 moved from open to pending, then closed. A later note says it reopened. What is the latest ticket state? A) Closed B) Open C) Pending D) Deleted | B | A | A | A | positive_expected_gain |
| qwen2.5:0.5b | learned | 오답 유지 | 81a6937ffde1 | logic | Reason about integer constraints: among integers from 9109 through 9120, including both ends, how many are divisible by 3? A) 6 B) 3 C) 5 D) 4 | D | None | None | None | revision_unresolved |

동일 검증 요청 600개 그룹 중 원문이 달라진 그룹 0개, 추출 답이 달라진 그룹 0개다. 동일 candidate에 두 Guard를 다시 적용한 `VRG_IDENTICAL_CANDIDATE_DIAGNOSTIC.csv`는 이 변동을 분리하는 사후 진단이며 실제 새 실행·추가 비용 절감 수치가 아니다.


동일 수정안 진단(사후·반복 평균):
| 모델 | Guard | 막은 CW | 포기한 WC | 가상 최종 정확도 |
| --- | --- | --- | --- | --- |
| qwen2.5:1.5b | heuristic | 48.0 | 25.0 | 59.33% |
| qwen2.5:1.5b | learned | 48.0 | 25.0 | 59.33% |
| qwen2.5:0.5b | heuristic | 30.0 | 6.0 | 51.67% |
| qwen2.5:0.5b | learned | 24.0 | 4.0 | 50.33% |

## 8. 한계와 기존 연구와의 관계
Heuristic은 유효한 최초 오답도 보존하므로 정답 훼손을 줄이는 대신 복구를 포기할 수 있다. Learned는 과거 학습200문제의 드문 복구/훼손 사건으로 새 수정안의 유용성을 추정해 데이터 이동과 과신 위험이 있다. 같은 소형 모델을 생성과 검증에 사용하므로 오류가 상관될 수 있고 외부 증거는 없다. 이러한 설명은 관측과 설계상의 한계이며 내부 추론 원인을 입증한 것이 아니다.

평가는 합성 단답/선택형이며 장문·한국어 실서비스·다른 모델 계열 검증은 하지 않았다. 지식은 HTTP/Python/HTML에 편중됐고 생성 템플릿 일부가 보정/평가에서 공유된다. 정확도는 고정 내용 추출 규칙에 의존한다. 신규 형식의 명확한 자연어 답변을 여전히 놓칠 수 있으며 평가 뒤 편리한 채점 규칙을 추가하지 않았다. 반복 출력은 독립적인 새 문제도 아니고 최초 생성의 무작위성을 포괄하지 않는다. 노트북을 전용 자원으로 완전히 격리하지 않았으며 CPU 계측은 요청별 GPU 작업을 추적하는 계측이 아니다.

[Learning to Defer (Mozannar & Sontag, 2020)](https://proceedings.mlr.press/v119/mozannar20b.html)는 예측을 다른 의사결정자에게 위임할지 학습하는 기존 틀이다. [Selective Classification (Geifman & El-Yaniv, 2017)](https://arxiv.org/abs/1705.08500)는 위험과 답변 범위의 교환을 다룬다. VRG는 요청 자체를 거절하기보다 이미 생성한 두 후보 중 하나를 선택하는 적용 형태이지만 독립적인 새 기반 기술이라는 근거는 아니다.

[Self-Refine (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651)는 자기 피드백을 통한 반복 개선을 연구했고, [Huang et al. (ICLR2024)](https://arxiv.org/abs/2310.01798)는 외부 피드백 없는 자기 교정에서 성능 저하도 관찰했다. 작업과 모델 조건이 다르므로 어느 논문도 이번 로컬 결과를 직접 보증하거나 모든 자기 검증의 무용성을 증명하지 않는다.

프로젝트 기여는 전이별 진단, gold-blind 후보 채택, 기존 선택형 검증과의 실제 호출·CPU·토큰 비교다. verification gating, selective prediction, learning-to-defer, 경량 확률 분류 자체의 발명을 주장하지 않는다.


## 9. 최종 질문의 답
1. **정확도를 개선했는가?** 최초 대비 관측 변화는 qwen2.5:1.5b heuristic 59.33%→59.33%; qwen2.5:1.5b learned 59.33%→59.33%; qwen2.5:0.5b heuristic 51.67%→51.67%; qwen2.5:0.5b learned 51.67%→50.33%다. No Verification 대비 이익과 해로운 Always/Selective 대비 이익을 구분한다. 사전 기준의 일반화 성공은 입증하지 못했다.
2. **정답 파괴를 줄였는가?** 반복당 평균 CW는 qwen2.5:1.5b: always 48.0건, selective 21.0건, heuristic 0.0건, learned 0.0건; qwen2.5:0.5b: always 30.0건, selective 23.0건, heuristic 0.0건, learned 6.0건다. 따라서 관측상 훼손 억제와 복구 희생을 동시에 확인해야 하며 H1의 신뢰구간 및 H2를 함께 보고한다.
3. **전체 추론 비용이 줄었는가?** 감소를 입증하지 못했다. 검증 후 Guard는 검증 호출을 생략하지 않았고, 이번 두 모델 모두 Selective 대비 실측 단계 CPU와 검증 토큰이 증가했다. GPU·전력·생애주기 전체 연산량 감소는 별도로도 입증하지 않았다.
4. **Selective보다 우수한가?** 이번 정확도–비용 균형의 전체 판정은 아니다. 정확도만의 관측 우위와 비용 균형 우위는 구분한다. 두 모델의 정확도·CPU·복구 유지·재현 조건을 모두 통과하지 못했다.
5. **기존 기법과 차이?** 검증 호출 전 게이팅과 달리 수정안 생성 후 원답/수정안을 선택한다. 이미 발생한 비용을 줄일 수 없는 대신 정답 훼손을 직접 제어한다. 기존 선택/위임 기법의 적용 연구다.
6. **독립적인 새로운 기술인가?** 이를 주장할 이론·광범위 비교·독립 재현 근거는 없다.
7. **가장 먼저 해결할 문제?** 추가 검증 없이도 원답을 보존하는 단순 기준을 넘어, 실제 틀린 유효 원답을 복구할 근거를 확보해야 한다. 외부 증거가 있는 검증과 충분한 복구/훼손 학습 사례를 확보한 뒤 미사용 외부 업무 집합에서 사전 등록해 검증하는 것이 우선이다. 추가 연구는 이번에 실행하지 않았다.


## 10. 보존·감사·재현
기존 및 실행 전 파일 **494개 해시 유지**, 새 보정/평가와 과거 줄기 중복0, 학습/보정/평가 분리, 모든 최초 답변·프롬프트·Guard 판정·채점 재구성 통과. API 오류 0회. 정답은 실행 바깥의 채점과 Oracle에만 사용했고 Guard API는 정답 키를 전달하면 오류로 거부한다. 확률 학습용 과거 라벨은 학습 자료 안에서만 허용했다.

자동 테스트: **199개 통과 (64.654초, 실험 전 데이터 스냅샷의 전체 unittest)**. 모의 출력은 테스트의 임시 폴더에만 생성했으며 이 보고서/CSV/그래프는 실제 호출 로그에서 생성했다. 신규 문제 생성 검사는 이미 사용한 질문을 거부하므로, 최종 전체 테스트는 기록된 실험 전 데이터의 읽기 전용 링크를 둔 임시 스냅샷에서 실행했다. 실제 실험 데이터를 지우거나 중복 방지 장치를 끄지 않았다.

명령어(프로젝트 루트):

```bash
.venv/bin/python -B -u -m src.vrg_experiment
.venv/bin/python -B -m src.vrg_analysis data/vrg_6828c214e04841e1ab6ca4e5392de9a9 data/vrg_6828c214e04841e1ab6ca4e5392de9a9_reanalysis
.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json
.venv/bin/python -B -m src.vrg_report data/vrg_6828c214e04841e1ab6ca4e5392de9a9 "<실제 테스트 결과>" data/vrg_6828c214e04841e1ab6ca4e5392de9a9_reanalysis
```

위 첫 명령은 신규 실행용이며 이미 저장된 같은 신규 문제 줄기가 있으면 중복 방지를 위해 중단한다. 기존 실행 재현의 입력은 manifest·profiles·initial_cache·고정 seed에 보존했다. 분석/그래프/보고서는 덮어쓰기를 거부하므로 동일 출력에 다시 실행하지 않는다. 오프라인 감사는 모델을 새 호출하지 않는다. 그래프에는 matplotlib가 필요하며 기존 임시 설치를 사용했다.

[결과](VRG_RESULTS.csv) · [오류 전이](VRG_ERROR_TRANSITIONS.csv) · [비용 분석](VRG_COST_ANALYSIS.csv) · [통계 비교](VRG_COMPARISONS.csv) · [H1–H5](VRG_HYPOTHESES.csv) · [감사](VRG_AUDIT.json) · [그래프](VRG_COMPARISON.png)

원시 실행: `../vrg_6828c214e04841e1ab6ca4e5392de9a9_outcomes.csv`, `../vrg_6828c214e04841e1ab6ca4e5392de9a9_calls.csv`, `../vrg_6828c214e04841e1ab6ca4e5392de9a9_repeat_summary.csv`, `../vrg_6828c214e04841e1ab6ca4e5392de9a9_resources.csv`. 선택 프로필·실행 순서·예전 파일 해시는 `../vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json`에 있다.

지식 기준 출처: [IANA HTTP registry](https://www.iana.org/assignments/http-status-codes), [Python built-ins](https://docs.python.org/3/library/functions.html), [Python exceptions](https://docs.python.org/3/library/exceptions.html), [WHATWG attributes](https://html.spec.whatwg.org/multipage/indices.html#attributes-3). 코드 출력은 독립 Python 실행, 산술은 별도 기준 계산으로 검사했다. 나머지 수작업 의미 기준의 무오류를 자동 검사만으로 보증하지 않는다.

