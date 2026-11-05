"""Render the complete, unselected VVP study from audited measured artifacts."""
import csv,json,statistics,sys
from pathlib import Path
if __package__:
    from .controlled_analysis import read_decoded
else:
    from controlled_analysis import read_decoded

NAMES={'none':'No Verification','full':'Full Verification','selective':'Selective','rcav':'RCAV','improved':'Improved RCAV','vvp':'VVP'}
def number(x,d=3):return '미측정' if x is None else f'{x:.{d}f}'
def percent(x):return '미측정' if x is None else f'{100*x:.2f}%'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])

def render(prefix,test_result):
    p=Path(prefix).resolve();root=p.parent.parent
    def read(s):return read_decoded(f'{p}_{s}.csv')
    def js(s):return json.loads(Path(f'{p}_{s}.json').read_text())
    meta,audit,manifest=js('metadata'),js('audit'),js('manifest');agg=read('aggregate');acc=read('accounting')
    def link(s):return f'[{s}](data/{p.name}_{s}.csv)'
    chosen=[r for r in agg if (r['method']=='vvp' and r['budget_percent']==25) or (r['method']=='none' and r['budget_percent']==0) or (r['method'] in ('full','selective','rcav','improved') and r['budget_percent']==100)]
    out=['# Verification Value Predictor 연구 보고서',
        '\n## 결론\nVVP의 성공 조건을 충족하지 못했다. 사전에 지정한 25% 예산에서 두 모델 모두 검증하지 않는 방식보다 정답을 늘리지 못했다. 1.5B에서는 모든 예산에서 검증을 0회 선택했고, 0.5B에서는 검증을 수행했지만 순정답 개선이 없었다. 호출 감소를 새로운 알고리즘의 우월성으로 해석하지 않는다.',
        f'\n실험 식별자: `{p.name}`. 실제 로컬 Ollama 실행이며 유료 API를 사용하지 않았다. 평가 문제 300개, 모델 {len(meta["models"])}개, 6방식 × 6예산 × 5반복 = 360조건이다. 최종 검증 호출 {meta["actual_verification_calls"]:,}회, 최초 생성 {meta["actual_primary_calls"]}회, 학습·보정 수집 {meta["training_collection_calls"]:,}회, 결과 행 {meta["outcome_rows"]:,}개를 기록했다. 워밍업 호출은 별도 파일에 있다.',
        '\n## 1. 설계 및 정보 경계\n'+
        '방법과 주 판정 예산 25%는 평가 검증 전에 고정했다. 나머지 0/10/50/75/100%는 보조 분석이며 좋은 예산만 골라 성공을 선언하지 않는다. 검증 호출 상한과 공통 CPU 비용 예측에 기반한 예산을 동시에 적용한다. CPU 예산은 예측치이므로 실제 CPU의 강제 상한은 아니다. Full의 100%만 전수 검증이며, 낮은 예산에서는 무작위 도착 순서의 부분집합이다. Selective/RCAV/Improved의 기존 게이트는 유지하고 공통 예산 제한만 덧붙였다.\n\n'+
        '기존 평가 문제를 학습에 전용하지 않았다. 전용 학습 문제 200개/모델, 새 보정 문제 100개, 새 최종 평가 300개를 분리했다. 평가 문제는 산술·일반 지식·논리·코딩·문장 이해 각 60개이며 사전 고정 A/B 집단 각 150개로 나눴다. A/B는 동일 생성 규칙의 부분집합으로 독립 외부 데이터셋이 아니다. 문제의 정답 및 실제 검증 결과는 예측/선택 인터페이스에 전달되지 않는다. 정확도 계산과 Full100 대조는 실행 뒤에만 수행한다.\n\n'+
        '최초 답변은 모델별 한 번 생성해 모든 조건에서 그대로 사용했다. 산술 60개는 기존 안전한 계산 도구 경로를 공통으로 유지하므로 최초 모델 호출은 240회/모델이다. VVP에만 도구 이점을 주지 않았다. 동일 모델·검증 프롬프트·temperature=0·seed=42·최대 출력 64토큰을 사용했다. 반복별 조건 및 도착 순서는 무작위화했고 모델 언로드와 중립 워밍업을 기록했다. 모델 순서는 1.5B 다음 0.5B로 고정되어 두 모델의 속도 차이는 순서와 완전히 분리되지 않는다. OS/하드웨어 캐시 전체를 제거했다고 주장하지 않는다.',
        '\n## 2. VVP 알고리즘\n'+
        '다항 로지스틱 회귀로 P(오답→정답), P(정답→오답), 나머지 상태를 추정한다. 최초 관측 가능한 유형·형식·길이·숫자 수·AST 크기·추출/도구 여부만 사용한다. 학습 정답은 전이 라벨 생성에만 사용한다. 기대 가치 EV=P(오답→정답)−P(정답→오답)를 예상 CPU 시간으로 나눠 양의 가치 항목을 우선 배정한다. 이는 탐욕적 예산 배정이며 최적 배낭 해법이 아니다. 안전한 도구의 증명된 결과는 개선 확률을 0으로 제한한다.\n\n'+
        '검증 CPU 비용은 실제 Ollama 프로세스와 Python CPU 계측에 맞춘 로그 회귀다. 정규화 강도·온도는 보정 집합의 log loss로 선택하고 CPU 규모는 보정 집합의 실제/예측 비율 중앙값으로 맞췄다. 평가에서는 재학습·재보정·임계값 변경을 하지 않았다. 온라인 예측을 위한 추가 AI 호출은 없으며 실제 계획 계산 CPU와 벽시계 시간을 모두 포함했다. 비교군은 공통 예산용 비용 추정만 계산하며 VVP의 분류 비용을 대신 부담하지 않는다.',
        '\n## 3. 주 비교: 5회 평균\n전체 정확도와 답변 형식 준수율을 별도로 계산한다. 아래 CPU/시간/토큰은 **판단+검증 단계**의 실제 측정값이다. RSS는 Ollama 관련 프로세스 RSS를 100ms 간격으로 관측한 최대값의 반복 평균이며, 정확한 메모리 최고점이나 방식의 독점 메모리 사용량이 아니다. 모든 조건은 문제 300개 기준이다.\n']
    out.append(table(['모델','방식/예산','정확도','형식 준수','수정/훼손','놓친 개선','검증 호출','입력/출력 토큰','검증 단계 CPU초','검증 단계 벽시계초','관측 RSS MiB'],[
        [r['model'],NAMES[r['method']]+'/'+str(r['budget_percent'])+'%',percent(r['accuracy_mean']),percent(r['format_compliance_mean']),f'{r["repairs_mean"]:g}/{r["damage_mean"]:g}',number(r['missed_repairs_full_reference_mean'],1),number(r['verification_calls_mean'],1),f'{r["input_tokens_mean"]:g}/{r["output_tokens_mean"]:g}',number(r['measured_cpu_seconds_mean']),number(r['verification_wall_seconds_mean']),number(r['sampled_max_rss_mib_mean'],1)] for r in chosen]))
    out.append('\n호출수 0인 방식도 Python 판단 비용이 있다. “놓친 개선”은 같은 최초 답변에 대한 별도 실행 Full100 결과를 기준으로 세며 동시 실행의 완전한 반사실은 아니다. 정답 수 불변과 답변 문자열 불변도 구분해 원본 전이 CSV에 저장했다.\n')
    out.append('\n## 4. 최초 생성과 전체 비용의 합산\n최초 생성은 공유 캐시 구축 시 한 번 실제 측정했다. 아래 전체 값은 **실측 구성요소의 합**이며 각 방식을 독립적으로 처음부터 끝까지 실행한 지연 측정이 아니다. 반복 간 같은 최초 비용을 더해도 최초 생성의 독립적 변동을 추정할 수 없다. 학습 비용은 이 표에 상각하지 않았다.\n')
    out.append(table(['모델','최초 모델 호출','입력/출력 토큰','최초 CPU초(서버+Python)','최초 벽시계초'],[
        [r['model'],r['model_calls'],f'{r["input_tokens"]}/{r["output_tokens"]}',number(r['cpu_seconds']+r['router_cpu_seconds'] if r['cpu_seconds'] is not None else None),number(r['stage_wall_seconds'])] for r in read('primary_cost')]))
    selected={(r['model'],r['method'],r['budget_percent']) for r in chosen}
    out.append('\n'+table(['모델','방식/예산','합산 모델 호출','합산 CPU초','합산 전체 벽시계초'],[
        [r['model'],NAMES[r['method']]+'/'+str(r['budget_percent'])+'%',number(r['accounted_model_calls'],1),number(r['accounted_full_cpu']),number(r['accounted_full_wall'])] for r in acc if (r['model'],r['method'],r['budget_percent']) in selected]))
    out.append('\n학습 수집 및 피팅 실제 비용:\n'+table(['모델','수집 모델 호출','수집 입력/출력 토큰','수집 벽시계초','피팅 CPU초'],[[r['model'],r['model_calls'],f'{r["input_tokens"]}/{r["output_tokens"]}',number(r['collection_wall_seconds']),number(r['fit_cpu_seconds'])] for r in read('training_resources')]))
    out.append('\n학습 수집에는 검증별 CPU가 저장되어 있으나 최초 학습 답변 생성 전체의 Ollama CPU 합계는 별도로 완전히 계측하지 않았다. 공통 워밍업·모델 언로드·감사용 사전 계획 생성은 위 방식별 합산에 포함하지 않는다. 실제 선택 비용은 실행 시 다시 측정해 포함했다. 따라서 최초 도입·학습·콜드 스타트까지 포함한 생애주기 순 CPU 절감은 입증하지 못했다. 전력·GPU FLOPs·금전 비용은 모두 **미측정**이며 예측 CPU는 실제 비용과 구분한다. 서버 API의 자체 duration은 보조 원시 로그이며 CPU 비용 예측의 정답값으로 쓰지 않았다.\n')
    out.append('\n## 5. 불확실성 및 성공 판정\n정확도는 동일 문제의 반복을 묶은 paired bootstrap 4,000회와 보수적 정답 전이 하한을 사용한다. 같은 답변을 5회 채점한 것을 서로 다른 1,500문제로 보지 않는다. 비용 CI는 대응 반복 차이의 bootstrap이며 반복 5개만으로 꼬리·시간 드리프트를 충분히 추정하기 어렵다. 아래 정확도 차이는 퍼센트포인트다.\n')
    out.append(table(['모델','비교','정확도 차이 pp','paired 95% CI pp','보수적 하한 pp','CPU 차이 초','CPU 차이 95% CI 초'],[
        [r['model'],r['comparison'],number(100*r['accuracy_delta']),f'[{100*r["paired_bootstrap_low"]:.3f}, {100*r["paired_bootstrap_high"]:.3f}]',number(100*r['conservative_exact_lower95']),number(r['cpu_delta_mean']),f'[{number(r["cpu_delta_ci95_low"])}, {number(r["cpu_delta_ci95_high"])}]'] for r in read('comparisons')]))
    out.append('\n'+table(['모델','최종 성공','사전 지정 조건'],[[r['model'],r['success'],json.dumps(r['criteria'],ensure_ascii=False)] for r in read('verdict')]))
    out.append('\n정확도 차이가 관측상 0이어도 일반화 비열등성이 증명된 것은 아니다. 평가 점들의 Pareto frontier는 관측 그리드에 한정하며 통계적 우월성이나 선택되지 않은 예산까지 보장하지 않는다.\n')
    out.append('\n## 6. 전 예산·유형별 결과\n'+table(['모델','방식','예산','정확도 평균','정확도 SD','95% CI','호출 평균','CPU 평균초','CPU SD','CPU 95% CI'],[
        [r['model'],NAMES[r['method']],str(r['budget_percent'])+'%',percent(r['accuracy_mean']),percent(r['accuracy_sample_std']),f'[{percent(r["accuracy_ci95_low"])}, {percent(r["accuracy_ci95_high"])}]',number(r['verification_calls_mean'],1),number(r['measured_cpu_seconds_mean']),number(r['measured_cpu_seconds_sample_std']),f'[{number(r["measured_cpu_seconds_ci95_low"])}, {number(r["measured_cpu_seconds_ci95_high"])}]'] for r in agg]))
    out.append('\n'+table(['모델','방식/예산','유형','정확도','수정/훼손 평균'],[[r['model'],NAMES[r['method']]+'/'+str(r['budget_percent'])+'%',r['category'],percent(r['accuracy']),f'{r["repairs_per_repeat"]:g}/{r["damage_per_repeat"]:g}'] for r in read('category_summary') if (r['model'],r['method'],r['budget_percent']) in selected]))
    out.append('\n## 7. 予測・실패 분석\n아래 AUC/Brier는 Full100의 실제 전이를 사후 평가한 값이다. 도구를 포함한 전체 점수와 도구를 제외한 점수를 모두 제시한다. 전체 AUC가 높아도 이미 안전한 도구 출력을 구분한 효과일 수 있고, 예산 내 정답 개선을 의미하지 않는다.\n')
    training=read('training');distribution=[]
    for m in meta['models']:
        for phase in ('training','validation'):
            rr=[r for r in training if r['model']==m and r['phase']==phase]
            distribution.append([m,phase,len(rr),sum(r['label']==0 for r in rr),sum(r['label']==1 for r in rr),sum(r['label']==2 for r in rr)])
    out.append(table(['모델','역할','문제 수','복구 라벨','훼손 라벨','정오 유지'],distribution))
    out.append('\n1.5B의 학습·보정 자료에는 복구보다 훼손이 많았고, 두 모델 모두 복구 사건 자체가 적었다. 이런 자료에서 복잡한 조건별 이익을 안정적으로 추정하기 어렵다. 이는 평가 뒤 확인한 실패 설명이며 평가 정답을 이용해 판정 기준을 바꾸지 않았다.\n')
    metrics=read('predictive_metrics');groups={}
    for r in metrics:groups.setdefault((r['model'],r['scope'],r['event']),[]).append(r)
    out.append(table(['모델','대상','사건','AUC 평균','Brier 평균','예측 확률 평균','관측 비율'],[[*key,number(statistics.mean(r['roc_auc'] for r in rr)) if all(r['roc_auc'] is not None for r in rr) else '산출 불가',number(statistics.mean(r['brier'] for r in rr)),percent(statistics.mean(r['predicted_rate'] for r in rr)),percent(statistics.mean(r['observed_full_reference_rate'] for r in rr))] for key,rr in groups.items()]))
    variations=read('verifier_variation');out.append(f'\n동일 요청의 실제 재검증 {len(variations)}개 그룹 중 추출 답이 달라진 그룹 {sum(r["distinct_predictions"]>1 for r in variations)}개, 원문이 달라진 그룹 {sum(r["distinct_raw_responses"]>1 for r in variations)}개. 결정적 설정도 독립 샘플이나 새로운 질문을 뜻하지 않는다.\n')
    out.append('\n예산 10%의 0.5B 결과 등 일부 보조 조건이 다른 수치를 보이더라도 주 예산 25%의 실패를 대체하지 않는다. 검증 효과는 유형·프롬프트·모델에 따라 양방향이고 최초 형식이 올바르다고 내용이 맞는 것은 아니다. 새 평가의 지식은 HTTP/HTML/Python/SI에 치우친 합성 문제이며 템플릿과 생성 규칙이 보정 집합과 공유된다. 문자열·구조 중복 배제는 의미적 완전 독립을 증명하지 않는다. 장문·실제 사용자 업무·다른 아키텍처에 일반화할 근거는 없다.\n')
    out.append('\n## 8. 기존 실험의 진단\n기존 Controlled 실험의 동일 최초 답변과 전수 검증 로그를 사용해 다음 전이를 집계했다. 이는 진단 전용이며 VVP 학습에 넣지 않았다. 아래 수는 5반복 합계이므로 독립 문제 수로 해석하지 않는다. 기존 실험의 호출·토큰·CPU·처리 시간은 원본 '+ '[reporting_summary](data/controlled_1ee08ac18eaa428f9644ef66daa2fcaa_reporting_summary.csv)'+'에 보존되어 있다.\n')
    out.append(table(['모델','집합','방식','반복 행','오답→정답','정답→오답','검증 후 정오 유지','문자열/추출답 변경·정오 유지','놓친 개선'],[[r['model'],r['dataset'],r['method'],r['repeated_rows'],r['repairs'],r['damage'],r['verified_same_correctness'],r['changed_answer_same_correctness'],r['missed_repairs_paired_full_reference']] for r in read('historical_diagnosis')]))
    old=read_decoded(root/'data/controlled_1ee08ac18eaa428f9644ef66daa2fcaa_reporting_summary.csv')
    out.append('\n기존 실험의 반복 평균(집합별 문제 수가 다르므로 새 실험과 단순 속도 비율을 계산하지 않음):\n'+table(['모델','집합','방식','문제 수','정확도','검증 호출','입력/출력 토큰','검증 CPU초','검증 벽시계초'],[[r['model'],r['dataset'],r['method'],r['requests_mean'],percent(r['accuracy_mean']),number(r['verification_calls_mean'],1),f'{r["verification_input_tokens_mean"]:g}/{r["verification_output_tokens_mean"]:g}',number(r['measured_cpu_seconds_mean']),number(r['verification_wall_seconds_mean'])] for r in old]))
    out.append('\n## 9. 새로움과 기존 기법\n기대 효용에 따라 계산 행동을 고르는 관점은 [Selecting Computations: Theory and Applications (2012)](https://proceedings.mlr.press/r10/hay12a.html)의 메타추론과 연결된다. 예측 후 전문가에게 위임할지 학습하는 문제는 [Learning to Defer (2020)](https://proceedings.mlr.press/v119/mozannar20b.html) 등 기존 연구다. 자기 검증이 정답을 훼손할 수 있다는 우려는 [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)와 관련되지만 그 연구가 모든 검증의 무용성을 증명하는 것은 아니다.\n\n이 저장소에서 추가한 것은 오답 복구와 정답 훼손을 분리한 간단한 전이 예측기, 실제 CPU 예측을 결합한 예산 선택 및 공유 최초 답변을 이용한 통제 비교다. 로지스틱 회귀·확률 보정·기대 가치·탐욕적 배정 자체를 새로운 기술이라고 주장하지 않는다. 새로운 세계 최초 알고리즘이나 기업 규모 효율을 입증하지 못했다.\n')
    out.append('\n## 10. 최종 질문의 답\n1. **검증이 유용한 답변을 예측했는가?** 확률의 분별력은 위 지표로 확인할 수 있으나 주 조건의 순정답 개선으로 이어지지 않았다. 실용적인 선택 능력은 입증하지 못했다.\n2. **Selective보다 우수한가?** 아니다. 일부 CPU 절감 관측과 전체 성공 조건은 별개이며 사전 지정 성공 조건을 통과하지 못했다.\n3. **단순 검증 생략과 다른가?** 설계는 복구·훼손과 비용을 예측한다. 그러나 1.5B에서는 검증 생략과 같은 선택으로 퇴화했고 0.5B 주 조건도 추가 정확도 이점이 없었다.\n4. **전체 AI 연산량이 줄었는가?** Selective 대비 측정 단계 CPU·토큰 차이는 표와 CSV에 있다. 최초 생성 비용 합산은 구성요소 계산이며 학습까지 포함한 전체 생애주기 연산 감소나 GPU 연산 감소를 입증하지 못했다.\n5. **정확도를 유지했는가?** No Verification 대비 주 조건 관측 정확도는 동일하지만 Selective 대비 일반화 비열등성은 입증하지 못했다.\n6. **무엇이 새로운가?** 이 프로젝트에서의 전이별 가치·실측 CPU 결합과 통제 실험 추가다. 기반 학습/의사결정 기법은 기존 방법이다.\n7. **AI 기업에 제시할 근거가 있는가?** 재현 가능한 로컬 부정적 실험 자료로는 제시할 수 있다. 제품 성능이나 대형 모델의 비용 우월성을 주장할 증거는 부족하다.\n')
    out.append('\n## 11. 재현·산출물·테스트\n'+
        '실행: `.venv/bin/python -B -u src/vvp_experiment.py` (Ollama 필요, 새 UUID 파일 생성). 분석: `.venv/bin/python -B src/vvp_analysis.py data/<prefix>`. 그래프: `.venv/bin/python -B src/vvp_plots.py data/<prefix>` (matplotlib 필요). 분석/그래프/보고서는 덮어쓰기를 거부하므로 같은 이름으로 재실행하지 않는다. 그래프 의존성은 프로젝트 외부 임시 경로에 설치했으며 모델/유료 API는 추가하지 않았다.\n\n'+
        f'자동 테스트: **{test_result}**. 모의 모델은 테스트에만 사용했고 본 보고서 수치는 실제 호출 로그에서 산출했다. 보존 감사: 기존 파일 {audit["preserved_files_checked"]:,}개 해시 일치, 학습-기존 평가 중복 0, 새 보정/평가와 과거 문제 중복 0, 모든 최초 답변/프롬프트/예산 선택/채점 재구성 통과, API 오류 {audit["api_errors"]}회.\n\n'+
        '원본 '+link('outcomes')+' · '+link('calls')+' · '+link('repeat_summary')+' · '+link('resources')+'\n\n'+
        '분석 '+link('aggregate')+' · '+link('comparisons')+' · '+link('effects_repeat_summary')+' · '+link('failure_cases')+' · '+link('predictive_metrics')+' · '+link('budget_actual')+' · '+link('pareto')+' · '+link('accounting')+'\n\n'+
        f'그래프: [정확도–예산](data/{p.name}_accuracy_budget.png), [호출–예산](data/{p.name}_calls_budget.png), [정확도–검증 CPU](data/{p.name}_accuracy_cpu.png), [정확도–합산 CPU](data/{p.name}_accuracy_accounted_cpu.png), [정답 전이](data/{p.name}_transitions.png). 각 SVG도 저장했다.\n\n'+
        '지식 정답 근거: [IANA HTTP 상태 코드](https://www.iana.org/assignments/http-status-codes), [WHATWG HTML 요소](https://html.spec.whatwg.org/multipage/indices.html#elements-3), [Python 문장 정의](https://docs.python.org/3/reference/simple_stmts.html), [BIPM SI 접두어](https://www.bipm.org/en/measurement-units/si-prefixes). 코드 출력 정답은 별도 실행 대조, 산술은 별도 AST 기준과 비교했다. 다른 유형의 수작업 기준 정답은 자동 감사만으로 완전한 무오류를 보증하지 않는다.\n\n'+
        '향후 연구 후보는 다른 검증기에서 복구 사례를 충분히 확보하고 외부 업무 집합에서 사전 등록 재현하는 것이다. 이번 결과를 바탕으로 추가 알고리즘이나 후속 실험은 실행하지 않았다.')
    text='\n'.join(out).replace('予測・','예측 및 ')
    report=root/'VVP_RESEARCH_REPORT.md'
    with report.open('x',encoding='utf-8') as f:f.write(text+'\n')
    failure=p.parent/(p.name+'_failure_report.md')
    with failure.open('x',encoding='utf-8') as f:
        f.write('# VVP 실패 사례 분석\n\n주 예산 25%는 두 모델 모두 No Verification 대비 순정답 개선 0이다. 개별 사례를 반복 행으로 독립 표본처럼 세지 않는다.\n\n')
        f.write(table(['모델','예산','수정 평균','훼손 평균','놓친 개선 평균','미추출 복구 평균','추출된 오답 복구 평균'],[[r['model'],r['budget_percent'],r['repairs_mean'],r['damage_mean'],r['missed_repairs_full_reference_mean'],r['unresolved_output_recoveries_mean'],r['parsed_wrong_repairs_mean']] for r in agg if r['method']=='vvp']))
        f.write('\n\n1.5B의 EV가 양수인 후보가 없어 모두 생략하는 퇴화가 발생했다. 0.5B에서는 기존 채점상 복구와 훼손이 상쇄되는 조건을 관찰했다. 임계값을 평가 결과에 맞춰 변경하지 않았다. 미추출 복구와 이미 추출된 오답 복구는 위 표와 원본 CSV에 분리했다. 미추출 복구가 모두 형식만의 개선인 것은 아니며 별도 내용 감사에서 확인한다.\n\n')
        outcomes=read('outcomes');index={(r['model'],r['task_id'],r['method'],r['budget_percent']):r for r in outcomes if r['repetition']==1}
        forecasts={(r['model'],r['task_id']):r for r in read('forecasts')}
        examples=[]
        for m in meta['models']:
            rr=[r for r in outcomes if r['model']==m and r['repetition']==1 and r['method']=='vvp' and r['budget_percent']==25]
            kinds={'정답 훼손':[r for r in rr if r['right_to_wrong']], '오답 복구':[r for r in rr if r['wrong_to_right']],
                   '놓친 복구':[r for r in rr if not r['verification_executed'] and not r['primary_correct'] and index[m,r['task_id'],'full',100]['final_correct']],
                   '검증했으나 정오 유지':[r for r in rr if r['verification_executed'] and not r['wrong_to_right'] and not r['right_to_wrong']]}
            for kind,candidates in kinds.items():
                candidates.sort(key=lambda r:-forecasts[m,r['task_id']]['expected_value'])
                for r in candidates[:3]:
                    reference=index[m,r['task_id'],'full',100]
                    examples.append([m,kind,r['task_id'][:12],r['category'],r['question'].replace('\n',' ').replace('|','/'),r['expected'],r['primary_prediction'],r['final_prediction'],reference['final_prediction'],number(forecasts[m,r['task_id']]['expected_value'])])
        f.write('\n\n## 예시: 첫 반복의 유형별 최대 3건\n사후 설명용 사례이며 학습·선택에 사용하지 않았다. 정답 기회 기준인 Full100은 별도 시간에 실행한 검증이다.\n\n')
        f.write(table(['모델','사건','ID','유형','질문','정답','최초 답','VVP 답','Full100 답','예측 EV'],examples))
        budget=read('budget_actual');ratios=[r for r in budget if r['cpu_quota_ratio'] is not None and r['actual_calls']>0]
        ratios.sort(key=lambda r:-r['cpu_quota_ratio'])
        f.write('\n\n## CPU 예산 예측 오차\n다음은 실제/예산 비율이 큰 조건들이다. 예산은 선택된 항목 예상 비용의 합이 아닌 전체 허용량이므로 이 비율만으로 항목별 회귀 오차를 추정하지 않는다. 실제 총 CPU에는 판단 비용이 포함된다.\n\n')
        f.write(table(['모델','방식','예산','반복','호출','예측 허용 CPU초','실제 총 CPU초','비율'],[[r['model'],r['method'],r['budget_percent'],r['repetition'],r['actual_calls'],number(r['predicted_cpu_quota']),number(r['actual_total_cpu']),number(r['cpu_quota_ratio'])] for r in ratios[:12]]))
        f.write(f'\n\n개별 정답·응답·선택: [{p.name}_outcomes.csv]({p.name}_outcomes.csv). VVP 오답 행: [{p.name}_failure_cases.csv]({p.name}_failure_cases.csv). CPU 예측 오차: [{p.name}_budget_actual.csv]({p.name}_budget_actual.csv). 전체 해석: [연구 보고서](../VVP_RESEARCH_REPORT.md).\n')
    print(report);print(failure)

if __name__=='__main__':render(sys.argv[1],sys.argv[2])
