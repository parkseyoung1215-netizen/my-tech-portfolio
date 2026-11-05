"""Actual PVCG CSV -> Korean research report and standalone comparison figures."""
import json,os,sys,statistics
from pathlib import Path
from .controlled_analysis import read_decoded
from .vvp_report import table,percent,number
NAMES={'none':'No Verification','full':'Full Verification','selective':'Selective','pvcg':'PVCG'}

def plot(out):
    deps=Path(os.environ.get('VVP_PLOT_DEPS','/private/tmp/vvp_plot_deps_20261105'))
    if deps.exists():sys.path.insert(0,str(deps))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/pvcg_matplotlib_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read_decoded(out/'PVCG_RESULTS.csv');models=list(dict.fromkeys(r['model'] for r in rows));methods=list(NAMES);colors=['#777777','#cb6c32','#2875ab','#8551ab']
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for axs,m in zip(axes,models):
        rr=[next(r for r in rows if r['model']==m and r['method']==s) for s in methods];x=list(range(4));labels=['None','Full','Selective','PVCG']
        axs[0].bar(x,[100*r['accuracy_mean'] for r in rr],color=colors);axs[0].set_ylim(0,100);axs[0].set_ylabel('Final content accuracy (%)')
        vals=[r['stage_cpu_seconds_mean'] for r in rr];low=[r['stage_cpu_seconds_ci95_low'] for r in rr];hi=[r['stage_cpu_seconds_ci95_high'] for r in rr]
        axs[1].bar(x,vals,color=colors,yerr=[[max(0,a-b) for a,b in zip(vals,low)],[max(0,b-a) for a,b in zip(vals,hi)]],capsize=3);axs[1].set_ylabel('Measured gate + verification CPU (s)')
        axs[2].bar(x,[r['verification_calls_mean'] for r in rr],color=colors);axs[2].set_ylabel('Actual verifier calls per 300')
        for ax in axs:ax.set_xticks(x,labels);ax.set_title(m);ax.grid(axis='y',alpha=.2)
    fig.suptitle('PVCG: 5 actual repeats, same frozen initial answers\nCPU bars: repeat-mean bootstrap95 CI; no GPU/power/money measurement')
    fig.tight_layout()
    for ext in ('png','svg'):
        with (out/f'PVCG_COMPARISON.{ext}').open('xb') as f:fig.savefig(f,format=ext,dpi=180,bbox_inches='tight')
    plt.close(fig);fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,m in zip(axes,models):
        for method,color in zip(methods,colors):
            r=next(r for r in rows if r['model']==m and r['method']==method);xx=r['stage_cpu_seconds_mean'];yy=100*r['accuracy_mean']
            ax.scatter(xx,yy,color=color,label=NAMES[method],s=55,marker='s' if method=='pvcg' else 'o')
            ax.annotate(NAMES[method],(xx,yy),xytext=(5,-15 if method=='pvcg' else 7),textcoords='offset points',fontsize=8)
        ax.set_title(m);ax.set_xlabel('Measured gate + verification CPU (s)');ax.set_ylabel('Final content accuracy (%)');ax.grid(alpha=.2)
    fig.suptitle('Observed Pareto points; no statistical dominance claim; shared initial cost excluded')
    fig.tight_layout()
    for ext in ('png','svg'):
        with (out/f'PVCG_PARETO.{ext}').open('xb') as f:fig.savefig(f,format=ext,dpi=180,bbox_inches='tight')
    plt.close(fig)
    with (out/'PVCG_PLOT_METADATA.json').open('x') as f:json.dump({'matplotlib':matplotlib.__version__,'source':'PVCG_RESULTS.csv','scope':'Actual observed means; no synthetic performance or smoothed curve'},f,indent=2)

def report(prefix,test_result,output_dir=None):
    p=Path(prefix).resolve();out=Path(output_dir).resolve() if output_dir else p.parent/p.name
    def read(s):return read_decoded(f'{p}_{s}.csv')
    def result(s):return read_decoded(out/s)
    manifest=json.loads(Path(f'{p}_manifest.json').read_text());meta=json.loads(Path(f'{p}_metadata.json').read_text());profiles=json.loads(Path(f'{p}_profiles.json').read_text());audit=json.loads((out/'PVCG_AUDIT.json').read_text())
    rows=result('PVCG_RESULTS.csv');verdict=result('PVCG_VERDICT.csv');comp=result('PVCG_COMPARISONS.csv');cats=result('PVCG_CATEGORY_RESULTS.csv');pareto=result('PVCG_PARETO.csv')
    skipall=all(r['verification_calls_mean']==0 for r in rows if r['method']=='pvcg');success=all(r['success'] for r in verdict)
    text=['# Pre-Verification Compute Gate (PVCG) 연구 보고서',
      '\n## 결론\n'+('사전 성공 조건을 두 모델 모두에서 통과했다. 계측 범위와 독립 재현 한계는 아래와 같다.' if success else '**PVCG의 성공 조건을 충족하지 못했다. 새로운 비용 절감 기술을 발견했다고 주장하지 않는다.**')+
      (' 두 모델 모두 검증을 0회 선택해 No Verification으로 퇴화했다. 같은 정확도에 추가 판단 비용이 발생하므로 검증을 모두 생략하는 방식보다 유리하지 않다.' if skipall else ' 호출 감소만으로는 성공이 아니며 정확도·실측 CPU·No Verification 대비 이점을 함께 판단했다.'),
      f'\n실험 `{p.name}`. 신규 평가300문제, 2모델 × 4방식 × 5반복 = 40조건, 실제 결과 {meta["outcome_rows"]:,}행이다. 실제 평가 검증 호출 {meta["actual_verification_calls"]:,}회, 공통 최초 모델 생성 {meta["actual_primary_calls"]}회, 신규 보정 수집 {meta["new_calibration_calls"]}회, 검증 루프 벽시계 {meta["verification_loop_seconds"]:.3f}초(워밍업·언로드·저장 포함)를 기록했다. 모의 응답은 성능 수치에 사용하지 않았다.',
      '\n## 1. VRG 실패에서 출발한 설계\nVRG는 수정안 생성 뒤 거절해 정답 파괴를 줄였지만, 이미 지출한 검증 호출·토큰·CPU를 줄이지 못했다. PVCG는 최초 답변 직후 검증 호출 **전**에만 판단한다. 기존 Selective가 검증하려는 집합을 더 걸러내며 수정안이 나오면 모든 검증 방식에 같은 채택 규칙을 쓴다. 수정안을 본 뒤 PVCG 판단을 뒤집는 경로는 없다.\n\n'+
      'PVCG의 관측 입력은 문제에서 추정한 유형과 최초 답변의 추출/형식 상태, 계산 도구 여부다. 추가 LLM 호출은 없다. 학습 자료의 유형×상태별 검증 복구/훼손/유지 횟수에 중립 Dirichlet(1,1,2) 사전값을 두고 기대 순이익을 추정한다. 표본10개 미만은 유형 전체로 backoff하며 여전히 부족하면 생략한다. 기대 이익−불확실성 가중 표준오차가 비용 가중치×과거 검증 CPU 중앙값을 넘을 때만 검증한다. 이 점수는 보장된 신뢰 하한이나 실제 CPU 측정값이 아니다.\n\n'+
      '기존 학습 전용200문제/모델을 사용했고 이전 평가나 이전 VVP 보정 문제를 학습으로 전용하지 않았다. 새로운 보정100문제에서 불확실성 가중치0/.5/1, CPU 가중치0/.05/.2를 비교했다. 보정 정답 수 최대, 동률이면 보정 실측 검증 CPU 합계 최소, 다시 동률이면 보수적 설정을 택했다. 최종 평가의 정답이나 검증 결과로 재조정하지 않았다. No Verification으로 퇴화하면 실패로 보고하도록 사전에 정했다.',
      '\n## 2. 데이터·환경·공정성\n시드 **78112026**. 신규 보정100개와 최종 평가300개를 분리했으며 산술·문서 기반 지식 검색·논리·코딩·문장 이해를 각각20/60개 포함했다. 문서 검색은 제공된 가상 서비스 카탈로그에서 사실을 찾는 합성 작업이며 일반 상식 회상 능력으로 표현하지 않는다. 카탈로그의 작업 문맥은 정상 입력이고 실제 정답 선택지 라벨은 모델 입력에 붙이지 않았다.\n\n'+
      '과거 CSV/JSON과 정규화한 문제 줄기/보기 순서 교체/산술 중복을 확인했다. 새로운 문자열이 의미적 독립을 보장하지는 않으며 보정·평가 생성 템플릿도 일부 공유된다. 각 유형의 반씩을 고정 A/B 집단 각150개로 나눴다. 이는 외부 독립 평가셋이 아니다.\n\n'+
      '기존 `qwen2.5:1.5b`, `qwen2.5:0.5b`를 그대로 사용했다. temperature0, seed42, 최대 출력64토큰, 기존 생성/검증 프롬프트와 VRG의 고정 내용 추출 규칙을 공유했다. 형식 준수율은 내용 정답률과 별도 측정한다. 산술60개는 모든 방식에서 기존 안전 계산 도구로 처리해 최초 모델 호출은240회/모델이다. 도구의 이점을 PVCG 성과로 취급하지 않고 AI 최초 생성240문제의 하위 결과도 보고한다.\n\n'+
      '최초 답변을 모델별 한 번 생성해 모든 방식·반복에서 공유했다. 모든 평가용 검증 호출 전에 선택 계획까지 동결했고 실행 시 다시 판단해 동일성을 검사하며 판단 CPU/시간을 계측했다. 검증을 선택한 Full/Selective/PVCG는 유효한 수정안을 채택하고 미추출이면 원답을 유지한다. 이는 공통 추출 실패 처리이며 VRG식 수정안 선택기가 아니다. 네 방식 모두 공통 입력 경계 검사/유형 해석을 수행하므로 No Verification도 Python 비용이 있다. 가장 단순한 원답 반환 구현은 이보다 저렴할 수 있으며 이를 PVCG의 이점으로 계산하지 않는다.\n\n'+
      '방식의 무작위 기본 순서를 회전시켜 첫4반복 위치를 균형화했다. 5회차는 한 위치를 반복하므로 완전한 위치 균형이 아니다. 모델 순서는 교대하고 모델 언로드·중립 워밍업·문제 도착 순서를 기록했다. OS/하드웨어 캐시와 노트북의 모든 배경 활동을 완전히 통제하지는 못했다. 실제 검증 호출은 방식별로 별도 실행했다.',
      '\n## 3. 실제 결과: 5회 평균\n아래 CPU·시간·토큰은 공통 최초 생성 이후 **판단+검증 단계**의 실측값이다. CW는 정답 파괴, WC는 오답 복구 수다.\n']
    text.append(table(['모델','방식','최초 정확도','최종 정확도','CW/WC','검증 호출','입력/출력 토큰','단계 CPU초','단계 벽시계초','판단 CPU초'],[[r['model'],NAMES[r['method']],percent(r['initial_accuracy_mean']),percent(r['accuracy_mean']),f'{r["CW_mean"]:g}/{r["WC_mean"]:g}',number(r['verification_calls_mean'],1),f'{number(r["input_tokens_mean"],1)}/{number(r["output_tokens_mean"],1)}',number(r['stage_cpu_seconds_mean']),number(r['stage_wall_seconds_mean']),number(r['gate_cpu_seconds_mean'],6)] for r in rows]))
    text.append('\n답변 범위·형식·메모리도 분리했다. 메모리는100ms 간격으로 관측한 Ollama 프로세스 그룹 RSS 최대값으로 정확한 연속 peak나 정책의 독점 점유량이 아니다. No Verification에서도 상주 모델 RSS가 관측될 수 있다.\n'+table(['모델','방식','답변 비율','형식 준수','관측 최대 RSS MiB'],[[r['model'],r['method'],percent(r['answer_rate_mean']),percent(r['format_compliance_mean']),number(r['sampled_max_rss_mib_mean'],1)] for r in rows]))
    text.append('\n판단 CPU는 판단 구간 전후의 Python `process_time` 차이다. 메모리 샘플러 등 같은 프로세스의 다른 스레드 CPU를 완전히 분리한 함수 전용 계측은 아니다. 단계 전체 CPU는 Ollama 프로세스와 Python 프로세스의 합이며 노트북 전체의 모든 시스템 프로세스 CPU는 아니다.\n')
    text.append('\n## 4. 동일 정확도와 비용·Pareto 비교\nNo Verification은 핵심 대조군이다. 검증 호출을 줄여도 이 방식에 대해 정확도 이점이 없으면 PVCG의 추가 가치를 입증하지 못한다. 아래 정확도 차이는 PVCG−대조군의 퍼센트포인트다.\n')
    text.append(table(['모델','대조군','정확도 차이 pp','paired95 CI pp','보수적 하한 pp','CPU 차이초','CPU95 CI초','벽시계 차이초'],[[r['model'],r['control'],number(100*r['accuracy_delta']),f'[{100*r["paired_bootstrap_low"]:.3f}, {100*r["paired_bootstrap_high"]:.3f}]',number(100*r['conservative_exact_lower95']),number(r['cpu_delta_mean']),f'[{number(r["cpu_delta_low"])}, {number(r["cpu_delta_high"])}]',number(r['wall_delta_mean'])] for r in comp]))
    failed=[r['model'] for r in comp if r['control']=='selective' and r['conservative_exact_lower95']<0]
    if failed:text.append('\n동일 정확도에서의 절감을 주장하려면 먼저 Selective 대비 비열등성을 확인해야 하지만 '+', '.join(failed)+'는 허용 손실0 기준을 통과하지 못했다. 단순히 더 싼 점을 같은 정확도의 개선으로 부르지 않는다.\n')
    text.append('\nNo Verification과의 직접 동일 정확도 비교: '+ '; '.join(r['model']+' 정확도 차이 '+number(100*r['accuracy_delta'])+'pp, PVCG 추가 CPU '+number(r['cpu_delta_mean'],6)+'초/300문제' for r in comp if r['control']=='none')+'다.\n')
    text.append('\n'+table(['모델','방식','관측 frontier','지배하는 대안'],[[r['model'],r['method'],r['on_observed_frontier'],', '.join(r['dominated_by'])] for r in pareto]))
    text.append('\n이 Pareto 관계는 관측 점 비교이며 통계적 지배 보장은 아니다. 동일 최초 생성 비용을 더해도 점들의 지배 관계는 바뀌지 않는다. 고정된 네 방식의 점만 비교했으며 평가에서 유리한 임계값을 새로 선택하지 않았다.\n')
    text.append('\n## 5. 전체 비용과 학습 비용의 범위\n최초 생성은 한 번 실제 측정했고 다음의 합계는 실측 구성요소 회계다. 각 방식의 처음부터 끝까지 독립 실행을5번 했다고 표현하지 않는다. 워밍업·언로드·감사 비용은 아래 정책별 합산에 넣지 않았으며 원시 로그/전체 루프 시간으로 따로 기록했다. CPU 시간은 GPU 연산량과 동일하지 않다.\n')
    text.append(table(['모델','방식','합산 모델 호출','합산 입력/출력 토큰','최초+단계 CPU초','최초+단계 벽시계초'],[[r['model'],r['method'],number(r['accounted_total_model_calls_mean'],1),f'{number(r["accounted_total_input_tokens_mean"],1)}/{number(r["accounted_total_output_tokens_mean"],1)}',number(r['accounted_total_cpu_seconds_mean']),number(r['accounted_total_wall_seconds_mean'])] for r in rows]))
    training=read('training_cost')
    text.append('\n새 보정 수집 및 피팅의 실측값:\n'+table(['모델','과거 학습행','새 수집 호출','입력/출력 토큰','수집 CPU초','수집 벽시계초','피팅 CPU초'],[[r['model'],r['historical_training_rows'],r['new_calibration_calls'],f'{r["input_tokens"]}/{r["output_tokens"]}',number(r['collection_cpu_seconds']),number(r['collection_wall_seconds']),number(r['fit_cpu_seconds'],6)] for r in training]))
    text.append('\n과거 학습 출력 수집은 VVP에서 이미 지출한 비용이고 무료가 아니다. 그때의 모든 최초 학습 생성 CPU가 완전히 별도 계측되지 않았으므로 전체 생애주기 순 연산 절감은 입증하지 못했다. CPU 비용 예측은 과거 관측의 그룹 중앙값이며 이번 실제 CPU와 구분한다. **GPU FLOPs·전력·금전 비용은 미측정**이며 추정 절감률을 만들지 않았다.\n')
    text.append('\n## 6. 보정 선택·실패 원인\n'+table(['모델','선택된 불확실성 가중치','선택된 CPU 가중치','평가 검증 계획 수'],[[m,profiles[m]['uncertainty_weight'],profiles[m]['cpu_penalty'],next(r['verification_calls_mean'] for r in rows if r['model']==m and r['method']=='pvcg')] for m in meta['models']]))
    trials=[dict(model=m,**t) for m in meta['models'] for t in profiles[m]['trials']]
    text.append('\n아래는 **보정 자료**의 결과이며 최종 평가 성능으로 대체하지 않는다:\n'+table(['모델','불확실성 가중치','CPU 가중치','보정 정답/100','보정 선택 호출','보정 실측 검증 CPU 합'],[[r['model'],r['uncertainty_weight'],r['cpu_penalty'],r['validation_correct'],r['selected_verifications'],number(r['validation_measured_verifier_cpu_sum'])] for r in trials]))
    groups=[dict(model=m,group=k,**v) for m in meta['models'] for k,v in profiles[m]['groups'].items()]
    text.append('\n학습 집단의 기대 이익 추정(참값이나 안전성 보장 아님):\n'+table(['모델','집단','표본','복구/훼손','예상 순이익','표준오차','과거 CPU 중앙값'],[[r['model'],r['group'].replace('|','/'),r['samples'],f'{r["repair"]}/{r["damage"]}',number(r['mean_gain']),number(r['standard_error']),number(r['median_cpu'])] for r in groups]))
    text.append('\n복구 사건이 적고 유형·표현이 새 자료로 이동하므로 과거 평균만으로 유용한 검증을 구분하기 어렵다. 검증을 모두 생략하면 추가 이익을 얻지 못하고 판단 비용만 생긴다. 단지 불확실성 가중치를 평가 뒤 낮춰 호출을 늘리는 것은 근거 있는 개선이 아니므로 하지 않았다.\n')
    for m in meta['models']:
        g=profiles[m]['groups'].get('coding|*');r=next(r for r in cats if r['model']==m and r['method']=='selective' and r['category']=='coding')
        if g:text.append(f'\n사후 분포 진단: {m}의 과거 학습 코딩 집단은 {g["samples"]}개 중 복구{g["repair"]}·훼손{g["damage"]}였고, 새 평가의 Selective 코딩 집단은60개에서 반복당 복구{r["WC_per_repeat"]:g}·훼손{r["CW_per_repeat"]:g}였다. 이 차이는 새로운 조건에서 검증 이익의 이전이 불안정함을 보여주지만 내부 원인을 확정하지 않는다. 이 사후 결과로 게이트를 재조정하지 않았다.\n')
    text.append('\n동결된 PVCG 판단 사유(반복이 아닌 서로 다른300요청 기준):\n'+table(['모델','사유','요청 수'],[[r['model'],r['reason'],r['unique_requests']] for r in result('PVCG_GATE_REASONS.csv')]))
    text.append('\n전수 검증의 실제 결과를 사후 대조한 놓친 복구·막은 훼손:\n'+table(['모델','방식','놓친 복구','막은 훼손','검증해도 정오 유지'],[[r['model'],r['method'],number(r['missed_repairs_full_reference_mean'],1),number(r['prevented_harm_full_reference_mean'],1),number(r['verified_without_correctness_change_mean'],1)] for r in rows]))
    text.append('\nFull은 같은 최초 답변을 별도 시간/캐시 상태에서 검증했으므로 완전히 같은 실행의 반사실은 아니다. 이 값은 사후 설명용이며 PVCG 결정에 전달하지 않았다.\n')
    actual=read('outcomes');index={(r['model'],r['repetition'],r['method'],r['task_id']):r for r in actual};examples=[]
    for m in meta['models']:
        rr=[r for r in actual if r['model']==m and r['repetition']==1 and r['method']=='pvcg']
        for kind,chosen in [('놓친 복구',[r for r in rr if not r['verification_executed'] and not r['primary_correct'] and index[m,1,'full',r['task_id']]['final_correct']]),('정답 훼손',[r for r in rr if r['transition']=='CW'])]:
            for r in chosen[:3]:
                ref=index[m,1,'full',r['task_id']];examples.append([m,kind,r['task_id'][:12],r['category'],r['question'].replace('\n',' ').replace('|','/'),r['expected'],r['primary_prediction'],ref['final_prediction'],r['decision']['reason']])
    text.append('\n첫 반복의 사후 예시(학습/선택에 사용하지 않음):\n'+table(['모델','사건','ID','유형','질문','정답','최초','Full 답','PVCG 이유'],examples))
    text.append('\n## 7. 유형별 결과\nMODEL_ONLY는 계산 도구를 제외한 AI 최초 생성240문제다.\n'+table(['모델','방식','유형','초기 정확도','최종 정확도','CW/WC 평균','검증 호출 평균'],[[r['model'],r['method'],r['category'],percent(r['initial_accuracy']),percent(r['accuracy']),f'{r["CW_per_repeat"]:g}/{r["WC_per_repeat"]:g}',r['verification_calls_per_repeat']] for r in cats if r['category']!='ALL']))
    text.append('\n## 8. 불확실성과 성공 판정\nCSV에는 반복 평균·표본 SD·최소/최대·bootstrap95% CI를 저장했다. 반복 평균의 정확도 CI는 고정된 문제/최초 답변 조건부이며 독립적인1,500문제나 최초 생성 분산의 증거가 아니다. 정확도 차이는 동일 문제의5회 차이를 묶은 paired bootstrap4,000회와 보수적 전이 하한을 사용했다. CPU/벽시계는 대응 반복5개의 차이를 bootstrap했다. 5반복은 긴 시간 드리프트와 꼬리 위험 추정에 작다. 다중 하위 비교에서 유리한 결과만 골라 성공으로 보지 않는다.\n')
    text.append(table(['모델','방식','정확도 평균±SD','단계 CPU 평균±SD초','CPU95 CI초','단계 벽시계 평균±SD초','판단 CPU 평균±SD초'],[[r['model'],r['method'],percent(r['accuracy_mean'])+' ± '+percent(r['accuracy_sample_std']),number(r['stage_cpu_seconds_mean'])+' ± '+number(r['stage_cpu_seconds_sample_std']),f'[{number(r["stage_cpu_seconds_ci95_low"])}, {number(r["stage_cpu_seconds_ci95_high"])}]',number(r['stage_wall_seconds_mean'])+' ± '+number(r['stage_wall_seconds_sample_std']),number(r['gate_cpu_seconds_mean'],6)+' ± '+number(r['gate_cpu_seconds_sample_std'],6)] for r in rows]))
    text.append(table(['모델','성공','사전 조건'],[[r['model'],r['success'],json.dumps(r['criteria'],ensure_ascii=False)] for r in verdict]))
    text.append('\n정확도 허용 손실은0이며 동일 관측 정확도만으로 일반화 비열등성이 증명되지는 않는다. Selective 대비 비열등성과 CPU 감소, No Verification 대비 엄격한 정확도 이익, 두 고정 집단의 방향, 실제 검증이 일부 존재하는 조건을 두 모델 모두 만족해야 성공이다.\n')
    variation=result('PVCG_CANDIDATE_VARIATION.csv')
    text.append(f'\n동일 검증 요청 {len(variation)}개 중 추출 답이 달라진 그룹 {sum(r["distinct_predictions"]>1 for r in variation)}개를 기록했다. 이 변동을 숨기거나 반복 답변의 독립성을 과장하지 않는다.\n')
    text.append('\n## 9. 최종 판단과 다음 연구\n'+
       ('이번 설정은 두 모델 모두 No Verification과 동일한 선택으로 퇴화했다. Selective/Full 대비 호출·CPU가 줄어도 No Verification보다 유리하지 않고, 더 나은 정확도도 얻지 못했다. 따라서 의미 있는 새 연산 절감 방법을 발견하지 못했다.\n\n' if skipall else '성공 여부는 위 사전 판정과 비교 CI를 따른다. 일부 호출 감소만으로 새로운 효율 기술을 발견했다고 해석하지 않는다.\n\n')+
       'PVCG는 기존 verification gating 및 경험적 검증 이익 추정의 적용이며 VVP/Selective와 근본적으로 독립된 새 기술이라는 근거는 없다. 이 실험에서 실측한 것은 CPU·토큰·호출·벽시계다. 이를 GPU FLOPs나 전력·금전 절감으로 확장하지 않는다. 학습 비용까지 포함한 전체 생애주기 절감 역시 확인하지 못했다.\n\n'+
       '다음 우선 과제는 더 많은 게이트를 추가하는 것보다 **외부 근거가 있는 검증기가 실제 복구 이익을 내는지** 먼저 확인하는 것이다. 틀린 유효 답변을 복구하고 정답을 훼손하지 않는 검증 신호가 있어야 호출 전 예측도 학습할 수 있다. 복구/훼손 사례를 별도 학습 자료로 확보하고 외부 업무 분포에서 사전 등록해 검증해야 한다. 후속 개발은 이번에 진행하지 않았다.\n')
    text.append('\n## 10. 보존·테스트·재현\n'+f'원본 및 실행 전 파일 **{audit["preserved_files_unchanged"]}개 해시 유지**, 과거와 새 보정/평가 문제 줄기 중복0, 학습/보정/평가 분리, 모든 최초 프롬프트·동결 계획·채점·실제 호출 재구성 통과. API 오류 {audit["api_errors"]}회. 결정 API는 정답/검증 결과 키를 받으면 거부한다.\n\n'+
       f'자동 테스트: **{test_result}**. 기존 신규 문제 생성 테스트가 이미 기록된 질문을 거부하므로 전체 테스트는 실험 전의 고정된 데이터 입력 스냅샷을 임시 폴더에 두고 실행했다. 실제 데이터는 삭제하지 않았으며 모의 응답은 임시 테스트 산출물에만 사용했다.\n\n'+
       '프로젝트 루트 실행 명령:\n\n```bash\n.venv/bin/python -B -u -m src.pvcg_experiment\n'+
       f'.venv/bin/python -B -m src.pvcg_analysis data/{p.name} data/{p.name}_reanalysis\n'+
       '.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json\n'+
       f'.venv/bin/python -B -m src.pvcg_report data/{p.name} "<실제 테스트 결과>" data/{p.name}_reanalysis\n```\n\n'+
       '첫 명령은 신규 실행용이며 이미 저장된 동일 문제 줄기가 있으면 중복 방지를 위해 중단한다. 분석은 기존 원시 로그로 실행하며 검증 모델을 재호출하지 않는다. 출력 폴더는 새 이름을 지정해야 하며 덮어쓰지 않는다. 원래 데이터/프로필/초기 답변/seed는 manifest와 frozen 파일에 보존했다. 그래프는 기존 임시 matplotlib 설치를 사용했다.\n\n'+
       '[결과](PVCG_RESULTS.csv) · [원시 반복 지표](PVCG_REPEAT_METRICS.csv) · [비용 분석](PVCG_COST_ANALYSIS.csv) · [통계 비교](PVCG_COMPARISONS.csv) · [Pareto](PVCG_PARETO.csv) · [감사](PVCG_AUDIT.json) · [비교 그래프](PVCG_COMPARISON.png) · [Pareto 그래프](PVCG_PARETO.png)\n\n'+
       f'실제 원시 응답/호출: `../{p.name}_outcomes.csv`, `../{p.name}_calls.csv`. 학습/보정 설정·실행 순서·모델 digest·환경·보존 해시는 `../{p.name}_manifest.json`, 동결된 선택은 `../{p.name}_plans.json`이다.\n\n'+
       '한계: 합성 영어 단답·선택형 평가, 템플릿 공유, 내용 추출 규칙 의존, 같은 소형 모델의 상관된 오류, 노트북의 비전용 자원과 캐시,5반복, 외부 데이터/모델 계열 미검증이다. 문자열 중복0이 의미적 독립이나 모델 사전학습 데이터와의 중복0을 증명하지 않는다.\n')
    with (out/'PVCG_RESEARCH_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    plot(out);print(out/'PVCG_RESEARCH_REPORT.md')

if __name__=='__main__':report(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
