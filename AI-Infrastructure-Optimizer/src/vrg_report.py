"""Korean VRG report and standalone figure from audited real artifacts only."""
import json,os,statistics,sys
from pathlib import Path
from .controlled_analysis import read_decoded
from .vvp_report import table,percent,number
NAMES={'none':'No Verification','always':'Always Accept','selective':'Selective','heuristic':'VRG-Heuristic','learned':'VRG-Learned','oracle':'Oracle Upper Bound'}

def figure(out):
    deps=Path(os.environ.get('VVP_PLOT_DEPS','/private/tmp/vvp_plot_deps_20261105'))
    if deps.exists():sys.path.insert(0,str(deps))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/vrg_matplotlib_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read_decoded(out/'VRG_RESULTS.csv');models=list(dict.fromkeys(r['model'] for r in rows));methods=list(NAMES)
    fig,axes=plt.subplots(2,3,figsize=(15,9));colors=['#777777','#c94b45','#2677a9','#31935d','#8950ad','#d29526']
    for axs,m in zip(axes,models):
        rr=[next(r for r in rows if r['model']==m and r['method']==method) for method in methods];x=list(range(6));labels=['None','Always','Selective','Heuristic','Learned','Oracle*']
        y=[r['accuracy_mean']*100 for r in rr];err=[[max(0,r['accuracy_mean']-r['accuracy_ci95_low'])*100 for r in rr],[max(0,r['accuracy_ci95_high']-r['accuracy_mean'])*100 for r in rr]]
        axs[0].bar(x,y,color=colors,yerr=err,capsize=3);axs[0].set_ylabel('Final content accuracy (%)');axs[0].set_ylim(0,100)
        axs[1].bar([i-.18 for i in x],[r['CW_mean'] for r in rr],width=.36,color='#c94b45',label='Correct to wrong')
        axs[1].bar([i+.18 for i in x],[r['WC_mean'] for r in rr],width=.36,color='#31935d',label='Wrong to correct');axs[1].legend(fontsize=8);axs[1].set_ylabel('Mean transitions per 300')
        cpu=[r['stage_cpu_seconds_mean'] for r in rr[:-1]];lo=[r['stage_cpu_seconds_ci95_low'] for r in rr[:-1]];hi=[r['stage_cpu_seconds_ci95_high'] for r in rr[:-1]]
        axs[2].bar(x[:-1],cpu,color=colors[:-1],yerr=[[max(0,a-b) for a,b in zip(cpu,lo)],[max(0,b-a) for a,b in zip(cpu,hi)]],capsize=3);axs[2].set_ylabel('Measured decision + verification CPU (s)')
        for ax in axs:ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_title(m);ax.grid(axis='y',alpha=.2)
    fig.suptitle('VRG: 5 actual repeats; shared initial answers; *Oracle is analysis only\nError bars: conditional repeat-mean bootstrap CI; Oracle CPU unmeasured')
    fig.tight_layout()
    for suffix in ('png','svg'):
        with (out/f'VRG_COMPARISON.{suffix}').open('xb') as f:fig.savefig(f,format=suffix,dpi=180,bbox_inches='tight')
    plt.close(fig)
    with (out/'VRG_PLOT_METADATA.json').open('x') as f:json.dump({'matplotlib':matplotlib.__version__,'data':'VRG_RESULTS.csv','oracle_cost':'unmeasured; no bar'},f,indent=2)

def report(prefix,test_result,output_dir=None):
    p=Path(prefix).resolve();out=Path(output_dir).resolve() if output_dir is not None else p.parent/p.name
    def read(s):return read_decoded(f'{p}_{s}.csv')
    def result(s):return read_decoded(out/s)
    meta=json.loads(Path(f'{p}_metadata.json').read_text());manifest=json.loads(Path(f'{p}_manifest.json').read_text());audit=json.loads((out/'VRG_AUDIT.json').read_text())
    rows=result('VRG_RESULTS.csv');comparisons=result('VRG_COMPARISONS.csv');verdict=result('VRG_HYPOTHESES.csv');transition=result('VRG_ERROR_TRANSITIONS.csv');initial=read('primary_cost');train=read('training_cost')
    successes=[r for r in verdict if r['success']]
    text=['# Verification Revision Guard (VRG) 연구 보고서',
      '\n## 결론\n'+('사전 조건을 통과한 개별 모델/Guard가 있다. 아래 모델 간 재현과 비용 한계까지 확인해야 전체 성공을 판정할 수 있다.' if successes else '**사전 지정 성공 조건을 통과한 Guard는 없다.** 정답 훼손 감소와 정확도–비용 우월성은 별개이며 검증을 이미 수행한 뒤 판단하는 VRG가 검증 연산을 절약한다고 주장하지 않는다.'),
      f'\n실험 `{p.name}`: 새 최종 평가 300문제, 모델 2개, 실제 실행 방식 5개, 각 5회 반복이다. 실제 검증 {meta["verification_calls"]:,}회, 최초 모델 생성 {meta["primary_calls"]}회, 새 보정 수집 {meta["calibration_calls"]}회, 실제 평가 행 {meta["raw_rows"]:,}개다. 검증 루프 전체 벽시계 시간은 {meta["verification_loop_seconds"]:.3f}초(언로드·워밍업·기록 포함)이며 학습/공통 최초 생성은 별도다. Oracle 3,000행은 별도 사후 상한선이며 추가 실행이나 무료 시스템이 아니다. 워밍업 호출은 원시 파일에 별도 보존했다.',
      '\n## 1. 원인 분석: 기존 실험\n기존 VVP 결과를 모델·방식·유형별 CC/CW/WC/WW로 나눴다. 원본 점수는 보존하고 이전 내용 감사 규칙을 적용한 값도 별도 기록했다. 아래 표는 **기존** 내용 기준 전이의 5반복 평균이며, 같은 최초 답변을 반복한 1,500행을 독립 문제 1,500개로 보지 않는다.\n']
    hist=read('historical_transitions')
    text.append(table(['모델','기존 방식','CC','CW','WC','WW','초기 정답 중 파괴율','초기 오답 중 복구율'],[[r['model'],r['method'],*(number(r[x]/5,1) for x in ('CC','CW','WC','WW')),percent(r['CW_rate_initial_correct']),percent(r['WC_rate_initial_wrong'])] for r in hist if r['grading']=='explicit_content' and r['category']=='ALL']))
    text.append('\n기존 전수 검증의 유형별 평균:\n'+table(['모델','유형','CC','CW','WC','WW','전체 중 CW율','전체 중 WC율'],[[r['model'],r['category'],*(number(r[x]/5,1) for x in ('CC','CW','WC','WW')),percent(r['CW_rate_all']),percent(r['WC_rate_all'])] for r in hist if r['grading']=='explicit_content' and r['method']=='full' and r['category']!='ALL']))
    text.append('\n기존 로그는 자기 검증이 정답을 뒤집을 수 있음을 보여주지만 내부 추론의 심리적 원인을 입증하지 않는다. 도구 정답을 모델로 재계산하는 불필요한 수정, 추출 실패와 실제 내용 오류, 모델별 복구/훼손 비대칭을 구분했다. 원본 및 내용 기준 모든 유형/방식 전이는 `../'+p.name+'_historical_transitions.csv`에 있다. 기존 Full은 미추출 수정안을 최초 답변으로 fallback했으므로 이번 Always Accept와 정확히 같은 정의가 아니다. 과거/새 수치를 동일 실험처럼 직접 합치지 않는다.\n')
    text.append('\n## 2. 실험 설계와 알고리즘\n'+
       '시드 **68112026**. 모델은 기존 `qwen2.5:1.5b`, `qwen2.5:0.5b`이며 서버 태그·모델 digest·양자화·플랫폼은 manifest에 저장했다. temperature=0, seed=42, 최대 출력64토큰, 기존 최초 생성 및 검증 프롬프트를 유지했다. 산술·일반 지식·논리·코딩·문장 이해 각60문제이고, 같은 평가를 다섯 번 실행했다. 산술60개는 모든 방식의 공통 기존 계산 도구 경로를 사용해 최초 모델 호출은240회/모델이다. 산술 도구의 정확성을 VRG 자체 성과로 계산하지 않는다.\n\n'+
       '학습은 과거 **학습 전용**200문제/모델의 원시 최초·수정 답변만 재사용했다. 기존 평가 및 VVP 보정 문제는 학습에 넣지 않았다. 새 보정100문제는 정규화 .01/.1/1과 온도 .75/1/1.5의 log loss 선택에만 썼다. 새 평가300개에서는 학습·임계값 변경을 하지 않았다. 문제 줄기·보기 순서 교체·산술 정규화 기준 중복을 배제했으나 의미적·템플릿 독립까지 보증하지 않는다. 각 유형을 반씩 나눈 A/B 각150개는 방향 재현용으로만 쓰며 외부 독립 데이터셋이 아니다.\n\n'+
       '모델별 최초 답변은 한 번 생성해 동결했고 모든 방식·반복이 이를 공유했다. 기존 추출기와 이전 VVP 내용 감사 규칙을 **실행 전부터 고정**해 적용했다. 형식 준수는 별도 지표다. 내용이 미해결이면 오답으로 채점하며 자연어의 모든 의미를 이해하는 채점기는 아니다.\n\n'+
       '- **No Verification:** 최초 답변 그대로 반환.\n'+
       '- **Always Accept:** 미추출 수정안까지 원문 그대로 채택. 이를 fallback해 정답 훼손을 숨기지 않는다.\n'+
       '- **Selective:** 기존 게이트와 미추출 수정안 fallback 유지.\n'+
       '- **VRG-Heuristic:** 수정안 미해결이면 거절, 같은 내용은 허용, 안전한 계산 도구의 정답은 보존, 최초 답변 미해결일 때만 유효한 변경을 허용. 최초의 잘못된 유효 답변도 보존할 수 있는 보수적 규칙이다.\n'+
       '- **VRG-Learned:** 최초+수정안의 관측 가능한 길이·형식·파싱·유형·AST 크기·내용 일치 등으로 복구/훼손/정오 유지의 다항 로지스틱 회귀를 학습한다. 예상 복구확률−훼손확률>0인 변경을 채택하고 무효 후보/도구 정답 보호를 적용한다. 모델의 주장된 확신도를 정답 확률로 사용하지 않는다.\n'+
       '- **Oracle:** Always의 실제 최초/수정안 중 정답을 사후 선택한다. 선택 API와 실행 가능한 방식 목록에 포함하지 않았으며 비용을 0으로 표시하지 않는다.\n\n'+
       'Guard는 검증 후 판단하므로 두 Guard 모두 매 반복300회 검증을 실제로 실행한다. 같은 최초·도착 순서를 공유하지만 각 실제 방식은 별도로 모델을 호출했다. 무작위 기본 방식 순서를 반복마다 회전해 각 위치에 한 번씩 배치했고 모델 순서는 반복마다 교대했다. 모델 언로드·중립 워밍업과 실행 위치를 기록했다. OS/하드웨어 캐시를 완전히 제거하지는 못했다.\n')
    text.append('\n## 3. 실제 평가 결과: 5회 평균\nCW는 최초 정답이 최종 오답이 된 수, WC는 반대다. 아래 시간·CPU·토큰은 최초 생성 이후 **판단+검증 단계**이다. Oracle은 품질 상한선이며 비용은 미측정이다.\n')
    text.append(table(['모델','방식','최초 정확도','최종 정확도','CW/WC','답변 비율','형식 준수','검증 호출','입력/출력 토큰','단계 CPU초','단계 벽시계초'],[
       [r['model'],NAMES[r['method']],percent(r['initial_accuracy_mean']),percent(r['accuracy_mean']),f'{r["CW_mean"]:g}/{r["WC_mean"]:g}',percent(r['answer_rate_mean']),percent(r['format_compliance_mean']),number(r['verification_calls_mean'],1),f'{number(r["verification_input_tokens_mean"],1)}/{number(r["verification_output_tokens_mean"],1)}',number(r['stage_cpu_seconds_mean']),number(r['stage_wall_seconds_mean'])] for r in rows]))
    text.append('\n전체 전이와 분모를 구분한 비율:\n'+table(['모델','방식','CC','CW','WC','WW','전체 중 CW','초기 정답 중 CW','전체 중 WC','초기 오답 중 WC'],[[r['model'],NAMES[r['method']],*(number(r[x+'_per_repeat'],1) for x in ('CC','CW','WC','WW')),percent(r['CW_rate_all']),percent(r['CW_rate_initial_correct']),percent(r['WC_rate_all']),percent(r['WC_rate_initial_wrong'])] for r in transition if r['category']=='ALL']))
    text.append('\n## 4. 비용 계측과 학습 비용\nCPU는 Ollama 관련 프로세스 누적 CPU 차이와 Python CPU의 합이다. CPU 시간은 GPU 연산량이 아니다. 메모리는 Ollama 프로세스 그룹 RSS를100ms 간격으로 관측한 최대값으로, 연속 진짜 peak나 방식별 독점 점유량이 아니다. 모델이 이미 상주해 있어 No Verification에도 RSS가 관측될 수 있다.\n')
    text.append(table(['모델','최초 생성 호출','최초 입력/출력 토큰','최초 CPU초','최초 벽시계초'],[[r['model'],r['model_calls'],f'{r["input_tokens"]}/{r["output_tokens"]}',number(r['cpu_seconds']+r['router_cpu_seconds'] if r['cpu_seconds'] is not None else None),number(r['stage_wall_seconds'])] for r in initial]))
    text.append('\n최초 생성과 실행 단계의 실측 구성요소 합산:\n'+table(['모델','방식','합산 모델 호출','합산 입력/출력 토큰','합산 CPU초','합산 벽시계초','Guard CPU초','관측 최대 RSS MiB'],[[r['model'],NAMES[r['method']],number(r['accounted_total_model_calls_mean'],1),f'{number(r["accounted_total_input_tokens_mean"],1)}/{number(r["accounted_total_output_tokens_mean"],1)}',number(r['accounted_total_cpu_seconds_mean']),number(r['accounted_total_wall_seconds_mean']),number(r['guard_cpu_seconds_mean'],6),number(r['sampled_max_rss_mib_mean'],1)] for r in rows]))
    text.append('\n이 합계는 동일 최초 생성의 비용을 각 정책에 더한 **구성요소 회계**이며, 매번 최초 생성을 다시 한 독립 end-to-end 실행이 아니다. 최초 생성 변동을 반복 측정한 것으로 해석하지 않는다. 공통 워밍업·언로드·감사 비용은 방식별 표에 포함하지 않았으며 로그에 별도 기록했다.\n\n')
    text.append(table(['모델','기존 학습행','신규 보정행','새 수집 호출','수집 입력/출력 토큰','수집 CPU초','수집 벽시계초','피팅 CPU초'],[[r['model'],r['historical_training_rows'],r['new_calibration_rows'],r['new_collection_calls'],f'{r["new_collection_input_tokens"]}/{r["new_collection_output_tokens"]}',number(r['new_collection_cpu_seconds']),number(r['new_collection_wall_seconds']),number(r['fit_cpu_seconds'])] for r in train]))
    text.append('\n기존 학습 출력 수집은 이미 지출한 VVP 비용이며 이번에 다시 생성하지 않았다. 이를 무료 데이터라고 보지 않는다. 과거 전체 최초 학습 생성 CPU를 완전히 별도 계측하지 않았으므로 생애주기 총 CPU 절감을 주장하지 않는다. 새 보정 수집과 피팅은 별도 실제 측정값이다. **GPU FLOPs·전력·금전 비용은 미측정**이다. 서버 duration 값은 보조 원시 로그이며 실제 CPU와 교체하지 않는다.\n')
    text.append('\n## 5. 반복 변동과 통계 비교\n평균·표본 표준편차·최소/최대·bootstrap95% CI는 VRG_RESULTS.csv 및 VRG_COST_ANALYSIS.csv에 저장했다. 반복 평균 CI는 같은 고정 문제/최초 답변 조건부 변동이며 모집단 정확도의 확실성을 의미하지 않는다. 정확도 차이는 문제별5회 차이를 묶은 paired bootstrap4,000회 및 보수적 전이 하한을 사용했다. CPU 차이는 대응 반복5개 차이의 bootstrap이다. 5개 반복은 꼬리 위험과 시간 드리프트 추정에 작고, 별도 다중 비교 보정을 하지 않았으므로 탐색적 하위 비교만으로 성공을 주장하지 않는다.\n')
    text.append(table(['모델','Guard','대조군','정확도 차이 pp','95% CI pp','보수 하한 pp','CPU 차이초','CPU 95% CI초','CW 감소 pp','CW 감소 95% CI pp'],[[r['model'],r['guard'],r['control'],number(100*r['accuracy_delta']),f'[{100*r["paired_bootstrap_low"]:.3f}, {100*r["paired_bootstrap_high"]:.3f}]',number(100*r['conservative_exact_lower95']),number(r['cpu_delta_mean']),f'[{number(r["cpu_delta_low"])}, {number(r["cpu_delta_high"])}]',number(100*r['harm_reduction']),f'[{100*r["harm_reduction_low"]:.3f}, {100*r["harm_reduction_high"]:.3f}]'] for r in comparisons]))
    text.append('\n반복별 변동 요약:\n'+table(['모델','방식','정확도 평균±SD','단계 CPU 평균±SD초','CPU 95% CI초','단계 벽시계 평균±SD초'],[[r['model'],r['method'],percent(r['accuracy_mean'])+' ± '+percent(r['accuracy_sample_std']),number(r['stage_cpu_seconds_mean'])+' ± '+number(r['stage_cpu_seconds_sample_std']),f'[{number(r["stage_cpu_seconds_ci95_low"])}, {number(r["stage_cpu_seconds_ci95_high"])}]',number(r['stage_wall_seconds_mean'])+' ± '+number(r['stage_wall_seconds_sample_std'])] for r in rows]))
    text.append('\n## 6. H1–H5와 성공 판정\nH2는 Always 복구의80% 이상 유지라는 사전 지정 설명적 기준을 적용했다. 이는 자체적인 통계적 비열등성 증명은 아니다. 정확도 개선의 불확실성, 비용 증가, 평가 누출, Selective 대비 실질 이점 부재 중 하나라도 있으면 성공으로 보지 않는다. 각 Guard는 두 모델 모두에서 조건을 충족해야 한다.\n')
    text.append(table(['모델','Guard','Always 대비 복구 유지율','사전 기준','성공'],[[r['model'],r['guard'],percent(r['repair_retention']),json.dumps(r['criteria'],ensure_ascii=False),r['success']] for r in verdict]))
    text.append('\nH4: 두 Guard가 검증 호출을 줄이지 않는 설계이므로 추가 판단 CPU만으로 이미 지출한 검증 비용을 회수할 수 없다. 하위 시스템의 재시도 절감 등 측정하지 않은 미래 이익을 가정하지 않았다. 캐시/순서로 발생한 CPU 차이를 Guard의 연산 절감으로 설명하지 않는다. Heuristic이 처음부터 바꾸지 않을 유효 답변을 검증하는 일은 실제로 불필요할 수 있다. 이를 호출 전 생략으로 바꾸면 기존 verification gating의 문제가 되므로 이번 사후 채택 효과와 혼동하지 않았다.\n')
    pareto=result('VRG_PARETO.csv')
    text.append('\n관측 정확도–CPU 점들의 Pareto 관계(통계적 지배 증명은 아님):\n'+table(['모델','방식','관측 frontier','지배하는 대안'],[[r['model'],r['method'],r['on_observed_frontier'],', '.join(r['dominated_by'])] for r in pareto]))
    text.append('\n## 7. 유형별 결과와 실패 사례\n'+table(['모델','방식','유형','최종 정확도','CW/WC 평균'],[[r['model'],r['method'],r['category'],percent((r['CC']+r['WC'])/r['repeated_rows']),f'{r["CW_per_repeat"]:g}/{r["WC_per_repeat"]:g}'] for r in transition if r['category'] not in ('ALL','MODEL_ONLY')]))
    text.append('\n계산 도구를 제외하고 AI 모델이 직접 최초 답변을 생성한 **240문제**의 사후 하위 분석(별도 성공 기준이나 재학습에 쓰지 않음):\n'+table(['모델','방식','초기 정확도','최종 정확도','CW/WC 평균'],[[r['model'],r['method'],percent((r['CC']+r['CW'])/r['repeated_rows']),percent((r['CC']+r['WC'])/r['repeated_rows']),f'{r["CW_per_repeat"]:g}/{r["WC_per_repeat"]:g}'] for r in transition if r['category']=='MODEL_ONLY']))
    actual=read('outcomes');cases=[]
    for m in meta['models']:
        for guard in ('heuristic','learned'):
            rr=[r for r in actual if r['model']==m and r['method']==guard and r['repetition']==1]
            for kind,candidates in [('정답 훼손',[r for r in rr if r['transition']=='CW']),('복구 거절',[r for r in rr if not r['accepted_revision'] and not r['primary_correct'] and r['verification_prediction'] is not None and r['verification_prediction']==r['expected']]),('오답 유지',[r for r in rr if r['transition']=='WW'])]:
                for r in candidates[:2]:cases.append([m,guard,kind,r['task_id'][:12],r['category'],r['question'].replace('\n',' ').replace('|','/'),r['expected'],r['primary_prediction'],r['verification_prediction'],r['final_prediction'],r['accept_reason']])
    text.append('\n첫 반복의 사후 설명용 사례(선택/학습에 사용하지 않음):\n'+table(['모델','Guard','사건','ID','유형','질문','정답','최초','수정','최종','판정 이유'],cases))
    variation=result('VRG_CANDIDATE_VARIATION.csv')
    text.append(f'\n동일 검증 요청 {len(variation)}개 그룹 중 원문이 달라진 그룹 {sum(r["distinct_raw"]>1 for r in variation)}개, 추출 답이 달라진 그룹 {sum(r["distinct_predictions"]>1 for r in variation)}개다. 동일 candidate에 두 Guard를 다시 적용한 `VRG_IDENTICAL_CANDIDATE_DIAGNOSTIC.csv`는 이 변동을 분리하는 사후 진단이며 실제 새 실행·추가 비용 절감 수치가 아니다.\n')
    same=result('VRG_IDENTICAL_CANDIDATE_DIAGNOSTIC.csv');same_summary=[]
    for m in meta['models']:
        for guard in ('heuristic','learned'):
            rr=[r for r in same if r['model']==m and r['guard']==guard]
            same_summary.append([m,guard,sum(r['avoided_damage'] for r in rr)/meta['repeats'],sum(r['lost_repair'] for r in rr)/meta['repeats'],percent(statistics.mean(r['guard_correct'] for r in rr))])
    text.append('\n동일 수정안 진단(사후·반복 평균):\n'+table(['모델','Guard','막은 CW','포기한 WC','가상 최종 정확도'],same_summary))
    text.append('\n## 8. 한계와 기존 연구와의 관계\n'+
       'Heuristic은 유효한 최초 오답도 보존하므로 정답 훼손을 줄이는 대신 복구를 포기할 수 있다. Learned는 과거 학습200문제의 드문 복구/훼손 사건으로 새 수정안의 유용성을 추정해 데이터 이동과 과신 위험이 있다. 같은 소형 모델을 생성과 검증에 사용하므로 오류가 상관될 수 있고 외부 증거는 없다. 이러한 설명은 관측과 설계상의 한계이며 내부 추론 원인을 입증한 것이 아니다.\n\n'+
       '평가는 합성 단답/선택형이며 장문·한국어 실서비스·다른 모델 계열 검증은 하지 않았다. 지식은 HTTP/Python/HTML에 편중됐고 생성 템플릿 일부가 보정/평가에서 공유된다. 정확도는 고정 내용 추출 규칙에 의존한다. 신규 형식의 명확한 자연어 답변을 여전히 놓칠 수 있으며 평가 뒤 편리한 채점 규칙을 추가하지 않았다. 반복 출력은 독립적인 새 문제도 아니고 최초 생성의 무작위성을 포괄하지 않는다. 노트북을 전용 자원으로 완전히 격리하지 않았으며 CPU 계측은 요청별 GPU 작업을 추적하는 계측이 아니다.\n\n'+
       '[Learning to Defer (Mozannar & Sontag, 2020)](https://proceedings.mlr.press/v119/mozannar20b.html)는 예측을 다른 의사결정자에게 위임할지 학습하는 기존 틀이다. [Selective Classification (Geifman & El-Yaniv, 2017)](https://arxiv.org/abs/1705.08500)는 위험과 답변 범위의 교환을 다룬다. VRG는 요청 자체를 거절하기보다 이미 생성한 두 후보 중 하나를 선택하는 적용 형태이지만 독립적인 새 기반 기술이라는 근거는 아니다.\n\n'+
       '[Self-Refine (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651)는 자기 피드백을 통한 반복 개선을 연구했고, [Huang et al. (ICLR2024)](https://arxiv.org/abs/2310.01798)는 외부 피드백 없는 자기 교정에서 성능 저하도 관찰했다. 작업과 모델 조건이 다르므로 어느 논문도 이번 로컬 결과를 직접 보증하거나 모든 자기 검증의 무용성을 증명하지 않는다.\n\n'+
       '프로젝트 기여는 전이별 진단, gold-blind 후보 채택, 기존 선택형 검증과의 실제 호출·CPU·토큰 비교다. verification gating, selective prediction, learning-to-defer, 경량 확률 분류 자체의 발명을 주장하지 않는다.\n')
    initial_map={r['model']:r['initial_accuracy_mean'] for r in rows};changes='; '.join(r['model']+' '+r['method']+' '+percent(initial_map[r['model']])+'→'+percent(r['accuracy_mean']) for r in rows if r['method'] in ('heuristic','learned'))
    harm_counts='; '.join(m+': '+', '.join(r['method']+' '+number(r['CW_mean'],1)+'건' for r in rows if r['model']==m and r['method'] in ('always','selective','heuristic','learned')) for m in meta['models'])
    text.append('\n## 9. 최종 질문의 답\n'+
       '1. **정확도를 개선했는가?** 최초 대비 관측 변화는 '+changes+'다. No Verification 대비 이익과 해로운 Always/Selective 대비 이익을 구분한다. 사전 기준의 일반화 성공은 '+('일부 모델/Guard에서만 관측되어 전체 검토가 필요하다.' if successes else '입증하지 못했다.')+'\n'+
       '2. **정답 파괴를 줄였는가?** 반복당 평균 CW는 '+harm_counts+'다. 따라서 관측상 훼손 억제와 복구 희생을 동시에 확인해야 하며 H1의 신뢰구간 및 H2를 함께 보고한다.\n'+
       '3. **전체 추론 비용이 줄었는가?** 감소를 입증하지 못했다. 검증 후 Guard는 검증 호출을 생략하지 않았고, 이번 두 모델 모두 Selective 대비 실측 단계 CPU와 검증 토큰이 증가했다. GPU·전력·생애주기 전체 연산량 감소는 별도로도 입증하지 않았다.\n'+
       '4. **Selective보다 우수한가?** 이번 정확도–비용 균형의 전체 판정은 아니다. 정확도만의 관측 우위와 비용 균형 우위는 구분한다. 두 모델의 정확도·CPU·복구 유지·재현 조건을 모두 통과하지 못했다.\n'+
       '5. **기존 기법과 차이?** 검증 호출 전 게이팅과 달리 수정안 생성 후 원답/수정안을 선택한다. 이미 발생한 비용을 줄일 수 없는 대신 정답 훼손을 직접 제어한다. 기존 선택/위임 기법의 적용 연구다.\n'+
       '6. **독립적인 새로운 기술인가?** 이를 주장할 이론·광범위 비교·독립 재현 근거는 없다.\n'+
       '7. **가장 먼저 해결할 문제?** 추가 검증 없이도 원답을 보존하는 단순 기준을 넘어, 실제 틀린 유효 원답을 복구할 근거를 확보해야 한다. 외부 증거가 있는 검증과 충분한 복구/훼손 학습 사례를 확보한 뒤 미사용 외부 업무 집합에서 사전 등록해 검증하는 것이 우선이다. 추가 연구는 이번에 실행하지 않았다.\n')
    text.append('\n## 10. 보존·감사·재현\n'+
       f'기존 및 실행 전 파일 **{audit["preserved_files_unchanged"]}개 해시 유지**, 새 보정/평가와 과거 줄기 중복0, 학습/보정/평가 분리, 모든 최초 답변·프롬프트·Guard 판정·채점 재구성 통과. API 오류 {audit["api_errors"]}회. 정답은 실행 바깥의 채점과 Oracle에만 사용했고 Guard API는 정답 키를 전달하면 오류로 거부한다. 확률 학습용 과거 라벨은 학습 자료 안에서만 허용했다.\n\n'+
       f'자동 테스트: **{test_result}**. 모의 출력은 테스트의 임시 폴더에만 생성했으며 이 보고서/CSV/그래프는 실제 호출 로그에서 생성했다. 신규 문제 생성 검사는 이미 사용한 질문을 거부하므로, 최종 전체 테스트는 기록된 실험 전 데이터의 읽기 전용 링크를 둔 임시 스냅샷에서 실행했다. 실제 실험 데이터를 지우거나 중복 방지 장치를 끄지 않았다.\n\n'+
       '명령어(프로젝트 루트):\n\n```bash\n.venv/bin/python -B -u -m src.vrg_experiment\n'+
       f'.venv/bin/python -B -m src.vrg_analysis data/{p.name} data/{p.name}_reanalysis\n'+
       f'.venv/bin/python -B -m src.vrg_test_snapshot data/{p.name}_manifest.json\n'+
       f'.venv/bin/python -B -m src.vrg_report data/{p.name} "<실제 테스트 결과>" data/{p.name}_reanalysis\n```\n\n'+
       '위 첫 명령은 신규 실행용이며 이미 저장된 같은 신규 문제 줄기가 있으면 중복 방지를 위해 중단한다. 기존 실행 재현의 입력은 manifest·profiles·initial_cache·고정 seed에 보존했다. 분석/그래프/보고서는 덮어쓰기를 거부하므로 동일 출력에 다시 실행하지 않는다. 오프라인 감사는 모델을 새 호출하지 않는다. 그래프에는 matplotlib가 필요하며 기존 임시 설치를 사용했다.\n\n'+
       '[결과](VRG_RESULTS.csv) · [오류 전이](VRG_ERROR_TRANSITIONS.csv) · [비용 분석](VRG_COST_ANALYSIS.csv) · [통계 비교](VRG_COMPARISONS.csv) · [H1–H5](VRG_HYPOTHESES.csv) · [감사](VRG_AUDIT.json) · [그래프](VRG_COMPARISON.png)\n\n'+
       f'원시 실행: `../{p.name}_outcomes.csv`, `../{p.name}_calls.csv`, `../{p.name}_repeat_summary.csv`, `../{p.name}_resources.csv`. 선택 프로필·실행 순서·예전 파일 해시는 `../{p.name}_manifest.json`에 있다.\n\n'+
       '지식 기준 출처: [IANA HTTP registry](https://www.iana.org/assignments/http-status-codes), [Python built-ins](https://docs.python.org/3/library/functions.html), [Python exceptions](https://docs.python.org/3/library/exceptions.html), [WHATWG attributes](https://html.spec.whatwg.org/multipage/indices.html#attributes-3). 코드 출력은 독립 Python 실행, 산술은 별도 기준 계산으로 검사했다. 나머지 수작업 의미 기준의 무오류를 자동 검사만으로 보증하지 않는다.\n')
    with (out/'VRG_RESEARCH_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    figure(out);print(out/'VRG_RESEARCH_REPORT.md')

if __name__=='__main__':report(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
