"""Research report from actual ACS checkpoint statistics, with explicit uncertainty."""
import json,os,statistics,sys
from pathlib import Path
from .controlled_analysis import read_decoded
from .adaptive_experiment import write_json
LABEL={'baseline':'Baseline','fixed8':'Fixed Cap 8','fixed16':'Fixed Cap 16','boundary':'Answer Boundary','confidence':'Confidence-Aware','adaptive':'Adaptive Stop'}
def f(v,n=3):return '미측정/해당 없음' if v is None else f'{v:.{n}f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(str(x).replace('|','/').replace('\n',' ') for x in row)+' |' for row in rows])+'\n'

def plots(out,results):
    deps=Path('/private/tmp/vvp_plot_deps_20261105')
    if deps.exists():sys.path.insert(0,str(deps))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/acs_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Full','Cap8','Cap16','Boundary','Confidence','Adaptive'];colors=['grey','#dca340','#b87933','#47956a','#4e80ac','#7b57a1'];fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    for i,ds in enumerate(('replay64','fresh64')):
        rr=[r for r in results if r['dataset']==ds]
        axes[i,0].bar(labels,[r['content_accuracy_mean']*100 for r in rr],color=colors);axes[i,0].set(title=ds,ylabel='Limited requested-fact coverage audit (%)',ylim=(0,100))
        axes[i,1].bar(labels,[r['wall_seconds_mean'] for r in rr],yerr=[r['wall_seconds_sample_std'] for r in rr],capsize=3,color=colors);axes[i,1].set(title=ds,ylabel='Wall / 64 requests (s), error=repeat SD')
        axes[i,2].bar(labels,[r['incomplete_rate_mean']*100 for r in rr],color=colors);axes[i,2].set(title=ds,ylabel='Incomplete output proxy (%)',ylim=(0,100))
        for ax in axes[i]:ax.tick_params(axis='x',rotation=30)
    fig.suptitle('ACS: Qwen2.5-1.5B BF16 / MLX, 3 repeats, real local generation\nLimited fact audit: doctor=physician; added facts not fully adjudicated; not a semantic correctness oracle')
    for ext in ('png','svg'):fig.savefig(out/('ACS_COMPARISON.'+ext),dpi=160)
    plt.close(fig);fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    offsets=[(5,7),(5,-14),(5,7),(-70,-15),(5,7),(5,-15)]
    for ax,ds in zip(axes,('replay64','fresh64')):
        for r,label,offset,color in zip([r for r in results if r['dataset']==ds],labels,offsets,colors):
            x=r['wall_seconds_mean'];y=r['content_accuracy_mean']*100;ax.scatter(x,y,color=color);ax.annotate(label,(x,y),xytext=offset,textcoords='offset points')
        ax.set(title=ds,xlabel='Measured wall / 64 requests (s)',ylabel='Limited fact coverage audit (%)',ylim=(0,100))
    fig.suptitle('Observed accuracy-time tradeoff; not a population noninferiority proof')
    for ext in ('png','svg'):fig.savefig(out/('ACS_PARETO.'+ext),dpi=160)
    plt.close(fig);write_json(out/'PLOT_METADATA.json',{'matplotlib':matplotlib.__version__,'all_performance_from_real_traces':True,'error_bars':'Three repeated timing runs; same questions, not 192 independent tasks'})

def report(folder,tests):
    out=Path(folder).resolve();p=Path(json.loads((out/'SOURCE.json').read_text())['experiment']);m=json.loads((p/'MANIFEST.json').read_text());read=lambda n:read_decoded(out/(n+'.csv'));rr=read('RESULTS');comp=read('COMPARISONS');cases=read('CASES');fail=read('FAILURES');groups=read('GROUPS');audit=json.loads((out/'AUDIT.json').read_text());verdict=read('VERDICT');legacy=read('LEGACY_VS_CONTENT');prob=read('TOKEN_PROBABILITY_AUDIT');pareto=read('PARETO')
    audited=read('AUDITED_RESULTS');audited_comp=read('AUDITED_COMPARISONS');audited_verdict=read('AUDITED_VERDICT');audited_cases=read('AUDITED_TRANSITIONS');audited_idx={(r['dataset'],r['method'],r['task_id'],r['repetition']):r for r in audited_cases}
    joint=[method for method in LABEL if method!='baseline' and all(any(v['dataset']==ds and v['method']==method and v['limited_observed_candidate'] for v in audited_verdict) for ds in ('replay64','fresh64'))]
    display_prob=[];events=read('PROBABILITY_EVENTS')
    for r in prob:
        low,high=map(float,r['token_probability_bin'].split('-'));ee=[e for e in events if e['dataset']==r['dataset'] and e['method']==r['method'] and e['stop'] and low<=e['selected_probability']<high]
        display_prob.append(dict(r,stopped_content_accuracy=statistics.mean(audited_idx[e['dataset'],e['method'],e['task_id'],e['repetition']]['audited_fact_coverage'] for e in ee),stopped_incomplete_rate=statistics.mean(audited_idx[e['dataset'],e['method'],e['task_id'],e['repetition']]['audited_incomplete'] for e in ee)))
    doctor_ids=sorted({r['task_id'] for r in audited_cases if r['dataset']=='fresh64' and r['category']=='text_quality' and r['difficulty']=='moderate'})[:2]
    doctor_examples=[r for r in audited_cases if r['dataset']=='fresh64' and r['task_id'] in doctor_ids and r['repetition']==1 and r['method'] in ('baseline','boundary','confidence','adaptive')]
    highlights=[]
    for ds in ('replay64','fresh64'):
        c=next(c for c in audited_comp if c['dataset']==ds and c['target']=='adaptive' and c['control']=='baseline');a=next(a for a in audited if a['dataset']==ds and a['method']=='adaptive');highlights.append(f'{ds}: Adaptive의 제한적 요구사실 보존 감사값 {a["audited_fact_coverage"]*100:.2f}%, Baseline 대비 wall {(1-c["wall_seconds_ratio"])*100:.2f}% 감소 [95% CI {(1-c["wall_seconds_ci95_high"])*100:.2f}, {(1-c["wall_seconds_ci95_low"])*100:.2f}]%, process CPU {(1-c["cpu_seconds_ratio"])*100:.2f}%, 모델 matmul 추정 {(1-c["analytical_matmul_flops_ratio"])*100:.2f}% 감소. 의미적 완전 정답이나 전체 GPU 연산 감소를 입증한 값은 아니다.')
    examples=[]
    for ds in ('replay64','fresh64'):
        for method in LABEL:
            candidates=[c for c in fail if c['dataset']==ds and c['method']==method];candidates.sort(key=lambda c:(not(c['baseline_content_correct'] and not c['content_correct']),not c['incomplete_output'],c['task_id']));examples.extend(candidates[:2])
    lines=['# Adaptive Computation Stop (ACS) Research Report','',f'실행 `{p.name}`. 실제 무료 로컬 모델 실측. 제한적 사실 감사 후 양셋의 관측 후보: **{", ".join(LABEL[x] for x in joint) if joint else "없음"}**. 작은 표본·사후 채점 감사이며 일반적인 무손실/정답 완성 인식/새 기술 발견을 입증하지 않는다.',*highlights,'',
      '## 0. 최종 판단에 중요한 채점 감사',
      '사전 문자열 채점은 Baseline의 doctor→physician 패러프레이즈8개를 오답으로 처리했다. 그 결과 Boundary/Confidence가 Dr.만 출력해 사실을 잃어도 Wrong→Wrong으로 분류하여 손실을 가렸다. 원래 점수·통계·모델 답변·시간은 그대로 두고 별도 CONTENT_AUDIT 파일을 추가했다. 동의어 인정은 사후의 제한적 평가 보정이며 정책·임계값·모델 학습에는 사용하지 않았다.',
      '요구 사실 보존 감사는 doctor/physician을 동의 역할로 인정한다. 완전한 의미 평가가 아니며 새로운 점수를 무조건 진짜 정확도라고 부르지 않는다. Baseline과 Adaptive는 dedicated/diligently, 일부 환자·지역사회 관련 미제공 사실도 추가했다. 이 표현들은 별도 충실도 문제로 남아 있다. 따라서 요구 사실을 보존했다는 것이 추가 사실 없는 완전한 정답을 의미하지 않는다.',
      table(['셋','방식','원문자열 내용점수','동의어 사실보존 감사','형식 준수율','식별된 추가표현율','wall s'],[[r['dataset'],LABEL[r['method']],f(r['literal_content_accuracy']*100,2)+'%',f(r['audited_fact_coverage']*100,2)+'%',f(r['format_compliance']*100,2)+'%',f(r['identified_fidelity_failure_rate']*100,2)+'%',f(r['wall_seconds'])] for r in audited]),
      table(['셋','대상/대조','감사 사실보존 차이 pp [CI]','wall 감소율 % [CI]'],[[r['dataset'],LABEL[r['target']]+'/'+LABEL[r['control']],f'{r["accuracy_delta"]*100:+.2f} [{r["paired_bootstrap_low"]*100:+.2f},{r["paired_bootstrap_high"]*100:+.2f}]',f'{(1-r["wall_seconds_ratio"])*100:+.2f} [{(1-r["wall_seconds_ci95_high"])*100:+.2f},{(1-r["wall_seconds_ci95_low"])*100:+.2f}]'] for r in audited_comp]),
      table(['셋','방식','감사 품질 점추정 조건','제한적 관측 후보','모집단 무손실 입증'],[[v['dataset'],LABEL[v['method']],v['audited_fact_quality_not_worse'],v['limited_observed_candidate'],v['population_noninferiority_proven']] for v in audited_verdict]),
      'Boundary/Confidence는 신규셋에서 Baseline이 담았던 요구 사실8개를 잃어 실패다. Adaptive는 이 제한적 감사에서 같은 요구 사실 보존을 보였지만, 명시적으로 설계한8개 약어 스트레스 사례의 보호 규칙이 작동한 결과다. 무작위 실서비스의 의미적 완료 판단으로 일반화하지 않는다. 아래 원래 지표들도 감사 전후 비교를 위해 모두 제공한다.','',
      table(['문제 ID','방식','원문 응답','감사 사실보존','식별된 추가표현'],[[r['task_id'][:12],LABEL[r['method']],r['answer'],r['audited_fact_coverage'],r['observed_unrequested_terms']] for r in doctor_examples]),
      '## 1. 설계·모델·데이터',
      f'Qwen2.5-1.5B-Instruct revision `{m["model"]["revision"]}`, Apache-2.0, BF16, Apple M4/16GiB, MLX0.32.4/MLX-LM0.32.2. 기존 로컬 HF 체크포인트를 사용했으며 유료 API/모델 설치는 없다. Full28층을 모든 방식에서 실행하고 종료 시점만 다르게 한다. 가중치·sampling 분포·프롬프트·KV 조건은 공통이다.',
      'replay64는 앞선 Stable Runtime의 기존64문제다. fresh64는 새 객관식32(산술/논리/코딩/독해 각8), JSON16, 한 문장16이다. 이전75% MCQ 편중을 줄였지만 합성 평가이며 실제 서비스 분포는 아니다. fresh 문장 중8개는 Dr. 약어가 있는 의도적 경계 스트레스 사례다. Adaptive의 약어 보호 가설을 검증하는 구성이라 무작위 외부 검증으로 과장하지 않는다.',
      '정책과 임계값은 ACS_DESIGN.md 및 runtime source hash로 평가 전에 고정했다. 학습/평가 임계값 조정은 없고 정답은 채점에만 사용했다. 모델 생성 함수는 question·method만 받으며 reference나 expected값은 받지 않는다. 요청 JSON 키는 입력에 명시된 형식 계약으로만 사용한다.',
      '128문제와 방식 순서를 섞어 각3회 실행했다. 같은 질문의3회 출력을192개의 독립 표본으로 보지 않는다. 새 데이터 exact stem 중복0이나 의미적/템플릿 유사성을 완전히 배제하지 않았다. 기존64의 난이도는 미지정, fresh 난이도는 생성 단계/스키마에 따른 사전 coarse label이며 정답률로 사후 지정하지 않았다.','',
      '## 2. 사전 고정한 방식',
      table(['방식','규칙'],[['Baseline','기존 기본 EOS, 명시적 선택지 요청16/나머지48 안전 상한'],['Fixed Cap8 /16','기본 상한 내 최대8 또는16; 양쪽을 모두 보고하며 유리한 상한만 고르지 않음'],['Answer Boundary','순수 선택지 A-D, 요청 키를 갖춘 완전 JSON, 종결 문장부호'],['Confidence-Aware','같은 경계 + 현재 선택 token의 실제 softmax 확률>=.90'],['Adaptive Stop','선택지 .75 / JSON .60 / 문장 .95; 긴 입력>128token이면 .05 상향(.99상한); 문장 최소6단어와 흔한 약어 보호']]),
      '지원되는 신호: 현재 실제 logits의 selected-token probability와 EOS competitor mass. EOS mass는 현재 token 이전 분포이며 다음 시점 EOS 확률이 아니다. 정답 확률·미래 완성 여부·외부 판정 모델의 confidence는 구현하지 않았다. 선택 token 확률이 높아도 정답/완성을 보장하지 않는다. 확률은 경계 후보에서만 계산하고 그 logsumexp·readback 비용을 wall/CPU에 포함했다.',
      '경계 parser와 유형 인식에 실패하면 EOS/안전 상한으로 계속한다. 자유서술 전체의 의미적 완료를 판정하는 시스템은 아니다. JSON은 따옴표 안 괄호/중복키를 처리하며 fenced JSON은 성급하게 중단하지 않는다.','',
      '## 3. 원문자열 자동지표 비교: 3회 평균·표준편차·범위 (최종 사실 감사는 위0절)']
    for ds in ('replay64','fresh64'):
        rows=[r for r in rr if r['dataset']==ds];lines.extend([f'### {ds}',table(['방식','내용 정확도','형식 준수율','전체 과제 성공률','미완성 비율','wall s 평균±SD [min,max]','CPU s 평균±SD','입력/출력 token/회','TPS','모델호출/회','MLX peak GiB'],[[LABEL[r['method']],f(r['content_accuracy_mean']*100,2)+'%',f(r['format_compliance_mean']*100,2)+'%',f(r['task_success_mean']*100,2)+'%',f(r['incomplete_rate_mean']*100,2)+'%',f'{r["wall_seconds_mean"]:.3f}±{r["wall_seconds_sample_std"]:.3f} [{r["wall_seconds_min"]:.3f},{r["wall_seconds_max"]:.3f}]',f'{r["cpu_seconds_mean"]:.3f}±{r["cpu_seconds_sample_std"]:.3f}',f'{r["input_tokens_mean"]:.0f}/{r["output_tokens_mean"]:.0f}',f(r['aggregate_tokens_per_second_mean'],2),f(r['model_calls_mean'],0),f(r['mean_mlx_peak_bytes_mean']/2**30)] for r in rows]),table(['방식','실제 forward/회','실제 블록 호출/회','모델 matmul 추정 TFLOPs/회','종료판단 s/회','확률판단 s/회(중첩)','실제 확률query/회','decode TPS'],[[LABEL[r['method']],f(r['forward_calls_mean'],0),f(r['transformer_block_calls_mean'],0),f(r['analytical_matmul_flops_mean']/1e12),f(r['decision_wall_seconds_mean']),f(r['probability_wall_seconds_mean']),f(r['probability_query_calls_mean'],1),f(r['aggregate_decode_tps_mean'],2)] for r in rows])])
    lines.extend(['입출력·wall·CPU는64요청 한 회의 합계를3회 반복한 값이다. TPS는 출력 token 합계/전체 wall이며 prefill과 판단을 포함한다. decode TPS는 첫 step 이후 model-step 시간만 사용하므로 end-to-end TPS와 다르다. EOS도 생성 token에 포함한다. 더 짧은 답이 TPS가 낮아도 실제 latency는 줄 수 있어 TPS만으로 성공을 판정하지 않는다.',
      'MLX peak는 요청별 allocator peak의 평균이고 동일 프로세스의 공통 상주 모델을 포함한다. process lifetime RSS를 방식별 peak로 사용하지 않았다. GPU elapsed/전력/금전 비용은 **미측정**이다. 실제 forward·블록 진입 수는 로그/네이티브 감사로 확인했으나 GPU hardware FLOPs는 측정하지 않았다.',
      '모델 matmul FLOPs는 actual query/context/층 수에 기반한 dense MAC×2 추정이다. confidence의 softmax logsumexp·reduction 비용은 wall에는 들어가지만 matmul 추정에는 미포함이다. probability_elements_scanned에 실제 처리 어휘 원소 수를 따로 저장했으며 이를 전체 GPU FLOPs로 바꾸지 않는다.','',
      '## 4. 원문자열 지표 차이·실제 절감률·신뢰구간 (사실 손실은0절 감사 기준)',
      table(['셋','대상/대조','내용 차이 pp [95% CI]','wall 감소율 % [CI]','CPU 감소율 % [CI]','모델 matmul 추정 감소율 %'],[[r['dataset'],LABEL[r['target']]+'/'+LABEL[r['control']],f'{r["accuracy_delta"]*100:+.2f} [{r["paired_bootstrap_low"]*100:+.2f},{r["paired_bootstrap_high"]*100:+.2f}]',f'{(1-r["wall_seconds_ratio"])*100:+.2f} [{(1-r["wall_seconds_ci95_high"])*100:+.2f},{(1-r["wall_seconds_ci95_low"])*100:+.2f}]',f'{(1-r["cpu_seconds_ratio"])*100:+.2f} [{(1-r["cpu_seconds_ci95_high"])*100:+.2f},{(1-r["cpu_seconds_ci95_low"])*100:+.2f}]',f((1-r['analytical_matmul_flops_ratio'])*100,2)] for r in comp]),
      'CI는 질문 cluster와 반복 축의 paired bootstrap이다. 각64문제에서1문제는1.5625%p다. 차이0의 bootstrap CI가0이어도 모집단의 엄격한 무손실은 증명할 수 없다. 보수적인 유한 표본 하한도 COMPARISONS.csv에 저장했다. 기준을 만족해도 관측 후보로만 표시하고 일반적인 정확도 유지를 단정하지 않는다.',
      '### Adaptive의 고정 상한 대비 추가 가치 (동의어 사실 감사 기준)',table(['셋','대조','사실보존 차이 pp','시간 비율 Adaptive/대조 [CI]','해석'],[[r['dataset'],LABEL[r['control']],f(r['accuracy_delta']*100,2),f'{r["wall_seconds_ratio"]:.3f} [{r["wall_seconds_ci95_low"]:.3f},{r["wall_seconds_ci95_high"]:.3f}]','관측 우위 후보' if r['accuracy_delta']>=0 and r['wall_seconds_ci95_high']<1 else '품질·시간 동시 우위 미확인/교환관계'] for r in audited_comp if r['target']=='adaptive' and r['control'] in ('fixed8','fixed16')]),
      '고정 cap보다 정확하지만 느린 결과는 품질-비용 trade-off이며 엄밀한 동시 우위가 아니다. cap8/16 두 개만으로 모든 최적 고정 상한을 이겼다고 주장하지 않는다. PARETO.csv는 원문자열 지표, AUDITED_PARETO.csv와 그림은 제한적 사실 감사의 비지배 점추정이며 통계적 Pareto 우월성 증명은 아니다.','',
      '## 5. 문제별 실패·불완전 종료',
      table(['셋','방식','기준 정답→오답/회','기준 오답→정답/회','중단했으나 내용오답/회','미완성/회'],[[ds,LABEL[method],sum(c['transition']=='Correct -> Wrong' for c in cases if c['dataset']==ds and c['method']==method)/3,sum(c['transition']=='Wrong -> Correct' for c in cases if c['dataset']==ds and c['method']==method)/3,sum(not c['content_correct'] and c['completion_reason'] not in ('eos','max_tokens') for c in cases if c['dataset']==ds and c['method']==method)/3,sum(c['incomplete_output'] for c in cases if c['dataset']==ds and c['method']==method)/3] for ds in ('replay64','fresh64') for method in LABEL]),
      table(['셋','방식','전환','종료','질문','기준 응답','중단 응답','오류'],[[c['dataset'],LABEL[c['method']],c['transition'],c['completion_reason'],c['question'],c['baseline_answer'],c['answer'],c['error_type']] for c in examples]),
      '예시는 셋·방식별 최대2개이며 Baseline 정답 파괴와 미완성을 우선해 기계적으로 고른다. 유리한 사례만 고르지 않으며 전체 실패 원문은 별도 CSV에 보존한다.',
      'CASES.csv는 모든 문제/3반복/방식의 원문·정답·시간·완성 상태를 보존한다. FAILURES.csv는 실패 전부다. 유효한 선택지/JSON이라도 내용 오답일 수 있고, 경계 도달과 정답 완성은 다르다. 상한으로 불완전한 JSON·누락 사실이 생기면 실패다. 문장 내용은 명시된 사실 문자열 보존 surrogate이며 완전한 의미 동등성·문체 평가는 아니다.','',
      '## 6. 실제 토큰 확률 신호와 한계',table(['셋','방식','token 확률 bin','고유 문제','반복 관측','중단 답변 사실보존 감사','중단 미완성 감사'],[[r['dataset'],LABEL[r['method']],r['token_probability_bin'],r['unique_questions'],r['repeat_events'],f(r['stopped_content_accuracy']*100,2)+'%',f(r['stopped_incomplete_rate']*100,2)+'%'] for r in display_prob]),
      '표는 실제로 중단한 경계만 포함한다. 거절된 경계의 prefix 내용은 따로 채점하지 않았으므로 그 확률을 최종 답변 라벨에 억지로 매핑하지 않았다. 같은 token 확률은 next-token likelihood이며 답변 정답/완성 확률로 보정된 것이 아니다. PROBABILITY_EVENTS.csv에 현재 위치·token확률·EOS competitor mass·임계값·종료 여부를 전부 보존한다.','',
      '## 7. 길이·난이도·유형 (제한적 사실 감사)',table(['셋','방식','그룹 기준','그룹','고유문제','사실보존 감사','형식 준수율','미완성 감사','wall/회 s'],[[r['dataset'],LABEL[r['method']],r['group_key'],r['group'],r['unique_questions'],f(r['content_accuracy']*100,2)+'%',f(r['format_compliance']*100,2)+'%',f(r['incomplete_rate']*100,2)+'%',f(r['wall_seconds_per_repeat'])] for r in read('AUDITED_GROUPS')]),
      '길이 그룹은 같은 질문의Baseline 출력 평균 길이 또는 reference token 길이이며 사후 분석 전용이다. 짧게 중단한 방법의 결과 길이로 그룹을 다시 나누어 유리한 질문만 비교하지 않았다. reference/difficulty 라벨은 runtime에 전달하지 않는다.','',
      '## 8. 기존 점수 보존과 감사',table(['셋','방식','기존 점수','새 내용 지표','기존 형식','새 형식','완전 과제 성공'],[[r['dataset'],LABEL[r['method']],f(r['historical_score']*100,2)+'%',f(r['new_content']*100,2)+'%',f(r['historical_format']*100,2)+'%',f(r['new_format']*100,2)+'%',f(r['full_task_success']*100,2)+'%'] for r in legacy]),
      '기존 CSV나 원래 점수를 수정하지 않았다. 문장 약어 Dr.를 독립 문장으로 세는 오류를 보호하고, 명시된 직업 사실도 보존하는지 확인한다. 내용과 형식을 나눴으므로 기존 correct와 값이 달라질 수 있다. 새 점수를 기존 정확도 개선으로 포장하지 않는다. 모델 프롬프트/정답은 변경하지 않았다.',
      table(['검사','결과'],[[k,v] for k,v in audit.items()]),f'자동 테스트: **{tests}**. 테스트 모의 값으로 성능표를 만들지 않았다.','',
      '## 9. 원문자열 판정 보존·기존 기술과 차이·다음 연구',
      '아래는 원문자열 자동판정이며 동의어 누락으로 Boundary/Confidence를 잘못 후보로 남겼던 결과다. 최종 판정에는0절의 사후 사실 감사를 사용하며 사후 감사라는 한계도 함께 적용한다.',
      table(['셋','방식','원품질 점추정 조건','시간 CI 개선','실제 모델작업 감소','원관측 후보','모집단 무손실 입증'],[[v['dataset'],LABEL[v['method']],v['quality_point_not_worse'],v['latency_ci_improved'],v['model_work_reduced'],v['observed_candidate'],v['population_noninferiority_proven']] for v in verdict]),
      'ACS는 기존 custom stopping criteria에 요청 형식 parser·현재 실제 token 확률·유형별 고정 규칙을 조합한 실험이다. 확률분포/가중치를 바꾸지 않고 생성 종료만 바꾼다. 사전 정의된 규칙이 실제 정답 완성 시점을 의미적으로 알아낸다고 주장할 수 없다. [Hugging Face stopping criteria](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.StoppingCriteria), [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)와 범위를 구분한다. 독립적인 새로운 기술이라는 근거는 없다.',
      '다음은 고정 상한의 더 넓은 사전 고정 sweep, 작성자가 규칙을 예상하지 못하는 독립적인 자연 요청, 다양한 출력 길이/JSON스키마/약어/실제 코드 생성에서 실패율을 확인하는 연구가 우선이다. 이 연구는 모델1개·합성64+64·3반복이며 전체 서비스에 절감률을 일반화하지 않는다. 사후 결과에 맞춘 임계값 보정이나 추가 최적화는 수행하지 않았다.','',
      '## 10. 재현과 재개',
      '```bash\n# 현재 실행 재개: 완료 요청은 재실행하지 않음\n.alsr-venv/bin/python -B -u -m src.acs_experiment '+str(p)+'\n# 같은 고정 데이터로 새 폴더에서 실제 모델 전체 재실행\n.alsr-venv/bin/python -B -u -m src.acs_replay '+str(p/'MANIFEST.json')+'\n# 완료 후 새 분석 폴더 생성 및 별도 내용 감사\n.venv/bin/python -B -m src.acs_analysis <실행 폴더>\n.venv/bin/python -B -m src.acs_content_audit <ANALYSIS 폴더>\n.venv/bin/python -B -m src.acs_report <ANALYSIS 폴더> "<실제 테스트 결과>"\n```',
      '재현은 같은 문제를 재사용하며 새 미사용 평가 데이터라고 부르지 않는다. 코어 소스/프로파일/패키지가 달라지면 재개를 거부한다. 요청 JSON은 새파일+fsync로 저장하고 부분 파일도 보존한다. 실제 원본 CSV는 각 EXPORT 폴더, session load/워밍업·native 감사는 sessions에 저장된다. 요청 latency에는 모델·종료판단·tokenize/decode를 포함하며 오프라인 채점/체크포인트 쓰기/모델 load는 별도 연구 단계다.','',
      '[사실 감사 결과](AUDITED_RESULTS.csv) · [원자동점수](RESULTS.csv) · [반복 원본](REPEATS.csv) · [감사 CI](AUDITED_COMPARISONS.csv) · [전체 감사](AUDITED_TRANSITIONS.csv) · [감사 실패](AUDITED_FAILURES.csv) · [확률](PROBABILITY_EVENTS.csv) · [그래프](ACS_COMPARISON.png) · [정확도-시간](ACS_PARETO.png)'] )
    with (out/'ACS_RESEARCH_REPORT.md').open('x') as stream:stream.write('\n\n'.join(lines)+'\n')
    plot_rows=[]
    for original in rr:
        a=next(a for a in audited if a['dataset']==original['dataset'] and a['method']==original['method']);plot_rows.append(dict(original,content_accuracy_mean=a['audited_fact_coverage'],incomplete_rate_mean=a['audited_incomplete_rate']))
    plots(out,plot_rows);print(str(out/'ACS_RESEARCH_REPORT.md'))

if __name__=='__main__':report(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else '미입력')
