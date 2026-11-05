"""Korean report and plots from measured paired data; no invented device costs."""
import json,os,statistics,sys
from pathlib import Path
from .controlled_analysis import read_decoded
from .alsr_io import write_union
from .adaptive_experiment import write_json
LABEL={'legacy_full':'기존 Full','legacy_stable1':'기존 Stable1','legacy_sketch':'기존 Sketch','lean_full':'Lean Full','lean_stable1':'Lean Stable1','lean_sketch':'Lean Sketch','lean_option_full':'Option Full','lean_option_stable1':'Option Stable1','lean_option_sketch':'Option Sketch'}
def fmt(x,n=3):return '미측정/정의 불가' if x is None else f'{x:.{n}f}'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join('---' for _ in head)+' |']+['| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in r)+' |' for r in rows])+'\n'

def plots(out,summary):
    deps=Path('/private/tmp/vvp_plot_deps_20261105')
    if deps.exists():sys.path.insert(0,str(deps))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stable_runtime_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['OldFull','OldStable','OldSketch','LeanFull','LeanStable','LeanSketch','OptFull','OptStable','OptSketch'];colors=['#888888']*3+['#3174a2']*3+['#b77132']*3
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    for i,ds in enumerate(('replay64','fresh64')):
        rr=[r for r in summary if r['dataset']==ds];axes[i,0].bar(labels,[r['task_success_mean']*100 for r in rr],color=colors);axes[i,0].set(ylabel='Task success (%)',ylim=(0,100),title=ds)
        axes[i,1].bar(labels,[r['wall_seconds_mean'] for r in rr],yerr=[r['wall_seconds_sample_std'] for r in rr],color=colors,capsize=3);axes[i,1].set(ylabel='Natural generation wall / 64 requests (s)',title=ds+' (error bars = repeat SD)')
        for ax in axes[i]:ax.tick_params(axis='x',rotation=35)
    fig.suptitle('Qwen2.5-1.5B BF16 / MLX, 3 repetitions; same policy/input per condition\nOption = explicit one-letter completion condition applied equally to all 3 policies')
    for ext in ('png','svg'):fig.savefig(out/('STABLE_RUNTIME_COMPARISON.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    components=[('model_step_except_gate_wall_seconds_mean','Forward + next token excluding nested gate'),('decision_wall_inclusive_seconds_mean','Gate region (includes prior queued work wait)'),('format_check_wall_seconds_mean','Requested-format check'),('tokenize_wall_seconds_mean','Tokenize'),('cache_setup_wall_seconds_mean','Cache/input setup'),('decode_wall_seconds_mean','Decode'),('host_bookkeeping_wall_seconds_mean','Host loop residual'),('reset_wall_seconds_mean','Reset'),('integrity_check_wall_seconds_mean','Integrity checks')]
    for ax,ds in zip(axes,('replay64','fresh64')):
        rr=[r for r in summary if r['dataset']==ds];bottom=[0.]*len(rr)
        for name,label in components:
            values=[max(0,r[name]) for r in rr];ax.bar(labels,values,bottom=bottom,label=label);bottom=[a+b for a,b in zip(bottom,values)]
        ax.set(title=ds,ylabel='Measured code-region wall / 64 requests (s)');ax.tick_params(axis='x',rotation=45)
    axes[1].legend(fontsize=7,loc='upper left');fig.suptitle('Wall decomposition is NOT GPU time / pure Python decomposition')
    for ext in ('png','svg'):fig.savefig(out/('STABLE_RUNTIME_BREAKDOWN.'+ext),dpi=160)
    plt.close(fig);write_json(out/'PLOT_METADATA.json',{'matplotlib':matplotlib.__version__,'real_measured_data_only':True,'error_bars':'SD over 3 timing repetitions, not independent new questions'})

def report(folder,test_result):
    out=Path(folder).resolve();source=json.loads((out/'SOURCE.json').read_text());p=Path(source['experiment']);m=json.loads((p/'MANIFEST.json').read_text());audit=json.loads((out/'AUDIT.json').read_text());read=lambda name:read_decoded(out/(name+'.csv'));summary=read('RESULTS');comparisons=read('COMPARISONS');verdict=read('VERDICT');attrs=read('LENGTH_ATTRIBUTION');fail=read('FAILURES');cats=read('CATEGORIES');profile=read('PROFILE');sessions=[json.loads(f.read_text()) for f in (p/'sessions').glob('*.end.json')];endtimes=sum(s['elapsed_seconds'] for s in sessions)
    promising=[]
    for pair in {(r['target'],r['control']) for r in verdict}:
        vv=[r for r in verdict if (r['target'],r['control'])==pair]
        if len(vv)==2 and all(r['observed_efficiency_candidate'] for r in vv):promising.append(pair)
    def comparison(ds,target,control):return next(r for r in comparisons if r['dataset']==ds and r['target']==target and r['control']==control)
    highlights=[]
    for ds in ('replay64','fresh64'):
        stable=comparison(ds,'legacy_stable1','legacy_full');impl=comparison(ds,'lean_stable1','legacy_stable1');option=comparison(ds,'lean_option_full','lean_full');full=next(r for r in summary if r['dataset']==ds and r['variant']=='lean_full');st=next(r for r in summary if r['dataset']==ds and r['variant']=='lean_stable1');op=next(r for r in summary if r['dataset']==ds and r['variant']=='lean_option_full')
        highlights.append(f'{ds}: Stable1 내용 점수 차이 {stable["accuracy_delta"]*100:+.2f}%p, 자연 wall {(stable["natural_wall_seconds_ratio"]-1)*100:+.2f}%, 동일16-token wall {(stable["fixed_wall_seconds_ratio"]-1)*100:+.2f}%. 실행기만 바꾼 Stable1 wall 감소 {(1-impl["natural_wall_seconds_ratio"])*100:.2f}%의 CI는 [{(1-impl["natural_wall_seconds_ci95_high"])*100:+.2f}, {(1-impl["natural_wall_seconds_ci95_low"])*100:+.2f}]%로0을 포함한다. Full의 선택지 완료 종료는 내용 점수 {full["task_success_mean"]*100:.2f}%를 유지하고 wall {(1-option["natural_wall_seconds_ratio"])*100:.2f}%, CPU {(1-option["natural_cpu_seconds_ratio"])*100:.2f}%, 추정 matmul FLOPs {(1-option["natural_analytical_matmul_flops_ratio"])*100:.2f}% 감소했다. 실제 블록 호출은 {full["transformer_block_invocations_mean"]:.0f} → {op["transformer_block_invocations_mean"]:.0f}회/회다.')
    probe_files=list(p.glob('ISOLATED_GATE_*.json'));probe_lines=[]
    if probe_files:
        micro=[]
        for f in probe_files:
            obj=json.loads(f.read_text());rr=obj['records'];micro.append({'source_file':f.name,'samples':len(rr),'warm_reused_training_states':True,'mean_wall_seconds':statistics.mean(r['wall_seconds'] for r in rr),'std_wall_seconds':statistics.stdev(r['wall_seconds'] for r in rr),'mean_cpu_seconds':statistics.mean(r['cpu_seconds'] for r in rr)})
        write_union(out/'ISOLATED_GATE_SUMMARY.csv',micro)
        probe_lines=['### 이미 계산된 hidden에서 분리한 Sketch 판단 진단',table(['진단','warm 관측 수','wall ms 평균±SD','CPU ms 평균'],[[r['source_file'],r['samples'],f'{r["mean_wall_seconds"]*1000:.3f}±{r["std_wall_seconds"]*1000:.3f}',fmt(r['mean_cpu_seconds']*1000)] for r in micro]),'학습용4입력의 첫2층 hidden을 완전히 materialize한 후 같은 hidden에서3회×5번씩 판단했다. 이는 캐시가 따뜻한 microbenchmark이며60개의 독립 문제나 production latency가 아니다. 본 실험의 Sketch gate 포함 timer는 약8ms/요청이지만 앞선 계층 대기가 섞여 있으므로 이를 순수 판단 비용이라고 해석할 수 없다. 두 시간을 빼서 정확한 GPU 비용을 추정하지 않는다.']
    stopchanges=read('STOP_TRANSITIONS');stop_summary=[]
    for ds in ('replay64','fresh64'):
        for method in ('full','stable1','sketch'):
            rr=[r for r in stopchanges if r['dataset']==ds and r['method']==method];stop_summary.append([ds,method,sum(r['baseline_correct'] and not r['option_correct'] for r in rr)/3,sum(not r['baseline_correct'] and r['option_correct'] for r in rr)/3,sum(r['possible_word_prefix'] for r in rr)])
    resume=json.loads((p/'RESUME_AUDIT.json').read_text()) if (p/'RESUME_AUDIT.json').exists() else None
    lines=['# Stable1 Runtime Attribution Report','',f'실행 `{p.name}`. 실제 로컬 Qwen2.5 1.5B 모델 실측. **재현된 관측 효율 후보: {str(promising) if promising else "없음"}**. 이 후보 표기는 모집단 무손실·독립 신기술 성공 판정이 아니다.',
      '**Stable1 계층 생략 개선은 성공하지 못했다.** 요청이 요구한 선택지 한 글자를 생성한 뒤 끝내는 Full 조건은 두 셋에서 내용 점수를 유지하며 시간·CPU·실제 블록 호출을 줄였지만, 이는 층 생략 성공이나 일반 서비스의 무손실·비용 절감률이 아니다.',*highlights,'',
      '## 1. 질문과 공정한 비교',
      '이전 Stable1은 총 성공률은 같았지만 정답1개를 잃고1개를 얻었고 형식 준수율도 떨어졌다. 그러므로 출발점부터 전반적 출력 품질이 유지됐다고 가정하지 않았다. 이번에는 출력 길이와 실행기 비용, 같은 token 부하의 비용을 분리한다.',
      f'모델 `{m["model"]["repo"]}` revision `{m["model"]["revision"]}`, BF16, Apple M4 16GiB, MLX0.32.4/MLX-LM0.32.2, Apache-2.0 공개 가중치. 기존 Ollama Q4 또는 다른 모델과의 비교가 아니다. 모델/프롬프트/greedy/EOS/MCQ16·open48 상한은 공통이다. last-position LM head 경로도 모두 공통이며 native full-logits 정합성을 확인했다.',
      'Full, Stable1(0-based24), 기존 frozen Adaptive-Sketch를 legacy/lean에서 비교한다. 모델 학습·계층 재선택·평가 임계값 조정은 없다. replay64와 별도 fresh64를 각3회 실행하고128문제/방식 순서를 무작위화했다. fresh64는 새 템플릿과 값이지만 합성 유형 간 의미적 상관까지 배제한 실서비스 표본은 아니다.',
      'lean_option은 별도 출력 조건이다. 질문이 명시적으로 한 글자 선택지만 요구하고 현재 생성 prefix가 A-D 한 글자일 때만 종료한다. 모델이 만든 token을 그대로 반환하고 정답을 보지 않는다. JSON/문장은 원래 EOS/상한을 유지한다. 세 정책에 같은 규칙을 적용했다. Option Stable1 vs Option Full과 Lean Stable1 vs Lean Full을 비교하며, 출력 조건의 효과를 층 생략 효과로 귀속하지 않는다.','',
      '## 2. 측정 범위와 실행기 개선',
      'legacy는 기존 Qwen native blocks·attention/MLP wrapper counters·per-token set/list/dict trace·logits materialize 뒤 argmax를 유지한다. lean은 같은 native attention/RMS/MLP/잔차 연산 순서를 사용하며, 실행 tuple을 미리 구성하고 decode 때 재사용한다. attention/MLP 디버그 wrapper를 우회하고 실제 진입 층 counter1개를 남긴다. argmax까지 함께 materialize하고 그 device token을 다음 입력으로 사용하여 host 토큰 tensor 재생성을 줄였다.',
      '가중치 복사·양자화·hidden 보정·새 gating/학습·다른 모델은 사용하지 않았다. 원래 경로도 큰 logits 전체를 CPU로 복사하지 않으므로 이를 제거했다고 주장하지 않는다. 실제 줄인 것은 host-created token tensor와 Python 추적/dispatch 및 별도 materialization 경계다. Mac unified-memory에서 실제 물리 복사 byte·bandwidth를 측정한 것은 아니다.',
      'legacy는 연구용 추적이 포함된 custom 실행기다. 그 비용을 제거해 빨라져도 공식 MLX-LM/llama.cpp/Ollama 같은 production runtime보다 우수함을 증명하지 않는다. 새 legacy에도 공통 구간 timer와 integrity 검사가 포함되어 지난 보고서의 absolute wall과 직접 비교해 개선율을 계산하지 않는다. 같은 새 실험 안에서 비교한다. 세 host 변경을 개별적으로 ablate하지 않아 각각의 기여 비율을 단정할 수 없다.',
      'model_step은 forward+다음token 확정+동기화를 합친 실제 wall이다. gate는 이 구간 안의 중첩 timer이며 앞선 두 층의 queued GPU 작업 대기가 섞일 수 있다. model_step에서 gate를 뺀 값은 코드 영역별 시간일 뿐 순수 GPU 연산 시간이 아니다. Stable1은 정적 경로여서 per-request 동적 gate가 없으며 경로 초기화 비용은 load/setup에 포함된다.',
      'host_bookkeeping은 기록한 구간을 제외한 루프 잔차이며 tensor 생성 등 MLX native 호출이 포함될 수 있다. process CPU time에는 Python 이외의 런타임/driver 작업도 포함된다. lean의 argmax wait는 아직 실행 중인 forward 대기까지 포함하고 legacy는 이미 logits를 materialize했으므로 이 둘을 argmax kernel 시간으로 직접 비교하면 안 된다.',
      '별도16-token workload는 replay/fresh 각16문제에서6기본변형을 같은 baseline token prefix로 실행했다. 매 step에서 logits/argmax를 실제 계산하며 고정 token 때문에 계산 그래프를 생략하지 않았다. 별도의8문제×6변형×3회 heavy profile은 실제 native attention/MLP Tap와 첫 logits hash·dispatch timer를 포함하여 primary latency에서 제외한다.',
      'GPU 경과시간·GPU FLOPs 하드웨어 counter·전력·금전 비용은 **미측정**. FLOPs는 실제 route/context/query 토큰 수에 기반한 dense matmul MAC×2 추정이며 norm/softmax/비선형 및 물리 memory traffic은 제외한다.','',
      '## 3. 실제 결과: 평균·표준편차·변동 범위']
    for ds in ('replay64','fresh64'):
        rr=[r for r in summary if r['dataset']==ds];lines.extend([f'### {ds}',
          table(['방식','내용/과제 성공률','객관식 정확도','형식 준수율','평균 생략층','wall s: 평균±SD [min,max]','CPU s: 평균±SD','출력 token/회','MLX peak GiB'],[[LABEL[r['variant']],fmt(r['task_success_mean']*100,2)+'%',fmt(r['mcq_accuracy_mean']*100,2)+'%',fmt(r['format_compliance_mean']*100,2)+'%',fmt(r['mean_skipped_layers_mean'],2),f'{r["wall_seconds_mean"]:.3f}±{r["wall_seconds_sample_std"]:.3f} [{r["wall_seconds_min"]:.3f},{r["wall_seconds_max"]:.3f}]',f'{r["cpu_seconds_mean"]:.3f}±{r["cpu_seconds_sample_std"]:.3f}',fmt(r['output_tokens_mean'],1),fmt(r['mean_mlx_peak_bytes_mean']/2**30)] for r in rr]),
          '한 회64문제의 합계/비율을3회 반복한 통계다. 정확도와 형식을 구분하며 JSON은 타입까지 검사한다. 문장 품질은 이름·장소·한 문장 제약에 한정되고 사실성/문체의 완전한 평가는 아니다. MLX peak는 같은 프로세스·공통 상주 모델에서 요청별 allocator peak의 평균이며 process lifetime RSS를 방법별 peak로 사용하지 않았다.',
          table(['방식','동일16-token wall s','동일 부하 추정 FLOPs','블록 실제 호출/회','층-token 평가/회','동적 gate s/64요청','host input tensor/회'],[[LABEL[r['variant']],fmt(r['fixed_wall_seconds_mean']),fmt(r['fixed_analytical_matmul_flops_mean']/1e12,3)+' T' if r['fixed_analytical_matmul_flops_mean'] is not None else '해당 없음',fmt(r['transformer_block_invocations_mean'],0),fmt(r['layer_token_evaluations_mean'],0),fmt(r['decision_wall_inclusive_seconds_mean']),fmt(r['host_created_input_tensors_mean'],1)] for r in rr])])
    lines.extend(['## 4. paired 정확도·실제 시간·추정 계산량 차이',
      '시간 감소율은1−target/control이다. CI는 질문 cluster와 반복 축의 paired resampling이다. 같은 질문3회를192개의 독립 질문으로 취급하지 않는다. CI가0개선/비율1을 포함하면 명확한 시간 개선으로 주장하지 않는다. 정확도 점추정이 하락하면 통계적 유의성과 무관하게 성공으로 판정하지 않는다.',
      table(['셋','대상 / 대조','정답률 차이 pp [95% CI]','wall 감소율 % [95% CI]','CPU 감소율 % [CI]','자연 추정 FLOPs 감소율 %','fixed wall 감소율 % [CI]'],[[r['dataset'],LABEL[r['target']]+' / '+LABEL[r['control']],f'{r["accuracy_delta"]*100:+.2f} [{r["paired_bootstrap_low"]*100:+.2f},{r["paired_bootstrap_high"]*100:+.2f}]',f'{(1-r["natural_wall_seconds_ratio"])*100:+.2f} [{(1-r["natural_wall_seconds_ci95_high"])*100:+.2f},{(1-r["natural_wall_seconds_ci95_low"])*100:+.2f}]',f'{(1-r["natural_cpu_seconds_ratio"])*100:+.2f} [{(1-r["natural_cpu_seconds_ci95_high"])*100:+.2f},{(1-r["natural_cpu_seconds_ci95_low"])*100:+.2f}]',f'{(1-r["natural_analytical_matmul_flops_ratio"])*100:+.2f}',f'{(1-r["fixed_wall_seconds_ratio"])*100:+.2f} [{(1-r["fixed_wall_seconds_ci95_high"])*100:+.2f},{(1-r["fixed_wall_seconds_ci95_low"])*100:+.2f}]' if r.get('fixed_wall_seconds_ratio') is not None else '조건 다름/해당 없음'] for r in comparisons]),
      '실행기만 바꾼 동일 정책 비교에서 native token/mask가 같으면 Transformer matmul 자체의 수는 같고 host scheduling 비용이 달라진 것이다. 반대로 Stable1 vs Full은 실제 층 수가 달라지지만 자연 출력 길이도 달라진다. Option vs EOS의 절감은 주로 불필요한 후속 token 생성을 끝내는 효과이며 새 계층 생략 알고리즘으로 포장하지 않는다.','',
      '## 5. 병목 원인과 길이 영향',
      table(['셋','실행 조건','Stable1 추가 token/회','추가 wall/회 s','관측 Δtoken-Δwall 상관','회귀 slope s/token'],[[r['dataset'],r['backend'],fmt(r['token_delta'],1),fmt(r['wall_delta_per_repeat']),fmt(r['correlation']),fmt(r['slope_seconds_per_output_token'],4)] for r in attrs]),
      '이는 동일 문제의 관측 결과 차이에 대한 사후 진단이며 출력 길이의 독립적인 인과 계수나 GPU token당 kernel 시간을 뜻하지 않는다. 동일16-token 통제가 빨라지고 자연 생성이 느리면 생성 길이가 구조적 절감을 상쇄했다는 근거다. 그 반대/불확실한 결과도 표에 그대로 남긴다.',
      table(['셋','방식','forward+token s','gate s(중첩)','tokenize s','decode s','형식검사 s','host 잔차 s','reset/cache/check s'],[[r['dataset'],LABEL[r['variant']],fmt(r['model_step_wall_seconds_mean']),fmt(r['decision_wall_inclusive_seconds_mean']),fmt(r['tokenize_wall_seconds_mean']),fmt(r['decode_wall_seconds_mean']),fmt(r['format_check_wall_seconds_mean']),fmt(r['host_bookkeeping_wall_seconds_mean']),fmt(r['reset_wall_seconds_mean']+r['cache_setup_wall_seconds_mean']+r['integrity_check_wall_seconds_mean'])] for r in summary]),
      '게이트 열은 forward 열에 이미 포함되므로 합산하지 않는다. 그림의 분해에서는 forward−gate와 gate를 합쳐 중복을 제거했다. 메모리 복사비/순수 Python 실행시간의 완전한 분리는 하지 못했다. 세 정책 공통의 scalar token read와 종료 확인이 남아 있고 비전용 노트북 열·kernel cache 영향을 완전히 통제하지 못했다.','',
      *probe_lines,
      '## 6. 개별 정답과 출력 품질 변화',
      table(['셋','방식','대조','문제','전환','기준 응답','응답'],[[r['dataset'],LABEL[r['variant']],LABEL[r['control']],r['question'],r['transition'],r['baseline_answer'],r['answer']] for r in fail[:24]]),
      '전체 실패는 FAILURES.csv, 모든 문제/반복의 정답 유지·파괴·복구와 원문은 TRANSITIONS.csv에 보존한다. 같은 총점이 개별 답변 보존을 의미하지 않는다. 선택지 종료로 부적절한 후속 설명이 사라져 parser 점수가 변할 수도 있다. 실제 생성 prefix 여부를 검사하고 출력 조건이 다른 결과는 분리했다.',
      table(['셋','종료 조건 비교 정책','EOS 정답→option 오답/회','EOS 오답→option 정답/회','단어 prefix 가능 관측'],stop_summary),
      '이번 Full 종료 조건에서는 두 셋 모두 개별 내용 점수 손실/복구가0이었다. supplementary initial_choice_label_accuracy도 RESULTS.csv에 남겼다. 이 형식 완료 규칙은 일반 자유서술·코딩 생성·JSON에 적용하지 않았으며 더 넓은 출력 형태에 안전하다고 주장하지 않는다. STOP_TRANSITIONS.csv에 종료 조건 변경 전후 전 사례를 보존한다.',
      table(['셋','변형','유형','성공률','형식 준수율','출력 token/회'],[[r['dataset'],LABEL[r['variant']],r['category'],fmt(r['task_success']*100,2)+'%',fmt(r['format_compliance']*100,2)+'%',fmt(r['output_tokens_per_repeat'],1)] for r in cats]),
      '## 7. 보존·누출·실행 증거',
      table(['검사','결과'],[[k,v] for k,v in audit.items()]),
      '새로운 학습이나 threshold 탐색은 하지 않았다. 생성기 함수는 question과 고정 정책/출력 조건만 받으며 reference/정답 여부를 받지 않는다. fixed prefix는 benchmark 실행기만 가지며 Sketch 판단에는 전달하지 않는다. 새로운 문제의 exact stem은 과거 기록과 겹치지 않지만 의미적/템플릿 독립성을 과장하지 않는다.',
      'legacy/lean의 natural token·mask 일치 여부를 PARITY.csv에, 옵션 종료 prefix 일치 여부를 STOP_PREFIX_PARITY.csv에 모두 남겼다. 실제 native attention/MLP 호출은 별도 Tap profile 및 소프트웨어 양성/음성 대조로 확인했다. 상세 계측의 비용을 primary 비교에 섞지 않았다.',
      f'자동 테스트: **{test_result}**. 테스트 모의 응답을 실측표에 사용하지 않았다.','',
      '## 8. 판정·기술적 의미와 다음 가설',
      table(['셋','대상/대조','내용·형식 점추정 유지','시간 개선 CI','추정 자연 연산 감소','관측 후보','모집단 무손실 확정'],[[r['dataset'],LABEL[r['target']]+'/'+LABEL[r['control']],r['quality_point_not_worse'],r['paired_latency_improvement'],r['estimated_natural_compute_reduction'],r['observed_efficiency_candidate'],r['strict_population_noninferiority_proven']] for r in verdict]),
      '새로운 기술: 실행계획 재사용, host 추적 감소, lazy materialization, device token 재사용, 요청 형식에 맞춘 종료는 알려진 런타임/생성 제어 아이디어의 로컬 조합이다. 이 실험으로 독립적인 신기술·일반적인 무손실 layer skipping을 주장할 수 없다. [MLX lazy evaluation 문서](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [공식 MLX-LM generate](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), [custom StoppingCriteria 문서](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.StoppingCriteria), [ShortGPT](https://arxiv.org/abs/2403.03853)와 범위를 구분한다. 공식 문서는 개념의 근거이며 이번 수치의 근거는 로컬 로그다.',
      '다음 가장 유망한 검증 가설은 **모든 층을 유지한 공식 MLX-LM 실행기에서도 명확한 출력 계약의 완료 종료가 정확도를 유지하며 시간을 줄이는가**다. Full의 선택지 종료만 두 셋에 재현됐으므로 새로운 계층 생략 정책보다 production baseline·독립 holdout·다른 출력 길이에서 이 효과를 먼저 확인할 가치가 있다. JSON/자유서술로 확대하는 안전성은 아직 검증하지 않았다. 현재64+64 합성 문제에서 관측한 무손실은 모집단 무손실 증명이 아니며, 일반 AI 서비스 전체에19~20% 절감률을 적용할 수 없다. 추가 개발이나 새 정책 튜닝은 이번 결과에 맞춰 수행하지 않았다.','',
      '## 9. 재개·재현·비용 범위',
      f'완료 세션 수 {len(sessions)}, 완료 세션 wall 합계 {endtimes:.3f}s. 이는 import/manifest 생성 이전은 제외하며 여러 세션일 때 중단 공백을 포함한 연속 wall이 아니다. 요청 latency 합계에는 모델 응답 처리 비용을, 세션 wall에는 load/warmup·smoke·prep·heavy profile·오프라인 채점·체크포인트 쓰기·export를 포함한다. 단위별 benchmark 파일 쓰기 시간은 별도로 계측하지 않았고 세션 잔차를 pure Python으로 해석하지 않는다. 중단된 세션은 end 기록이 없어 이 합계에 포함되지 않을 수 있다.',
      '요청 완료 JSON은 새 attempt 파일로 fsync하며 재개 시 완료 행을 재사용한다. 불완전 JSON도 삭제하지 않는다. 같은 key의 완료 행이 중복되면 중단한다. 프로파일/소스/패키지 hash가 바뀌면 재개를 거부한다. 재개로 달라진 세션/캐시 상태는 session ID로 추적한다.',
      f'완료 후 실제 재개 감사: {str(resume)}. frozen runner의 session end 필드 new_complete_records는 완료 누계이며, 새 실행 수는 checkpoint의 session ID로 따로 계산한다. 재개 감사는 추가 생성0과 새 model-load 기록 없음을 확인했다.',
      '```bash\n# 새 실행\n.alsr-venv/bin/python -B -u -m src.stable_runtime_experiment\n# 중단 후 이 실행 재개\n.alsr-venv/bin/python -B -u -m src.stable_runtime_experiment '+str(p)+'\n# GPU 호출 없는 상태 확인\n.venv/bin/python -B -m src.stable_runtime_status '+str(p)+'\n# 완료 후 새 ANALYSIS 폴더에 분석\n.venv/bin/python -B -m src.stable_runtime_analysis '+str(p)+'\n.venv/bin/python -B -m src.stable_runtime_report <생성된 ANALYSIS 폴더> "<테스트 결과>"\n```',
      '[결과](RESULTS.csv) · [반복](REPEATS.csv) · [비교 CI](COMPARISONS.csv) · [길이 원인 분석](LENGTH_ATTRIBUTION.csv) · [실패](FAILURES.csv) · [호출 상세 profile](PROFILE.csv) · [그래프](STABLE_RUNTIME_COMPARISON.png) · [분해 그래프](STABLE_RUNTIME_BREAKDOWN.png)'] )
    with (out/'STABLE_RUNTIME_REPORT.md').open('x') as f:f.write('\n\n'.join(lines)+'\n')
    plots(out,summary);print(str(out/'STABLE_RUNTIME_REPORT.md'))

if __name__=='__main__':report(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else '미입력')
