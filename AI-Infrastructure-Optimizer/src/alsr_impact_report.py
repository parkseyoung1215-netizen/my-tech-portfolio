"""Render measured layer-impact CSVs into a Korean report and standalone figures."""
import json,os,statistics,sys
from pathlib import Path
from .controlled_analysis import read_decoded
from .adaptive_experiment import write_json
LABELS={'baseline':'Full','adaptive_sketch':'기존 Sketch','static':'기존 Static4','random':'Random4','stable1':'Stable1','random1':'Random1','prelayer_gate':'PreLayerGate'}

def num(v,n=3):return '미측정/정의 불가' if v is None else f'{v:.{n}f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in row)+' |' for row in rows])+'\n'

def plots(out):
    deps=Path('/private/tmp/vvp_plot_deps_20261105')
    if deps.exists():sys.path.insert(0,str(deps))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/alsr_impact_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read_decoded(out/'LAYER_SUMMARY.csv');summary=read_decoded(out/'RESULTS.csv');labels=['Full','OldSketch','Static4','Random4','Stable1','Random1','PreLayer']
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for position,color in [(0,'#3174a2'),(1,'#c36c30')]:
        rr=sorted([r for r in rows if r['position']==position],key=lambda r:r['layer']);x=[r['layer'] for r in rr]
        axes[0,0].plot(x,[r['mean_kl'] for r in rr],marker='.',color=color,label=f'baseline prefix length {position}')
        axes[0,1].plot(x,[100*r['top1_agreement'] for r in rr],marker='.',color=color,label=f'baseline prefix length {position}')
    axes[0,0].set(yscale='symlog',xlabel='Skipped layer (zero based)',ylabel='Mean KL(full || skip), nats');axes[0,0].legend()
    axes[0,1].set(xlabel='Skipped layer (zero based)',ylabel='Top-1 agreement (%)',ylim=(0,103));axes[0,1].legend()
    axes[1,0].bar(labels,[r['task_success_mean']*100 for r in summary],color=['grey','#4976a0','#b97030','#d2a050','#419876','#9fabc1','#7551a5']);axes[1,0].set(ylabel='Task success (%)',ylim=(0,100));axes[1,0].tick_params(axis='x',rotation=25)
    axes[1,1].bar(labels,[r['wall_seconds_mean'] for r in summary],color='#4976a0');axes[1,1].set(ylabel='Natural generation wall / 64 prompts (s)');axes[1,1].tick_params(axis='x',rotation=25)
    fig.suptitle('Qwen2.5-1.5B BF16 / MLX: 64 held-out prompts, 3 repeated timing runs\nAblation distributions use same full-model prefix; generation uses its own history')
    for suffix in ('png','svg'):
        path=out/('LAYER_IMPACT_COMPARISON.'+suffix)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=160)
    plt.close(fig)
    base=summary[0];fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    offsets={'Full':(5,10),'PreLayer':(5,-18),'Stable1':(-50,10),'OldSketch':(-65,-18),'Static4':(5,7),'Random4':(5,-15),'Random1':(5,-15)}
    for label,r in zip(labels,summary):
        x=r['fixed_analytical_matmul_flops_mean']/base['fixed_analytical_matmul_flops_mean'];y=r['task_success_mean']*100;ax.scatter(x,y,s=65);ax.annotate(label,(x,y),xytext=offsets[label],textcoords='offset points')
    ax.set(xlim=(.84,1.035),ylim=(0,100))
    ax.set(xlabel='Analytical dense matmul FLOPs / Full, same16-token workload\n(not hardware FLOPs; generation quality evaluated separately)',ylabel='Natural generation task success (%)',title='Quality vs estimated compute; weights remain resident')
    for suffix in ('png','svg'):
        path=out/('LAYER_IMPACT_EFFICIENCY.'+suffix)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=160)
    plt.close(fig)
    write_json(out/'PLOT_METADATA.json',{'matplotlib':matplotlib.__version__,'all_performance_from_actual_local_traces':True,'flops_analytical':True})

def report(folder,test_result='아직 실행 결과 미입력'):
    out=Path(folder).resolve();read=lambda name:read_decoded(out/(name+'.csv'));m=json.loads((out/'MANIFEST.json').read_text());meta=json.loads((out/'METADATA.json').read_text());profile=json.loads((out/'PROFILE.json').read_text());audit=json.loads((out/'AUDIT.json').read_text())
    results=read('RESULTS');comparison=read('COMPARISONS');layers=read('LAYER_SUMMARY');signals=read('SIGNAL_POOLED');paired=read('SIGNAL_PAIRED');verdict=read('VERDICT');categories=read('CATEGORY_RESULTS');agreement=read('TOP1_VS_SEQUENCE');base=next(r for r in results if r['method']=='baseline');gate=next(r for r in results if r['method']=='prelayer_gate');candidate=profile['candidate'];stable=sorted([r for r in layers if r['position']==0],key=lambda r:r['mean_kl']);promising=[r['method'] for r in verdict if r['promising']]
    signal_pair=next(r for r in paired if r['control']=='frozen_old_sketch');signal_better=signal_pair['auroc_delta_ci95_low'] is not None and signal_pair['auroc_delta_ci95_low']>0 and signal_pair['risk_mae_delta_ci95_high']<0
    signal_verdict='동일8층/첫 위치에서 AUROC와 MAE 모두의 CI가 기존 Sketch보다 유리하다. 다만 학습 조건 차이가 있고 이것만으로 실제 연산 절감은 입증하지 못한다.' if signal_better else '동일8층/첫 위치에서 기존 Sketch보다 예측력이 확실히 우수하다고 입증하지 못했다. AUROC와 MAE의 효과 방향과 CI를 함께 봐야 한다.'
    design_link=os.path.relpath(Path(__file__).resolve().parents[1]/'ALSR_IMPACT_DESIGN.md',out)
    stable_result=next(r for r in results if r['method']=='stable1');stable_compare=next(r for r in comparison if r['method']=='stable1' and r['control']=='baseline');candidate_pair=next(r for r in paired if r['scope']=='candidate1');examples=[r for r in read('FAILURE_CASES') if r['method']=='stable1' and r['baseline_correct']!=r['final_correct']]
    cpu=read('CPU_COMPARISONS') if (out/'CPU_COMPARISONS.csv').exists() else []
    random_losses=[r for r in read('TRANSITIONS') if r['method']=='random1' and r['initial_next_token_agreement'] is True and r['transition']=='Correct -> Wrong']
    pool_new=next(r for r in signals if r['scope']=='old_sketch_pool8' and r['position']==0 and r['signal']=='prelayer_ridge');pool_old=next(r for r in signals if r['scope']=='old_sketch_pool8' and r['position']==0 and r['signal']=='frozen_old_sketch')
    correction=[]
    if meta.get('corrected_tie_rule'):
        tie=json.loads((out/'TIE_AUDIT.json').read_text());phases=read('PHASE_COMPARISON')
        correction=['## 동점 처리 오류와 재측정 공개',
          f'새 진단 코드의 argpartition 기반 top2 정렬이 BF16 동점에서 실제 greedy argmax와 다른 top1을 선택했다. full top1 기록 {tie["changed_full_top1"]}행, flip 라벨 {tie["changed_flip_labels"]}/{tie["original_rows"]}행이 달랐다. 과거 ALSR의 kl_effect는 이미 np.argmax를 사용하므로 이 오류를 과거 ALSR 전체의 오류로 확대하지 않는다.',
          f'원래 로그·프로파일·생성/시간 기록은 보존했다. top10에 모든 최대값 동점 토큰이 포함됨을 확인해 최소 token ID라는 greedy 규칙으로 라벨을 복원했다. signed margin 복원이 불가능했던 {tie["targeted_margin_rechecks"]}건은 실제 재실행하여 원래 logits SHA 일치를 확인했다. KL/JS/top2 margin 원측정은 그대로다. 최초 실제 MLX 생성의 첫 token도 NumPy greedy 선택과 모두 같음을 확인했다.',
          '수정 라벨의 학습32/보정16만으로 다시 fit했다. grid·KL 기준·후보 선택 규칙은 변경하지 않았다. 수정 후에도 후보24/threshold0이다. 단, 회귀 계수가 달라져7가지 방식 전체를 같은 순서 조건으로 다시3회 실측했다. **아래 주 결과는 수정 후3회만**이며 원래3회를 독립적인 추가3회로 합치거나 유리한 성능으로 골라 쓰지 않았다.',
          '이 평가셋은 처음 연구 시에는 새로운 holdout이었지만 오류 감사 후 재사용됐으므로 수정본에 대해 완전히 새롭고 눈가림된 외부 검증이라고 주장하지 않는다. 평가 라벨/효과로 policy 임계값을 조정하지는 않았다.',
          '진단 분포·고정 token 시퀀스·참조 NLL은 원래 실측을 재사용했다. 참조 NLL의 모든 마스크가 수정 후 생성의 마스크와 같음을 검사했다. 이를 새 NLL 반복 측정으로 표현하지 않는다. 준비/진단의 원래 비용과 재측정 비용은 구분한다.',
          table(['방식','수정 전 성공률','수정 후 성공률','응답 변화(192관측)','수정 전 wall/회','수정 후 wall/회'],[[LABELS[r['method']],num(r['original_success']*100,2)+'%',num(r['corrected_success']*100,2)+'%',r['changed_responses'],num(r['original_wall']),num(r['corrected_wall'])] for r in phases]),
          '위 두 phase의 시간 차이는 kernel cache·실행 시점·원래1회차의 NLL 진단 교차 실행 등과 섞여 있다. 알고리즘 절감 효과로 해석하지 않는다. 상세: ORIGINAL_PHASE_RESULTS.csv, PHASE_COMPARISON.csv, TIE_LABEL_CHANGES.csv.']
    lines=['# ALSR Layer Impact Research Report','',f'실행 `{out.name}`. **실제 로컬 모델 실측**. 사전 설계: [ALSR_IMPACT_DESIGN.md]({design_link}). 기존 파일과 지난 ALSR 결과를 보존했다.','',
      '## 1. 핵심 판정',f'확대 조건을 충족한 후보: **{", ".join(promising) if promising else "없음"}**. 성공을 가정하거나 평가에서 임계값을 재조정하지 않았다. 계층 번호는 모두 **0부터 시작**한다.',
      '**새 판단 신호 판정:** '+signal_verdict,
      f'새 gate는 학습 데이터에서 고른 층 **{candidate}** 하나만 대상으로 한다. 보정에서 고정한 위험 임계값은 **{profile["threshold"]}**다. 추정 위험은 KL+top1-flip 회귀값이며 확률로 보정된 오류 확률이 아니다.',
      f'Full 과제 성공률 {base["task_success_mean"]*100:.2f}%, 새 gate {gate["task_success_mean"]*100:.2f}%. 각 방법의 개별 정답 파괴/복구를 아래에 따로 보고한다. 같은 총점이 같은 응답을 의미하지 않는다.','',
      *correction,'## 2. 실행 환경과 공정성',
      f'Apple M4 / 16GiB Mac, 기존 공개 HF 모델 `{m["model"]["repo"]}`, revision `{m["model"]["revision"]}`, Apache-2.0. MLX 0.32.4/MLX-LM0.32.2, 실제 BF16, 28층(hidden1536, FFN8960, attention12/KV2). 기존 Ollama Q4와의 속도 비교가 아니다. Ollama 계층 생략 API를 가정하지 않았다.',
      '모든 방식은 같은 chat template, greedy, MCQ 최대16/open최대48 생성토큰 및 EOS 규칙이다. 3회, 각 문제 안에서 방식 순서와 문제 순서를 무작위화하고 MANIFEST에 기록했다. 같은 문제를 3회 실행한 것을 독립된 192문제로 해석하지 않는다. 전용 장비/완전한 열·캐시 통제는 아니다.',
      '기존 ALSR과 같은 last-position LM head 경로를 모든 방식에 공통 적용했다. native 전체-position logits와 수동 전체-position logits 정합 검사를 통과했다. BF16의 head 행렬 shape에 따른 작은 수치 차이는 SMOKE.json에 별도로 기록하며 공통 head 최적화의 효과를 새 생략 효과에 포함하지 않는다.',
      '자연 생성 wall은 tokenization·실제 prefill/decode·gate·동기화·detokenize를 포함한다. 별도16문제는 동일한 prompt와 full이 만든 동일16-token 시퀀스를 모든 방식에 제공하여 출력 길이 차이를 제거했다. 미래 token은 benchmark 실행기만 가지며 gate는 현재 prompt hidden만 본다.',
      '각 방식 중립 입력 워밍업 및 KV 신규 할당. 한 prompt의 첫 prefill에서 정한 skip mask를 끝까지 유지한다. 생략 캐시offset0, 나머지 offset 정합, 실제 블록/attention/MLP 미호출을 확인한다. 가중치를 제거하지 않아 대부분 메모리는 계속 상주한다.','',
      '## 3. 데이터 및 정보 누출 방지',
      '새 학습32/보정16/평가64. 평가: 산술·논리·코딩·독해 각12, JSON8, 문장제약8. 학습/보정에도 열린 출력 유형을 포함했다. 과거 exact stem 중복0, split 교집합0. 합성 템플릿과 기본 능력은 공유하므로 의미적 중복·실서비스 다양성을 완전히 배제한 데이터는 아니다.',
      '학습만으로 unembedding shortlist·각 층 회귀·후보 층을 결정하고, 보정의 분포 변화만으로 임계값을 선택했다. 정답, 정답 여부, 실제 생략 결과, 기준 미래 token은 live gate에 주지 않는다. 평가 ablation 결과가 생성 전에 존재하지만 PROFILE hash와 재구성 검사로 평가 결과를 policy에 사용하지 않았음을 확인했다.',
      '기존 ALSR의 고정 parser를 유지하고 내용과 형식을 별도 측정한다. 오프라인 감사에서 Python의 true==1 비교로 JSON boolean을 숫자와 혼동하지 않도록 모든 방식에 동일한 정확한 타입 검사를 추가했다. 원시 GENERATION의 기존 grade는 수정하지 않았고 LEGACY_SCORE_COMPARISON.csv에 전후 점수를 보존한다. 문장 제약 통과는 전반적인 사실성·문체 품질 보장이 아니다. 참조 NLL은 각 방식/문제1회이며 비용은 진단 항목으로 분리했다. 64문제로 모집단 정확도 손실1%p 이내를 증명할 수 없다.',
      f'JSON 타입 감사로 판정이 달라진 반복 관측 수: **{audit["json_type_scoring_disagreements"]}**. 상세는 STRICT_SCORE_AUDIT.csv. 이 오프라인 채점 검사는 policy 학습·보정·생략 판단에 사용하지 않는다.','',
      '## 4. 단일 계층 인과 진단',
      '모든28층을 하나씩 제거하여 두 context에서 측정했다: 첫 답변 직전(position0), full 모델의 첫 생성 token 뒤(position1). 첫 token이 EOS면 두 번째 context는 제외한다. 두 번째 context는 오프라인 고정-prefix 진단이다. 자연 생성은 각 방식 자신의 token 이력을 따른다.',
      '전체 어휘 분포로 KL(P_full || P_skip), JS divergence, top-1 변화, 양쪽 top2 logit margin, full top1의 skip 분포 내 signed margin을 계산했다. CSV에는 top10 확률과 전체 float32 logits SHA를 기록한다. top10만으로 KL을 근사하지 않았다. feature 수집 baseline과 일반 baseline logits는 모든 context에서 정확히 같았다.',
      '### position0 평균 KL이 낮은5개 층',table(['층','평균 KL','p90 KL','최대 KL','top1 일치율'],[[r['layer'],num(r['mean_kl']),num(r['p90_kl']),num(r['max_kl']),num(r['top1_agreement']*100,2)+'%'] for r in stable[:5]]),
      '### position0 평균 KL이 높은5개 층',table(['층','평균 KL','p90 KL','최대 KL','top1 일치율'],[[r['layer'],num(r['mean_kl']),num(r['p90_kl']),num(r['max_kl']),num(r['top1_agreement']*100,2)+'%'] for r in reversed(stable[-5:])]),
      '위 순위는 평가 진단 결과이며 새 gate의 후보 선택에 사용하지 않았다. 낮은 평균 KL이 모든 입력·전체 생성에서 안전함을 의미하지 않는다. 전체28층/두 위치 지표는 [LAYER_SUMMARY.csv](LAYER_SUMMARY.csv), 원시 측정은 [LAYER_ABLATIONS.csv](LAYER_ABLATIONS.csv).','',
      '단일 층의 효과를 합쳐4층 동시 생략의 효과로 간주하지 않는다. 층 사이 보상·상호작용 때문에 Static4/Random4의 결과는 별도 실제 생성으로 검증했다.',
      '### 학습에서 선택한 층의 두 위치 비교',table(['후보 층','위치','평균 KL','top1 일치율','full margin','skip margin'],[[r['layer'],r['position'],num(r['mean_kl']),num(r['top1_agreement']*100,2)+'%',num(r['mean_full_margin']),num(r['mean_skip_margin'])] for r in layers if r['layer']==candidate]),
      'position1은 동일 baseline token을 붙여 context 전체를 다시 prefill한 비교다. 실제 incremental KV decode와 수치적으로 완전히 동일하다는 보장은 하지 않는다. 유형별 층 효과는 LAYER_CATEGORY_SUMMARY.csv, margin과 KL의 관계는 LAYER_MARGIN_RELATION.csv에 기록한다. 최종 full logit margin은 오프라인 설명 변수이며 live 신호가 아니다.','',
      '## 5. 저비용 입력 신호의 예측력',
      '현재 층 입력의8차원 고정 투영·RMS·길이, 직전 층 변화, 학습 전용 작은 LM-head shortlist margin/entropy를 사용한다. 현재 층 출력이나 full final logits를 live에서 계산하지 않는다. Shortlist는 전체 어휘에 대한 중간층 confidence가 아니며 잘 보정된 확률로 취급하지 않는다.',
      '층마다 ridge로 KL+flip을 예측한다. layer-mean은 입력을 보지 않는 깊이 기준선이다. 기존 Sketch는 원래 ALSR profile을 그대로 고정하고 같은8층/position0에서 비교했다. 기존 Sketch와 새 신호의 학습 표본·특징 시점이 다르므로 예측력 차이를 신호 하나만의 인과 효과로 주장할 수 없다.',
      '### 보정에서 고정한 생략 기준',table(['임계값','허용된 문제','top1 flip','허용군 평균 KL','조건 통과'],[[r['threshold'],r['selected'],r['flips'],num(r['mean_kl'],5),r['eligible']] for r in profile['trials']]),
      '보정에서 후보 층의 모든 문제는 top1을 유지했지만 비어 있지 않은 모든 허용군이 mean KL<=.02 조건을 넘었다. 따라서 임계값0은 top1 분류 실패 때문이라고 단정할 수 없으며 엄격한 분포 보존 기준과 위험 회귀의 선택력에 함께 기인한다. 이 기준을 평가에서 완화하지 않았다.',
      '임계값0이면 gate를 상수 Full로 접어 추가 probe를 제거할 수 있다. 이번 실측은 판단 오버헤드를 확인하기 위해 probe를 유지했다. 상수 접기를 해도 Baseline과 같아질 뿐 실제 계층 연산 절감은 생기지 않는다. 이 비용을 정책의 본질적인 필수 비용이라고 과장하지 않는다.',
      table(['범위','위치','신호','flip AUROC','위험 MAE','위험 Spearman'],[[r['scope'],r['position'],r['signal'],num(r['flip_auroc']),num(r['risk_mae']),num(r['risk_spearman'])] for r in signals]),
      '### 신호 비교의 질문 단위 bootstrap 95% CI',table(['범위','대조','AUROC 차이 [CI]','MAE 차이 [CI]'],[[r['scope'],r['control'],f'{num(r["auroc_delta"])} [{num(r["auroc_delta_ci95_low"])}, {num(r["auroc_delta_ci95_high"])}]',f'{num(r["risk_mae_delta"])} [{num(r["risk_mae_delta_ci95_low"])}, {num(r["risk_mae_delta_ci95_high"])}]'] for r in paired]),
      'AUROC 차이 양수/MAE 차이 음수가 새 신호에 유리하다. CI가0을 포함하면 우월성은 불확실하다. 전체 층 pooled AUROC는 깊이 식별만으로도 높아질 수 있으므로 [SIGNAL_METRICS.csv](SIGNAL_METRICS.csv)의 층별 평가와 candidate1을 함께 본다. 모든 회귀/원시 단일 신호는 탐색적 비교이며 다중 비교 보정된 기술 우월성 증거가 아니다.','',
      f'공통8층/position0의 AUROC는 기존 {pool_old["flip_auroc"]:.3f} → 새 신호 {pool_new["flip_auroc"]:.3f}다. 하지만 실제 선택 후보 층의 MAE는 layer-mean보다 {candidate_pair["risk_mae_delta"]:+.4f} 크고 CI [{candidate_pair["risk_mae_delta_ci95_low"]:+.4f}, {candidate_pair["risk_mae_delta_ci95_high"]:+.4f}]다. 따라서 전체 pool에서의 예측 개선을 저위험 허용 구간의 정확한 보정으로 간주할 수 없다.',
      '## 6. 실제 생성 결과와 계산 비용',
      '표는64문제 한 회의 합계/비율을3회 평균한 값이다. Static4/Random4와 1층 방식의 예산은 다르다. Stable1/Random1은 1층 기준선이며 gate가 거절한 경우 gate의 평균 생략량은 더 작다. 이를 동일 연산 예산 비교로 포장하지 않는다.',
      '기존 Sketch/Static4/Random4의 후보는8개 중간층, 새 Stable1/gate의 학습 탐색과 Random1은 내부25층(2..26)이다. 후보24는 기존8개 pool 밖에 있다. 후보 범위·생략 예산·학습 표본이 동시에 다르므로 정책 결과의 차이를 새 입력 신호만의 효과로 해석하지 않는다. 신호 자체는 공통8층 비교를 따로 제공한다.',
      table(['방식','과제 성공률','객관식 정답률','형식 준수율','평균 생략층','생성 wall s','CPU s','출력 토큰','gate ms/요청','MLX peak GiB'],[[LABELS[r['method']],num(r['task_success_mean']*100,2)+'%',num(r['mcq_accuracy_mean']*100,2)+'%',num(r['format_compliance_mean']*100,2)+'%',num(r['mean_skipped_layers_mean'],2),num(r['wall_seconds_mean']),num(r['cpu_seconds_mean']),num(r['output_tokens_mean'],1),num(r['mean_gate_wall_seconds_mean']*1000),num(r['mean_mlx_peak_bytes_mean']/2**30)] for r in results]),
      'CPU는 이 Python 프로세스의 실제 CPU 시간이며 GPU 전체 연산량이 아니다. MLX peak는 요청별 allocator peak의 평균이다. 프로세스 lifetime max RSS는 원시 로그에 따로 남겼고 방식별 peak로 해석하지 않는다. GPU 경과 시간·FLOPs 하드웨어 카운터·전력·금전 비용은 **미측정**이다.',
      '방식들을 같은 프로세스에서 교차 실행하므로 모델과 작은 두 probe의 가중치는 공통 상주한다. 이 메모리 표는 공통 실행기 안에서 요청의 활성화/KV 포함 peak를 비교한 것이며 각 방식을 독립 프로세스로 배포한 최소 메모리 비교가 아니다.',
      '새 gate timer는 앞선 hidden 계산을 materialize한 뒤 feature+prediction 시간을 잰다. 이때 생기는 동기화의 전체 latency 영향은 요청 wall에 포함된다. 기존 Sketch timer에는 앞서 queue된 GPU 계산 대기가 섞일 수 있어 둘의 gate timer만 직접 비교해 우월성을 주장하지 않는다.',
      '### 동일16-token workload 및 분석 matmul FLOPs',table(['방식','fixed wall s','fixed CPU s','FLOPs/Full','자연 생성 FLOPs/Full','참조 NLL'],[[LABELS[r['method']],num(r['fixed_wall_seconds_mean']),num(r['fixed_cpu_seconds_mean']),num(r['fixed_analytical_matmul_flops_mean']/base['fixed_analytical_matmul_flops_mean'],4),num(r['analytical_matmul_flops_mean']/base['analytical_matmul_flops_mean'],4),num(r['reference_nll_once'])] for r in results]),
      'FLOPs는 dense Q/K/V/O+MLP+attention matmul+LM head와 probe matmul의 MAC×2 분석 추정이다. norms/softmax/비선형/메모리 전송·kernel fusion 효과 등은 제외한다. 실제 미호출 검사와 결합해 구조적 연산 감소를 확인하지만 GPU FLOPs를 측정한 것으로 표현하지 않는다. 자연 생성 FLOPs에는 응답 길이 변화가 섞이므로 동일 token 부하를 별도 제시한다.',
      '### Full 대비 paired 95% CI',table(['방식','성공률 차이 pp [CI]','자연 wall 비율 [CI]','fixed wall 비율 [CI]'],[[LABELS[r['method']],f'{r["accuracy_delta"]*100:.2f} [{r["paired_bootstrap_low"]*100:.2f}, {r["paired_bootstrap_high"]*100:.2f}]',f'{r["natural_wall_ratio"]:.3f} [{r["natural_wall_ci95_low"]:.3f}, {r["natural_wall_ci95_high"]:.3f}]',f'{r["fixed_wall_ratio"]:.3f} [{r["fixed_wall_ci95_low"]:.3f}, {r["fixed_wall_ci95_high"]:.3f}]'] for r in comparison if r['control']=='baseline']),
      'CI는 질문 cluster와 반복 축을 재표본화한다. 3회 반복은 길이·입력 다양성에 대한 독립 표본을 늘리지 않는다.0차이 CI도1%p 모집단 비열등의 증명이 아니므로 보수적 exact 하한을 COMPARISONS.csv에 병기했다. 반복 평균/표준편차/최소/최대는 RESULTS.csv, 개별 반복은 REPEAT_METRICS.csv. 실행 순서 상관은 ORDER_DIAGNOSTICS.csv의 기술통계이며 캐시·열에 대한 인과 통제가 아니다.','',
      '### CPU 시간의 paired 비율과95% CI',table(['방식','부하','CPU/Full [CI]'],[[LABELS[r['method']],r['workload'],f'{r["ratio"]:.3f} [{r["ci95_low"]:.3f}, {r["ci95_high"]:.3f}]'] for r in cpu]),
      '새 gate의 자연 생성 CPU 시간 점추정은 줄었지만 CI가1을 포함한다. fixed 부하에서는 CPU 점추정도 줄지 않았다. 실제 계층 생략0이며 dense matmul 추정량도 감소하지 않아 이 CPU 변동을 전체 모델 추론 연산 절감으로 해석하지 않는다.',
      'Static4는 Full보다 평균 참조 NLL이 낮으면서도 과제 성공률은 크게 나빠졌다. NLL·첫 token 안정성·형식·내용 정답률은 서로 다른 지표이며 어느 하나의 개선만으로 품질 유지라고 판단하지 않는다.',
      '## 7. 총점에 가려진 개별 변화',table(['방식','정답→오답 / 회','오답→정답 / 회','응답 변화 / 회'],[[LABELS[r['method']],num(r['correct_to_wrong_mean'],2),num(r['wrong_to_correct_mean'],2),num(r['different_responses_mean'],2)] for r in results]),
      f'Stable1은 Full과 총 성공률이 같지만 매회 정답 {stable_result["correct_to_wrong_mean"]:.0f}개를 잃고 {stable_result["wrong_to_correct_mean"]:.0f}개를 얻었다. 형식 준수율은 {base["format_compliance_mean"]*100:.2f}% → {stable_result["format_compliance_mean"]*100:.2f}%다. 출력 token은 {base["output_tokens_mean"]:.0f} → {stable_result["output_tokens_mean"]:.0f}개/회로 증가했다. 같은 token 부하에서는 약 {(1-stable_compare["fixed_wall_ratio"])*100:.2f}% 빠르지만 자연 생성은 {(stable_compare["natural_wall_ratio"]-1)*100:.2f}% 느리다. 총점 보존을 출력 품질 보존으로 해석하면 안 된다.',
      table(['문제','정답','Full 응답','Stable1 응답','전환'],[[r['question'],r['reference'],r['baseline_answer'],r['answer'],'오답→정답' if r['final_correct'] else '정답→오답'] for r in examples]),
      '### 첫 top1이 같아도 전체 생성이 달라지는가?',table(['방식','관측 수(반복 포함)','고유 문제','응답 변화','정답 파괴','오답 복구'],[[LABELS[r['method']],r['n_repeat_observations'],r['unique_questions'],r['response_changed'],r['correct_destroyed'],r['correct_repaired']] for r in agreement]),
      '위 표는 실제1층을 생략했고 최초 top1이 같았던 경우만 집계한다. 모든 생략 정책에 일반화한 비율이 아니다. 조건부 표본 및 반복 수를 고유 문제 수와 분리했다. [TRANSITIONS.csv](TRANSITIONS.csv)는 모든 문제/방식/반복의 원문과 전환을 보존한다.',
      f'Random1에서는 최초 top1이 같아도 정답을 잃은 관측이{len(random_losses)}건(고유{len({r["task_id"] for r in random_losses})}문제) 있었다. 첫 token이 구조 기호/답변 접두어일 수도 있어 최초 token 안정성은 의미 내용이나 전체 생성 안전성의 충분조건이 아니다. Stable1의 최초 top1 유지군에서는 정답 파괴가 없었지만 응답 변화가 발생했다. 이 제한된 조건부 결과를 모든 층에 일반화하지 않는다.',
      '### 유형별 과제 성공률',table(['유형']+[LABELS[r['method']] for r in results],[[cat]+[num(next(x['success'] for x in categories if x['category']==cat and x['method']==r['method'])*100,2)+'%' for r in results] for cat in sorted({r['category'] for r in categories})]),
      '실패/응답 변화 사례의 prompt·reference·기준/수정 원문·생략층은 [FAILURE_CASES.csv](FAILURE_CASES.csv)에 전부 기록한다. 생성 길이 상한에 따른 형식 실패와 내용 오류를 구분한다.','',
      '## 8. 기존 기술과의 관계',
      '* [ShortGPT](https://arxiv.org/abs/2403.03853)는 층 입출력 유사성으로 Block Influence를 정의하고 정적 pruning을 연구했다. 본 연구는 단일 층 제거의 최종 분포 인과 효과를 분석하고 현재 층 실행 전 특징을 진단한다. ShortGPT 전체 재현이나 우월성 증명이 아니다.',
      '* [Tuned Lens](https://arxiv.org/abs/2303.08112)는 중간 hidden에서 최종 분포를 읽는 학습 affine probe를 연구했다. 여기서는 작은 shortlist와 ridge를 쓰며 전체 tuned-lens 학습을 재현하지 않았다. 중간 표현 probe라는 발상은 기존 기법이다.',
      '* [LayerSkip](https://arxiv.org/abs/2404.16710)는 layer dropout/early-exit loss 학습과 self-speculative decoding을 결합한다. 여기서는 모델 가중치를 학습하지 않고 원래 Qwen에 branch-before-block을 적용했다.',
      '* [SkipDecode](https://arxiv.org/abs/2307.02628)는 token별 exit와 batching/KV 제약을 다룬다. 여기서는 첫 prefill의 마스크를 고정하므로 동적 decode 재진입이나 batch 최적화를 주장하지 않는다.',
      '문헌 검토는 대표 기법과의 범위를 구분한 것으로 전 세계 새로움을 확정하는 전수 조사가 아니다. 새로운 발명/독립 기술이라고 주장하지 않는다.','',
      '## 9. 판정과 한계',table(['방식','품질 점추정 목표','10% latency CI 목표','실제 skip/추정 연산 감소','확대 후보'],[[LABELS[r['method']],r['quality_point_within_target'],r['speed10_confident'],r['true_skips_and_estimated_compute_reduction'],r['promising']] for r in verdict]),
      '가중치 학습 없이 raw single-layer skip은 downstream 계산 및 여러 token의 오차를 바꾼다. 단일 next-token KL/flip을 잘 예측하는 것과 전체 정답·형식·길이를 보존하는 것은 별개다. 단일 안정 층, 작은 보정셋, 합성 영어 단문, BF16/Qwen1.5B 한 모델,3회, Python 계측 및 host-GPU 동기화 비용이라는 한계가 있다.',
      '10% 지연 감소는 앞선 ALSR에서 이어온 확대 목표다. 이번 1층 gate는 구조상 전체28층 중 최대1층만 생략하므로 큰 속도 향상을 기대하기 어렵다. 10% 미달 자체를 모든 작은 효과가 없다는 증거로 해석하지 않으며 실제 비율/CI를 그대로 제시한다. 신호의 예측력, 안정 층의 존재, 실용적인 전체 연산 절감은 서로 다른 판정이다.',
      '새 신호가 예측 지표에서 일부 좋아도 실제 품질-지연 동시 조건을 충족하지 못하면 연산 절감 성공으로 판단하지 않는다. 실패를 숨기거나 test 기준으로 정책을 보정하지 않는다. 다음 단계는 평가 결과를 다시 학습시키는 것이 아니라 별도 학습 데이터에서 다중-token/완성답변 변화와 단일-token 위험의 연결을 검증하는 것이다.','',
      '## 10. 비용 범위·보존·재현',
      f'실측 모델 단계 wall 합계: {meta["all_stages_wall_seconds"]:.3f}s. '+(f'원래 모델 단계 {meta["original_stage_wall_seconds"]:.3f}s + 수정 재측정 단계 {meta["canonical_rerun_wall_seconds"]:.3f}s의 합이며 중간 감사·테스트·보고서 작성 공백을 포함한 연속 경과시간이 아니다. 수정 fit 비용은 FIT_COST.json, margin 재검사 비용은 TIE_MARGIN_RECHECK.csv. ' if meta.get('corrected_tie_rule') else '')+'오프라인 원래 단계 비용: OFFLINE_COSTS.csv; 원래 shortlist/fixed-sequence 준비: PREPARATION.csv; 이번 load: LOAD.json; 이번 워밍업: WARMUPS.csv. 원래 fit 비용은 별도 미측정이며 원래 job wall에 포함된다. 오프라인 연구비와 live inference비를 혼합해 절감률을 만들지 않았다.',
      f'기존 {audit["preserved_files_unchanged"]}개 파일 해시 확인. 실제 호출/캐시/토큰/마스크 검사, probe baseline parity, profile train/validation 재구성, reference gate 호출0. 자동 테스트: **{test_result}**. 테스트 모의 응답을 실측표에 사용하지 않았다.',
      'GATE_BRANCH_SMOKE.json은 중립 입력으로 실제 모델의 gate 생략 분기를 강제한 소프트웨어 검사다. 후보 블록에 trap을 설치해 생략 시 미호출과 비생략 시 예외라는 양성 대조를 확인하고 원래 프로파일을 보존한다. 강제 임계값의 답변·속도를 평가 성능이나 성공 사례로 사용하지 않는다.',
      '```bash\n# 동점 수정을 진단 수집 전부터 적용하고 모든 진단/생성을 실제 재계산\n.alsr-venv/bin/python -B -u -m src.alsr_impact_fixed_run '+str(Path(m.get('diagnostic_source',out))/'MANIFEST.json')+'\n# 분석/보고서는 기존 산출물이 있으면 덮어쓰지 않음\n.alsr-venv/bin/python -B -m src.alsr_impact_analysis <새 실행 폴더>\n.venv/bin/python -B -m src.alsr_impact_cpu <새 실행 폴더>\n.alsr-venv/bin/python -B -u -m src.alsr_impact_smoke <새 실행 폴더>\n.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json\n.alsr-venv/bin/python -B -m unittest discover -s tests -p "test_alsr_impact*.py"\n.venv/bin/python -B -m src.alsr_impact_report <새 실행 폴더> "<실제 테스트 결과>"\n```',
      '고정 HF revision 캐시가 있어야 하며 MLX 실제 실행에는 Metal 권한이 필요하다. seed와 모든 실행 순서는 MANIFEST에 저장된다.','',
      '[결과](RESULTS.csv) · [층별 효과](LAYER_SUMMARY.csv) · [신호 비교 CI](SIGNAL_PAIRED.csv) · [전환](TRANSITIONS.csv) · [비교 그림](LAYER_IMPACT_COMPARISON.png) · [효율 그림](LAYER_IMPACT_EFFICIENCY.png) · [감사](AUDIT.json)']
    with (out/'ALSR_LAYER_IMPACT_REPORT.md').open('x') as f:f.write('\n\n'.join(lines)+'\n')
    plots(out);print(str(out/'ALSR_LAYER_IMPACT_REPORT.md'))

if __name__=='__main__':report(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else '미입력')
