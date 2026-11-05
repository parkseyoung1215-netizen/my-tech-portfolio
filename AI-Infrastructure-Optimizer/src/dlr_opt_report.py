"""Korean measured optimization report and standalone plots, never overwrite."""
import json,os,statistics,sys
from pathlib import Path
from collections import Counter
from .controlled_analysis import read_decoded
from .alsr_io import write_union
from .adaptive_experiment import write_json
METHODS=('baseline','old_dlr','device','lightweight')
LABEL={'baseline':'Baseline','old_dlr':'기존 DLR','device':'Device-Resident','lightweight':'Lightweight'}
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in row)+' |' for row in rows])

def render(folder):
    a=Path(folder);p=Path(json.loads((a/'SOURCE.json').read_text())['experiment']);m=json.loads((p/'MANIFEST.json').read_text());audit=json.loads((a/'AUDIT.json').read_text())
    results=read_decoded(a/'DLR_OPTIMIZATION_RESULTS.csv');comparisons=read_decoded(a/'DLR_OPTIMIZATION_COMPARISONS.csv');cases=read_decoded(a/'DLR_OPTIMIZATION_CASES.csv');verdict=read_decoded(a/'DLR_OPTIMIZATION_VERDICTS.csv');repeats=read_decoded(a/'DLR_OPTIMIZATION_REPEATS.csv')
    qualified=[r for r in verdict if r['observed_minimum_pass'] and r['method'] in ('device','lightweight')];grade='B' if qualified else 'C'
    conclusion='일부 개선은 있으나 추가 검증 필요' if qualified else '최적화 실패'
    fidelity=[]
    for path in p.glob('FRESH_KERNEL_ABLATION_*.csv'):fidelity.extend(read_decoded(path))
    lines=['# DLR_OPTIMIZATION_REPORT','',f'**최종 결론: {grade} — {conclusion}.** 실행 `{p.name}`. 무료 로컬 실제 추론. 단일 M4 16GiB, Qwen2.5-1.5B-Instruct BF16. 기존 파일/가중치/데이터 보존; 유료 API 및 GitHub 업로드 없음.','',
    f'최소 관측 기준을 충족한 제한적 조건 수: {len(qualified)}. 작은 표본, 단일 장치, synthetic 문제, Device fresh-kernel 수치 차이 때문에 모집단 무손실·일반적인 전체 연산 감소를 입증하지 않는다. 관측 기준을 통과하지 못한 방식/집합도 모두 아래에 포함한다.','',
    '## 1. 기존 DLR의 병목','',
    '기존 src/dlr_runtime.py는 8/12/16/24번층에서 Attention 및 KV cache를 정상 계산한 다음, post-attention RMSNorm의 현재 입력과 최근 fresh MLP anchor 입력의 상대 L2 거리가0.5이하이면 gate/up/down projection을 포함한 MLP residual 전체를 이전 값으로 대체한다. 최대1토큰 연속 재사용, prefill 재사용 없음. KV Cache와 달리 새 토큰의 FFN 근사값을 사용한다.','',
    '각 후보 판단 시 mx.eval(z)로 앞선 GPU 작업을 기다리고, 거리 reduction을 mx.eval한 뒤 distance.item()으로 Python 분기한다. 이전 보고서에서 자연생성 평균 DLR 재사용은 기존32문제8회/신규32문제16회였고, 모델 matmul 추정 감소는0.00858%/0.01972%였는데 wall은5.9%/7.8% 증가했다. 이는 이번 새 측정과 섞지 않은 과거 결과다. gate당 대략5*d+4개 scalar arithmetic 대비, 재사용하면 MLP3projection의6*d*I FLOPs를 생략한다. Qwen d=1536,I=8960이면 gate약7684ops, MLP약82575360FLOPs다. 하지만 낮은 재사용 빈도, host round-trip, pipeline serialization 및 cache/선택 비용 때문에 이 산술 비율만으로 속도를 예측할 수 없다.','',
    '특히 pre_gate_sync 시간에는 원래 필요한 Attention/앞선 모델 연산 대기가 포함된다. 이를 전부 추가 오버헤드로 더하거나 wall에서 빼지 않는다. gate_wall과 pre_gate_sync는 서로 다른 범위이며 둘 모두 전체 wall에 포함돼 있다. 단일 profiler 분해 없이 원인별 정확한 인과 비중은 식별하지 못한다.','',
    '## 2. 실제 구현과 가능성','',
    'A Baseline/B 기존 DLR은 원래 DLRRuntime을 그대로 호출해 이전 정책·native 커널을 유지했다. C Device는 동일 거리 기준을 MLX array로 계산하고, 자체 Metal GEMV 커널의 uniform if(reuse) 분기가 weight 읽기와 MAC loop 이전에 return하도록 구현했다. gate/up과 down의 GPU 출력 counter는 fresh일 때 합3, reuse일 때0이다. mx.where(fresh_mlp,cached)로 양쪽 projection을 모두 계산하는 가짜 생략 방식이 아니다. 재사용 중에도 커널 launch·중간 zero write·cached output copy는 발생한다.','',
    '설치된 MLX에는 mx.cond가 없었지만 custom Metal kernel은 가능했다. 따라서 device 방식 자체가 불가능하다고 처리하지 않았다. 동적 선택/anchor 갱신에는 mx.where를 쓰되, 고비용 projection 자체는 커널 내부에서 조건부 실행한다. 세부 소스는 dlr_opt_kernels.py다.','',
    'D Lightweight는 이미 argmax 과정에서 읽은 현재 입력 token ID가 최근 fresh anchor의 ID와 같고 age=0일 때 네 후보 MLP 출력을 재사용한다. 추가 LLM 호출·hidden-state 거리 계산·GPU scalar read가 없다. 단, 동일 token ID라도 attention 문맥과 hidden state는 다르므로 안전성 보장은 없고 실패는 실제 출력으로 판정한다. 후보4층과 최대연속1회 제한은 유지했다.','',
    'Device/Lightweight forward에는 .item 또는 mx.eval로 gate를 읽는 코드가 없다. 일반 greedy token 읽기(토큰당1회)와 요청 시작/끝 동기화는 여전히 있다. Device branch counter는 요청 종료 시1 batch로 회수하며 그 시간도 wall에 포함한다. 따라서 모든 호스트 동기화를 제거했다고 주장하지 않는다.','',
    'Device fresh GEMV는 stock MLX GEMV와 reduction/rounding 순서가 다르다. 개발 random activation에서 native MLP 최대 절대 차이0.125, 상대 L2약0.00388을 관측했다. 개발용 두 prompt의 forced-fresh 생성 token은 Baseline과 같았지만, 모든 평가 질문에서 fresh-kernel parity가 입증된 것은 아니다. Device와 기존 DLR 차이는 sync뿐 아니라 kernel 구현 및 수치 오차가 섞인다.','',
    f'별도 사후 fresh-kernel 점검은 고정된 부분집합 {len(set(r["task_id"] for r in fidelity))}문제를3회, 총{len(fidelity)}건 실행했다. 재사용을 끈 Device의 Baseline token열 일치는 {sum(r["tokens_match_baseline"] for r in fidelity)}/{len(fidelity)}건이었다. 이를 학습/임계값 조정에 쓰지 않았으며 모든 평가 문제의 수치 동일성을 증명하지 않는다. 주 비교와 실행 순서가 달라 이 추가 점검의 latency는 주 성능 비교에서 제외했다.','',
    '개발 파일럿에서는 반복 hello8회를 요청했을 때 Baseline/Lightweight는8개, 기존 DLR/Device는9개를 출력했다. 이는 실제 재사용의 품질 실패 사례이며 버리지 않았다. 최초 Metal 컴파일은 scalar reuse 입력을 pointer처럼 사용해 실패했고, 입력을 길이1 array로 수정한 뒤 통과했다. 실패한 protocol 폴더도 보존했다. 평가 중 정책/커널/채점을 조정하지 않았다.','',
    '[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [where](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.where.html), [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html). 기존 [activation caching 연구](https://arxiv.org/abs/2512.16843)와 겹치는 로컬 엔지니어링 실험이며 새 발명의 증거가 아니다.','',
    '## 3. 공정 조건과 평가 범위','',
    f'기존32문제와 신규32문제, 반복복사 long_high4문제와 서로 다른 단어 복사 long_low4문제를 사용했다. high/low는 사전 입력 설계 가설이며 결과를 보고 유리한 문제를 선별한 것이 아니다. 전체72문제×4방식×3회 자연생성864건, 동일 Baseline token stream을 teacher forcing으로 실제 실행하는864건, 총1728건이다. 시드{m["seed"]}. 같은 system/user template, greedy EOS, short32/long96토큰 상한. 모든 방식에 동일하게 적용했다. 정답은 scoring에만 사용한다. 복사 과제의 원문이 prompt에 포함된 것은 과제 정의이며, 채점 결과나 정답 여부를 gate에 제공한 것은 아니다.','',
    '워밍업/컴파일 후 측정하고 질문 및 방식 순서를 무작위 교차했다. schedule·sequence·position·session·원본 체크포인트를 보존한다. 저장·모델 로드·개발 audit·warmup 시간은 요청 latency 밖에서 별도 기록했다. 각 요청은 fresh KV/cache 상태에서 시작한다. 장치 온도/OS 부하를 완전히 통제하지 못했으며 반복3회는 독립적 모델 출력 표본3개가 아니다.','',
    '신규 exact stem 중복은 과거 JSON/CSV를 재귀 검사했다. 합성 템플릿/문제 개념의 유사성은 남아 외부 일반화를 증명하지 않는다. MCQ 선택지 추출, typed JSON, 사전 제한한 color 표현, 40단어 전체 literal copying으로 채점했다. 형식과 내용은 분리한다. 긴 답변의 일부 prefix만 맞아도 정답으로 처리하지 않는다. 열린 문장의 의미·환각 전반을 평가하는 데이터는 아니다.','',
    f'가중치 revision `{m["model"]["revision"]}`, 패키지 `{json.dumps(m["packages"],sort_keys=True)}`. 모든 pinned 가중치/설정 파일 해시를 로드 시 확인했다.','',
    '## 4. 전체 자연생성 비교','']
    for ds in ('all72',*m['datasets']):
        rr=[r for r in results if r['phase']=='natural' and r['dataset']==ds]
        lines+=['### '+ds,'',table(['방식','정답률','형식 준수','출력 token열 일치','wall 평균±SD(s)','모델시간(s)','CPU 평균±SD(s)','입력/출력토큰','ms/출력토큰','MLX peak MiB','실제재사용/반복'],[[LABEL[r['method']],f'{r["content_accuracy_mean"]*100:.2f}%',f'{r["format_compliance_mean"]*100:.2f}%',f'{r["output_agreement_mean"]*100:.2f}%',f'{r["wall_seconds_mean"]:.3f} ± {r["wall_seconds_sample_std"]:.3f}',f'{r["model_wall_seconds_mean"]:.3f}',f'{r["cpu_seconds_mean"]:.3f} ± {r["cpu_seconds_sample_std"]:.3f}',f'{r["input_tokens_mean"]:.0f}/{r["output_tokens_mean"]:.0f}',f'{r["mean_seconds_per_output_token_mean"]*1000:.3f}',f'{r["mlx_peak_bytes_mean"]/(1<<20):.2f}',f'{r["reused_mlp_calls_mean"]:.0f}'] for r in rr]),'']
    lines+=['wall·CPU·모델시간·토큰은 각 집합의 전체 문제 합계다. 출력 토큰 수는 실제 실행한 EOS 생성 단계도 포함한다. 토큰당 시간은 prefill을 포함한 전체 wall/output 수이며 decode-only 지연이 아니다. RSS는 프로세스 lifetime peak라 방식별 독립 peak가 아니고, MLX peak도 장치 전체 메모리가 아니다.','',
    '## 5. 동일 토큰 수·동일 입력 위치 비교','',
    '자연생성 1회차 Baseline의 토큰열을 고정해 모든 방식에서 실제 forward를 실행했다. 고정 응답의 채점을 독립적인 생성 정답률로 이용하지 않는다. 입력·출력 토큰 수와 prefix 위치가 일치하는지 감사했다.','']
    for control in ('baseline','old_dlr'):
        lines+=['### 고정열 비교 대상: '+LABEL[control],'',table(['집합','방식','wall 감소율 [95%CI]','CPU 감소율','모델 matmul 추정 감소','gate 포함 산술 proxy 감소','3회 wall 비율'],[[r['dataset'],LABEL[r['method']],f'{(1-r["wall_seconds_ratio"])*100:.2f}% [{(1-r["wall_seconds_ci95_high"])*100:.2f}, {(1-r["wall_seconds_ci95_low"])*100:.2f}]',f'{(1-r["cpu_seconds_ratio"])*100:.2f}%',f'{(1-r["analytical_matmul_flops_ratio"])*100:.4f}%',f'{(1-r["limited_net_arithmetic_proxy_ratio"])*100:.4f}%',', '.join(f'{x:.4f}' for x in r['repeat_wall_ratios'])] for r in comparisons if r['phase']=='fixed' and r['control']==control]),'']
    lines+=['양의 감소율은 감소, 음수는 증가다. CI는 질문과 반복 축을 함께 재표집한 paired bootstrap이며, 3회 평균·SD는 RESULTS에 모두 저장했다. long은 각 4문제로 특히 불확실성이 크다. 정확도 차이 0 또는 CI[0,0]으로 모집단 비열등성을 증명하지 않는다.','',
    '## 6. 판단 추가 비용과 실제 생략','',table(['집합','방식','host gate 읽기/반복','pre-gate eval/반복','거리 GPU 평가/반복','ID 비교/반복','기존 gate 시간(s)','device graph 구성 CPU(s)','device gate GPU 시간','추정 gate 산술 ops'],[[r['dataset'],LABEL[r['method']],f'{r["gate_host_scalar_reads_mean"]:.0f}',f'{r["explicit_pre_gate_evals_mean"]:.0f}',f'{r["gate_queries_mean"]:.0f}',f'{r["id_comparisons_mean"]:.0f}',f'{r["gate_wall_seconds_mean"]:.4f}',f'{r["gate_graph_host_cpu_seconds_mean"]:.4f}','미측정',f'{r["gate_arithmetic_estimate_mean"]:.0f}'] for r in results if r['phase']=='natural']),'',
    '위 표의 기존 gate wall과 Device graph 구성 CPU는 범위가 달라 직접 차감하지 않는다. Device gate GPU 단독 시간은 추가 동기화 없이 분리할 수 없어 미측정이며, 계산·대기는 wall/model 시간에 포함된다. Lightweight ID 정책 점검 비용은 gate_wall/gate_cpu에 기록했다. id_comparisons는 decode 단계별 정책 점검 수이며 age 단락 조건이면 ID equality 자체는 실행하지 않는다. CPU 정수 비교를 GPU FLOPs로 표현하지 않았다. counter batch 읽기, 선택·anchor 갱신, 커널 launch도 실제 wall에 포함한다.','',
    'FLOPs는 dense MAC=2의 분석적 추정이며 GPU 실측값이 아니다. gate 산술 약5*d+4, 선택 element 수는 별도 저장했다. norm·activation·비교·메모리 이동을 포함한 완전한 GPU 명령 계수가 아니므로 제한적 산술 proxy를 전체 AI 연산량으로 부르지 않는다. 실제 MAC 루프 우회 counter와 추정 FLOPs를 구분한다. GPU 전력·금전 비용·GPU 실행시간은 미측정이다.','',
    '## 7. 품질 실패와 개별 변화','',table(['집합','방식','Correct→Wrong/반복','Wrong→Correct/반복','출력 불일치/전체요청'],[[ds,LABEL[method],f'{sum(c["transition"]=="Correct->Wrong" for c in cases if c["phase"]=="natural" and c["dataset"]==ds and c["method"]==method)/3:.2f}',f'{sum(c["transition"]=="Wrong->Correct" for c in cases if c["phase"]=="natural" and c["dataset"]==ds and c["method"]==method)/3:.2f}',f'{sum(not c["output_match"] for c in cases if c["phase"]=="natural" and c["dataset"]==ds and c["method"]==method)}/{len(m["datasets"][ds])*3}'] for ds in m['datasets'] for method in METHODS[1:]]),'']
    losses=[c for c in cases if c['phase']=='natural' and c['repetition']==1 and c['transition']=='Correct->Wrong']
    differences=[c for c in cases if c['phase']=='natural' and c['repetition']==1 and not c['output_match'] and c not in losses]
    for c in (losses+differences)[:16]:lines += [f'### {c["dataset"]}/{LABEL[c["method"]]}/{c["task_id"][:10]}',f'- 질문: {c["question"].replace(chr(10)," / ")}',f'- 기준: {c["reference"]!r}',f'- Baseline: {c["baseline_answer"]!r}',f'- 비교 출력: {c["answer"]!r}; 전환 {c["transition"]}, 재사용{c["reused_mlp_calls"]}회.','']
    lines+=['전체 응답·채점·실패·순서는 CASES/FAILURES/GROUPS CSV에 보존했다. 위 예시만으로 결론을 선택하지 않는다. Baseline이 원래 전부 실패한 긴 생성 조건에서 같은 0%를 얻은 것을 품질 유지 성공으로 처리하지 않는다.','',
    '## 8. 최소 성공 기준과 결론','',table(['집합','방식','동일 관측 정답률','정답 손실 없음','5% 동일열 감소','3회 모두5%감소','순산술proxy감소','최소 관측 통과'],[[r['dataset'],LABEL[r['method']],r['equal_observed_accuracy'],r['no_any_observed_correct_loss'],r['equal_stream_wall_gain_at_least_5pct'],r['all_three_repeats_gain_at_least_5pct'],r['equal_stream_net_proxy_reduced'],r['observed_minimum_pass']] for r in verdict]),'',
    f'**최종 분류 {grade}: {conclusion}.** 관측 통과 조건만으로 무손실 일반화나 신규 기술 발견을 주장하지 않는다. CPU 시간도 반드시 따로 보고, wall이 줄어도 CPU가 늘면 CPU 절감이라고 표현하지 않는다. 메모리가 줄지 않으면 메모리 절감 성공이 아니다.','',
    '이번 반복 단어 집합에서 Baseline 정답률50%가 기존 DLR/Device에서는25%로 떨어졌다. Lightweight는50%로 같았지만 Correct→Wrong1개와 Wrong→Correct1개가 상쇄됐으며, go 복사는40단어 대신 상한96단어까지 생성했다. 집합 정답률만 보면 이 손상을 놓치므로 토큰열 일치율과 개별 전환을 별도 판정했다. 동일열 Lightweight의 평균 속도 이득은약3.23%로 목표5% 미만이다.','',
    '## 자동 테스트와 실제 실행 검증','',
    '기존 전체 snapshot 테스트251개:247개 통과, NumPy 환경 사유4개 건너뜀(73.387초). 격리 MLX/NumPy 환경의 관련17개 테스트는 전부 통과(0.210초)했으며 건너뛴4개 수치 검사를 포함한다. 추가 실제 모델 MLP tap은4개 forward에서 후보별 실제 MLP3회와 재사용1회를 확인했다. 작은 poisoned-weight GPU 제어 실험에서도 reuse 분기는 finite cached output/MAC counter0, fresh 분기는 NaN/counter3으로 확인했다. 이 소프트웨어 제어 실험을 실제 과제 성능으로 보고하지 않았다.','',
    '## 9. 질문별 답변','',
    '1. **기존 DLR이 느려진 원인:** 낮은 재사용률에 비해 후보별 host 결과 회수와 직렬화가 많았다. 추가 거리 산술은 작지만 latency 비용은 FLOPs만으로 설명되지 않는다. 이번 표의 pre-gate eval·host read·고정열 시간을 함께 보며, pre-gate 대기 전체를 추가 overhead로 오인하지 않는다.','',
    '2. **동기화 제거가 도움 됐는가:** Device의 gate item()/eval는 실제로0이며 요청 종료 counter1 batch와 greedy token sync는 남는다. 기존 DLR 대비 시간비는고정열 표에 있다. 새 GEMV kernel/수치 차이가 섞여 순수 sync 인과효과만은 식별하지 못한다.','',
    '3. **경량화하면 얼마나 빨라지는가:** Lightweight의 기존 DLR 대비 감소율과 Baseline 대비 감소율을 모두 위 표에서 구분했다. hidden-state distance 비용은0이나 동일토큰 문맥 변화 위험은 남는다. 재사용0인 요청은 연산 절감의 증거가 아니다.','',
    '4. **큰 모델에 유리할 가능성:** MLP 절감량은d*I, 거리gate는d 또는ID O(1)이므로 산술적 여지는 있다. 그러나 memory bandwidth, cache, prefill 비중, 모델별 hidden-state 변화, custom kernel 효율이 달라져 속도/품질 우위를 외삽하지 않는다. 다른 모델은 이번에 미실험이다.','',
    f'5. **정확도와 시간을 함께 만족한 조건:** 위 최소 관측 통과 {len(qualified)}개를 명시했다. 작은 표본과 반복 상관이 남아 모집단 정확도 유지가 입증된 것은 아니다.','',
    '6. **기존 activation reuse와 차이:** 이전 MLP residual 재사용 자체는 기존 계열이다. 이번 차이는 이 로컬 Qwen 구현에서 GPU 조건부 projection과 이미 관측된 token ID proxy를 비교한 점이다. 독립적 새로운 알고리즘/발명/일반적인 SOTA 근거는 없다.','',
    '## 10. 재현·보존','',f'원래{audit["original_files_unchanged"]}개 파일의 해시 보존 검사 통과. 모든 raw request는 append-only checkpoints, 원본 CSV는EXPORT 폴더다. 재시작은 완료 요청을 재생성하지 않고 source hash를 대조한다.','',
    '```sh',f'.alsr-venv/bin/python -B -m src.dlr_opt_experiment {p}',f'.venv/bin/python -B -m src.dlr_opt_analysis {p}', '.venv/bin/python -B -m src.dlr_opt_report <새 ANALYSIS 폴더>', '.venv/bin/python -B -m unittest tests.test_dlr_optimization', '.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json','```','',
    '그래프는 기존 /private/tmp/vvp_plot_deps_20261105의 plotting 전용 의존성을 사용하며 모델 환경에 패키지를 설치하지 않는다. 모델 로컬 경로/revision/패키지/seed는manifest에 있다. 신규 문제는 한번 저장하면 다시 미사용 문제라고 주장하지 않는다.']
    # Korean report: keep labels/analysis in Korean, including technical table captions.
    text='\n'.join(lines)+'\n'
    with (a/'DLR_OPTIMIZATION_REPORT.md').open('x') as f:f.write(text)
    write_json(a/'FINAL_CLASSIFICATION.json',{'classification':grade,'description':conclusion,'qualified_observed_conditions':qualified,'population_accuracy_noninferiority_proven':False,'gpu_power_money_measured':False})
    positions=[]
    for ds in m['datasets']:
        for method in METHODS:
            for pos in range(4):
                rr=[c for c in cases if c['phase']=='natural' and c['dataset']==ds and c['method']==method and c['position']==pos]
                if rr:positions.append(dict(dataset=ds,method=method,position=pos,requests=len(rr),mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in rr)))
    write_union(a/'DLR_OPTIMIZATION_ORDER_EFFECTS.csv',positions);plots(a,results);print('DLR OPT REPORT '+str(a/'DLR_OPTIMIZATION_REPORT.md'),flush=True)

def plots(a,results):
    deps=Path('/private/tmp/vvp_plot_deps_20261105')
    if deps.exists():sys.path.insert(0,str(deps))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/dlr_opt_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(14,14));colors=['#4477aa','#ee6677','#228833','#aa3377']
    for row,ds in enumerate(('replay32','fresh32','long_high','long_low')):
        rr=[next(r for r in results if r['phase']=='natural' and r['dataset']==ds and r['method']==method) for method in METHODS]
        for col,(key,label,scale) in enumerate((('content_accuracy','Task accuracy (%)',100),('wall_seconds','Natural wall seconds / cohort',1),('cpu_seconds','Process CPU seconds / cohort',1))):
            ax=axes[row,col];ax.bar(METHODS,[r[key+'_mean']*scale for r in rr],yerr=[r[key+'_sample_std']*scale for r in rr],capsize=3,color=colors);ax.set_title(ds+' — '+label);ax.tick_params(axis='x',rotation=20)
    fig.suptitle('DLR optimization: real local measurements, mean ± SD over 3 repeats');fig.tight_layout();fig.savefig(a/'DLR_OPTIMIZATION_COMPARISON.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for ax,ds in zip(axes,('replay32','fresh32','long_high','long_low')):
        for method,color in zip(METHODS,colors):
            r=next(r for r in results if r['phase']=='natural' and r['dataset']==ds and r['method']==method);ax.scatter(r['wall_seconds_mean'],r['content_accuracy_mean']*100,color=color);ax.annotate(method,(r['wall_seconds_mean'],r['content_accuracy_mean']*100),xytext=(3,5),textcoords='offset points',fontsize=7)
        ax.set_title(ds);ax.set_xlabel('Wall seconds');ax.set_ylabel('Task accuracy (%)');ax.margins(.3)
    fig.suptitle('Observed points, not population dominance');fig.tight_layout();fig.savefig(a/'DLR_OPTIMIZATION_PARETO.png',dpi=160);plt.close(fig)

if __name__=='__main__':render(sys.argv[1])
