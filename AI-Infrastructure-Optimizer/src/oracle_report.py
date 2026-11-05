"""Korean scoped Oracle capability report and measured rate/quality plots."""
import json,os,statistics,sys
from pathlib import Path
from .controlled_analysis import read_decoded
from .alsr_io import write_union
from .adaptive_experiment import write_json
from .controlled_statistics import describe
LABEL={'baseline':'Baseline','ideal':'Ideal Oracle','random':'Random Skip','existing':'Existing DLR','overhead':'Oracle+Actual Overhead'}
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|']+['| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in row)+' |' for row in rows])
def fmt(x,n=3):return '미측정' if x is None else f'{x:.{n}f}'
def overall_results(a):
    rr=read_decoded(a/'ORACLE_REPEATS.csv');allreps=[];summaries=[];conditions=list(dict.fromkeys((r['phase'],r['method'],r['rate']) for r in rr))
    sums=('wall_seconds','cpu_seconds','model_wall_seconds','input_tokens','output_tokens','reused_mlp_calls','gate_queries','gate_wall_seconds','gate_arithmetic_estimate','analytical_matmul_flops')
    for phase,method,rate in conditions:
        values=[]
        for rep in range(1,6):
            rows=[r for r in rr if (r['phase'],r['method'],r['rate'],r['repetition'])==(phase,method,rate,rep)];n=sum(r['requests'] for r in rows);assert n==20
            row=dict(phase=phase,method=method,rate=rate,repetition=rep,requests=n)
            for k in ('content_accuracy','format_compliance','output_agreement'):row[k]=sum(r[k]*r['requests'] for r in rows)/n
            for k in sums:row[k]=sum(r[k] for r in rows)
            row['mlx_peak_bytes']=max(r['mlx_peak_bytes'] for r in rows);values.append(row);allreps.append(row)
        summary=dict(phase=phase,method=method,rate=rate,unique_questions=20)
        for k in sums+('content_accuracy','format_compliance','output_agreement','mlx_peak_bytes'):
            for key,value in describe([r[k] for r in values],draws=1000).items():summary[k+'_'+key]=value
        summaries.append(summary)
    write_union(a/'ORACLE_OVERALL_REPEATS.csv',allreps);write_union(a/'ORACLE_OVERALL_RESULTS.csv',summaries);return summaries

def render(folder):
    a=Path(folder);p=Path(json.loads((a/'SOURCE.json').read_text())['experiment']);m=json.loads((p/'MANIFEST.json').read_text());audit=json.loads((a/'AUDIT.json').read_text());cls=json.loads((a/'CLASSIFICATION.json').read_text())
    results=read_decoded(a/'ORACLE_RESULTS.csv');comp=read_decoded(a/'ORACLE_COMPARISONS.csv');cases=read_decoded(a/'ORACLE_CASES.csv');verdicts=read_decoded(a/'ORACLE_VERDICTS.csv');classification=cls['classification'];meaning={'A':'가능성 확인','B':'조건부 가능','C':'실패'}[classification]
    equiv=json.loads((a/'OVERHEAD_EQUIVALENCE.json').read_text())
    overall=overall_results(a)
    lines=['# ORACLE_EXPERIMENT_REPORT','',f'**최종 판정: {classification} — {meaning}.** 실제 무료 로컬 Qwen2.5-1.5B-Instruct BF16 실험 `{p.name}`. 기존 실험·코드·데이터 보존, 유료 API·GitHub 업로드 없음.','',
    '판정의 의미는 사전에 정한 **정확도/형식 비감소·개별 정답 손실 없음·동일열 평균5% 이상·CI/반복 방향**의 동시 충족 여부다. C를 단순히 물리적인 속도 이득이 전혀 없거나 모든 DLR이 불가능하다는 뜻으로 사용하지 않는다. 5%는 이번 실험의 사전 실용 기준이며 그보다 작은 관측 이득도 아래에 숨기지 않고 보고한다.','',
    '**중요한 범위:** 이 Oracle은 평가 정답과 미래 Baseline 활성값을 사용해 후보3개 중 최선의 결정표를 고른다. 실제 서비스 알고리즘이 아니다. 수학적으로 최적인 Oracle도 아니므로, 부정적 결과가 모든 DLR의 불가능성을 증명하지 않는다. 긍정적 결과도 배포 가능한 판단 기준의 발견이나 모집단 정확도 보존 증거가 아니다.','',
    '## 1. 실험의 질문과 기존 구현','',
    '기존 DLR은8/12/16/24번 MLP에서 최신 fresh 입력 대비 상대 L2가0.5이하면 이전 MLP residual을 재사용한다. Attention과 KV는 매 토큰 갱신한다. Device-Resident는 GPU 조건 분기로 projection MAC 루프를 실제 우회하지만 fresh GEMV 수치 순서가 native와 달랐다. Lightweight는 이미 읽은 token ID의 반복을 신호로 썼으며 문맥 변화 및 EOS 손상이 발생했다.','',
    '이번에는 모든 비교의 실제 MLP를 기존 native MLX 연산으로 실행해 custom GEMV 수치 오차 요인을 제거했다. 생략 시 gate/up/down projection과 activation을 호출하지 않고 최신 fresh MLP 출력을 residual로 사용한다. zero residual Layer Skipping이 아니며 KV caching과도 다르다.','',
    '## 2. Oracle과 기준선 정의','',
    '- Baseline: 기존 정상 native 추론.\n- Ideal Oracle: 미리 고정한 표 조회만 수행, runtime 거리 계산/판단 없음.\n- Random Skip: 같은 토큰별 quota/cooldown에서 무작위 MLP 재사용.\n- Existing DLR: 기존4후보·0.5기준 그대로. 목표 생략률별 재튜닝하지 않았다.\n- Oracle+Actual Overhead: Ideal과 같은 표를 적용하면서 기존4후보의 실제 거리/eval/item 계산과 입력 anchor 관리를 추가하고, 그 판단은 버린다.','',
    'Ideal/Random/Overhead는 최대28개 MLP를 대상으로 하지만 Existing DLR은4개 후보 제한을 유지한다. 따라서 Existing을50% matched-rate 기법이라고 비교하지 않는다. 한 번 재사용 뒤 fresh 계산을 요구하므로 기존4후보의 긴 decode 이론상 평균 상한도 약2/28=7.14%다(유한 짧은 출력에서는 달라짐). 이는 구조적 호출 상한이며 실측 FLOPs/시간 감소율이 아니다. Baseline/Existing은 조건당 한 번 측정했고 그래프의 수평선은 동일 측정값이다. 여섯 생략률에서 독립 측정한 것으로 과장하지 않았다.','',
    'Random은 질문/rate별 독립 seed의 표 한 개를 고정해5회 반복했다. 따라서 반복 CI는 서로 다른 random plan 다섯 개의 불확실성을 포함하지 않는다. Oracle은 정답으로3후보를 선택하고 Random은 한 표만 쓰므로, 둘의 품질 차이를 실용적인 예측기의 우수성으로 해석하지 않는다.','',
    'Zero-overhead는 **판단 계산을 뺀 비교군**이라는 뜻이다. 표 생성·로드·경로 준비 비용은 따로 있고, runtime 표 인덱싱/분기와 output cache 관리는 남는다. Ideal/Random/Overhead는0%에서도28개 output anchor를 관리한다. 0%와 Baseline 차이에는 cache/lookup뿐 아니라 새 loop·계수·timer dispatch 차이도 있어 순수 memory-copy 비용으로 해석하지 않는다. Overhead는4개 입력 anchor를 더 보관한다. 이를 완전한 모든 overhead0으로 부르지 않는다.','',
    '생략률0/10/20/30/40/50%는 decode의28개 MLP 호출 기준이다. Prefill은 항상 전체 계산하고 한 층을 연속2번 재사용하지 않는다. 정수 quota는 누적 floor로 맞췄다. 실제 decode 비율, 전체 layer-token 비율, 전체 모델 FLOPs 추정 감소율을 각각 구분한다. 특히50% target은 전체 추론 연산량50% 감소가 아니다.','',
    '## 3. 사전 결정표 생성과 특권 정보','',
    f'정상 모델을 공통 최대 길이까지 실행해 각 토큰·계층의 실제 MLP 출력을 수집했다. EOS 이후에도 미래 활성값 확보를 위해 계속 생성한 부분이 있다. 그 출력을 최근 fresh anchor와 비교한 최소 drift 표1개와 random 표2개를 만들고 실제 자연생성으로 평가했다. 선택 기준은 내용 정답, 형식, Baseline token열 일치, 길이 차이, 후보 index 순서로 사전에 고정했다. 평가 정답을 사용한 의도적 특권 선택이며 정보 누출 없는 실용 모델로 표현하지 않는다. Random 비교군은 별도 독립 seed다.','',
    f'오프라인 trace 모델 호출 {audit["offline_trace_model_calls"]}회, 후보 생성 호출 {audit["offline_candidate_model_calls"]}회. 표 생성 wall 합계 {audit["offline_table_generation_seconds"]:.3f}초, CPU 합계 {audit["offline_table_generation_cpu_seconds"]:.3f}초. 여기에는 활성값 수집·후보 생성·채점·파일 저장이 포함된다. 모든 후보와 실패는 tables 하위 TRACE/CANDIDATE 파일에 보존했다. 원시 activation 전체를 장기 보관하지 않고 shape/hash와 표를 저장했다.','',
    '이 오프라인 비용을 공짜로 보아 전체 서비스 비용이 감소했다고 주장하지 않는다. 성공 판정은 이미 표가 주어진 온라인 추론 단계의 가능성에 한정한다. 새 질문마다 표를 새로 만들면 이 비용이 다시 발생하며, 동일 질문의 표를 반복 사용하는 상황은 최종 답변 캐싱과도 구분해야 한다.','',
    '각 표의 SHA256를 반복 실행 전 기록하고 분석 시 검증했다. runtime은 질문·표·토큰예산을 받으며 정답/미래 활성값은 직접 받지 않지만, 표에 이미 그 정보가 반영되어 있다. 신규 문제의 정답도 Oracle 표 선택에 썼으므로 Oracle 정확도는 독립 holdout 일반화 측정이 아니다.','',
    '## 4. 공정 조건과 측정','',
    f'기존8문제(객관식6/JSON2), 신규8문제(객관식4/JSON2/color2), 신규 긴 복사4문제(반복2/다양단어2), 총20문제다. 각 조건5회, 자연생성 및 동일 토큰열 각각2000건, 주 비교4000건을 실제 실행했다. seed{m["seed"]}. 동일 prompt/chat template/greedy/EOS, short32/long64토큰 cap. 쉬운/어려운 실패를 제외하지 않았다. 합성 exact stem 중복만 검사했으며 개념적 유사성은 남는다.','',
    f'가중치 revision `{m["model"]["revision"]}`, 패키지 `{json.dumps(m["packages"],sort_keys=True)}`. 기존 로컬 파일 해시 확인. M4 16GiB, 권장 working-set80%/MLX allocator cache128MiB 제한을 유지했다.','',
    '워밍업 후 질문별 조건 순서를 섞고5회 반복했다. 자연생성이 모두 끝난 뒤 Baseline1회차의 실제 토큰열(EOS 포함)을 모든 방식에 동일하게 입력하는 fixed-stream을 실행했다. 후자는 실제 forward 비용 측정이며 강제 출력의 정답률을 독립적인 생성 품질로 해석하지 않는다. 실제 예측 token도 가능한 경로에서 별도 보존한다.','',
    '요청 wall은 기존 DLR과 같은 내부 측정 범위로, tokenization 전부터 decode 및 최종 MLX 동기화까지다. 이후 무결성 검사/일부 metadata 조립/함수 반환·최종 참조 해제/저장은 별도여서 전체 controller 시간과 같지 않다. 루프 안 cache 갱신·반환값 준비에 필요한 GPU 작업·판단 비용은 포함한다. COMPLETE 로그의 controller elapsed는 표 생성/로드/워밍업/저장도 포함하므로 방식별 추론 절감률의 분모로 사용하지 않았다.','',
    '내용과 형식을 기존 채점 함수로 분리했다. JSON은 중복키/type혼동을 거부하고, color는 기존 제한 문법, copy는24단어 전체 순서 일치를 요구했다. 부분 prefix만 맞은 불완전 응답은 실패다. 일반 자연어 의미/환각 전체를 평가하는 지표가 아니다.','',
    '## 5. 자연생성 전체 결과','',
    '### 전체20문제 합계','',table(['방식','목표%','내용%','형식%','Baseline token 일치%','wall 평균±SD(s)','CPU(s)','출력토큰'],[[LABEL[r['method']],r['rate'] if r['rate'] is not None else '고유',f'{r["content_accuracy_mean"]*100:.2f}',f'{r["format_compliance_mean"]*100:.2f}',f'{r["output_agreement_mean"]*100:.2f}',f'{r["wall_seconds_mean"]:.3f} ± {r["wall_seconds_sample_std"]:.3f}',f'{r["cpu_seconds_mean"]:.3f}',f'{r["output_tokens_mean"]:.0f}'] for r in overall if r['phase']=='natural']),'',
    '전체 정확도는 집합 평균을 단순 평균하지 않고 실제 문제 수8/8/4로 가중했다. 전체 표의95%CI는 고정20문제의5반복 합계 bootstrap이고, paired 질문/반복 비교 CI는 아래 별도 표다.','']
    for ds in m['datasets']:
        rr=[r for r in results if r['phase']=='natural' and r['dataset']==ds]
        lines+=['### '+ds,'',table(['방식','목표 decode%','실제 decode%','내용%','형식%','Baseline 일치%','wall 평균±SD(s)','CPU(s)','출력토큰','MLX peak MiB'],[[LABEL[r['method']],r['rate'] if r['rate'] is not None else '고유 설정',f'{r["achieved_decode_skip_fraction_mean"]*100:.2f}',f'{r["content_accuracy_mean"]*100:.2f}',f'{r["format_compliance_mean"]*100:.2f}',f'{r["output_agreement_mean"]*100:.2f}',f'{r["wall_seconds_mean"]:.3f} ± {r["wall_seconds_sample_std"]:.3f}',f'{r["cpu_seconds_mean"]:.3f}',f'{r["output_tokens_mean"]:.0f}',f'{r["mlx_peak_bytes_mean"]/(1<<20):.2f}'] for r in rr]),'']
    lines+=['시간/토큰은 각 집합 전체 문제의 합계 평균이다. 토큰당 지연은prefill 포함 wall/output수이며 decode-only 시간과 다르다. 모든 평균·SD·최소·최대·95%CI는 ORACLE_RESULTS.csv, 반복별 값은 ORACLE_REPEATS.csv에 있다.','',
    '## 6. 동일 토큰 수의 실제 비용 비교','']
    for ds in m['datasets']:
        cc=[r for r in comp if r['phase']=='fixed' and r['dataset']==ds and r['control']=='baseline']
        lines+=['### '+ds,'',table(['방식','목표%','wall 감소% [95%CI]','CPU 감소%','모델 matmul 추정 감소%','5회 wall 비율'],[[LABEL[r['method']],r['rate'] if r['rate'] is not None else '고유',f'{(1-r["wall_seconds_ratio"])*100:.2f} [{(1-r["wall_seconds_ci95_high"])*100:.2f}, {(1-r["wall_seconds_ci95_low"])*100:.2f}]',f'{(1-r["cpu_seconds_ratio"])*100:.2f}',f'{(1-r["analytical_matmul_flops_ratio"])*100:.3f}',', '.join(f'{v:.3f}' for v in r['repeat_wall_ratios'])] for r in cc]),'']
    lines+=['양의 감소율은 절감, 음수는 증가다. CI는 고유 질문 및 반복 축 paired bootstrap이다. 다수 rate/집합 탐색에 대한 동시 신뢰구간 보정은 하지 않았으므로 탐색적 nominal95%CI다. 작은20문제와 deterministic 반복을 독립100개 문제로 과장하지 않는다. 정확도 차이0 또는 bootstrap[0,0]만으로 모집단 비열등성을 입증하지 않는다.','',
    '### 모델·판단·cache·실제 생략 계수','',table(['집합','방식','목표%','모델 step(s)','gate(s)','cache Python(s)','표조회(s)','wall ms/토큰','실제 생략 MLP','전체 layer-token 감소%'],[[r['dataset'],LABEL[r['method']],r['rate'] if r['rate'] is not None else '고유',fmt(r['model_wall_seconds_mean']),fmt(r['gate_wall_seconds_mean']),fmt(r['cache_python_wall_seconds_mean']),fmt(r['route_lookup_wall_seconds_mean']),fmt(r['mean_seconds_per_output_token_mean']*1000),f'{r["reused_mlp_calls_mean"]:.0f}',f'{r["achieved_all_layer_token_skip_fraction_mean"]*100:.3f}'] for r in results if r['phase']=='fixed']),'',
    '## 7. 실제 판단 비용과 cache 비용','',table(['집합','목표%','Overhead/Ideal wall 비율 [95%CI]','추가 CPU 비율'],[[r['dataset'],r['rate'],f'{r["wall_seconds_ratio"]:.3f} [{r["wall_seconds_ci95_low"]:.3f},{r["wall_seconds_ci95_high"]:.3f}]',f'{r["cpu_seconds_ratio"]:.3f}'] for r in comp if r['phase']=='fixed' and r['control']=='ideal']),'',
    'Overhead/Ideal은 실제로 gate 연산을 실행한 비교다. gate_wall은 입력 준비 후 거리 계산·GPU eval·item 회수를 포함하고, pre_gate_sync는 앞선 모델 연산 대기를 포함한다. 후자를 전부 추가 overhead로 해석하거나 두 값을 전체 wall에 다시 더하지 않는다. cache_python_wall/route_lookup은 Python·MLX graph dispatch 시간이지 GPU memcpy 시간이 아니다.','',
    '추가 gate는 기존 거리/eval/item 계산과 입력 anchor 관리를 실제 수행하지만, event dict 로깅·age bookkeeping까지 원래 구현과 byte 단위로 같지는 않다. 따라서 E−Ideal을 원래 DLR의 모든 Python/로깅 비용의 정확한 인과 분해로 표현하지 않는다.','',
    f'Overhead/Ideal의 자연생성600쌍에서 token열 차이 {equiv["natural_token_mismatches"]}개, 동일열 실제 예측600쌍에서 차이 {equiv["fixed_prediction_mismatches"]}개를 기록했다. 강제된 최종 문자열 일치만으로 연산 동일성을 주장하지 않았다.','',
    'GPU activation을 host로 내보내는 경로는 주 추론에 없다. output anchor는 logical84KiB, 추가4 input anchor는12KiB다. slice/contiguous가 backing storage를 유지하거나 필요 시 복사할 수 있어 실제 메모리 이동량은 별도 미측정이다. MLX peak는 실제 allocator 관측이지만 장치 전체 memory가 아니고 RSS는 프로세스 lifetime high-water라 방식별 독립 peak가 아니다.','',
    '모델 matmul FLOPs는 MAC=2의 분석 추정이고 실제 hardware FLOPs가 아니다. 실제 native MLP 호출 우회 수는 별도로 검증했다. gate 산술약5*d+4를 별도 저장하되 norm/activation/lookup/cache/메모리traffic/실제 instruction 전체를 계산한 것은 아니다. native projection call 수를 GPU kernel launch 수로 바꾸지 않는다. 현재 개발 환경에서 xcrun --find xctrace가 실패해 해당 도구를 사용할 수 없었다. 실제 GPU kernel count/GPU timestamp·전력·금전 비용은 미측정이다.','',
    '## 8. 별도 침습적 병목 프로파일','']
    profiles=[f for f in p.glob('PROFILE_*/ORACLE_INTRUSIVE_PROFILE.csv') if json.loads((f.parent/'SCOPE.json').read_text()).get('warmed_per_condition')]
    if profiles:
        profile=max(profiles,key=lambda f:f.stat().st_mtime_ns);pr=read_decoded(profile);lines += [f'조건별1회 워밍업 후 측정한 보조 프로파일 `{profile.parent.name}`를 사용했다. 첫 프로파일의 cold-start 가능성 자료도 보존했지만 아래 표에는 혼합하지 않았다. 주4000건과 별도 실행이다.','',table(['집합','방식','목표%','동기화 wall(s)','Attention 구간(s)','MLP 구간(s)','cache 동기화 구간(s)'],[[r['dataset'],LABEL[r['method']],r['rate'] if r['rate'] is not None else '고유',fmt(r['wall_seconds']),fmt(r['profile_attention_sync_wall_seconds']),fmt(r['profile_mlp_sync_wall_seconds']),fmt(r['profile_cache_sync_wall_seconds'])] for r in pr]),'']
    else:lines+=['별도 프로파일이 아직 없으면 이 항목은 미측정이다.','']
    lines+=['각 Attention/MLP/cache에 동기화를 추가한 wall 구간이다. 정상 pipeline overlap을 바꾸므로 주 성능값·GPU 실행시간·원인별 정확한 기여율로 취급하지 않는다. Existing wrapper의 Attention은 inner 모듈만, 다른 profile 경로는 input norm/attention/residual add를 포함하므로 그 bucket을 직접 동일 범위로 비교하지 않는다. MLP 구간은 post-norm 입력 준비 뒤 측정한다. 작은 모델의 MLP 절감 여지는 matched-stream 곡선에서 직접 보고, Python/cache/Attention/head의 잔여 비용은 이 프로파일을 보조로 사용한다.','',
    '## 9. 품질 손상과 판정','',table(['집합','목표%','Oracle 관측 기회','정답 손실 문제','동일열 시간 감소%','동일열 CI 상한','5회 모두 빨라짐'],[[r['dataset'],r['rate'],r['conditional_opportunity'],r['individual_loss_questions'],f'{r["matched_wall_saving"]*100:.2f}',f'{r["matched_ci_high"]:.3f}',r['all_five_repeats_faster']] for r in verdicts]),'',
    '기회 판정은 평균5% 이상 matched-stream 감소·CI 상한<1·5회 모두 개선·관측 내용/형식 비감소·개별 Baseline 정답 손실 없음·실제 MLP우회를 동시에 요구한다. Baseline 정답률0인 집합은 품질 성공으로 처리하지 않는다. 이는 소규모 관측 기준이며 일반화나 상용 정책의 정확도 유지 증명이 아니다.','']
    losses=[c for c in cases if c['phase']=='natural' and c['repetition']==1 and c['method']=='ideal' and c['transition']=='Correct->Wrong']
    for c in losses[:12]:lines += [f'### {c["dataset"]}/Oracle{c["rate"]}%/{c["task_id"][:10]}',f'- 질문: {c["question"].replace(chr(10)," / ")}',f'- 기준: {c["reference"]!r}',f'- Baseline: {c["baseline_answer"]!r}',f'- Oracle: {c["answer"]!r}; 출력{c["output_tokens"]}토큰, 완료 사유 {c["completion_reason"]}.','']
    lines+=['모든 응답·오답·개별 전환·순서는 ORACLE_CASES/FAILURES CSV에 보존했다. 위 예시나 가장 유리한 rate만으로 결론을 정하지 않았다.','',
    '## 10. 한계와 다음 연구 권고','',
    f'**최종 {classification}: {meaning}.** 공통 통과 rate는 {cls["common_qualifying_rates"]}, 통과 집합/rate 수는 {len(cls["qualified_conditions"])}다. 판단은 이 native 구현/표/소규모 synthetic 데이터에만 해당한다. throughput만 빨라지고 정확도가 나빠진 rate는 성공이 아니다.','',
    '더 큰 모델에서는 MLP weight traffic 대비 작은 lookup 비용의 비율이 유리해질 가능성이 있으나 다른 모델은 실행하지 않아 미검증이다. 긴 출력은 decode 비중이 커질 수 있지만 재사용 오차 누적과 EOS 오류도 커질 수 있다. 메모리나 전력 절감을 실측하지 않고 주장하지 않는다.','',
    '다음 단계는 이 결과에서 품질을 유지한 표가 존재하는 경우에만 그 선택을 재현하는 저비용 사전 신호를 별도 학습/평가 데이터로 검증하는 것이다. 조건이 없으면 gate 최적화를 확대하기 전에 full-model native kernel/compiled execution과 다른 비용 절감 방향을 우선 검토한다. 이번에 새 서비스 알고리즘이나 독립적 기술을 발견했다고 주장하지 않는다.','',
    '## 11. 재현 명령과 보존','',
    '자동 테스트: 전체 snapshot256개 중252개 통과, NumPy 환경 사유4개 건너뜀(76.046초). 해당4개 수치 검사와 새 Oracle 테스트를 포함한 별도 MLX/NumPy 환경13개는 전부 통과(0.077초). 실제 모델 smoke는0% native 출력 일치·생략 계수·MLP trap 우회·fresh positive control을 확인했다.','',
    '```sh',f'.alsr-venv/bin/python -B -m src.oracle_experiment {p}',f'.venv/bin/python -B -m src.oracle_analysis {p}',f'.alsr-venv/bin/python -B -m src.oracle_profile {p}', '.venv/bin/python -B -m src.oracle_report <새 ANALYSIS 경로>', '.venv/bin/python -B -m unittest tests.test_oracle', '.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json','```','',
    f'기존 {audit["original_files_unchanged"]}개 파일 SHA256를 보존했다. checkpoints는 append-only이며 재시작은 완료 요청을 재생성하지 않는다. 표 생성·모델 로드·워밍업·별도 profile은 주 비교 밖에서 기록한다. 새 데이터는 최초 저장 후 다시 미사용 문제로 주장하지 않는다.','',
    '[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [synchronize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html). 공식 문서는 비동기/동기화 해석의 근거이며 본 실험 성능의 근거는 원본 측정 파일이다.']
    if m.get('replay_from'):lines.insert(4,'이 실행은 기존 문제를 명시적으로 재실행한 replay이며 신규 미사용 평가가 아니다. 원본 cohort 이름을 유지했지만 독립적인 새로운 표본으로 세지 않는다.')
    lines+=['','전체 실제 모델 재실행은 다음 명령으로 새 UUID에 생성한다. 같은 문제를 재사용하므로 신규 평가라고 주장하지 않는다.','```sh',f'.alsr-venv/bin/python -B -m src.oracle_replay {p}/MANIFEST.json','```']
    with (a/'ORACLE_EXPERIMENT_REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    plots(a,results,comp);print('ORACLE REPORT '+str(a/'ORACLE_EXPERIMENT_REPORT.md'),flush=True)

def plots(a,results,comparisons):
    deps=Path('/private/tmp/vvp_plot_deps_20261105')
    if deps.exists():sys.path.insert(0,str(deps))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/oracle_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8));colors={'ideal':'#228833','random':'#ee6677','overhead':'#aa3377'}
    for col,ds in enumerate(('replay8','fresh8','long4')):
        for method in ('ideal','random','overhead'):
            cc=sorted([r for r in comparisons if r['phase']=='fixed' and r['dataset']==ds and r['method']==method and r['control']=='baseline'],key=lambda r:r['rate']);axes[0,col].plot([r['rate'] for r in cc],[r['wall_seconds_ratio'] for r in cc],marker='o',label=LABEL[method],color=colors[method]);axes[0,col].fill_between([r['rate'] for r in cc],[r['wall_seconds_ci95_low'] for r in cc],[r['wall_seconds_ci95_high'] for r in cc],color=colors[method],alpha=.12)
            rr=sorted([r for r in results if r['phase']=='natural' and r['dataset']==ds and r['method']==method],key=lambda r:r['rate']);axes[1,col].plot([r['rate'] for r in rr],[r['content_accuracy_mean']*100 for r in rr],marker='o',label=LABEL[method],color=colors[method])
        old=next(r for r in comparisons if r['phase']=='fixed' and r['dataset']==ds and r['method']=='existing');axes[0,col].axhline(1,color='black',ls='--',label='Baseline');axes[0,col].axhline(old['wall_seconds_ratio'],color='#4477aa',ls=':',label='Existing DLR (one setting)')
        for method,color,style in [('baseline','black','--'),('existing','#4477aa',':')]:
            r=next(r for r in results if r['phase']=='natural' and r['dataset']==ds and r['method']==method);axes[1,col].axhline(r['content_accuracy_mean']*100,color=color,ls=style,label=LABEL[method])
        axes[0,col].set(title=ds,ylabel='Equal-stream wall / Baseline (95% CI)');axes[1,col].set(ylabel='Natural closed-task accuracy (%)',ylim=(-3,103))
        for row in (0,1):axes[row,col].set_xlabel('Decode MLP target skip % (prefill unchanged)');axes[row,col].grid(alpha=.15)
    axes[0,0].legend(fontsize=7);fig.suptitle('Privileged best-of-three Oracle: actual local measurements / 5 repeats\nUses evaluation labels and future Baseline activations; not deployable or globally optimal');fig.tight_layout();fig.savefig(a/'ORACLE_COMPARISON.png',dpi=160);fig.savefig(a/'ORACLE_COMPARISON.svg');plt.close(fig)

if __name__=='__main__':render(sys.argv[1])
