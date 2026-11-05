"""Fresh holdout, frozen RCAV calibration, two installed models, three real repeats."""
import csv,hashlib,json,random,statistics,time,platform,urllib.request
from pathlib import Path
from uuid import uuid4
if __package__:
    from . import rcav_policy as policy
    from .rcav_dataset import build,ROOT,SEED
    from .rcav_runtime import LocalClient,verifier_prompt
    from . import rcav_runtime as runtime
    from .skipwork_runtime import make_profile,batch_plan
    from .skipwork_experiment import score
    from .efficiency_repeat import total,read_csv,frozen_check
    from .solver300_experiment import check_references
    from .verify_routing200 import process_snapshot,resource_delta
    from .routing_runner import check_ollama
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    import rcav_policy as policy
    from rcav_dataset import build,ROOT,SEED
    from rcav_runtime import LocalClient,verifier_prompt
    import rcav_runtime as runtime
    from skipwork_runtime import make_profile,batch_plan
    from skipwork_experiment import score
    from efficiency_repeat import total,read_csv,frozen_check
    from solver300_experiment import check_references
    from verify_routing200 import process_snapshot,resource_delta
    from routing_runner import check_ollama
    from tool_experiment import write_records
    from adaptive_experiment import write_json
LARGE='qwen2.5:1.5b';SMALL='qwen2.5:0.5b';REPEATS=3

def summarize(rows,calls,blocks,model,rep):
    result=[]
    for method in policy.METHODS:
        rr=[r for r in rows if r['method']==method];aa=[a for a in calls if a['method']==method];bb=[b for b in blocks if b['method']==method]
        correct=sum(r['final_correct'] for r in rr);answered=sum(r['answered'] for r in rr);n=len(rr)
        prefill=total(aa,'prefill_seconds');generation=total(aa,'eval_seconds');rss=[b['boundary_max_rss_mib'] for b in bb if b['boundary_max_rss_mib'] is not None]
        result.append({'model':model,'repetition':rep,'method':method,'requests':n,'correct':correct,'accuracy':correct/n,
            'answer_rate':answered/n,'answered_accuracy':correct/answered if answered else None,
            'format_compliance':sum(r['final_format_compliant'] for r in rr)/n,
            'primary_correct':sum(r['primary_correct'] for r in rr),'model_calls':len(aa),
            'verification_calls':sum(a['stage']=='verify' for a in aa),'tool_calls':sum(r['tool_calls'] for r in rr),
            'input_tokens':total(aa,'input_tokens'),'output_tokens':total(aa,'output_tokens'),
            'prefill_seconds':prefill,'generation_seconds':generation,
            'model_processing_seconds':prefill+generation if prefill is not None and generation is not None else None,
            'server_total_seconds':total(aa,'total_model_seconds'),'processing_seconds':total(bb,'processing_seconds'),
            'mean_service_seconds':sum(r['service_seconds'] for r in rr)/n,
            'mean_completion_latency_seconds':sum(r['completion_latency_seconds'] for r in rr)/n,
            'ollama_cpu_seconds':total(bb,'cpu_seconds'),'router_cpu_seconds':total(bb,'router_cpu_seconds'),
            'cpu_recorded_blocks':sum(b['cpu_seconds'] is not None for b in bb),'resource_blocks':len(bb),
            'rss_boundary_max_mib':max(rss) if rss else None,
            'harmful_verifications':sum(r['verification_outcome']=='harmful' for r in rr),
            'beneficial_verifications':sum(r['verification_outcome']=='beneficial' for r in rr),
            'redundant_correct_by_outcome':sum(r['verification_outcome']=='redundant_correct_by_outcome' for r in rr),
            'verification_parse_failures':sum(r['verification_preserved_primary'] for r in rr),
            'api_errors':sum(a['status']=='api_error' for a in aa),'cost':None,'cost_status':'unmeasured'})
    return result

def aggregate(summaries):
    result=[]
    for model in dict.fromkeys(r['model'] for r in summaries):
        for method in policy.METHODS:
            rr=[r for r in summaries if r['model']==model and r['method']==method];row={'model':model,'method':method,'repeats':len(rr)}
            for key in rr[0]:
                if key in ('model','method','repetition','cost_status'):continue
                vs=[r[key] for r in rr];valid=all(isinstance(v,(int,float)) for v in vs)
                row.update({key+'_mean':statistics.mean(vs) if valid else None,key+'_min':min(vs) if valid else None,
                    key+'_max':max(vs) if valid else None,key+'_sample_std':statistics.stdev(vs) if valid and len(vs)>1 else None})
            row['cost_status']='unmeasured';result.append(row)
    return result

def compare(summaries,rows):
    ix={(r['model'],r['repetition'],r['method']):r for r in summaries};ri={(r['model'],r['repetition'],r['method'],r['question']):r for r in rows};out=[]
    for s in summaries:
        if s['method'] not in policy.WEIGHTS:continue
        model,rep=s['model'],s['repetition'];rr=[r for r in rows if r['model']==model and r['repetition']==rep and r['method']==s['method']]
        for control in ('baseline','selective'):
            b=ix[model,rep,control]
            row={'model':model,'repetition':rep,'method':s['method'],'control':control,
                'quality_maintained_observed':s['accuracy']>=b['accuracy'] and s['answer_rate']>=b['answer_rate'],
                'correct_to_wrong':sum(ri[model,rep,control,r['question']]['final_correct'] and not r['final_correct'] for r in rr),
                'wrong_to_correct':sum(not ri[model,rep,control,r['question']]['final_correct'] and r['final_correct'] for r in rr)}
            for key in ('accuracy','answer_rate','model_calls','input_tokens','output_tokens','model_processing_seconds','processing_seconds','ollama_cpu_seconds'):
                row[key+'_delta']=s[key]-b[key] if s[key] is not None and b[key] is not None else None
            out.append(row)
    return out

def pareto(averages):
    out=[]
    for r in averages:
        for cost in ('model_processing_seconds','processing_seconds','ollama_cpu_seconds'):
            candidates=[s for s in averages if s['model']==r['model'] and s[cost+'_mean'] is not None]
            dominators=[s['method'] for s in candidates if s['method']!=r['method'] and s['accuracy_mean']>=r['accuracy_mean'] and s['answer_rate_mean']>=r['answer_rate_mean'] and s[cost+'_mean']<=r[cost+'_mean'] and
                (s['accuracy_mean']>r['accuracy_mean'] or s['answer_rate_mean']>r['answer_rate_mean'] or s[cost+'_mean']<r[cost+'_mean'])] if r[cost+'_mean'] is not None else []
            out.append({'model':r['model'],'method':r['method'],'cost_axis':cost,'accuracy':r['accuracy_mean'],
                'answer_rate':r['answer_rate_mean'],'measured_mean_cost':r[cost+'_mean'],'on_observed_frontier':not dominators if r[cost+'_mean'] is not None else None,
                'dominated_by':dominators,'scope':'Observed means only; not significance-tested or a global Pareto frontier'})
    return out

def report(averages,pairs,frontier,meta):
    def span(r,k,scale=1,n=3):
        values=[r[k+s] for s in ('_mean','_min','_max')]
        return '미측정' if any(v is None for v in values) else f'{values[0]*scale:.{n}f} [{values[1]*scale:.{n}f}, {values[2]*scale:.{n}f}]'
    lines=['# Risk-Cost Adaptive Verification (RCAV)','',f"실행 ID `{meta['run_id']}`. 실제 로컬 모델 실행. 모든 값은 평균 [최소, 최대].",'',
        '## 사전 설계','',
        '새 보정100문제와 새 평가150문제(5유형×30)를 과거 질문 줄기·교환 산술·선택지 재배열 중복과 분리했다. '
        '세 번 같은150문제를 반복했으며 독립450문제로 보지 않는다. 각 모델은 별도로 보정했고 평가 전에 프로필을 고정했다. '
        '온도0,seed42,출력64토큰,동일 해결/검증 프롬프트·내용 파서·산술 도구를 적용했다.',
        '', 'RCAV는 문제를 해결한 뒤 검증 전에 요청 유형·난이도·답변 해석 상태로 오류위험p를 추정한다. '
        '보정의 오답 수정확률r와 정답 악화확률h를 사용해 예상 순오류 감소 Δ=p×r−(1−p)×h를 계산한다. '
        '유형/형식 층화 표본8개 미만은 유형10개 이상/전체 통계로 후퇴하고, 전역Beta(1,1)과 사전강도10의 경험적 수축을 사용한다. '
        '해석 불능은 유효답 실패위험1, 안전 산술 도구 결과는 위험0으로 다룬다. 이러한 수치는 확률 보장/모델 자체 confidence가 아니다.',
        '', '검증 입력 문자수와 보정에서 실제 측정한 prefill+generation초를 비음수 기울기 선형회귀로 연결한다. '
        '예측값은 보정 최소 관측시간으로 하한을 두며 비용 기준Cref는 보정 중앙값이다. Δ>λ×예상시간/Cref일 때만 검증한다. '
        '주 RCAV λ=0.05, 사전 감도분석 λ=0와0.15를 모두 실행·공개했다. 평가로 λ를 고르거나 조정하지 않았다.',
        '', 'Baseline은 전수 검증하는 연구용 대조군, Selective는 기존 최소20보정·쉬운 유형·수정≤악화이면 검증 생략 규칙 그대로다. '
        '같은 새 보정100개로 각 모델의 Selective 통계도 구했으므로 과거 프로필의 수치를 그대로 쓴 것은 아니다. '
        '원본 기존 코드/실험은 변경하지 않았다. 검증 실패 시 원래 답을 유지하며 추가 재시도는 없다.',
        '', '## 전체 측정','',
        '| 모델 | 방식 | 내용 정확도 % | 답변율 % | 호출 수 | 입력/출력 토큰 | 처리시간(초) | 모델 처리시간(초) |',
        '|---|---|---:|---:|---:|---|---|---|']
    for r in averages:lines.append(f"| {r['model']} | {r['method']} | {span(r,'accuracy',100,2)} | {span(r,'answer_rate',100,2)} | {span(r,'model_calls',1,0)} | {span(r,'input_tokens',1,0)}/{span(r,'output_tokens',1,0)} | {span(r,'processing_seconds')} | {span(r,'model_processing_seconds')} |")
    lines+=['','| 모델 | 방식 | 평균 서비스(ms) | 평균 완료 지연(ms) | Ollama CPU(초) | RSS 경계최대(MiB) |','|---|---|---|---|---|---|']
    for r in averages:lines.append(f"| {r['model']} | {r['method']} | {span(r,'mean_service_seconds',1000,2)} | {span(r,'mean_completion_latency_seconds',1000,2)} | {span(r,'ollama_cpu_seconds')} | {span(r,'rss_boundary_max_mib',1,1)} |")
    lines+=['','서비스 시간은 해당 요청의 생성·의사결정·검증을 포함한다. 완료 지연은 블록 시작 동시도착에서 앞 요청 대기를 포함한다. '
        '처리시간은 방식별 블록 벽시계 합계이며 모델 처리시간은 서버 prefill+generation합계다. CPU는 Ollama 프로세스군 실제 카운터 차다. '
        'RSS는 양끝 표본 최대로 피크/전용 메모리가 아니고 다른 상주 모델도 포함할 수 있다.',
        '', '## 기존 Selective 대비 관측 판정','', '| 모델 | RCAV 조건 | 품질 유지 반복 수 | 처리시간 감소 | 모델시간 감소 | CPU 감소 |','|---|---|---:|---:|---:|---:|']
    for model in dict.fromkeys(r['model'] for r in averages):
        for method in policy.WEIGHTS:
            pp=[p for p in pairs if p['model']==model and p['method']==method and p['control']=='selective']
            cnt=lambda key:sum(p[key] is not None and p[key]<0 for p in pp)
            lines.append(f"| {model} | {method} | {sum(p['quality_maintained_observed'] for p in pp)}/{len(pp)} | {cnt('processing_seconds_delta')}/{len(pp)} | {cnt('model_processing_seconds_delta')}/{len(pp)} | {cnt('ollama_cpu_seconds_delta')}/{len(pp)} |")
    lines+=['','품질 유지는 같은 반복의 정확도·답변율 이상이라는 관측 기준이며 통계적 비열등성 보장이 아니다. '
        'paired.csv는 Baseline/Selective 각각에 대한 문항별 손실도 기록한다. λ=0은 비용항이 없는 절감 효과 대조군이므로 '
        '비용항이 없는 변형과 주 RCAV의 행동이 같으면 비용항의 추가 효용을 입증한 것이 아니다.',
        '', '## 관측 Pareto 관계','', '```json',json.dumps(frontier,ensure_ascii=False,indent=2),'```','',
        '정확도 최대·측정시간/CPU 최소·답변율 비감소 기준의 다섯 관측점이다. 표본 오차/타이밍 잡음에 따른 극소 차이로 frontier가 달라질 수 있으며 '
        '전역 최적화나 독립 일반화의 증거가 아니다. 가중치 중 유리한 조건만 선택해 성공이라고 하지 않는다.',
        '', '## 추정과 실측·한계','',
        'decision의 risk_estimate/expected_error_reduction/estimated_verification_model_seconds는 보정 데이터 기반 **추정값**이다. '
        '실제 정확도·토큰·서버시간·클라이언트 벽시계·CPU·RSS는 실행 로그의 **측정값**이다. '
        '호출 감소와 총 연산량 감소를 동일시하지 않는다. **금전비용·GPU FLOPs·전력은 미측정이며 추정 절감률을 만들지 않았다.**',
        '', '보정 비용은 calibration_summary.csv에 분리했으며 초기 비용 상각을 가정하지 않았다. '
        f"평가 전체 루프 {meta['evaluation_seconds']:.3f}초는 계측·사후채점·파일저장도 포함한다. "
        '워밍업은 별도 호출 로그에 저장했다. 모델마다 연속 실행해 서버/캐시/발열/메모리 상태가 다를 수 있다. 모델 간 절대속도 인과 비교에는 한계가 있다.',
        '', '작은 층화 표본·회귀 비용의 캐시 민감도·효과 크기의 점추정·분포 이동 때문에 위험/비용/검증 효과가 틀릴 수 있다. '
        'CPU/GPU별 비용 모델이나 신뢰상한을 학습한 시스템이 아니다. 임계값의 물리적·경제적 최적성을 주장하지 않는다. '
        '영어 객관식, 합성 논리/지문과 코드 출력 예측이며 의미상 템플릿 독립을 보장하지 않는다. '
        '내용 추출은 정답과 무관한 명확한 표기만 해석하며 자유서술의 의미 정답을 판정하지 않는다.',
        '', '## 파일','',*[f'- [{s}]({meta["prefix_name"]}_{s})' for s in ('manifest.json','profiles.json','calibration.csv','calibration_calls.csv','calibration_summary.csv','warmup_calls.csv','evaluation.csv','calls.csv','blocks.csv','repeat_summary.csv','aggregate.csv','paired.csv','pareto.csv','metadata.json')],'']
    return '\n'.join(lines)

def run(data_dir=ROOT/'data',repeats=REPEATS):
    if repeats<2:raise ValueError('At least two repetitions')
    data_dir=Path(data_dir);calibration,tasks=build(data_dir)
    references=check_references(calibration)+check_references(tasks)
    check_ollama([LARGE])
    with urllib.request.urlopen('http://localhost:11434/api/tags',timeout=10) as f:tags=json.load(f)
    names={m.get('name',m.get('model')) for m in tags['models']};models=[LARGE]+([SMALL] if SMALL in names else [])
    preserved={}
    for folder in (ROOT/'src',ROOT/'tests',data_dir):
        for path in folder.iterdir():
            if path.is_file():preserved[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in ROOT.iterdir():
        if path.is_file():preserved[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    questions=[t['question'] for t in tasks];batches=batch_plan(questions);orders=[]
    for rep in range(repeats):
        plan=[b[:] for b in batches];random.Random(SEED+rep).shuffle(plan);orders.append(plan)
    run_id=uuid4().hex;prefix=data_dir/f'rcav_{run_id}'
    manifest={'run_id':run_id,'version':policy.VERSION,'seed':SEED,'temperature':0,'model_seed':42,'output_tokens':64,
        'models':models,'optional_small_unavailable':SMALL not in names,'tags':tags,'calibration':calibration,'evaluation':tasks,
        'repeats':repeats,'orders':orders,'weights':policy.WEIGHTS,'main_method':'rcav','prior_strength':policy.PRIOR,
        'preserved_sha256':preserved,'platform':platform.platform(),'python':platform.python_version()}
    write_json(Path(f'{prefix}_manifest.json'),manifest);frozen_manifest=hashlib.sha256(Path(f'{prefix}_manifest.json').read_bytes()).hexdigest()
    profiles={};calrows=[];calcalls=[];calsummaries=[];warmups=[]
    for model in models:
        client=LocalClient(model);warmups.append(client.send('Return only A.','warmup'))
        before=process_snapshot();start=time.perf_counter();rr=[];aa=[]
        for i,t in enumerate(calibration):
            p=client.primary(t['question']);a,v=client.verify(t['question'],p['answer']);aa.extend(p['calls']);aa.append(a)
            final=v['prediction'] if v['prediction'] is not None else p['prediction']
            seconds=a['prefill_seconds']+a['eval_seconds'] if a['prefill_seconds'] is not None and a['eval_seconds'] is not None else None
            rr.append({**t,'model':model,'inferred_category':runtime.features(t['question'])['category'],
                'observed':policy.observable(t['question'],p),'primary_answer':p['answer'],'primary_prediction':p['prediction'],
                'verification_answer':a['answer'],'final_prediction':final,'primary_correct':p['prediction']==t['expected'],
                'final_correct':final==t['expected'],'verification_prompt_chars':len(verifier_prompt(t['question'],p['answer'])),
                'verification_model_seconds':seconds})
            if (i+1)%20==0:print(f'RCAV calibration {model} {i+1}/{len(calibration)}',flush=True)
        wall=time.perf_counter()-start;after=process_snapshot()
        profiles[model]={'rcav':policy.fit(rr),'selective':make_profile(rr)}
        calsummaries.append({'model':model,'requests':len(rr),'model_calls':len(aa),'processing_seconds':wall,
            'input_tokens':total(aa,'input_tokens'),'output_tokens':total(aa,'output_tokens'),
            'prefill_seconds':total(aa,'prefill_seconds'),'generation_seconds':total(aa,'eval_seconds'),
            'before':before,'after':after,**resource_delta(before,after)})
        calrows.extend(rr);calcalls.extend(aa)
    write_json(Path(f'{prefix}_profiles.json'),profiles);frozen_profiles=hashlib.sha256(Path(f'{prefix}_profiles.json').read_bytes()).hexdigest()
    for suffix,records in [('calibration',calrows),('calibration_calls',calcalls),('calibration_summary',calsummaries)]:write_records(Path(f'{prefix}_{suffix}.csv'),records)
    print('All profiles frozen before any evaluation',flush=True)
    allrows=[];allcalls=[];allblocks=[];summaries=[];start=time.perf_counter()
    for mi,model in enumerate(models):
        client=LocalClient(model);warmups.append(client.send('Return only A.','warmup'))
        for rep,order in enumerate(orders):
            rows=[];calls=[];blocks=[]
            for block_index,indices in enumerate(order):
                offset=(block_index+rep)%len(policy.METHODS);qq=[questions[i] for i in indices]
                for method in policy.METHODS[offset:]+policy.METHODS[:offset]:
                    before=process_snapshot();block=runtime.execute_block(qq,method,client,profiles[model]['rcav'],profiles[model]['selective']);after=process_snapshot()
                    block_id=f'M{mi}:R{rep+1}:B{block_index:03d}:{method}'
                    for row,index in zip(block['rows'],indices):rows.append({**score(row,tasks[index]),'repetition':rep+1,'block_id':block_id})
                    calls.extend({**a,'repetition':rep+1,'method':method,'block_id':block_id} for a in block['calls'])
                    blocks.append({'model':model,'repetition':rep+1,'method':method,'block_id':block_id,'question_indices':indices,
                        'processing_seconds':block['processing_seconds'],'router_cpu_seconds':block['router_cpu_seconds'],
                        'before':before,'after':after,**resource_delta(before,after)})
                if (block_index+1)%10==0:print(f'RCAV actual {model} repeat{rep+1}/{repeats}: {block_index+1}/{len(order)} blocks',flush=True)
            ss=summarize(rows,calls,blocks,model,rep+1);summaries.extend(ss)
            for suffix,records in [('evaluation',rows),('calls',calls),('blocks',blocks),('summary',ss)]:write_records(Path(f'{prefix}_m{mi}_r{rep+1}_{suffix}.csv'),records)
            allrows.extend(rows);allcalls.extend(calls);allblocks.extend(blocks)
            print('REPEAT '+json.dumps(ss),flush=True)
    evaluation_seconds=time.perf_counter()-start;averages=aggregate(summaries);pairs=compare(summaries,allrows);frontier=pareto(averages)
    frozen_check(manifest)
    if hashlib.sha256(Path(f'{prefix}_manifest.json').read_bytes()).hexdigest()!=frozen_manifest or hashlib.sha256(Path(f'{prefix}_profiles.json').read_bytes()).hexdigest()!=frozen_profiles:raise RuntimeError('Frozen configuration changed')
    if len({a['call_id'] for a in allcalls})!=len(allcalls):raise RuntimeError('Duplicate real calls')
    meta={'run_id':run_id,'prefix_name':prefix.name,'real_inference':True,'evaluation_seconds':evaluation_seconds,
        'evaluated_rows':len(allrows),'unique_evaluation_calls':len(allcalls),'models':models,'repeats':repeats,
        'preserved_files_checked':len(preserved),'preserved_unchanged':True,'manifest_sha256':frozen_manifest,
        'profiles_sha256':frozen_profiles,'cost_status':'unmeasured'}
    for suffix,records in [('evaluation',allrows),('calls',allcalls),('blocks',allblocks),('repeat_summary',summaries),
        ('aggregate',averages),('paired',pairs),('pareto',frontier),('warmup_calls',warmups),('reference_checks',references)]:write_records(Path(f'{prefix}_{suffix}.csv'),records)
    write_json(Path(f'{prefix}_metadata.json'),meta)
    with Path(f'{prefix}_report.md').open('x',encoding='utf-8') as f:f.write(report(averages,pairs,frontier,meta))
    print(f'REPORT: {prefix}_report.md',flush=True)
    return prefix

if __name__=='__main__':
    try:run()
    except (OSError,ValueError,RuntimeError) as error:raise SystemExit(f'RCAV 연구 중단: {error}')
