"""Frozen original RCAV vs additive guard on development replay and fresh150."""
import hashlib,json,random,statistics,time,urllib.request,platform
from pathlib import Path
from uuid import uuid4
if __package__:
    from .rcav_guard_dataset import build,ROOT,SEED
    from . import rcav_guard_policy as guard
    from . import rcav_guard_runtime as runtime
    from .rcav_runtime import LocalClient
    from .efficiency_protocol import extract
    from .efficiency_repeat import total,read_csv,frozen_check
    from .efficiency_audit import decode_row
    from .rcav_experiment import LARGE
    from .skipwork_experiment import score
    from .skipwork_runtime import batch_plan
    from .solver300_experiment import check_references
    from .verify_routing200 import process_snapshot,resource_delta
    from .routing_runner import check_ollama
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    from rcav_guard_dataset import build,ROOT,SEED
    import rcav_guard_policy as guard
    import rcav_guard_runtime as runtime
    from rcav_runtime import LocalClient
    from efficiency_protocol import extract
    from efficiency_repeat import total,read_csv,frozen_check
    from efficiency_audit import decode_row
    from rcav_experiment import LARGE
    from skipwork_experiment import score
    from skipwork_runtime import batch_plan
    from solver300_experiment import check_references
    from verify_routing200 import process_snapshot,resource_delta
    from routing_runner import check_ollama
    from tool_experiment import write_records
    from adaptive_experiment import write_json
SOURCE=ROOT/'data/rcav_c038afc6c5df422aa9e58080a4fa5847'

def historical_loss(source=SOURCE):
    rows=[r for r in read_csv(f'{source}_evaluation.csv') if r['model']==LARGE]
    ix={(r['repetition'],r['method'],r['question']):r for r in rows};result=[]
    lost=[r for r in rows if r['repetition']=='1' and r['method']=='rcav' and r['final_correct']=='False' and ix['1','selective',r['question']]['final_correct']=='True']
    for r in lost:
        matches=[(ix[str(rep),'rcav',r['question']],ix[str(rep),'selective',r['question']]) for rep in range(1,4)]
        result.append({'question':r['question'],'category':r['category'],'family':guard.family(r['question']),
            'expected':r['expected'],'frozen_primary_candidate':r['primary_answer'],
            'same_primary_all3':all(a['primary_answer']==b['primary_answer'] for a,b in matches),
            'rcav_skipped_all3':all(a['verification_executed']=='False' for a,b in matches),
            'selective_repaired_all3':all(b['verification_executed']=='True' and b['verification_prediction']==r['expected'] for a,b in matches),
            'rcav_call_ids_by_repeat':[json.loads(a['call_ids']) for a,b in matches],
            'selective_call_ids_by_repeat':[json.loads(b['call_ids']) for a,b in matches],
            'interpretation':'Same initial error; skipping removes a measured repair opportunity, not the origin of the primary reasoning error.'})
    if len(result)!=11:raise ValueError('Historical eleven changed')
    return result

def pool_record(question,expected,category,primary_answer,verification_answer,seconds,source):
    kind='integer' if category=='arithmetic' else 'choice'
    p=extract(question,primary_answer,kind);v=extract(question,verification_answer,kind)
    final=v['prediction'] if v['prediction'] is not None else p['prediction']
    return {'question':question,'expected':expected,'category':category,'primary_answer':primary_answer,
        'primary_prediction':p['prediction'],'verification_answer':verification_answer,'final_prediction':final,
        'primary_correct':p['prediction'] is not None and p['prediction']==expected,
        'final_correct':final is not None and final==expected,
        'response_status':'unresolved' if p['prediction'] is None else 'bare' if p['format_compliant'] else 'decorated',
        'verification_model_seconds':seconds,'calibration_source':source}

def summarize(rows,calls,blocks,dataset,rep):
    out=[]
    for method in runtime.METHODS:
        rr=[r for r in rows if r['method']==method];aa=[a for a in calls if a['method']==method];bb=[b for b in blocks if b['method']==method]
        n=len(rr);correct=sum(r['final_correct'] for r in rr);answered=sum(r['answered'] for r in rr)
        prefill=total(aa,'prefill_seconds');gen=total(aa,'eval_seconds');rss=[b['boundary_max_rss_mib'] for b in bb if b['boundary_max_rss_mib'] is not None]
        out.append({'dataset':dataset,'repetition':rep,'method':method,'requests':n,'correct':correct,'accuracy':correct/n,
            'answer_rate':answered/n,'answered_accuracy':correct/answered if answered else None,
            'format_compliance':sum(r['final_format_compliant'] for r in rr)/n,'primary_correct':sum(r['primary_correct'] for r in rr),
            'model_calls':len(aa),'verification_calls':sum(a['stage']=='verify' for a in aa),'tool_calls':sum(r['tool_calls'] for r in rr),
            'input_tokens':total(aa,'input_tokens'),'output_tokens':total(aa,'output_tokens'),'prefill_seconds':prefill,'generation_seconds':gen,
            'model_processing_seconds':prefill+gen if prefill is not None and gen is not None else None,
            'server_total_seconds':total(aa,'total_model_seconds'),'processing_seconds':total(bb,'processing_seconds'),
            'mean_service_seconds':sum(r['service_seconds'] for r in rr)/n,
            'mean_completion_latency_seconds':sum(r['completion_latency_seconds'] for r in rr)/n,
            'ollama_cpu_seconds':total(bb,'cpu_seconds'),'router_cpu_seconds':total(bb,'router_cpu_seconds'),
            'rss_boundary_max_mib':max(rss) if rss else None,'cpu_recorded_blocks':sum(b['cpu_seconds'] is not None for b in bb),
            'resource_blocks':len(bb),'harmful_verifications':sum(r['verification_outcome']=='harmful' for r in rr),
            'beneficial_verifications':sum(r['verification_outcome']=='beneficial' for r in rr),
            'historical11_correct':sum(r['historical_loss_item'] and r['final_correct'] for r in rr) if dataset=='replay150' else None,
            'api_errors':sum(a['status']=='api_error' for a in aa),'cost':None,'cost_status':'unmeasured'})
    return out

def aggregate(summaries):
    out=[]
    for dataset in dict.fromkeys(r['dataset'] for r in summaries):
        for method in runtime.METHODS:
            rr=[r for r in summaries if r['dataset']==dataset and r['method']==method];row={'dataset':dataset,'method':method,'repeats':len(rr)}
            for key in rr[0]:
                if key in ('dataset','method','repetition','cost_status'):continue
                vs=[r[key] for r in rr];numeric=all(isinstance(v,(int,float)) for v in vs)
                row.update({key+'_mean':statistics.mean(vs) if numeric else None,key+'_min':min(vs) if numeric else None,
                    key+'_max':max(vs) if numeric else None,key+'_sample_std':statistics.stdev(vs) if numeric and len(vs)>1 else None})
            row['cost_status']='unmeasured';out.append(row)
    return out

def report(averages,metadata):
    def span(r,k):return f"{r[k+'_mean']:.3f} [{r[k+'_min']:.3f}, {r[k+'_max']:.3f}]" if r[k+'_mean'] is not None else '미측정'
    lines=['# RCAV 정답 손실 구제 실험','',f"실행 ID `{metadata['run_id']}`. 수치는 실제 로컬1.5B, 평균 [최소, 최대].",'',
        '기존150문제는 개발 진단 재생, 새150문제는 사용하지 않은 확인 평가로 분리한다. 각3회·온도0·seed42·64토큰·동일 프롬프트/파서/도구다. '
        '기존 RCAV와 Selective는 원래 보정 프로필을 그대로 사용한다. guarded는 기존 RCAV가 생략한 경우에만 추가 검증한다.',
        '', '추가 게이트는 과거 별도 보정100과 새 별도 보정100을 합친200개로만 학습한다. 코드AST/논리 문법의 구조 가족·답변 형식에 대해 '
        '표본≥8, 오답 수정률95% Wilson하한>0.05, 수정건수>악화건수, 평균 검증시간≤기존 보정 중앙값×1.5이면 작동한다. '
        '형식별표본이 부족하면 같은구조 전체로만 후퇴하고, 구조 표본도 부족하면추가하지 않는다. 평가 정답/검증 결과는 판단에 쓰지 않는다. '
        'Wilson하한은 수정빈도에 대한 탐색적 기준이며 순정확도 보장이나 다중검정 보정이 아니다.',
        '', '| 데이터 | 방식 | 정확도 | 답변율 | 검증 호출 | 전체 호출 | 입력/출력 토큰 | 처리시간(초) | 모델시간(초) | CPU(초) |',
        '|---|---|---:|---:|---:|---:|---|---|---|---|']
    for r in averages:lines.append(f"| {r['dataset']} | {r['method']} | {r['accuracy_mean']:.2%} | {r['answer_rate_mean']:.2%} | {r['verification_calls_mean']:.2f} | {r['model_calls_mean']:.2f} | {r['input_tokens_mean']:.2f}/{r['output_tokens_mean']:.2f} | {span(r,'processing_seconds')} | {span(r,'model_processing_seconds')} | {span(r,'ollama_cpu_seconds')} |")
    lines+=['','새150개는 각유형30개, 기본형태100·구조변형40·별도지식10으로 구성했다. 구조변형은 역방향달력/역순관계/3집합/비트추가/3-hop, 복합코드와 지문 예외변경 등을 포함한다. '
        '동일템플릿 계열이 일부공유돼 의미독립을 보장하지 않는다. 기존11문항은 코드6·논리5이고 같은최초답/Selective수정이3회모두확인됐다. '
        '추가로 원래후보를고정한 검증 재실행33회를 평가후 별도 저장했다. 최초오답과 복구기회 생략을구분해야한다.',
        '', '시간은 실제블록벽시계, 모델시간은서버prefill+generation,CPU는Ollama카운터차다. '
        'RSS는공유/상주프로세스경계표본으로피크/전용메모리가아니다. API오류·실패·추출도서비스시간에포함하고ps/채점은제외한다. '
        f"평가전체루프{metadata['evaluation_seconds']:.3f}초는계측/채점/파일저장을포함한다. 보정과재검증진단비용은별도다.",
        '', '금전비용·전력·GPU FLOPs는미측정이다. 정답률차이가있더라도개별문항을잃을수있다. '
        '3회는동일150문제재실행으로450개독립표본이아니며고정seed/기기/캐시/부하한계가있다. '
        '별도분석보고서는질문군집bootstrap으로정확도차이불확실성을확인하고개발재생과미사용평가를구분한다.',
        '',*[f'- [{s}]({metadata["prefix_name"]}_{s})' for s in ('manifest.json','guard_profile.json','historical_losses.csv','calibration_pool.csv','new_calibration_calls.csv','evaluation.csv','calls.csv','blocks.csv','repeat_summary.csv','aggregate.csv','loss_rechecks.csv','loss_recheck_calls.csv','metadata.json')],'']
    return '\n'.join(lines)

def run(data_dir=ROOT/'data',repeats=3):
    if repeats<3:raise ValueError('At least3 repetitions required')
    data_dir=Path(data_dir);newcal,fresh=build(data_dir)
    source=json.loads(Path(f'{SOURCE}_manifest.json').read_text());base_profiles=json.loads(Path(f'{SOURCE}_profiles.json').read_text())[LARGE]
    oldtasks=[{**t,'regime':'development_replay'} for t in source['evaluation']]
    losses=historical_loss();lost_set={r['question'] for r in losses}
    references=check_references(newcal+fresh)+check_references(oldtasks)
    check_ollama([LARGE])
    with urllib.request.urlopen('http://localhost:11434/api/tags',timeout=10) as f:tags=json.load(f)
    preserved={}
    for folder in (ROOT/'src',ROOT/'tests',data_dir):
        for path in folder.iterdir():
            if path.is_file():preserved[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in ROOT.iterdir():
        if path.is_file():preserved[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    datasets={'replay150':oldtasks,'fresh150':fresh};orders={}
    for name,tasks in datasets.items():
        plan=batch_plan([t['question'] for t in tasks]);orders[name]=[]
        for rep in range(repeats):
            order=[b[:] for b in plan];random.Random(SEED+rep).shuffle(order);orders[name].append(order)
    run_id=uuid4().hex;prefix=data_dir/f'rcav_guard_{run_id}'
    manifest={'run_id':run_id,'source':str(SOURCE),'datasets':datasets,'new_calibration':newcal,'repeats':repeats,'orders':orders,
        'model':LARGE,'seed':42,'temperature':0,'num_predict':64,'base_profiles':base_profiles,
        'guard_constraints':{'min_samples':guard.MIN_SAMPLES,'wilson_z':guard.Z,'repair_lower_threshold':guard.REPAIR_LOWER_THRESHOLD,'max_cost_ratio':guard.MAX_COST_RATIO},
        'preserved_sha256':preserved,'tags':tags,'platform':platform.platform()}
    write_json(Path(f'{prefix}_manifest.json'),manifest);mh=hashlib.sha256(Path(f'{prefix}_manifest.json').read_bytes()).hexdigest()
    write_records(Path(f'{prefix}_historical_losses.csv'),losses)
    client=LocalClient(LARGE);warmup=client.send('Return only A.','warmup');write_json(Path(f'{prefix}_warmup.json'),warmup)
    pool=[]
    for r in read_csv(f'{SOURCE}_calibration.csv'):
        if r['model']==LARGE:pool.append(pool_record(r['question'],r['expected'],r['category'],r['primary_answer'],r['verification_answer'],float(r['verification_model_seconds']),'historical_training100'))
    newcalls=[];newrows=[];before=process_snapshot();start=time.perf_counter()
    for i,t in enumerate(newcal):
        p=client.primary(t['question']);a,v=client.verify(t['question'],p['answer']);newcalls.extend(p['calls']);newcalls.append(a)
        seconds=a['prefill_seconds']+a['eval_seconds'] if a['prefill_seconds'] is not None and a['eval_seconds'] is not None else None
        newrows.append(pool_record(t['question'],t['expected'],t['category'],p['answer'],a['answer'],seconds,'new_training100'))
        if (i+1)%20==0:print(f'Guard calibration {i+1}/100',flush=True)
    cal_seconds=time.perf_counter()-start;after=process_snapshot();pool.extend(newrows);profile=guard.fit(pool)
    write_json(Path(f'{prefix}_guard_profile.json'),profile);gh=hashlib.sha256(Path(f'{prefix}_guard_profile.json').read_bytes()).hexdigest()
    write_records(Path(f'{prefix}_calibration_pool.csv'),pool);write_records(Path(f'{prefix}_new_calibration_calls.csv'),newcalls)
    write_json(Path(f'{prefix}_calibration_resource.json'),{'processing_seconds':cal_seconds,'before':before,'after':after,**resource_delta(before,after)})
    print('Guard frozen '+json.dumps(profile),flush=True)
    allrows=[];allcalls=[];allblocks=[];summaries=[];start=time.perf_counter()
    for name,tasks in datasets.items():
        for rep,order in enumerate(orders[name]):
            rows=[];calls=[];blocks=[]
            for bi,indices in enumerate(order):
                qs=[tasks[i]['question'] for i in indices];offset=(bi+rep)%len(runtime.METHODS)
                for method in runtime.METHODS[offset:]+runtime.METHODS[:offset]:
                    before=process_snapshot();block=runtime.execute_block(qs,method,client,base_profiles['rcav'],profile,base_profiles['selective']);after=process_snapshot()
                    bid=f'{name}:R{rep+1}:B{bi:03d}:{method}'
                    for r,i in zip(block['rows'],indices):rows.append({**score(r,tasks[i]),'dataset':name,'repetition':rep+1,'block_id':bid,
                        'regime':tasks[i]['regime'],'historical_loss_item':r['question'] in lost_set})
                    calls.extend({**a,'dataset':name,'repetition':rep+1,'method':method,'block_id':bid} for a in block['calls'])
                    blocks.append({'dataset':name,'repetition':rep+1,'method':method,'block_id':bid,'question_indices':indices,
                        'processing_seconds':block['processing_seconds'],'router_cpu_seconds':block['router_cpu_seconds'],
                        'before':before,'after':after,**resource_delta(before,after)})
                if (bi+1)%10==0:print(f'Guard actual {name} repeat{rep+1}/{repeats}: {bi+1}/{len(order)} blocks',flush=True)
            ss=summarize(rows,calls,blocks,name,rep+1);summaries.extend(ss)
            for suffix,records in [('evaluation',rows),('calls',calls),('blocks',blocks),('summary',ss)]:write_records(Path(f'{prefix}_{name}_r{rep+1}_{suffix}.csv'),records)
            allrows.extend(rows);allcalls.extend(calls);allblocks.extend(blocks);print('REPEAT '+json.dumps(ss),flush=True)
    evaluation_seconds=time.perf_counter()-start
    rechecks=[];recheck_calls=[];start=time.perf_counter()
    for rep in range(repeats):
        for loss in losses:
            a,v=client.verify(loss['question'],loss['frozen_primary_candidate']);recheck_calls.append(a)
            rechecks.append({'repetition':rep+1,'question':loss['question'],'category':loss['category'],'expected':loss['expected'],
                'fixed_candidate':loss['frozen_primary_candidate'],'verification_answer':a['answer'],'prediction':v['prediction'],
                'correct':v['prediction'] is not None and v['prediction']==loss['expected'],'call_id':a['call_id']})
    recheck_seconds=time.perf_counter()-start;averages=aggregate(summaries);frozen_check(manifest)
    if hashlib.sha256(Path(f'{prefix}_manifest.json').read_bytes()).hexdigest()!=mh or hashlib.sha256(Path(f'{prefix}_guard_profile.json').read_bytes()).hexdigest()!=gh:raise RuntimeError('Frozen policy changed')
    if len({a['call_id'] for a in allcalls})!=len(allcalls):raise RuntimeError('Duplicate call accounting')
    metadata={'run_id':run_id,'prefix_name':prefix.name,'real_inference':True,'manifest_sha256':mh,'guard_profile_sha256':gh,
        'preserved_files_checked':len(preserved),'preserved_unchanged':True,'evaluation_seconds':evaluation_seconds,
        'calibration_seconds':cal_seconds,'new_calibration_calls':len(newcalls),'recheck_seconds':recheck_seconds,
        'diagnostic_recheck_calls':len(recheck_calls),'evaluation_rows':len(allrows),'actual_evaluation_calls':len(allcalls),'cost_status':'unmeasured'}
    for suffix,records in [('evaluation',allrows),('calls',allcalls),('blocks',allblocks),('repeat_summary',summaries),('aggregate',averages),
        ('loss_rechecks',rechecks),('loss_recheck_calls',recheck_calls),('reference_checks',references)]:write_records(Path(f'{prefix}_{suffix}.csv'),records)
    write_json(Path(f'{prefix}_metadata.json'),metadata)
    with Path(f'{prefix}_report.md').open('x',encoding='utf-8') as f:f.write(report(averages,metadata))
    print(f'REPORT: {prefix}_report.md',flush=True)
    return prefix

if __name__=='__main__':
    try:run()
    except (OSError,ValueError,RuntimeError) as exc:raise SystemExit(f'RCAV guard 중단: {exc}')
