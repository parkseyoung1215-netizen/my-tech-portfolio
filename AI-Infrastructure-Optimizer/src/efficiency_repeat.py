"""Three real repetitions of existing policies, corrected scoring, no new algorithm."""
import csv,hashlib,json,random,statistics,time,platform
from pathlib import Path
from uuid import uuid4
import urllib.request
if __package__:
    from . import efficiency_runtime as runtime
    from .efficiency_protocol import extract
    from .skipwork_experiment import score
    from .solver300_experiment import check_references
    from .solver300_dataset import task_key,ROOT
    from .verify_routing200 import process_snapshot,resource_delta
    from .routing_runner import check_ollama
    from .adaptive_experiment import write_json
    from .tool_experiment import write_records
else:
    import efficiency_runtime as runtime
    from efficiency_protocol import extract
    from skipwork_experiment import score
    from solver300_experiment import check_references
    from solver300_dataset import task_key,ROOT
    from verify_routing200 import process_snapshot,resource_delta
    from routing_runner import check_ollama
    from adaptive_experiment import write_json
    from tool_experiment import write_records

SOURCE=ROOT/'data/skipwork_7a7cc6bdb72e495ab9af664ba5b15db7'
REPEATS=3

def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def total(records,key):
    values=[r[key] for r in records]
    return sum(values) if values and all(v is not None for v in values) else None

def summarize(rows,calls,blocks,repetition):
    result=[]
    for m in runtime.METHODS:
        rr=[r for r in rows if r['method']==m];aa=[a for a in calls if a['method']==m];bb=[b for b in blocks if b['method']==m]
        n=len(rr);correct=sum(r['final_correct'] for r in rr);answered=sum(r['answered'] for r in rr)
        prefill=total(aa,'prefill_seconds');generation=total(aa,'eval_seconds')
        rss=[b['boundary_max_rss_mib'] for b in bb if b['boundary_max_rss_mib'] is not None]
        result.append({'repetition':repetition,'method':m,'requests':n,'correct':correct,'accuracy':correct/n,
            'answered':answered,'answer_rate':answered/n,'answered_accuracy':correct/answered if answered else None,
            'final_format_compliance':sum(r['final_format_compliant'] for r in rr)/n,
            'model_format_compliance':sum(a['format_compliant'] for a in aa)/len(aa) if aa else None,
            'strict_correct':sum(r['final_correct'] and r['final_format_compliant'] for r in rr),
            'unresolved_content':n-answered,'parsed_content_errors':answered-correct,
            'model_calls':len(aa),'primary_calls':sum(a['stage']=='primary' for a in aa),
            'verification_calls':sum(a['stage']!='primary' for a in aa),'tool_calls':sum(r['tool_calls'] for r in rr),
            'input_tokens':total(aa,'input_tokens'),'output_tokens':total(aa,'output_tokens'),
            'prefill_seconds':prefill,'generation_seconds':generation,
            'model_processing_seconds':prefill+generation if prefill is not None and generation is not None else None,
            'server_total_seconds':total(aa,'total_model_seconds'),'load_seconds':total(aa,'load_seconds'),
            'processing_seconds':total(bb,'processing_seconds'),
            'mean_completion_latency_seconds':sum(r['completion_latency_seconds'] for r in rr)/n,
            'ollama_cpu_seconds':total(bb,'cpu_seconds'),'router_cpu_seconds':total(bb,'router_cpu_seconds'),
            'cpu_recorded_blocks':sum(b['cpu_seconds'] is not None for b in bb),'resource_blocks':len(bb),
            'rss_boundary_max_mib':max(rss) if rss else None,
            'verified_requests':sum(r['verification_executed'] for r in rr),
            'preserved_primary_after_verifier_failure':sum(r['verification_preserved_primary'] for r in rr),
            'harmful_verifications':sum(r['verification_outcome']=='harmful' for r in rr),
            'beneficial_verifications':sum(r['verification_outcome']=='beneficial' for r in rr),
            'redundant_correct_by_outcome':sum(r['verification_outcome']=='redundant_correct_by_outcome' for r in rr),
            'api_errors':sum(a['status']=='api_error' for a in aa),'cost':None,'cost_status':'unmeasured'})
    return result

def aggregate(summaries):
    output=[]
    for method in runtime.METHODS:
        rr=[r for r in summaries if r['method']==method];row={'method':method,'repetitions':len(rr)}
        for key in rr[0]:
            if key in ('repetition','method','cost_status'):continue
            values=[r[key] for r in rr]
            valid=all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in values)
            row.update({f'{key}_mean':statistics.mean(values) if valid else None,
                        f'{key}_min':min(values) if valid else None,f'{key}_max':max(values) if valid else None,
                        f'{key}_sample_std':statistics.stdev(values) if valid and len(values)>1 else None})
        row['cost_status']='unmeasured';output.append(row)
    return output

def paired(summaries,rows):
    result=[];lookup={(r['repetition'],r['method']):r for r in summaries}
    ix={(r['repetition'],r['method'],r['question']):r for r in rows}
    for s in summaries:
        if s['method']=='baseline':continue
        rep=s['repetition'];base=lookup[rep,'baseline'];rr=[r for r in rows if r['repetition']==rep and r['method']==s['method']]
        d={'repetition':rep,'method':s['method'],
            'quality_maintained_observed':s['accuracy']>=base['accuracy'] and s['answer_rate']>=base['answer_rate'],
            'correct_to_wrong':sum(ix[rep,'baseline',r['question']]['final_correct'] and not r['final_correct'] for r in rr),
            'wrong_to_correct':sum(not ix[rep,'baseline',r['question']]['final_correct'] and r['final_correct'] for r in rr)}
        for key in ('accuracy','answer_rate','model_calls','input_tokens','output_tokens','model_processing_seconds','processing_seconds','mean_completion_latency_seconds','ollama_cpu_seconds','rss_boundary_max_mib'):
            d[key+'_delta']=s[key]-base[key] if s[key] is not None and base[key] is not None else None
        result.append(d)
    return result

def legacy_rows(source):
    rows=read_csv(f'{source}_evaluation.csv');summary=[]
    for m in runtime.METHODS:
        rr=[r for r in rows if r['method']==m]
        parsed=[extract(r['question'],r['final_answer'],r['task_format']) for r in rr]
        summary.append({'dataset':'historical_single_run','method':m,'requests':len(rr),
            'old_official_accuracy':sum(r['final_correct']=='True' for r in rr)/len(rr),
            'old_selected_answer_content_accuracy':sum(p['prediction'] is not None and p['prediction']==r['expected'] for r,p in zip(rr,parsed))/len(rr),
            'old_selected_answer_format_compliance':sum(p['format_compliant'] for p in parsed)/len(rr),
            'interpretation':'Old selected final responses only, not a replay of changed verification adoption; no timing resimulation.'})
    return summary

def frozen_check(manifest):
    for path,digest in manifest['preserved_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Original changed: {path}')

def report(averages,summaries,pairs,legacy,metadata):
    def span(r,key,scale=1,n=3):
        vals=[r[key+s] for s in ('_mean','_min','_max')]
        return '미측정' if any(v is None for v in vals) else f'{vals[0]*scale:.{n}f} [{vals[1]*scale:.{n}f}, {vals[2]*scale:.{n}f}]'
    lines=['# 기존 절감 방법 반복 실측','',f"실행 ID `{metadata['run_id']}`. 수치는 실제 Ollama 실행이며 평균 [최소, 최대]를 표기한다.",'',
        '## 고정 설계와 형식 수정','',
        '기존 skipwork150문제(유형별30)를 동일하게3회 반복했다. 이것은 새 holdout 일반화 평가가 아니라 기존 평가의 신뢰성·변동 검증이다. '
        '모델 qwen2.5:1.5b, temperature=0, seed=42, 프롬프트·출력예산64/8/128·최대4개 묶음을 유지했다. '
        '새 라우팅 규칙을 추가하지 않았다. 블록 순서를 반복마다 사전 섞고, 방식 순서를 블록·반복에 걸쳐 순환했다.',
        '', 'Baseline은 기존 Hybrid 해결(도구 또는1.5B) 후 전수 검증하는 연구용 대조군이다. '
        'Early Exit는 안전 산술 도구의 재검증 생략, Selective는 별도 보정에서 순이득이 없던 쉬운 유형의 검증 생략이다. '
        '수정된 판정기로 별도100문제 보정을 한 번 수행하고 기존 규칙 그대로 프로필과 모든 사전 결정을 고정했다. 평가 정답은 실행 후 채점에만 사용했다.',
        '', '내용 추출은 전체 응답의 단일 문자/정수, 단일 FINAL_ANSWER, 정확한 옵션 표기/본문, 단일 코드펜스를 허용한다. '
        '기대 정답을 받지 않으며 애매한 답·옵션 본문 충돌·중복JSON키·누락/추가ID는 거부한다. '
        '묶음의 JSON 코드펜스는 내용으로 해석하지만 원래JSON출력 계약 준수로 인정하지 않는다. 검증이 해석되지 않으면 원래 답을 보존한다.',
        '', '내용 정확도는 해석된 정답/전체 요청이고 미해결도 분모에 넣는다. 미해결과 해석된 오답은 CSV에 분리했다. '
        '형식 준수율은 정규화 전 실제 채택 응답이 원래 계약(단일 대문자/정수, 묶음은 코드펜스 없는JSON)을 지킨 비율이다. '
        '시스템이 반환할 정규화된 답의 형식100%와 혼동하지 않는다. 모든 고유 모델 호출의 형식 준수율도 별도 기록한다.',
        '', '## 반복 평균과 범위','',
        '| 방식 | 내용 정확도 % | 답변율 % | 채택 응답 형식 준수 % | 모델 호출 | 입력 토큰 | 출력 토큰 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in averages:lines.append('| '+r['method']+' | '+' | '.join([span(r,'accuracy',100,2),span(r,'answer_rate',100,2),span(r,'final_format_compliance',100,2),span(r,'model_calls',1,0),span(r,'input_tokens',1,0),span(r,'output_tokens',1,0)])+' |')
    lines+=['','| 방식 | 처리 벽시계(초) | 평균 완료 지연(ms) | prefill+generation(초) | 서버 total(초) | Ollama CPU(초) | RSS 경계 최대(MiB) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in averages:lines.append('| '+r['method']+' | '+' | '.join([span(r,'processing_seconds'),span(r,'mean_completion_latency_seconds',1000,2),span(r,'model_processing_seconds'),span(r,'server_total_seconds'),span(r,'ollama_cpu_seconds'),span(r,'rss_boundary_max_mib',1,1)])+' |')
    lines+=['','각 블록 시작에 최대4개 요청이 함께 도착한다. 완료 지연은 순차 대기/묶음 종료까지 기다린 시간을 포함한다. '
        '처리 벽시계는 방식별 블록 합계이며, 서버 total은 API가 보고한 total_duration 합계다. '
        '모델 처리시간은 prompt_eval_duration+eval_duration 합계이며 로딩·HTTP·Python·계측 비용과 다르다. '
        '서버 duration은 내부 벽시계 계측으로 GPU 총FLOPs의 직접 측정이 아니다.',
        '', '## Baseline 대비 반복별 판정','',
        '| 방식 | 품질 유지 반복 수 | 더 짧은 처리시간 반복 수 | 더 적은 모델 처리시간 반복 수 | 더 적은 CPU 반복 수 |',
        '|---|---:|---:|---:|---:|']
    for m in runtime.METHODS[1:]:
        pp=[p for p in pairs if p['method']==m]
        count=lambda k:sum(p[k] is not None and p[k]<0 for p in pp)
        lines.append(f"| {m} | {sum(p['quality_maintained_observed'] for p in pp)}/{len(pp)} | {count('processing_seconds_delta')}/{len(pp)} | {count('model_processing_seconds_delta')}/{len(pp)} | {count('ollama_cpu_seconds_delta')}/{len(pp)} |")
    lines+=['','정답률 유지란 각 반복에서 Baseline 이상인 내용 정확도와 답변율이라는 관측 기준이다. '
        '통계적 동등성/비열등성 검정이나 모든 개별 문항의 보장을 뜻하지 않는다. 문항별 정답→오답 손실은 paired.csv에 별도 기록했다. '
        '세 반복은 같은150문제와 같은seed를 재사용하므로 독립450문제로 보지 않으며, 최소/최대 범위는 신뢰구간이 아니다.',
        '', '## 과거 결과와 구분','',
        '| 방식 | 과거 공식 정확도 % | 과거 채택 답 내용만 재해석 % | 새 반복 평균 내용 정확도 % |',
        '|---|---:|---:|---:|']
    for old,new in zip(legacy,averages):lines.append(f"| {old['method']} | {old['old_official_accuracy']*100:.2f} | {old['old_selected_answer_content_accuracy']*100:.2f} | {new['accuracy_mean']*100:.2f} |")
    lines+=['','과거 공식 CSV/보고서는 수정하지 않았다. 과거 채택 답 재해석은 이미 반환된 최종 텍스트만 새 파서로 읽은 내용 지표다. '
        '이전 묶음 검증은 형식 실패로 검증 답을 버렸지만 새 실행은 코드펜스를 안전하게 읽어 채택한다. '
        '따라서 새 결과의 변화에는 실제 검증 답 채택과 보정 재산출도 포함되며, 숫자 차이를 새 알고리즘 효과로 주장하지 않는다.',
        '', '## 비용·자원·보정 한계','',
        f"보정 {metadata['calibration_calls']}호출, {metadata['calibration_seconds']:.3f}초, 입력{metadata['calibration_input_tokens']}/출력{metadata['calibration_output_tokens']}토큰은 반복 표에서 분리했다. "
        f"평가 전체 루프 {metadata['evaluation_seconds']:.3f}초는 모니터링·채점·반복 파일 저장을 포함한다. 워밍업1회는 별도JSON에 기록했다. "
        '처리 벽시계에는 추출·라우팅·검증 실패·원래 답 보존을 포함하며 ps수집/사후채점은 제외한다.',
        '', 'Ollama 프로세스군 CPU 카운터와 Python process_time을 실제 수집했다. RSS는 블록 양끝 실제 표본으로 '
        '피크/전용 메모리나 모델별 독립 메모리가 아니다. 같은 로컬 서버·가중치·prefix cache를 공유하는 warm 상태 실험이다. '
        '캐시를 강제로 비우지 않았고 백그라운드 부하·발열·순서 효과를 완전히 분리하지 못했다. '
        '입력토큰은 서버 카운트이며 캐시를 제외한 실제prefill FLOPs를 뜻하지 않는다.',
        '', '**금전 비용·전력·GPU FLOPs는 미측정이며 추정 절감률도 만들지 않았다.** '
        '호출·토큰·모델 처리시간·CPU·사용자 지연은 각각 다른 지표로 판단한다. 보정 최초 비용의 상각도 가정하지 않는다. '
        '영어 객관식·수도 지식·합성 논리/지문·코드 출력 예측이라는 평가 범위가 한계다.',
        '', '## 산출물','',*[f'- [{s}]({metadata["prefix_name"]}_{s})' for s in ('manifest.json','profile.json','plans.json','warmup.json','calibration.csv','calibration_calls.csv','evaluation.csv','calls.csv','blocks.csv','repeat_summary.csv','aggregate.csv','paired.csv','legacy_regrade.csv','metadata.json')],'']
    return '\n'.join(lines)

def run(data_dir=ROOT/'data',source=SOURCE,repeats=REPEATS):
    if repeats<2:raise ValueError('At least two real repetitions required')
    data_dir=Path(data_dir);source=Path(source)
    old=json.loads(Path(f'{source}_manifest.json').read_text());tasks=old['evaluation'];calibration=old['calibration']
    if len(tasks)!=150 or len({task_key(t['question']) for t in tasks})!=150:raise ValueError('Expected same150')
    if {task_key(t['question']) for t in tasks}&{task_key(t['question']) for t in calibration}:raise ValueError('Calibration overlap')
    references=check_references(tasks);check_ollama([runtime.MODEL])
    with urllib.request.urlopen('http://localhost:11434/api/tags',timeout=10) as f:tags=json.load(f)
    preserved={}
    for folder in (ROOT/'src',ROOT/'tests',data_dir):
        for path in folder.iterdir():
            if path.is_file():preserved[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in ROOT.iterdir():
        if path.is_file():preserved[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    run_id=uuid4().hex;prefix=data_dir/f'efficiency_repeat_{run_id}'
    questions=[t['question'] for t in tasks];plan=runtime.batch_plan(questions);plans=[]
    for rep in range(repeats):
        order=[b[:] for b in plan];random.Random(42000+rep).shuffle(order);plans.append(order)
    manifest={'run_id':run_id,'repeats':repeats,'source':str(source),'source_manifest_sha256':hashlib.sha256(Path(f'{source}_manifest.json').read_bytes()).hexdigest(),
        'tasks':tasks,'calibration':calibration,'block_orders':plans,'temperature':0,'seed':42,
        'protocol':'efficiency-protocol-v1','preserved_sha256':preserved,'platform':platform.platform(),
        'python':platform.python_version(),'ollama_tags':tags}
    write_json(Path(f'{prefix}_manifest.json'),manifest)
    frozen={'manifest':hashlib.sha256(Path(f'{prefix}_manifest.json').read_bytes()).hexdigest()}
    warmup=runtime.send('Choose a color. A) Red B) Table C) Chair D) Door. Return only A, B, C or D.','warmup')
    write_json(Path(f'{prefix}_warmup.json'),warmup)
    calrows=[];calcalls=[];start=time.perf_counter()
    for i,t in enumerate(calibration):
        p=runtime.primary(t['question']);a,v=runtime.verify(t['question'],p['answer']);calcalls.extend(p['calls']);calcalls.append(a)
        final=v['prediction'] if v['prediction'] is not None else p['prediction']
        calrows.append({**t,'inferred_category':runtime.features(t['question'])['category'],
            'primary_answer':p['answer'],'primary_prediction':p['prediction'],'verification_answer':a['answer'],
            'final_prediction':final,'primary_correct':p['prediction']==t['expected'],'final_correct':final==t['expected']})
        if (i+1)%20==0:print(f'Corrected-protocol calibration {i+1}/{len(calibration)}',flush=True)
    calibration_seconds=time.perf_counter()-start;profile=runtime.make_profile(calrows)
    write_json(Path(f'{prefix}_profile.json'),profile)
    decisions=[{'question':q,'method':m,'decision':runtime.preflight(q,m,profile)} for q in questions for m in runtime.METHODS]
    write_json(Path(f'{prefix}_plans.json'),decisions)
    for key in ('profile','plans'):frozen[key]=hashlib.sha256(Path(f'{prefix}_{key}.json').read_bytes()).hexdigest()
    write_records(Path(f'{prefix}_calibration.csv'),calrows);write_records(Path(f'{prefix}_calibration_calls.csv'),calcalls)
    print('Frozen recalibration '+json.dumps(profile),flush=True)
    allrows=[];allcalls=[];allblocks=[];summaries=[];start=time.perf_counter()
    for rep,order in enumerate(plans):
        rows=[];calls=[];blocks=[]
        for block_index,indices in enumerate(order):
            qq=[questions[i] for i in indices];offset=(block_index+rep)%len(runtime.METHODS)
            for method in runtime.METHODS[offset:]+runtime.METHODS[:offset]:
                before=process_snapshot();block=runtime.execute_block(qq,method,profile);after=process_snapshot()
                block_id=f'R{rep+1}:B{block_index:03d}:{method}'
                for row,index in zip(block['rows'],indices):rows.append({**score(row,tasks[index]),'repetition':rep+1,'block_id':block_id})
                calls.extend({**a,'repetition':rep+1,'method':method,'block_id':block_id} for a in block['calls'])
                blocks.append({'repetition':rep+1,'block_id':block_id,'method':method,'request_count':len(indices),
                    'question_indices':indices,'processing_seconds':block['processing_seconds'],'router_cpu_seconds':block['router_cpu_seconds'],
                    'before':before,'after':after,**resource_delta(before,after)})
            if (block_index+1)%10==0:print(f'Real repeat {rep+1}/{repeats}: blocks {block_index+1}/{len(order)}',flush=True)
        ss=summarize(rows,calls,blocks,rep+1);summaries.extend(ss)
        for suffix,records in [('evaluation',rows),('calls',calls),('blocks',blocks),('summary',ss)]:write_records(Path(f'{prefix}_r{rep+1}_{suffix}.csv'),records)
        allrows.extend(rows);allcalls.extend(calls);allblocks.extend(blocks)
        print('REPEAT SUMMARY '+json.dumps(ss),flush=True)
    evaluation_seconds=time.perf_counter()-start
    averages=aggregate(summaries);pairs=paired(summaries,allrows);legacy=legacy_rows(source)
    frozen_check(manifest)
    for key,digest in frozen.items():
        if hashlib.sha256(Path(f'{prefix}_{key}.json').read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen policy modified')
    if len({a['call_id'] for a in allcalls})!=len(allcalls):raise RuntimeError('Duplicate actual call accounting')
    metadata={'run_id':run_id,'prefix_name':prefix.name,'real_inference':True,'repetitions':repeats,
        'frozen_sha256':frozen,'preserved_files_checked':len(preserved),'preserved_unchanged':True,
        'calibration_seconds':calibration_seconds,'calibration_calls':len(calcalls),
        'calibration_input_tokens':total(calcalls,'input_tokens'),'calibration_output_tokens':total(calcalls,'output_tokens'),
        'evaluation_seconds':evaluation_seconds,'evaluation_rows':len(allrows),'unique_evaluation_calls':len(allcalls),'cost_status':'unmeasured'}
    for suffix,records in [('evaluation',allrows),('calls',allcalls),('blocks',allblocks),('repeat_summary',summaries),
                           ('aggregate',averages),('paired',pairs),('legacy_regrade',legacy),('reference_checks',references)]:
        write_records(Path(f'{prefix}_{suffix}.csv'),records)
    write_json(Path(f'{prefix}_metadata.json'),metadata)
    with Path(f'{prefix}_report.md').open('x',encoding='utf-8') as f:f.write(report(averages,summaries,pairs,legacy,metadata))
    print(f'REPORT: {prefix}_report.md',flush=True)
    return prefix

if __name__=='__main__':
    try:run()
    except (OSError,ValueError,RuntimeError) as error:raise SystemExit(f'반복 실측 중단: {error}')
