"""Pre-solve omission experiment: unique calls, real finite-batch latency and resource accounting."""
from collections import Counter
import hashlib,json,random,time
from pathlib import Path
from uuid import uuid4
import urllib.request
if __package__:
    from . import skipwork_runtime as runtime
    from .skipwork_dataset import build,ROOT,SEED
    from .solver300_experiment import check_references
    from .verify_routing200 import process_snapshot,resource_delta
    from .routing_runner import check_ollama
    from .adaptive_experiment import write_json
    from .tool_experiment import write_records
else:
    import skipwork_runtime as runtime
    from skipwork_dataset import build,ROOT,SEED
    from solver300_experiment import check_references
    from verify_routing200 import process_snapshot,resource_delta
    from routing_runner import check_ollama
    from adaptive_experiment import write_json
    from tool_experiment import write_records

CATEGORIES=('arithmetic','knowledge','logic','coding','comprehension')


def score(row,task):
    p=row['primary_prediction'];f=row['final_prediction'];v=row['verification_prediction']
    pc=p is not None and p==task['expected'];fc=f is not None and f==task['expected']
    if not row['verification_executed']:outcome='not_executed'
    elif not pc and fc:outcome='beneficial'
    elif pc and not fc:outcome='harmful'
    elif pc and fc:outcome='redundant_correct_by_outcome'
    else:outcome='still_wrong_or_unresolved'
    return {**row,'category':task['category'],'expected':task['expected'],'task_format':task['task_format'],
        'primary_correct':pc,'final_correct':fc,'answered':f is not None,'verification_outcome':outcome,
        'verification_format_error':row['verification_executed'] and v is None,
        'tool_misapplication':row['primary_source']=='tool' and (task['category']!='arithmetic' or not pc)}


def summarize(rows,calls,blocks):
    result=[]
    for method in runtime.METHODS:
        rr=[r for r in rows if r['method']==method];aa=[a for a in calls if a['method']==method];bb=[b for b in blocks if b['method']==method]
        inputs=[a['input_tokens'] for a in aa];outputs=[a['output_tokens'] for a in aa];evals=[a['eval_seconds'] for a in aa]
        cpu=[b['cpu_seconds'] for b in bb];rss=[b['boundary_max_rss_mib'] for b in bb if b['boundary_max_rss_mib'] is not None]
        correct=sum(r['final_correct'] for r in rr);answered=sum(r['answered'] for r in rr)
        result.append({'method':method,'requests':len(rr),'primary_correct':sum(r['primary_correct'] for r in rr),
            'correct':correct,'accuracy':correct/len(rr),'answered':answered,'answer_rate':answered/len(rr),
            'answered_accuracy':correct/answered if answered else None,'model_calls':len(aa),
            'primary_model_calls':sum(a['stage']=='primary' for a in aa),'verification_model_calls':sum(a['stage']!='primary' for a in aa),
            'verified_requests':sum(r['verification_executed'] for r in rr),'skipped_verifications':sum(not r['verification_executed'] for r in rr),
            'tool_calls':sum(r['tool_calls'] for r in rr),'input_tokens':sum(inputs) if all(v is not None for v in inputs) else None,
            'output_tokens':sum(outputs) if all(v is not None for v in outputs) else None,
            'token_recorded_calls':sum(i is not None and o is not None for i,o in zip(inputs,outputs)),
            'ollama_eval_seconds':sum(evals) if all(v is not None for v in evals) else None,
            'processing_seconds':sum(b['processing_seconds'] for b in bb),
            'mean_completion_latency_seconds':sum(r['completion_latency_seconds'] for r in rr)/len(rr),
            'ollama_cpu_seconds':sum(cpu) if all(v is not None for v in cpu) else None,
            'cpu_recorded_blocks':sum(v is not None for v in cpu),'resource_blocks':len(bb),
            'router_cpu_seconds':sum(b['router_cpu_seconds'] for b in bb),'rss_boundary_max_mib':max(rss) if rss else None,
            'verification_outcomes':dict(Counter(r['verification_outcome'] for r in rr)),
            'verification_format_failures':sum(r['verification_format_error'] for r in rr),
            'primary_format_failures':sum(r['primary_prediction'] is None for r in rr),
            'api_errors':sum(a['status']=='api_error' for a in aa),
            'tool_misapplications':sum(r['tool_misapplication'] for r in rr),'cost':None,'cost_status':'unmeasured'})
    return result


def comparisons(rows,summaries):
    ix={(r['method'],r['question']):r for r in rows};base=next(s for s in summaries if s['method']=='baseline');result=[]
    for s in summaries:
        if s['method']=='baseline':continue
        rr=[r for r in rows if r['method']==s['method']]
        result.append({'method':s['method'],'accuracy_delta':s['accuracy']-base['accuracy'],
            'answer_rate_delta':s['answer_rate']-base['answer_rate'],'model_call_delta':s['model_calls']-base['model_calls'],
            'input_token_delta':s['input_tokens']-base['input_tokens'] if s['input_tokens'] is not None and base['input_tokens'] is not None else None,
            'output_token_delta':s['output_tokens']-base['output_tokens'] if s['output_tokens'] is not None and base['output_tokens'] is not None else None,
            'processing_seconds_delta':s['processing_seconds']-base['processing_seconds'],
            'mean_latency_delta':s['mean_completion_latency_seconds']-base['mean_completion_latency_seconds'],
            'correct_to_wrong':sum(ix[('baseline',r['question'])]['final_correct'] and not r['final_correct'] for r in rr),
            'wrong_to_correct':sum(not ix[('baseline',r['question'])]['final_correct'] and r['final_correct'] for r in rr),
            'quality_non_degradation_observed':s['accuracy']>=base['accuracy'] and s['answer_rate']>=base['answer_rate'],
            'lower_latency_observed':s['mean_completion_latency_seconds']<base['mean_completion_latency_seconds'],
            'lower_processing_time_observed':s['processing_seconds']<base['processing_seconds'],
            'interpretation':'Calls and tokens are measured workload indicators, not FLOPs or money'})
    return result


def report(summaries,comparison,profile,metadata):
    def fmt(v,n=3):return f'{v:.{n}f}' if v is not None else '미측정'
    lines=['# 불필요한 추론·검증 생략 연구','',f"실행 ID `{metadata['run_id']}`.",'',
        '## 사전에 고정한 설계','',
        'Baseline은 기존 Hybrid의 해결 경로(산술 도구, 비산술1.5B)에 전수1.5B 재검증을 추가한 연구용 대조군이다. '
        '이것이 이전 Hybrid보다 우수한 생산 기준선이라는 뜻은 아니다. 검증 결과가 파싱되면 최종 답으로 채택하고, '
        '검증 형식이 실패하면 원래 답을 유지한다. 검증이 오히려 정답을 오답으로 바꿀 수 있다.',
        '', '- Early Exit: 실행 전 안전 산술식임을 확인하면 도구 답 이후 모델 재검증을 생략한다. 기존 도구 방식의 통제 실험이며 새 발명으로 주장하지 않는다.',
        '- Selective: 안전 도구 증명은 생략하고, 쉬운 유형의 보정20개 이상에서 검증 수정건수≤악화건수이면 검증을 생략한다. 어려운 요청·근거 부족은 검증한다.',
        '- 가설1(Selective): 기본 답의 자신감이 아니라 검증의 예상 순이득을 기준으로 판단하면 개선하지 못하는 검증을 줄일 수 있다. 보정의 점추정 규칙으로 정확도 보장이 아니다.',
        '- 가설2(Budget Cap): 객관식의 첫 모델 출력 상한을64에서8토큰으로 줄이면 호출 수가 같아도 출력 연산을 줄일 수 있다. 검증 예산은 그대로다. 잘림 위험을 기록한다.',
        '- 가설3(Batch Verify): 해결 전 유형·입력 길이만으로 최대4개·2600자 묶음을 고정해 검증을 한 호출로 합친다. 호출 감소가 문맥·출력·대기 증가를 상쇄할지는 실제 측정으로 판단한다.',
        '', '새150문제는 유형별30개다. 보정은 별도 저장된100문제이며 현재 평가와 과거 모든 질문 줄기·교환 중복을 제외했다. '
        '보정·정책·출력 계약·묶음계획은 평가 전에 고정했다. 생략 함수는 질문과 보정 통계만 받고 답안·기대 정답·유형 라벨을 받지 않는다.',
        '', '## 실제 결과','',
        '| 방식 | 정확도 | 답변율 | 모델 호출 | 입력/출력 토큰 | 처리 벽시계 합계(초) | 평균 완료 지연(ms) | 생성 시간(초) |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['method']} | {s['accuracy']:.2%} | {s['answer_rate']:.2%} | {s['model_calls']} | {fmt(s['input_tokens'],0)}/{fmt(s['output_tokens'],0)} | {s['processing_seconds']:.3f} | {s['mean_completion_latency_seconds']*1000:.3f} | {fmt(s['ollama_eval_seconds'])} |")
    lines+=['','모든 요청은 해당 블록 시작에 도착한 것으로 정의했다. 순차 방식도 앞 요청 대기를 포함하고 묶음 방식은 같은 묶음 검증 종료를 기다린다. '
        '처리 시간은 메서드별 블록의 실제 벽시계 합계이며, 겹치는 요청 지연을 더한 값과 다르다. 공유 검증 호출의 토큰·시간을 요청별로 복제하거나 임의 분할하지 않았다.',
        '', '## 실제 자원과 검증 효과','',
        '| 방식 | Ollama CPU(초) | Python CPU(초) | RSS 경계최대(MiB) | 검증 요청/생략 요청 | 결과별 검증 효과 |',
        '|---|---:|---:|---:|---|---|']
    for s in summaries:
        lines.append(f"| {s['method']} | {fmt(s['ollama_cpu_seconds'])} | {fmt(s['router_cpu_seconds'])} | {fmt(s['rss_boundary_max_mib'],1)} | {s['verified_requests']}/{s['skipped_verifications']} | {s['verification_outcomes']} |")
    lines+=['','redundant_correct_by_outcome은 사후에 원래 정답과 최종 정답이 모두 맞았던 검증을 뜻한다. '
        '실행 시 정답을 알고 생략한 것이 아니며, 이런 검증이 모든 목적에서 무가치하다고 단정하지 않는다. '
        'correct→wrong 악화와 wrong→correct 개선, 파싱 실패를 문항별로 분리했다.',
        '', '## 기준선 대비 판정','', '```json',json.dumps(comparison,ensure_ascii=False,indent=2),'```','',
        '모델 호출이 감소해도 입력/출력 토큰·실제 생성 시간·CPU·처리 시간까지 감소했는지는 각각 확인해야 한다. '
        '이 값들은 작업량과 시간의 대리 지표이며 GPU FLOPs나 총 하드웨어 연산량의 직접 측정이 아니다. '
        '정확도가 떨어지면 무조건 우수하다고 하지 않는다. 같은 답변율의 단일 실행 관측값으로 일반적 우위를 확정하지 않는다.',
        '', '## 보정과 계측 한계','', '```json',json.dumps(profile,ensure_ascii=False,indent=2),'```','',
        f"보정 벽시계 {metadata['calibration_seconds']:.3f}초, 평가 전체 루프 {metadata['evaluation_seconds']:.3f}초. "
        '보정 호출·토큰은 calibration_calls.csv에 분리했다. 최초 보정 비용을 회수했다고 가정하지 않는다. '
        '점추정 기반 검증 순이득이 다른 문제로 일반화될 보장은 없으며 어려움 판정은 요청의 문법/길이 휴리스틱이다.',
        '', 'Ollama 프로세스군의 CPU 카운터 차·RSS 경계 표본과 Python process_time을 분리해 실제 수집했다. '
        'RSS는 상주·공유 메모리의 표본으로 전용/피크 메모리가 아니다. GPU·전력·FLOPs·금전 비용은 미측정이다. '
        '입출력 토큰은 서버 기록이며 캐시를 제외한 실제 prefill 연산량을 의미하지 않는다. '
        '자원 수집·사후 채점·워밍업은 처리 시간에서 제외하고 전체 루프에는 모니터·채점도 포함한다.',
        '', '검증 묶음은 JSON 출력과 정확한 id 집합을 요구했다. 파싱 실패면 원래 답을 유지하고 추가 모델 호출로 숨겨 복구하지 않았다. '
        '호출 감소가 실패한 검증이나 단순 답안 보존에서 나온 경우도 효과 분석에 남겼다. '
        '같은 문제를 반복하는 로컬 측정이라 prefix cache·발열·배경 부하 영향이 있고 블록 순서만 균형화했다.',
        '', '문제는 영어 객관식·코드 출력 예측·합성 논리/지문 템플릿이며 지식은 수도 문제에 편중된다. '
        '지식 보정은 다른 주제이므로 분포 이동 한계도 있다. 새 질문 줄기는 중복이 없지만 개념·템플릿 계열이 모두 독립적이지 않다. '
        '원본 파일·고정 명세·프로필 해시를 검사했고, 평가 라벨은 결과 생성 후 채점에만 사용했다. 유료 API나 모델 설치는 사용하지 않았다.',
        '', '## 파일','',*[f"- [{s}]({metadata['prefix_name']}_{s})" for s in ('manifest.json','profile.json','plans.json','calibration.csv','calibration_calls.csv','evaluation.csv','calls.csv','blocks.csv','summary.csv','comparison.csv','reference_checks.csv','metadata.json')],'']
    return '\n'.join(lines)


def run(data_dir=ROOT/'data'):
    data_dir=Path(data_dir)
    preserved={p:hashlib.sha256(p.read_bytes()).hexdigest() for folder in (ROOT/'src',data_dir) for p in folder.iterdir() if p.is_file()}
    calibration,evaluation=build(data_dir);references=check_references(evaluation)
    questions=[t['question'] for t in evaluation];plan=runtime.batch_plan(questions)
    random.Random(SEED).shuffle(plan)
    check_ollama([runtime.MODEL])
    with urllib.request.urlopen('http://localhost:11434/api/tags',timeout=10) as response:tags=json.load(response)
    run_id=uuid4().hex;prefix=data_dir/f'skipwork_{run_id}'
    manifest={'version':runtime.VERSION,'seed':SEED,'calibration':calibration,'evaluation':evaluation,'batch_plan':plan,
        'max_batch_items':4,'max_batch_chars':2600,'budget_cap_choice_tokens':8,'default_tokens':64,'batch_verification_tokens':128,
        'preserved_sha256':{str(p):h for p,h in preserved.items()}}
    manifest_path=Path(f'{prefix}_manifest.json');write_json(manifest_path,manifest)
    manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runtime.send('Choose a color. A) Red B) Table C) Chair D) Door. Return only A, B, C or D.','warmup')
    calrows=[];calcalls=[];start=time.perf_counter()
    for i,t in enumerate(calibration):
        p=runtime.primary(t['question']);a,pred,mode=runtime.verify(t['question'],p['answer'])
        calcalls.extend(p['calls']);calcalls.append(a)
        final=pred if pred is not None else p['prediction']
        calrows.append({**t,'inferred_category':runtime.features(t['question'])['category'],
            'primary_answer':p['answer'],'primary_prediction':p['prediction'],'verification_answer':a['answer'],
            'final_prediction':final,'primary_correct':p['prediction']==t['expected'],'final_correct':final==t['expected']})
        if (i+1)%10==0:print(f'Calibration {i+1}/100',flush=True)
    cal_seconds=time.perf_counter()-start;profile=runtime.make_profile(calrows)
    write_json(Path(f'{prefix}_profile.json'),profile);profile_hash=hashlib.sha256(Path(f'{prefix}_profile.json').read_bytes()).hexdigest()
    preplans=[{'question':q,'method':m,'decision':runtime.preflight(q,m,profile)} for q in questions for m in runtime.METHODS]
    write_json(Path(f'{prefix}_plans.json'),preplans);plans_hash=hashlib.sha256(Path(f'{prefix}_plans.json').read_bytes()).hexdigest()
    write_records(Path(f'{prefix}_calibration.csv'),calrows);write_records(Path(f'{prefix}_calibration_calls.csv'),calcalls)
    print('Frozen profile '+json.dumps(profile,ensure_ascii=False),flush=True)
    rows=[];calls=[];blocks=[];start=time.perf_counter()
    for bi,indices in enumerate(plan):
        qq=[questions[i] for i in indices];offset=bi%len(runtime.METHODS)
        for method in runtime.METHODS[offset:]+runtime.METHODS[:offset]:
            before=process_snapshot();block=runtime.execute_block(qq,method,profile);after=process_snapshot()
            block_id=f'B{bi:03d}:{method}'
            for row,index in zip(block['rows'],indices):rows.append({**score(row,evaluation[index]),'block_id':block_id})
            calls.extend({**a,'method':method,'block_id':block_id} for a in block['calls'])
            blocks.append({'block_id':block_id,'method':method,'request_count':len(indices),'question_indices':indices,
                'processing_seconds':block['processing_seconds'],'router_cpu_seconds':block['router_cpu_seconds'],
                'before':before,'after':after,**resource_delta(before,after)})
        if (bi+1)%5==0:print(f'Actual evaluation blocks {bi+1}/{len(plan)}',flush=True)
    eval_seconds=time.perf_counter()-start;summaries=summarize(rows,calls,blocks);comparison=comparisons(rows,summaries)
    for path,h in preserved.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise RuntimeError(f'Original changed: {path}')
    for path,h in [(manifest_path,manifest_hash),(Path(f'{prefix}_profile.json'),profile_hash),(Path(f'{prefix}_plans.json'),plans_hash)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen data/policy changed')
    if len({a['call_id'] for a in calls})!=len(calls):raise RuntimeError('Shared calls counted more than once')
    metadata={'run_id':run_id,'prefix_name':prefix.name,'calibration_seconds':cal_seconds,'evaluation_seconds':eval_seconds,
        'manifest_sha256':manifest_hash,'profile_sha256':profile_hash,'plans_sha256':plans_hash,
        'ollama_tags':tags,'preserved_files_unchanged':True,'real_inference':True,'cost_status':'unmeasured'}
    write_records(Path(f'{prefix}_evaluation.csv'),rows);write_records(Path(f'{prefix}_calls.csv'),calls)
    write_records(Path(f'{prefix}_blocks.csv'),blocks);write_records(Path(f'{prefix}_summary.csv'),summaries)
    write_records(Path(f'{prefix}_comparison.csv'),comparison);write_records(Path(f'{prefix}_reference_checks.csv'),references)
    write_json(Path(f'{prefix}_metadata.json'),metadata)
    with Path(f'{prefix}_report.md').open('x',encoding='utf-8') as f:f.write(report(summaries,comparison,profile,metadata))
    for s in summaries:print(json.dumps(s,ensure_ascii=False),flush=True)
    print(f'REPORT: {prefix}_report.md',flush=True)
    return prefix


if __name__=='__main__':
    try:run()
    except (OSError,ValueError,RuntimeError) as error:raise SystemExit(f'생략 연구 중단: {error}')
