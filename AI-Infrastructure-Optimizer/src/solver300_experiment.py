"""Local-only, preregistered four-strategy comparison; calibration separate from evaluation."""
from collections import Counter
import ast
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import urllib.request
from uuid import uuid4

if __package__:
    from . import solver300_runtime as runtime
    from .solver300_dataset import build,prior_keys,task_key,ROOT,SEED
    from .solver300_protocol import content_extract
    from .verify_routing200 import process_snapshot,resource_delta,reference_answer
    from .routing_runner import check_ollama
    from .tool_experiment import write_records
    from .adaptive_experiment import write_json
else:
    import solver300_runtime as runtime
    from solver300_dataset import build,prior_keys,task_key,ROOT,SEED
    from solver300_protocol import content_extract
    from verify_routing200 import process_snapshot,resource_delta,reference_answer
    from routing_runner import check_ollama
    from tool_experiment import write_records
    from adaptive_experiment import write_json

CATEGORIES=('arithmetic','knowledge','logic','coding','comprehension')


def check_references(tasks):
    checks=[]
    for t in tasks:
        if t['task_format']=='integer':
            if str(reference_answer(t['question']))!=t['expected']: raise ValueError('Arithmetic reference mismatch')
        if t['category']!='coding': continue
        code=t['reference_code']
        if code not in t['question']: raise ValueError('Code reference is not the question code')
        tree=ast.parse(code)
        if any(isinstance(n,(ast.Import,ast.ImportFrom,ast.While)) for n in ast.walk(tree)): raise ValueError('Unexpected reference code')
        functions={n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                safe=(isinstance(n.func,ast.Name) and n.func.id in {'print','len','range','sum'}|functions) or (isinstance(n.func,ast.Attribute) and n.func.attr in {'get','append','copy','strip','split'})
                if not safe: raise ValueError('Unsafe reference function')
        result=subprocess.run([sys.executable,'-B','-I','-c',code],capture_output=True,text=True,timeout=3,check=True)
        option=next(line[3:] for line in t['question'].splitlines() if line.startswith(t['expected']+') '))
        if result.stdout.strip()!=option: raise ValueError('Coding reference mismatch')
        checks.append({'question':t['question'],'reference_output':result.stdout.strip(),'expected_option':option,'matches':True})
    return checks


def summarize(rows):
    result=[]
    for strategy in runtime.STRATEGIES:
        for category in ('all',)+CATEGORIES:
            group=[r for r in rows if r['strategy']==strategy and (category=='all' or r['category']==category)]
            if not group: continue
            aa=[a for r in group for a in r['attempts']]
            output=[a['output_tokens'] for a in aa];inputs=[a['input_tokens'] for a in aa]
            backend=[r['cpu_seconds'] for r in group];rss=[r['boundary_max_rss_mib'] for r in group if r['boundary_max_rss_mib'] is not None]
            answered=sum(r['status']=='answered' for r in group);correct=sum(r['correct'] for r in group)
            result.append({'strategy':strategy,'category':category,'requests':len(group),'correct':correct,'answered':answered,
                'accuracy':correct/len(group),'answer_rate':answered/len(group),'answered_accuracy':correct/answered if answered else None,
                'mean_latency_seconds':sum(r['elapsed_seconds'] for r in group)/len(group),
                'total_seconds':sum(r['elapsed_seconds'] for r in group),'model_calls':len(aa),
                'small_calls':sum(a['model']==runtime.SMALL for a in aa),'large_calls':sum(a['model']==runtime.LARGE for a in aa),
                'tool_calls':sum(r['tool_calls'] for r in group),'fallback_requests':sum(r['fallback'] for r in group),
                'fallback_seconds':sum(r['fallback_seconds'] for r in group),'format_errors':sum(r['format_errors'] for r in group),
                'noncompliant_but_parseable':sum(not a.get('format_compliant',False) and a.get('prediction') is not None for a in aa),
                'content_errors':sum(r['status']=='answered' and not r['correct'] for r in group),
                'api_errors':sum(a['status']=='api_error' for a in aa),
                'input_tokens':sum(inputs) if all(v is not None for v in inputs) else None,
                'output_tokens':sum(output) if all(v is not None for v in output) else None,
                'token_recorded_calls':sum(i is not None and o is not None for i,o in zip(inputs,output)),
                'ollama_cpu_seconds':sum(backend) if all(v is not None for v in backend) else None,
                'cpu_samples':sum(v is not None for v in backend),'router_cpu_seconds':sum(r['router_cpu_seconds'] for r in group),
                'rss_boundary_max_mib':max(rss) if rss else None,'rss_samples':len(rss),
                'classification_errors':sum(r['decision']['category']!=r['category'] for r in group),
                'tool_misapplications':sum(r['source']=='tool' and (r['category']!='arithmetic' or not r['correct']) for r in group),
                'cost':None,'cost_status':'unmeasured'})
    return result


def hybrid_comparison(rows,summaries,bootstrap_samples=2000):
    index={(r['strategy'],r['question']):r for r in rows}
    groups=[[r for r in rows if r['strategy']=='rule' and r['category']==c] for c in CATEGORIES]
    base=[r for g in groups for r in g]
    rng=random.Random(SEED+1)
    accuracy_deltas=[];latency_deltas=[]
    for _ in range(bootstrap_samples):
        sample=[rng.choice(g) for g in groups for _ in range(len(g))]
        accuracy_deltas.append(sum(int(index[('hybrid',r['question'])]['correct'])-int(r['correct']) for r in sample)/len(sample))
        latency_deltas.append(sum(index[('hybrid',r['question'])]['elapsed_seconds']-r['elapsed_seconds'] for r in sample)/len(sample))
    def interval(values):
        values=sorted(values);return [values[int(.025*len(values))],values[int(.975*len(values))-1]]
    overall={s['strategy']:s for s in summaries if s['category']=='all'}
    r,h=overall['rule'],overall['hybrid']
    common=[(r,index[('hybrid',r['question'])]) for r in base if r['status']=='answered' and index[('hybrid',r['question'])]['status']=='answered']
    bt=sum(a['elapsed_seconds'] for a,b in common);ht=sum(b['elapsed_seconds'] for a,b in common)
    quality=h['accuracy']>=r['accuracy'] and h['answer_rate']>=r['answer_rate']
    return {'baseline':'rule','candidate':'hybrid','accuracy_delta':h['accuracy']-r['accuracy'],
        'answer_rate_delta':h['answer_rate']-r['answer_rate'],'mean_latency_delta':h['mean_latency_seconds']-r['mean_latency_seconds'],
        'accuracy_delta_bootstrap_95':interval(accuracy_deltas),'latency_delta_bootstrap_95':interval(latency_deltas),
        'common_answered':len(common),'rule_common_seconds':bt,'hybrid_common_seconds':ht,
        'common_speed_ratio':bt/ht if ht else None,
        'correct_to_wrong':sum(a['correct'] and not b['correct'] for a,b in common),
        'lost_answers':sum(r['status']=='answered' and index[('hybrid',r['question'])]['status']!='answered' for r in base),
        'quality_non_degradation_observed':quality,'lower_total_latency':h['total_seconds']<r['total_seconds'],
        'no_more_model_calls':h['model_calls']<=r['model_calls'],
        'no_more_output_tokens':h['output_tokens']<=r['output_tokens'] if h['output_tokens'] is not None and r['output_tokens'] is not None else None,
        'strict_dominance_observed':quality and h['total_seconds']<r['total_seconds'] and h['model_calls']<=r['model_calls'] and h['output_tokens'] is not None and r['output_tokens'] is not None and h['output_tokens']<=r['output_tokens'],
        'bootstrap_scope':'stratified question pairs within one run; not independent repeated-run uncertainty'}


def report(summaries,comparison,profile,meta):
    def num(v,n=3): return f'{v:.{n}f}' if v is not None else '미측정'
    def pct(v): return f'{v:.1%}' if v is not None else '해당 없음'
    overall={s['strategy']:s for s in summaries if s['category']=='all'}
    lines=['# 해결 방법 자동 선택: 신규300문제 실제 실험','',f"실행 ID `{meta['run_id']}`.",'',
        '## 사전에 고정한 정책','',
        '- Fixed: 모든 문제에1.5B 한 번.','- Rule: 명확한 정수 산술은 도구, 짧은 지식·지문은0.5B, 그 외는1.5B.',
        '- Confidence: 유형별0.5B 보정 정확도의95% Wilson 하한이0.8 이상이고 최소20표본·쉬운 요청일 때0.5B, 그 외1.5B.',
        '- Hybrid: 산술 도구 + Confidence의 모델 선택. 난이도는 연산·문장길이·코드구조의 휴리스틱이며 정답 라벨이 아니다.',
        '- 작은 모델의 파싱/API 실패만1.5B로 fallback. 잘못된 비산술 내용을 정답을 보고 전환하지 않는다. 큰 모델이 실패하면 원인을 기록하고 중단한다.',
        '', '기존 혼합100문제 중 모호한 소유권4개를 제외한96개와 명시적 새 보정4개로 두 모델을 실제 보정했다. '
        '평가300개는 보정 및 과거 모든 문제와 선택지 순서/정규화/교환 중복까지 제외했다. 각 유형60개이며 '
        '지식은 기초 사실 객관식, 코딩은 출력 예측, 지문·논리는 합성 템플릿이다. 같은 템플릿 계열의 일반화 한계가 있다.',
        '', '프로필·문제·출력 계약·재시도 예산은 평가 전에 고정했다. 모든 모델 호출에 같은 프롬프트 규칙·온도0·seed42·출력64토큰 상한을 사용했다. '
        '주 정확도는 유일한 최종 답 또는 응답 전체의 단일 문자/정수로 채점하며, 형식 미준수와 해석 불가능 형식 오류를 별도로 기록했다. '
        '이는 이전 엄격한 FINAL_ANSWER 점수와 다른 사전 정의 계약이므로 과거 점수와 직접 비교하지 않는다.',
        '', '## 전체 결과','',
        '| 방식 | 정확도 | 답변율 | 답변 정확도 | 평균 지연(ms) | 처리 시간(초) | 0.5B/1.5B 호출 | 도구 | 입력/출력 토큰 |',
        '|---|---:|---:|---:|---:|---:|---|---:|---|']
    for s in overall.values():
        lines.append(f"| {s['strategy']} | {pct(s['accuracy'])} | {pct(s['answer_rate'])} | {pct(s['answered_accuracy'])} | {s['mean_latency_seconds']*1000:.3f} | {s['total_seconds']:.3f} | {s['small_calls']}/{s['large_calls']} | {s['tool_calls']} | {num(s['input_tokens'],0)}/{num(s['output_tokens'],0)} |")
    lines+=['','## 유형별 정확도','','| 유형 | Fixed | Rule | Confidence | Hybrid |','|---|---:|---:|---:|---:|']
    ix={(s['strategy'],s['category']):s for s in summaries}
    for c in CATEGORIES: lines.append('| '+c+' | '+' | '.join(pct(ix[(s,c)]['accuracy']) for s in runtime.STRATEGIES)+' |')
    lines+=['','## 자원 및 실패','','| 방식 | Ollama CPU(초) | 라우터 Python CPU(초) | RSS 경계최대(MiB) | fallback 건/초 | 파싱 형식 오류 | 승인한 내용 오답 |','|---|---:|---:|---:|---|---:|---:|']
    for s in overall.values():
        lines.append(f"| {s['strategy']} | {num(s['ollama_cpu_seconds'])} | {num(s['router_cpu_seconds'])} | {num(s['rss_boundary_max_mib'],1)} | {s['fallback_requests']}/{s['fallback_seconds']:.3f} | {s['format_errors']} | {s['content_errors']} |")
    lines+=['','CPU는 실제 Ollama 프로세스군 경계 카운터와 실행 함수의Python process_time을 따로 기록했다. '
        'RSS는 상주 모델/공유 페이지를 포함할 수 있는 경계 샘플이며 전략별 전용 메모리나 피크가 아니다. '
        'GPU·전력·FLOPs·금전 비용은 미측정이며 호출/토큰을 추정 금액으로 바꾸지 않았다. '
        '실패 호출에 토큰이 없으면 전체 합계도 미측정으로 남긴다. 거절·선택·파싱·도구·fallback은 요청 시간에 포함하고 '
        '워밍업·사후 채점·ps 수집은 제외한다. 전체 루프 시간에는 모니터와 채점이 포함된다.',
        '', '## Hybrid vs Rule 판정','', '```json',json.dumps(comparison,ensure_ascii=False,indent=2),'```','',
        '품질·답변 범위를 낮추지 않으면서 시간·호출·출력 토큰도 늘리지 않는 경우에만 관측상 우위로 판정한다. '
        '정확도가 늘어도 시간/토큰이 늘면 trade-off이며 무조건 우수하다고 표현하지 않는다. '
        '신뢰구간은 한 실행의 유형별 문항쌍 bootstrap이며 독립 반복 실행의 불확실성이나 캐시/발열 영향을 포함하지 않는다.',
        '', '## 보정 프로필과 보정 비용','', '```json',json.dumps(profile,ensure_ascii=False,indent=2),'```','',
        f"보정 벽시계 시간 {meta['calibration_seconds']:.3f}초, 평가 루프 {meta['evaluation_seconds']:.3f}초. "
        '보정 호출·입출력 토큰은 calibration.csv에 별도 저장했고 요청 합계에 숨겨 포함하거나 비용 회수를 가정하지 않았다. '
        '1.5B 선택은 가장 큰 가용 모델로의 전환이며 품질 보장이 아니다. 유형별 보정표본20개의 분포가 새 문제 전체를 대표하지 않을 수 있다.',
        '', '## 무결성·누출·평가 한계','',
        '라우터는 질문·전략·고정 보정 통계만 받는다. 기대 정답/유형 라벨/참조 코드는 모델 호출에 전달하지 않는다. '
        '보정 정답은 보정 프로필에만 사용하고 평가 정답은 생성 후 채점에만 사용했다. '
        '과거 MC 문제는 선택지 순서를 제거한 질문 줄기로도 중복 검사했고, 코드60개는 수작업 기준 프로그램의 격리 출력과 선택지를 실행 전에 대조했다. '
        '생성된 모델 코드는 실행하지 않았다. 지식 라벨은 기초 사실 수작업 구성으로 외부 기관의 독립 검증을 뜻하지 않는다.',
        '', '문항별 실제 유형 추정·난이도·선택 사유·모델 호출 순서·직접 도구·내용오류·fallback을 evaluation.csv에 보존했다. '
        '클라이언트 request_id와 실제 직렬화 본문 해시/요청옵션·입출력 토큰을 기록했지만 서버 내부 캐시 hit와GPU는 측정하지 않았다. '
        '같은 문제 네 방식을 순환 실행해 위치를 균형화했으나 같은 입력의prefix cache 영향을 완전히 제거하지 못했다. '
        '영어 객관식·합성 템플릿·단일 실행이며 자유 코드 생성·장문·외부 검색·상용 서비스 부하의 성능으로 일반화할 수 없다.',
        '', '원본 소스와 과거 실험 데이터는 해시로 보존을 확인했다. 유료 API와 추가 모델 설치는 사용하지 않았다.',
        '', '## 파일','',*[f"- [{suffix}]({meta['prefix_name']}_{suffix})" for suffix in ('manifest.json','calibration.csv','profile.json','evaluation.csv','summary.csv','comparison.csv','resources.csv','reference_checks.csv','metadata.json')],'']
    return '\n'.join(lines)


def run(data_dir=ROOT/'data'):
    data_dir=Path(data_dir)
    preserved={p:hashlib.sha256(p.read_bytes()).hexdigest() for folder in (ROOT/'src',data_dir) for p in folder.iterdir() if p.is_file()}
    calibration,evaluation=build(data_dir)
    references=check_references(evaluation)
    check_ollama([runtime.SMALL,runtime.LARGE])
    with urllib.request.urlopen('http://localhost:11434/api/tags',timeout=10) as response: tags=json.load(response)
    run_id=uuid4().hex;prefix=data_dir/f'solver300_{run_id}'
    manifest={'version':runtime.VERSION,'seed':SEED,'calibration':calibration,'evaluation':evaluation,
        'calibration_policy':'96 prior mixed tasks + 4 new explicit ownership tasks; no ambiguous gold',
        'confidence_threshold':.8,'confidence_min_samples':20,'bootstrap_samples':2000,
        'preserved_sha256':{str(p):h for p,h in preserved.items()}}
    manifest_path=Path(f'{prefix}_manifest.json');write_json(manifest_path,manifest)
    manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for m in (runtime.SMALL,runtime.LARGE): runtime.generate(m,'2 + 2')
    calrows=[];start=time.perf_counter()
    for i,t in enumerate(calibration):
        for model in ((runtime.SMALL,runtime.LARGE) if i%2==0 else (runtime.LARGE,runtime.SMALL)):
            a=runtime.generate(model,t['question']);prediction,mode=content_extract(a['answer'],runtime.features(t['question'])['answer_kind'])
            calrows.append({**t,**a,'inferred_category':runtime.features(t['question'])['category'],
                           'prediction':prediction,'parse_mode':mode,'correct':prediction==t['expected']})
        if (i+1)%10==0: print(f'Calibration {i+1}/{len(calibration)}',flush=True)
    cal_seconds=time.perf_counter()-start
    profile=runtime.make_profile(calrows);write_json(Path(f'{prefix}_profile.json'),profile)
    write_records(Path(f'{prefix}_calibration.csv'),calrows)
    profile_hash=hashlib.sha256(Path(f'{prefix}_profile.json').read_bytes()).hexdigest()
    print('Frozen calibration profile: '+json.dumps(profile,ensure_ascii=False),flush=True)
    rows=[];resources=[];start=time.perf_counter()
    for i,t in enumerate(evaluation):
        offset=i%4
        for strategy in runtime.STRATEGIES[offset:]+runtime.STRATEGIES[:offset]:
            before=process_snapshot();r=runtime.execute(t['question'],strategy,profile);after=process_snapshot()
            r.update(category=t['category'],expected=t['expected'],task_format=t['task_format'],
                     correct=r['status']=='answered' and r['prediction']==t['expected'],**resource_delta(before,after))
            rows.append(r);resources.append({'question':t['question'],'strategy':strategy,'before':before,'after':after})
        if (i+1)%20==0: print(f'Actual evaluation {i+1}/300',flush=True)
    eval_seconds=time.perf_counter()-start
    summaries=summarize(rows);comparison=hybrid_comparison(rows,summaries)
    for p,h in preserved.items():
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h: raise RuntimeError(f'Original changed: {p}')
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=manifest_hash or hashlib.sha256(Path(f'{prefix}_profile.json').read_bytes()).hexdigest()!=profile_hash: raise RuntimeError('Frozen policy/data changed')
    meta={'run_id':run_id,'prefix_name':prefix.name,'calibration_seconds':cal_seconds,'evaluation_seconds':eval_seconds,
          'manifest_sha256':manifest_hash,'profile_sha256':profile_hash,'ollama_tags':tags,'preserved_files_unchanged':True,
          'real_inference':True,'cost_status':'unmeasured'}
    write_records(Path(f'{prefix}_evaluation.csv'),rows);write_records(Path(f'{prefix}_summary.csv'),summaries)
    write_records(Path(f'{prefix}_comparison.csv'),[comparison]);write_records(Path(f'{prefix}_resources.csv'),resources)
    write_records(Path(f'{prefix}_reference_checks.csv'),references)
    write_json(Path(f'{prefix}_metadata.json'),meta)
    with Path(f'{prefix}_report.md').open('x',encoding='utf-8') as f: f.write(report(summaries,comparison,profile,meta))
    for s in summaries:
        if s['category']=='all': print(json.dumps(s,ensure_ascii=False),flush=True)
    print('COMPARISON: '+json.dumps(comparison,ensure_ascii=False),flush=True)
    print(f'REPORT: {prefix}_report.md',flush=True)
    return prefix


if __name__=='__main__':
    try: run()
    except (OSError,ValueError,RuntimeError) as error: raise SystemExit(f'실험 중단: {error}')
