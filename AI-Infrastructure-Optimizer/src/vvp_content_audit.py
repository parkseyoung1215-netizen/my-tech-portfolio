"""Posthoc content-only audit of returned text. Never changes decisions or scores."""
import re,statistics,sys
from collections import defaultdict
from pathlib import Path
if __package__:
    from .efficiency_protocol import extract
    from .controlled_analysis import read_decoded
    from .tool_experiment import write_records
else:
    from efficiency_protocol import extract
    from controlled_analysis import read_decoded
    from tool_experiment import write_records

def interpret(question,answer,kind):
    """No reference answer. Accept explicit, nonconflicting choice assertions only."""
    old=extract(question,answer,kind)
    if old['prediction'] is not None:return old['prediction'],'original_parser'
    if kind!='choice':return None,'unresolved'
    text=answer.strip();options=dict(re.findall(r'(?m)^([A-D])\) (.*)$',question))
    dot=re.fullmatch(r'([A-D])\.',text)
    if dot:return dot.group(1),'standalone_letter_period'
    match=re.fullmatch(r'The Python program outputs the list (\[[^\]]+\])\. Therefore, the correct option is ([A-D])\.',text)
    if match:
        body,letter=match.groups()
        return (letter,'explicit_list_and_choice') if options.get(letter)==body else (None,'conflicting_list_and_choice')
    match=re.fullmatch(r'The Python program outputs option ([A-D]): (.+)\.',text)
    if match:
        letter,body=match.groups()
        return (letter,'explicit_output_choice') if options.get(letter)==body else (None,'conflicting_output_choice')
    match=re.match(r'The Python program outputs the following:\n\n([A-D])\) ([^\n]+)(?:\n|$)',text)
    if match:
        letter,body=match.groups()
        return (letter,'explicit_output_before_truncated_reasoning') if options.get(letter)==body else (None,'conflicting_output_before_reasoning')
    return None,'unresolved'

def audit(prefix):
    p=Path(prefix);rows=read_decoded(f'{p}_outcomes.csv');groups=defaultdict(list);responses={}
    for r in rows:
        kind='integer' if r['category']=='arithmetic' else 'choice'
        before,bmode=interpret(r['question'],r['primary_answer'],kind);after,amode=interpret(r['question'],r['final_answer'],kind)
        bc=before is not None and before==r['expected'];ac=after is not None and after==r['expected']
        groups[r['model'],r['repetition'],r['method'],r['budget_percent']].append((r,bc,ac))
        for stage,raw,pred,mode in [('initial',r['primary_answer'],before,bmode),('returned',r['final_answer'],after,amode)]:
            if mode!='original_parser':
                key=r['model'],r['task_id'],raw
                responses[key]={'model':r['model'],'task_id':r['task_id'],'category':r['category'],'question':r['question'],
                    'raw_response':raw,'supplemental_prediction':pred,'interpretation':mode,'expected':r['expected'],
                    'supplemental_correct':pred is not None and pred==r['expected'],
                    'scope':'Posthoc explicit-content audit; no policy, training label, original score or returned text changed.'}
    repeat=[]
    for key,rr in groups.items():
        repeat.append(dict(zip(('model','repetition','method','budget_percent'),key),
            original_accuracy=statistics.mean(r['final_correct'] for r,b,a in rr),
            supplemental_initial_accuracy=statistics.mean(b for r,b,a in rr),supplemental_final_accuracy=statistics.mean(a for r,b,a in rr),
            supplemental_repairs=sum(not b and a for r,b,a in rr),supplemental_damage=sum(b and not a for r,b,a in rr),
            original_repairs=sum(r['wrong_to_right'] for r,b,a in rr),original_damage=sum(r['right_to_wrong'] for r,b,a in rr)))
    aggregate=[];grouped=defaultdict(list)
    for r in repeat:grouped[r['model'],r['method'],r['budget_percent']].append(r)
    for key,rr in grouped.items():
        out=dict(zip(('model','method','budget_percent'),key));out['repeats']=len(rr)
        for name in ('original_accuracy','supplemental_initial_accuracy','supplemental_final_accuracy','supplemental_repairs','supplemental_damage','original_repairs','original_damage'):
            out[name+'_mean']=statistics.mean(r[name] for r in rr)
        aggregate.append(out)
    write_records(Path(f'{p}_content_audit_responses.csv'),list(responses.values()))
    write_records(Path(f'{p}_content_audit_repeats.csv'),repeat)
    write_records(Path(f'{p}_content_audit_aggregate.csv'),aggregate)
    print('Posthoc supplementary content audit; original scores unchanged. Unique inspected responses:',len(responses))
    for r in aggregate:
        if (r['method'],r['budget_percent']) in [('none',0),('selective',100),('vvp',25)]:print(r)

def report(prefix):
    if __package__:
        from .vvp_report import table,percent
    else:
        from vvp_report import table,percent
    p=Path(prefix);rows=read_decoded(f'{p}_content_audit_aggregate.csv')
    initial={r['model']:r['initial_correct_mean']/r['requests_mean'] for r in read_decoded(f'{p}_aggregate.csv')}
    selected=[r for r in rows if (r['method'],r['budget_percent']) in [('none',0),('full',100),('selective',100),('rcav',100),('improved',100),('vvp',25)]]
    text='# VVP 추출 실패 응답의 별도 내용 감사\n\n'
    text+='기존 점수·학습 라벨·선택·성공 판정은 변경하지 않았다. 기존 파서는 응답 전체가 인정 형식에 맞아야 추출하므로 명확한 설명문도 실패할 수 있다. 실제 반환 문장의 모든 미추출 응답 25개를 따로 검사했다. 이는 평가 후 발견한 표현에 대한 **사후 보조 분석**이며 사전 등록 독립 검증 결과가 아니다.\n\n'
    text+='추가 규칙은 정답을 입력받지 않고 단독 `A.` 형태, 선택지 본문과 일치하는 명시적 출력 주장, 목록과 선택지가 서로 일치하는 명시적 결론만 인정한다. 추론 중단이나 상충 주장은 계속 미해결로 둔다. 실제 코드 실행의 fallback 결과를 바꾸지 않고 최종 반환된 원문만 다시 해석했다. 이 제한된 감사가 모든 자연어 의미 오류를 해소한다고 주장하지 않는다.\n\n'
    text+=table(['모델','방식/예산','기존 최초 정확도','보조 최초 정확도','기존 최종 정확도','보조 최종 정확도','기존 복구/훼손','보조 복구/훼손'],[[r['model'],r['method']+'/'+str(r['budget_percent'])+'%',percent(initial[r['model']]),percent(r['supplemental_initial_accuracy_mean']),percent(r['original_accuracy_mean']),percent(r['supplemental_final_accuracy_mean']),f'{r["original_repairs_mean"]:g}/{r["original_damage_mean"]:g}',f'{r["supplemental_repairs_mean"]:g}/{r["supplemental_damage_mean"]:g}'] for r in selected])
    text+='\n\n0.5B의 기존 최초 정답은 149개였으나 명확한 내용 기준으로 151개다. VVP25 최종은 149개로, 내용 기준에서는 **2개 순손실(−0.67pp)**이다. 기존 복구 3건 중 1건은 이미 올바른 목록과 선택지 C를 제시해 내용상 정답이었다. 또 다른 최초 정답 `A.`가 검증 후 오답으로 바뀌어 기존 채점의 훼손 수에서 빠졌다. 따라서 보조 지표에서는 복구 2건·훼손 4건이다. 이 발견은 VVP 성공을 뒷받침하지 않고 최초의 부정적 결론을 강화한다.\n\n'
    text+=f'전체 예산/반복 수치는 [{p.name}_content_audit_aggregate.csv]({p.name}_content_audit_aggregate.csv), [{p.name}_content_audit_repeats.csv]({p.name}_content_audit_repeats.csv), 응답별 근거는 [{p.name}_content_audit_responses.csv]({p.name}_content_audit_responses.csv)에 저장했다. 원래의 모든 CSV는 그대로 보존했다.\n'
    path=Path(f'{p}_content_audit_report.md')
    with path.open('x',encoding='utf-8') as f:f.write(text)
    print(path)

if __name__=='__main__':
    audit(sys.argv[1]);report(sys.argv[1])
