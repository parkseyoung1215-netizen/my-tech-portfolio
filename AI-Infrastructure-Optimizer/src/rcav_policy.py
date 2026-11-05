"""Exploratory empirical expected-error reduction versus measured-time prediction."""
import statistics
if __package__:
    from .solver300_runtime import features
else:
    from solver300_runtime import features
VERSION='rcav-v1'
WEIGHTS={'rcav_zero':0.0,'rcav':0.05,'rcav_high':0.15}
METHODS=('baseline','selective','rcav_zero','rcav','rcav_high')
PRIOR=10

def observable(question,primary):
    f=features(question)
    status='unresolved' if primary['prediction'] is None else 'bare' if primary['format_compliant'] else 'decorated'
    return {**f,'response_status':status,'question_chars':len(question),'candidate_chars':len(primary['answer'])}

def group_key(f):return f"{f['category']}|{f['difficulty']}|{f['response_status']}"

def stats(rows):
    wrong=sum(not r['primary_correct'] for r in rows);correct=len(rows)-wrong
    fixed=sum(not r['primary_correct'] and r['final_correct'] for r in rows)
    harmed=sum(r['primary_correct'] and not r['final_correct'] for r in rows)
    return {'samples':len(rows),'wrong':wrong,'correct':correct,'fixed':fixed,'harmed':harmed}

def fit(rows):
    if not rows:raise ValueError('Calibration required')
    if any(r['verification_model_seconds'] is None or r['verification_model_seconds']<=0 for r in rows):raise ValueError('Measured calibration time required')
    global_stats=stats(rows);categories={};groups={}
    for c in {r['observed']['category'] for r in rows}:categories[c]=stats([r for r in rows if r['observed']['category']==c])
    for k in {group_key(r['observed']) for r in rows}:groups[k]=stats([r for r in rows if group_key(r['observed'])==k])
    xs=[r['verification_prompt_chars'] for r in rows];ys=[r['verification_model_seconds'] for r in rows]
    mx,my=statistics.mean(xs),statistics.mean(ys);den=sum((x-mx)**2 for x in xs)
    slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den if den else 0.
    # Nonnegative length slope is a prespecified physical prior; no eval fitting.
    slope=max(0.,slope);intercept=my-slope*mx
    return {'version':VERSION,'prior_strength':PRIOR,'min_group_samples':8,'min_category_samples':10,
        'weights':WEIGHTS,'global':global_stats,'categories':categories,'groups':groups,
        'cost_model':{'target':'measured verifier prefill+generation seconds','intercept':intercept,'slope_per_char':slope,
            'floor_seconds':min(ys),'reference_seconds':statistics.median(ys),'train_min_chars':min(xs),'train_max_chars':max(xs),
            'train_mean_seconds':my,'samples':len(ys)}}

def decide(question,primary,verification_prompt_chars,profile,weight=0.05):
    if profile.get('version')!=VERSION:raise ValueError('RCAV profile mismatch')
    if weight<0:raise ValueError('Negative risk-cost weight')
    f=observable(question,primary);g=profile['global'];s=profile['groups'].get(group_key(f),{})
    level='group'
    if s.get('samples',0)<profile['min_group_samples']:
        s=profile['categories'].get(f['category'],{});level='category'
    if s.get('samples',0)<profile['min_category_samples'] and level=='category':s=g;level='global'
    # Beta(1,1) global rates, then empirical shrinkage within observable strata.
    pe=(g['wrong']+1)/(g['samples']+2);rescue=(g['fixed']+1)/(g['wrong']+2);harm=(g['harmed']+1)/(g['correct']+2)
    strength=profile['prior_strength']
    risk=(s['wrong']+strength*pe)/(s['samples']+strength)
    rescue=(s['fixed']+strength*rescue)/(s['wrong']+strength)
    harm=(s['harmed']+strength*harm)/(s['correct']+strength)
    tool_proof=primary['source']=='tool' and primary['prediction'] is not None
    if tool_proof:risk=0.
    elif primary['prediction'] is None:risk=1.
    reduction=risk*rescue-(1-risk)*harm
    cost=profile['cost_model'];seconds=max(cost['floor_seconds'],cost['intercept']+cost['slope_per_char']*verification_prompt_chars)
    threshold=weight*seconds/cost['reference_seconds']
    execute=not tool_proof and reduction>threshold
    return {'observed':f,'risk_estimate':risk,'rescue_probability_estimate':rescue,'harm_probability_estimate':harm,
        'expected_error_reduction':reduction,'estimated_verification_model_seconds':seconds,'normalized_cost':seconds/cost['reference_seconds'],
        'weight':weight,'required_error_reduction':threshold,'verify':execute,'calibration_level':level,'calibration_samples':s['samples'],
        'cost_extrapolation':not cost['train_min_chars']<=verification_prompt_chars<=cost['train_max_chars'],
        'reason':'safe_tool_proof' if tool_proof else 'expected_reduction_exceeds_cost' if execute else 'estimated_benefit_not_worth_cost'}
