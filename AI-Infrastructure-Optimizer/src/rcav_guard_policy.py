"""Additive calibration-only repair-risk guard; original RCAV stays frozen."""
import ast,re,math,statistics
if __package__:
    from . import rcav_policy as base
else:
    import rcav_policy as base
VERSION='rcav-repair-guard-v1'
MIN_SAMPLES=8;REPAIR_LOWER_THRESHOLD=.05;MAX_COST_RATIO=1.5;Z=1.959963984540054

def family(question):
    category=base.features(question)['category'];low=question.lower()
    if category=='coding':
        match=re.search(r'```python\s*\n([\s\S]*?)\n```',question)
        try:tree=ast.parse(match.group(1)) if match else None
        except SyntaxError:tree=None
        if tree is not None:
            calls=[n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)]
            if 'sum' in calls:return 'coding:aggregate'
            if 'split' in calls or 'strip' in calls:return 'coding:text'
            if any(isinstance(n,ast.ListComp) for n in ast.walk(tree)):return 'coding:filter'
            if 'get' in calls:return 'coding:mapping'
            if any(isinstance(n,ast.Subscript) for n in ast.walk(tree)):return 'coding:index'
        return 'coding:other'
    if category=='logic':
        if '->' in question:return 'logic:graph'
        if 'parity' in low or 'bits=' in low:return 'logic:parity'
        if 'weekday' in low or 'weekly calendar' in low:return 'logic:calendar'
        if 'finishes before' in low or 'finishes after' in low:return 'logic:order'
        if 'set membership' in low:return 'logic:set'
    return category

def response_status(primary):return 'unresolved' if primary['prediction'] is None else 'bare' if primary['format_compliant'] else 'decorated'

def lower_rate(k,n):
    if not n:return 0.
    z2=Z*Z;p=k/n
    return (p+z2/(2*n)-Z*math.sqrt(p*(1-p)/n+z2/(4*n*n)))/(1+z2/n)

def describe(rows):
    n=len(rows);fixed=sum(not r['primary_correct'] and r['final_correct'] for r in rows)
    harmed=sum(r['primary_correct'] and not r['final_correct'] for r in rows)
    costs=[r['verification_model_seconds'] for r in rows]
    if not costs or any(v is None or v<=0 for v in costs):raise ValueError('Actual calibration costs required')
    return {'samples':n,'fixed':fixed,'harmed':harmed,'repair_rate':fixed/n,
        'repair_wilson_lower':lower_rate(fixed,n),'mean_measured_verification_seconds':statistics.mean(costs)}

def fit(rows):
    if not rows:raise ValueError('Separate calibration required')
    families={};groups={}
    for r in rows:
        f=family(r['question']);status=r['response_status']
        families.setdefault(f,[]).append(r);groups.setdefault(f+'|'+status,[]).append(r)
    return {'version':VERSION,'min_samples':MIN_SAMPLES,'repair_lower_threshold':REPAIR_LOWER_THRESHOLD,
        'max_cost_ratio':MAX_COST_RATIO,'wilson_z':Z,'families':{k:describe(v) for k,v in families.items()},
        'groups':{k:describe(v) for k,v in groups.items()},'training_rows':len(rows),
        'warning':'Wilson bound is a per-bucket exploratory repair-rate screen, not a simultaneous guarantee of net quality.'}

def decide(question,primary,prompt_chars,base_profile,guard_profile):
    if guard_profile.get('version')!=VERSION:raise ValueError('Wrong guard profile')
    original=base.decide(question,primary,prompt_chars,base_profile,.05)
    f=family(question);status=response_status(primary);s=guard_profile['groups'].get(f+'|'+status,{})
    level='family_status'
    if s.get('samples',0)<guard_profile['min_samples']:s=guard_profile['families'].get(f,{});level='family'
    enough=s.get('samples',0)>=guard_profile['min_samples']
    cost_ratio=s['mean_measured_verification_seconds']/base_profile['cost_model']['reference_seconds'] if enough else None
    proof=primary['source']=='tool' and primary['prediction'] is not None
    eligible=bool(enough and s['repair_wilson_lower']>guard_profile['repair_lower_threshold'] and s['fixed']>s['harmed'] and cost_ratio<=guard_profile['max_cost_ratio'] and not proof)
    added=not original['verify'] and eligible
    return {'verify':original['verify'] or added,'reason':'original_rcav' if original['verify'] else 'calibrated_repair_guard' if added else 'guard_not_supported',
        'base_decision':original,'family':f,'response_status':status,'calibration_level':level,
        'calibration_samples':s.get('samples',0),'calibrated_repair_lower':s.get('repair_wilson_lower'),
        'calibrated_fixed':s.get('fixed'),'calibrated_harmed':s.get('harmed'),
        'estimated_guard_cost_ratio':cost_ratio,'additional_verification':added}
