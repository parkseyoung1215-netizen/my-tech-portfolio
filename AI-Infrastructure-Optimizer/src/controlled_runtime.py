"""Verification-only execution. Primary answers are immutable inputs, never generated."""
import json,time,threading,subprocess,urllib.request
if __package__:
    from .rcav_runtime import verifier_prompt
    from . import rcav_policy as base
    from . import rcav_guard_policy as guard
    from .skipwork_runtime import preflight
else:
    from rcav_runtime import verifier_prompt
    import rcav_policy as base
    import rcav_guard_policy as guard
    from skipwork_runtime import preflight
METHODS=('baseline','selective','rcav','improved')

class MemorySampler:
    """Observed process-group RSS maximum; not a continuous true peak."""
    def __init__(self,snapshot,interval=.1):
        self.ids=list(snapshot['processes']);self.interval=interval;self.samples=[];self.errors=[];self.stop_event=threading.Event();self.thread=None
        if self.ids:self.samples.append([snapshot['time'],sum(r['rss_kib'] for r in snapshot['processes'].values())/1024])
    def start(self):
        def poll():
            while not self.stop_event.wait(self.interval):
                try:
                    r=subprocess.run(['ps','-o','rss=','-p',','.join(self.ids)],capture_output=True,text=True,timeout=3,check=True)
                    values=[int(v) for v in r.stdout.split()]
                    if len(values)!=len(self.ids):raise ValueError('Process set changed during RSS sample')
                    self.samples.append([time.perf_counter(),sum(values)/1024])
                except Exception as exc:self.errors.append(str(exc))
        if self.ids:self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
    def stop(self,snapshot):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=4)
        if snapshot['processes']:self.samples.append([snapshot['time'],sum(r['rss_kib'] for r in snapshot['processes'].values())/1024])
        return {'sampled_max_rss_mib':max(v[1] for v in self.samples) if self.samples else None,
            'rss_samples':self.samples,'rss_sample_count':len(self.samples),'rss_sampling_interval_seconds':self.interval,'rss_sampling_errors':self.errors}

def http_json(path,payload=None):
    req=urllib.request.Request('http://localhost:11434'+path,data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)

def reset_and_warm(client,experiment_models):
    start=time.perf_counter();record={'target_model':client.model,'events':[],'reset_observed':False,'error':''}
    try:
        record['before']=http_json('/api/ps')
        for model in experiment_models:
            payload={'model':model,'keep_alive':0};record['events'].append({'request':payload,'response':http_json('/api/generate',payload)})
        record['after_unload']=http_json('/api/ps')
        names={m.get('name',m.get('model')) for m in record['after_unload']['models']}
        record['reset_observed']=not any(model in names for model in experiment_models)
    except Exception as exc:record['error']=str(exc)
    warmup=client.send('Return only A.','controlled_warmup')
    try:record['after_warmup']=http_json('/api/ps')
    except Exception as exc:record['error']+='; '+str(exc)
    record['elapsed_seconds']=time.perf_counter()-start
    return record,warmup

def choose(question,primary,method,profiles):
    chars=len(verifier_prompt(question,primary['answer']))
    if method=='baseline':return {'verify':True,'reason':'exhaustive_verification'}
    if method=='selective':
        d=preflight(question,'selective',profiles['selective']);return {'verify':not d['skip_verifier'],'reason':d['reason']}
    if method=='rcav':return base.decide(question,primary,chars,profiles['rcav'],.05)
    if method=='improved':return guard.decide(question,primary,chars,profiles['rcav'],profiles['guard'])
    raise ValueError('Unknown method')

def execute_block(frozen_primaries,method,client,profiles):
    start,cpu=time.perf_counter(),time.process_time();rows=[];calls=[]
    for frozen in frozen_primaries:
        request_start=time.perf_counter();p=frozen;q=p['question'];d=choose(q,p,method,profiles)
        row={'task_id':p['task_id'],'question':q,'primary_cache_hash':p['cache_entry_sha256'],
            'method':method,'model':client.model,'decision':d,'primary_answer':p['answer'],'primary_prediction':p['prediction'],
            'primary_format_compliant':p['format_compliant'],'primary_source':p['source'],
            'final_answer':p['answer'],'final_prediction':p['prediction'],'final_format_compliant':p['format_compliant'],
            'verification_executed':False,'verification_answer':'','verification_prediction':None,
            'verification_format_compliant':None,'call_ids':[]}
        if d['verify']:
            a,v=client.verify(q,p['answer']);calls.append(a);row['call_ids'].append(a['call_id'])
            row.update(verification_executed=True,verification_answer=a['answer'],verification_prediction=v['prediction'],verification_format_compliant=v['format_compliant'])
            if v['prediction'] is not None:row.update(final_answer=a['answer'],final_prediction=v['prediction'],final_format_compliant=v['format_compliant'])
        row.update(verification_service_seconds=time.perf_counter()-request_start,completion_latency_seconds=time.perf_counter()-start)
        rows.append(row)
    return {'rows':rows,'calls':calls,'verification_wall_seconds':time.perf_counter()-start,'router_cpu_seconds':time.process_time()-cpu}
