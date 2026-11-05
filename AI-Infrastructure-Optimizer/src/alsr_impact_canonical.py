"""Real randomized rerun after a canonical-greedy tie bug fix; original data immutable."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import json,shutil,sys,time
from pathlib import Path
from uuid import uuid4
import mlx.core as mx
import numpy as np
from mlx_lm import load
from .alsr_impact_engine import ImpactEngine
from .alsr_impact_run import Journal,sha,rss,OLD,SETUP,unload_ollama
from .alsr_impact_ties import correct,fit_profile
from .alsr_impact_policy import METHODS
from .alsr_impact_data import SEED
from .alsr_dataset import grade
from .alsr_io import write_union
from .adaptive_experiment import write_json
from .efficiency_repeat import frozen_check

def run(source):
    startjob=time.perf_counter();source=Path(source).resolve();m=json.loads((source/'MANIFEST.json').read_text());originalmeta=json.loads((source/'METADATA.json').read_text());setup=json.loads(SETUP.read_text())
    out=source/('CANONICAL_'+uuid4().hex);out.mkdir();read=lambda name:[json.loads(s) for s in (source/(name+'.jsonl')).read_text().splitlines()]
    source_hashes={str(p):sha(p) for p in source.iterdir() if p.is_file()};ab=correct(read('LAYER_ABLATIONS'),read('DIAGNOSTIC_CONTEXTS'));ctx=read('DIAGNOSTIC_CONTEXTS');shortlist=json.loads((source/'SHORTLIST.json').read_text())['ids']
    start,cpu=time.perf_counter(),time.process_time();profile=fit_profile(ab,shortlist);fitcost={'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'scope':'Corrected training/calibration only; same threshold grid and policy, no evaluation adjustment.'}
    m=dict(m,diagnostic_source=str(source),diagnostic_source_sha256=source_hashes,correction='Canonical min-index greedy tie rule, no threshold/policy/hyperparameter change. Original test is reused as bug-audit evaluation, not a new untouched holdout.')
    write_json(out/'MANIFEST.json',m);write_json(out/'PROFILE.json',profile);profilehash=sha(out/'PROFILE.json');write_json(out/'FIT_COST.json',fitcost)
    print('CANONICAL OUTPUT '+str(out),flush=True);print('CANONICAL candidate/threshold '+str(profile['candidate'])+'/'+str(profile['threshold']),flush=True)
    write_json(out/'OLLAMA_UNLOAD.json',unload_ollama());mx.set_memory_limit(int(setup['mlx_device']['max_recommended_working_set_size']*.8));mx.set_cache_limit(128*(1<<20));mx.clear_cache()
    start,cpu=time.perf_counter(),time.process_time();model,tok,config=load(m['model']['path'],return_config=True);model.eval();engine=ImpactEngine(model,tok,config);engine.prepare_profile(json.loads(OLD.read_text()));engine.prepare_local(profile)
    write_json(out/'LOAD.json',{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'dtype':str(model.model.embed_tokens.weight.dtype),'mlx_active_bytes':mx.get_active_memory(),'device':mx.device_info(),'rss_bytes':rss()});write_json(out/'SMOKE.json',{**engine.check_native(),**engine.check_skip_trap()})
    tasks={t['task_id']:t for ts in m['splits'].values() for t in ts};ci={(r['task_id'],r['position']):r for r in ctx};rechecks=[]
    # Ten corrected-reference margins outside saved top10 require actual targeted forwards.
    for r in ab:
        if r['full_top1_skip_margin'] is not None:continue
        c=ci[r['task_id'],r['position']];ids=engine.prompt_ids(tasks[r['task_id']]['question'])+c['prefix_tokens'];engine.reset();mx.synchronize();start,cpu=time.perf_counter(),time.process_time();logits,_=engine.forward(mx.array([ids]),skip=[r['layer']]);a=np.array(logits[0,-1].astype(mx.float32));mx.synchronize()
        import hashlib
        assert hashlib.sha256(a.tobytes()).hexdigest()==r['skip_distribution']['sha256_f32'],'Recheck logits differ: preserve and investigate, never silently replace original distributions'
        r['full_top1_skip_margin']=float(a[r['full_top1']]-np.max(np.delete(a,r['full_top1'])));r['margin_status']='actual_targeted_recheck_hash_equal'
        rechecks.append(dict(task_id=r['task_id'],position=r['position'],layer=r['layer'],wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,logits_sha256_equal=True,corrected_margin=r['full_top1_skip_margin']))
    write_union(out/'TIE_MARGIN_RECHECK.csv',rechecks)
    with (out/'LAYER_ABLATIONS.jsonl').open('x') as f:
        for r in ab:f.write(json.dumps(r)+'\n')
    write_union(out/'LAYER_ABLATIONS.csv',ab)
    for name in ('DIAGNOSTIC_CONTEXTS.jsonl','DIAGNOSTIC_CONTEXTS.csv','SHORTLIST.json','FIXED_SEQUENCES.json','PREPARATION.jsonl','PREPARATION.csv','OFFLINE_COSTS.csv','QUALITY.jsonl','QUALITY.csv'):
        with (out/name).open('xb') as target:target.write((source/name).read_bytes())
    write_json(out/'TIE_AUDIT.json',{'original_rows':len(ab),'changed_flip_labels':sum(r['top1_changed']!=r['legacy_top1_changed'] for r in ab),'changed_full_top1':sum(r['full_top1']!=r['legacy_full_top1'] for r in ab),'top10_tie_groups_complete':True,'targeted_margin_rechecks':len(rechecks),'original_kl_js_and_top2_margins_preserved':True,'raw_artifacts_preserved':True,'retrained_only_training_and_calibration':True,'evaluation_not_new_after_bug_audit':True})
    fixedseq=json.loads((out/'FIXED_SEQUENCES.json').read_text());warm=[]
    for method in METHODS:warm.append(dict(engine.generate('Return only A.',method,SEED,max_tokens=4),method=method))
    write_union(out/'WARMUPS.csv',warm);gen=Journal(out/'GENERATION.jsonl');fixed=Journal(out/'FIXED_WORK.jsonl')
    for s in m['schedule']:
        task=tasks[s['task_id']];method=s['method'];before=rss();r=engine.generate(task['question'],method,SEED+s['repetition'],max_tokens=16 if task['score_kind']=='choice' else 48)
        gen.add(dict(r,**s,question=task['question'],reference=task['reference'],category=task['category'],score_kind=task['score_kind'],grade=grade(task,r['answer']),rss_before_bytes=before,rss_after_bytes=rss()))
        if s['task_id'] in fixedseq:
            f=engine.generate(task['question'],method,SEED+s['repetition'],max_tokens=16,fixed_tokens=fixedseq[s['task_id']]);assert f['skip_layers']==r['skip_layers'];fixed.add(dict(f,**s))
        if s['position']==len(METHODS)-1 and (s['sequence']+1)%8==0:print(f'CANONICAL repeat{s["repetition"]}/3 question{s["sequence"]+1}/64',flush=True)
    gen.close(out/'GENERATION.csv');fixed.close(out/'FIXED_WORK.csv');frozen_check(m)
    for p,h in source_hashes.items():assert sha(p)==h
    assert sha(out/'PROFILE.json')==profilehash
    write_json(out/'METADATA.json',{'real_inference':True,'all_stages_wall_seconds':originalmeta['all_stages_wall_seconds']+time.perf_counter()-startjob,'original_stage_wall_seconds':originalmeta['all_stages_wall_seconds'],'canonical_rerun_wall_seconds':time.perf_counter()-startjob,'profile_sha256':profilehash,'preserved_files':len(m['preserved_sha256']),'generation_rows':len(gen.rows),'ablation_rows':len(ab),'fixed_rows':len(fixed.rows),'quality_rows':len(read('QUALITY')),'GPU_elapsed_power_money':'unmeasured','corrected_tie_rule':True,'diagnostics_and_reference_quality_source':str(source),'reference_quality':'Reused actual original NLL once, masks unchanged; analysis validates masks, not falsely reported as repeated measurement.','cost_scope':'Original + canonical model-stage wall sum, not continuous job wall; intermediate audit/tests/report gaps excluded.'})
    print('CANONICAL COMPLETE '+str(out),flush=True)

if __name__=='__main__':run(sys.argv[1])
