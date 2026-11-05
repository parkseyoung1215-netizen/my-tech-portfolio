"""Preserve all original performance traces; recheck ONLY lost auxiliary evidence.

Never regenerate pilot outputs or replace original calibration profiles.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import gc,hashlib,json,sys,time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from .alsr_engine import Engine,calibrate
from .alsr_io import write_union
from .alsr_experiment import sha
from .controlled_analysis import read_decoded
from .efficiency_repeat import frozen_check
from .adaptive_experiment import write_json

def recover(prefix):
    p=Path(prefix).resolve();manifest=json.loads(Path(f'{p}_manifest.json').read_text());setup=json.loads(Path(manifest['setup_path']).read_text())
    original_hashes={str(f):sha(f) for f in p.parent.glob(p.name+'*') if f.is_file()};frozen_check(manifest)
    rows=read_decoded(f'{p}_generation.csv');fixed=read_decoded(f'{p}_fixed_work.csv');quality=read_decoded(f'{p}_quality.csv')
    assert (len(rows),len(fixed),len(quality))==(1536,384,512)
    mx.set_memory_limit(int(min(setup['mlx_device']['max_recommended_working_set_size']*.8,9*(1<<30))));mx.set_cache_limit(128*(1<<20))
    profiles={};records=[];costs=[];loads=[];smokes=[];checks=[];start_all=time.perf_counter()
    for mi,info in enumerate(setup['hf_models']):
        start,cpu=time.perf_counter(),time.process_time();model,tok,config=load(info['path'],return_config=True);model.eval();engine=Engine(model,tok,config)
        loads.append({'model':info['repo'],'load_wall_seconds':time.perf_counter()-start,'load_cpu_seconds':time.process_time()-cpu,'active_bytes':mx.get_active_memory(),'dtype':str(model.model.embed_tokens.weight.dtype),'config':config,'phase':'postrun_auxiliary_recheck'})
        smokes.append({'model':info['repo'],**engine.check_native(),**engine.check_skip_trap(),'phase':'postrun_auxiliary_recheck'})
        original=json.loads(Path(f'{p}_m{mi}_profile.json').read_text());recomputed,trace,cost=calibrate(engine,manifest['splits']['training'],manifest['splits']['calibration'])
        same=recomputed==original;checks.append({'model':info['repo'],'recomputed_profile_exactly_matches_original':same})
        if not same:raise RuntimeError('Recheck differs from frozen original profile; do not substitute it')
        profiles[info['repo']]=original;records.extend(dict(r,model=info['repo']) for r in trace);costs.append(dict(cost,model=info['repo'],phase='postrun_auxiliary_recheck_only'))
        print('ALSR auxiliary recheck '+info['repo']+' original profile reproduced exactly',flush=True)
        del engine,model,tok;gc.collect();mx.clear_cache();mx.synchronize()
    for name,h in original_hashes.items():assert sha(name)==h
    for name,rr in [('calibration_recheck',records),('calibration_cost_recheck',costs),('loads_recheck',loads),('native_parity_recheck',smokes),('profile_recheck_status',checks)]:write_union(Path(f'{p}_{name}.csv'),rr)
    write_json(Path(f'{p}_profiles.json'),profiles)
    metadata={'prefix':p.name,'models':[i['repo'] for i in setup['hf_models']],'repeats':3,'pilot_questions':32,'generation_rows':len(rows),'fixed_work_rows':len(fixed),'quality_rows':len(quality),
      'real_inference':True,'recovered_after_export_error':True,'total_trial_wall_seconds':None,'total_trial_wall_status':'Unmeasured: process stopped before end metadata export; never inferred from timestamps.',
      'original_auxiliary_cost_status':'Original calibration/load/warmup/preparation costs not recovered; recheck costs separately measured, not substituted.',
      'postrun_recheck_wall_seconds':time.perf_counter()-start_all,'memory_limit_bytes':int(min(setup['mlx_device']['max_recommended_working_set_size']*.8,9*(1<<30))),
      'preserved_files':len(manifest['preserved_sha256']),'profile_hashes':{str(mi):sha(Path(f'{p}_m{mi}_profile.json')) for mi in range(2)},
      'original_trace_sha256':original_hashes,'calibration_trace_suffix':'calibration_recheck','native_parity_suffix':'native_parity_recheck','loads_suffix':'loads_recheck','calibration_cost_suffix':'calibration_cost_recheck',
      'serialization_error':'Legacy CSV writer chose first training-row keys; validation-row fields caused final calibration export failure. All per-repeat and combined generation/fixed/quality traces preserved.',
      'GPU_elapsed_power_monetary_cost':'unmeasured'}
    write_json(Path(f'{p}_metadata.json'),metadata);frozen_check(manifest);print('ALSR recovered original performance, no pilot rerun: '+str(p),flush=True)

if __name__=='__main__':recover(sys.argv[1])
