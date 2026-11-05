"""Preserve reported durations; invalidate inconsistent model-time aggregates."""
from pathlib import Path
import sys
if __package__:
    from .controlled_analysis import read_decoded
    from .tool_experiment import write_records
else:
    from controlled_analysis import read_decoded
    from tool_experiment import write_records

def duration_problem(call):
    pre,gen,total,elapsed=[call.get(k) for k in ('prefill_seconds','eval_seconds','total_model_seconds','elapsed_seconds')]
    if any(v is None for v in (pre,gen,total,elapsed)):return 'duration_unavailable'
    if min(pre,gen,total,elapsed)<0:return 'negative_duration'
    # Arithmetic sanity check only. Do not repair units, cap values or estimate a replacement.
    if pre+gen>total+.001:return 'reported_components_exceed_reported_total'
    if total>elapsed+.001:return 'reported_total_exceeds_client_elapsed'
    return ''

def validate(prefix):
    prefix=Path(prefix).resolve();anomalies=[];invalid=set();counts={}
    for stage in ('calls','primary_calls','calibration_calls','warmup_calls'):
        rows=read_decoded(f'{prefix}_{stage}.csv');counts[stage]=0
        for a in rows:
            error=duration_problem(a)
            if error:
                counts[stage]+=1
                anomalies.append({'phase':stage,'reason':error,**a})
                if stage=='calls':invalid.add((a['model'],a['dataset'],a['method']))
    original=read_decoded(f'{prefix}_aggregate.csv');summary=[]
    for row in original:
        key=row['model'],row['dataset'],row['method'];bad=key in invalid
        summary.append({**row,'model_duration_integrity':'untrusted_inconsistent_server_fields' if bad else 'no_arithmetic_inconsistency_detected',
            'validated_model_processing_seconds_mean':None if bad else row['model_processing_seconds_mean'],
            'wall_cpu_token_metrics':'Unchanged independent measurements; sampler overhead and clock/counter limitations still apply.',
            'raw_duration_columns_warning':'model_* duration columns retain uncorrected server reports; use validated_model_processing_seconds_mean.'})
    # Different call phases have different fields: union schema without losing evidence.
    fields=list(dict.fromkeys(k for a in anomalies for k in a)) or ['phase','reason','call_id']
    filled=[{k:a.get(k) for k in fields} for a in anomalies]
    write_records(Path(f'{prefix}_duration_anomalies.csv'),filled,fields=fields)
    write_records(Path(f'{prefix}_reporting_summary.csv'),summary)
    print({'duration_anomalies_by_phase':counts,'affected_verification_cells':len(invalid)})
    return counts

if __name__=='__main__':validate(sys.argv[1])
