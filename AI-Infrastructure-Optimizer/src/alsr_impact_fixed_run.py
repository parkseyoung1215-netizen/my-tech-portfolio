"""Future clean full reruns: fix greedy ties before any causal labels are collected."""
import json,sys
from pathlib import Path
from uuid import uuid4
from . import alsr_impact_engine as engine
from .alsr_impact_ties import distribution_effect
from . import alsr_impact_run as experiment
from .adaptive_experiment import write_json

def run(manifest_path):
    source=Path(manifest_path).resolve();m=json.loads(source.read_text());runid=uuid4()
    assert m['seed']==experiment.SEED
    assert m['model']==json.loads(experiment.SETUP.read_text())['hf_models'][1]
    engine.distribution_effect=distribution_effect
    experiment.build=lambda _:m['splits']
    experiment.uuid4=lambda:runid
    experiment.run()
    write_json(experiment.ROOT/'data'/('alsr_impact_'+runid.hex)/'CANONICAL_TIE_RULE.json',{'source_manifest':str(source),'source_sha256':experiment.sha(source),'rule':'Minimum token index among exact max ties, matching actual greedy argmax','real_models_recomputed':True,'all_diagnostic_and_quality_calls_recomputed':True})

if __name__=='__main__':run(sys.argv[1])
