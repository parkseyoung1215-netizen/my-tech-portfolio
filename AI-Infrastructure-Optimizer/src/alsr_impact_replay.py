"""Explicitly replay immutable splits; this does not bypass model inference or grading."""
import json,sys
from pathlib import Path
from uuid import uuid4
from . import alsr_impact_run as experiment
from .adaptive_experiment import write_json

def replay(source):
    source=Path(source).resolve();manifest=json.loads(source.read_text());runid=uuid4()
    assert manifest['seed']==experiment.SEED
    assert manifest['model']==json.loads(experiment.SETUP.read_text())['hf_models'][1]
    assert experiment.sha(experiment.OLD)==manifest['old_profile_sha256']
    experiment.build=lambda _:manifest['splits']
    experiment.uuid4=lambda:runid
    experiment.run()
    write_json(experiment.ROOT/'data'/('alsr_impact_'+runid.hex)/'REPLAY_SOURCE.json',{'manifest':str(source),'sha256':experiment.sha(source),'scope':'Explicit replay of immutable train/cal/evaluation; real model recomputation, no threshold tuning on test.'})

if __name__=='__main__':replay(sys.argv[1])
