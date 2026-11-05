"""Explicit new real run of immutable saved ACS inputs; never claims a new holdout."""
import json,sys
from pathlib import Path
from uuid import uuid4
from . import acs_experiment as exp
from .adaptive_experiment import write_json

def replay(source):
    source=Path(source).resolve();m=json.loads(source.read_text())
    for f,h in m['runtime_source_sha256'].items():assert exp.sha(f)==h,'Use the archived source/environment to reproduce this version'
    preserved={str(p.resolve()):exp.sha(p) for folder in (exp.ROOT/'src',exp.ROOT/'tests',exp.ROOT/'data') for p in folder.rglob('*') if p.is_file()};preserved.update({str(p.resolve()):exp.sha(p) for p in exp.ROOT.iterdir() if p.is_file()})
    out=exp.ROOT/'data'/('acs_'+uuid4().hex);out.mkdir();new=dict(m,preserved_sha256=preserved,freshness_reference_preserved_sha256=m.get('freshness_reference_preserved_sha256',m['preserved_sha256']),explicit_replay=True,replay_source_manifest=str(source),replay_source_sha256=exp.sha(source));write_json(out/'MANIFEST.json',new);write_json(Path(str(out)+'_manifest.json'),{'datasets':m['datasets'],'explicit_replay':True,'manifest':str(out/'MANIFEST.json')});exp.create=lambda:out;exp.run()

if __name__=='__main__':replay(sys.argv[1])
