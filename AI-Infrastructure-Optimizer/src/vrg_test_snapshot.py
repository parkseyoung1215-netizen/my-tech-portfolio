"""Run all tests against the recorded pre-study data snapshot, without deleting data.

Fresh-dataset tests deliberately reject a dataset once recorded in the real data
directory. Isolate their historical inputs instead of disabling that protection.
All original data are read-only symlink inputs; mocked outputs stay in temp dirs.
"""
import json,shutil,subprocess,sys,tempfile
from pathlib import Path

def run(manifest_path):
    path=Path(manifest_path).resolve();root=path.parent.parent;manifest=json.loads(path.read_text())
    with tempfile.TemporaryDirectory(prefix='vrg_test_snapshot_') as folder:
        target=Path(folder)
        for name in ('src','tests'):
            shutil.copytree(root/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (target/'data').mkdir()
        for original in manifest['preserved_sha256']:
            source=Path(original)
            if source.parent==root/'data':(target/'data'/source.name).symlink_to(source)
            elif source.parent==root:(target/source.name).symlink_to(source)
        print('Full suite, isolated pre-study dataset inputs; production data untouched.',flush=True)
        result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests'],cwd=target)
        return result.returncode

if __name__=='__main__':raise SystemExit(run(sys.argv[1]))
