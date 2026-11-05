"""Explicit public-copy hash adapter for historical module entry points.

This checks approved public bytes, never rewrites recorded fingerprints and never
pretends a transformed copy is byte-identical to the private execution environment.
It does not provision models, repair missing old versions, or waive arbitrary hashes.
"""
import hashlib, json, runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    if len(sys.argv) < 2 or not sys.argv[1].startswith("src."):
        raise SystemExit("Usage: python3 -B tools/run_recorded.py src.module [arguments]")
    if Path.cwd().resolve() != ROOT: raise SystemExit("Run from the project directory.")
    sys.path.insert(0, str(ROOT))
    from src import efficiency_repeat
    lineage = json.loads((ROOT / "publication/LINEAGE.json").read_text())
    by_path = {item["path"]: item for item in lineage["files"]}
    def public_check(manifest):
        transformed = 0
        for recorded, expected in manifest["preserved_sha256"].items():
            path = Path(recorded)
            if not path.is_absolute(): path = ROOT / path
            path = path.resolve()
            relative = path.relative_to(ROOT).as_posix()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual == expected: continue
            item = by_path.get(relative)
            if item and item["private_source_sha256"] == expected and item["public_sha256"] == actual:
                transformed += 1
                continue
            raise RuntimeError(f"Historical hash mismatch not covered by declared public transform: {relative}")
        if transformed: print(f"PUBLIC-COPY MODE: accepted {transformed} declared transforms; not private-byte identity.", flush=True)
    efficiency_repeat.frozen_check = public_check
    module, args = sys.argv[1], sys.argv[2:]
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__")
if __name__ == "__main__": main()
