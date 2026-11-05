"""Offline full-suite runner using the historical pre-study fixture (no models)."""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    manifest = json.loads((ROOT / "data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json").read_text())
    with tempfile.TemporaryDirectory(prefix="public_research_tests_") as folder:
        target = Path(folder)
        for name in ("src", "tests"):
            shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (target / "data").mkdir()
        count = 0
        for recorded in manifest["preserved_sha256"]:
            source = Path(recorded)
            if not source.is_absolute(): source = ROOT / source
            source = source.resolve()
            if source.parent == ROOT / "data" or source.parent == ROOT:
                if source.is_file():
                    relative = source.relative_to(ROOT)
                    (target / relative).symlink_to(source)
                    count += 1
                else: raise FileNotFoundError(source)
        print(f"Offline full suite; {count} historical fixture files; original/public data untouched.", flush=True)
        return subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"], cwd=target).returncode
if __name__ == "__main__": raise SystemExit(main())
