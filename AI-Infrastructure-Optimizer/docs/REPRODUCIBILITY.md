# Reproduction and public-copy limitations

## Offline validation

From `AI-Infrastructure-Optimizer/`, run the three `tools/` commands in the README.
These are standard-library checks and mocked unit tests, not model performance runs.
The full suite uses the same pre-VRG fixture as `src.vrg_test_snapshot`, resolving
sanitized relative paths before linking read-only inputs into a temporary directory.
Fresh-data collision protection stays active; evaluation data are not removed to pass tests.
Four NumPy-dependent tests may be skipped by a minimal Python environment. Install NumPy
in a separate environment to run the optional numerical command in the README.

`verify_readme.py` recalculates the 142 numeric facts in `README_FACTS.json` from CSV.
`verify_publication.py` checks the approved public inventory and original-to-public lineage.
It checks consistency, not the authenticity of a historical execution or scientific validity.

## Actual local execution

Existing Ollama examples in the README use the local server and existing model tags.
Do not represent their output as a rerun of a historical measurement. Check installed
models first; no paid API is required. New generated CSV names must be unique.

MLX experiments require an Apple Silicon Mac, the recorded MLX/MLX-LM versions, local
Qwen2.5-Instruct checkpoints, and sufficient unified memory. Original manifest/model
profile JSON files record checkpoint revisions, dtype, package and hardware information.
Model weights, model-download caches and virtual environments are intentionally omitted.
Temporary checkpoint paths such as `/private/tmp/...` describe the original machine;
recreate the recorded checkpoint locally rather than assuming it is bundled.

Representative historical entry points (run only after checking local prerequisites):

```sh
python3 -B tools/run_recorded.py src.dlr_experiment
python3 -B tools/run_recorded.py src.dlr_opt_experiment
python3 -B tools/run_recorded.py src.oracle_experiment
```

No folder argument requests a **new run**; inspect the module's `create()` and model
profile paths before execution. Supplying a recorded folder requests resume in several
modules and can write new exports/session receipts. Work on a copy of that folder to
preserve historical artifacts. Each original design/report retains detailed commands.
The publication task itself runs offline checks only; it does not execute these MLX jobs.

## Why old byte-hash checks can fail

Original run receipts and `preserved_sha256` values refer to private historical bytes.
The public copy redacts private absolute paths, makes local Markdown links relative,
and adds an English synthesis. Unlike the first publication, this update does not rebase
historical fingerprints. The first publication's earlier transformations remain documented
in `PUBLICATION_HASH_HISTORY.json`, and changed previous public documents are archived.

`tools/run_recorded.py` is an explicit adapter for modules importing
`efficiency_repeat.frozen_check`: it accepts exact original hashes or a declared
original-to-public transformation with a matching current public hash. It logs accepted
transformations and does not disable arbitrary guards. Additional module-specific checks,
missing historical versions and excluded caches may still prevent a replay; this is
reported as a reproducibility limitation rather than silently bypassed. The adapter does
not change model prompts, scoring, thresholds or stored measurement values.

## Interpretation

Repeated greedy outputs from shared initial answers do not form independent question
samples. Different studies use different hardware/backend/precision and timing scopes.
Posthoc scoring audits remain separate from original scores. Oracle reference/future-aware
selection is privileged and not a deployable gate. FLOPs are analytical unless a report
explicitly supplies a hardware measurement; GPU power/energy/money remain unmeasured.
