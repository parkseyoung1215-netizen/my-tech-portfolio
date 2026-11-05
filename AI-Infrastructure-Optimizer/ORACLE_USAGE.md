# Zero-Overhead Oracle research

Existing local Qwen2.5-1.5B weights and `.alsr-venv` are required. Metal access
must be exposed by the execution environment. All model files are hash checked;
HF offline mode is forced. No paid API, download, upload, or original-file edit.

```sh
.venv/bin/python -B -m unittest tests.test_oracle
.alsr-venv/bin/python -B -m src.oracle_experiment --smoke
# First-time only: rejects duplicate new question stems after they are recorded.
.alsr-venv/bin/python -B -m src.oracle_experiment
# Continue exactly the printed UUID directory (completed requests not regenerated).
.alsr-venv/bin/python -B -m src.oracle_experiment <run-folder>
.alsr-venv/bin/python -B -m src.oracle_profile <run-folder>
.venv/bin/python -B -m src.oracle_analysis <run-folder>
.venv/bin/python -B -m src.oracle_report <new-analysis-folder>
# Full REAL inference replay into a NEW UUID, same questions, explicitly NOT unseen.
.alsr-venv/bin/python -B -m src.oracle_replay <run-folder>/MANIFEST.json
.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json
```

The report and all statistics/graphs are exclusive-write artifacts. Do not
render twice into the same analysis directory. Construction traces and every
candidate result are append-only under tables; primary requests under checkpoints.
Partial attempts are retained, and completed TABLES.json files are reused on
resume. All primary schedule/source/table hashes are frozen before comparisons.

Oracle table selection intentionally uses evaluation references AND future
Baseline activations. It is optimistic best-of-three, not a deployable method,
not exhaustive optimal scheduling, and not an independent holdout accuracy test.
Random control uses an independent seed with the same per-step quotas/cooldown.
Baseline and Existing DLR each have one setting, not six independent rate points.

Skip targets refer to decode MLP invocations, not prefill or total-model FLOPs.
Fixed-stream text is forced: use natural generations for quality. Intrusive
profiling adds synchronization and is excluded from primary latency comparisons.
MLX cache dispatch is not GPU memcpy time; native projection call count is not
GPU kernel count. Actual GPU FLOPs/timestamps/power/money are unmeasured.

Graphs reuse existing plotting-only dependencies in
`/private/tmp/vvp_plot_deps_20261105`, without altering the model environment.
