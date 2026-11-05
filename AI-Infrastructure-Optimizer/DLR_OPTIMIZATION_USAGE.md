# DLR optimization 2

Existing model and `.alsr-venv` required. No model/package download or paid API.
Metal access requires the host execution environment to expose the GPU.

```sh
.venv/bin/python -B -m unittest tests.test_dlr_optimization
.alsr-venv/bin/python -B -m src.dlr_opt_experiment --pilot
# A NEW run: only before this exact synthetic holdout has ever been stored.
.alsr-venv/bin/python -B -m src.dlr_opt_experiment --new <successful-pilot-folder>
# Resume: existing append-only checkpoints are not regenerated.
.alsr-venv/bin/python -B -m src.dlr_opt_experiment <run-folder>
.venv/bin/python -B -m src.dlr_opt_analysis <run-folder>
.venv/bin/python -B -m src.dlr_opt_report <newly-printed-analysis-folder>
.alsr-venv/bin/python -B -m src.dlr_opt_controls <run-folder>
.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json
```

All reports/CSVs live in a UUID study directory, preserving every earlier study.
Do not render twice into the same analysis folder. Model/policy/source hashes
are frozen before evaluation; existing code uses the unmodified old DLR runtime.
Plotting reuses `/private/tmp/vvp_plot_deps_20261105` without installing anything
into the model environment. Weights/model revision/package versions are pinned
in MANIFEST.json, with per-request results in checkpoints and exports in EXPORT.

Device DLR uses true conditional Metal MAC loops, NOT compute-both `where`.
It replaces stock fresh GEMV kernels, so floating-point changes are a documented
confound. Device GPU gate duration alone is unmeasured; graph dispatch time is
not GPU compute time. Gate-inclusive request wall and CPU remain primary.
The always-present token argmax host synchronization is not claimed eliminated.

Lightweight gates only on previously available input token IDs and an age cap.
Equal token IDs do not imply equal hidden states. Report damage and missed EOS.
Fixed streams execute actual model work but their text is forced: their accuracy
is not independent output quality. FLOPs are analytical estimates; no measured
GPU energy, chip FLOPs, GPU duration, or money savings are claimed.
