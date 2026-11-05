# Dynamic Layer Reuse local research

The existing Qwen2.5-1.5B-Instruct weights and `.alsr-venv` are required.
No paid API, download, upload, or original-file replacement is used. Metal GPU
access must be allowed by the execution environment. Offline mode is forced.

The new modules are `src/dlr_policy.py`, `dlr_runtime.py`, `dlr_data.py`,
`dlr_experiment.py`, `dlr_analysis.py`, `dlr_analysis_final.py`, `dlr_report.py`; tests are
`tests/test_dlr.py`. Policies are frozen in `DLR_DESIGN.md`.

```sh
.venv/bin/python -B -m unittest tests.test_dlr
.alsr-venv/bin/python -B -m src.dlr_experiment --pilot
# First-time fresh study only: rejects already recorded "new" questions.
.alsr-venv/bin/python -B -m src.dlr_experiment
# Resume the printed UUID folder; completed requests are never regenerated.
.alsr-venv/bin/python -B -m src.dlr_experiment data/dlr_<uuid>
.venv/bin/python -B -m src.dlr_analysis_final data/dlr_<uuid>
.alsr-venv/bin/python -B -m src.dlr_report data/dlr_<uuid>/ANALYSIS_<uuid>
# Existing dataset builders require the historical read-only test snapshot:
.venv/bin/python -B -m src.vrg_test_snapshot data/vrg_6828c214e04841e1ab6ca4e5392de9a9_manifest.json
```

Every analysis creates a new directory; rendering uses exclusive writes and
must not be rerun into an existing rendered directory. All main natural/fixed
requests have append-only checkpoints. Model loads, native parity, forced-reuse
controls, and warmups are administrative costs separately recorded; they are
not silently included in some policies but excluded from others.

Natural generation is real greedy output under one common 32-token safety cap.
Fixed-stream requests execute real model computations on equal token positions;
their forced text is not an independent quality result. FLOPs are dense-matmul
estimates, and process CPU time is not GPU time or total chip computation.

`dlr_analysis.py` is the original analysis version frozen in the run manifest.
The final analysis adds explicit verdict output in `dlr_analysis_final.py`.
An export was blocked after this newly added analysis source was edited during
execution; its frozen bytes were restored and the extension was moved to the
new module. All 960 saved measurements were retained. The completed resume
exported them with zero new model requests; see `EXECUTION_RECOVERY.json`.

An unmodified Baseline and fresh attention KV updates are used. Only the MLP
residual at four candidate layers can be approximated using a previous token.
This is activation reuse research, not a claim of a new independent invention.
Do not treat fewer output tokens, zero reuse, or slower execution as success.
