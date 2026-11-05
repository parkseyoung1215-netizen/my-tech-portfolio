# Automatic test results

Full historical snapshot suite: 247 tests, 243 passed, 4 NumPy-environment skips, 74.013 seconds.

Isolated MLX/NumPy environment supplement: 13 tests passed in 0.047 seconds; includes all four previously skipped numerical tests and the five new DLR tests.

Native real-model audit: exact same last-position-head baseline logits, max absolute difference 0; forced-reuse bypassed trapped MLP; fresh-compute positive control raised.

All 960 primary requests checked attention/MLP counters and KV offsets. Equal-stream inputs and token counts are identical. Complete resume added zero new requests.

Plot rendering initially lacked matplotlib in the model venv; reused the existing plotting dependencies under /private/tmp/vvp_plot_deps_20261105. No package installation or model rerun. All graph artifacts were visually checked.
