# PVCG: prespecified local controlled study

Seed78112026. Train on200 historical TRAINING-ONLY tasks/model from VVP raw
training outputs. Never use old VRG/VVP/RCAV evaluation or VVP validation as training.
New calibration100 and final evaluation300: arithmetic, context-document lookup,
logic, coding, reading (20/60 per type). Context lookup is synthetic supplied-fact
retrieval, NOT a claim of new general-knowledge recall. Five-type results separate.
No old normalized stem/choice-shuffle duplication; template semantics remain a limit.

PVCG filters the frozen Selective eligible set before verify(): category + initial
resolution/format state; support>=10 with category backoff. Symmetric neutral
Dirichlet(1,1,2) repair/damage/unchanged prior; mean gain minus uncertainty weight*SE
must exceed CPU penalty*historical group median actual verifier CPU. These are
forecasts/utility scores, not actual costs or a guaranteed risk bound. No extra LLM
predictor call. Calibrate weight0/.5/1 and penalty0/.05/.2 ONLY on calibration100:
max correctness, ties lower actual verifier CPU sum, then conservative parameters.
Unknown/unsupported groups skip with explicit reason; a skip-all profile is not success.
This is empirical verification gating, not a claim of a new foundational technology.

Four executable methods: None, Full, frozen Selective, PVCG; same immutable initial
outputs per model, same verifier prompt/temp0/seed42/max64 and content parser as VRG.
All verified methods adopt a resolved candidate else retain primary; no post-verifier
guard. Exact calculator path common to all methods; report non-tool subgroup too.
Freeze every test selection before any test verifier call. Recompute and charge gate
CPU/wall at actual execution, assert identical plan; reject gold/future keys in API.

Both qwen2.5 1.5B/0.5B; each method5 actual repeats, independent verifier calls.
Random base method order rotates across repeats; first4 positions balanced, repeat5
repeats one position, recorded limitation. Alternate model order; reset and neutral
warmup each model/repeat; common random question arrival. Caches not fully removed.

Success requires BOTH models: conservative paired unique-question lower95 accuracy
delta vs Selective>=0, vs None>0, paired-repeat CPU upper95 vs Selective<0, actual
input/output tokens not increased, some verifier calls, positive None accuracy direction
in both fixed150 cohorts. Zero accuracy tolerance; uncertain differences not success.
Bootstrap4000; repetitions of frozen outputs are not independent questions. No
evaluation threshold changes, posthoc best model selection, or hidden extra models.

Measure actual server+Python CPU, wall, gate overhead, tokens/calls, observed RSS100ms.
Initial generation measured once; initial+stage sum is component accounting, NOT an
independently repeated production run. New calibration and fitting separately measured;
historical acquisition sunk and not free. GPU FLOPs/power/money unmeasured. Compare all
four observed Pareto points; None dominance or accuracy loss means no efficiency success.

Existing files recursively hashed under src/tests/data plus root files, never modified.
All outputs new UUID; canonical report/CSV/PNG in its own subdirectory. Tests use the
pre-VRG input snapshot to rerun fixed fresh-dataset tests without deleting real data.
