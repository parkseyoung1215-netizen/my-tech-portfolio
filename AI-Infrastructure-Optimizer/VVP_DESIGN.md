# Verification Value Predictor: prespecified local study

Run `.venv/bin/python -B -u src/vvp_experiment.py`; all existing code/data preserved.

Historical evaluation is diagnosis only. Training uses two verified training-only
200-question cohorts, recollected for each model; no past evaluation stems.
New validation100 selects L2 (.01,.1,1) and temperature (.75,1,1.5) by log loss,
and calibrates CPU regression by median validation cost ratio. New test300 is
untouched by fitting, split into two prespecified balanced150 cohorts. All split
stems checked against stored data. Semantic template overlap remains a limitation.

Multinomial logistic regression predicts unconditional repair, damage, unchanged
correctness. EV = P(repair)-P(damage). Features: inferred category, syntax family,
AST size, input/output lengths, numeric counts, difficulty, parsing status, tool
source; no gold, future verifier output, IDs or memorized question lookup. Proven
calculator outputs have repair probability0. No AI inference in the predictor.
Cost model learns log actual verifier CPU (Ollama+Python) and is validated separately.

Six methods: No Verification, Full (uniform under limited budget), Selective,
RCAV, Improved RCAV, VVP. Frozen existing profiles. Common initial answers once
per model; actual independent verifier calls for every condition, not a replayed
cost simulation. Six budgets0/10/25/50/75/100%, five repeats on both installed
models. Each budget caps calls floor(N*pct) and predicted CPU quota. Other methods
consume budget in common randomized arrival order; VVP chooses positive EV by
EV/predicted CPU, then executes in that same arrival order. Greedy, not optimal
knapsack. Prediction errors can exceed actual CPU budget; log these overruns.

All allocations freeze before any final-test verification. Recompute and charge
planning at execution, asserting the same allocation. Full100 supplies each
repeat's actual matched-primary reference to diagnose missed improvements after
execution only. Different responses to identical verifier hashes are recorded.

Main success budget fixed25%, not chosen from test curves. Require conservative
paired lower95 accuracy delta >=0 vs Selective100, >0 vs NoVerification0, CPU
upper95 delta <0 vs Selective100, and directional quality replication in both
150-question cohorts and both models. All other budget points are secondary.
Question-cluster inference keeps repeated tasks together. No significance claim
from an uncertain difference, reused initial output or posthoc best budget.

Initial generation measured once; adding it to per-method actual planning plus
verification is labeled component accounting, not an independently executed
production end-to-end pipeline. Training/fitting overhead recorded separately,
no amortization assumption. No FLOPs/power/money estimation. CPU/RSS measurement
and runner-reset protocol reuse controlled validation. Server durations are raw
telemetry only: inconsistent fields cannot be used as trusted computation costs.

Expected-value metareasoning, learning-to-defer, multinomial regression, shrinkage,
greedy budget allocation and temperature calibration are existing techniques.
Contribution is a local empirical transition/value predictor comparison, not a
claim of a newly invented foundational technology or frontier-model evidence.
