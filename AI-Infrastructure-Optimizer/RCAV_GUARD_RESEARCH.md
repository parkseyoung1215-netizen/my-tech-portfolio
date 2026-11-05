# Repair-risk guard for RCAV

All original code and data stay immutable. Run:
`.venv/bin/python -B -u src/rcav_guard_experiment.py`

Tests: `.venv/bin/python -B -m unittest discover -s tests -q`

The old150 is a development replay: its observed11 losses informed the diagnostic
question and general feature design. It is not an independent confirmation set.
New150 is locked before fitting: 30/category, 100 basic variants,40 structural
variants,10 additional fact-only probes. Semantic template overlap remains.

Use frozen original1.5B RCAV/Selective profiles. Add a guard only when original
RCAV skips. Features: syntax-only code AST family, logic request structure and
primary response parsing status; never execute code or inspect evaluation gold.
Calibrate guard on historical training100 plus new disjoint training100, never
the old150 or new150. Rules fixed before measurement: >=8 samples, 95% Wilson
lower bound of unconditional repair frequency >.05, fixed>harmed, measured mean
verification time <=1.5x original calibration median. Status strata fall back
only to the same family; insufficient evidence skips. Safe calculator results
are never additionally verified. These screens do not guarantee net accuracy.

Compare original RCAV, guarded RCAV and Selective on both datasets,3 repeats,
fixed seed42/temp0/64tokens, identical clients/prompts/parser, rotated method
order and preshuffled blocks. Unique model calls, server tokens/time, wall/CPU
and boundary RSS are actual measurements. Cost forecasts are estimates, not
money/FLOPs. Calibration and33 fixed-candidate diagnostic rechecks are separate.

Question-cluster paired bootstrap of repeated accuracy is planned with4000 draws
and95% intervals, stratified by category. Repeats are not independent450 tasks.
Report per-repeat paired losses/repairs, type and shift results, net verification
damage, retained savings versus Selective, and existing11 recovery separately.
Do not tune gates on either evaluation set. Do not claim generalization from
development replay gains or cherry-pick an insignificant fresh-set result.
