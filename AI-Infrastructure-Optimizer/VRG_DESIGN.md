# VRG: prespecified controlled local study

Seed 68112026. Five repetitions, both installed qwen2.5 0.5B and 1.5B.
Historical VVP/RCAV/Selective results are diagnosis only. Train only on the
200 dedicated historical training questions per model, not any previous evaluation
or VVP validation. New validation100 chooses L2 and temperature by log loss.
New untouched test300 has five types of60, fixed balanced A/B cohorts of150.
Fresh stems checked against all stored CSV/JSON questions; semantic/template
independence is not guaranteed. Gold labels stay outside runtime APIs.

All methods share a single frozen initial output/model/question. Existing safe
calculator handles exact arithmetic for every method; this is not a VRG advantage.
Existing model, verifier prompt, temperature0, seed42, max64 output tokens preserved.
Content interpretation freezes original parsing plus previous VVP explicit-content
audit (gold-blind); missing content is incorrect, format scored separately.

No Verification keeps primary. Always Accept returns the verifier raw output even
if unresolved. Selective uses its unchanged existing gate and unresolved-output
fallback. Heuristic/Learned call the verifier on every request then decide on
acceptance: rejection does NOT save a verifier call. Heuristic preserves resolved
answers unless the candidate agrees, and accepts valid candidates for unresolved
primaries. Learned predicts repair/damage/unchanged via lightweight multinomial
logistic regression on observable initial+candidate characteristics, accepts
positive expected gain, and protects resolved calculator answers. Invalid
candidates rejected. No model-provided verbal confidence is assumed calibrated.
Oracle is analysis-only upper bound from Always candidates, never online or free.

Five executable arms run independently with actual verifier calls, shared arrivals,
random base method order rotated across five repetitions (each position once),
alternating model order, unload and neutral warmup. Record candidate variation.
Oracle uses the exact Always candidate only after execution; no measured oracle cost.

Success assessed for each guard, with no best-model selection: paired question-cluster
bootstrap95 and conservative exact accuracy lower bound >0 vs Selective, harm
reduction vs Always, retain >=80% of Always repairs (descriptive prespecified
floor), paired-repeat CPU upper95 <=0 vs Selective, direction on both fixed
150-question cohorts, BOTH models required. Uncertain differences cannot establish
success. Report H1-H5 independently, including failure. Secondary comparisons with
Always and No Verification; no tuning on test curves or error cases.

Measure actual server+Python CPU, guard/gate Python CPU, wall, tokens, calls,
Ollama RSS sampled100ms; same initial generation measured once then added only as
component accounting (not independent production runs). New calibration collection
and fit costs separately measured; historical collection is sunk reused data, not
free. GPU FLOPs/power/money unmeasured. A post-verification guard cannot reclaim
already-spent compute. H4 savings beyond measured costs are not assumed.
