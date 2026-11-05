# DLR optimization 2: prospective protocol

Preserve original DLR and all other studies. Local existing Qwen2.5-1.5B BF16,
same chat template, greedy EOS, 32-token cap on short tasks, 96 on long tasks.
Seed 2026110605. No training, no threshold selection from final evaluation.

Baseline and Old DLR call the unmodified previous runtime. Old DLR reuses MLP
residuals at layers 8,12,16,24, relative L2 <= .5, at most one reused step.
Attention/KV always update; prefill never reused.

Device-Resident DLR prototype: identical gate computed as MLX arrays, no gate
item()/eval to branch Python during decoding. Custom Metal kernels implement
uniform GPU conditional branches BEFORE gate/up/down weight reads and dot
products. Their output counters show whether actual matmuls execute. This is
not mx.where(full_mlp, cached) compute-both selection. Fresh GEMV reductions
may differ numerically from stock MLX: compare fresh-only kernel controls and
disclose this implementation confound. If build/correctness or time fails,
record infeasibility and never replace missing results with estimates.

Lightweight DLR: compare current input token ID (already obtained by greedy
generation) to the most recent fresh anchor's input ID; reuse the four MLP
residuals only if IDs equal and age zero. One consecutive reuse maximum.
This proxy is NOT a guarantee of equal hidden state/context. No hidden-state
distance, extra LLM call, or extra GPU gate read. Record losses, not just speed.

Development pilot is neutral JSON + repeated-word prompts, excluded from
evaluation. If a custom-kernel request exceeds 20s for <=16 tokens, use a
smaller Device evaluation subset and report missing conditions, not fabricate
four-way full comparison. Otherwise run all four methods in randomized blocks.

Evaluation: original DLR replay32; previously unused fresh32 (same format mix
as DLR fresh32 but new templates/values); additional four repetitive long
requests and four varied long requests. "high/low" are input-design hypotheses,
not labels derived by selecting favorable output. Each method3 repeats.
Fresh32 novelty checks against recursive prior stem keys; semantic/template
overlap is disclosed. References are offline scoring only.

All72 questions additionally run3 repeats in fixed-stream mode, using the
first Baseline output token IDs, so total primary requests=72*4*3*2=1728.
Teacher-forced text accuracy is not independent model accuracy. Additionally
record each forced-step predicted token, to diagnose internal output changes.
No early stopping beyond EOS/common cap. No sample or failure exclusion.

Record gate-inclusive synchronized wall/CPU, model step time, input/output
tokens, time per output token, per-request MLX peak and RSS lifetime peak,
real MLP reuse/bypass counters, selected device flags, actual Python gate
scalar reads, old gate/eval sites, and GPU gate graph construction time.
Device gate standalone GPU duration cannot be isolated without adding
synchronization; mark unmeasured, include in wall instead. CPU process_time
is not GPU time. Administrative native audits/warmups/load are separate.

Dense matmul FLOPs are analytical (MAC=2), not hardware measurements.
Estimate old/device gate arithmetic as ~5*d+4 operations per evaluated gate;
compare same cap/fixed stream; selection/copy/age tensor element counts
separate. Exp/norm/activation and hardware memory costs are not full FLOPs.
Lightweight ID comparisons counted, not represented as GPU FLOPs. Report a
limited model-matmul+gate-arithmetic proxy, not full measured compute cost.
No measured GPU power/FLOPs/money. GPU kernels still launch on reused steps
and intermediate zeros/output copies are not free.

Minimum success: equal observed content accuracy on both short cohorts,
no content/quality loss on long tasks, >=5% equal-stream speed improvement
including gates, reduced estimated net arithmetic, consistent direction all3
repeats. Report paired question/repetition bootstrap and uncertainty; equal
accuracy in32 synthetic tasks cannot prove population noninferiority. Label
A only with substantive supporting evidence; B for limited improvements with
unresolved fidelity/uncertainty; C for no qualifying optimization.

Primary implementation sources verified:
https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.where.html
https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
Previous activation-reuse literature remains relevant; no novelty claim.
