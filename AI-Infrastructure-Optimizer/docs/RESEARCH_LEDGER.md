# Research ledger

Every entry refers to an actually recorded local study. Detailed original reports may be in Korean; source/data are retained. Time definitions differ across studies and must not be pooled into a universal speed ranking. Unknown measurements are not filled with estimates.

## Initial inference benchmarks and answer validation

- **Goal / hypothesis:** Compare small/large model accuracy, extraction failures and latency.
- **Implemented method:** Integer extraction distinguishes correct, wrong and unextractable answers. CSV-based routing rejects unsupported quality constraints.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Arithmetic benchmark, one recorded pass per model; repeated rows are not repeated independent trials.
- **Accuracy / actual timing:** qwen2.5:0.5b: 10.00% accuracy, 0.1022 s mean request wall; qwen2.5:1.5b: 53.33% accuracy, 0.2093 s mean request wall.
- **Resources / accounting:** Request wall, server durations and output tokens; CPU/money not claimed where absent.
- **Comparison:** Same arithmetic questions, two existing local models.
- **Failures / limitations:** Tiny arithmetic-only dataset; legacy cost/cache examples contain illustrative values, excluded from empirical claims.
- **Finding for subsequent research:** Use measured CSV evidence with explicit unsuitable/uncertain outcomes; benchmark accuracy is not a service guarantee.
- **Evidence:** [report](../ROUTING_USAGE.md), [implementation](../src/benchmark_v2.py).

## Adaptive Routing

- **Goal / hypothesis:** Use the small model for calibrated easy requests and retain fallback/refusal.
- **Implemented method:** Separate calibration statistics and confidence bounds, with parser/API fallback rather than gold-guided routing.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Separate calibration and 40-question evaluation; one pass.
- **Accuracy / actual timing:** Fixed old-prompt:55%/12.352 s; old automatic50%/4.428 s; adaptive50%/2.814 s.
- **Resources / accounting:** Coverage50% for both routers, tokens/server durations and calibration cost reported.
- **Comparison:** Common answered requests compared separately; prompt-matched fixed control included.
- **Failures / limitations:** Rejecting half the requests is not equal-coverage acceleration; one run.
- **Finding for subsequent research:** Shared answered-subset analysis is necessary when coverage differs.
- **Evidence:** [report](../ADAPTIVE_RESULTS.md), [implementation](../src/adaptive_router.py).

## Coverage Routing

- **Goal / hypothesis:** Improve answered coverage without simply lowering trust thresholds.
- **Implemented method:** Generate, validate and escalate through explicit small/large paths; retain failures and refusal reasons.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Existing200 and fresh100 problems, separate cohort summaries; one recorded run.
- **Accuracy / actual timing:** Replay200 Coverage: 64.50% accuracy, 44.173 s request total.
- **Resources / accounting:** Coverage, answered accuracy, handoffs, fallback and per-model time in CSV.
- **Comparison:** Fixed, old Adaptive and Coverage use matched questions.
- **Failures / limitations:** Validation is not semantic proof; calibration support and refusal effects limit generalization.
- **Finding for subsequent research:** Record whether large-model handoff really occurred; do not hide rejected requests.
- **Evidence:** [report](../data/coverage_951686cca01344ed8b732fc1e66d7cab_report.md), [implementation](../src/coverage_runner.py).

## Tool-Assisted Routing

- **Goal / hypothesis:** Avoid LLM arithmetic when a bounded exact tool is appropriate.
- **Implemented method:** Allowlisted AST integer arithmetic without eval; return computed results directly and route unsupported input to models.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Existing200 and fresh100 arithmetic evaluations; one recorded run.
- **Accuracy / actual timing:** Fresh100 Tool-Assisted: 100.00% accuracy, 0.012 s request total.
- **Resources / accounting:** Tool/model calls, tokens, latency and inappropriate-tool/fallback records.
- **Comparison:** Fixed, Adaptive, Coverage and Tool on the same corpus.
- **Failures / limitations:** Arithmetic success is domain-specific, not general AI correctness or monetary saving.
- **Finding for subsequent research:** Use strict task applicability; preserve fallback and misapplication logs.
- **Evidence:** [report](../data/tool_routing_73d749e404ea4347bbba5883b485a75e_report.md), [implementation](../src/safe_calculator.py).

## Mixed tasks and scoring reliability

- **Goal / hypothesis:** Check whether arithmetic gains transfer and separate format failure from content error.
- **Implemented method:** Keep original strict scores; add a separate content-based audit and inspect actual selected model/tool.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** 100 mixed arithmetic, knowledge, reasoning, coding and reading tasks; original run plus offline audit.
- **Accuracy / actual timing:** Original strict Tool-Assisted score 45.00%, total 5.902 s; audited content metrics remain separate.
- **Resources / accounting:** Per-type accuracy, calls, tokens, CPU/RSS where recorded; audit is not a new inference run.
- **Comparison:** Four strategies shared the general adapter on non-arithmetic requests.
- **Failures / limitations:** Identical responses do not demonstrate effective differentiated routing; scoring changes cannot be credited as algorithm gains.
- **Finding for subsequent research:** Maintain raw text, output contracts and pre/post-audit metrics.
- **Evidence:** [report](../data/mixed_reliability_318dcf5fdf2045179fd24fd3a25866fa_report.md), [implementation](../src/mixed_reliability_audit.py).

## Rule-Based / Confidence-Based / Hybrid

- **Goal / hypothesis:** Select tool/small/large solution paths using request-time evidence.
- **Implemented method:** Rule heuristic, calibrated small-model Wilson bound, and their Hybrid combination.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Separate100 calibration and new300 evaluation tasks, five types; one pass.
- **Accuracy / actual timing:** See README CSV-derived table: Hybrid80.33%/14.549 s vs Rule75.33%/20.327 s.
- **Resources / accounting:** Tokens, calls, fallback, server-group/Python CPU and sampled RSS.
- **Comparison:** Fixed and Confidence coincide because no small-model category passed calibration.
- **Failures / limitations:** Six individual correct answers regressed; shared tool advantage and calibration overhead matter.
- **Finding for subsequent research:** A localized routing benefit exists; it is not multi-model general superiority.
- **Evidence:** [report](../data/solver300_c22b83397f0945b4adfb8304f7b2f268_report.md), [implementation](../src/solver300_runtime.py).

## Pre-solve omission and repeated verification

- **Goal / hypothesis:** Skip verification with nonpositive expected benefit; test output caps and batching.
- **Implemented method:** Early Exit for proven arithmetic, Selective calibration rule, fixed cap and grouped JSON verification.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Original new150; later same150 repeated3 times, separate100 calibration.
- **Accuracy / actual timing:** CSV-derived README table: Selective66.67%/17.379 s vs Baseline58%/19.461 s; batching68%/36.231 s.
- **Resources / accounting:** Server processing vs request wall, calls/tokens, queue latency, CPU and boundary RSS.
- **Comparison:** Baseline is a deliberately constructed full-verification ablation, not the best production router.
- **Failures / limitations:** Early Exit did not consistently improve wall; batching reduced calls but increased tokens/time. Later parser changes are separate.
- **Finding for subsequent research:** Model-call reductions must be checked against real workload and latency.
- **Evidence:** [report](../data/efficiency_repeat_1b6d963181b249e0889270ca4590ea2a_report.md), [implementation](../src/skipwork_runtime.py).

## RCAV and Improved RCAV

- **Goal / hypothesis:** Compare predicted repair benefit and verification cost, then selectively guard high-risk omissions.
- **Implemented method:** Calibrated error risk, repair/harm estimates and cost; independent improved guard.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Original150 and later replay150/new150,3 repeats; models separately calibrated.
- **Accuracy / actual timing:** Original1.5B RCAV 75.33%/5.588 s; this is not the later controlled result.
- **Resources / accounting:** Full/request/model processing, calls/tokens, CPU/RSS and retained/lost answers in linked CSVs.
- **Comparison:** Baseline, Selective, RCAV sensitivities and improved variant.
- **Failures / limitations:** Initial shared/cache/order confounds;0.5B accuracy trade-offs and missed repairs.
- **Finding for subsequent research:** Control identical initial answers before interpreting verification efficiency.
- **Evidence:** [report](../data/rcav_guard_b2e19463ef974c54a65b0770f2cf475d_report.md), [implementation](../src/rcav_policy.py).

## RCAV Controlled Efficiency Validation

- **Goal / hypothesis:** Separate cached initial generation, verification work and execution-order effects.
- **Implemented method:** Same frozen first answers, randomized condition blocks and actual verification executions.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Replay150 and fresh300,2 models,5 repeats.
- **Accuracy / actual timing:** 1.5B/fresh300 RCAV 62.00% with 0.011 s verification-stage wall; initial generation is additional.
- **Resources / accounting:** Stage CPU/tokens, initial-generation costs, sampled RSS, transitions and Pareto records.
- **Comparison:** Baseline, Selective, RCAV, Improved on identical initial outputs.
- **Failures / limitations:** RCAV often performs no verification; superiority to simply preserving the first answer remains unresolved.
- **Finding for subsequent research:** Shared-component sums are not independently measured end-to-end service latency.
- **Evidence:** [report](../RCAV_CONTROLLED_VALIDATION.md), [implementation](../src/controlled_validation.py).

## Verification Value Predictor (VVP)

- **Goal / hypothesis:** Predict repair minus damage probability per estimated verification CPU cost.
- **Implemented method:** Lightweight learned transition/cost predictors and budgeted selection; no predictor LLM calls.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Fresh300;2 models, six methods/budgets,5 repeats; disjoint training/calibration.
- **Accuracy / actual timing:** 1.5B VVP25 61.00%, stage wall 0.011 s, but no verification selected;0.5B had no net audited repair gain.
- **Resources / accounting:** Prediction/training costs, actual CPU quota overruns, tokens/calls and content-audit sidecars.
- **Comparison:** No Verification, Full, Selective, RCAV and Improved; all budget sweeps retained.
- **Failures / limitations:** Degeneration to no verification, imperfect cost calibration and scoring sensitivity; not a successful new technology.
- **Finding for subsequent research:** Measure incremental value against No Verification, not only Full Verification.
- **Evidence:** [report](../VVP_RESEARCH_REPORT.md), [implementation](../src/vvp_model.py).

## Verification Revision Guard (VRG)

- **Goal / hypothesis:** Avoid correct-to-wrong revisions without excessive repair loss.
- **Implemented method:** Post-verification heuristic/learned adoption guards, shared first answers and a labelled Oracle upper bound.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Fresh300,2 models,5 repeats; training/calibration separated.
- **Accuracy / actual timing:** 1.5B heuristic 59.33%, verification/decision stage 49.977 s.
- **Resources / accounting:** CC/CW/WC/WW, actual verification calls, CPU/tokens and separate training cost.
- **Comparison:** No Verification, Always Accept and Selective; Oracle is not deployable.
- **Failures / limitations:** No guard passed success criteria; accepting/rejecting after verification cannot recover already spent inference.
- **Finding for subsequent research:** Move compute gating before verification, while retaining an explicit no-verification comparator.
- **Evidence:** [report](../data/vrg_6828c214e04841e1ab6ca4e5392de9a9/VRG_RESEARCH_REPORT.md), [implementation](../src/vrg_policy.py).

## Pre-Verification Compute Gate (PVCG)

- **Goal / hypothesis:** Gate verification using already available first-answer features.
- **Implemented method:** Calibrated transition uncertainty and measured-cost proxy before revision generation.
- **Models / hardware:** Local Ollama Qwen2.5 tags as recorded; per-run digests/platform metadata retained. Do not retroactively infer identical hardware, quantization or load from later measurements.
- **Data / repetitions:** Fresh300,2 models,5 repeats; separate training and calibration.
- **Accuracy / actual timing:** 1.5B PVCG 61.33%, stage wall 0.009 s; both models selected zero verifications.
- **Resources / accounting:** Gate CPU/wall, calls, tokens and shared initial-generation accounting.
- **Comparison:** No Verification, Full and existing Selective.
- **Failures / limitations:** Degenerated to No Verification with extra decision overhead; success conditions failed.
- **Finding for subsequent research:** A gate must add value over preserving the first answer, not merely suppress calls.
- **Evidence:** [report](../data/pvcg_f4de31fe19654eaabb9a17700f3d6a26/PVCG_RESEARCH_REPORT.md), [implementation](../src/pvcg_policy.py).

## Adaptive Layer Skipping Research (ALSR)

- **Goal / hypothesis:** Test true Transformer-block bypass rather than fewer calls.
- **Implemented method:** Static/random/BI controls plus low-cost Adaptive-Sketch/Leverage; no weight training.
- **Models / hardware:** Native local Qwen2.5-Instruct BF16 via MLX, Apple M4/16 GiB; revisions/packages in the manifest. Not directly comparable to Ollama Q4 timings.
- **Data / repetitions:** 32 new evaluation tasks,3 repeats,2 models; separate training12/calibration8 and fixed-load subset.
- **Accuracy / actual timing:** 1.5B baseline task success 78.12%, natural total 12.805 s; no quality-and-timing candidate passed.
- **Resources / accounting:** Native bypass controls, synchronized wall/CPU, MLX peak and analytical FLOPs; original whole-job clock missing after export error.
- **Comparison:** Eight controls including matched-random skipped-layer counts.
- **Failures / limitations:** Tiny/OOD calibration, numeric head differences and recovered export; not a full LayerSkip reproduction.
- **Finding for subsequent research:** Actual bypass and lower estimated FLOPs do not guarantee better latency.
- **Evidence:** [report](../data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f/ALSR_RESEARCH_REPORT.md), [implementation](../src/alsr_engine.py).

## Layer impact and Stable Runtime

- **Goal / hypothesis:** Explain Stable1 quality/latency discrepancy before adding policies.
- **Implemented method:** Single-layer output-distribution ablation, KL/top1/margin analysis and legacy-vs-lean execution.
- **Models / hardware:** Native local Qwen2.5-Instruct BF16 via MLX, Apple M4/16 GiB; revisions/packages in the manifest. Not directly comparable to Ollama Q4 timings.
- **Data / repetitions:** Layer-impact64 evaluation; runtime replay64/fresh64 with3 repeats and fixed-stream diagnostics.
- **Accuracy / actual timing:** Fresh64: Full 85.94%/27.984 s; Stable1 82.81%/31.060 s; Sketch 67.19%/25.796 s. No quality-preserving acceleration demonstrated.
- **Resources / accounting:** Model step/gate/Python timing scopes, CPU/MLX peak, tokens, counted attention/MLP and estimated FLOPs.
- **Comparison:** Full, Stable1 and Adaptive-Sketch with common conditions.
- **Failures / limitations:** Small tasks, reused outputs and API/kernel bookkeeping; no measured GPU-time/power saving.
- **Finding for subsequent research:** Profile actual runtime overhead independently of analytical model FLOPs.
- **Evidence:** [report](../data/stable_runtime_7437bcc4afc44d6d958ba98ed2f41628/ANALYSIS_91160fd91aef42cb8177350d31138a01/STABLE_RUNTIME_REPORT.md), [implementation](../src/stable_runtime_engine.py).

## Adaptive Computation Stop (ACS)

- **Goal / hypothesis:** Stop completed answers without truncating required content.
- **Implemented method:** Fixed caps, structural boundaries, real token-probability queries and request-aware stop rules.
- **Models / hardware:** Native local Qwen2.5-Instruct BF16 via MLX, Apple M4/16 GiB; revisions/packages in the manifest. Not directly comparable to Ollama Q4 timings.
- **Data / repetitions:** Existing64/fresh64,3 repeats, six conditions including two cap controls.
- **Accuracy / actual timing:** Fresh limited fact coverage 82.81%→82.81%; wall 39.687→36.424 s. This is not full semantic correctness.
- **Resources / accounting:** Real tokens, process CPU, allocator peak, gate/probability overhead and separate literal/audited scores.
- **Comparison:** All28 layers retained; generation stopping changes output work.
- **Failures / limitations:** Doctor/physician scoring bug, Dr. truncation, unrequested additions and intentional stress-case bias.
- **Finding for subsequent research:** A high next-token probability does not prove task completion.
- **Evidence:** [report](../data/acs_16a7001279c947c0acb730672deb1a97/ANALYSIS_f4146c33e0184a62aef399eaaaa7f53d/ACS_RESEARCH_REPORT.md), [implementation](../src/acs_runtime.py).

## DLR / temporal activation reuse

- **Goal / hypothesis:** Reuse similar adjacent-token MLP residuals while updating attention/KV.
- **Implemented method:** Four candidate MLPs, relative-L2 gate, cached fresh anchors and one-step cooldown.
- **Models / hardware:** Native local Qwen2.5-Instruct BF16 via MLX, Apple M4/16 GiB; revisions/packages in the manifest. Not directly comparable to Ollama Q4 timings.
- **Data / repetitions:** Existing32/fresh32,3 repeats; equal-stream subset,960 primary requests.
- **Accuracy / actual timing:** Fresh DLR 84.38% content accuracy, 13.273 s total; slower than Full despite actual reuse.
- **Resources / accounting:** Actual MLP call avoidance, CPU/wall, gate synchronization and analytical matmul savings.
- **Comparison:** Baseline, Static, DLR and Conservative.
- **Failures / limitations:** Very sparse reuse, host synchronization and cache cost; no broadly valid latency improvement.
- **Finding for subsequent research:** This is activation reuse, not an independently established new invention.
- **Evidence:** [report](../data/dlr_5a4656b8d37f48dd8a55a0e82b647b37/ANALYSIS_a61966a5b3314aa38e3b4b54517e94f9/DLR_RESEARCH_REPORT.md), [implementation](../src/dlr_runtime.py).

## Device-Resident / Lightweight DLR

- **Goal / hypothesis:** Reduce gate synchronization and similarity overhead.
- **Implemented method:** True conditional Metal projection branches; cheap observed-token-ID proxy; native controls.
- **Models / hardware:** Native local Qwen2.5-Instruct BF16 via MLX, Apple M4/16 GiB; revisions/packages in the manifest. Not directly comparable to Ollama Q4 timings.
- **Data / repetitions:** Existing32/fresh32 plus8 long tasks,3 repeats, natural+fixed1728 primary requests.
- **Accuracy / actual timing:** CSV-derived README table; no method met the accuracy and matched-stream timing goal.
- **Resources / accounting:** GPU branch counters/software controls, host scalar reads, tokens, wall/CPU and MLX peak.
- **Comparison:** Baseline and original DLR, with separate fresh-kernel fidelity checks.
- **Failures / limitations:** Fresh kernel rounding confound; lightweight counting/EOS damage; offsetting correct/wrong cases.
- **Finding for subsequent research:** Removing host reads helps a component but does not establish joint efficiency.
- **Evidence:** [report](../data/dlr_opt_2c8a76633b5e450f88242020fce6a899/ANALYSIS_66d8ddb5e3e247a4be9489dcad563887/DLR_OPTIMIZATION_REPORT.md), [implementation](../src/dlr_opt_kernels.py).

## Zero-Overhead Oracle

- **Goal / hypothesis:** Measure a privileged scheduling ceiling before inventing another gate.
- **Implemented method:** Frozen best-of-three label/future-aware tables; quota/cooldown-matched Random; real added gate/cache overhead.
- **Models / hardware:** Native local Qwen2.5-Instruct BF16 via MLX, Apple M4/16 GiB; revisions/packages in the manifest. Not directly comparable to Ollama Q4 timings.
- **Data / repetitions:** Existing8/fresh8/long4, five repeats, zero-to-half decode-MLP targets,4000 primary requests.
- **Accuracy / actual timing:** README shows representative natural outcomes. Higher targets shorten matched-stream time but damage accuracy; prespecified joint classification C.
- **Resources / accounting:** Actual native MLP bypass, phase-synchronized diagnostic, CPU/tokens/memory and separate table-generation cost.
- **Comparison:** Baseline, Existing, Ideal, Random and Oracle+Actual Overhead.
- **Failures / limitations:** Uses evaluation labels/future activations; not globally optimal/deployable, not a proof every DLR is impossible.
- **Finding for subsequent research:** Small quality-preserving gains do not establish the specified meaningful benefit; prioritize rigorous native/runtime baselines.
- **Evidence:** [report](../data/oracle_8ef05974726245a1b273f178fc668cb7/ANALYSIS_04d1920008d24b7eba424bc4c451b5f4/ORACLE_EXPERIMENT_REPORT.md), [implementation](../src/oracle_experiment.py).

## Complete report inventory

- [ACS_RESEARCH_REPORT.md](../data/acs_16a7001279c947c0acb730672deb1a97/ANALYSIS_f4146c33e0184a62aef399eaaaa7f53d/ACS_RESEARCH_REPORT.md)
- [ALSR_RESEARCH_REPORT.md](../data/alsr_1c1ea4a321cd4b78859d3ac33f4aba5f/ALSR_RESEARCH_REPORT.md)
- [ALSR_LAYER_IMPACT_REPORT.md](../data/alsr_impact_e8eea7cd19db4d8796ea54cfe6f57b2c/CANONICAL_aca75ae9b3014233ba48c949c0186bb4/ALSR_LAYER_IMPACT_REPORT.md)
- [controlled_1ee08ac18eaa428f9644ef66daa2fcaa_failure_report.md](../data/controlled_1ee08ac18eaa428f9644ef66daa2fcaa_failure_report.md)
- [coverage_951686cca01344ed8b732fc1e66d7cab_assessment.md](../data/coverage_951686cca01344ed8b732fc1e66d7cab_assessment.md)
- [coverage_951686cca01344ed8b732fc1e66d7cab_report.md](../data/coverage_951686cca01344ed8b732fc1e66d7cab_report.md)
- [DLR_RESEARCH_REPORT.md](../data/dlr_5a4656b8d37f48dd8a55a0e82b647b37/ANALYSIS_a61966a5b3314aa38e3b4b54517e94f9/DLR_RESEARCH_REPORT.md)
- [DLR_TEST_RESULTS.md](../data/dlr_5a4656b8d37f48dd8a55a0e82b647b37/ANALYSIS_a61966a5b3314aa38e3b4b54517e94f9/DLR_TEST_RESULTS.md)
- [DLR_OPTIMIZATION_REPORT.md](../data/dlr_opt_2c8a76633b5e450f88242020fce6a899/ANALYSIS_66d8ddb5e3e247a4be9489dcad563887/DLR_OPTIMIZATION_REPORT.md)
- [efficiency_repeat_1b6d963181b249e0889270ca4590ea2a_assessment.md](../data/efficiency_repeat_1b6d963181b249e0889270ca4590ea2a_assessment.md)
- [efficiency_repeat_1b6d963181b249e0889270ca4590ea2a_report.md](../data/efficiency_repeat_1b6d963181b249e0889270ca4590ea2a_report.md)
- [mixed_5f6833b806564466821eb9f81873848f_assessment.md](../data/mixed_5f6833b806564466821eb9f81873848f_assessment.md)
- [mixed_5f6833b806564466821eb9f81873848f_report.md](../data/mixed_5f6833b806564466821eb9f81873848f_report.md)
- [mixed_reliability_318dcf5fdf2045179fd24fd3a25866fa_assessment.md](../data/mixed_reliability_318dcf5fdf2045179fd24fd3a25866fa_assessment.md)
- [mixed_reliability_318dcf5fdf2045179fd24fd3a25866fa_report.md](../data/mixed_reliability_318dcf5fdf2045179fd24fd3a25866fa_report.md)
- [ORACLE_EXPERIMENT_REPORT.md](../data/oracle_8ef05974726245a1b273f178fc668cb7/ANALYSIS_04d1920008d24b7eba424bc4c451b5f4/ORACLE_EXPERIMENT_REPORT.md)
- [PVCG_RESEARCH_REPORT.md](../data/pvcg_f4de31fe19654eaabb9a17700f3d6a26/PVCG_RESEARCH_REPORT.md)
- [rcav_c038afc6c5df422aa9e58080a4fa5847_assessment.md](../data/rcav_c038afc6c5df422aa9e58080a4fa5847_assessment.md)
- [rcav_c038afc6c5df422aa9e58080a4fa5847_report.md](../data/rcav_c038afc6c5df422aa9e58080a4fa5847_report.md)
- [rcav_guard_b2e19463ef974c54a65b0770f2cf475d_assessment.md](../data/rcav_guard_b2e19463ef974c54a65b0770f2cf475d_assessment.md)
- [rcav_guard_b2e19463ef974c54a65b0770f2cf475d_report.md](../data/rcav_guard_b2e19463ef974c54a65b0770f2cf475d_report.md)
- [skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_assessment.md](../data/skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_assessment.md)
- [skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_report.md](../data/skipwork_7a7cc6bdb72e495ab9af664ba5b15db7_report.md)
- [solver300_c22b83397f0945b4adfb8304f7b2f268_assessment.md](../data/solver300_c22b83397f0945b4adfb8304f7b2f268_assessment.md)
- [solver300_c22b83397f0945b4adfb8304f7b2f268_report.md](../data/solver300_c22b83397f0945b4adfb8304f7b2f268_report.md)
- [STABLE_RUNTIME_REPORT.md](../data/stable_runtime_7437bcc4afc44d6d958ba98ed2f41628/ANALYSIS_91160fd91aef42cb8177350d31138a01/STABLE_RUNTIME_REPORT.md)
- [tool_routing_73d749e404ea4347bbba5883b485a75e_assessment.md](../data/tool_routing_73d749e404ea4347bbba5883b485a75e_assessment.md)
- [tool_routing_73d749e404ea4347bbba5883b485a75e_report.md](../data/tool_routing_73d749e404ea4347bbba5883b485a75e_report.md)
- [verification200_faeda373c1e6443f9775a4adb75be465_audited_report.md](../data/verification200_faeda373c1e6443f9775a4adb75be465_audited_report.md)
- [verification200_faeda373c1e6443f9775a4adb75be465_report.md](../data/verification200_faeda373c1e6443f9775a4adb75be465_report.md)
- [VRG_RESEARCH_REPORT.md](../data/vrg_6828c214e04841e1ab6ca4e5392de9a9/VRG_RESEARCH_REPORT.md)
- [vvp_2a90be1867974c2cbc6862856d207d44_content_audit_report.md](../data/vvp_2a90be1867974c2cbc6862856d207d44_content_audit_report.md)
- [vvp_2a90be1867974c2cbc6862856d207d44_failure_report.md](../data/vvp_2a90be1867974c2cbc6862856d207d44_failure_report.md)
