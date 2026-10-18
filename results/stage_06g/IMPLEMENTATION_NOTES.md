# Stage 06G implementation notes

Implementation status: written for independent review. No actual cached probabilities have been evaluated by this implementation locally. No outcome is implied by the implementation or its synthetic fixtures.

`scripts/stage06g_pair_complement.py` implements exactly the frozen `plans/STAGE_06G_PAIR_COMPLEMENT.md` candidate. It imports the unchanged Stage 06F module only after checking its SHA-256 `a8bb68181087640fb563539870840117531a4d57e529cbeaf69e73d4c0a05be7`, and calls only `_load_inputs` and `_source_gate`. It does not call the F diagnostic or its bootstrap. Those unchanged gates bind the original official train, E output bytes, alignment, labels, prompts, folds, original-order evidence replay, source scores, confusion matrices and historical counts. The G plan hash is `ece986eb6244bd322564cd7fd7a437239e5f74635a2cea2b9a39968bd891b927`.

## Entry points and execution freeze

- `run_complement(input_root, output_root, freeze_path)` is the production entry point. A freeze is mandatory and the output directory must be absent or empty and outside the immutable input root. Failure writes `status.json` with `FAILED_PRESERVE_OUTPUTS` and re-raises; it never launches another attempt.
- CLI variables are `NTL_STAGE06G_FREEZE` (required), `NTL_STAGE06G_INPUT` (default `/kaggle/input`), and `NTL_STAGE06G_OUTPUT` (default `/kaggle/working/stage06g_pair_complement`).
- Freeze keys are `implementation_source_sha256`, `source_gate_source_sha256`, `plan_sha256`, `input_version_ids`, `accelerator`, `internet_enabled`; optional explicit `source_gate_path` and `plan_path` locate packaged source files. The source hashes and plan hash must match the actual files; F and plan hashes also match hardcoded reviewed values. Inputs must be `[170484,359514801]`, accelerator `none`, internet disabled. Authoritative notebook source metadata verifies those configuration declarations before execution.
- `derive_candidate(classes, evidence, source_prediction, qwen_probability, deb_probability)` is a pure synthetic-testable helper. It applies whole-matrix native-dtype `log(clip(P,1e-9,1))`, exactly ordered equal pair-margin addition, stable descending evidence top-two routing and native positive/negative/exact-zero handling. It never consumes ground truth.
- `comparison_invariants` independently reconciles source/candidate confusion matrices, scoped changes, unaffected entries and other-class correct indicators. `evaluate_candidate` applies the immutable actual-data scope count assertions and all complete-development endpoints. `paired_prompt_interval` reproduces the inherited D/E joint fixed-prediction prompt bootstrap; production uses 3,000 draws and seed 20261017.

## Evaluation accounting

The two target supports are exactly 150, and all other class recalls are structurally unchanged. Consequently each complete 18-class balanced-accuracy gain is asserted equal to `(corrected-broken)/2700`, and the three-context primary mean equals total net corrections divided by 8100. The denominator is not 2699 row accuracy. Counts of changed predictions also include scoped nonpair changes that remain wrong. Both false-entry destinations are reported for every other true class.

The single inherited screen is strictly positive mean gain, strictly positive lower conditional 95% bound and every context gain at least -0.005, after integrity gates pass. No class-specific veto, extra threshold, alternate member, ensemble weight or second candidate is tested. Class recall harms are reported plainly.

## Outputs and recovery

Terminal result/status are `pair_complement_result.json` and `status.json`, status `COMPLETE_FIXED_ATTENTION_PAIR_COMPLEMENT`. The result distinguishes the unchanged official S02 0.549 from this incomplete reused-development proxy. It records source and candidate raw balanced accuracy, all context changes, Gemini recalls, conditional uncertainty, screen outcome, source freeze receipt, input gates, array hashes and zero fit/inference/regrouping counters.

`pair_complement_arrays.npz` stores only numeric/Unicode arrays: aligned IDs/classes/labels/prompts/folds, unchanged source predictions, the one separate candidate prediction, top-two indices, fixed scope, exact native pair margins, masks and complete confusion matrices. Every array is read back with `allow_pickle=False`, checked for identical dtype and values, and checked against a dtype/shape/bytes SHA-256. The source evidence, saved probabilities and predictions are hash-checked as unchanged across evaluation. No passage text is persisted or printed.

CSV outputs cover all 18 class recalls, complete context summaries, folds, prompts, other-class pair destinations, scoped outcome accounting and all six complete confusion matrices. All aggregate CSVs, source gate, result and manifest are emitted as machine-readable log records. A small Matplotlib `context_gain.png` shows context gains and corrected/broken counts. `output_manifest.json` binds every output other than itself, including final status; the chart and NPZ remain output artifacts.

These are conditional reused-development results. Contexts repeat the same passages. The bootstrap does not include refitting, scope-selection or future-data uncertainty. A positive result would justify only the next matched full-system investigation; a negative result is retained without an automatic variant or retry. No official submission, final-selection change, GPU, paid service, model inference, fitting, optimization or batch balancing is implemented.
