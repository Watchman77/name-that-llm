# Stage 06G: reject the fixed attention-pair correction

The single predeclared correction failed its improvement screen. [Private Kaggle V2 359531626](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06g-pair-complement?scriptVersionId=359531626) completed successfully in 22.5 seconds on CPU with internet disabled. Source, freeze, scope, numerical accounting and stored-array readback gates passed. The rule ran as intended; its measured development performance did not justify adoption. **Retain the unchanged control and retire this exact rule. Official S02 remains separately confirmed at 54.9%.**

## The recipe that was tested

Only when the unchanged raw-control ensemble's top two labels were exactly Gemini 3.7 Flash and Gemini 3.8 Flash, replace the pair choice with the sign of the equal sum of the saved Qwen and DeBERTa token-attention log-probability differences. Each member coefficient was 1.0, clipping was `[1e-9,1]`, the addition order and stable top-two gate were fixed, exact-zero ties retained control, and all outside-scope predictions were preserved. There were no exact-zero margins in the actual data. No fitting, model inference, regrouping, balancing or parameter search occurred.

This result rejects adoption of that precise equal-sum/top-two recipe on the available evidence. It does not prove that attention evidence, pair models or all possible corrections are useless. It supplies no license to tune another weight or threshold on the same result.

## Complete 18-class raw balanced accuracy

| Context | Control | Candidate | Change, percentage points | Corrected | Broken | Changed, both still wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 52.638081% | 52.489933% | -0.148148 | 26 | 30 | 7 |
| 3 | 55.306736% | 54.973403% | -0.333333 | 27 | 36 | 6 |
| 6 | 56.491921% | 56.528958% | +0.037037 | 29 | 28 | 5 |
| Mean / repeated-context totals | **54.812246%** | **54.664098%** | **-0.148148** | **82** | **94** | **18** |

The 3,000-draw paired-prompt bootstrap, seed 20261017, gives a conditional 95% interval of **-0.641975 to +0.358025 percentage points** for the mean difference. The interval includes zero. Both the positive-mean and positive-lower-bound requirements fail; every context remains above the predeclared -0.5-point floor. The point estimate is adverse, but this conditional interval does not establish a population-wide deterioration.

These are the same 2,699 passages reused across three development contexts, not 8,097 independent examples. The 82 corrected and 94 broken predictions are repeated-context observations. Their net -12 reconciles to the complete balanced-accuracy change as `-12/8100`, because each of the two affected classes has support 150 and the other sixteen class recalls are invariant. This is balanced accuracy, not row accuracy divided by 2,699.

The scope comprised 193, 196 and 190 rows. Of the 194 total changed context predictions, 176 were corrected or broken true-pair predictions and 18 were nonpair passages switched between two wrong Gemini destinations. Pair-entry masks remained identical. All sixteen other class recalls, outside-scope decisions and true-pair rows originally assigned another class remained exactly unchanged. The small positive context-6 change cannot override the failed primary mean and interval criteria.

## The class tradeoff explains the negative result

| Model | Mean control recall | Mean candidate recall | Mean change, percentage points | Corrected / broken, repeated contexts |
| --- | ---: | ---: | ---: | ---: |
| Gemini 3.7 Flash | 38.666667% | 45.555556% | +6.888889 | 50 / 19 |
| Gemini 3.8 Flash | 54.666667% | 45.111111% | -9.555556 | 32 / 75 |

The correction recovered more Gemini 3.7 cases, but broke substantially more correct Gemini 3.8 decisions. It redistributed the pair's recall and lost twelve net correct context predictions. The earlier attention-member disagreement counts were therefore an opportunity to test, not a measured reservoir of safe fixes. This is the exact risk Stage 06F identified: a member can oppose existing mistakes and also oppose correct answers.

The result remains an incomplete available-component proxy, not a new official public score or full-S02 comparison. Pair, members and routing were chosen after previous development diagnostics. The bootstrap fixes all learned evidence and routing and omits refitting, scope-selection and future-data uncertainty. The full numeric candidate/source arrays remain on immutable Kaggle output unless independently retrieved; archived aggregate/log hashes and production exact-readback gates do not amount to a second-machine row-level recomputation.

## Recommended next direction: representation complementarity, with provenance first

Stop iterating on this Gemini correction. The next substantive prerequisite is to determine which **missing representation supplies different prompt-held-out evidence across all eighteen classes**, rather than assume that another pair gate or a larger model will improve the ensemble. This is a recommendation for a prerequisite audit, not a frozen new experiment, a training authorization or a promise of a gain.

The existing inventory needs to be reconciled with completed work before choosing that representation. `results/stage_06/COMPONENT_INVENTORY.json` is a historical 17 October snapshot: its GLM-cache gap was resolved in Stage 06C, and stylegram was reproduced for Stage 06D/E. The current Stage 06E proxy contains stylegram, two plain LoRA outputs, two token-attention outputs, and frozen DeBERTa plain/centered evidence. It does not reproduce full S02. Its documented gaps include the other frozen centered representations, compatible LoRA centered evidence, the LightGBM branch and adequately verified held-out fine-tuned encoder evidence.

Here, missing means absent from the development proxy. GLM and ELECTRA already contribute to full S02. Reproducing them would close baseline coverage and improve the reference for testing future changes; it would not introduce a novel model or establish improvement over official 54.9%. A genuinely novel competitive complement has not been identified by this audit.

Two concrete cached leads already recorded in this repository merit provenance review before any new extraction: the completed own GLM layer 8/3/24 cache, and the ELECTRA-large mean/style cache located in pinned encoder output 358745836. Existing GLM features supported mixed grouping; their availability does not mean a matched GLM classifier has been evaluated in this proxy. The ELECTRA cache was located historically but has not been validated here for a new head. Neither is presented as a proven complementary model. The Qwen frozen caches and desklib passage cache were not located in the recorded inventory, and UltraBERT outer-prompt/checkpoint provenance remains unresolved; final-fit or full-ensemble inference on training rows cannot fill those held-out gaps.

The useful audit deliverable is one source-backed coverage table: exact representation/checkpoint and source version; cache hash and row/class alignment; pooling/truncation/precision; whether every proposed supervised prediction excludes its entire scored prompt; compatibility with common context batches; existing native output coverage; and the specific extraction or validation gap. Native outputs already available may describe shared errors, but all-case disagreement or oracle recovery is not an adoptable gain. No learned stacking should be fitted to the globally reused OOF table.

Only after that audit should one missing, feasible representation receive a separately frozen matched experiment, if warranted. Its primary endpoint must remain marginal full-18-class ensemble benefit, not its own component score. Stage 06E already showed that component improvement need not improve this ensemble; Stage 06G now shows that apparent error disagreement need not yield a successful correction. No new candidate, model availability claim, public-source lookup, training run, submission or follow-on plan was created as part of these findings. There is no evidence here that 70% is within reach.

## Evidence

The frozen prospective definition is `plans/STAGE_06G_PAIR_COMPLEMENT.md`. Actual aggregate outcomes are `pair_complement_result.json`, `context_summary.csv`, `class_recall.csv`, `fold_scores.csv`, `prompt_counts.csv`, `scope_outcomes.csv`, `other_class_false_entries.csv` and six complete confusion matrices. Source/freeze checks are in `source_gate.json`; runtime bindings are in `output_manifest.json` and `KAGGLE_V2.log`. Component-coverage provenance comes from `results/stage_06/COMPONENT_INVENTORY.json`, `results/stage_06/LOra_COMPONENT_PROVENANCE.json`, `results/stage_06b/REPORT.md`, `results/stage_06c/GLM_CACHE_COMPLETE.json`, `plans/STAGE_06D_FIXED_ENSEMBLE_PROXY.md` and `results/stage_06e/REPORT.md`.
