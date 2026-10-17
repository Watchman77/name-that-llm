# Stage 06: audit of the proposed route to 70%
Date: 2026-10-17. Status: source audit and one public LoRA prediction-artifact audit complete. Full S02 reconstruction and centering experiment remain pending. See REPORT.md.

## Verified starting point
Our unchanged S02 reproduction scored 0.549 official public balanced accuracy. S01 scored 0.354: +19.5 percentage points. At this live check, the leader is 0.600, tenth 0.559, and Watchman 777 is rank 181/1212 with two entries. These standings can change; final private results are unknown.
- Official leaderboard: https://www.kaggle.com/competitions/name-that-llm/leaderboard
- Own S02 V1: https://www.kaggle.com/code/watchman777/name-that-llm-stage-05-public-reproduction?scriptVersionId=359332774
- Exact source V3: https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering?scriptVersionId=359126546

70% is +15.1 percentage points above S02, equivalent to reducing remaining mean class error by 15.1/45.1 = 33.48%. This is a goal, not an evidence-backed forecast. Claims that 58–60% is very achievable or 62–65% likely are not established by our experiments.

## Advice compared with actual executed code
Audit basis: vendor/public_549_v3.py, actual code-cell SHA256 e3d97306f3185ea5f73d7904d81331b27803e3ceb45970596590ec306969762e (archive adds one terminal newline).
| Proposal | Existing S02 implementation | Genuine remaining question |
|---|---|---|
| Style and character diversity | Word/character/shape/function n-grams, stylometric LightGBM | Are additional features complementary on held-out prompts? |
| Larger representations | Qwen/GLM frozen features, DeBERTa-large, ELECTRA-large, desklib detector | Which new representation fixes errors others share? |
| Fine-tuning / LoRA | Two independently trained LoRA24 classifiers, UltraBERT fold ensemble | Additional training must earn its cost through complementary errors. |
| Token style pooling | Function-word/punctuation pooling and two learned token-attention heads | A particular missing signal, not generic more pooling. |
| Prompt centering | Mixed lexical/GLM clustering, five frozen centered twins and one LoRA centered twin | Cluster quality, small-group mismatch, robustness. |
| Class balancing | Sinkhorn-style marginal balancing, alpha 0.5 | Validate changed distribution assumptions rather than add another arbitrary correction. |
| Error analysis | We have Stage 03 diagnostics, not verified full S02 OOF diagnostics | Build an honest component/ensemble validation inventory first. |

The four frozen LLM plain weights are zero but their centered twins contribute. Reading only the plain-weight dictionary would miss these active components.

## Public OOF lead inspected
Public source: https://www.kaggle.com/code/lightningv08/ntl-lora24-fivefold-training?scriptVersionId=358831649
Its Copy/Edit version matches the kernel output attached to S02. Apache 2.0 is displayed. Public Output lists lora24_oof.npz and lora24_comparison.json plus outer/final checkpoints. The training and evaluation scripts are embedded in its visible first code cell.

Static review of train_lora24.py shows:
- Five outer fits and a separate final fit; explicit disjoint fit/validation rows and prompt groups.
- Fixed three epochs; validation labels do not select epochs/checkpoints in this implementation.
- Separate validation_features.npz and readouts.npz per job, train-data digest and class metadata.
- Centered-head training uses same-prompt triples; this differs from the frozen heads fitted on complete prompt groups.

Static review of evaluate_lora24.py shows:
- SHA256 validation of train.csv and reference inputs; explicit IDs/classes/labels/groups checks.
- Reconstructs plain and centered log probabilities from each outer checkpoint's saved validation features and heads.
- Requires each training row be covered exactly once; skips the final fit.
- Three context scenarios (1,3,6) with inferred lexical groups and singleton plain fallback.
- Public mixtures reference a separate Gemma baseline and exclude UltraBERT. They are NOT full S02 OOF predictions.
- Report explicitly calls three fixed mixture comparisons descriptive and notes development data reuse.

Update: private CPU V1 359394412 has now independently reconstructed this component's numeric predictions from the saved outer feature banks and heads; all values and component metrics matched. The earlier source-only limitation is resolved for this component. See REPORT.md and LOra_COMPONENT_PROVENANCE.json. Limitations: A private helper input appears in the source notebook; embedded code is public, but complete training reproducibility/dependency provenance is not yet certified. No published score is silently promoted to our own result.

## Evaluation corrections
1. Full-fit adapters trained on all 2,699 rows cannot generate honest training OOF predictions. Use actual outer checkpoints. Fixed predeclared averaging can combine different fold schemes if every member holds out each scored row's entire prompt; learned weights/stacking require a shared outer evaluation with all component fits excluding its scoring prompts.
2. Hidden test labels are unavailable. Claimed current ensemble confusion pairs remain unverified. Stage 03 errors are not evidence of S02's current hardest pairs.
3. Learn vocabulary, scalers, supervised features, heads, blend weights, calibration and gates without scoring-fold labels. Fit a meta-model inside outer training prompts; scoring it on the same OOF rows used to tune it is optimistic.
4. A single positive temperature applied to final logits preserves argmax and cannot alone improve balanced accuracy. Member-wise calibration or calibration before balancing may change decisions and requires held-out evaluation.
5. There are only 150 prompts and about 150 examples per class. Isotonic calibration and pair specialists can overfit.
6. Report per-class regressions and uncertainty. Requiring literally no per-class regression is not the competition objective; compare mean recall and severity of tradeoffs.
7. Our failed supervised DeBERTa trial (22.76% development BA) rules out that fixed recipe as a replacement, not every transformer or every possible complementary blend.
8. All existing folds are development data; nested comparisons improve discipline but do not undo historical adaptation to this dataset.

Next protocol: plans/STAGE_06_OOF_AND_CENTERING.md. No new GPU run, paid service or official submission was launched by this audit.
