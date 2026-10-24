# Stage 09 — matched public-data pretraining pilot

Prospective protocol, 24 October 2026. No Stage 09 model has been fitted when this protocol is written. The account's verified official best is S03 / Stage 08C V2 (361732127), 0.620 public balanced accuracy. Stage 09 produces development evidence only.

## Question

Does authorship pretraining on the already reviewed, own-prompt direct-OpenAI subset improve a small encoder after the same official-data fine-tuning? This deliberately narrow pilot tests the whole staged procedure, including its extra pretraining compute. It does not isolate data value from extra compute, reproduce the upstream 83,827-row recipe, or measure a contribution to the S03 ensemble.

The prior Stage 04B recipe achieved only 22.7639% grouped development balanced accuracy. Repeating that recipe is unjustified. The new arms both use an attention-mask mean of the last encoder layer and a single linear classifier, with separate encoder/head learning rates, motivated by section 3 of Nishant Dahal's frozen V1 361311282. This is a new base-size pilot, not an exact reproduction of the author's large encoder.

## Data and split, fixed before fitting

- Official train: 2,699 rows / 150 prompt groups / 18 labels; SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- Use only fold 1 of the existing five GroupKFold splits: 2,159 training rows / 120 prompts; 540 validation rows / 30 prompts; all 18 labels in both. Validate the ordered validation-ID hash and the saved resource reference. This fold is reused development data, not an untouched holdout.
- External: prvsiyan's Extended Name That LLM Dataset V4, `new` split, direct `openai` route only. Exactly 750 responses in 150 complete five-GPT prompt blocks, 150 per label. No proxy route, reconstructed official prompt or `train_regen` row is admitted.
- Re-run `stage09_data_audit.py` on the actual mounted inputs. Require `PASS_FROZEN_PILOT`, exact immutable source hashes and the frozen admission fingerprint. Map every prompt to its actual source declaration and raw request envelope. Drop whole external groups on failed origin/response screens; the frozen run additionally refuses a changed admitted corpus.
- The executed CPU audit passed 13 contract/failure checks, matched all 750 calls, and found no exact/normalized/substring/shared-8gram/high-cosine response overlap. These checks do not certify semantic prompt independence; the official prompt wording is unavailable and source declarations are publisher supplied.

## Fixed arms

Use untouched `microsoft/deberta-v3-base` revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, from private Stage 04B Resources V1 359317729. Hash the full original weight, configs, tokenizer and dependency wheels against the manifest. Never load any S03 adapted checkpoint.

| Setting | Control | Candidate |
|---|---|---|
| Initial encoder and linear head | Same vendor encoder / seeded head | Same vendor encoder / seeded head |
| External pretraining | None | Two full epochs of the 750 admitted rows |
| Pretraining encoder/head LR | — | 0.000015 / 0.001 |
| Before official training | Reset head to the initial seeded state | Reset head to that identical state; retain pretrained encoder |
| Official training | Same 2,159 rows, three complete epochs | Identical rows, order, loss and schedule |
| Official encoder/head LR | 0.00001 / 0.0005 | 0.00001 / 0.0005 |
| Evaluation | Final epoch only | Final epoch only |

Seed 20261024. Max length 384 including specials, slow SentencePiece tokenizer, right truncation, dynamic padding to a multiple of 8. Microbatch 4, accumulation 4; sum weighted losses divided by actual example count in each accumulation group. The final short group is never divided by a nominal full-batch count. External full-epoch shuffling without replacement preserves equal five-class exposure; official class weights use only the official training split and normalize inverse support to mean one. All 18 labels remain in the classifier.

AdamW betas (0.9,0.999), eps 1e-8, weight decay 0.01 except bias/normalization parameters; separate encoder/head rate groups; linear decay with 10% rounded-up warmup, norm clipping 1.0. Fresh optimizer, scheduler, AMP scaler and official-phase RNG/loading schedule for each arm. FP32 master weights and FP16 autocast on T4; scaler starts at 1024. Abort on nonfinite loss/gradients, skipped optimizer updates or OOM; no silent batch/length/model changes.

Expected updates: external 94; official 405 per arm. No early stopping, validation-driven epoch/rate selection, new seed, ensemble or parameter search.

## Execution budget and evidence

Kaggle included compute only, Internet off. Maximum three allocated GPU-hours across the attempt: on T4 x2 this is at most 5,400 wall seconds, with time reserved for preservation. Record actual environment and allocation. No paid API, new provider generation, competition submission or final-selection change belongs to Stage 09.

Save complete-phase checkpoints with an immutable freeze/hash receipt. An incomplete phase is a failed attempt, never silently treated as a completed shortened run. Any later retry or resume must retain prior failure and completed-phase evidence and obey the cumulative budget.

Keep original row IDs, prompt IDs, labels/predictions/probabilities and model weights private. Publish code, source/version/hash receipts, aggregate confusion/per-class metrics, interval, timing, status and external-data identifiers only. Public repository visibility does not authorize uploading competition text or adapted weights.

## Measurement and decision

After both final epochs, require finite, bounded 540 x 18 probabilities, canonical sorted label order, normalized rows, exact unique ordered validation identity, and prediction/argmax parity. Report both development balanced accuracies, candidate-minus-control difference, all per-class recalls/confusions, corrected/broken decisions, changed predictions and truncation incidence.

Use paired resampling of the same 30 validation prompt groups for a conditional 95% interval, 10,000 draws with seed 20261024. The interval excludes refitting and adaptive experiment-selection uncertainty. A positive difference with lower endpoint above zero and no technical validation failure passes the pilot screen; report every class regression even if it passes. A pass earns broader declared grouped evaluation, not a new official score or a claim that 62.0% has improved. A failure retains the control and does not trigger an automatic weight/rate search.

## Sources

- S03 submission: https://www.kaggle.com/competitions/name-that-llm/submissions
- Method inspiration: https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two?scriptVersionId=361311282
- Reviewed external source: https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset (V4, source-declared CC BY 4.0; attribution retained)
- Immutable vendor resources: https://www.kaggle.com/code/watchman777/name-that-llm-stage-04b-resources?scriptVersionId=359317729

The author's current larger-pool recipe supplies motivation only; its source-trained features cannot provide honest S03 OOF predictions.
