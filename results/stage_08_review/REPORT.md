# Stage 08 — review and preparation of the public 0.600 recipe
**24 October update:** The real-input audit is complete and the unchanged recipe is blocked by a missing private LoRA24 checkpoint and changed old-merge counts. See [UPDATE_20261024.md](UPDATE_20261024.md). The earlier preparation notes below are retained as the 20 October record.

Reviewed on 20 October 2026. **V20 is a technically plausible reproduction target, with a real author public score displayed as 0.600. Our reproduction has not executed model code and has no score.** Source review is complete; exact input receipts and the real-input preflight remain prerequisites to GPU execution. Our last verified official S02 score is still 0.549; it was not freshly rechecked here.

## Version correction and observed evidence

The earlier search link pointed to `359708344`, **V14**, whose own public score is **0.557** despite the notebook's current 0.600 title. Its best-score link identifies **V20, 360088024**, which was opened directly and displays **Public Score 0.600**, Version 20 of 20. V14 also showed two unavailable datasets; that finding must not be attributed automatically to V20.

- Author V20: https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt?scriptVersionId=360088024
- V20's visible preview completed in **45m39s on two T4 GPUs**, with 2,699 train rows and **three example test rows**. That preview is separate from the author's displayed public score and from any result in our account.
- The source lists 23 attachments: one competition, nine datasets, four base-model versions and nine notebook outputs. The public input explorer reports **42.19 GB**. No missing-dataset marker was observed in V20's list.
- `nishantdahal/ntl-two-stage-lora` is visible at dataset version 1. Its explorer exposes `pool_v1.parquet`, `lora8b_v1/` and `lora14b_v1/`; both adapter directories, metadata and classifier-head files are listed. The 8B metadata preview has the expected 18-label order, keep=24, eight pooled layers and max_len=447. File listing/metadata availability does not establish binary checksums or independent training provenance.

## What this changes

The main branch trains a four-view n-gram SVC on official data plus a declared **34,940-row public pool**. Two published Qwen3 **8B and 14B classification adapters** were reportedly trained first on the public pool, then on official data. A shared-prompt Gaussian classifier on new 8B features contributes to inferred groups of 2–18 passages. Their complete probabilities are mixed 85/15 with an older full feature ensemble containing four other Gaussian heads. A single final full Sinkhorn balance is applied for larger test sets.

This is materially different from our Stage 07B character/style pilot and its failed Stage 07C added-vote diagnostic. It does not require us to buy generation API access or retrain the large adapters to test the published inference recipe. GPU runtime is still required for inference, plus CPU fitting of the recipe's readouts/style branches; the three-row runtime is not a forecast for the hidden test.

Two separate reviews found no obvious source repair blocker. `SOURCE_CODE_REVIEW.md` inventories mandatory files, model/config contracts, software compatibility and silent fallback risks. `DATA_REVIEW.md` distinguishes the declared 34,940-row pool, the older branch's reported 7,644-row merge, published final adapters and unverified training/CV assertions. The source has one complete Python code cell plus an embedded 889-line older stack. We parsed both without executing them.

## Private preparation and a caught input change

A private copy was created under watchman777 as **Name That LLM - Stage 08 Public 600 Reproduction**. The complete editor code was copied through the UI and compared with all **71,625 characters** of the rendered V20 code: exact match. The first Quick Save is **V1 360471311**. Its source has no execution counts/output and the draft session was off. The saved viewer confirms Private and a +0/-0 source diff.

However, copying inherited **latest** dataset versions. Our V1 explorer reported **42.22 GB**, while the author V20 reports 42.19 GB. A direct comparison found one definite change: prvsiyan's extended corpus was **V4** in the scored source but **V5** in our copy. V5's version note adds a `train_regen` split based on reconstructed official training prompts and changes the stated licence. It must not silently enter an unchanged V20 reproduction or later be treated as independent validation data.

The draft was restored to **extended corpus V4** using its version selector; reopening that selector confirmed V4 checked and latest unchecked. A second Quick Save was requested with the version name `V20 frozen; extended data pinned V4; no run`. The final save receipt is in `KAGGLE_PREPARATION.json`. Preserve V1 as the record of the inherited-version issue. Other attachment byte identities are not certified by matching names or aggregate input size, and incomplete UI captures are not treated as successful version checks.

## Concrete next execution gate

`scripts/stage08_input_preflight.py` is a separate read-only CPU audit. Its 12 synthetic checks pass. It performs **no neural inference, fitting, network use, public-source execution or real input mutation**. On the mounted Kaggle inputs it will:

1. Require unambiguous mandatory pool/checkpoint/base/encoder/attention/UltraBERT files; inspect safe metadata and tensor headers without importing model/custom code.
2. Hash the new pool, adapters, classifier heads, metadata and small configurations, and inventory large backbone shards by name and size. Compare these receipts with the intended immutable attachment versions; the inventory alone cannot prove source-version identity.
3. Validate the pool's declared 34,940 rows, 18 labels, text schema and class support, and reject exact/normalized official full-response overlap. Report duplicate counts without publishing text or original-row predictions.
4. List the older stack's extra CSV membership. Its original 7,644-row merged training table still needs reconciliation; a successful run after silently losing CSVs would be a changed recipe.

The script has **not run on the real Kaggle inputs**. Its synthetic pass is not a data, GPU compatibility or performance result. Finish input/version reconciliation and the aggregate pool/source review, then execute one frozen private reproduction using the intended source. Preserve all upstream credit, keep checkpoints and fitted outputs private, and inspect the candidate parquet and terminal logs before considering an official evaluation. Do not launch a weight search or claim a new original contribution from reproducing this public method.

## Evaluation boundary

The author-reported grouped CV is not independently reproduced by this inference notebook. The final adapters reportedly saw all official labels during stage 2. Holding out prompts only when fitting their Gaussian/readout heads would not undo that supervised exposure. Use fold-specific adapted checkpoints or fully prompt-excluded training for genuine development predictions. The author's preview, author's official public score, our future private preview, and any future official score in our account are four distinct milestones.

## Storage and attribution

Captured source and decoded embedded source are retained for static review, with the notebook's displayed Apache 2.0 licence and upstream notices. See `NOTICE.md` and `LICENSE-APACHE-2.0.txt`. External passage-pool/checkpoint licences are separate; their text and model weights are not republished in this repository. No API purchase, new provider generation, model inference, trained-weight publication, official submission or final-selection change occurred in this stage.
