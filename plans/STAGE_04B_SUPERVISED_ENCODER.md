# Stage 04B — prospective supervised encoder adaptation

## Status and question

**Planned, not executed.** Stage 04's frozen MPNet recipe failed (14.59% alone, 34.12% with character/style, versus the 35.41% reference). This distinct trial asks whether adapting a pretrained contextual encoder to the 18 generator labels improves unseen-prompt development performance. It does not assume that a larger model will win.

## Fixed candidate and resources

- One candidate only: `microsoft/deberta-v3-base`, public ungated MIT-licensed checkpoint, Hugging Face revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
- Source: https://huggingface.co/microsoft/deberta-v3-base . Retain its model card and checkpoint hashes with the run.
- Add a randomly initialized 18-class sequence-classification head and fine-tune the encoder and head together. No synthetic passages or external competition-labelled dataset.
- Prepare model/dependency files online separately, then require local-only loading and offline execution for the training experiment. No remote inference or paid services.
- Intended execution: one Kaggle T4 GPU, sequential folds, within an initial maximum **three GPU-hours** of the account's included quota. The UI listed GPU T4 x2, but actual quota allocation and runtime compatibility are not yet verified. Use one device; do not silently switch to paid hardware or CPU training.
- Prefer the validated Transformers 4.56.2 API; capture actual Python, torch, Transformers, tokenizers, SentencePiece and protobuf versions before running. If the offline environment cannot load and run the model, preserve the failure and repair the environment before fitting the trial. Pin and record dependencies and hashes used in that environment.

## Fixed training recipe

- Same official train SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`; all 2,699 rows, 150 prompts and 18 labels. Reuse the exact saved five folds; IDs and prompt IDs are never features.
- Original text, tokenizer from the pinned checkpoint, max length 384 including specials, right truncation, dynamic padding to a multiple of 8. Report untruncated token lengths and truncation incidence.
- Reset pretrained weights and random head for every fold; seed Python, NumPy, torch and the shuffled training loader with 20261016. Report GPU nondeterminism limitations.
- Three complete training epochs, selecting the **final epoch only**. No early stopping, best-epoch choice, validation-driven scheduler or hyperparameter search. Held-out predictions are computed after training; validation labels do not enter training.
- Micro-batch size 4, gradient accumulation 4, effective batch size 16 except the final partial accumulation group. Scale the final partial group by its actual row count so each example has the same loss weight.
- AdamW, learning rate 2e-5, betas (0.9,0.999), eps 1e-8, weight decay 0.01 on non-bias/non-LayerNorm parameters; no decay on biases or normalization parameters. Linear learning-rate decay, 10% optimizer-step warmup, gradient norm clipping 1.0.
- Mixed float16 with gradient scaling on T4. Keep optimizer master parameters in float32. Record skipped/overflow steps and reject a nonfinite run as incomplete rather than reporting a valid score.
- Cross-entropy class weights computed only from the training fold, inverse to training support and normalized to mean one. No label smoothing, augmentation, extra losses, layer freezing, ensemble or temperature tuning.
- Stop on runtime/resource failure, nonfinite loss, or the declared budget limit. Preserve partial logs and checkpoints. An out-of-memory batch-size change requires a recorded protocol amendment before another fit; do not quietly change the trial.

## Evaluation and stop rule

Primary metric: pooled OOF balanced accuracy, compared with the frozen Stage 03 reference. Save every final-epoch fold score, OOF logits/predictions in fixed label order, per-model recall, length-bin diagnostics, runtime, warnings, code/data/fold/checkpoint hashes and environment details. Retain failed attempts with explicit status; never combine partial attempts as one clean trial.

Report the paired development difference with 2,000 prompt-cluster bootstrap resamples, seed 20261016, and the same conditional/selection caveats as Stage 04. Inspect all five folds and class regressions before selecting a provisional candidate. No supplied prompt is an untouched confirmation set.

This is one fixed supervised recipe. Do not add learning rates, epochs, seeds or architectures after inspecting outcomes under this protocol. No competition submission or final-selection change is part of this trial. A successful candidate must pass a separate offline Kaggle inference/version check before any official submission.

## Implementation clarifications recorded before the first fit

Use the checkpoint's slow SentencePiece tokenizer (`use_fast=False`) with SentencePiece 0.2.1. Use Transformers 4.56.2, tokenizers 0.22.2, huggingface-hub 0.36.2 and protobuf 6.31.1 from prepared offline wheels; record the Kaggle torch version (minimum 2.6 for the original checkpoint format). Set AMP GradScaler initial scale to 1024 and growth interval to 2000; an overflow/nonfinite gradient ends the attempt as incomplete with its record preserved. Warmup steps round upward to the nearest whole optimizer step. Accumulated summed weighted losses divide by the actual total row count in that accumulation group, including its final partial group. Keep final checkpoint parameters in float32. Reserve the final minute of the three-hour allowance for status/checkpoint preservation.

Kaggle's accelerator dialog reported **30 included hours remaining** on 17 October 2026 before resource preparation. Resource preparation is a separate private CPU notebook; no classifier is fitted there. Actual T4 runtime availability still requires the training preflight check.
