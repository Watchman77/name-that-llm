# Stage 06: honest error analysis, then matched-context centering
Date: 2026-10-17.
Status: public LoRA component artifact audit executed successfully as private CPU V1 359394412. Full S02 artifact inventory and the centering candidate comparison remain pending. No S03 submission. See results/stage_06/REPORT.md.

## Question and sequence
Can training a frozen centered classifier on smaller same-prompt groups improve generalisation to inferred small test groups, versus fitting centered features on all 17–18 responses? S02's frozen twins use full prompt groups; its LoRA centered head already uses triples. This is one specific hypothesis, not a claim that centering is absent.

First finish a CPU artifact audit. Preserve S02 V1 and its score as the official reference. Do not refit full-data competition checkpoints on a validation fold and call the result OOF.

## A. Prediction provenance and error audit (CPU)
Use the exact public LoRA V1 358831649 already attached to S02 as the first audit target. Read numeric NPZ files with allow_pickle=False.
1. Verify official train SHA256 40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707; IDs unique and ordered; labels/classes/groups aligned.
2. Read training_spec.json and each outer COMPLETE.json. Verify file digests, disjoint row/prompt sets, exact once-only validation coverage, fixed epoch provenance. Never include final checkpoint training predictions.
3. Recompute saved plain logits from validation_features and readouts and compare against lora24_oof.npz. Recompute declared source metrics before trusting diagnostics.
4. Produce per-class recall, directed confusion counts/rates, fold scores and disagreements. Label every table with the precise component and scenario; never title LoRA-only diagnostics “54.9% ensemble errors.”
5. Inventory equivalent artifacts for the second LoRA, token-attention, UltraBERT and frozen features. Record train prompts/fold definitions for every supervised component. Different outer fold schemes cannot simply be stacked as if all scoring labels were held out from every fitted member.
6. If complete S02 OOF cannot be reconstructed, report that gap explicitly. Evaluate the frozen-head change separately first. Do not use in-sample full-fit outputs as a substitute.

Outputs: provenance.json, component_inventory.csv, fold_scores.csv, per_class_recall.csv, confusion_counts.csv, confusion_rates.csv and component_disagreement.csv. Exclude raw passage text. Exact missing files must be named before deciding whether additional GPU extraction/training is necessary.

## B. One centering comparison, conditional on valid frozen feature caches
Hold the published feature extraction, regularisation C, downstream weights and clustering thresholds fixed. Apply only to S02's five frozen centered twins; leave LoRA centered training unchanged.

Control: current frozen twin, with per-prompt training means from all available training responses.
Candidate: for each training prompt, make fixed-seed partitions at group sizes 2, 3 and 6. For each partition, subtract the within-group feature mean. Each row contributes once per partition scheme; weight each copy 1/3 so total row weight stays unchanged. Merge any singleton remainder into the preceding group and record realised sizes. The three sizes are one declared training mixture, not three competing tuned candidates. Seed 20261017.

Fit centered StandardScaler and LogisticRegression using only outer-training prompts. Retain plain heads and singleton fallback. Validation uses inferred groups only; true prompt IDs are permitted for diagnostic oracle grouping, never deployment predictions.

Use the already saved five prompt folds. On each held-out fold, simulate pseudo-test contexts with 1, 3 and 6 responses per prompt, fixed seeds and identical rows for paired comparisons. Shuffle/mix prompts before inference; hide prompt IDs from group recovery. Use mixed GLM/lexical clustering only where correct cached GLM features exist; otherwise label results a lexical-only approximation rather than S02-equivalent validation.

Primary development endpoint: unweighted mean balanced accuracy across the three predeclared context scenarios for control vs candidate frozen-head evidence, with the same eligible-row rule (size 2–18). Report each scenario separately and paired prompt bootstrap uncertainty (3,000 draws). Include true-group diagnostic scores, cluster pair precision/recall and results by inferred group size to separate clustering errors from classifier errors.

Only claim an ensemble gain after all evaluated supervised members have compatible honest outer predictions. No zero-weight/full-fit component shortcuts.

## Decision rule
Advance only if the paired primary gain is positive, its 95% prompt-bootstrap lower bound is above zero, and no scenario loses more than 0.5 percentage points. This is a conservative development screening rule, not proof of private-test transfer. Report all per-class changes and the largest regression. One tested candidate only; further ideas require a new recorded protocol and recognition of selection bias.

Fit full-data candidate only after this screen. Check offline hidden-run budget <=9 hours, memory and input provenance. A future official submission gets a new receipt; keep S02 unchanged. Public feedback is confirmation, not a parameter-selection loop.

## Resources and progress limits
CPU audit first; no new GPU training justified by this document alone. Cached pretrained frozen embeddings can support cheap fold-local heads, but availability is not yet certified. No paid API, paid compute or synthetic data. 70% remains a stretch target; 56%, 58%, 60% are milestones to measure, not promised outcomes.
