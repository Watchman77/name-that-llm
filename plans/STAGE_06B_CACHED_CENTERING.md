# Stage 06B: cached-feature inventory and one DeBERTa centering test
Declared 2026-10-17 before candidate outcomes. Scope narrows the Stage 06 protocol to the one S02 centered twin whose frozen features were located in public input.

## Source inventory
Exact input versions from frozen S02 V1 359332774:
- encoder features: lucasboesen/name-that-llm-extract-encoders2 V1 358745836. Visible source extracts official pretrained encoders in eval/no_grad mode and saves emb, emb_style, layers and IDs in original competition row order.
- second LoRA: lucasboesen/name-that-llm-train-lora24-s7 V1 358911769. Separate jobs 0..4 with saved held-out features/readouts, plus final fit; prompt folds use sorted prompts shuffled with RandomState(0), round-robin across five folds.
- token attention: lucasboesen/name-that-llm-train-tokattn V1 358928013; round2 V1 358948428. OOF files include probabilities/classes/fold and IDs; source fits fold-local scalers/heads.
- public first LoRA already verified separately in Stage 06A. Its folds differ; never assume a cross-component stack is fully held out merely because each member has OOF.
- GLM, ELECTRA weight and desklib weight notebook trees contain weights/tokenizers and placeholder submissions, with no passage-feature cache listed. UltraBERT availability still needs an explicit file/fold audit.

## CPU run
Attach competition data and these four public notebook outputs, with CPU and internet off. Never execute attached Python, load torch .pt pickle files, retrain a backbone or submit to the competition.

Inspect NPZ keys, shapes, numeric ranges and SHA256. Numeric arrays use allow_pickle=False. Some sources saved IDs as object dtype: inspect the NPY header and pickle bytecode as data only (pickletools), extract unique official ID string literals, require exact original order/count, and never unpickle. If this check fails, stop rather than bypass.

Verify second LoRA folds, IDs, classes, once-only coverage and completion reports; reconstruct plain predictions. Report any fp16 rounding difference from its fp32 training-log accuracy. For token-attention OOF, verify ID/classes/fold, finite normalised probabilities and recomputed metrics. These files alone cannot independently prove retraining lineage; source review and source artifact provenance are required and recorded.

## One controlled candidate
Use only enc_debv3l2_train.npz, mean pooling of layers [24,6], matching S02's frozen DeBERTa representation but stored as float16 rather than S02's in-memory fp32. Convert numeric features to float32 for computation. Do not select other cache layers/models based on scores.

Use GroupKFold(5) by prompt_id, matching our original development fold method; verify the prompt-to-fold mapping against the archived reference mapping embedded in the runner. Fit all StandardScalers and LogisticRegression(C=0.005,max_iter=5000) using outer training prompts only.
- Control centered head: subtract each complete training prompt's feature mean.
- Candidate centered head: the declared mixture of partitions at sizes 2,3,6 with seed 20261017 + fold index. Each row appears once per size; all three copies get sample weight 1/3. Merge singleton remainders into the prior group.
- Plain head unchanged, fit once per fold. Preserve S02 test-batch standardisation, singleton fallback, eligible inferred cluster size 2..18, and fixed DeBERTa evidence weights plain=0.05/centered=0.25.
- Simulate context batches with target 1,3,6 responses per prompt using seed 20261017+1000+fold index, same batches for control and candidate. Clustering sees mixed held-out texts only, TF-IDF English stopwords removed, sublinear TF, average linkage cosine cut0.85 (S02 fallback setting). Oracle true groups are diagnostic only.
- No GLM features were located, so this is a lexical-only component approximation, NOT validation of S02's mixed clustering or complete ensemble.
- No new blend weight, threshold search, calibration or class balancing.

Primary endpoint: mean balanced accuracy over the three scenarios for this component's fixed weighted plain+centered evidence. Paired 3,000-draw prompt bootstrap interval. Screen passes only if mean gain>0, lower95% bound>0, and no scenario loses>0.5 percentage points. Report all scenarios, per-class changes, oracle diagnostics and convergence warnings. Historical development reuse remains.

A positive result licenses only further validation; it does not justify adding this measured gain to S02's official54.9% or an immediate official submission. A negative/inconclusive result is retained without retuning.
