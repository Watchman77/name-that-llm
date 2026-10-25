# Stage 12 source review — Dhruv V1 362013743

Static review of the six captured cells is complete. **GPU admission remains pending real mounted-input and executable-asset checks.** The source-author **0.651** and **5,251.1-second T4 x2 preview** were supplied from the parent's live check; neither is our result. No upstream code, model or serializer payload was executed locally. All six captured strings match their split files and parse after omitting IPython command lines; exact hashes are in [SOURCE_REVIEW.json](SOURCE_REVIEW.json).

The active recipe averages log probabilities from five folds plus a full-data adapter for each of Qwen3-14B, Qwen3-8B and Llama3.1-8B, adds two Nishant checkpoints and DeBERTa-large, plus train/pool SVC, centered twins and two full-data Gaussian heads. It mixes **15% complement**, then **10% ATLAS on selected low-margin rows**, applies **100 Sinkhorn iterations**, typography exclusions and Hungarian assignment for inferred groups of 12–18 rows. It does **not** include S05's full-text 27B Qwen branch. The complement also **intentionally excludes Qwen3.5 token-attention** via NTL_DROP_Q35=1; its absence is part of this exact recipe.

## Material gates before GPU

| Gate | Concrete source evidence and required check |
| --- | --- |
| Exact discovery | Cell1 resolves **37 heads.pt adapters: 19 legacy and 18 active v5**, four metadata checkpoints and seven config selectors, plus fixed files. Unused legacy globals still hard-fail if absent. Main mdir uses sorted config paths; adir/nmeta use unsorted first glob matches. Complement selects by model type/hidden size, so Base/chat identity must be reconciled. Enumerate every match and selected hash against the pinned inputs. |
| Expert inclusion | Cell5's shell pipeline does not enforce child success. infer.py catches missing/bad complement and all ATLAS exceptions and proceeds. Complement skips absent LoRA markers. A valid parquet alone cannot prove source recipe coverage. Reconcile the author's optional-member set and inspect complete own completion/fallback logs. |
| Runtime | No hard full-process deadline exists. ATLAS uses a projected budget, checked periodically after more than 50 rows; overrun skips ATLAS. Earlier experts/model loading and complement are unbounded internally. The competition's verified **9-hour cap** applies. Read actual free GPU quota and T4 x2 allocation; CPU checks and the public preview do not establish hidden runtime. |
| Executable assets | Hash and inspect mounted ATLAS/UltraBERT custom code, all selected offline wheels and payload identities before GPU. Runtime imports ntl_atlas/ntl_rlcd, uses UltraBERT trust_remote_code, Torch heads and PEFT. Offline bitsandbytes install only prints its return code. This review does not certify those unseen payloads. |
| Output evidence | A separate guard should read actual saved parquet and require exact IDs/order/count, valid18labels and schema. Validate saved comp_logit.npy and comp_meta.json: finite n-by18, canonical named axes and matching IDs. Main individual probabilities are not saved, so full mixture reconstruction and every internal finite check cannot be claimed from existing files. |

Unlike S05's shortcut, this preview does real GPU inference and repeats 40 seen training examples for small test files; Gaussian branches also extract features for all official training rows. The complement forwards both all train and test rows through seven backbone families. Its preview duration therefore cannot be used as a direct hidden-test ETA.

## Interpretation and remaining risks

Fold averaging is an inference ensemble, **not OOF evaluation**. Gaussian training features use the last full-data adapter. The 40-row checks are explicitly in-sample. The optional --sim prompt split retrains some heads but does not automatically retrain or exclude pretrained adapters/readouts, so it does not establish unbiased development accuracy. Imported comments about older grouped-OOF weights are provenance claims rather than new verified CV.

No explicit paid API, outbound transmission, download command or production hidden-test label lookup appears in the six visible cells. That does not establish checkpoint/external-pool provenance or the behavior of unseen custom code. Keep Internet off and clear smoke/fake-test diagnostic overrides for production.

Frozen numeric limitations include fp16 adapter merge/unmerge roundoff, BF16 retry without a second finite assertion, complement nan_to_num replacements, and final argmax that can still produce a valid label despite unsaved nonfinite intermediates. Sinkhorn's uniform counts and injective group assumptions are heuristics. Preserve them for exact reproduction; do not silently repair or tune the source. Some complement gates are computed but never applied to the saved running LOGIT, so comments are not proof of active contributions.

**Practical verdict:** no definitive visible-source fatal algorithm or explicit network/test-label violation was found. If the real input, source parity, executable-asset and free-resource gates pass, one frozen free-Kaggle reproduction is reasonable. It may reproduce 0.651; that is not a promised own score or evidence of reaching 70%.

[Exact upstream version](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27?scriptVersionId=362013743) · [machine-readable findings and source hashes](SOURCE_REVIEW.json).
