# Stage 07A private pilot — frozen before model fitting

Question: does one fixed addition of verified, first-party GPT responses improve the character-plus-style classifier on original official held-out prompt groups?

## Scope amendment to Stage 07

Use only `prvsiyan/prvsiyans-extended-name-that-llm-dataset` **V4**, `passages_clean.csv`, `split=new`, `provider_route=openai`. This is 750 rows: 150 complete five-GPT-model prompt blocks, with 150 examples for each of gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra and gpt-6-astra. Other labels still receive original official training data and remain in validation. External coverage of all 18 classes is not required for this data-effect experiment.

The source declares CC BY 4.0. Attribution: prvsiyan, Extended Name That LLM Dataset, V4, October 2026. Its published API envelopes were checked for every selected row in the preceding review. Direct request and served IDs are exact, including dated gpt-5.5. The source CSV and prompt bank are checksum-pinned. This pilot admits no Kaggle proxy, OpenRouter, Google or Anthropic responses; their separate route/use assessments do not block this bounded test.

Use scope: private, undistributed classifier. The current OpenAI Services Agreement's Permitted Exception explicitly includes qualifying classifiers that are not distributed or made commercially available to third parties: https://openai.com/policies/services-agreement/ . This is a bounded working admission for the local SVM experiment, not approval for distributing weights, a generative fine-tune or an official competition deployment. No model is serialized; no provider inference API is called. A fresh repository metadata check found the GitHub repository is PUBLIC, correcting the earlier private-repository description. Only code, external-public-data IDs, audit receipts and aggregate results may be committed. Keep official row-level labels/predictions and native margins out of that repository. The local classifier is not distributed by publishing its unfitted source code. No repository visibility setting is changed.

## Admission and leakage checks

The original training snapshot has 2,699 rows / 150 prompt groups / 18 labels. Check its frozen SHA-256. Independently check candidate uniqueness, prompt/class structure, word bounds, stop status, request/served IDs, and existing mechanical quality flags.

Before fitting, compare all candidate response texts to all official training texts, ignoring their labels. Mark a response suspect if it has an exact/normalized full-response overlap, either complete normalized passage is a substring of the other, it shares at least three distinct normalized word 8-grams with the union of official text, or nearest-official character TF-IDF cosine is at least 0.85. Exclude its entire external prompt block globally. This retrieval feature map is discarded; it is not used by either classifier. The preflight manifest fixes both the screen and its selected rows before any development score exists.

The 150 admitted named topics were inspected against all 150 published reconstructed official topics; no same named topic was identified. A separate review inspected the 30 closest retrieved instruction pairs from the full V4 new/clean candidate. This is not a guarantee of semantic independence: official prompt wording is unavailable and reconstructions are guesses. Treat a positive result as a provisional development signal, not sufficient grounds for candidate promotion or an official submission.

## Fixed comparison

Both arms use identical `GroupKFold(5)` original `prompt_id` splits, evaluating the same original validation passages and all 18 classes. All official passages from a held-out prompt stay out of that fold's training. No external passage is used for validation. External prompt IDs are a separate namespace, not joined to official IDs by numeric suffix.

Use the previously specified SurfaceStyleFeatures (111 fixed features), standardized and L2 normalized, plus character TF-IDF (3–5 grams, min_df=3, sublinear TF, 120,000 features maximum). Fit this mapping **once on original fold-training data only**, then reuse it in both arms. Candidate data does not affect vocabulary, IDF or style scaling. Raw passage text and punctuation are preserved identically in both arms.

Control trains a LinearSVC head on original fold-training rows. Candidate trains the identical head on those rows plus the frozen admitted external rows. Parameters for both: C=1, seed=20261018, dual=True, max_iter=10000, tol=0.0001. Original example weights are n_official/(18*n_class) in both arms. Each augmented class's external examples receive a combined weight of 0.5*n_official/18, evenly divided among its external rows. All external classes have at most 150 examples. These weights, C, feature settings and candidate scope will not be searched or revised in this pilot.

The optimization settings match, not the number of rows or wall time. Record head fit time, iterations and convergence warnings. Do not serialize a final model or run test inference.

## Measurements and fixed decision

Primary measure: pooled **development balanced accuracy** on original held-out passages, all 18 classes. Save fold results, predictions, confusion matrices, per-class recall and native decision margins. Margins are not calibrated probabilities.

Compute the candidate-minus-control difference and a paired 95% interval by jointly resampling 150 official prompt groups (10,000 draws, seed=20261018), recomputing macro recall including the missing Fable passage correctly. The interval is conditional on the fixed fitted folds and does not include every training or prompt-independence uncertainty. Report corrected and broken original passages.

The fixed improvement screen requires a positive overall difference, lower interval endpoint above zero, and no recorded fit warnings. A pass means **positive component signal requiring independence and ensemble review**, not a promoted competition candidate. A fail retains the control with no automatic variant, cap change or reweighting. This standalone character/style experiment is distinct from full S02 (previously verified official 0.549) and the incomplete Stage 06E proxy.

## Pre-execution record

The preflight passed on the real files with all 750 rows retained and no suspect blocks. Maximum nearest-official character cosine was 0.233093; no full-response, substring or thresholded 8-gram overlap was found. These are data checks, not development results. `preflight.json` pins environment, hashes and the complete candidate manifest hash. The runner, this protocol, compact admitted IDs and preflight are committed before fitting; a local freeze receipt records the immutable commit and runner checksum. Any source or manifest change makes the runner refuse execution.

Before execution, checks for class-weight mass, paired bootstrap on a known all-wrong-vs-all-correct contrast, and rejection of an altered freeze passed. The runner compiles. No independent agent review was performed.
