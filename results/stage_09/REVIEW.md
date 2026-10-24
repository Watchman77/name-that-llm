# Stage 09 implementation review

Prepared and independently reviewed on 24 October 2026 before GPU fitting. The real-data CPU admission audit passed 13 checks; runner algebra/alignment/schema checks passed 24; nbformat schema/readback and all three code cells compile. These are not model-training or CUDA results.

The review checked whole-prompt memberships, all 18 official labels, source association and immutable data admission, identical initial heads, fresh official optimizers/RNG schedules, last-epoch-only evaluation, actual-row accumulation normalization, finite ordered probability/logit readback, private checkpoint/prediction paths and the two-allocated-device budget.

Repairs made before fitting: ignore only the same `.cache` snapshot metadata excluded by the original vendor manifest; collect uncapped token lengths and truncation incidence; mark training as unstarted during preflight; test the real CUDA graph on arbitrary fixtures without an optimizer step; serialize independent checkpoint tensor copies; and validate NPZ column order, logits, IDs and softmax parity on both fresh readback and resume. The final freeze and notebook contain these repairs.

Actual Torch model loading, forward/backward, AMP and memory behavior must pass the saved Kaggle CUDA preflight. A failure is an incomplete run, never a valid score. Checkpoints are resumable only after complete phases with matching source/data freeze and cumulative budget. No silent hardware, schedule, length or corpus fallback is allowed.

The larger S03 pool was not admitted for this development pilot: the inference attachment omits the actual prompt wording required by the prospective review. Instead the fixed 750-row subset has publisher-declared own-written prompts and actual call/request associations. Official prompt wording is unavailable, so semantic independence is not certified. The same development groups have been reused; a positive conditional interval earns broader grouped evaluation only.

The successful source's architectural idea and rates are credited to Nishant Dahal's frozen part-two V1 361311282. Our bounded base-size implementation is new code; it neither initializes from the author's label-exposed checkpoints nor claims to reproduce the author's larger training corpus. Original competition text, row IDs, row predictions and trained weights stay outside public repository updates.
