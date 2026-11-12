# Stage 13I: a four-variant Gemini specialist on unassigned passages

The Stage 13H audit found 91 errors between Gemini variants among 425 unassigned errors. This is one fixed follow-up on the same previously used development prompts. It cannot estimate the full production ensemble or an official score.

## Fixed comparison

Reconstruct the original five prompt-exclusive folds and the exact Stage 13F assignment-on baseline from competition train 170484, GLM OOF 368292220, and pretrained DeBERTa features 358745836. Require the Stage 13F geometry digest, original Stage 13C score anchor, and Stage 13F's balanced accuracy and confusion digest. Keep the original 12–18 passage rule, fold balancing and all existing scores fixed. Verify the expected 223 target rows and 91 baseline within-Gemini errors against Stage 13H.

Train one four-class Gemini model per fold. Use the frozen pretrained DeBERTa layer 24/6 features and the unchanged Stage 13C scaler and logistic regression recipe (C=.005, LBFGS). Fit both scaler and head only on passages actually written by one of the four Gemini models in that fold's training prompts. The scorer does not use a held-out true label. The candidate replaces only the model variant on unassigned passages whose frozen baseline prediction is already Gemini. Never change a family decision, an assigned passage, or any other unassigned passage. A false Gemini family prediction remains wrong in both arms.

Report both arms on all 2,699 held-out rows. Include balanced accuracy, exact paired delta, corrections and regressions, class changes, and a conditional paired 95% interval using 5,000 fixed-seed prompt draws within folds. Positive screen requires at least +0.50 percentage points, strictly positive lower interval endpoint, and no class recall loss exceeding five points. The candidate was motivated by an audit of these same prompts, so even a pass would require separate validation. No threshold sweep, alternate specialist, class-score blending, or production change in this run.

One private offline CPU run, aggregate output only. No GPU, paid service, competition-test access, official submission or final-selection change.
