"""Review the overly broad V1 warning flag without refitting or changing results."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]/'results/stage_06b'
r=json.loads((root/'centering_result_V1.json').read_text())
expected='scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.'
messages=r['convergence_warnings']
assert len(messages)==15 and all(message==expected for message in messages)
gains=np.asarray(r['scores']['candidate'])-np.asarray(r['scores']['control'])
np.testing.assert_allclose(gains,r['scenario_gains'],rtol=0,atol=1e-15)
assert abs(float(gains.mean())-r['primary_mean_gain'])<1e-15
statistical_pass=bool(gains.mean()>0 and r['paired_prompt_95_interval'][0]>0 and min(gains)>=-.005)
review=dict(status='NUMERICAL_SCREEN_PASSES_AFTER_WARNING_REVIEW',original_v1_screen_pass=r['screen_pass'],statistical_screen_pass=statistical_pass,warning_review=dict(count=len(messages),exact_message=expected,classification='Future SciPy API deprecation, not an optimizer convergence failure',unrecognised_messages=[],v1_bug='V1 stored all caught warnings under convergence_warnings and rejected any warning.'),model_or_prediction_changes=False,refit_performed=False,primary_mean_gain=r['primary_mean_gain'],paired_prompt_95_interval=r['paired_prompt_95_interval'],decision='Advance only to further matched full-ensemble validation; no official submission justified.',limits=['Review uses preserved completed-run numeric results; does not recreate unobserved training.','One fp16-cache DeBERTa component with lexical-only grouping; no full S02 score gain established.'])
assert statistical_pass and not r['screen_pass']
(root/'WARNING_REVIEW.json').write_text(json.dumps(review,indent=2))
print(json.dumps(review,indent=2))
