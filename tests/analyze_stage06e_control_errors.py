"""Aggregate-only error analysis of three verified Stage06E control confusions.

No new prediction, classifier, parameter search or model execution is performed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/stage_06e'
SCENARIOS=(1,3,6)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    manifest=json.loads((OUT/'output_manifest.json').read_text())
    matrices=[]
    for s in SCENARIOS:
        path=OUT/f'raw_control_scenario{s}_confusion.csv'
        assert digest(path)==manifest[path.name]['sha256']
        matrices.append(pd.read_csv(path,index_col=0))
    labels=matrices[0].index.tolist()
    values=np.stack([m.to_numpy() for m in matrices])
    assert values.shape==(3,18,18) and np.all(values>=0)
    supports=values.sum(2)
    assert np.array_equal(supports[0],supports[1]) and np.array_equal(supports[0],supports[2])
    recalls=np.stack([np.diag(v)/sup for v,sup in zip(values,supports)])
    control=pd.DataFrame(recalls.T,index=labels,columns=SCENARIOS)
    control['mean']=control.mean(axis=1)
    perclass=pd.read_csv(OUT/'per_class_recall.csv')
    for s in SCENARIOS:
        recorded=perclass[(perclass.endpoint=='raw_control')&(perclass.scenario==s)].set_index('model').reindex(labels)
        np.testing.assert_allclose(control[s],recorded.recall,rtol=0,atol=1e-15)
    rows=[]
    for i,a in enumerate(labels):
        for j,b in enumerate(labels):
            if i==j:continue
            counts=values[:,i,j]
            rows.append(dict(pair_type='directed_true_to_predicted',model_a=a,model_b=b,
                             context1_errors=int(counts[0]),context3_errors=int(counts[1]),context6_errors=int(counts[2]),
                             mean_errors_per_context=float(counts.mean()),repeated_context_error_count=int(counts.sum())))
            if i<j:
                counts=values[:,i,j]+values[:,j,i]
                rows.append(dict(pair_type='undirected_mutual',model_a=a,model_b=b,
                             context1_errors=int(counts[0]),context3_errors=int(counts[1]),context6_errors=int(counts[2]),
                             mean_errors_per_context=float(counts.mean()),repeated_context_error_count=int(counts.sum())))
    pairs=pd.DataFrame(rows).sort_values(['pair_type','mean_errors_per_context','model_a','model_b'],ascending=[True,False,True,True])
    pairs['rank_within_type']=pairs.groupby('pair_type').cumcount()+1
    pairs=pairs[['pair_type','rank_within_type','model_a','model_b','context1_errors','context3_errors','context6_errors','mean_errors_per_context','repeated_context_error_count']]
    pairs.to_csv(OUT/'error_pairs.csv',index=False)
    assert len(pairs)==18*17+18*17//2
    errors=values.sum((1,2))-np.trace(values,axis1=1,axis2=2)
    for kind in ('directed_true_to_predicted','undirected_mutual'):
        selected=pairs[pairs.pair_type==kind]
        assert np.array_equal(selected[['context1_errors','context3_errors','context6_errors']].sum().to_numpy(),errors)
        assert int(selected.repeated_context_error_count.sum())==int(errors.sum())
    candidate=perclass[perclass.endpoint=='raw_candidate'].pivot(index='model',columns='scenario',values='recall').reindex(labels)
    delta=candidate-control[list(SCENARIOS)];delta['mean']=delta.mean(axis=1)
    weak=control.sort_values(['mean'],ascending=True).head(6)
    directed=pairs[pairs.pair_type=='directed_true_to_predicted'].head(6)
    mutual=pairs[pairs.pair_type=='undirected_mutual'].head(6)
    harms=delta.sort_values('mean').head(4)
    def weak_table():
        text='| True class | Context 1 | Context 3 | Context 6 | Mean recall |\n|---|---:|---:|---:|---:|\n'
        for name,row in weak.iterrows():text+=f'| {name} | {100*row[1]:.2f}% | {100*row[3]:.2f}% | {100*row[6]:.2f}% | {100*row["mean"]:.2f}% |\n'
        return text
    def pair_table(frame,directed=False):
        text=('| True → predicted | Mean errors/context | Counts in contexts 1 / 3 / 6 |\n|---|---:|---|\n' if directed else '| Mutual confusion | Mean errors/context | Counts in contexts 1 / 3 / 6 |\n|---|---:|---|\n')
        arrow='→' if directed else '↔'
        for row in frame.itertuples():text+=f'| {row.model_a} {arrow} {row.model_b} | {row.mean_errors_per_context:.2f} | {row.context1_errors} / {row.context3_errors} / {row.context6_errors} |\n'
        return text
    harm_table='| Class | Mean recall change | Context 6 change |\n|---|---:|---:|\n'
    for name,row in harms.iterrows():harm_table+=f'| {name} | {100*row["mean"]:+.2f} pp | {100*row[6]:+.2f} pp |\n'
    top_share=float(mutual.iloc[0].mean_errors_per_context/errors.mean())
    doc=f'''# Stage 06E control error analysis

The centering candidate failed the primary comparison: **54.8122% → 54.6767% mean raw development balanced accuracy (−0.1356 percentage points)**. Retain the original control. These results do not change the confirmed official S02 score.

This analysis uses the three complete 18×18 raw-control confusion previews from [V2 359514801](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06e-canonical-cpu-evidence/output?scriptVersionId=359514801). Their local bytes and hashes match the preserved runtime output manifest; supports and diagonals reconcile with printed recalls and all three balanced accuracies. Each context predicts the **same 2,699 passages**. Counts averaged or summed across contexts are repeated predictions, never additional independent examples or counts of unique passages. Control errors are {int(errors[0])}, {int(errors[1])} and {int(errors[2])} in contexts 1/3/6.

## Weakest classes

Ranked by mean class recall across the fixed contexts:

{weak_table()}
Fable has 149 passages; every other class has 150. That one-row support difference does not explain its low recall. Fable is frequently confused with both GLM classes, so a same-family-only strategy would miss a material error source.

## Largest confusion pairs

Mutual counts add the two directions within each context:

{pair_table(mutual)}
The Gemini 3.7/3.8 pair accounts for **{100*top_share:.2f}% of control errors averaged across contexts**. It remains the largest mutual confusion in all three settings. This is an observed target for investigation, not an estimate of recoverable accuracy.

The largest directed confusions are:

{pair_table(directed,True)}
Full rankings, including zero-count pairs, are in `error_pairs.csv`. For directed rows, `model_a` is the true class and `model_b` the predicted class; undirected rows combine both directions. `repeated_context_error_count` sums three context-specific prediction counts and is not a unique-passage count.

## What the rejected candidate harmed

Raw candidate-minus-control class recall changes:

{harm_table}
The candidate improves GPT-5.6-sol by +2.00 pp mean recall, but its gains do not outweigh the losses. In context 6 it corrects 41 passages and breaks 52: a net loss of 11 correct predictions. The slight balanced diagnostic increase cannot rescue a failed raw endpoint.

## Next hypotheses and evidence needed

1. **Prioritize Gemini 3.7 versus 3.8.** Test whether a dedicated separation signal can address the persistent two-way confusion. Retrieve the already-saved `ensemble_arrays.npz` first to identify row IDs, margins and prompt distribution; the aggregate matrices alone cannot show whether the same passages fail in every context.
2. **Investigate GLM 5.3 versus Flash next.** Both have low recall and a persistent mutual confusion. Inspect cross-confused passages and existing component probabilities before choosing features or a pair classifier. Fable-to-GLM errors deserve a separate check; label names alone do not establish the mechanism.
3. **Require a prospective comparison for any correction.** Freeze one hypothesis and its feature/gating recipe, preserve prompt exclusion in every supervised fit, and evaluate the complete fixed raw ensemble plus every class. Track false entry from other classes into the targeted pair. Reused development errors are hypothesis-generation evidence; they do not create an untouched validation set. A successful pair-only metric is insufficient if total raw balanced accuracy worsens.

No model, fitting, inference or parameter search was run for this analysis. The full raw NPZ files are not locally available, so no row-level overlap, probability-margin, paired-bootstrap or causal error-mechanism claim is made.
'''
    (OUT/'ERROR_ANALYSIS.md').write_text(doc)
    print(json.dumps(dict(status='COMPLETE_AGGREGATE_ERROR_ANALYSIS',models_fitted=0,new_predictions=0,
        control_errors_by_context=errors.tolist(),top_mutual_pair=mutual.iloc[0].to_dict(),
        weakest_class=weak.index[0],weakest_mean_recall=float(weak.iloc[0]['mean']),
        error_pairs_rows=len(pairs),error_pairs_sha256=digest(OUT/'error_pairs.csv'),
        report_sha256=digest(OUT/'ERROR_ANALYSIS.md')),indent=2))

if __name__=='__main__':main()
