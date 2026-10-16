# Frozen Stage 03 candidate: character patterns + 111 style features.

from pathlib import Path
import hashlib, json, platform, os, time, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
import sklearn
import pyarrow
from IPython.display import display
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

candidates = [Path(os.environ.get('NAME_THAT_LLM_DATA_DIR', 'data')),
              Path('/kaggle/input/competitions/name-that-llm'),
              Path('/kaggle/input/name-that-llm')]
DATA_DIR = next((p for p in candidates if (p/'train.csv').is_file() and (p/'test.csv').is_file()), None)
if DATA_DIR is None:
    raise FileNotFoundError('Attach the competition on Kaggle or put train.csv and test.csv in data/.')
OUT = Path('baseline_outputs')
OUT.mkdir(exist_ok=True)
RUN_START = time.perf_counter()
VERSIONS = {'python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__,
            'scipy':scipy.__version__, 'sklearn':sklearn.__version__, 'pyarrow':pyarrow.__version__}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                     'axes.spines.right':False, 'figure.facecolor':'white', 'axes.facecolor':'white'})
print('Input directory:', DATA_DIR)
print('Environment:', VERSIONS)

from sklearn.preprocessing import StandardScaler, Normalizer
"""Fixed surface-style features; no labels, vocabulary fitting or external data."""
import re
import string
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FUNCTION_WORDS = tuple('a an the and or but if then than because while although however therefore thus so as at by for from in into of on to with without is are was were be been being it its this that these those we you they he she i not no can could may might must should would will'.split())
PUNCTUATION = tuple(string.punctuation) + tuple('–—…“”‘’')
BASE_NAMES = ('uppercase_letter_fraction','digit_fraction','whitespace_fraction',
              'newline_fraction','non_ascii_fraction','mean_word_length','std_word_length',
              'mean_words_per_sentence','std_words_per_sentence','contraction_fraction',
              'bullet_line_fraction','heading_line_fraction','table_line_fraction',
              'fence_markers_per_line','blank_line_fraction')
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

class SurfaceStyleFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(BASE_NAMES)+[f'punct_U{ord(p):04X}' for p in PUNCTUATION]
                          +[f'function_{w}' for w in FUNCTION_WORDS], dtype=object)
    def transform(self, X):
        rows=[]
        for value in X:
            text=str(value)
            chars=max(len(text),1)
            words=WORD_PATTERN.findall(text)
            nw=max(len(words),1)
            alpha=max(sum(c.isalpha() for c in text),1)
            lengths=np.asarray([len(w) for w in words], dtype=float)
            sentences=[WORD_PATTERN.findall(s) for s in re.split(r'[.!?]+',text) if WORD_PATTERN.search(s)]
            sentence_lengths=np.asarray([len(s) for s in sentences], dtype=float)
            lines=text.splitlines() or ['']
            nonempty=[s for s in lines if s.strip()]
            line_den=max(len(nonempty),1)
            low=[w.lower() for w in words]
            base=[sum(c.isupper() for c in text)/alpha,
                  sum(c.isdigit() for c in text)/chars,
                  sum(c.isspace() for c in text)/chars,
                  text.count('\n')/chars,
                  sum(ord(c)>127 for c in text)/chars,
                  float(lengths.mean()) if len(lengths) else 0.,
                  float(lengths.std()) if len(lengths) else 0.,
                  float(sentence_lengths.mean()) if len(sentence_lengths) else 0.,
                  float(sentence_lengths.std()) if len(sentence_lengths) else 0.,
                  sum("'" in w or '’' in w for w in words)/nw,
                  sum(bool(re.match(r'^\s*(?:[-*+]|\d+[.)])\s+',s)) for s in nonempty)/line_den,
                  sum(bool(re.match(r'^\s*#{1,6}\s+',s)) for s in nonempty)/line_den,
                  sum(bool(re.match(r'^\s*\|.*\|\s*$',s)) for s in nonempty)/line_den,
                  text.count('```')/line_den,
                  sum(not s.strip() for s in lines)/len(lines)]
            rows.append(base+[text.count(p)/chars for p in PUNCTUATION]
                        +[low.count(w)/nw for w in FUNCTION_WORDS])
        if not rows:
            return np.empty((0,len(self.get_feature_names_out())),dtype=float)
        matrix=np.asarray(rows,dtype=float)
        assert matrix.shape[1]==len(self.get_feature_names_out())
        assert np.isfinite(matrix).all()
        return matrix



train = pd.read_csv(DATA_DIR/'train.csv', dtype=str)
test = pd.read_csv(DATA_DIR/'test.csv', dtype=str)
EXPECTED_LABELS = ['gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra',
    'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
    'claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
    'qwen3.8-max','glm-5.3','glm-5.3-flash','kimi-k3']
assert {'id','prompt_id','text','model'} <= set(train.columns)
assert {'id','text'} <= set(test.columns)
assert not train[['id','prompt_id','text','model']].isna().any().any()
assert not test[['id','text']].isna().any().any()
assert train.id.is_unique and test.id.is_unique
assert train.text.str.strip().ne('').all() and test.text.str.strip().ne('').all()
assert set(train.model) == set(EXPECTED_LABELS), 'Unexpected labels: verify official instructions'
assert not train.text.duplicated().any(), 'Investigate duplicate passages before validation'
normalized = train.text.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
assert not normalized.duplicated().any(), 'Investigate normalized duplicates before validation'
assert not train.duplicated(['prompt_id','model']).any(), 'Investigate repeated prompt/model pairs'
labels = sorted(EXPECTED_LABELS)
counts = train.model.value_counts().sort_index()
words = train.text.str.split().str.len()
audit = {'training_rows':len(train), 'prompts':int(train.prompt_id.nunique()),
         'models':len(labels), 'preview_test_rows':len(test), 'duplicate_ids':0,
         'duplicate_exact_text':0, 'duplicate_normalized_text':0,
         'missing_required_training_values':0, 'class_counts':counts.to_dict(),
         'models_per_prompt':{str(k):int(v) for k,v in train.groupby('prompt_id').model.nunique().value_counts().items()},
         'word_count_min_median_max':[int(words.min()), float(words.median()), int(words.max())]}
print('Audit:', {k:v for k,v in audit.items() if k != 'class_counts'})
display(counts.rename('passages').to_frame())
hashes = {f:hashlib.sha256((DATA_DIR/f).read_bytes()).hexdigest()
          for f in ['train.csv','test.csv','sample_submission.parquet'] if (DATA_DIR/f).is_file()}
_ = (OUT/'data_audit.json').write_text(json.dumps({'audit':audit,'sha256':hashes}, indent=2))


def build_model():
    style = Pipeline([('extract', SurfaceStyleFeatures()),
                      ('scale', StandardScaler()), ('normalize', Normalizer())])
    return Pipeline([
        ('features', FeatureUnion([
            ('character', TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=3,
                                         sublinear_tf=True, max_features=120000)),
            ('style', style)])),
        ('classifier', LinearSVC(C=1.0, class_weight='balanced', random_state=20261016,
                                 max_iter=10000, dual=True))])

oof = np.empty(len(train), dtype=object)
fold_id = np.full(len(train), -1)
fold_results = []
convergence_warnings = []
for fold, (tr, va) in enumerate(GroupKFold(5).split(train.text, train.model, train.prompt_id)):
    started = time.perf_counter()
    assert set(train.prompt_id.iloc[tr]).isdisjoint(set(train.prompt_id.iloc[va]))
    assert set(train.model.iloc[tr]) == set(labels) and set(train.model.iloc[va]) == set(labels)
    estimator = build_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        estimator.fit(train.text.iloc[tr], train.model.iloc[tr])
    convergence_warnings += [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    oof[va] = estimator.predict(train.text.iloc[va])
    fold_id[va] = fold
    result = {'fold':fold+1,'validation_rows':len(va),
              'validation_prompts':int(train.prompt_id.iloc[va].nunique()),
              'balanced_accuracy':float(balanced_accuracy_score(train.model.iloc[va], oof[va])),
              'seconds':time.perf_counter()-started}
    fold_results.append(result)
    print(f"Fold {fold+1}: balanced accuracy={result['balanced_accuracy']:.4f}; seconds={result['seconds']:.1f}", flush=True)
assert (fold_id >= 0).all()
score = float(balanced_accuracy_score(train.model, oof))
cm = confusion_matrix(train.model, oof, labels=labels)
per_class_recall = cm.diagonal() / cm.sum(axis=1)
assert np.isclose(score, per_class_recall.mean()), 'Independent metric reconciliation failed'
assert cm.sum() == len(train)
report = pd.DataFrame(classification_report(train.model, oof, labels=labels,
    output_dict=True, zero_division=0)).T
print(f'Pooled balanced accuracy: {score:.4f}')
print(f'Fold SD: {np.std([r["balanced_accuracy"] for r in fold_results], ddof=1):.4f} (not a confidence interval)')
print(f'Uniform-random reference expectation: {1/len(labels):.4f}')
print('Convergence warnings:', len(convergence_warnings))
display(report.loc[labels,['precision','recall','f1-score','support']])
oof_table = train[['id','prompt_id','model']].assign(fold=fold_id+1, prediction=oof)
oof_table.to_csv(OUT/'baseline_oof.csv', index=False)
pd.DataFrame(fold_results).to_csv(OUT/'baseline_folds.csv', index=False)
report.to_csv(OUT/'classification_report.csv')


recall_table = report.loc[labels,['recall','support']].sort_values('recall')
fig, ax = plt.subplots(figsize=(10,8))
bars = ax.barh(recall_table.index, recall_table.recall, color='#2864A5')
ax.axvline(1/len(labels), color='#555555', linestyle='--', linewidth=1,
           label='Uniform-random expected recall (5.56%)')
for bar, value in zip(bars, recall_table.recall):
    ax.text(value+0.012, bar.get_y()+bar.get_height()/2, f'{value:.1%}', va='center', fontsize=10)
ax.set_xlim(0,1.08)
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel('Recall on held-out prompts')
ax.set_title(f'Stage 03: pooled balanced accuracy {score:.2%}\n2,699 passages; five prompt-grouped folds')
ax.legend(loc='lower right', frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT/'per_model_recall.png', dpi=160, bbox_inches='tight')
plt.show()
plt.close(fig)

norm_cm = cm/cm.sum(axis=1,keepdims=True)
short_labels = [x.replace('gemini-','G: ').replace('claude-','C: ').replace('gpt-','O: ') for x in labels]
fig, ax = plt.subplots(figsize=(12,11))
im = ax.imshow(norm_cm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(labels)), short_labels, rotation=55, ha='right', fontsize=10)
ax.set_yticks(range(len(labels)), short_labels, fontsize=10)
ax.set_xlabel('Predicted model')
ax.set_ylabel('True model')
ax.set_title('Stage 03: confusion on unseen prompts\nRows sum to 100%; O = OpenAI, G = Google, C = Anthropic')
cbar=fig.colorbar(im, ax=ax, shrink=0.78)
cbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
cbar.set_label('Share of passages from the true model')
fig.tight_layout()
fig.savefig(OUT/'confusion_matrix.png', dpi=160, bbox_inches='tight')
plt.show()
plt.close(fig)


started = time.perf_counter()
final_model = build_model()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', ConvergenceWarning)
    final_model.fit(train.text, train.model)
convergence_warnings += [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
submission = pd.DataFrame({'id':test.id, 'model':final_model.predict(test.text)})
assert list(submission.columns)==['id','model']
assert len(submission) == len(test) and submission.id.is_unique
assert not submission.isna().any().any()
assert set(submission.model) <= set(labels)
submission.to_parquet('submission.parquet', index=False)
pd.testing.assert_frame_equal(submission, pd.read_parquet('submission.parquet'))
if (DATA_DIR/'sample_submission.parquet').is_file():
    example = pd.read_parquet(DATA_DIR/'sample_submission.parquet')
    assert list(example.columns)==['id','model']
    # Preview example IDs do not constrain the hidden test supplied by Kaggle.
print('Created submission.parquet:', len(submission), 'rows; scoring is handled by Kaggle.')
metrics={'experiment':'stage_03_character_plus_style', 'balanced_accuracy':score,
    'fold_results':fold_results, 'source_sha256':hashes, 'environment':VERSIONS,
    'convergence_warnings':convergence_warnings, 'full_fit_preview_seconds':time.perf_counter()-started,
    'total_seconds':time.perf_counter()-RUN_START, 'executed_utc':datetime.now(timezone.utc).isoformat(),
    'validation':'GroupKFold(n_splits=5) by prompt_id; fold-local vocabulary and IDF',
    'test_rows':len(test), 'test_status':'Output generated; any official score is provided by Kaggle.'}
(OUT/'baseline_metrics.json').write_text(json.dumps(metrics, indent=2))
print(f"Total notebook computation: {metrics['total_seconds']:.1f} seconds")

