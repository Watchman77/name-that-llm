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

if __name__=='__main__':
    f=SurfaceStyleFeatures()
    names=list(f.get_feature_names_out())
    # Manually check denominators using a five-character, two-word example.
    x=f.transform(['A, a!', ''])
    assert np.isclose(x[0,names.index('punct_U002C')],1/5)
    assert np.isclose(x[0,names.index('uppercase_letter_fraction')],1/2)
    assert np.isclose(x[0,names.index('function_a')],1.)
    assert np.isfinite(x).all() and len(set(names))==len(names)
    print(f'Feature checks passed: {len(names)} fixed features.')
