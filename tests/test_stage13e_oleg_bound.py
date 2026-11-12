import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stage13e_oleg_bound import AuditError, FOCUS, LABELS, PASS_FLAGS, STOP_FLAGS, analyze


def fixture():
    wide, meta, rows, provenance = [], [], [], []
    for block in range(2):
        prompt = f'synthetic user request {block}'
        group = f'p{block}'
        row = f'w{block}'
        wide.append({'id': row, 'prompt': prompt})
        rows.append({'view': 'short', 'id': row, 'prompt_group_id': group})
        for idx, model in enumerate(sorted(LABELS)):
            cell = f'c{block}-{idx}'
            text = ('I cannot provide that answer. ' if block == 0 and model == FOCUS else '') + ('plain ' * 45)
            wide[-1][model] = text
            meta.append(dict(view='short', id=row, model=model, cell_id=cell, response_id=cell,
                             prompt_group_id=group, prompt_exact=True, word_count=len(text.split())))
            origin = ('speechmap' if block == 0 else 'artificialanalysis_microevals') if model == FOCUS else 'fixture'
            occurrence = dict(cell_id_short=cell, occurrence_id=f'o{cell}', source=origin, model=model,
                              response_id=cell, prompt_group_id=group)
            occurrence.update({name: True for name in PASS_FLAGS})
            occurrence.update({name: False for name in STOP_FLAGS})
            occurrence['license_unknown'] = model == FOCUS and block == 1
            provenance.append(occurrence)
    return tuple(pd.DataFrame(x) for x in (wide, meta, rows, provenance))


class OlegBoundTest(unittest.TestCase):
    def test_complete_blocks_cannot_outrun_bottleneck_label(self):
        r = analyze(*fixture())
        self.assertEqual(r['complete_exact_user_text_groups_before_screen'], 2)
        self.assertEqual(r['complete_exact_user_text_groups_after_refusal_marker_screen'], 1)
        self.assertEqual(r['complete_exact_user_text_groups_after_refusal_and_source_flags'], 0)
        self.assertEqual(r['minimum_per_label_cell_upper_bound_after_refusal_marker_screen'], 1)
        self.assertEqual(r['gemini_3_8_flash_source_counts']['speechmap']['opening_refusal_marker_cells'], 1)
        self.assertEqual(r['gemini_3_8_flash_source_counts']['artificialanalysis_microevals']['cells_with_unknown_license_flag'], 1)
        self.assertEqual(r['decision'], 'BLOCKED_UPPER_BOUND_BELOW_30')
        self.assertFalse(r['panel_admitted'])
        self.assertFalse(r['models_trained_or_scored'])
        self.assertNotIn('synthetic user request', str(r))

    def test_reject_mismatched_provenance_link(self):
        wide, meta, rows, provenance = fixture()
        provenance.loc[0, 'prompt_group_id'] = 'not-linked'
        with self.assertRaisesRegex(AuditError, 'provenance_prompt_group_id_link'):
            analyze(wide, meta, rows, provenance)


if __name__ == '__main__':
    unittest.main()
