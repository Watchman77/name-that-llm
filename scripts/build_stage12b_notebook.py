#!/usr/bin/env python3
"""Build the one fixed Stage12B counterfactual without executing model code."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json
import re
import nbformat

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/stage_12b';RESULT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'notebooks/Stage_11_Public_642_Reproduction.ipynb'
OUTPUT=ROOT/'notebooks/Stage_12B_SVC_Topic_Correction.ipynb'
HELPER=ROOT/'scripts/stage12b_intervention_validation.py'
def sha(data):return hashlib.sha256(data).hexdigest()
def source(cell):return ''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
def canonical(text):return sha((text.rstrip()+'\n').encode())
nb=nbformat.read(BASE,as_version=4);codes=[c for c in nb.cells if c.cell_type=='code']
assert len(codes)==34
original=source(codes[29])
needle="        total+=.25*clipped_logp(aligned(work/'deberta.npz',ids))\n    if family=='part_two_complement_atlas_0636':"
assert original.count(needle)==1
insertion="""    # Stage12B: one fixed counterfactual; preserve every model and blend weight.
    svc_topic_before_total = total.copy()
    total = adaptive_svc_topic_correction(total, svc_scores, groups)
    if not (np.isfinite(svc_scores).all() and np.isfinite(svc_topic_before_total).all() and np.isfinite(total).all()):
        raise FloatingPointError('Stage12B nonfinite topic-correction evidence')
    np.savez_compressed(work/'svc-topic-correction-evidence.npz', ids=np.asarray(ids), labels=np.asarray(LABELS),
                        groups=np.asarray(groups,dtype=np.int64), svc_scores=np.asarray(svc_scores,dtype=np.float64),
                        before_total=svc_topic_before_total, after_total=total)
"""
replacement="        total+=.25*clipped_logp(aligned(work/'deberta.npz',ids))\n"+insertion+"    if family=='part_two_complement_atlas_0636':"
modified=original.replace(needle,replacement,1)
common=source(codes[4]).split('\n',1)[1]
helper=HELPER.read_text()
helper=re.sub(r"EXPECTED_ORCHESTRATOR_SHA256 = '[^']*'", "EXPECTED_ORCHESTRATOR_SHA256 = '"+canonical(modified.split('\n',1)[1])+"'",helper,count=1)
helper=re.sub(r"EXPECTED_COMMON_SHA256 = '[^']*'", "EXPECTED_COMMON_SHA256 = '"+canonical(common)+"'",helper,count=1)
HELPER.write_text(helper)
new=copy.deepcopy(nb);new_codes=[c for c in new.cells if c.cell_type=='code'];new_codes[29].source=modified
new.cells[0].source="""# Stage12B — one fixed SVC topic-correction counterfactual

Parent: our official S05 V6 (362129392), public balanced accuracy **0.642**. Preserve all weights, assets, model cells and final balance; activate only the existing fixed topic correction after the complete SVC/decoder/DeBERTa log-vote sum and before complementary mixing.

The Stage12A component study improved by 2.149391 percentage points on reused prompt-disjoint development folds. Its SVC-only gate differs from the complete-ensemble gate here; this is one blind official experiment with an unknown gain, not full-S05 OOF evidence or a new scientific contribution. The three-row preview skips models and the intervention; it is format only.

Upstream Apache-2.0 credits retained. Public source: Lightning / Vladislav Lavrentev V28 362014737. Only upstream code cell30 changes; the original independent Stage11 output guard remains byte-for-byte unchanged. A final separate guard validates the intervention's evidence and production mixture.
"""
new.cells.append(nbformat.v4.new_markdown_cell('## Stage12B intervention validation\n\nReplay the one fixed correction from private saved numeric evidence. A format preview explicitly reports that the intervention was not executed.',id='stage12b-guard-introduction'))
guard = (
    "# Independent Stage12B evidence guard; no model loading or fitting.\n"
    "import hashlib\n"
    "INTERVENTION_VALIDATION_SOURCE = " + repr(helper) + "\n"
    "assert hashlib.sha256(INTERVENTION_VALIDATION_SOURCE.encode()).hexdigest() == '" + sha(helper.encode()) + "'\n"
    "intervention_validation_path=Path('/kaggle/working/stage12b_intervention_validation.py')\n"
    "intervention_validation_path.write_text(INTERVENTION_VALIDATION_SOURCE,encoding='utf-8')\n"
    "intervention_validation_namespace={'__name__':'stage12b_saved_intervention_guard','__file__':str(intervention_validation_path)}\n"
    "exec(compile(INTERVENTION_VALIDATION_SOURCE,str(intervention_validation_path),'exec'),intervention_validation_namespace)\n"
    "intervention_validation_report=intervention_validation_namespace['validate_intervention'](\n"
    "    '/kaggle/working', DATA, temporary_dir='/kaggle/temp/ntl636-fulltext-qwen-query',\n"
    "    runtime_dir='/kaggle/temp/ntl636-fulltext-qwen-external3200-runtime', namespace=globals())\n"
)
new.cells.append(nbformat.v4.new_code_cell(guard,id='stage12b-intervention-guard',execution_count=None,outputs=[]))
actual_metadata=json.loads((ROOT/'results/stage_11/FORMAT_PREVIEW_V6.json').read_text())['kaggle_metadata']
new.metadata['kaggle']=copy.deepcopy(actual_metadata)
new.metadata['kaggle']['isGpuEnabled']=True  # Intended candidate allocation, not a claim of actual execution.
new.metadata['parent_stage11_metadata']=new.metadata.pop('stage11')
new.metadata['stage12b']={'parent_own_script_version_id':362129392,'parent_own_official_score':.642,'upstream_source_version_id':362014737,'upstream_code_cells_unchanged':32,'upstream_code_cell_modified':30,'original_output_guard_code_cell':34,'original_output_guard_unchanged':True,'intervention_guard_code_cell':35,'coefficient':.62,'topic_threshold':.85,'one_official_test_only':True,'actual_model_execution_by_build':False,'preview_intervention_executed':False,'intended_accelerator':'GPU T4 x2','intended_internet':False,'input_source_versions_unchanged':28}
nbformat.validate(new)
new_codes=[c for c in new.cells if c.cell_type=='code']
assert len(new_codes)==35
changes=[i+1 for i in range(33) if source(codes[i])!=source(new_codes[i])]
assert changes==[30] and source(codes[33])==source(new_codes[33])
for c in new_codes:
    text=source(c)
    if text.startswith('%%writefile '):text=text.split('\n',1)[1]
    compile(text,'stage12b_static_syntax','exec')
nbformat.write(new,OUTPUT)
patch=''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='S05_code_cell30',tofile='Stage12B_code_cell30'))
(RESULT/'SOURCE_PATCH.diff').write_text(patch)
receipt={'stage':'12B','parent_notebook':'notebooks/Stage_11_Public_642_Reproduction.ipynb','parent_notebook_sha256':sha(BASE.read_bytes()),'parent_own_script_version_id':362129392,'parent_own_official_score':.642,'source_version_id':362014737,'new_notebook':'notebooks/Stage_12B_SVC_Topic_Correction.ipynb','new_notebook_sha256':sha(OUTPUT.read_bytes()),'code_cells':35,'original_source_code_cells':33,'unchanged_original_source_code_cells':32,'modified_original_code_cells':changes,'original_output_guard_unchanged':True,'original_output_guard_code_sha256':sha(source(codes[33]).encode()),'original_orchestrator_code_sha256':sha(original.encode()),'new_orchestrator_code_sha256':sha(modified.encode()),'new_orchestrator_canonical_file_sha256':canonical(modified.split('\n',1)[1]),'unchanged_common_canonical_file_sha256':canonical(common),'intervention_validation_module_sha256':sha(helper.encode()),'all_35_code_cells_static_compile_passed':True,'notebook_format_validated':True,'input_versions_unchanged':new.metadata.kaggle.dataSources==actual_metadata['dataSources'],'input_source_count':len(actual_metadata['dataSources']),'input_source_versions':actual_metadata['dataSources'],'intended_candidate_gpu_boolean':True,'parent_document_gpu_boolean':actual_metadata['isGpuEnabled'],'actual_gpu_allocation_verified_by_build':False,'models_executed':False,'classifier_fitted':False,'source_patch_sha256':sha(patch.encode()),'visibility':'private','blind_official_tests_declared':1}
(RESULT/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['stage','new_notebook_sha256','modified_original_code_cells','original_output_guard_unchanged','code_cells','input_source_count','models_executed']}))
