"""Independent no-fit contract fixtures for prospective Stage06E evidence.

Only synthetic numeric/string arrays and source inspection are used. No model is
fitted, loaded or invoked. The main block writes an auditable fixture report.
"""
from __future__ import annotations
import copy
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import stage06e_cache_checks as checks
import stage06e_cached_ensemble as consumer


def synthetic_capture():
    # Five complete prompt folds, two classes, four passages per prompt.
    n, c = 20, 2
    folds = np.repeat(np.arange(1, 6), 4)
    groups = np.repeat(np.asarray(['p0','p1','p2','p3','p4']), 4)
    ids = np.asarray([f'id{i}' for i in range(n)])
    y = np.tile([0, 1, 0, 1], 5)
    p = np.empty((3, n, c), dtype=np.float64)
    p[:] = np.asarray([[.8,.2],[.3,.7],[1.,0.],[.4,.6]] * 5)
    logp = np.log(np.clip(p, 1e-9, 1))
    batches = np.tile(np.asarray([np.arange(4), np.arange(4)//2, np.zeros(4,int)]), (1,5))
    sizes = np.tile(np.asarray([[1]*4,[2]*4,[4]*4]), (1,5))
    active = (sizes >= 2) & (sizes <= 18)
    a = dict(ids=ids, classes=np.asarray(['A','B']), y=y, groups=groups, fold=folds,
             scenarios=np.asarray([1,3,6]), plain_probability=p,
             plain_log_probability=logp, plain_coverage=np.ones((3,n),int),
             context_coverage=np.ones((3,n),int), batches=batches,
             global_batches=batches + folds[None]*1000000)
    for mode in ('inferred','oracle'):
        a[mode]=batches.copy()
        a['cluster_size_'+mode]=sizes.copy()
        a['eligible_'+mode]=active.copy()
    for key in checks.KEYS:
        cp=np.zeros_like(p)
        cp[active]=np.tile([.6,.4], (active.sum(),1))
        if key.endswith('candidate'):
            cp[1,1]=[.1,.9]
        cl=np.zeros_like(cp)
        cl[active]=np.log(np.clip(cp[active],1e-9,1))
        evidence=.05*logp.copy()
        evidence[active]+=.25*cl[active]
        a['centered_probability_'+key]=cp
        a['centered_log_probability_'+key]=cl
        a['component_log_evidence_'+key]=evidence
        a['prediction_'+key]=evidence.argmax(-1)
        a['component_coverage_'+key]=np.ones((3,n),int)
    return a


class CanonicalContract(unittest.TestCase):
    def setUp(self): self.a = synthetic_capture()
    def reject(self, change):
        a=copy.deepcopy(self.a); change(a)
        with self.assertRaises((AssertionError,ValueError,IndexError,TypeError,KeyError)):
            checks.validate_canonical_arrays(a)
    def test_valid_contract_and_independent_confusion(self):
        result=checks.validate_canonical_arrays(self.a)
        for key in checks.KEYS:
            for s in range(3):
                pred=self.a['prediction_'+key][s]
                expected=np.mean([np.mean(pred[self.a['y']==k]==k) for k in range(2)])
                self.assertEqual(expected,result[key][s]['balanced_accuracy'])
        # A zero probability must retain the original clipping floor exactly.
        self.assertEqual(self.a['plain_log_probability'][0,2,1],np.log(1e-9))
        self.assertTrue(np.all(self.a['centered_log_probability_control'][0]==0))
    def test_duplicate_ids_rejected(self): self.reject(lambda a:a['ids'].__setitem__(1,a['ids'][0]))
    def test_duplicate_classes_rejected(self): self.reject(lambda a:a['classes'].__setitem__(1,a['classes'][0]))
    def test_probability_normalization_rejected(self): self.reject(lambda a:a['plain_probability'].__setitem__((0,0,0),.7))
    def test_log_probability_roundtrip_tamper_rejected(self): self.reject(lambda a:a['plain_log_probability'].__setitem__((0,0,0),-1.))
    def test_nonfinite_rejected(self): self.reject(lambda a:a['component_log_evidence_control'].__setitem__((1,0,0),np.nan))
    def test_inactive_normalized_probability_rejected(self): self.reject(lambda a:a['centered_probability_control'].__setitem__((0,0),[.5,.5]))
    def test_inactive_log_contribution_rejected(self): self.reject(lambda a:a['centered_log_probability_control'].__setitem__((0,0,0),-.1))
    def test_eligibility_tamper_rejected(self): self.reject(lambda a:a['eligible_inferred'].__setitem__((0,0),True))
    def test_cluster_size_tamper_rejected(self): self.reject(lambda a:a['cluster_size_inferred'].__setitem__((1,0),3))
    def test_clusters_crossing_batches_rejected(self): self.reject(lambda a:a['inferred'].__setitem__((0,0),1))
    def test_global_batch_fold_collision_rejected(self): self.reject(lambda a:a['global_batches'].__setitem__((1,0),1000001))
    def test_prediction_tamper_rejected(self): self.reject(lambda a:a['prediction_candidate'].__setitem__((2,0),1-self.a['prediction_candidate'][2,0]))
    def test_component_evidence_tamper_rejected(self): self.reject(lambda a:a['component_log_evidence_candidate'].__setitem__((1,0,0),-.1))
    def test_plain_coverage_missing_and_duplicate_rejected(self):
        for x in (0,2):self.reject(lambda a:a['plain_coverage'].__setitem__((0,0),x))
    def test_component_coverage_duplicate_rejected(self): self.reject(lambda a:a['component_coverage_control'].__setitem__((1,0),2))
    def test_negative_batch_rejected(self):
        def change(a):
            a['batches'][0,0]=-1;a['global_batches'][0,0]=999999
        self.reject(change)
    def test_whole_prompt_split_fold_rejected(self):
        def change(a): a['groups'][4]=a['groups'][0]
        self.reject(change)
    def test_context_coverage_missing_rejected(self): self.reject(lambda a:a['context_coverage'].__setitem__((0,0),0))
    def test_wrong_scores_rejected(self):
        scores={key:[.5,.5,.5] for key in checks.KEYS}
        scores['control'][0]=.123
        with self.assertRaises(AssertionError):checks.validate_canonical_arrays(self.a,scores)
    def test_reference_fold_drift_rejected(self):
        refs={f'p{k}':k+1 for k in range(5)}; refs['p0']=2
        with self.assertRaises(AssertionError):checks.validate_canonical_arrays(self.a,reference_folds=refs)
    def test_readback_schema_values_and_object_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'capture.npz'
            np.savez_compressed(path,**self.a)
            self.assertTrue(checks.exact_saved_arrays(path,self.a))
            bad=copy.deepcopy(self.a);bad['y']=bad['y'].astype(float)
            np.savez_compressed(path,**bad)
            with self.assertRaises(AssertionError):checks.exact_saved_arrays(path,self.a)
            bad=copy.deepcopy(self.a);bad['classes']=bad['classes'].astype(object)
            np.savez_compressed(path,**bad)
            with self.assertRaises((AssertionError,ValueError)):checks.exact_saved_arrays(path,bad)


class SourcePreservationContract(unittest.TestCase):
    @staticmethod
    def tree(name):return ast.parse((ROOT/'scripts'/name).read_text())
    @staticmethod
    def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    def test_original_grouping_partition_centering_unchanged(self):
        original=self.tree('stage06c_mixed_confirmation.py')
        captured=self.tree('stage06e_capture_original.py')
        for name in ('centered','partition_training','contexts','safe_ids'):
            self.assertEqual(ast.dump(self.function(original,name),include_attributes=False),
                             ast.dump(self.function(captured,name),include_attributes=False))
    def test_original_fit_and_logprob_preserved(self):
        original=self.tree('stage06c_mixed_confirmation.py')
        captured=self.tree('stage06e_capture_original.py')
        f=copy.deepcopy(self.function(captured,'fit_head'))
        f.body=[n for n in f.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
            and n.value.func.value.id=='CAP')]
        self.assertEqual(ast.dump(self.function(original,'fit_head'),include_attributes=False),ast.dump(f,include_attributes=False))
        lf=self.function(captured,'logprob')
        p=next(n.value for n in lf.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='probability')
        value=copy.deepcopy(next(n.value for n in lf.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='value'))
        class Inline(ast.NodeTransformer):
            def visit_Name(self,node):return copy.deepcopy(p) if node.id=='probability' else node
        value=Inline().visit(value)
        expected=next(n.value for n in self.function(original,'logprob').body if isinstance(n,ast.Return))
        self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(value,include_attributes=False))
    def test_original_main_loop_preserved_after_capture_removal(self):
        old=self.tree('stage06c_mixed_confirmation.py');new=self.tree('stage06e_capture_original.py')
        def loop(t):return next(n for n in t.body if isinstance(n,ast.For) and ast.unparse(n.target)=='(number, (fit, valid))')
        copied=copy.deepcopy(loop(new))
        plain_call=ast.parse('plain_clf.predict_proba(standardized)',mode='eval').body
        class RemoveCapture(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='CAP':return None
                return self.generic_visit(node)
            def visit_For(self,node):
                if 'capture_label' in ast.unparse(node.target):return None
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='probability':return None
                return self.generic_visit(node)
            def visit_Name(self,node):return copy.deepcopy(plain_call) if node.id=='probability' else node
        copied=RemoveCapture().visit(copied)
        self.assertEqual(ast.dump(loop(old),include_attributes=False),ast.dump(copied,include_attributes=False))
    def test_no_extra_inference_or_rng_sites(self):
        def calls(t):
            return [ast.unparse(n) for n in ast.walk(t) if isinstance(n,ast.Call) and
                (ast.unparse(n.func).endswith('.predict_proba') or ast.unparse(n.func).startswith('np.random.'))]
        self.assertEqual(sorted(calls(self.tree('stage06c_mixed_confirmation.py'))),sorted(calls(self.tree('stage06e_capture_original.py'))))
        consumer_tree=self.tree('stage06e_cached_ensemble.py')
        banned=[]
        for n in ast.walk(consumer_tree):
            if isinstance(n,ast.Call):
                func=ast.unparse(n.func)
                if any(func.endswith('.'+v) for v in ('fit','fit_transform','predict','predict_proba','transform')):banned.append(func)
                if func in ('linkage','fcluster','LogisticRegression','StandardScaler','TfidfVectorizer'):banned.append(func)
        self.assertEqual(banned,[])


class InertPackageContract(unittest.TestCase):
    def test_five_cells_metadata_and_exact_source_literals(self):
        nb=json.loads((ROOT/'notebooks/Stage_06E_Canonical_CPU_Evidence.ipynb').read_text())
        old=json.loads((ROOT/'notebooks/Stage_06D_Fixed_Ensemble.ipynb').read_text())
        def source(cell):return ''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
        self.assertEqual(len(nb['cells']),5)
        cells=[source(c) for c in nb['cells']]
        self.assertEqual(cells[:4],[source(c) for c in old['cells'][:4]])
        self.assertTrue(all(len(c)<190000 for c in cells))
        self.assertEqual((ROOT/'scripts/kaggle_stage06e_multicell.py').read_text(),''.join(cells))
        for i,c in enumerate(cells):compile(c,f'cell{i}','exec')
        self.assertTrue(all(c['outputs']==[] and c['execution_count'] is None for c in nb['cells']))
        metadata=nb['metadata']['kaggle']
        self.assertEqual(metadata['accelerator'],'none')
        self.assertFalse(metadata['isInternetEnabled']);self.assertFalse(metadata['isGpuEnabled'])
        self.assertEqual([i['sourceId'] for i in metadata['dataSources']],checks.EXPECTED_INPUT_VERSIONS)
        tree=ast.parse(cells[4]);values={}
        for n in tree.body:
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
                if n.targets[0].id in ('STAGE06E_SOURCE_TEXT','STAGE06E_FILE_SHA256','STAGE06E_EXPECTED_SOURCES'):
                    values[n.targets[0].id]=ast.literal_eval(n.value)
        self.assertEqual(set(values),{'STAGE06E_SOURCE_TEXT','STAGE06E_FILE_SHA256','STAGE06E_EXPECTED_SOURCES'})
        expected=values['STAGE06E_EXPECTED_SOURCES']
        paths={'producer':'scripts/stage06e_capture_original.py','capture_helper':'scripts/stage06e_cache_checks.py',
               'static_preparation':'scripts/stage06e_static_members.py','consumer':'scripts/stage06e_cached_ensemble.py',
               'original_C':'scripts/stage06c_mixed_confirmation.py','plan':'plans/STAGE_06E_CANONICAL_CPU_EVIDENCE.md',
               'freeze':'results/stage_06e/EXECUTION_FREEZE.json'}
        self.assertEqual(set(expected),set(paths))
        for k,p in paths.items():
            name=Path(p).name
            content=values['STAGE06E_SOURCE_TEXT'][name].encode()
            self.assertEqual(content,(ROOT/p).read_bytes())
            self.assertEqual(expected[k],checks.sha256(ROOT/p))
            self.assertEqual(values['STAGE06E_FILE_SHA256'][name],expected[k])
        freeze=json.loads(values['STAGE06E_SOURCE_TEXT']['EXECUTION_FREEZE.json'])
        self.assertEqual(freeze['capture_source_sha256'],expected['producer'])
        self.assertEqual(freeze['cache_checks_source_sha256'],expected['capture_helper'])
        self.assertEqual(freeze['consumer_source_sha256'],expected['consumer'])
        self.assertEqual(freeze['static_preparation_source_sha256'],expected['static_preparation'])
        self.assertEqual(freeze['plan_sha256'],expected['plan'])


class FrozenFusionContract(unittest.TestCase):
    def test_all_seven_literal_weights(self):
        self.assertEqual(consumer.WEIGHTS, {'stylegram':1.516,'lora':.177,'lora2':.113,
            'tokattn':.153,'tokattn_deb':.185,'deb_plain':.05,'deb_center':.25})
    def test_fusion_scalar_oracle_and_inactive_rows(self):
        rng=np.random.default_rng(106)
        n=18
        def p(shape):
            a=rng.uniform(.01,1,shape);return a/a.sum(-1,keepdims=True)
        static={}
        for key in consumer.MEMBERS:
            static['probability_'+key]=p((n,18))
            static['fold_'+key]=np.ones(n,int)
        plain=np.log(p((3,n,18)))
        center0=np.log(p((3,n,18)))
        center1=np.log(p((3,n,18)))
        mask=np.tile(np.arange(n)%2==0,(3,1))
        canonical=dict(ids=np.asarray([str(k) for k in range(n)]),fold=np.ones(n,int),
            eligible_inferred=mask,plain_log_probability=plain,
            centered_log_probability_control=center0,centered_log_probability_candidate=center1,
            global_batches=np.tile(np.repeat([1000000,1000001],9),(3,1)))
        actual=consumer._fixed_fusion(canonical,static)
        for name,center in [('control',center0),('candidate',center1)]:
            expected=np.zeros((3,n,18))
            for s in range(3):
                for r in range(n):
                    for k in range(18):
                        base=1.516*np.log(static['probability_stylegram'][r,k])
                        base+=.177*np.log(static['probability_lora'][r,k])
                        base+=.113*np.log(static['probability_lora2'][r,k])
                        base+=.153*np.log(static['probability_tokattn'][r,k])
                        base+=.185*np.log(static['probability_tokattn_deb'][r,k])
                        base+=.05*plain[s,r,k]
                        if mask[s,r]:base+=.25*center[s,r,k]
                        expected[s,r,k]=base
            np.testing.assert_array_equal(actual['evidence'][name],expected)
        np.testing.assert_array_equal(actual['evidence']['control'][~mask],actual['evidence']['candidate'][~mask])
        self.assertEqual(len(actual['balancing_records']),12)
    def test_exact_frozen_balance_and_batch_separation(self):
        tree=ast.parse((ROOT/'scripts/stage06d_fixed_ensemble.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage06d_balance')
        scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<frozen-D-balance>','exec'),scope)
        frozen=scope['stage06d_balance']
        value=np.asarray([[9.,0.,-3.],[8.,1.,-4.],[0.,1.,2.],[0.,0.,6.]])
        np.testing.assert_array_equal(consumer.stage06e_balance(value),frozen(value))
        np.testing.assert_array_equal(consumer.stage06e_balance(value[:1]),frozen(value[:1]))
        separate=np.concatenate([consumer.stage06e_balance(value[:2]),consumer.stage06e_balance(value[2:])])
        self.assertFalse(np.allclose(separate,consumer.stage06e_balance(value),rtol=0,atol=1e-6))
    def test_bootstrap_independent_passage_resample(self):
        y=np.tile(np.arange(18),3)
        groups=np.repeat(np.asarray(['g0','g1','g2']),18)
        ctrl=np.tile(y,(3,1))
        cand=ctrl.copy()
        # Control errs on two rows, candidate corrects both and breaks one.
        ctrl[:,0]=1;ctrl[:,19]=2;cand[:,38]=3
        pred={'raw_control':ctrl,'raw_candidate':cand,'balanced_control':ctrl,'balanced_candidate':cand}
        def ba(p):return float(np.mean([np.mean(p[y==k]==k) for k in range(18)]))
        scores={k:[ba(row) for row in v] for k,v in pred.items()}
        result=consumer._paired_intervals(y,groups,pred,scores)
        delta=((cand==y[None]).astype(float)-(ctrl==y[None]).astype(float)).mean(0)
        draws=np.random.default_rng(20261017).integers(3,size=(3000,3))
        per_group=[np.flatnonzero(groups==f'g{k}') for k in range(3)]
        oracle=[]
        for draw in draws:
            rows=np.concatenate([per_group[k] for k in draw])
            oracle.append(np.mean([np.mean(delta[rows][y[rows]==k]) for k in range(18)]))
        interval=np.quantile(oracle,[.025,.975])
        for endpoint in ('raw','balanced'):
            self.assertEqual(result[endpoint]['bootstrap_draws'],3000)
            self.assertEqual(result[endpoint]['bootstrap_seed'],20261017)
            self.assertAlmostEqual(result[endpoint]['mean_gain'],1/54)
            np.testing.assert_allclose(result[endpoint]['conditional_prompt_95_interval'],interval,rtol=0,atol=1e-16)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    output=ROOT/'results/stage_06e/INDEPENDENT_CONTRACT_FIXTURES.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        model_fits=0,model_inference_calls=0,kind='independent synthetic corruption fixtures',
        checks_source_sha256=checks.sha256(ROOT/'scripts/stage06e_cache_checks.py'),
        consumer_source_sha256=checks.sha256(ROOT/'scripts/stage06e_cached_ensemble.py')),
        indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
