#!/usr/bin/env python3
"""Offline milestone verification using frozen predictions. Never fits real data."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, platform, re, shutil, subprocess, sys
from pathlib import Path

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    logs=out/'logs';logs.mkdir(exist_ok=True); art=root/'artifacts/exp05';v2=root/'code/evaluation_v2'
    protected={str(x.relative_to(root)):digest(x) for d in [art,v2] for x in d.rglob('*') if x.is_file() and '__pycache__' not in x.parts}
    results=[]
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    def command(name,args,cwd=root):
        q=subprocess.run([str(x) for x in args],cwd=cwd,env=env,capture_output=True,text=True)
        text=q.stdout+q.stderr;(logs/(name+'.log')).write_text(text,encoding='utf-8')
        tests=re.search(r'Ran (\d+) tests?',text)
        row={'name':name,'argv':[str(x) for x in args],'cwd':str(cwd),'exit_code':q.returncode,'log':'logs/'+name+'.log','tests_run':int(tests.group(1)) if tests else None}
        results.append(row);print(name,'exit',q.returncode,'tests',row['tests_run'],flush=True)
        if q.returncode: raise RuntimeError(f'{name} failed; inspect {logs/(name+".log")}')
    report={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'real_training_performed':False,'network_required':False,'commands':results}
    try:
        repo=out/'verification_repo'
        if repo.exists(): raise ValueError('Use a new --out directory; an existing verification_repo will not be overwritten')
        command('01_offline_clone',['git','clone','--branch','work/wiki300-exp05-real',root/'code/exp05_history.bundle',repo])
        command('02_checkout_base',['git','switch','--detach','exp05-io-contract-v1'],repo)
        command('03_patch_check',['git','apply','--check',art/'exp05_wiki_real.patch'],repo)
        command('04_patch_apply',['git','apply','--index',art/'exp05_wiki_real.patch'],repo)
        # Validate reconstructed tracked files byte-for-byte against the archived adapter commit.
        command('05_code_tree_matches_adapter',['git','diff','--exit-code','57555b536f2395dd39d4c329917a7eef1acdf77c'],repo)
        command('06_validate_real_io',[sys.executable,'scripts/45_exp04_validate_io.py','--bundle',art/'data_bundle'],repo)
        command('07_existing_exp04_exp05_tests',[sys.executable,'-m','unittest','discover','-s','tests','-p','test_exp0*.py','-v'],repo)
        command('08_evaluation_v2_tests',[sys.executable,'-m','unittest','discover','-s','tests','-p','test_evaluate_addon.py','-v'],v2)
        v1=out/'original_scorer';v1.mkdir()
        for fn in ['predictions.csv','run_manifest.json']:shutil.copy2(art/'real_run'/fn,v1/fn)
        command('09_original_scoring',[sys.executable,'scripts/50_exp05_evaluate_wiki.py','--bundle',art/'data_bundle','--predictions',v1],repo)
        command('10_evaluation_v2_scoring',[sys.executable,v2/'scripts/evaluate_addon.py','--bundle',art/'data_bundle','--run',art/'real_run','--source',art/'frozen_source/SafePlus_combined_current.jsonl','--out',out/'evaluation_v2'])
        command('11_dataset_audit',[sys.executable,root/'scripts/audit_dataset.py','--artifact',art,'--out',out/'audit'])
        command('12_milestone_contract_tests',[sys.executable,'-m','unittest','discover','-s',root/'tests','-p','test_milestone_contract.py','-v'])
        command('13_review_queue_rebuild',[sys.executable,root/'scripts/build_review_queue.py','--out',out/'review_queue_rebuilt'])
        queue_reference=root/'history/review_before_rules_v1_1/HUMAN_REVIEW_QUEUE'
        if not queue_reference.exists():queue_reference=root/'HUMAN_REVIEW_QUEUE'
        for fn in ['HUMAN_REVIEW_QUEUE.jsonl','HUMAN_REVIEW_QUEUE.csv','SELECTION_MANIFEST.json','REVIEW_ALL_12.html']:
            assert (queue_reference/fn).read_bytes()==(out/'review_queue_rebuilt'/fn).read_bytes(),f'Frozen-v1 review output not reproducible: {fn}'
        report['review_queue_reproduced_exactly']=True
        old=json.loads((art/'real_run/wiki_metrics.json').read_text());new=json.loads((v1/'wiki_metrics.json').read_text())
        assert old==new,'Original scorer metrics differ'
        comparisons={}
        for fn in ['exploratory_score_v2.json','dataset_quality_audit_v2.json','per_account_exploratory_v2.csv','independent_review_queue_36.csv','train_prior_evidence_review_36.csv']:
            x=v2/'reports'/fn;y=out/'evaluation_v2'/fn
            same=json.loads(x.read_text())==json.loads(y.read_text()) if x.suffix=='.json' else x.read_bytes()==y.read_bytes()
            comparisons[fn]={'equal':same,'old_sha256':digest(x),'new_sha256':digest(y)}
            assert same,f'v2 output differs: {fn}'
        report['scoring_matches_saved_original']=True;report['v2_saved_output_comparison']=comparisons
        after={rel:digest(root/rel) for rel in protected};report['protected_files_unchanged']=protected==after
        assert report['protected_files_unchanged'],'A frozen artifact changed'
        report['test_count']=sum(r['tests_run'] or 0 for r in results);report['status']='PASS'
    except Exception as e:
        report['status']='FAIL';report['error']=repr(e)
    finally:
        report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out/'acceptance_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
