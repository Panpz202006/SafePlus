#!/usr/bin/env python3
"""Check a current publication patch on clean Tag2; no training or remote writes."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='57555b536f2395dd39d4c329917a7eef1acdf77c'
RELEASE='wiki300-v1.2-eval-v2.1-20261025'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def allowed(p):
    rel=p.relative_to(ROOT);parts=rel.parts
    if '__pycache__' in parts or '.ipynb_checkpoints' in parts:return False
    if parts[0] in ['artifacts','history','github_ready','review_revision_work']:return False
    if str(rel) in ['manifest_sha256.json','code/exp05_history.bundle','data_review/SafePlus_combined_current.jsonl','reports/packaging_validation.json','reports/RELEASE_VALIDATION.json']:return False
    if parts[:2] in [('data_review','evidence'),('logs','publication'),('HUMAN_REVIEW_QUEUE','accounts')]:return False
    if parts[:3]==('code','evaluation_v2','reports'):return False
    return str(rel)!='HUMAN_REVIEW_QUEUE/HUMAN_REVIEW_QUEUE.jsonl'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work-dir',type=Path,required=True);a=ap.parse_args();work=a.work_dir.resolve()
    if (work/'staging_repo').exists() or (work/'check_repo').exists():raise ValueError('Use a fresh --work-dir; existing repositories are never overwritten')
    work.mkdir(parents=True,exist_ok=True);out=ROOT/'github_ready';out.mkdir(exist_ok=True);logs=ROOT/'logs/publication';logs.mkdir(parents=True,exist_ok=True)
    commands=[];env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    def run(name,args,cwd=ROOT):
        q=subprocess.run(list(map(str,args)),cwd=cwd,env=env,capture_output=True,text=True)
        (logs/(name+'.log')).write_text(q.stdout+q.stderr,encoding='utf-8')
        commands.append({'name':name,'argv':list(map(str,args)),'cwd':str(cwd),'exit_code':q.returncode,'log':'logs/publication/'+name+'.log'})
        print(name,q.returncode,flush=True)
        if q.returncode:raise RuntimeError(name+' failed; inspect saved log')
    assets=[ROOT/'data_review/SafePlus_combined_current.jsonl',ROOT/'code/exp05_history.bundle',ROOT/'artifacts/exp05/frozen_source/SafePlus_combined_current.jsonl']
    (ROOT/'data_review/ASSET_REQUIREMENTS.json').write_text(json.dumps({'release_id':RELEASE,'installation':'Extract full milestone ZIP; code patch excludes source datasets, raw API caches, history and model artifacts','assets':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in assets}},ensure_ascii=False,indent=2)+'\n')
    tracked=sorted(p for p in ROOT.rglob('*') if p.is_file() and allowed(p))
    staging=work/'staging_repo';check=work/'check_repo';bundle=ROOT/'code/exp05_history.bundle'
    run('01_clone_adapter',['git','clone','--branch','work/wiki300-exp05-real',bundle,staging])
    target=staging/'milestones/wiki300_20261025'
    for p in tracked:
        dest=target/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    run('02_stage_milestone',['git','add','--force','milestones/wiki300_20261025'],staging)
    patch=subprocess.check_output(['git','diff','--cached','--binary'],cwd=staging);(out/'milestone_files.patch').write_bytes(patch)
    filemap={'milestones/wiki300_20261025/'+str(p.relative_to(ROOT)):sha(p) for p in tracked}
    (out/'INCLUDED_FILES.json').write_text(json.dumps({'release_id':RELEASE,'base_commit':BASE,'file_count':len(filemap),'files':filemap,'large_assets_in_complete_zip':True,'remote_writes':False},ensure_ascii=False,indent=2)+'\n')
    run('03_clone_clean_tag2',['git','clone','--branch','exp05-io-contract-v1',bundle,check])
    run('04_original_adapter_check',['git','apply','--check',ROOT/'artifacts/exp05/exp05_wiki_real.patch'],check)
    run('05_original_adapter_apply',['git','apply','--index',ROOT/'artifacts/exp05/exp05_wiki_real.patch'],check)
    run('06_current_milestone_check',['git','apply','--check',out/'milestone_files.patch'],check)
    run('07_current_milestone_apply',['git','apply','--index',out/'milestone_files.patch'],check)
    run('08_frozen_code_unchanged',['git','diff','--exit-code',BASE,'--','.',':(exclude)milestones/wiki300_20261025'],check)
    assert {name:sha(check/name) for name in filemap}==filemap,'Applied patch differs from current files'
    installed=check/'milestones/wiki300_20261025';n_assets=0
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if rel.parts[0] in ['review_revision_work','github_ready'] or '__pycache__' in rel.parts:continue
        if 'milestones/wiki300_20261025/'+str(rel) in filemap:continue
        dest=installed/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);n_assets+=1
    run('09_current_record_validation',[sys.executable,'scripts/validate_review_revision.py'],installed)
    run('10_current_review_tests',[sys.executable,'-m','unittest','discover','-s','tests','-p','test_review_revision.py','-v'],installed)
    run('11_frozen_queue_tests',[sys.executable,'-m','unittest','discover','-s','tests','-p','test_milestone_contract.py','-v'],installed)
    run('12_evaluation_v2_1_tests',[sys.executable,'-m','unittest','discover','-s','code/evaluation_v2/tests','-p','test_evaluate*.py','-v'],installed)
    def test_count(name):
        matches=re.findall(r'^Ran (\d+) tests? in ',(logs/(name+'.log')).read_text(),re.M)
        if len(matches)!=1:raise ValueError('Cannot establish actual test count: '+name)
        return int(matches[0])
    counts={name:test_count(name) for name in ['10_current_review_tests','11_frozen_queue_tests','12_evaluation_v2_1_tests']}
    for p in assets:assert sha(installed/p.relative_to(ROOT))==sha(p),'A dataset/history asset changed'
    validation=json.loads((installed/'data_review/VALIDATION_REPORT.json').read_text())
    result={'status':'PASS','release_id':RELEASE,'base_commit':BASE,'patch_file_count':len(filemap),'patch_sha256':sha(out/'milestone_files.patch'),'patch_bytes':len(patch),'patch_applied_on_clean_tag2_after_original_adapter':True,'patch_tracked_files_match_current_source':True,'frozen_model_code_unchanged':True,'current_review_validation':validation['status'],'current_review_conditions_passed':validation['checks_passed'],'current_review_tests':counts['10_current_review_tests'],'historical_queue_tests':counts['11_frozen_queue_tests'],'evaluation_tests':counts['12_evaluation_v2_1_tests'],'test_count_executed':sum(counts.values()),'runtime_assets_hydrated':n_assets,'real_model_trained':False,'io_converted':False,'samples_reaudited':False,'push_performed':False,'commands':commands}
    for p in [ROOT/'reports/RELEASE_VALIDATION.json',out/'PATCH_CHECK.json']:p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'STATUS.json').write_text(json.dumps({'status':'READY_FOR_REVIEW_NOT_PUSHED','release_id':RELEASE,'formal_review_revision':'review12_user_rules_v1_1_20261024','evaluation_version':'2.1','patch_sha256':result['patch_sha256'],'old_patch_location':'history/before_metrics_v2_1/github_ready/milestone_files.patch','current_patch_checked':True,'push_performed':False},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='commands'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
