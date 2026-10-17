from __future__ import annotations
import importlib.util, math
from pathlib import Path
import numpy as np, pandas as pd

BASE04A=Path(r"scripts/34_exp04_learned_behavior_evidence.py")
BASE03Y=Path(r"scripts/32_exp03_behavior_likelihood_identification.py")
OUT=Path(r"reports/experiment04/learned_behavior_delay_fusion")
SEED=0
BETAS=[0.0,0.1,0.25,0.5,1.0,2.0,4.0]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def norm(s):
    s=np.asarray(s,float); sd=float(s.std())
    return np.zeros_like(s) if sd<1e-8 else (s-s.mean())/sd


def med(c,p):
    return int(c[min(int(np.searchsorted(np.cumsum(p),0.5)),len(c)-1)])


def bucket(d):
    return 'delay_1_2' if d<=2 else ('delay_3_5' if d<=5 else 'delay_6_plus')


def infer(xu,td,params,beta,a,y):
    d=int(td); obs=xu[:d+1]; c=np.arange(0,d,dtype=np.int64)
    bs=[]; ds=[]
    for k in c:
        bs.append(a.candidate_behavior_log_likelihood(obs,int(k),params))
        dp=max(y.delay_probability(d-int(k)),1e-30); ds.append(math.log(dp))
    bs=norm(bs); ds=np.asarray(ds,float)
    post=a.stable_softmax(beta*bs+ds)
    bp=a.stable_softmax(bs); dp=a.stable_softmax(ds)
    def est(p):
        return float(np.sum(p*c)),med(c,p),int(c[np.argmax(p)])
    pm,pmed,pmo=est(post); bm,bmed,bmo=est(bp); dm,dmed,dmo=est(dp)
    return dict(posterior_mean_tf=pm,posterior_median_tf=pmed,posterior_mode_tf=pmo,
                behavior_mean_tf=bm,behavior_median_tf=bmed,behavior_mode_tf=bmo,
                delay_mean_tf=dm,delay_median_tf=dmed,delay_mode_tf=dmo)


def eval_beta(beta,idxs,x,det,td,tf,eid,params,a,y):
    rows=[]
    for i in idxs:
        if det[i]!=1: continue
        pr=infer(x[i],td[i],params,beta,a,y); t=int(tf[i]); d=int(td[i])
        rows.append(dict(index=int(i),entity_id=int(eid[i]),beta=beta,true_tf=t,observed_td=d,
                         true_delay=d-t,delay_bucket=bucket(d-t),**pr))
    df=pd.DataFrame(rows)
    out={}
    for col in ['posterior_mean_tf','posterior_median_tf','posterior_mode_tf']:
        e=df[col].to_numpy()-df.true_tf.to_numpy(); ae=np.abs(e)
        out[col]=dict(mae=float(ae.mean()),signed_error=float(e.mean()),within_1=float((ae<=1).mean()),within_2=float((ae<=2).mean()))
    return out,df


def summarize(df,name,col):
    e=df[col].to_numpy()-df.true_tf.to_numpy(); ae=np.abs(e)
    return dict(method=name,n=len(df),mae=float(ae.mean()),signed_error=float(e.mean()),within_1=float((ae<=1).mean()),within_2=float((ae<=2).mean()))


def boot(base,new,seed):
    imp=np.asarray(base)-np.asarray(new); rng=np.random.default_rng(seed); n=len(imp)
    vals=np.array([imp[rng.integers(0,n,n)].mean() for _ in range(5000)])
    return dict(mean_gain=float(imp.mean()),ci_low=float(np.quantile(vals,.025)),ci_high=float(np.quantile(vals,.975)),
                improved_rate=float((imp>1e-10).mean()),worsened_rate=float((imp<-1e-10).mean()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print('='*80); print('EXPERIMENT 04B: LEARNED BEHAVIOR + DELAY FUSION'); print('='*80)
    print('MAE = Mean Absolute Error（平均绝对误差，越小越好）')
    print('Signed Error = predicted tf - true tf（负数偏早，正数偏晚）')
    print('Validation = 验证集，只用于选 beta；Test = 测试集，只做最终评估')
    a=load(BASE04A,'exp04a'); y=load(BASE03Y,'exp03y')
    data=np.load(a.DATA_PATH,allow_pickle=True); x=data['x'].astype(float); det=data['detected'].astype(int); td=data['t_detect'].astype(int); tf=data['t_onset'].astype(int)
    labels=a.get_entity_fraud_label(data); eid=data['entity_id'].astype(int) if 'entity_id' in data else np.arange(len(x))
    tr,va,te=a.split_indices(len(x),seed=SEED)
    print(f'Split: train={len(tr)} val={len(va)} test={len(te)}')
    params,hist=a.train_emission_em(x,labels,det,td,tr)

    print('\nVALIDATION BETA SWEEP')
    vr=[]
    for b in BETAS:
        sm,_=eval_beta(b,va,x,det,td,tf,eid,params,a,y); m=sm['posterior_median_tf']
        vr.append(dict(beta=b,median_mae=m['mae'],median_signed_error=m['signed_error'],mean_mae=sm['posterior_mean_tf']['mae'],mode_mae=sm['posterior_mode_tf']['mae']))
        print(f"beta={b:>4} | median MAE={m['mae']:.6f} | signed={m['signed_error']:.6f} | mean MAE={sm['posterior_mean_tf']['mae']:.6f} | mode MAE={sm['posterior_mode_tf']['mae']:.6f}")
    vdf=pd.DataFrame(vr); vdf['abs_signed']=vdf.median_signed_error.abs(); best=vdf.sort_values(['median_mae','abs_signed','beta']).iloc[0]; beta=float(best.beta)
    print(f'\nSelected beta on VALIDATION = {beta}')

    sm,test=eval_beta(beta,te,x,det,td,tf,eid,params,a,y)
    methods=pd.DataFrame([
        summarize(test,'delay_only','delay_median_tf'),
        summarize(test,'learned_behavior_only','behavior_median_tf'),
        summarize(test,'learned_behavior_plus_delay','posterior_median_tf')])
    print('\nPRIMARY TEST COMPARISON (posterior median)'); print(methods.to_string(index=False))

    er=[]
    for col in ['posterior_mean_tf','posterior_median_tf','posterior_mode_tf']:
        er.append(dict(estimator=col,selected_beta=beta,**sm[col]))
    edf=pd.DataFrame(er); print('\nFUSION MEAN / MEDIAN / MODE'); print(edf.to_string(index=False))

    br=[]
    for bu in ['delay_1_2','delay_3_5','delay_6_plus']:
        sub=test[test.delay_bucket==bu]
        for name,col in [('delay_only','delay_median_tf'),('learned_behavior_only','behavior_median_tf'),('learned_behavior_plus_delay','posterior_median_tf')]:
            ae=np.abs(sub[col].to_numpy()-sub.true_tf.to_numpy()); br.append(dict(delay_bucket=bu,method=name,n_users=len(sub),mae=float(ae.mean())))
    bdf=pd.DataFrame(br); print('\nTEST BY TRUE DELAY'); print(bdf.to_string(index=False))

    truth=test.true_tf.to_numpy(); de=np.abs(test.delay_median_tf.to_numpy()-truth); be=np.abs(test.behavior_median_tf.to_numpy()-truth); fe=np.abs(test.posterior_median_tf.to_numpy()-truth)
    pdf=pd.DataFrame([dict(comparison='fusion_vs_delay_only',**boot(de,fe,SEED)),dict(comparison='fusion_vs_behavior_only',**boot(be,fe,SEED+1))])
    print('\nPAIRED USER-LEVEL IMPROVEMENT'); print(pdf.to_string(index=False))

    hist.to_csv(OUT/'em_history.csv',index=False); vdf.to_csv(OUT/'validation_beta_sweep.csv',index=False); test.to_csv(OUT/'test_entity_predictions.csv',index=False); methods.to_csv(OUT/'test_method_summary.csv',index=False); edf.to_csv(OUT/'test_fusion_estimators.csv',index=False); bdf.to_csv(OUT/'test_delay_bucket_summary.csv',index=False); pdf.to_csv(OUT/'test_user_level_improvement.csv',index=False)
    print('\nInterpretation: fusion is strongest if MAE improves over both single evidences, signed error moves toward 0, long-delay performance stays strong, and bootstrap CI is >0.')

if __name__=='__main__': main()
