from __future__ import annotations
import importlib.util, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE04A=Path(r"scripts/34_exp04_learned_behavior_evidence.py")
BASE03Y=Path(r"scripts/32_exp03_behavior_likelihood_identification.py")
OUT=Path(r"reports/experiment04/entropy_adaptive_fusion")
SEED=0
BETAS=[0.,.1,.25,.5,1.,2.,4.]
AMINS=[0.,.25,.5,1.,2.]
AMAXS=[.5,1.,2.,4.,8.]
BOOT=5000

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def med(c,p):
    return int(c[min(np.searchsorted(np.cumsum(p),.5),len(c)-1)])

def bucket(d):
    return "delay_1_2" if d<=2 else ("delay_3_5" if d<=5 else "delay_6_plus")

def ent(p):
    if len(p)<=1:return 0.
    h=-float(np.sum(p*np.log(np.maximum(p,1e-30))))
    return h/math.log(len(p))

def bscores(xu,td,pars,a):
    d=int(td); obs=xu[:d+1]; c=np.arange(0,d,dtype=np.int64)
    s=np.array([a.candidate_behavior_log_likelihood(obs,int(k),pars) for k in c],dtype=float)
    return c,s

def dscores(c,td,y):
    d=int(td)
    return np.array([math.log(max(y.delay_probability(d-int(k)),1e-30)) for k in c],dtype=float)

def gscale(x,lab,det,td,tr,pars,a):
    vals=[]
    for i in tr:
        if lab[i]!=1 or det[i]!=1: continue
        c,s=bscores(x[i],td[i],pars,a)
        if len(c)>1 and np.std(s)>1e-8: vals.append(np.std(s))
    if not vals: raise RuntimeError("No TRAIN scale values.")
    return max(float(np.median(vals)),1e-8)

def infer(xu,td,pars,scale,a,y,kind,beta=None,amin=None,amax=None):
    c,b=bscores(xu,td,pars,a); d=dscores(c,td,y)
    z=(b-b.mean())/scale
    pb=a.stable_softmax(z); H=ent(pb); conf=1-H
    if kind=="delay": q=d; alpha=0.
    elif kind=="behavior": q=z; alpha=np.nan
    elif kind=="fixed": alpha=float(beta); q=alpha*z+d
    else:
        alpha=float(amin)+(float(amax)-float(amin))*conf
        q=alpha*z+d
    p=a.stable_softmax(q)
    return med(c,p),float(np.sum(p*c)),int(c[np.argmax(p)]),H,conf,alpha

def evaluate(ids,x,det,td,tf,eid,pars,scale,a,y,kind,beta=None,amin=None,amax=None):
    rows=[]
    for i in ids:
        if det[i]!=1: continue
        m,mean,mode,H,conf,alpha=infer(x[i],td[i],pars,scale,a,y,kind,beta,amin,amax)
        t=int(tf[i]); d=int(td[i])
        rows.append(dict(index=int(i),entity_id=int(eid[i]),true_tf=t,observed_td=d,true_delay=d-t,
                         delay_bucket=bucket(d-t),median_tf=m,mean_tf=mean,mode_tf=mode,
                         entropy=H,confidence=conf,alpha=alpha))
    df=pd.DataFrame(rows)
    e=df.median_tf.to_numpy()-df.true_tf.to_numpy()
    return dict(n=len(df),mae=float(np.mean(abs(e))),signed=float(np.mean(e)),
                within1=float((abs(e)<=1).mean()),within2=float((abs(e)<=2).mean()),
                mean_alpha=float(np.nanmean(df.alpha.to_numpy(float))) if not np.all(pd.isna(df.alpha)) else np.nan),df

def boot(base,new,seed=0):
    imp=np.asarray(base)-np.asarray(new); rng=np.random.default_rng(seed); n=len(imp)
    vals=np.array([imp[rng.integers(0,n,n)].mean() for _ in range(BOOT)])
    return dict(mean_gain=float(imp.mean()),ci_low=float(np.quantile(vals,.025)),
                ci_high=float(np.quantile(vals,.975)),improved=float((imp>1e-10).mean()),
                worsened=float((imp<-1e-10).mean()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print("="*80); print("EXPERIMENT 04C: ENTROPY-BASED ADAPTIVE FUSION"); print("="*80)
    print("Entropy = 熵（低=Behavior更确定）")
    print("MAE = Mean Absolute Error（平均绝对误差，越小越好）")
    print("Signed Error = predicted tf - true tf（负数偏早，正数偏晚）")
    a=load(BASE04A,"a"); y=load(BASE03Y,"y")
    d=np.load(a.DATA_PATH,allow_pickle=True)
    x=d["x"].astype(float); det=d["detected"].astype(int); td=d["t_detect"].astype(int); tf=d["t_onset"].astype(int)
    lab=a.get_entity_fraud_label(d); eid=d["entity_id"].astype(int) if "entity_id" in d else np.arange(len(x))
    tr,va,te=a.split_indices(len(x),seed=SEED)
    print(f"Split: train={len(tr)} val={len(va)} test={len(te)}")
    pars,hist=a.train_emission_em(x,lab,det,td,tr)
    scale=gscale(x,lab,det,td,tr,pars,a); print(f"\nGlobal behavior scale from TRAIN = {scale:.6f}")
    fixed=[]
    for b in BETAS:
        s,_=evaluate(va,x,det,td,tf,eid,pars,scale,a,y,"fixed",beta=b)
        fixed.append(dict(beta=b,**s))
    fdf=pd.DataFrame(fixed); fdf["abs_signed"]=abs(fdf.signed)
    fs=fdf.sort_values(["mae","abs_signed","beta"]).iloc[0]
    beta=float(fs.beta)
    print(f"\nVALIDATION-SELECTED FIXED: beta={beta} | MAE={fs.mae:.6f} | signed={fs.signed:.6f}")
    adapt=[]
    print("\nVALIDATION ADAPTIVE GRID")
    for amin in AMINS:
        for amax in AMAXS:
            if amax<amin: continue
            s,_=evaluate(va,x,det,td,tf,eid,pars,scale,a,y,"adaptive",amin=amin,amax=amax)
            adapt.append(dict(alpha_min=amin,alpha_max=amax,**s))
            print(f"amin={amin:>4} amax={amax:>4} | MAE={s['mae']:.6f} | signed={s['signed']:.6f} | mean alpha={s['mean_alpha']:.6f}")
    adf=pd.DataFrame(adapt); adf["abs_signed"]=abs(adf.signed)
    sel=adf.sort_values(["mae","abs_signed","alpha_min","alpha_max"]).iloc[0]
    amin=float(sel.alpha_min); amax=float(sel.alpha_max)
    print(f"\nVALIDATION-SELECTED ADAPTIVE: alpha_min={amin} alpha_max={amax} | MAE={sel.mae:.6f} | signed={sel.signed:.6f}")
    cfg=[("delay_only","delay",{}),("behavior_only","behavior",{}),
         ("fixed_global_fusion","fixed",{"beta":beta}),
         ("entropy_adaptive_fusion","adaptive",{"amin":amin,"amax":amax})]
    rows=[]; dfs={}
    for name,kind,kw in cfg:
        s,df=evaluate(te,x,det,td,tf,eid,pars,scale,a,y,kind,**kw); rows.append(dict(method=name,**s)); dfs[name]=df
    overall=pd.DataFrame(rows)
    print("\nFIXED TEST OVERALL"); print(overall[["method","n","mae","signed","within1","within2","mean_alpha"]].to_string(index=False))
    br=[]
    for name,df in dfs.items():
        for bk in ["delay_1_2","delay_3_5","delay_6_plus"]:
            sub=df[df.delay_bucket==bk]; e=abs(sub.median_tf.to_numpy()-sub.true_tf.to_numpy())
            br.append(dict(delay_bucket=bk,method=name,n=len(sub),mae=float(e.mean()),
                           mean_alpha=float(np.nanmean(sub.alpha.to_numpy(float))) if not np.all(pd.isna(sub.alpha)) else np.nan))
    bdf=pd.DataFrame(br); print("\nTEST BY TRUE DELAY"); print(bdf.to_string(index=False))
    ad=dfs["entropy_adaptive_fusion"]; ae=abs(ad.median_tf.to_numpy()-ad.true_tf.to_numpy())
    pr=[]
    for name in ["delay_only","behavior_only","fixed_global_fusion"]:
        df=dfs[name]; be=abs(df.median_tf.to_numpy()-df.true_tf.to_numpy())
        pr.append(dict(comparison=f"adaptive_vs_{name}",**boot(be,ae,SEED)))
    pdf=pd.DataFrame(pr); print("\nPAIRED USER-LEVEL IMPROVEMENT"); print(pdf.to_string(index=False))
    hist.to_csv(OUT/"em_history.csv",index=False); fdf.to_csv(OUT/"validation_fixed.csv",index=False)
    adf.to_csv(OUT/"validation_adaptive.csv",index=False); overall.to_csv(OUT/"test_overall.csv",index=False)
    bdf.to_csv(OUT/"test_by_delay.csv",index=False); pdf.to_csv(OUT/"test_paired.csv",index=False)
    for name,df in dfs.items(): df.to_csv(OUT/f"test_{name}.csv",index=False)

if __name__=="__main__":
    main()
