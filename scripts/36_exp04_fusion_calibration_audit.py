from __future__ import annotations
import importlib.util, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE04A=Path(r"scripts/34_exp04_learned_behavior_evidence.py")
BASE03Y=Path(r"scripts/32_exp03_behavior_likelihood_identification.py")
OUT=Path(r"reports/experiment04/fusion_calibration_audit")
BETAS=[0.0,0.1,0.25,0.5,1.0,2.0,4.0]
SEED=0

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def med(c,p):
    return int(c[min(int(np.searchsorted(np.cumsum(p),0.5)),len(c)-1)])

def bucket(d):
    return "delay_1_2" if d<=2 else ("delay_3_5" if d<=5 else "delay_6_plus")

def arrays(xu,d,params,a,y):
    c=np.arange(0,int(d),dtype=np.int64); obs=xu[:int(d)+1]
    b=np.asarray([a.candidate_behavior_log_likelihood(obs,int(k),params) for k in c],dtype=float)
    q=np.asarray([math.log(max(y.delay_probability(int(d)-int(k)),1e-30)) for k in c],dtype=float)
    return c,b,q

def global_scale(x,det,td,idx,params,a,y):
    s=[]
    for i in idx:
        if det[i]!=1 or int(td[i])<2: continue
        _,b,_=arrays(x[i],td[i],params,a,y)
        z=float(np.std(b))
        if z>1e-8: s.append(z)
    if not s: raise RuntimeError("No valid train behavior scales.")
    return float(np.median(s))

def infer(xu,d,params,a,y,strategy,beta,gscale):
    c,b,q=arrays(xu,d,params,a,y)
    if strategy=="per_user_z":
        z=float(np.std(b)); bs=np.zeros_like(b) if z<1e-8 else (b-b.mean())/z
    elif strategy=="global_scale":
        bs=(b-b.mean())/gscale
    else:
        raise ValueError(strategy)
    p=a.stable_softmax(beta*bs+q)
    pb=a.stable_softmax(b)
    pd=a.stable_softmax(q)
    return {
        "fusion_mean":float(np.sum(p*c)),"fusion_median":med(c,p),"fusion_mode":int(c[np.argmax(p)]),
        "behavior_median":med(c,pb),"delay_median":med(c,pd),
    }

def eval_split(name,idx,x,det,td,tf,eid,params,a,y,strategy,beta,gscale):
    rows=[]
    for i in idx:
        if det[i]!=1: continue
        pred=infer(x[i],td[i],params,a,y,strategy,beta,gscale)
        t=int(tf[i]); d=int(td[i])
        rows.append({"split":name,"index":int(i),"entity_id":int(eid[i]),"true_tf":t,
                     "observed_td":d,"true_delay":d-t,"delay_bucket":bucket(d-t),
                     "strategy":strategy,"beta":beta,**pred})
    df=pd.DataFrame(rows)
    e=df["fusion_median"].to_numpy()-df["true_tf"].to_numpy()
    return {"mae":float(np.mean(np.abs(e))),"signed":float(np.mean(e))},df

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print("="*80); print("EXPERIMENT 04B-CALIBRATION AUDIT"); print("="*80)
    print("MAE = Mean Absolute Error（平均绝对误差，越小越好）")
    print("Calibration = 校准；比较 per-user z-score 与 global scale")
    print("Confidence = 置信度；global scale 保留不同用户证据强弱差异")
    a=load(BASE04A,"exp04a"); y=load(BASE03Y,"exp03y")
    data=np.load(a.DATA_PATH,allow_pickle=True)
    x=data["x"].astype(float); det=data["detected"].astype(int); td=data["t_detect"].astype(int)
    tf=data["t_onset"].astype(int); labels=a.get_entity_fraud_label(data)
    eid=data["entity_id"].astype(int) if "entity_id" in data else np.arange(len(x))
    tr,va,te=a.split_indices(len(x),seed=SEED)
    params,hist=a.train_emission_em(x,labels,det,td,tr)
    gs=global_scale(x,det,td,tr,params,a,y)
    print(f"\nGlobal behavior scale from TRAIN = {gs:.6f}")
    vals=[]; frames=[]
    print("\nVALIDATION CALIBRATION SWEEP")
    for st in ["per_user_z","global_scale"]:
        for beta in BETAS:
            m,df=eval_split("validation",va,x,det,td,tf,eid,params,a,y,st,beta,gs)
            vals.append({"strategy":st,"beta":beta,**m}); frames.append(df)
            print(f"{st:12s} beta={beta:>4} | median MAE={m['mae']:.6f} | signed={m['signed']:.6f}")
    vs=pd.DataFrame(vals); vs["abs_signed"]=vs["signed"].abs()
    best=vs.sort_values(["mae","abs_signed","beta"]).iloc[0]
    st=str(best["strategy"]); beta=float(best["beta"])
    print(f"\nVALIDATION-SELECTED CALIBRATION: strategy={st}, beta={beta}")
    tm,tdf=eval_split("test",te,x,det,td,tf,eid,params,a,y,st,beta,gs)
    print("\nFIXED TEST RESULT")
    print(f"fusion median MAE={tm['mae']:.6f} | signed={tm['signed']:.6f}")
    methods=[]
    for name,col in [("delay_only","delay_median"),("behavior_only_raw","behavior_median"),("selected_fusion","fusion_median")]:
        e=tdf[col].to_numpy()-tdf["true_tf"].to_numpy()
        methods.append({"method":name,"mae":float(np.mean(np.abs(e))),"signed_error":float(np.mean(e))})
    ms=pd.DataFrame(methods); print("\nTEST OVERALL"); print(ms.to_string(index=False))
    br=[]
    for b in ["delay_1_2","delay_3_5","delay_6_plus"]:
        sub=tdf[tdf["delay_bucket"]==b]
        for name,col in [("delay_only","delay_median"),("behavior_only_raw","behavior_median"),("selected_fusion","fusion_median")]:
            e=np.abs(sub[col].to_numpy()-sub["true_tf"].to_numpy())
            br.append({"delay_bucket":b,"method":name,"n":len(sub),"mae":float(e.mean())})
    bs=pd.DataFrame(br); print("\nTEST BY TRUE DELAY"); print(bs.to_string(index=False))
    hist.to_csv(OUT/"em_history.csv",index=False); vs.to_csv(OUT/"validation_calibration_sweep.csv",index=False)
    pd.concat(frames,ignore_index=True).to_csv(OUT/"validation_predictions.csv",index=False)
    tdf.to_csv(OUT/"test_predictions.csv",index=False); ms.to_csv(OUT/"test_overall.csv",index=False); bs.to_csv(OUT/"test_by_delay.csv",index=False)
    print("\nInterpretation:")
    print("If global_scale is selected and long-delay MAE improves, per-user z-score was erasing useful confidence.")
    print("If per_user_z still wins, fixed global scaling is not enough; next test adaptive/user-specific fusion.")

if __name__=="__main__":
    main()
