"""First SAFEPlus verification: exact likelihood and a constant-hazard synthetic diagnostic.

Core verification/estimation uses only Python's standard library. Plotting is optional.
The model allows onset and subsequent detection in the same discrete bin.
No neural model or original SAFE implementation is trained here.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics


def forward(a, b):
    if len(a) != len(b):
        raise ValueError('Hazard sequences must have equal length')
    u, v, detection = 1.0, 0.0, []
    for ak, bk in zip(a, b):
        if not 0 <= ak <= 1 or not 0 <= bk <= 1:
            raise ValueError('Hazards must be probabilities')
        at_risk = v + u * ak
        detection.append(at_risk * bk)
        u, v = u * (1 - ak), at_risk * (1 - bk)
    return detection, u, v


def explicit_sum(a, b):
    g, survival = [], 1.0
    for ak in a:
        g.append(survival * ak)
        survival *= 1 - ak
    detection = [sum(g[k] * math.prod(1 - b[r] for r in range(k, d)) * b[d]
                     for k in range(d + 1)) for d in range(len(a))]
    latent = sum(g[k] * math.prod(1 - b[r] for r in range(k, len(a))) for k in range(len(a)))
    return detection, survival, latent


def enumerate_paths(a, b):
    detection = [0.0] * len(a)
    terminal = [0.0, 0.0]
    def visit(k, state, mass):
        if k == len(a):
            terminal[state] += mass
        elif state == 0:
            visit(k + 1, 0, mass * (1 - a[k]))
            visit(k + 1, 1, mass * a[k] * (1 - b[k]))
            detection[k] += mass * a[k] * b[k]
        else:
            visit(k + 1, 1, mass * (1 - b[k]))
            detection[k] += mass * b[k]
    visit(0, 0, 1.0)
    return detection, *terminal


def logadd(x, y):
    high, low = max(x, y), min(x, y)
    return high if high == -math.inf else high + math.log1p(math.exp(low - high))


def log_forward(a, b):
    def lp(p):
        return math.log(p) if p > 0 else -math.inf
    def lq(p):
        return math.log1p(-p) if p < 1 else -math.inf
    u, v, detection = 0.0, -math.inf, []
    for ak, bk in zip(a, b):
        at_risk = logadd(v, u + lp(ak))
        detection.append(at_risk + lp(bk))
        u, v = u + lq(ak), at_risk + lq(bk)
    return detection, u, v


def nll(counts, a, b):
    k = len(counts) - 1
    detection, u, v = log_forward([a] * k, [b] * k)
    log_probs = detection + [logadd(u, v)]
    return -sum(c * p for c, p in zip(counts, log_probs) if c) / sum(counts)


def nll_gradient(counts, a, b):
    u, v = 1.0, 0.0
    ua = ub = va = vb = 0.0
    grad_a = grad_b = 0.0
    for count in counts[:-1]:
        w = v + u * a
        wa, wb = va + ua * a + u, vb + ub * a
        d, da, db = w * b, wa * b, wb * b + w
        if count:
            grad_a -= count * da / d
            grad_b -= count * db / d
        ua, ub, va, vb = ua * (1-a) - u, ub * (1-a), wa * (1-b), wb * (1-b) - w
        u, v = u * (1-a), w * (1-b)
    grad_a -= counts[-1] * (ua + va) / (u + v)
    grad_b -= counts[-1] * (ub + vb) / (u + v)
    return grad_a / sum(counts), grad_b / sum(counts)


def responsibilities(a, b, time, detected):
    g = [a * (1-a)**k for k in range(time)]
    onset = [g[k] * (1-b)**(time-1-k) * b if detected else g[k] * (1-b)**(time-k)
             for k in range(time)]
    masses = onset if detected else [(1-a)**time] + onset
    total = sum(masses)
    return [mass / total for mass in masses]


def verify():
    rng = random.Random(20261110)
    maximum, posterior_error, cases = 0.0, 0.0, 0
    for length in range(0, 9):
        for _ in range(20):
            a, b = [rng.random() for _ in range(length)], [rng.random() for _ in range(length)]
            versions = [forward(a,b), explicit_sum(a,b), enumerate_paths(a,b)]
            for det, u, v in versions:
                maximum = max(maximum, abs(sum(det)+u+v-1))
                for det2, u2, v2 in versions:
                    maximum = max(maximum, abs(u-u2), abs(v-v2), *(abs(x-y) for x,y in zip(det,det2)))
            cases += 1
    for a in (0.0, 0.2, 1.0):
        for b in (0.0, 0.7, 1.0):
            det,u,v = forward([a]*4,[b]*4)
            maximum = max(maximum,abs(sum(det)+u+v-1))
            ref=enumerate_paths([a]*4,[b]*4)
            assert all(abs(x-y)<1e-12 for x,y in zip(det+[u,v],ref[0]+list(ref[1:])))
            cases += 1
    for time in range(1,21):
        for detected in (False,True):
            posterior_error=max(posterior_error,abs(sum(responsibilities(.08,.30,time,detected))-1))
    counts=[1,4,8,5,12,20]
    analytic=nll_gradient(counts,.17,.31)
    step=1e-6
    numeric=((nll(counts,.17+step,.31)-nll(counts,.17-step,.31))/(2*step),
             (nll(counts,.17,.31+step)-nll(counts,.17,.31-step))/(2*step))
    gradient_error=max(abs(x-y) for x,y in zip(analytic,numeric))
    stress=log_forward([.8]*1000,[.9]*1000)
    assert math.isfinite(stress[0][-1]) and math.isfinite(logadd(stress[1],stress[2]))
    assert maximum<1e-12 and posterior_error<1e-12 and gradient_error<1e-6
    return {'cases':cases,'max_probability_error':maximum,'max_posterior_sum_error':posterior_error,
            'max_gradient_error':gradient_error,'long_horizon_log_prob_finite':True,
            'gradient_check':'analytic two-parameter sensitivity vs central finite difference; not neural autodiff'}


def bounded_minimum(function, left, right, iterations=70):
    ratio=(math.sqrt(5)-1)/2
    x=right-ratio*(right-left)
    y=left+ratio*(right-left)
    fx,fy=function(x),function(y)
    for _ in range(iterations):
        if fx<fy:
            right,y,fy=y,x,fx
            x=right-ratio*(right-left)
            fx=function(x)
        else:
            left,x,fx=x,y,fy
            y=left+ratio*(right-left)
            fy=function(y)
    return (left+right)/2


def simulate(n, a, b, horizon, seed):
    rng=random.Random(seed)
    rows=[]
    for _ in range(n):
        onset=1+int(math.log1p(-rng.random())/math.log1p(-a))
        delay=int(math.log1p(-rng.random())/math.log1p(-b))
        detection=onset+delay
        rows.append((onset,detection,min(detection,horizon),int(detection<=horizon)))
    return rows


def observed_counts(rows,horizon):
    counts=[0]*(horizon+1)
    for _,_,time,detected in rows:
        counts[time-1 if detected else -1]+=1
    return counts


def risk(a,b,horizon):
    _,u,v=forward([a]*horizon,[b]*horizon)
    return v/(u+v)


def onset_mae(rows,a,b):
    errors=[]
    for onset,_,time,detected in rows:
        if detected:
            posterior=responsibilities(a,b,time,True)
            prediction=sum((j+1)*p for j,p in enumerate(posterior))
            errors.append(abs(prediction-onset))
    return statistics.mean(errors)


def experiment(out,n=5000):
    horizon,true_a,true_b=20,.08,.30
    seeds=[7,11,19,23,31]
    grid=[.02+i*.005 for i in range(97)]
    # Symmetry lets us search unordered pairs; both labeled orientations are retained.
    candidates=[(a,b) for i,a in enumerate(grid) for b in grid[i:]]
    cached={pair:log_forward([pair[0]]*horizon,[pair[1]]*horizon) for pair in candidates}
    def cached_loss(counts,pair):
        det,u,v=cached[pair]
        return -sum(c*p for c,p in zip(counts,det+[logadd(u,v)]) if c)/sum(counts)
    rows=[]
    first=None
    for seed in seeds:
        train=simulate(n,true_a,true_b,horizon,seed)
        test=simulate(n,true_a,true_b,horizon,seed+10000)
        train_counts,test_counts=observed_counts(train,horizon),observed_counts(test,horizon)
        pair=min(candidates,key=lambda p:cached_loss(train_counts,p))
        coarse_anchor=min(grid,key=lambda a:nll(train_counts,a,true_b))
        anchored=bounded_minimum(lambda a:nll(train_counts,a,true_b),
                                 max(.001,coarse_anchor-.005),min(.999,coarse_anchor+.005))
        undetected=[r for r in test if not r[3]]
        actual=statistics.mean(r[0]<=horizon for r in undetected)
        model_pairs={'unanchored_low_a':pair,'unanchored_swapped':pair[::-1],'known_delay':(anchored,true_b)}
        for name,(a,b) in model_pairs.items():
            rows.append({'seed':seed,'model':name,'a':a,'b':b,'test_nll_per_account':nll(test_counts,a,b),
                         'mean_onset_bins':1/a,'detected_onset_mae_bins':onset_mae(test,a,b),
                         'undetected_fraud_probability':risk(a,b,horizon),
                         'empirical_undetected_fraud_fraction':actual,'test_censoring_fraction':len(undetected)/n})
        if first is None:
            first={'train_counts':train_counts,'test_counts':test_counts,'grid':grid,
                   'losses':{pair:cached_loss(train_counts,pair) for pair in candidates}}
    aggregates={}
    for model in model_pairs:
        subset=[r for r in rows if r['model']==model]
        aggregates[model]={}
        for metric in ('a','b','test_nll_per_account','mean_onset_bins','detected_onset_mae_bins',
                       'undetected_fraud_probability','empirical_undetected_fraud_fraction'):
            values=[r[metric] for r in subset]
            aggregates[model][metric]={'mean':statistics.mean(values),'sample_sd':statistics.stdev(values)}
    true_det,u,v=forward([true_a]*horizon,[true_b]*horizon)
    swap_det,us,vs=forward([true_b]*horizon,[true_a]*horizon)
    swap_error=max(abs(x-y) for x,y in zip(true_det+[u+v],swap_det+[us+vs]))
    assert swap_error<1e-12
    summary={'verification':verify(),'design':{'train_n_per_seed':n,'test_n_per_seed':n,'seeds':seeds,
             'horizon_bins':horizon,'true_onset_hazard':true_a,'true_detection_hazard':true_b,
             'grid_step':.005,'known_delay_fitting':'grid initialization followed by bounded scalar refinement','covariates':False,'same_bin_detection':True,
             'fitting_uses':'detection indicator and observed time only; no latent onset labels',
             'known_delay_anchor':'true b supplied as external oracle information; not estimated from hidden labels',
             'unanchored_branches':'same fitted unordered pair; low-a is arbitrary display order, not identification'},
             'truth':{'mean_onset_bins':1/true_a,'undetected_fraud_probability':v/(u+v),
                      'swapped_undetected_fraud_probability':vs/(us+vs),'censoring_probability':u+v,
                      'max_swap_detection_probability_difference':swap_error},'aggregates':aggregates,
             'limitations':['Synthetic constant hazards only; no original SAFE or GRU model was trained.',
                            'One parameter setting, five seeds, and a bounded grid are a diagnostic, not a benchmark.',
                            'Constant positive onset hazard implies eventual fraud with probability one; no cure fraction is tested.',
                            'Mean posterior onset is evaluated only for detected test accounts.',
                            'Known delay is oracle information; recovery here does not prove real-world identifiability.']}
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'replicates.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return summary,rows,first


def plot(out,summary,rows,first):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    horizon=summary['design']['horizon_bins']
    a,b=summary['design']['true_onset_hazard'],summary['design']['true_detection_hazard']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    grid=first['grid'];minimum=min(first['losses'].values())
    surface=[[first['losses'][tuple(sorted((x,y)))]-minimum for x in grid] for y in grid]
    im=axes[0].contourf(grid,grid,surface,levels=[0,.001,.005,.01,.03,.1,.3,.8,2],cmap='cividis',extend='max')
    axes[0].scatter([a,b],[b,a],marker='x',s=65,c='#d95f02',linewidths=2,label='True pair and rate-swapped pair')
    axes[0].set(xlabel='Onset probability a (per bin)',ylabel='Detection probability b (per bin)',title='Detection-only likelihood is symmetric')
    axes[0].legend(fontsize=7,loc='upper right')
    fig.colorbar(im,ax=axes[0],label='Training NLL above minimum (nats/account)')
    x=list(range(1,horizon+1))
    d,u,v=forward([a]*horizon,[b]*horizon)
    ds,_,_=forward([b]*horizon,[a]*horizon)
    axes[1].plot(x,[sum(d[:k]) for k in x],label='True a=0.08, b=0.30',linewidth=2.8)
    axes[1].plot(x,[sum(ds[:k]) for k in x],'--',label='Swapped a=0.30, b=0.08',linewidth=1.8)
    empirical=[sum(first['test_counts'][:k])/sum(first['test_counts']) for k in x]
    axes[1].plot(x,empirical,'.',color='black',label='Synthetic held-out observations')
    axes[1].set(xlabel='Time (bins)',ylabel='P(detected by time)',title='Same detection CDF, different latent onset',ylim=(0,1))
    axes[1].legend(fontsize=7)
    names=['unanchored_low_a','unanchored_swapped','known_delay']
    means=[summary['aggregates'][name]['undetected_fraud_probability']['mean'] for name in names]
    sd=[summary['aggregates'][name]['undetected_fraud_probability']['sample_sd'] for name in names]
    axes[2].bar(range(3),means,yerr=sd,color=['#497b99','#969696','#477c5a'],capsize=4)
    axes[2].axhline(summary['truth']['undetected_fraud_probability'],color='black',linestyle='--',label='True probability')
    axes[2].set(xticks=range(3),xticklabels=['Unanchored:\nlow-a branch','Unanchored:\nswapped branch','Known delay'],
                ylabel='P(onset by bin 20 | undetected)',title='Undetected-fraud risk is not identified',ylim=(0,1.07))
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('SAFEPlus first synthetic diagnostic — no covariates or neural network',fontsize=13)
    fig.supxlabel('Synthetic data: train/test N=5,000 each per seed; 5 seeds; 20 bins. Bars: mean ± sample SD. Likelihood/CDF panels: seed 7.',fontsize=8)
    fig.savefig(out/'diagnostic.png',dpi=180)
    plt.close(fig)


def main():
    import sys

    # Support direct script execution as well as Python's module entry point.
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.run_paths import new_run_directory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='Override the default run directory')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    output_dir = args.out if args.out is not None else new_run_directory('synthetic_likelihood')
    summary, rows, first = experiment(output_dir)
    if args.plot:
        plot(output_dir, summary, rows, first)
    print(json.dumps(summary, indent=2))
    print(f'Results: {output_dir}', file=sys.stderr)


if __name__=='__main__':
    main()
