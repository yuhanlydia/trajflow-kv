"""Paired episode analysis; task clusters, not tokens/blocks, determine uncertainty."""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np


def paired_summary(a:list[dict],b:list[dict],*,n_boot:int=10000,seed:int=2027) -> dict:
    def index(rows):
        out={}
        for r in rows:
            key=str(r['case_id'])
            if key in out: raise ValueError('Duplicate paired case')
            if r.get('status','complete')!='complete': raise ValueError('Infrastructure failures require explicit resolution')
            if r.get('success') not in (True,False,0,1): raise ValueError('Binary success required')
            out[key]=r
        return out
    aa,bb=index(a),index(b)
    if not aa or set(aa)!=set(bb): raise ValueError('Paired manifests differ or are empty; no silent intersection')
    groups=defaultdict(list); wins=losses=0
    for k in sorted(aa):
        if str(aa[k]['task_id'])!=str(bb[k]['task_id']): raise ValueError('Task mismatch')
        d=int(aa[k]['success'])-int(bb[k]['success']); groups[str(aa[k]['task_id'])].append(d)
        wins+=d==1; losses+=d==-1
    n=wins+losses
    p=min(1.,2.*sum(math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2)) for k in range(min(wins,losses)+1))) if n else 1.
    means=np.array([np.mean(v) for v in groups.values()])
    rng=np.random.default_rng(seed)
    boot=np.array([rng.choice(means,len(means),replace=True).mean() for _ in range(n_boot)])
    return {'cases':len(aa),'task_clusters':len(groups),'wins':int(wins),'losses':int(losses),
            'ties':len(aa)-n,'delta_pp':100.*float(means.mean()),
            'ci95_pp':(100*np.quantile(boot,[.025,.975])).tolist(),'mcnemar_p':p,'mcnemar_note':'Descriptive paired-case test; clustered task bootstrap is the primary inference when seeds share tasks.',
            'success_a':sum(int(r['success']) for r in a),'success_b':sum(int(r['success']) for r in b),
            'estimand':'macro-task paired success difference',
            'warning':'Fewer than 10 independent task clusters: interval is exploratory.' if len(groups)<10 else None}


def credit_summary(records:list[dict]) -> dict:
    values=np.array([r['effect'] for r in records],dtype=float)
    if not len(values) or not np.isfinite(values).all(): raise ValueError('No finite effects')
    return {'independent_prefixes':len({r['prefix_id'] for r in records}),
            'block_measurements':len(records),'mean_effect':float(values.mean()),
            'positive_fraction':float((values>0).mean()),
            'note':'Sign frequency is not causal-ground-truth accuracy; report per-block/role only as analysis.'}
