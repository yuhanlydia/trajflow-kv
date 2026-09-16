"""Two-stage supervised credit fitting and matched policy refinement.

All model parameters are frozen except the small controller and low-rank bank.
This is offline supervised/candidate-policy training, not unbiased REINFORCE.
"""
from __future__ import annotations
import argparse,copy,json,random
from pathlib import Path
import torch
from .qwen import QwenEngine
from .schema import read_jsonl,normalize_records,validate_splits,digest
from .objectives import action_objective,credit_loss
from .runtime import candidate_vjp
from .evaluate import evaluate_records

METHODS=['action_sft','ce','successful_ce','global_return','action_advantage','memory','shuffle_memory','flip_memory','unsigned','uniform']
MEMORY={'memory','shuffle_memory','flip_memory','unsigned'}

def selection_score(metrics):
    """Rank checkpoints without converting single demonstrations into fake accuracy."""
    for key in ('critical_accuracy','accuracy'):
        if metrics.get(key) is not None:return float(metrics[key])
    if metrics.get('mean_action_nll') is not None:return -float(metrics['mean_action_nll'])
    raise ValueError('Validation has neither discriminative candidates nor action targets')

def train(config):
    cfg=copy.deepcopy(config);seed=cfg.get('seed',0);random.seed(seed);torch.manual_seed(seed)
    train_rows=normalize_records(read_jsonl(cfg['train']));val=normalize_records(read_jsonl(cfg['validation']))
    split=validate_splits({'train':train_rows,'validation':val},cfg.get('group_key','task_id'))
    if any(r.get('split')=='test' for r in train_rows+val):raise ValueError('Test records cannot be used for training/selection')
    if not train_rows or not val:raise ValueError('Nonempty train and validation splits required')
    method=cfg['method']
    if method not in METHODS:raise ValueError(method)
    if method in {'global_return','successful_ce'}:
        for r in train_rows:
            if 'trajectory_return' not in r and 'return' not in r:raise ValueError(f'{method} requires actual logged trajectory return')
            if method=='global_return' and 'logged_action_index' not in r:raise ValueError('Global-return baseline requires logged_action_index, not oracle target_index')
    e=QwenEngine(cfg['model'],cfg.get('layers'),cfg.get('target','k'),cfg.get('device','cuda'),
        cfg.get('max_pixels',100352),cfg.get('max_tokens',8192),cfg.get('revision'),cfg.get('four_bit',False))
    e.setup(cfg.get('controller','gru'),cfg.get('hidden',128),cfg.get('rank',8),cfg.get('alpha',8),
        'uniform' if method=='uniform' else ('unsigned' if method=='unsigned' else 'signed'))
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'split_manifest.json').write_text(json.dumps(split,indent=2))
    # Compact pooled features only: not a persistent full-KV dataset.
    compact=[];absolute=[]
    for r in train_rows:
        c=e.context(r);compact.append((c.features.cpu(),c.query.cpu()))
        if method in MEMORY:
            if 'credits' not in r or len(r['credits'])!=len(c.spans):raise ValueError('Missing aligned measured memory credits')
            if r.get('credit_kind') not in {'local_action_logprob','candidate_value_proxy','paired_rollout_return'}:raise ValueError('Credit kind missing')
            absolute.extend(abs(x) for x in r['credits'])
    scale=max(float(torch.tensor(absolute).quantile(.75)),1e-5) if absolute else 1.
    deadzone=cfg.get('credit_deadzone',0.002)
    targets=[]
    for r in train_rows:
        t=list(r.get('credits',[]))
        if method=='shuffle_memory':random.shuffle(t)
        if method=='flip_memory':t=[-x for x in t]
        targets.append(t)
    # Avoid the signed-gate/zero-up double-zero initialization trap by pretraining the gate.
    if method in MEMORY:
        opt=torch.optim.AdamW(e.controller.parameters(),lr=cfg.get('credit_lr',1e-3),weight_decay=.01)
        for epoch in range(cfg.get('credit_epochs',20)):
            order=list(range(len(train_rows)));random.shuffle(order);losses=[]
            e.controller.train()
            for i in order:
                f,q=(x.to(e.device) for x in compact[i]);target=torch.tensor(targets[i],device=e.device)
                if not len(target):continue
                opt.zero_grad(set_to_none=True)
                loss=credit_loss(e.controller(f,q),target,scale=scale,deadzone=deadzone)
                loss.backward();torch.nn.utils.clip_grad_norm_(e.controller.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            print(json.dumps({'stage':'credit','epoch':epoch,'loss':sum(losses)/max(len(losses),1)}),flush=True)
    params=list(e.bank.parameters())+list(e.controller.parameters())
    opt=torch.optim.AdamW(params,lr=cfg.get('policy_lr',1e-4),weight_decay=.01)
    accumulation=cfg.get('accumulation',4);best=-float('inf');log=[]
    # Fixed baseline per task family from TRAIN only.
    groups={}
    for r in train_rows:groups.setdefault(r.get('family','all'),[]).append(float(r.get('return',r.get('trajectory_return',1.))))
    baselines={k:sum(v)/len(v) for k,v in groups.items()}
    for epoch in range(cfg.get('policy_epochs',3)):
        indices=list(range(len(train_rows)));random.shuffle(indices);opt.zero_grad(set_to_none=True);losses=[]
        # eval keeps controller dropout deterministic for the exact two-pass VJP; gradients still enabled.
        e.controller.eval();e.bank.eval()
        for step,i in enumerate(indices):
            r=train_rows[i];c=e.context(r); n=len(r['candidates'])
            objective=method if method in {'action_sft','ce','successful_ce','global_return','action_advantage'} else cfg.get('memory_policy_objective','ce')
            if n<2 and objective!='action_sft':raise ValueError('Candidate-policy training requires at least two proposals; use action_sft for demonstrations')
            with torch.no_grad():ref_scores=torch.stack([e.score(c,a) for a in r['candidates']]);ref=torch.softmax(ref_scores,0)
            objective=method if method in {'action_sft','ce','successful_ce','global_return','action_advantage'} else cfg.get('memory_policy_objective','ce')
            q=torch.tensor(r['action_values'],device=e.device) if 'action_values' in r else None
            def lf(s):
                main=action_objective(s,objective,target_index=r.get('logged_action_index') if objective=='global_return' else r.get('target_index'),values=q,reference=ref,
                    trajectory_return=float(r.get('return',r.get('trajectory_return',1.))),baseline=baselines[r.get('family','all')])
                lp=torch.log_softmax(s,0);kl=(ref*(ref.clamp_min(1e-12).log()-lp)).sum()
                return main+cfg.get('lambda_kl',.05)*kl
            # Divide by actual microbatch group size, including the last partial group.
            group_size=min(accumulation,len(indices)-(step//accumulation)*accumulation)
            def scorer(j):return e.score(c,r['candidates'][j],controlled=True)
            # Add energy during the gradient replay, avoiding a third full VLM pass.
            reg=(lambda:e.bank.energy()*cfg.get('lambda_energy',.01)) if cfg.get('lambda_energy',.01)>0 else None
            loss,_=candidate_vjp(scorer,n,lf,divisor=group_size,replay_regularizer=reg);losses.append(loss)
            if method in MEMORY and len(c.spans):
                target=torch.tensor(targets[i],device=e.device)
                aux=credit_loss(e.controller(c.features,c.query),target,scale=scale,deadzone=deadzone)
                (cfg.get('lambda_credit',.2)*aux/group_size).backward()
            if (step+1)%accumulation==0 or step+1==len(indices):
                torch.nn.utils.clip_grad_norm_(params,1.);opt.step();opt.zero_grad(set_to_none=True)
        metrics=evaluate_records(e,val,True)
        score=selection_score(metrics)
        if score is None:raise ValueError('Validation has no labelled examples')
        log.append(dict(epoch=epoch,train_loss=sum(losses)/max(len(losses),1),validation=score))
        extra=dict(method=method,seed=seed,credit_scale=scale,credit_deadzone=deadzone,split_hash=digest(split),
                   objective_type='offline_candidate_surrogate',training_config=cfg)
        e.save(out/'last.pt',extra)
        if score>best:best=score;e.save(out/'best.pt',extra)
        (out/'training_log.json').write_text(json.dumps(log,indent=2));print(json.dumps(log[-1]),flush=True)
    return log

def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    a=p.parse_args();train(yaml.safe_load(Path(a.config).read_text()))
if __name__=='__main__':main()
