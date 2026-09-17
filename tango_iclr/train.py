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
from .credit_diagnostics import evaluate_credit

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
    setup_extra={}
    if cfg.get('controller')=='cross_gru':
        setup_extra=dict(feature_layers=cfg.get('feature_layers'),feature_chunks=cfg.get('feature_chunks',8),
                         gate_temperature=cfg.get('gate_temperature',1.))
    e.setup(cfg.get('controller','gru'),cfg.get('hidden',128),cfg.get('rank',8),cfg.get('alpha',8),
        'uniform' if method=='uniform' else ('unsigned' if method=='unsigned' else 'signed'),**setup_extra)
    if hasattr(e,'temperature'):e.temperature=float(cfg.get('gate_temperature',1.))
    if hasattr(e,'metadata'):e.metadata['gate_temperature']=float(cfg.get('gate_temperature',1.))
    if cfg.get('expected_model_type') and getattr(e,'model_type',None)!=cfg['expected_model_type']:
        raise ValueError('Loaded architecture differs from requested model')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'split_manifest.json').write_text(json.dumps(split,indent=2))
    # Compact pooled features only: not a persistent full-KV dataset.
    compact=[];absolute=[]
    for r in train_rows:
        c=e.context(r);compact.append((c.features.cpu(),c.query.cpu()))
        if method in MEMORY:
            if 'credits' not in r or len(r['credits'])!=len(c.spans):raise ValueError('Missing aligned measured memory credits')
            if r.get('credit_kind') not in {'local_action_logprob','candidate_value_proxy','paired_rollout_return','contrastive_action'}:raise ValueError('Credit kind missing')
            absolute.extend(abs(x) for x in r['credits'])
    scale=max(float(torch.tensor(absolute).quantile(.75)),1e-5) if absolute else 1.
    deadzone=cfg.get('credit_deadzone',0.002)
    targets=[]
    for r in train_rows:
        t=list(r.get('credits',[]))
        if method=='shuffle_memory':random.shuffle(t)
        if method=='flip_memory':t=[-x for x in t]
        targets.append(t)
    # Save treatment-specific targets separately: controls never select against undisclosed gold credit.
    target_rows=copy.deepcopy(train_rows)
    for row,t in zip(target_rows,targets):row['credits']=t
    val_targets=copy.deepcopy(val)
    vrng=random.Random(seed+1000003)
    for row in val_targets:
        if method=='shuffle_memory' and 'credits' in row:vrng.shuffle(row['credits'])
        if method=='flip_memory' and 'credits' in row:row['credits']=[-x for x in row['credits']]
    val_compact=[]
    if method in MEMORY:
        for row in val:
            c=e.context(row);val_compact.append((c.features.cpu(),c.query.cpu()))
    def closs(pred,target):
        return credit_loss(pred,target,scale=scale,deadzone=deadzone,
            rank_weight=cfg.get('credit_rank_weight',.1),sign_weight=cfg.get('credit_sign_weight',0.))
    def diagnostics(stage):
        result={}
        for label,features,rows in [('train',compact,train_rows),('validation',val_compact,val)]:
            if features:result[label]=evaluate_credit(e.controller,features,rows,e.device,scale,deadzone)
        (out/f'credit_{stage}.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    if method in MEMORY:
        opt=torch.optim.AdamW(e.controller.parameters(),lr=cfg.get('credit_lr',1e-3),weight_decay=.01)
        best_credit=float('inf');best_state=None;credit_log=[]
        for epoch in range(cfg.get('credit_epochs',20)):
            order=list(range(len(train_rows)));random.shuffle(order);losses=[];e.controller.train()
            for i in order:
                f,q=(x.to(e.device) for x in compact[i]);target=torch.tensor(targets[i],device=e.device)
                if not len(target):continue
                opt.zero_grad(set_to_none=True);loss=closs(e.controller(f,q),target)
                loss.backward();torch.nn.utils.clip_grad_norm_(e.controller.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            diagnostics('stage1_last')
            selected=evaluate_credit(e.controller,val_compact,val_targets,e.device,scale,deadzone);source='validation'
            if not selected.get('available'):
                selected=evaluate_credit(e.controller,compact,target_rows,e.device,scale,deadzone);source='train'
            score=selected['normalized_rmse']
            entry=dict(epoch=epoch,train_loss=sum(losses)/max(len(losses),1),selection_source=source,
                       selection_rmse=score,sign_accuracy=selected['signed_accuracy'])
            credit_log.append(entry)
            if score<best_credit:
                best_credit=score;best_state={k:v.detach().cpu().clone() for k,v in e.controller.state_dict().items()}
                torch.save(dict(controller=best_state,credit_scale=scale,epoch=epoch,selection_source=source),out/'credit_controller_best.pt')
            print(json.dumps(dict(stage='credit',**entry)),flush=True)
        if best_state is not None:e.controller.load_state_dict(best_state)
        first=diagnostics('stage1_best')
        e.save(out/'stage1.pt',dict(method=method,seed=seed,credit_scale=scale,training_config=cfg,stage='credit_only'))
        (out/'credit_training_log.json').write_text(json.dumps(credit_log,indent=2)+'\n')
        check=first.get('validation',{})
        if not check.get('available'):check=first['train']
        sign_acc=check.get('signed_accuracy')
        warning=check.get('correlation') is None or sign_acc is None or sign_acc<cfg.get('credit_warning_sign_accuracy',.6)
        (out/'credit_fit_status.json').write_text(json.dumps(dict(warning=warning,
            strict=cfg.get('stop_on_credit_failure',False),not_a_significance_test=True),indent=2))
        if warning and cfg.get('stop_on_credit_failure',False):raise RuntimeError('Inspect stage-1 credit diagnostics before refinement')
    freeze=bool(cfg.get('freeze_controller_policy',False) and method in MEMORY)
    e.controller.requires_grad_(not freeze)
    params=list(e.bank.parameters())+([] if freeze else list(e.controller.parameters()))
    opt=torch.optim.AdamW(params,lr=cfg.get('policy_lr',1e-4),weight_decay=.01)
    accumulation=cfg.get('accumulation',4);best=-float('inf');log=[]
    # Fixed baseline per task family from TRAIN only.
    groups={}
    for r in train_rows:groups.setdefault(r.get('family','all'),[]).append(float(r.get('return',r.get('trajectory_return',1.))))
    baselines={k:sum(v)/len(v) for k,v in groups.items()}
    for epoch in range(cfg.get('policy_epochs',3)):
        indices=list(range(len(train_rows)));random.shuffle(indices);opt.zero_grad(set_to_none=True);losses=[]
        # Policy refinement backpropagates through the controller and bank.
        # These modules are deterministic (the Transformer explicitly uses
        # dropout=0), so training mode preserves exact VJP replay while also
        # retaining the cuDNN GRU state required for backward.
        e.controller.train();e.bank.train()
        for step,i in enumerate(indices):
            r=train_rows[i];c=e.context(r); n=len(r['candidates'])
            objective=method if method in {'action_sft','ce','successful_ce','global_return','action_advantage'} else cfg.get('memory_policy_objective','ce')
            if n<2 and objective!='action_sft':raise ValueError('Candidate-policy training requires at least two proposals; use action_sft for demonstrations')
            normalization=cfg.get('score_normalization','mean')
            if objective=='action_sft' and cfg.get('lambda_kl',.05)==0:ref=None
            else:
                with torch.no_grad():ref_scores=torch.stack([e.score(c,a,normalization=normalization) for a in r['candidates']]);ref=torch.softmax(ref_scores,0)
            objective=method if method in {'action_sft','ce','successful_ce','global_return','action_advantage'} else cfg.get('memory_policy_objective','ce')
            q=torch.tensor(r['action_values'],device=e.device) if 'action_values' in r else None
            def lf(s):
                main=action_objective(s,objective,target_index=r.get('logged_action_index') if objective=='global_return' else r.get('target_index'),values=q,reference=ref,
                    trajectory_return=float(r.get('return',r.get('trajectory_return',1.))),baseline=baselines[r.get('family','all')])
                lp=torch.log_softmax(s,0);kl=(ref*(ref.clamp_min(1e-12).log()-lp)).sum() if ref is not None else s.sum()*0.
                return main+cfg.get('lambda_kl',.05)*kl
            # Divide by actual microbatch group size, including the last partial group.
            group_size=min(accumulation,len(indices)-(step//accumulation)*accumulation)
            def scorer(j):return e.score(c,r['candidates'][j],controlled=True,normalization=normalization)
            # Add energy during the gradient replay, avoiding a third full VLM pass.
            reg=(lambda:e.bank.energy()*cfg.get('lambda_energy',.01)) if cfg.get('lambda_energy',.01)>0 else None
            loss,_=candidate_vjp(scorer,n,lf,divisor=group_size,replay_regularizer=reg);losses.append(loss)
            if method in MEMORY and not freeze and len(c.spans):
                target=torch.tensor(targets[i],device=e.device)
                aux=closs(e.controller(c.features,c.query),target)
                (cfg.get('lambda_credit',.2)*aux/group_size).backward()
            if (step+1)%accumulation==0 or step+1==len(indices):
                torch.nn.utils.clip_grad_norm_(params,1.);opt.step();opt.zero_grad(set_to_none=True)
        metrics=evaluate_records(e,val,True,cfg.get('score_normalization','mean'))
        if method in MEMORY:diagnostics(f'stage2_epoch{epoch}')
        score=selection_score(metrics)
        if score is None:raise ValueError('Validation has no labelled examples')
        log.append(dict(epoch=epoch,train_loss=sum(losses)/max(len(losses),1),validation=score,validation_metrics={k:v for k,v in metrics.items() if k!='details'}))
        extra=dict(method=method,seed=seed,credit_scale=scale,credit_deadzone=deadzone,split_hash=digest(split),
                   objective_type='offline_candidate_surrogate',training_config=cfg)
        e.save(out/'last.pt',extra)
        if score>best:best=score;e.save(out/'best.pt',extra)
        (out/'training_log.json').write_text(json.dumps(log,indent=2));print(json.dumps(log[-1]),flush=True)
    if method in MEMORY:diagnostics('stage2_last')
    return log

def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    a=p.parse_args();train(yaml.safe_load(Path(a.config).read_text()))
if __name__=='__main__':main()
