"""Matched, clean-donor KV labels. These are local/proxy labels, NOT live returns."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import torch
from .schema import read_jsonl,write_jsonl,normalize_records,digest,public_record
from .qwen import QwenEngine

def teacher_utility(scores,kind,target_index=None,values=None):
    if scores.ndim!=1 or not torch.isfinite(scores).all():raise ValueError('Finite 1-D scores required')
    if kind=='candidate_value_proxy':
        if values is None or values.shape!=scores.shape:raise ValueError('Aligned continuation values required')
        return (torch.softmax(scores,0)*values).sum()
    if target_index is None or not 0<=target_index<len(scores):raise ValueError('Valid target required')
    if kind=='local_action_logprob':return scores[target_index]
    if kind=='contrastive_action':
        if len(scores)<2:raise ValueError('Contrastive credit requires at least two candidates')
        mask=torch.arange(len(scores),device=scores.device)!=target_index
        return scores[target_index]-torch.logsumexp(scores[mask],0)
    raise ValueError(kind)


def generation_signature(engine,kind,donor_limit,normalization,donor_policy='first'):
    return digest(dict(kind=kind,donor_limit=donor_limit,normalization=normalization,
        model_path=engine.model_path,revision=engine.revision,target=engine.target,
        layers=engine.layers,max_pixels=engine.max_pixels,max_tokens=engine.max_tokens,
        four_bit=engine.four_bit,donor_policy=donor_policy,protocol_version=2))

@torch.no_grad()
def label_record(engine,row,kind,donor_limit=3,normalization='mean',donor_policy='first'):
    if donor_limit<1:raise ValueError('donor_limit must be positive')
    context=engine.context(row,keep_donors=True)
    actions=row['candidates']
    if kind=='local_action_logprob':
        if 'target_index' not in row: raise ValueError('Local teacher requires training-only target action')
        target_action=actions[row['target_index']]
        baseline=engine.score(context,target_action,normalization=normalization)
        reference_probs=None
    elif kind=='contrastive_action':
        if 'target_index' not in row:raise ValueError('Contrastive teacher requires target')
        scores=torch.stack([engine.score(context,a,normalization=normalization) for a in actions])
        utility=lambda x:teacher_utility(x,kind,row['target_index'])
        baseline=utility(scores);reference_probs=torch.softmax(scores,0).cpu().tolist()
    elif kind=='candidate_value_proxy':
        if 'action_values' not in row: raise ValueError('Candidate-value teacher requires labelled continuation values')
        scores=torch.stack([engine.score(context,a,normalization=normalization) for a in actions])
        values=torch.tensor(row['action_values'],device=scores.device)
        utility=lambda x:(torch.softmax(x,0)*values).sum()
        baseline=utility(scores)
        reference_probs=torch.softmax(scores,0).cpu().tolist()
    else: raise ValueError('Live paired return labels must be collected by an environment rollout, not this script')
    baseline=float(baseline); credits=[]; records=[]
    for j,span in enumerate(context.spans):
        # No semantic ground-truth donor role is required at inference or used by the controller.
        eligible=row.get('donor_indices',list(range(len(context.spans))))
        donors=[d for d in eligible if d!=j and 0<=d<len(context.spans)
                and context.spans[d][1]-context.spans[d][0]==span[1]-span[0]]
        if donor_policy=='seeded':donors.sort(key=lambda d:digest([row['prefix_id'],j,d,'sigma-donor-v2']))
        elif donor_policy!='first':raise ValueError('Unknown donor policy')
        donors=donors[:donor_limit]
        if not donors: raise ValueError(f'{row["prefix_id"]}: no equal-length donor for block {j}')
        effects=[]
        for d in donors:
            if kind=='local_action_logprob':
                changed_utility=engine.score(context,target_action,patch=(j,d),normalization=normalization)
            else:
                changed=torch.stack([engine.score(context,a,patch=(j,d),normalization=normalization) for a in actions])
                changed_utility=utility(changed)
            effects.append(baseline-float(changed_utility))
        credits.append(sum(effects)/len(effects))
        records.append(dict(block=j,donors=donors,effects=effects,mean=credits[-1],
            donor_std=float(torch.tensor(effects).std(unbiased=True)) if len(effects)>1 else None))
    out=dict(row,teacher_source_hash=digest(row),generation_signature=generation_signature(engine,kind,donor_limit,normalization,donor_policy),credits=credits,credit_kind=kind,reference_probs=reference_probs,
        teacher_context_hash=digest(public_record(row)),teacher_protocol=dict(kind=kind,normalization=normalization,donor_policy=donor_policy,
        model_path=engine.model_path,model_revision=engine.revision,target=engine.target,layers=engine.layers,donor_capture='clean_pre_rope',records=records,
        donor_relative=True,not_live_rollout=True))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',required=True);p.add_argument('--revision');p.add_argument('--device',default='cuda')
    p.add_argument('--layers',nargs='+',type=int,default=list(range(12,24)));p.add_argument('--target',choices=['k','v','both'],default='k')
    p.add_argument('--kind',choices=['local_action_logprob','candidate_value_proxy','contrastive_action'],default='candidate_value_proxy')
    p.add_argument('--max-pixels',type=int,default=100352);p.add_argument('--max-tokens',type=int,default=8192)
    p.add_argument('--donor-policy',choices=['first','seeded'],default='first')
    p.add_argument('--donors',type=int,default=3);p.add_argument('--max-prefixes',type=int)
    p.add_argument('--normalization',choices=['mean','sum'],default='mean');p.add_argument('--four-bit',action='store_true')
    a=p.parse_args(); rows=normalize_records(read_jsonl(a.data)); rows=rows[:a.max_prefixes] if a.max_prefixes else rows
    if any(r.get('split')=='test' for r in rows): raise ValueError('Teacher generation refuses test data')
    e=QwenEngine(a.model,a.layers,a.target,a.device,a.max_pixels,a.max_tokens,a.revision,a.four_bit)
    done=read_jsonl(a.output) if Path(a.output).exists() else []; byid={r['prefix_id']:r for r in done}
    for row in rows:
        if row['prefix_id'] in byid:
            prior=byid[row['prefix_id']]
            if prior.get('teacher_context_hash')!=digest(public_record(row)) or prior.get('teacher_source_hash')!=digest(row) or prior.get('generation_signature')!=generation_signature(e,a.kind,a.donors,a.normalization,a.donor_policy):
                raise ValueError('Resume source/protocol/model mismatch; use a new output path')
            continue
        done.append(label_record(e,row,a.kind,a.donors,a.normalization,a.donor_policy));write_jsonl(a.output,done)
        print(json.dumps({'completed':len(done),'total':len(rows),'prefix_id':row['prefix_id']}),flush=True)
if __name__=='__main__':main()
