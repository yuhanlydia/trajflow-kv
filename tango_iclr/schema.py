"""Prefix data contracts. Gold/teacher information never enters public_record."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any

PUBLIC_KEYS = ('instruction', 'image', 'history_images', 'history_actions', 'prompt', 'messages')
CREDIT_KINDS = {'local_action_logprob', 'candidate_value_proxy', 'paired_rollout_return','contrastive_action'}


def read_jsonl(path: str | Path) -> list[dict]:
    result=[]
    with Path(path).open(encoding='utf8') as f:
        for n,line in enumerate(f,1):
            if line.strip():
                x=json.loads(line)
                if not isinstance(x,dict): raise ValueError(f'{path}:{n}: expected object')
                result.append(x)
    return result


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix(p.suffix+'.tmp')
    with temporary.open('w',encoding='utf8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(p)


def public_record(record: dict) -> dict:
    return {k:record[k] for k in PUBLIC_KEYS if k in record}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def _flatten(row: dict) -> dict:
    x=dict(row.get('prefix',{})); x.update({k:v for k,v in row.items() if k!='prefix'})
    x.setdefault('instruction',x.get('task',''))
    x.setdefault('history_images',[]); x.setdefault('history_actions',[])
    x.setdefault('family',x.get('task_family','unspecified'))
    if not x.get('task_id'):
        if x.get('trajectory_id'):
            x['task_id']=str(x['trajectory_id']); x['_group_source']='trajectory_id'
        elif x['family']!='unspecified' and x.get('seed') is not None:
            x['task_id']=f"{x['family']}:seed:{x['seed']}"; x['_group_source']='family_seed'
        else:
            x['task_id']=str(x.get('prefix_id','')); x['_group_source']='prefix_only'
    x.setdefault('split','unspecified')
    if 'memory_advantages' in x and 'credits' not in x:
        x['credits']=x['memory_advantages']
        # Legacy estimator computes softmax(candidate score) dot enumerated Q.
        if x.get('memory_advantage_source')=='matched_latent_kv_patch_q':
            x.setdefault('credit_kind','candidate_value_proxy')
    return x


def normalize_records(rows: list[dict]) -> list[dict]:
    groups:dict[str,list[dict]]={}
    for row in rows:
        x=_flatten(row)
        if not x.get('prefix_id'): raise ValueError('Every row requires prefix_id')
        groups.setdefault(str(x['prefix_id']),[]).append(x)
    output=[]
    for key,group in groups.items():
        x=dict(group[0]); expected=digest(public_record(x))
        if any(digest(public_record(r))!=expected for r in group):
            raise ValueError(f'Candidate-dependent context at prefix {key}')
        if 'candidates' not in x:
            x['candidates']=[r['action'] for r in group]
            if all('Q' in r for r in group): x['action_values']=[float(r['Q']) for r in group]
            opt=x.get('optimal_actions',[])
            if opt: x.setdefault('target_index',next((i for i,a in enumerate(x['candidates']) if a in opt),-1))
        elif len(group)!=1:
            raise ValueError(f'Duplicate grouped prefix: {key}')
        x['critical_step']=any(bool(r.get('critical_step',r.get('critical',False)))
                               or bool(r.get('is_critical_action',False)) for r in group)
        opt=set(a for r in group for a in r.get('optimal_actions',[]))
        if opt:
            x['optimal_indices']=[i for i,a in enumerate(x['candidates']) if a in opt]
            if not x['optimal_indices']: raise ValueError(f'{key}: oracle actions absent')
            x['target_index']=x['optimal_indices'][0]
        candidates=x.get('candidates',[])
        if not candidates or not all(isinstance(a,str) for a in candidates):
            raise ValueError(f'{key}: candidates must be nonempty strings')
        if len(set(candidates))!=len(candidates): raise ValueError(f'{key}: duplicate candidates')
        if 'action_values' in x:
            values=x['action_values']
            if len(values)!=len(candidates) or not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f'{key}: invalid candidate values')
            x['action_values']=[float(v) for v in values]
            x.setdefault('target_index',max(range(len(values)),key=lambda i:values[i]))
            x.setdefault('optimal_indices',[i for i,v in enumerate(values) if float(v)==max(map(float,values))])
        if 'target_index' in x and not 0<=int(x['target_index'])<len(candidates):
            raise ValueError(f'{key}: target is absent from candidate set')
        if 'target_index' in x:x['target_index']=int(x['target_index'])
        if 'logged_action' in x:
            if x['logged_action'] not in candidates:raise ValueError(f'{key}: logged action missing from candidate support')
            x['logged_action_index']=candidates.index(x['logged_action'])
        if 'logged_action_index' in x and not 0<=int(x['logged_action_index'])<len(candidates):raise ValueError('Invalid logged action index')
        if 'credits' in x:
            if len(x['credits'])!=len(x['history_images']): raise ValueError(f'{key}: credit/history mismatch')
            if not all(math.isfinite(float(v)) for v in x['credits']): raise ValueError('Nonfinite credit')
            if x.get('credit_kind') not in CREDIT_KINDS:
                raise ValueError(f'{key}: explicitly set credit_kind; do not conflate scores and returns')
        output.append(x)
    return output


def validate_splits(splits: dict[str,list[dict]], group_key: str='task_id') -> dict:
    owners={}; fingerprints={}; report={}
    for split,rows in splits.items():
        for row in rows:
            if group_key=='task_id' and row.get('_group_source')=='prefix_only':
                raise ValueError('Explicit task grouping is required; a prefix_id is not an episode/task split')
            group=str(row[group_key])
            if group in owners and owners[group]!=split: raise ValueError(f'{group_key} leakage: {group}')
            owners[group]=split
            fp=digest(public_record(row))
            if fp in fingerprints and fingerprints[fp]!=split: raise ValueError('Identical public input crosses splits')
            fingerprints[fp]=split
        report[split]={'prefixes':len(rows),'groups':len({str(x[group_key]) for x in rows}),
                       'manifest_sha256':digest(rows)}
    return report


def resolve_images(rows: list[dict], root: str | Path) -> list[dict]:
    root=Path(root)
    for x in rows:
        for k in ('image',):
            if x.get(k) and not Path(x[k]).is_absolute(): x[k]=str((root/x[k]).resolve())
        x['history_images']=[str((root/p).resolve()) if not Path(p).is_absolute() else p for p in x['history_images']]
    return rows
