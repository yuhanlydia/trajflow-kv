"""Paired synthetic candidate results, never native task success."""
import json
from pathlib import Path


def compare(base,method):
    b={r['prefix_id']:r for r in base['details']};m={r['prefix_id']:r for r in method['details']}
    if b.keys()!=m.keys():raise ValueError('Different prefix sets')
    keys=[k for k in b if b[k]['correct'] is not None and m[k]['correct'] is not None]
    if not keys:raise ValueError('No discriminative pairs')
    rescue=sum(not b[k]['correct'] and m[k]['correct'] for k in keys)
    regression=sum(b[k]['correct'] and not m[k]['correct'] for k in keys)
    return dict(pairs=len(keys),base_correct=sum(b[k]['correct'] for k in keys),
                method_correct=sum(m[k]['correct'] for k in keys),rescues=rescue,regressions=regression,
                delta_pp=100*(rescue-regression)/len(keys),changed_decisions=sum(b[k]['selected']!=m[k]['selected'] for k in keys))


def summarize(root):
    root=Path(root);p=root/'base'/'sealed_test.json'
    if not p.exists():raise FileNotFoundError('Missing completed sealed base test')
    base=json.loads(p.read_text());results={};loaded={}
    for f in sorted(root.glob('*/sealed_test.json')):
        if f==p:continue
        loaded[f.parent.name]=json.loads(f.read_text());results[f.parent.name]=compare(base,loaded[f.parent.name])
    vs_sft={}
    for name,value in loaded.items():
        if name.startswith('memory_s'):
            sft='action_sft_s'+name.rsplit('_s',1)[-1]
            if sft in loaded:vs_sft[name]=compare(loaded[sft],value)
    return dict(metric='synthetic_candidate_accuracy_NOT_native_GUI_success',vs_base=results,vs_action_sft=vs_sft)
