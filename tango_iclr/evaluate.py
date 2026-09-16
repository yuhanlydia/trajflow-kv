"""Offline candidate diagnostic. Never report these metrics as live task success."""
import argparse,json
from pathlib import Path
import torch
from .qwen import QwenEngine
from .schema import read_jsonl,write_jsonl,normalize_records

@torch.no_grad()
def evaluate_records(engine,rows,controlled=True,normalization='mean',fixed_projectors=None):
    details=[]
    if engine.controller is not None:engine.controller.eval();engine.bank.eval()
    for r in rows:
        c=engine.context(r)
        s=torch.stack([engine.score(c,a,controlled,normalization=normalization,fixed_projectors=fixed_projectors) for a in r['candidates']])
        selected=int(s.argmax()); true=r.get('target_index')
        good=r.get('optimal_indices',[true] if true is not None else [])
        prob=torch.softmax(s,0)
        item=dict(prefix_id=r['prefix_id'],task_id=r['task_id'],selected=selected,
                  correct=selected in good if good and len(r['candidates'])>1 else None,critical=bool(r.get('critical_step',r.get('critical',True))),
                  scores=s.cpu().tolist(),probabilities=prob.cpu().tolist())
        if true is not None:item['action_nll']=-float(s[int(true)])
        if controlled:item['normalized_energy']=float(engine.bank.energy().detach())
        if r.get('action_values') is not None:item['selected_label_value']=r['action_values'][selected]
        if engine.controller is not None and len(c.spans):
            item['predicted_credit']=engine.controller(c.features,c.query).cpu().tolist()
        details.append(item)
    labelled=[x for x in details if x['correct'] is not None]; critical=[x for x in labelled if x['critical']]
    nll=[x['action_nll'] for x in details if 'action_nll' in x]
    return dict(metric_type='offline_candidate_accuracy_not_task_success',prefixes=len(details),
                mean_action_nll=sum(nll)/len(nll) if nll else None,
                accuracy=sum(x['correct'] for x in labelled)/len(labelled) if labelled else None,
                critical_accuracy=sum(x['correct'] for x in critical)/len(critical) if critical else None,
                critical_prefixes=len(critical),details=details)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint');p.add_argument('--model');p.add_argument('--device',default='cuda')
    p.add_argument('--normalization',choices=['mean','sum'],default='mean');p.add_argument('--fixed-projectors');p.add_argument('--adapter')
    p.add_argument('--max-pixels',type=int,default=100352);p.add_argument('--max-tokens',type=int,default=8192)
    p.add_argument('--revision');p.add_argument('--four-bit',action='store_true')
    a=p.parse_args()
    e=QwenEngine.from_checkpoint(a.checkpoint,a.device,a.model) if a.checkpoint else QwenEngine(a.model,device=a.device,max_pixels=a.max_pixels,max_tokens=a.max_tokens,revision=a.revision,four_bit=a.four_bit)
    if a.adapter:
        if a.checkpoint:raise ValueError('Student adapter evaluation must not attach teacher checkpoint')
        e.model.load_adapter(a.adapter)
    proj=torch.load(a.fixed_projectors,map_location='cpu',weights_only=True)['projectors'] if a.fixed_projectors else None
    result=evaluate_records(e,normalize_records(read_jsonl(a.data)),bool(a.checkpoint),a.normalization,proj)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='details'}))
if __name__=='__main__':main()
