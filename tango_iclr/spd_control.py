"""Adapted SPD-style GUI control: calibration CE gradients -> fixed right subspace.

This is NOT a claimed reproduction of SPD's full published data-generation/SFT
pipeline. It isolates a fixed correctness-gradient subspace on the same GUI data.
"""
import argparse,json
from pathlib import Path
import torch
from .schema import read_jsonl,normalize_records
from .qwen import QwenEngine
from .interventions import hooks

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--rank',type=int,default=8);p.add_argument('--max-prefixes',type=int,default=100)
    p.add_argument('--layers',nargs='+',type=int,default=list(range(12,24)));p.add_argument('--target',choices=['k','v','both'],default='k')
    p.add_argument('--device',default='cuda');a=p.parse_args()
    rows=normalize_records(read_jsonl(a.data))[:a.max_prefixes]
    if any(r.get('split')=='test' for r in rows):raise ValueError('Calibration cannot use test')
    e=QwenEngine(a.model,a.layers,a.target,a.device)
    cov={n:torch.zeros(w,w,device=e.device) for n,w in e.widths.items()}
    for r in rows:
        c=e.context(r); activations={}
        def capture(n):
            def fn(_m,_i,x):
                x.requires_grad_(True);x.retain_grad();activations[n]=x;return x
            return fn
        with hooks(e.model,{n:capture(n) for n in e.names}):
            loss=-e.score(c,r['candidates'][r['target_index']],normalization='sum');loss.backward()
        for n,x in activations.items():
            if x.grad is None:raise RuntimeError(f'No activation gradient: {n}')
            g=torch.cat([x.grad[:,s:end].reshape(-1,x.shape[-1]) for s,end in c.spans],0).float() if c.spans else x.grad[:,:c.prompt_ids.shape[1]].reshape(-1,x.shape[-1]).float()
            cov[n]+=g.T@g/max(len(g),1)
    proj={}
    for n,m in cov.items():
        _,u=torch.linalg.eigh(m);u=u[:,-min(a.rank,u.shape[1]):];proj[n]=(u@u.T).cpu()
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'projectors':proj,'metadata':dict(label='SPD-style GUI gradient-subspace control',layers=a.layers,target=a.target,
        rank=a.rank,calibration_prefixes=len(rows),operator='history_x_times_P_not_residual',not_full_spd_reproduction=True)},path)
if __name__=='__main__':main()
