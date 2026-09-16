"""Decoder-only, pre-RoPE interventions with clean donors and bounded spans.

A signed residual does not mathematically imply monotonic attention enhancement.
The learned operator must pass held-out policy evaluation, separately from credit prediction.
"""
from __future__ import annotations
import re
from contextlib import contextmanager
import torch
from torch import nn


def decoder_projection_names(model:nn.Module,layers:list[int],target:str) -> list[str]:
    if target not in {'k','v','both'}: raise ValueError(target)
    kinds={'k','v'} if target=='both' else {target}
    names=[]
    for name,module in model.named_modules():
        m=re.search(r'(?:^|\.)layers\.(\d+)\.self_attn\.([kv])_proj$',name)
        if m and 'visual' not in name and int(m[1]) in layers and m[2] in kinds and hasattr(module,'out_features'):
            names.append(name)
    if not names: raise ValueError('No selected decoder k/v projections; check architecture and layer indices')
    found={int(re.search(r'layers\.(\d+)',n)[1]) for n in names}
    if found!=set(layers): raise ValueError(f'Missing requested layers: {set(layers)-found}')
    return names


def visual_spans(input_ids:torch.Tensor,image_token_id:int) -> list[tuple[int,int]]:
    if input_ids.ndim!=2 or input_ids.shape[0]!=1: raise ValueError('Intervention requires batch size one')
    ids=input_ids[0].tolist(); spans=[]; start=None
    for i,x in enumerate(ids+[None]):
        if x==image_token_id and start is None: start=i
        elif x!=image_token_id and start is not None: spans.append((start,i)); start=None
    return spans


def patch_span(x:torch.Tensor,span:tuple[int,int],donor:torch.Tensor) -> torch.Tensor:
    s,e=span
    if x.ndim!=3 or not 0<=s<e<=x.shape[1] or donor.shape!=x[:,s:e].shape:
        raise ValueError('Donor and recipient must have identical token length and width')
    y=x.clone(); y[:,s:e]=donor.to(x); return y


class Residual(nn.Module):
    def __init__(self,width,rank,alpha):
        super().__init__()
        if not 0<rank<=width: raise ValueError('Invalid rank')
        self.down=nn.Linear(width,rank,bias=False); self.up=nn.Linear(rank,width,bias=False)
        nn.init.orthogonal_(self.down.weight); nn.init.zeros_(self.up.weight)
        self.scale=alpha/rank
    def forward(self,x): return self.up(self.down(x.float()))*self.scale


class LowRankBank(nn.Module):
    def __init__(self,widths:dict[str,int],rank:int=8,alpha:float=8):
        super().__init__(); self.names=list(widths); self.mapping={n:f'p{i}' for i,n in enumerate(widths)}
        self.blocks=nn.ModuleDict({self.mapping[n]:Residual(w,rank,alpha) for n,w in widths.items()})
        self.energies=[]
    def reset_energy(self): self.energies=[]
    def apply(self,name,x,spans,gates):
        if len(gates)!=len(spans): raise ValueError('One gate per history span required')
        y=x.clone(); module=self.blocks[self.mapping[name]]
        for j,(s,e) in enumerate(spans):
            if e>x.shape[1]: continue  # decode step reads already transformed prefix cache
            delta=module(x[:,s:e])*gates[j]
            y[:,s:e]=x[:,s:e]+delta.to(x.dtype)
            den=x[:,s:e].detach().float().square().mean().clamp_min(1e-6)
            self.energies.append(delta.square().mean()/den)
        return y
    def energy(self):
        return torch.stack(self.energies).mean() if self.energies else next(self.parameters()).sum()*0.
    def merge_static(self,modules:dict,dynamic:bool=True):
        if dynamic: raise ValueError('Input-dependent signed gates cannot be algebraically merged')
        with torch.no_grad():
            for name in self.names:
                p=self.blocks[self.mapping[name]]; layer=modules[name]
                d=p.up.weight@p.down.weight*p.scale
                layer.weight.add_((d@layer.weight.float()).to(layer.weight))
                if layer.bias is not None: layer.bias.add_((d@layer.bias.float()).to(layer.bias))


@contextmanager
def hooks(model,callbacks:dict):
    handles=[]; modules=dict(model.named_modules())
    try:
        for name,fn in callbacks.items(): handles.append(modules[name].register_forward_hook(fn))
        yield
    finally:
        for h in handles: h.remove()
