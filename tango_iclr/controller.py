"""Small history-aware predictor. Inputs are detached, target-free prefix features."""
from __future__ import annotations
import math
import torch
from torch import nn


class CreditController(nn.Module):
    def __init__(self,input_dim:int,hidden:int=128,kind:str='gru'):
        super().__init__()
        if kind not in {'mlp','gru','transformer'}: raise ValueError(kind)
        self.kind=kind; self.input_dim=input_dim; self.hidden=hidden
        self.project=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,hidden),nn.GELU())
        self.query=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,hidden),nn.GELU())
        if kind=='gru': self.encoder=nn.GRU(hidden,hidden,num_layers=1,batch_first=True)
        elif kind=='transformer':
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,4,2*hidden,
                             dropout=0.,batch_first=True,norm_first=True),num_layers=1,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.Linear(3*hidden,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,blocks:torch.Tensor,query:torch.Tensor) -> torch.Tensor:
        if blocks.ndim!=2 or query.shape!=(self.input_dim,) or blocks.shape[-1]!=self.input_dim:
            raise ValueError('Expected [history,input_dim] and [input_dim]')
        if len(blocks)==0: return blocks.new_empty((0,))
        z=self.project(blocks.float()); q=self.query(query.float())
        if self.kind=='gru':
            _,s=self.encoder(z.unsqueeze(0)); context=s[-1,0].expand_as(z)
        elif self.kind=='transformer':
            pos=torch.arange(len(z),device=z.device)[:,None]
            freq=torch.exp(torch.arange(0,self.hidden,2,device=z.device)*(-math.log(10000.)/self.hidden))
            pe=torch.zeros_like(z); pe[:,0::2]=torch.sin(pos*freq); pe[:,1::2]=torch.cos(pos*freq)
            encoded=self.encoder((z+pe).unsqueeze(0))[0]; context=encoded
        else: context=torch.zeros_like(z)
        return self.head(torch.cat([z,context,q.expand_as(z)],dim=-1)).squeeze(-1)

    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
