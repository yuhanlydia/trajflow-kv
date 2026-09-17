"""CPU integration using a differentiable stand-in, never a GUI result."""
import json
from types import SimpleNamespace
import torch
from tango_iclr.schema import write_jsonl
from tango_iclr.controller import CreditController
from tango_iclr.interventions import LowRankBank
from tango_iclr.runtime import gate_values
from tango_iclr.train import train

class FakeEngine:
    def __init__(self,*args,**kwargs): self.device=torch.device('cpu');self.controller=None
    def setup(self,kind,hidden,rank,alpha,gate_mode):
        self.controller=CreditController(8,hidden,kind)
        self.bank=LowRankBank({'k':8},min(rank,8),alpha);self.mode=gate_mode;self.metadata={}
    def context(self,r):
        return SimpleNamespace(features=torch.ones(2,8),query=torch.ones(8),spans=[(0,1),(1,2)])
    def score(self,c,a,controlled=False,**kwargs):
        x=torch.arange(24.).reshape(1,3,8)/24
        if controlled:
            if torch.is_grad_enabled():
                assert self.controller.training
                assert self.bank.training
            self.bank.reset_energy();g=gate_values(self.controller(c.features,c.query),self.mode)
            x=self.bank.apply('k',x,c.spans,g)
        return x.sum()*(1 if a=='A' else -1)
    def save(self,path,extra):torch.save({'metadata':extra},path)

def test_training_end_to_end_cpu(tmp_path,monkeypatch):
    import tango_iclr.train as module
    monkeypatch.setattr(module,'QwenEngine',FakeEngine)
    def row(i):return dict(prefix_id=i,task_id=i,instruction=i,image='a.png',history_images=['b.png','c.png'],
                           candidates=['A','B'],action_values=[1,0],target_index=0,credits=[.1,-.1],credit_kind='candidate_value_proxy')
    write_jsonl(tmp_path/'tr.jsonl',[row('tr')]);write_jsonl(tmp_path/'va.jsonl',[row('va')])
    result=train(dict(train=str(tmp_path/'tr.jsonl'),validation=str(tmp_path/'va.jsonl'),
                      model='fake',output=str(tmp_path/'out'),method='memory',device='cpu',hidden=16,
                      credit_epochs=1,policy_epochs=1,accumulation=4))
    assert (tmp_path/'out'/'best.pt').exists() and len(result)==1
