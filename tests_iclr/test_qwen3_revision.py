"""Regression tests for the 500/50/50 protocol; no pretrained GPU model required."""
import importlib.util
from collections import Counter
from pathlib import Path
import pytest
import torch


def api(module,name):
    assert importlib.util.find_spec('tango_iclr.'+module),f'missing module {module}'
    m=importlib.import_module('tango_iclr.'+module)
    assert hasattr(m,name),f'missing function {module}.{name}'
    return getattr(m,name)


def test_protocol():
    c=api('controlled','default_protocol')()
    assert c['model']=='Qwen/Qwen3-VL-8B-Instruct'
    assert (c['train_count'],c['validation_count'],c['test_count'])==(500,50,50)
    assert c['test_template']=='D' and c['layers']==list(range(12,24))
    assert c['rank']==8 and c['credit_kind']=='local_action_logprob' and c['donors']==1


def test_balanced_unique_data():
    fn=api('controlled_data','make_specs');s=fn('validation','B',50,812341)
    assert s==fn('validation','B',50,812341)
    counts=Counter(x['target_index'] for x in s)
    assert len(counts)==8 and max(counts.values())-min(counts.values())<=1
    codes=[v for x in s for v in x['nonce_pool']]
    assert len(codes)==len(set(codes))
    assert set(codes).isdisjoint(v for x in fn('test','D',50,912341) for v in x['nonce_pool'])
    with pytest.raises(ValueError,match='Template C'):fn('test','C',50,912341)


def test_rendered_splits(tmp_path):
    generate=api('controlled_data','generate_dataset')
    from tango_iclr.schema import read_jsonl,normalize_records,validate_splits,public_record
    m=generate(tmp_path,16,8,8)
    s={k:normalize_records(read_jsonl(tmp_path/f'{k}.jsonl')) for k in ['train','validation','test']}
    validate_splits(s);assert [len(v) for v in s.values()]==[16,8,8]
    from PIL import Image
    for rows in s.values():
        for r in rows:
            assert len(r['history_images'])==5 and len(r['candidates'])==8
            assert len({Image.open(f).size for f in r['history_images']+[r['image']]})==1
            assert not {'target_index','credits','donor_indices','nonce_pool'}&public_record(r).keys()
            assert 'action_values' not in r and 'trajectory_return' not in r
    assert m['data_kind']=='synthetic_controlled_not_native_gui'
    with pytest.raises(FileExistsError):generate(tmp_path,16,8,8)


def test_pool_preserves_order():
    pool=api('features','pool_chunks');x=torch.randn(4,8)
    assert torch.equal(pool(x,4),x)
    assert not torch.equal(pool(x,4),pool(x.flip(0),4))
    with pytest.raises(ValueError):pool(x,0)


def test_cross_gru_gradient():
    from tango_iclr.controller import CreditController
    try:c=CreditController(8,16,'cross_gru')
    except ValueError:pytest.fail('cross_gru not implemented')
    torch.manual_seed(4);b=torch.randn(5,6,8);q=torch.randn(7,8)
    y=c(b,q);assert y.shape==(5,);y.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in c.parameters())
    assert not torch.allclose(y.detach(),c(b,q.flip(0)))
    assert c(torch.empty(0,6,8),q).shape==(0,)


def test_contrastive_offset_invariance():
    fn=api('teacher','teacher_utility');s=torch.tensor([1.,2.,3.])
    assert torch.allclose(fn(s,'contrastive_action',1),fn(s+5,'contrastive_action',1))
    with pytest.raises(ValueError):fn(torch.ones(1),'contrastive_action',0)


def test_credit_metrics():
    fn=api('credit_diagnostics','credit_metrics');y=torch.tensor([[.1,-.1,0.],[-.1,.1,0.]])
    m=fn(torch.ones_like(y)*.5,y,scale=1.,deadzone=.005)
    assert m['signed_accuracy']==.5 and m['neutral_false_positive_rate']==1.
    assert m['within_prefix_correlation'] is None
    m=fn(y,y,scale=1.,deadzone=.005)
    assert m['signed_accuracy']==1. and m['balanced_sign_accuracy']==1.


def test_sign_loss_gradient():
    from tango_iclr.objectives import credit_loss
    p=torch.tensor([.2,.2],requires_grad=True)
    try:l=credit_loss(p,torch.tensor([-.1,.1]),scale=.1,sign_weight=.5)
    except TypeError:pytest.fail('signed loss not implemented')
    l.backward();assert p.grad[0]>0 and p.grad[1]<0


def test_selection_lock(tmp_path):
    lock=api('controlled','freeze_selection');verify=api('controlled','verify_selection')
    ck=tmp_path/'best.pt';ck.write_bytes(b'v1');data=tmp_path/'test.jsonl';data.write_text('{}\n')
    rec=lock({'memory':str(ck)},data,{'model':'official'},tmp_path/'lock.json')
    verify(rec,data,{'model':'official'});ck.write_bytes(b'v2')
    with pytest.raises(ValueError,match='checkpoint'):verify(rec,data,{'model':'official'})


def test_no_test_in_train_jobs():
    jobs=api('controlled','build_plan')(api('controlled','default_protocol')(),'train')
    assert all('test.jsonl' not in str(j) for j in jobs)
    assert any('action_sft' in str(j) for j in jobs) and any('memory' in str(j) for j in jobs)


def test_frozen_stage2_keeps_stage1_controller(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tango_iclr.controller import CreditController
    from tango_iclr.interventions import LowRankBank
    from tango_iclr.runtime import gate_values
    from tango_iclr.schema import write_jsonl
    import tango_iclr.train as module
    class Engine:
        def __init__(self,*args,**kwargs):self.device=torch.device('cpu');self.controller=None
        def setup(self,kind,hidden,rank,alpha,gate_mode,**kwargs):
            self.controller=CreditController(8,hidden,kind);self.bank=LowRankBank({'k':8},rank,alpha);self.mode=gate_mode;self.metadata={}
        def context(self,row):
            # Deterministic detached miniature features; not an actual Qwen/GUI result.
            z=torch.arange(64.).reshape(2,4,8)/32
            return SimpleNamespace(features=z,query=torch.arange(40.).reshape(5,8)/20,spans=[(0,1),(1,2)])
        def score(self,c,a,controlled=False,**kwargs):
            x=torch.arange(24.).reshape(1,3,8)/24
            if controlled:
                self.bank.reset_energy();g=gate_values(self.controller(c.features,c.query),self.mode)
                x=self.bank.apply('k',x,c.spans,g)
            return x.sum()*(1 if a=='A' else -1)
        def save(self,path,extra):
            torch.save(dict(metadata=extra,controller=self.controller.state_dict(),bank=self.bank.state_dict()),path)
    monkeypatch.setattr(module,'QwenEngine',Engine)
    def row(i):return dict(prefix_id=i,task_id=i,instruction=i,image='a.png',history_images=['b.png','c.png'],
        candidates=['A','B'],target_index=0,credits=[.1,-.1],credit_kind='local_action_logprob')
    write_jsonl(tmp_path/'train.jsonl',[row('train')]);write_jsonl(tmp_path/'val.jsonl',[row('val')])
    module.train(dict(train=str(tmp_path/'train.jsonl'),validation=str(tmp_path/'val.jsonl'),model='fake',method='memory',
        output=str(tmp_path/'out'),device='cpu',controller='cross_gru',hidden=16,rank=4,
        credit_epochs=2,policy_epochs=1,freeze_controller_policy=True,memory_policy_objective='action_sft',lambda_kl=0.))
    a=torch.load(tmp_path/'out'/'stage1.pt',weights_only=True);b=torch.load(tmp_path/'out'/'last.pt',weights_only=True)
    assert all(torch.equal(a['controller'][k],b['controller'][k]) for k in a['controller'])
    assert any(not torch.equal(a['bank'][k],b['bank'][k]) for k in a['bank'])
    assert (tmp_path/'out'/'credit_stage1_best.json').is_file()
    assert (tmp_path/'out'/'credit_stage2_last.json').is_file()


def test_plans_have_architecture_compatible_variants_and_same_teacher_cache():
    from tango_iclr.controlled import build_plan,default_protocol
    c=default_protocol();primary=build_plan(c,'train');abl=build_plan(c,'ablations')
    assert len({j['id'] for j in primary})==len(primary)
    assert len({j['id'] for j in abl})==len(abl)
    assert sum(j['id'].startswith('teacher_') for j in primary)==2
    for j in primary+abl:
        if 'config' in j:
            cfg=j['config'];assert cfg['model']=='Qwen/Qwen3-VL-8B-Instruct'
            assert set(cfg['feature_layers']).issubset(cfg['layers'])
        assert 'test.jsonl' not in str(j)


def test_balanced_complete_500_50_50_nonce_manifests():
    from tango_iclr.controlled_data import make_specs
    used=set();rows=[]
    for split,template,n,seed in [('train','A',500,7),('validation','B',50,100007),('test','D',50,200007)]:
        part=make_specs(split,template,n,seed,used);assert len(part)==n
        counts=Counter(r['target_index'] for r in part)
        assert max(counts.values())-min(counts.values())<=1
        rows+=part
    assert len(rows)==600 and len(used)==6000


def test_teacher_donor_selection_reproducible_and_target_unchanged():
    from types import SimpleNamespace
    from tango_iclr.teacher import label_record
    class Engine:
        model_path='fake';revision='rev';target='k';layers=[0];max_pixels=1;max_tokens=64;four_bit=False
        def __init__(self):self.calls=[]
        def context(self,row,keep_donors=False):return SimpleNamespace(spans=[(0,2),(2,4),(4,6)],donors={})
        def score(self,context,action,patch=None,normalization='mean'):
            self.calls.append((action,patch));return torch.tensor(1. if patch is None else .5)
    e=Engine();r=dict(prefix_id='x',instruction='same',candidates=['A','B'],target_index=1,history_images=['a','b','c'])
    a=label_record(e,r,'local_action_logprob',1,'mean','seeded');f=Engine();b=label_record(f,r,'local_action_logprob',1,'mean','seeded')
    assert a==b and e.calls==f.calls
    assert len(e.calls)==4 and all(action=='B' for action,_ in e.calls)


def test_qwen_checkpoint_restores_feature_settings(tmp_path,monkeypatch):
    from tango_iclr.qwen import QwenEngine
    from tango_iclr.controller import CreditController
    from tango_iclr.interventions import LowRankBank
    def fake_init(self,model_path,layers=None,target='k',device='cpu',max_pixels=1,max_tokens=64,revision=None,four_bit=False):
        self.device=torch.device('cpu');self.model_path=model_path;self.layers=layers or [0,1,2]
        self.target=target;self.max_pixels=max_pixels;self.max_tokens=max_tokens;self.revision=revision
        self.four_bit=four_bit;self.feature_dim=8;self.widths={'k':8};self.model_type='qwen3_vl';self.num_hidden_layers=36
    monkeypatch.setattr(QwenEngine,'__init__',fake_init)
    e=QwenEngine('fake');e.setup('cross_gru',16,4,8.,'signed',feature_layers=[0,2],feature_chunks=4,gate_temperature=.7)
    path=tmp_path/'checkpoint.pt';e.save(path,{'credit_scale':.1})
    r=QwenEngine.from_checkpoint(path,device='cpu')
    assert r.feature_mode=='layer_chunks' and r.feature_layers==[0,2] and r.feature_chunks==4 and r.temperature==.7
    b=torch.randn(5,8,8);q=torch.randn(10,8)
    assert torch.allclose(e.controller(b,q),r.controller(b,q))
    assert r.metadata['credit_scale']==.1


def test_locked_images_cannot_change(tmp_path):
    import json
    from tango_iclr.controlled import freeze_selection,verify_selection
    from tango_iclr.schema import write_jsonl
    im=tmp_path/'img.png';im.write_bytes(b'original');ck=tmp_path/'ck.pt';ck.write_bytes(b'weights')
    data=tmp_path/'test.jsonl';write_jsonl(data,[dict(image=str(im),history_images=[])])
    r=freeze_selection({'memory':str(ck)},data,{'model':'fake'},tmp_path/'lock.json')
    im.write_bytes(b'modified')
    with pytest.raises(ValueError,match='image'):verify_selection(r,data,{'model':'fake'})


def test_paired_report_compares_sigma_with_sft(tmp_path):
    import json
    from tango_iclr.controlled_report import summarize
    def save(name,correct):
        p=tmp_path/name/'sealed_test.json';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(details=[dict(prefix_id=f'p{i}',correct=c,selected=int(c)) for i,c in enumerate(correct)])))
    save('base',[False,False,True]);save('action_sft_s11',[True,False,True]);save('memory_s11',[True,True,True])
    r=summarize(tmp_path)
    assert r['vs_action_sft']['memory_s11']['rescues']==1
    assert r['vs_action_sft']['memory_s11']['regressions']==0
