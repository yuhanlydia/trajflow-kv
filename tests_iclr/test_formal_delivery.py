"""Regression tests for delivery contracts; not GUI performance experiments."""
import json
from types import SimpleNamespace
import pytest
import torch
from tango_iclr.objectives import action_objective
from tango_iclr.suite import build_jobs
from tango_iclr.evaluate import evaluate_records


def cfg():
    return dict(model='qwen', revision='fixed', train='train.jsonl', validation='val.jsonl',
                test='test.jsonl', output='runs', seeds=[11], online_manifests=['real.jsonl'],
                history_limit=8, max_steps=40,
                base=dict(max_pixels=100352, max_tokens=8192, hidden=256, four_bit=False,
                          layers=list(range(12,24)), target='k', rank=8, credit_epochs=5,
                          policy_epochs=5, memory_policy_objective='action_sft'),
                credit_kind='local_action_logprob')


def test_sft_uses_action_logprob_not_candidate_softmax():
    scores=torch.tensor([-2., -4.], requires_grad=True)
    loss=action_objective(scores,'action_sft',target_index=0)
    assert loss.item()==2.
    loss.backward()
    assert torch.equal(scores.grad,torch.tensor([-1.,0.]))


def test_primary_has_only_base_sft_and_sigma_and_no_distillation():
    jobs=build_jobs(cfg(),'primary')
    trains=[j for j in jobs if j['stage']=='train']
    assert {j['config']['method'] for j in trains}=={'action_sft','memory'}
    assert not any(j['stage']=='distill' for j in jobs)


def test_sft_is_not_blocked_by_missing_causal_teacher():
    jobs=build_jobs(cfg(),'primary')
    j=next(j for j in jobs if j.get('config',{}).get('method')=='action_sft')
    assert j['config']['train']=='train.jsonl'
    assert not any(d.startswith('teacher_') for d in j.get('depends',[]))


def test_base_budget_is_same_as_controlled_budget():
    jobs=build_jobs(cfg(),'primary')
    for j in jobs:
        if j['id'] in {'base_test','base_online_0'}:
            cmd=j['command']
            assert cmd[cmd.index('--max-pixels')+1]=='100352'
            assert cmd[cmd.index('--max-tokens')+1]=='8192'
            assert cmd[cmd.index('--revision')+1]=='fixed'


def test_all_does_not_launch_unverified_policy_distillation():
    jobs=build_jobs(cfg(),'all')
    assert not any(j['stage']=='distill' for j in jobs)


class EvalEngine:
    controller=None
    def context(self,row):return SimpleNamespace(spans=[])
    def score(self,context,action,*args,**kwargs):return torch.tensor(-2.5)


def test_single_demonstrated_action_is_not_100_percent_accuracy():
    r=dict(prefix_id='p',task_id='t',candidates=['click'],target_index=0,critical=True)
    out=evaluate_records(EvalEngine(),[r],controlled=False)
    assert out['accuracy'] is None
    assert out['critical_accuracy'] is None
    assert out['mean_action_nll']==2.5


def test_real_log_export_excludes_failures_and_future_screens():
    from tango_iclr.real_data import trajectory_prefixes
    record={'task_id':'task1','success':True,'status':'complete','backend':'androidworld_http',
            'trajectory':[{'image':f'{i}.png','action':f'action{i}'} for i in range(4)]}
    rows=trajectory_prefixes(record,split='train',min_history=2,history_limit=2)
    assert len(rows)==2
    assert rows[0]['image']=='2.png'
    assert rows[0]['history_images']==['0.png','1.png']
    assert rows[0]['candidates']==['action2']
    assert 'action_values' not in rows[0]  # Never invent Q labels.
    assert rows[0]['data_origin']=='native_gui_demonstration'
    assert trajectory_prefixes(dict(record,success=False),split='train')==[]
    assert trajectory_prefixes(dict(record,status='infra_error'),split='train')==[]


def test_validation_selection_uses_nll_for_demonstrations():
    from tango_iclr.train import selection_score
    assert selection_score({'critical_accuracy':None,'accuracy':None,'mean_action_nll':2.5})==-2.5
    with pytest.raises(ValueError):selection_score({'critical_accuracy':None,'accuracy':None})


def test_memory_and_sft_train_with_real_demo_schema_on_cpu(tmp_path,monkeypatch):
    from test_smoke_training import FakeEngine
    from tango_iclr.schema import write_jsonl
    import tango_iclr.train as module
    monkeypatch.setattr(module,'QwenEngine',FakeEngine)
    def row(i):
        return dict(prefix_id=i,task_id=i,instruction=i,image='now.png',history_images=['a.png','b.png'],
                    candidates=['A'],target_index=0,credits=[.1,-.1],credit_kind='local_action_logprob')
    write_jsonl(tmp_path/'train.jsonl',[row('train')]);write_jsonl(tmp_path/'val.jsonl',[row('val')])
    for method in ['memory','action_sft']:
        result=module.train(dict(train=str(tmp_path/'train.jsonl'),validation=str(tmp_path/'val.jsonl'),
            model='fake',output=str(tmp_path/method),method=method,hidden=16,device='cpu',
            memory_policy_objective='action_sft',credit_epochs=1,policy_epochs=1,accumulation=8))
        assert len(result)==1 and (tmp_path/method/'best.pt').is_file()
